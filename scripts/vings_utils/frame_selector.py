"""
Frame selector for VINGS-Mono (or any online mono/depth SLAM).

Two-stage filter:
  1. Pose filter: reject frames too close to existing keyframes
  2. Geometric information gain: reject frames whose view directions
     are redundant with what's already stored

The geometric information gain follows VISTA (Nagami et al. RA-L 2026, Eq. 1):
    g_I(d_new) = (min(-d_v . d_new) + 1) / 2     per ray
    G_I        = mean over rays                   per frame

Differences vs. the VISTA paper:
  - Use case: binary mapper-skip gate per frame, not trajectory scoring.
    The paper's Eq. 2 (path summation with semantic term and discount) and
    its MPC planner (Algorithm 1) are intentionally NOT used.
  - No voxel traversal: VINGS gives us per-pixel depth, so we just
    back-project and hash the resulting 3D point to a voxel cell. No
    occlusion check against pre-existing geometry.
  - Sparse storage: only voxels that have actually been hit live in the
    dict. No pre-allocated 3D array, no UNOBSERVED/FREE/OCCUPIED state.
  - No semantic gain: pure geometric novelty.
  - Per-voxel direction list bounded via reservoir sampling (Vitter, 1985)
    so the loop-revisit bias of FIFO is avoided.
  - Extra Stage 1 pose filter (translation + rotation thresholds) in front
    of the gain computation; not in the paper, see docs/VISTA.md.

Pose convention assumed (c2w, common in NeRF/3DGS pipelines):
    p_world = R @ p_cam + t
where R is the rotation matrix mapping camera-frame vectors into world frame,
and t is the camera origin in world coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


# =============================================================================
# Config
# =============================================================================

@dataclass
class FrameSelectorConfig:
    voxel_size: float = 0.10                # meters per voxel cell
    max_views_per_voxel: int = 16           # reservoir size -> bounded D_v

    # Pose filter thresholds (Stage 1)
    trans_thresh_m: float = 0.15            # meters
    rot_thresh_deg: float = 10.0            # degrees

    # Gain filter (Stage 2)
    gain_thresh: float = 0.30               # accept if G_I > this
    n_rays_score: int = 256                 # rays used to *score* a frame
    n_rays_integrate: int = 2048            # rays used to *update* state after accept

    # Numerical
    min_depth: float = 0.2                  # ignore depths below this (noise)
    max_depth: float = 15.0                 # ignore depths above this (unreliable)


# =============================================================================
# Frame selector
# =============================================================================

class FrameSelector:
    """
    Online frame filter for streaming RGB-D-like input (RGB + estimated depth).

    Usage:
        sel = FrameSelector(cfg, K, image_hw=(H, W))
        for frame in stream:
            accept, score = sel.should_accept(frame.depth, frame.t, frame.R)
            if accept:
                feed_to_training(frame)
    """

    def __init__(self, cfg: FrameSelectorConfig, K: np.ndarray, image_hw: tuple[int, int]):
        self.cfg = cfg
        self.K_inv = np.linalg.inv(K).astype(np.float32)
        self.H, self.W = image_hw

        # Sparse voxel hash. Per cell we keep:
        #   seen   : int        — total number of rays ever inserted (for reservoir)
        #   views  : list[arr]  — up to max_views_per_voxel reservoir samples (unit vecs, world)
        self.voxel_views: dict[tuple[int, int, int], dict[str, object]] = {}

        # Keyframe poses for Stage 1 (translation, rotation matrix)
        self.keyframes: list[tuple[np.ndarray, np.ndarray]] = []

        # RNG for ray subsampling
        self._rng = np.random.default_rng(0)

        # Precompute pixel grid once
        vs, us = np.meshgrid(np.arange(self.H), np.arange(self.W), indexing="ij")
        self._pixels = np.stack([us.ravel(), vs.ravel(), np.ones_like(us.ravel())],
                                axis=1).astype(np.float32)  # (H*W, 3)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, cfg_dict: dict, K: np.ndarray,
                    image_hw: tuple[int, int]) -> "FrameSelector":
        """Build a FrameSelector from a plain dict (e.g. parsed YAML)."""
        fields = {f for f in FrameSelectorConfig.__dataclass_fields__.keys()}
        kwargs = {k: v for k, v in cfg_dict.items() if k in fields}
        return cls(FrameSelectorConfig(**kwargs), K, image_hw)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def should_accept(
        self,
        depth: np.ndarray,        # (H, W) in meters, <=0 or NaN = invalid
        t: np.ndarray,            # (3,) camera translation in world
        R: np.ndarray,            # (3, 3) camera rotation, c2w
        rgb: Optional[np.ndarray] = None,   # ignored; for selector-API parity
        **_: object,                          # absorb unknown kwargs (depth_cov, ...)
    ) -> tuple[bool, float]:
        """
        Decide whether to keep this frame. Returns (accept, geometric_gain).
        If accept=True, internal state is updated. `rgb` is accepted for
        API parity with feature-based selectors but ignored.
        """
        t = np.asarray(t, dtype=np.float32)
        R = np.asarray(R, dtype=np.float32)

        # First frame is always accepted (otherwise we have nothing to compare against)
        if not self.keyframes:
            score = 1.0
            self._integrate(depth, t, R)
            return True, score

        # ---- Stage 1: Pose filter --------------------------------------
        if self._pose_is_redundant(t, R):
            return False, 0.0

        # ---- Stage 2: Geometric gain -----------------------------------
        score = self._geometric_gain(depth, t, R, n_rays=self.cfg.n_rays_score)
        if score < self.cfg.gain_thresh:
            return False, score

        # Accept -> update state
        self._integrate(depth, t, R)
        return True, score

    # ------------------------------------------------------------------
    # Stage 1: Pose filter
    # ------------------------------------------------------------------

    def _pose_is_redundant(self, t: np.ndarray, R: np.ndarray) -> bool:
        """
        Return True if there is any existing keyframe within BOTH the
        translation and rotation thresholds. "Both" means: a frame at a
        nearby position AND similar orientation. If either is far enough,
        the new frame is potentially useful.
        """
        rot_th = np.deg2rad(self.cfg.rot_thresh_deg)
        for t_i, R_i in self.keyframes:
            d_t = np.linalg.norm(t - t_i)
            cos_a = (np.trace(R_i.T @ R) - 1.0) * 0.5
            d_r = np.arccos(np.clip(cos_a, -1.0, 1.0))
            if d_t < self.cfg.trans_thresh_m and d_r < rot_th:
                return True
        return False

    # ------------------------------------------------------------------
    # Stage 2: Geometric gain (VISTA-style)
    # ------------------------------------------------------------------

    def _geometric_gain(self, depth: np.ndarray, t: np.ndarray, R: np.ndarray,
                        n_rays: int) -> float:
        """
        VISTA paper Eq. 1, but with depth-based back-projection instead of
        voxel traversal. Returns G_I in [0,1].
        """
        rays_world, points_world = self._sample_back_projected(depth, t, R, n_rays)
        if len(rays_world) == 0:
            return 0.0

        gains = np.empty(len(rays_world), dtype=np.float32)
        for n, (d_new, p_world) in enumerate(zip(rays_world, points_world)):
            cell = self._point_to_cell(p_world)
            entry = self.voxel_views.get(cell)
            if entry is None or not entry["views"]:
                # Voxel never hit before -> like UNOBSERVED in the paper
                gains[n] = 1.0
            else:
                stored_arr = np.asarray(entry["views"], dtype=np.float32)  # (M, 3)
                cos_max = float((stored_arr @ d_new).max())                # closest existing dir
                gains[n] = (-cos_max + 1.0) * 0.5                          # Eq. 1

        return float(gains.mean())

    # ------------------------------------------------------------------
    # State update
    # ------------------------------------------------------------------

    def _integrate(self, depth: np.ndarray, t: np.ndarray, R: np.ndarray) -> None:
        """Insert direction into each hit voxel via reservoir sampling.

        Reservoir keeps a uniform sample of size <= cap over the full history
        of insertions, so loop-revisits don't evict the original viewpoints
        (which FIFO would). Bounded list keeps the scoring step O(M*cap).
        """
        rays_world, points_world = self._sample_back_projected(
            depth, t, R, self.cfg.n_rays_integrate
        )
        cap = self.cfg.max_views_per_voxel
        for d_new, p_world in zip(rays_world, points_world):
            cell = self._point_to_cell(p_world)
            entry = self.voxel_views.get(cell)
            if entry is None:
                entry = {"seen": 0, "views": []}
                self.voxel_views[cell] = entry
            views: list = entry["views"]  # type: ignore[assignment]
            entry["seen"] = int(entry["seen"]) + 1  # type: ignore[assignment]
            if len(views) < cap:
                views.append(d_new.astype(np.float32))
            else:
                # Algorithm R (Vitter, 1985): replace random slot with prob cap/seen
                j = int(self._rng.integers(0, entry["seen"]))  # type: ignore[arg-type]
                if j < cap:
                    views[j] = d_new.astype(np.float32)

        self.keyframes.append((t.copy(), R.copy()))

    # ------------------------------------------------------------------
    # Back-projection helper (the only "geometry" we need)
    # ------------------------------------------------------------------

    def _sample_back_projected(
        self,
        depth: np.ndarray,
        t: np.ndarray,
        R: np.ndarray,
        n_rays: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Subsample pixels with valid depth, back-project to world.
        Returns (rays_world (N,3) unit vectors, points_world (N,3)).
        """
        d_flat = depth.ravel()
        valid = (d_flat > self.cfg.min_depth) & (d_flat < self.cfg.max_depth) & np.isfinite(d_flat)
        if not valid.any():
            return np.empty((0, 3)), np.empty((0, 3))

        valid_idx = np.flatnonzero(valid)
        if len(valid_idx) > n_rays:
            valid_idx = self._rng.choice(valid_idx, size=n_rays, replace=False)

        pix = self._pixels[valid_idx]                  # (N, 3)
        d = d_flat[valid_idx].astype(np.float32)       # (N,)

        # Camera-frame rays (unnormalized): K^-1 [u v 1]^T (z-component = 1)
        rays_cam_un = pix @ self.K_inv.T               # (N, 3)
        rays_cam = rays_cam_un / np.linalg.norm(rays_cam_un, axis=1, keepdims=True)
        rays_world = rays_cam @ R.T                    # (N, 3)

        # World-frame points: scale unnormalized rays by depth-along-z, then transform
        points_cam = rays_cam_un * d[:, None]
        points_world = points_cam @ R.T + t            # (N, 3)

        return rays_world.astype(np.float32), points_world.astype(np.float32)

    def _point_to_cell(self, p_world: np.ndarray) -> tuple[int, int, int]:
        idx = np.floor(p_world / self.cfg.voxel_size).astype(int)
        return int(idx[0]), int(idx[1]), int(idx[2])


# =============================================================================
# Smoke test
# =============================================================================

if __name__ == "__main__":
    rng = np.random.default_rng(42)

    # Fake pinhole intrinsics: 640x480, fov ~ 70deg
    H, W = 240, 320
    fx = fy = 0.5 * W / np.tan(np.deg2rad(70.0) / 2)
    cx, cy = W / 2, H / 2
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

    cfg = FrameSelectorConfig(gain_thresh=0.25)
    sel = FrameSelector(cfg, K, (H, W))

    def yaw_rot(psi: float) -> np.ndarray:
        c, s = np.cos(psi), np.sin(psi)
        return np.array([
            [-s, 0, c],
            [ c, 0, s],
            [ 0, -1, 0],
        ], dtype=np.float32)

    def fake_depth(t, R) -> np.ndarray:
        """Cheap synthetic depth: assume a box room centered at origin, 3m radius."""
        pix = sel._pixels
        rays_cam_un = pix @ sel.K_inv.T
        rays_cam = rays_cam_un / np.linalg.norm(rays_cam_un, axis=1, keepdims=True)
        rays_world = rays_cam @ R.T
        ts = []
        for axis, lim in [(0, 3.0), (0, -3.0), (1, 3.0), (1, -3.0)]:
            num = lim - t[axis]
            denom = rays_world[:, axis]
            with np.errstate(divide="ignore", invalid="ignore"):
                tt = np.where(np.abs(denom) > 1e-6, num / denom, np.inf)
            tt = np.where(tt > 0.1, tt, np.inf)
            ts.append(tt)
        ts = np.stack(ts, axis=1)
        depth_along_ray = ts.min(axis=1)
        depth_z = depth_along_ray * rays_cam[:, 2]
        depth_z = np.where(np.isfinite(depth_z) & (depth_z > 0), depth_z, 0.0)
        return depth_z.reshape(H, W).astype(np.float32)

    accepted = 0
    total = 0
    for i, psi in enumerate(np.linspace(0, 2 * np.pi, 60, endpoint=False)):
        t = np.array([0.0, 0.0, 0.8], dtype=np.float32)
        R = yaw_rot(psi)
        depth = fake_depth(t, R)
        ok, score = sel.should_accept(depth, t, R)
        total += 1
        accepted += int(ok)
        if i % 5 == 0:
            print(f"frame {i:2d}  yaw={np.degrees(psi):+6.1f}°  "
                  f"score={score:.3f}  {'ACCEPT' if ok else 'skip'}  "
                  f"keyframes={len(sel.keyframes)}  voxels={len(sel.voxel_views)}")

    print(f"\nTotal: {accepted}/{total} accepted, {len(sel.voxel_views)} voxels stored")

    print("\n--- Replaying same trajectory ---")
    accepted2 = 0
    for psi in np.linspace(0, 2 * np.pi, 60, endpoint=False):
        t = np.array([0.0, 0.0, 0.8], dtype=np.float32)
        R = yaw_rot(psi)
        depth = fake_depth(t, R)
        ok, _ = sel.should_accept(depth, t, R)
        accepted2 += int(ok)
    print(f"Replay: {accepted2}/60 accepted (low number = filter works)")
