"""
MM3DGS-inspired keyframe selector.

Implements the keyframe selection step from Sun L.C., Bhatt N.P. et al.,
"MM3DGS SLAM: Multi-modal 3D Gaussian Splatting for SLAM Using Vision, Depth,
and Inertial Measurements", IROS 2024, Section III.E.

Two gates per Tracker-KF candidate:

    1. Covisibility-Gate (matches the direction of the reference code at
       VITA-Group/MM3DGS-SLAM `slam/mapper.py:is_covisible`).
         Sample pixels of PREV_KF's stored depth, back-project to world using
         the PREV_KF pose, then project those world points into the CURRENT
         frame. covis = fraction of points that land inside the current image
         with positive z. Accept candidate iff covis < covis_thresh
         (default 0.95). Reference renders depth from the Gaussian map at the
         prev-KF pose with a silhouette > 0.99 mask; we use the tracker depth
         that was current at the moment prev_kf was committed (see Adaptations).

    2. Quality-Gate ("NIQE min in sliding window")
         Maintain a sliding window of the per-frame Laplacian variance.
         The window is populated UNCONDITIONALLY (matches reference
         `slam/mapper.py` push-every-frame loop) — both candidate and
         non-candidate frames enter so the recent quality baseline is
         honest. We accept the current frame iff it is the window-best
         (highest lap-var) at the moment covis triggers. The reference
         instead commits `niqe_window[0]` (window-min-NIQE / window-max-
         quality) as the KF, which may be an OLDER frame; the per-frame
         `should_accept` interface in VINGS cannot retroactively choose
         an older frame (viz_out is ephemeral), so we approximate by only
         accepting when the current frame happens to be the window-best.
         Quality is approximated by the variance of the Laplacian on
         grayscale (sharp = high Laplacian variance, blurry = low). The
         original paper uses NIQE.

Adaptations vs the paper / reference code:
  - Depth source: VINGS' DBA-Fusion / motion3d tracker depth (viz_out['depths'])
    instead of rendering depth from the Gaussian map. Reference renders
    depth from the map at the prev-KF pose, silhouette-masked at > 0.99
    (`slam/mapper.py:145-151`). VINGS adaptation: store the tracker depth at
    the moment prev_kf was committed and back-project from there. Robust
    during early/under-optimized map and consistent with VISTA / NURBS-LVI.
  - NIQE -> Variance-of-Laplacian. Same intent (motion-blur suppression),
    no extra dependency, microsecond-level cost.
  - Covisibility comparison is against the single most-recent keyframe rather
    than a covisibility graph; VINGS does not maintain a KF graph for the
    selector slot. Reference is also singular here: `self.keyframes[-1]`.
  - Invalid-depth frames (e.g. sky-only / aerial frames where the tracker
    depth is entirely outside [min_depth, max_depth]) return covis=1.0
    ("skip"), instead of treating "no data" as "new terrain". Reference has
    no such guard because rendered+silhouette-masked depth always has *some*
    valid pixels by construction.
  - No retroactive KF emission. Reference: when covis triggers, commits
    `niqe_window[0]` (the recent-window's best-quality frame, possibly older
    than the current one). VINGS: emits only the current frame. Effect: with
    monotonically increasing blur within a candidate streak we lose the
    sharp leading frame; `force_accept_after` is the failsafe.
  - `min_gap_after_kf` mirrors reference `kf_every` (TUM-default 5) as a
    MINIMUM frame gap before a spawn is allowed. The reference's check
    (`mapper.py:170`) is a post-covis-drop gate, NOT an unconditional
    force-accept: if covis stays > 0.95, no spawn happens regardless of
    elapsed frames. Default 0 (off) — VINGS sequences with `frame_skip`
    already enforce coarser tracker-KF spacing.

Same calling convention as the other selectors:

    sel = Mm3dgsSelector(cfg, K, (H, W))
    accept, score = sel.should_accept(depth, t, R, rgb=rgb_uint8)
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    import cv2
except ImportError as e:
    raise ImportError("Mm3dgsSelector requires opencv-python.") from e

from vings_utils.nurbs_lvi_selector import backproject


# =============================================================================
# Config / data classes
# =============================================================================

@dataclass
class Mm3dgsConfig:
    # Covisibility gate
    covis_thresh: float = 0.95          # accept iff covis < this
    n_samples: int = 2048               # # depth pixels for back-projection
    min_depth: float = 0.2
    max_depth: float = 35.0

    # Quality gate (NIQE-min-in-window proxy)
    niqe_window: int = 5                # sliding window size (frames)

    # Quality-gate failsafe: long below-threshold streak with no window-best.
    # Counts consecutive spawn-eligible frames (below_thresh AND gap-eligible)
    # that failed Gate 2 since last KF. 0 disables.
    force_accept_after: int = 0

    # Paper's `kf_every` (TUM config = 5). Minimum frame gap since last KF
    # BEFORE a spawn is allowed. Combined with covis-drop, mirrors the
    # reference's `idx - keyframes[-1].idx >= kf_every` post-covis check.
    # 0 disables (allow back-to-back KFs as soon as covis drops).
    min_gap_after_kf: int = 0


@dataclass
class Mm3dgsScore:
    covis: float = 1.0
    lap_var: float = 0.0
    window_size: int = 0
    is_window_best: bool = False
    forced: bool = False
    forced_reason: str = ""
    gap_frames: int = 0
    accepted: bool = False


# =============================================================================
# Helpers
# =============================================================================

def _laplacian_var(img: np.ndarray) -> float:
    """Variance-of-Laplacian blur proxy. Accepts HxW or HxWx3 uint8."""
    if img.ndim == 3 and img.shape[2] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif img.ndim == 2:
        gray = img
    else:
        gray = img[..., 0]
    if gray.dtype != np.uint8:
        gray = gray.astype(np.uint8)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _project_to_kf(world_pts: np.ndarray,
                   R_kf: np.ndarray, t_kf: np.ndarray,
                   K: np.ndarray, H: int, W: int) -> np.ndarray:
    """
    world_pts: (N,3) world-frame points. R_kf, t_kf: c2w of the target keyframe.
    Returns boolean (N,) mask: True where the point projects inside the image
    with positive depth.
    """
    valid_in = np.isfinite(world_pts[:, 0])
    if not valid_in.any():
        return np.zeros(world_pts.shape[0], dtype=bool)
    # c2w inverse: pts_cam = R.T @ (P - t)
    pts_cam = (world_pts - t_kf) @ R_kf
    z = pts_cam[:, 2]
    in_front = z > 1e-6
    safe_z = np.where(in_front, z, 1.0)
    uv = pts_cam @ K.T
    u = uv[:, 0] / safe_z
    v = uv[:, 1] / safe_z
    inside = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    return valid_in & in_front & inside


# =============================================================================
# Selector
# =============================================================================

class Mm3dgsSelector:
    """MM3DGS-style keyframe selector. Same shape as FrameSelector / NurbsLviSelector."""

    def __init__(self, cfg: Mm3dgsConfig, K: np.ndarray, image_hw: tuple[int, int]):
        self.cfg = cfg
        self.K = np.asarray(K, dtype=np.float32)
        self.K_inv = np.linalg.inv(self.K)
        self.H, self.W = image_hw

        self.prev_kf_R: Optional[np.ndarray] = None
        self.prev_kf_t: Optional[np.ndarray] = None
        self.prev_kf_depth: Optional[np.ndarray] = None
        # Sliding window of lap_var for the last N frames (every frame pushed).
        self.window: deque[float] = deque(maxlen=cfg.niqe_window)
        self._stalled = 0
        self._gap = 0

        # Cache of pixel-grid samples (deterministic, reused across calls).
        self._uv_samples = self._make_uv_samples()

    # ------------------------------------------------------------------
    @classmethod
    def from_config(cls, cfg_dict: dict, K: np.ndarray,
                    image_hw: tuple[int, int]) -> "Mm3dgsSelector":
        fields = set(Mm3dgsConfig.__dataclass_fields__.keys())
        kwargs = {k: v for k, v in cfg_dict.items() if k in fields}
        return cls(Mm3dgsConfig(**kwargs), K, image_hw)

    # ------------------------------------------------------------------
    def should_accept(
        self,
        depth: np.ndarray,
        t: np.ndarray,
        R: np.ndarray,
        rgb: Optional[np.ndarray] = None,
        **_: object,
    ) -> tuple[bool, Mm3dgsScore]:
        score = Mm3dgsScore()

        # Quality proxy. If no rgb was passed, fall back to a normalised depth
        # rendering -- gives a valid (if noisier) blur signal so the gate still
        # functions in rgb-less smoketests.
        if rgb is not None:
            score.lap_var = _laplacian_var(rgb)
        else:
            d = np.nan_to_num(depth, nan=0.0)
            d_norm = np.clip(d / max(self.cfg.max_depth, 1e-6), 0, 1) * 255.0
            score.lap_var = _laplacian_var(d_norm.astype(np.uint8))

        t = np.asarray(t, dtype=np.float32)
        R = np.asarray(R, dtype=np.float32)

        # First frame: nothing to compare to, accept and seed prev_kf.
        if self.prev_kf_R is None:
            self._commit(R, t, depth)
            score.covis = 1.0
            score.is_window_best = True
            score.accepted = True
            self.window.append(score.lap_var)
            score.window_size = 1
            return True, score

        self._gap += 1
        score.gap_frames = self._gap

        # Gate 1: Covisibility (paper direction — prev_kf depth → current frame).
        # Reference returns is_covisible=True iff percent_inside > 0.95, then
        # need_new_keyframe returns False (no spawn). So spawn-eligibility is
        # iff percent_inside <= 0.95. We use strict `<` (off-by-epsilon at the
        # exact boundary; functionally identical with random samples).
        score.covis = self._covisibility(t, R)
        below_thresh = score.covis < self.cfg.covis_thresh

        # Gate 2: NIQE-min sliding window. Push every frame (matches reference
        # `slam/mapper.py` push-every-frame loop). Current is window-best iff
        # its lap_var equals the recent maximum.
        self.window.append(score.lap_var)
        score.window_size = len(self.window)
        score.is_window_best = score.lap_var >= max(self.window)

        # Reference spawn-trigger: below_thresh AND (gap >= kf_every).
        # is_window_best is our (non-paper) quality gate to approximate the
        # retroactive niqe_window[0]-pick we cannot perform.
        gap_eligible = self._gap >= self.cfg.min_gap_after_kf
        spawn_eligible = below_thresh and gap_eligible
        accept = spawn_eligible and score.is_window_best

        # Quality-gate failsafe: long spawn-eligible streak with no window-best.
        # Only counts spawn-eligible frames (matches reference branch — high-
        # covis frames never advance the failsafe).
        if spawn_eligible and not accept:
            self._stalled += 1
            if self.cfg.force_accept_after > 0 and self._stalled >= self.cfg.force_accept_after:
                accept = True
                score.forced = True
                score.forced_reason = "force_accept_after"
        else:
            self._stalled = 0

        if accept:
            self._commit(R, t, depth)
            score.accepted = True

        return accept, score

    # ------------------------------------------------------------------
    def _commit(self, R: np.ndarray, t: np.ndarray, depth: np.ndarray) -> None:
        self.prev_kf_R = R.copy()
        self.prev_kf_t = t.copy()
        self.prev_kf_depth = depth.copy()
        self._stalled = 0
        self._gap = 0

    def _covisibility(self, t_curr: np.ndarray, R_curr: np.ndarray) -> float:
        """Paper-direction: backproject prev_kf depth at prev_kf pose, project
        the resulting world points into the current frame, return the inside-
        image fraction. Mirrors `slam/mapper.py:is_covisible` in the reference.
        """
        uv = self._uv_samples
        world = backproject(uv, self.prev_kf_depth, self.K_inv,
                            self.prev_kf_t, self.prev_kf_R,
                            self.cfg.min_depth, self.cfg.max_depth)
        n_valid_world = int(np.isfinite(world[:, 0]).sum())
        if n_valid_world == 0:
            # No usable prev-KF depth — cannot judge covisibility. Treat as
            # "fully covered" so the frame is skipped; missing data is not
            # evidence of new terrain. Defensive guard not present in the
            # reference (which renders silhouette-masked map depth and is
            # never empty by construction).
            return 1.0
        visible = _project_to_kf(world, R_curr, t_curr,
                                 self.K, self.H, self.W)
        return float(visible.sum()) / float(n_valid_world)

    def _make_uv_samples(self) -> np.ndarray:
        """Deterministic, stride-based sample grid covering the image."""
        n = max(64, int(self.cfg.n_samples))
        # aspect-aware grid count
        aspect = self.W / max(self.H, 1)
        ny = max(8, int(np.sqrt(n / aspect)))
        nx = max(8, int(n / ny))
        xs = np.linspace(0.5, self.W - 0.5, nx, dtype=np.float32)
        ys = np.linspace(0.5, self.H - 0.5, ny, dtype=np.float32)
        uu, vv = np.meshgrid(xs, ys)
        return np.stack([uu.ravel(), vv.ravel()], axis=1)


# =============================================================================
# Smoke test
# =============================================================================

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    H, W = 240, 320
    fx = fy = 0.5 * W / np.tan(np.deg2rad(70.0) / 2)
    K = np.array([[fx, 0, W / 2], [0, fy, H / 2], [0, 0, 1]], np.float32)

    # min_gap_after_kf mirrors paper TUM `kf_every: 5`. Combined with
    # below_thresh, it requires at least N frames since last KF.
    cfg = Mm3dgsConfig(covis_thresh=0.95, niqe_window=5,
                       n_samples=1024, min_depth=0.2, max_depth=35.0,
                       min_gap_after_kf=5, force_accept_after=10)
    sel = Mm3dgsSelector(cfg, K, (H, W))

    def sharp_rgb() -> np.ndarray:
        return rng.integers(0, 255, (H, W, 3), dtype=np.uint8)

    def blurry_rgb() -> np.ndarray:
        return cv2.GaussianBlur(rng.integers(0, 255, (H, W, 3), dtype=np.uint8),
                                (21, 21), 0)

    depth = np.full((H, W), 3.0, dtype=np.float32)
    I = np.eye(3, dtype=np.float32)

    def report(i, accept, sc, label):
        flag = "ACCEPT" + (f"(force={sc.forced_reason})" if sc.forced else "")
        print(f"frame {i:2d} {label:>14s}  covis={sc.covis:.3f}  "
              f"lap={sc.lap_var:8.1f}  win={sc.window_size}  "
              f"best={sc.is_window_best}  gap={sc.gap_frames:2d}  "
              f"{flag if accept else 'skip'}")

    n_accept = 0
    ok, sc = sel.should_accept(depth, np.zeros(3, np.float32), I, sharp_rgb())
    n_accept += int(ok); report(0, ok, sc, "first")

    for i in range(1, 5):
        ok, sc = sel.should_accept(depth, np.zeros(3, np.float32), I, sharp_rgb())
        n_accept += int(ok); report(i, ok, sc, "stationary")

    for i in range(5, 12):
        t = np.array([1.0 * (i - 4), 0.0, 0.0], np.float32)
        rgb = blurry_rgb() if (i % 2 == 0) else sharp_rgb()
        ok, sc = sel.should_accept(depth, t, I, rgb)
        n_accept += int(ok); report(i, ok, sc, "translating")

    print(f"\nTotal accepted: {n_accept}/12  (expected: 2-3 — first frame, "
          f"plus a sharp translating frame once below_thresh and gap>=5)")
