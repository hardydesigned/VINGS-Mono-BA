"""
NURBS-LVI-inspired keyframe selector.

Implements the adaptive keyframe selection from Wu et al., IEEE/ASME TMECH 2026
("NURBS-Based Continuous LiDAR-Visual-Inertial SLAM with Adaptive Keyframe
Selection"), Section III.A.

Adaptive threshold:
    gamma = (2 - N/2) * (1 - Mc/Mr)
    beta  = Oc/Mc - Oc/Mr
    alpha = Oc/Mc - 0.5
    phi   = s + gamma - beta - alpha
    Q     = phi * (Mc*Or/Mr + Dc*Or/Dr)

Decision (Eq. 4 of the paper):
    accept if  Or + Oc > Q

Q is the *threshold*, not the score; Or + Oc is the migration evidence. The
public C++ reference's `shouldCreateKeyframe(sd, threshold_Q) -> Q >= ...` is
a test harness, not the algorithm itself.

Frame roles (paper III.A.3):
    prev_kf    most recently accepted keyframe
    reference  the IMMEDIATE successor frame of prev_kf
    current    candidate frame, several non-KF frames after reference

Variables (paper):
    Mc, Mr   tracked features (current, reference) — both relative to prev_kf
    Dc, Dr   total *extracted* features in (current, reference)
    Or       sector migrations between prev_kf and reference
    Oc       sector migrations between prev_kf and current
    N        frames between previous keyframe and current frame
    s        exp(-lambda * symmetric_chamfer(P1, P2)),  lambda = 1/|P_2|  (Eq. 3)

The reference frame is the immediate successor of prev_kf and serves *only* as
an anchor for the principal axis. It is NEVER returned as a keyframe (paper
Sec. III.A.3, "not selected as keyframes").

Adaptations vs the original (NTU-VIRAL-style LiDAR-VIO):
  - No IMU propagation; we consume VINGS-Mono's pose estimate directly.
  - No LiDAR; "valid depth" comes from VINGS' dense depth map.
  - Sparse ORB+BF matching at decision time instead of frame-to-frame LK
    tracking. Simpler architecture, ~30-60ms per decision call. The
    feature-tracking strategy is exchangeable (see LKFeatureTracker for an
    optional swap-in).

Same calling convention as FrameSelector:
    sel = NurbsLviSelector(cfg, K, (H, W))
    accept, score = sel.should_accept(depth, t, R, rgb=rgb_uint8)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

try:
    import cv2
except ImportError as e:
    raise ImportError("NurbsLviSelector requires opencv-python.") from e


# =============================================================================
# Config / data classes
# =============================================================================

@dataclass
class NurbsLviConfig:
    # ORB
    orb_n_features: int = 800

    # Sector binning (3 sectors around principal axis). Paper-Default is 15.0,
    # passend zu LiDAR-VIO mit grossen Inter-KF-Baselines. Bei dichten Video-
    # Frames mit kleinem baseline/depth ist die Parallaxe ~0.5-2° -- mit 15°
    # waere Or strukturell 0. 2° ist empirisch besser fuer VINGS.
    sector_angle_deg: float = 2.0

    # Failsafe-Pfade
    min_matches: int = 15           # below this -> force accept (tracking unstable)
    force_accept_all: bool = False  # diagnostic: log Q/Or/Oc but always accept

    # Depth gating (same convention as FrameSelector)
    min_depth: float = 0.2
    max_depth: float = 35.0


@dataclass
class KeyframeStats:
    Mc: int = 0
    Mr: int = 0
    Dc: int = 0
    Dr: int = 0
    Or_: int = 0     # 'Or' is a Python keyword-ish builtin; rename
    Oc: int = 0
    N: int = 0
    s: float = 1.0


@dataclass
class ScoreDetail:
    phi: float = 0.0
    gamma: float = 0.0
    beta: float = 0.0
    alpha: float = 0.0
    term_feat: float = 0.0
    Q: float = 0.0                   # adaptive threshold
    migration: float = 0.0           # Or + Oc; accepted when migration > Q
    forced: bool = False             # accept was forced (too few matches / diag)
    n_matches: int = 0


# Internal cache: everything we need to remember about an accepted frame.
@dataclass
class _FrameMemo:
    rgb_gray: np.ndarray
    depth: np.ndarray
    t: np.ndarray
    R: np.ndarray
    kp_uv: np.ndarray                # (N, 2) pixel coords
    desc: np.ndarray                 # (N, 32) ORB descriptor bytes
    world_pts: np.ndarray            # (N, 3), NaN row where depth invalid


# =============================================================================
# Pure helpers
# =============================================================================

def safe_div(a: float, b: float, eps: float = 1e-12) -> float:
    return 0.0 if abs(b) < eps else a / b


def symmetric_chamfer(P1: np.ndarray, P2: np.ndarray) -> float:
    """Squared-distance symmetric Chamfer between two (N,3)/(M,3) point sets."""
    if P1.size == 0 or P2.size == 0:
        return float("inf")
    diff = P1[:, None, :] - P2[None, :, :]      # (N, M, 3)
    d2 = np.einsum("ijk,ijk->ij", diff, diff)   # (N, M)
    return float(d2.min(axis=1).mean() + d2.min(axis=0).mean())


def backproject(uv: np.ndarray, depth: np.ndarray,
                K_inv: np.ndarray, t: np.ndarray, R: np.ndarray,
                min_d: float, max_d: float) -> np.ndarray:
    """
    Sub-pixel depth lookup at uv ((N,2)), back-project to world.
    Returns (N,3) with NaN rows where depth is invalid.
    """
    H, W = depth.shape
    u = np.clip(uv[:, 0], 0, W - 1).astype(np.float32)
    v = np.clip(uv[:, 1], 0, H - 1).astype(np.float32)
    # Nearest-pixel depth sample (cheap, ORB keypoints are sub-pixel but depth is dense)
    d = depth[v.astype(int), u.astype(int)]
    valid = np.isfinite(d) & (d > min_d) & (d < max_d)

    pix = np.stack([u, v, np.ones_like(u)], axis=1)        # (N, 3)
    rays_cam = pix @ K_inv.T                                # (N, 3)
    pts_cam = rays_cam * d[:, None]                         # scale along z
    pts_world = pts_cam @ R.T + t                           # (N, 3)
    pts_world[~valid] = np.nan
    return pts_world


def assign_sector(view_vec: np.ndarray, principal_axis: np.ndarray,
                  sector_deg: float) -> int:
    """
    Return 0 / 1 / 2 — center sector is 1, outer ±`sector_deg` are 0 and 2.
    `view_vec` and `principal_axis` need not be unit; we normalize.
    """
    a = view_vec / (np.linalg.norm(view_vec) + 1e-12)
    b = principal_axis / (np.linalg.norm(principal_axis) + 1e-12)
    cos_a = float(np.clip(a @ b, -1.0, 1.0))
    ang = np.degrees(np.arccos(cos_a))
    if ang < sector_deg * 0.5:
        return 1
    return 0 if (np.cross(a, b)[2] > 0) else 2  # rough left/right split via z-axis


# =============================================================================
# Score (verbatim transcription of the C++ reference)
# =============================================================================

def compute_keyframe_score(stats: KeyframeStats) -> ScoreDetail:
    gamma = (2.0 - 0.5 * stats.N) * (1.0 - safe_div(stats.Mc, stats.Mr))
    beta  = safe_div(stats.Oc, stats.Mc) - safe_div(stats.Oc, stats.Mr)
    alpha = safe_div(stats.Oc, stats.Mc) - 0.5
    phi   = stats.s + gamma - beta - alpha

    t1 = safe_div(stats.Mc * stats.Or_, stats.Mr)
    t2 = safe_div(stats.Dc * stats.Or_, stats.Dr)
    term = t1 + t2

    return ScoreDetail(phi=phi, gamma=gamma, beta=beta, alpha=alpha,
                       term_feat=term, Q=phi * term)


# =============================================================================
# Selector
# =============================================================================

class NurbsLviSelector:
    """Keyframe selector compatible with FrameSelector's should_accept() shape."""

    def __init__(self, cfg: NurbsLviConfig, K: np.ndarray, image_hw: tuple[int, int]):
        self.cfg = cfg
        self.K = np.asarray(K, dtype=np.float32)
        self.K_inv = np.linalg.inv(self.K)
        self.H, self.W = image_hw

        self.orb = cv2.ORB_create(nfeatures=cfg.orb_n_features)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        self.prev_kf: Optional[_FrameMemo] = None
        self.reference: Optional[_FrameMemo] = None
        self.ref_pkmatches: Optional[tuple[np.ndarray, np.ndarray]] = None  # (idx_r, idx_p)
        self.Mr_cached: int = 0
        self.Dr_cached: int = 0
        self.Or_cached: int = 0
        self.N: int = 0

    # ------------------------------------------------------------------
    @classmethod
    def from_config(cls, cfg_dict: dict, K: np.ndarray,
                    image_hw: tuple[int, int]) -> "NurbsLviSelector":
        fields = set(NurbsLviConfig.__dataclass_fields__.keys())
        kwargs = {k: v for k, v in cfg_dict.items() if k in fields}
        return cls(NurbsLviConfig(**kwargs), K, image_hw)

    # ------------------------------------------------------------------
    def should_accept(
        self,
        depth: np.ndarray,
        t: np.ndarray,
        R: np.ndarray,
        rgb: Optional[np.ndarray] = None,
        **_: object,
    ) -> tuple[bool, Optional[ScoreDetail]]:
        """
        Per the paper (Sec. III.A.3):
            prev_kf   most recently accepted keyframe
            reference IMMEDIATE successor frame of prev_kf
            current   candidate, several non-KF frames after reference
            accept iff (Or + Oc) > Q

        rgb may be (H, W) gray or (H, W, 3) BGR/RGB uint8.
        """
        if rgb is None:
            return True, None

        gray = self._to_gray(rgb)
        memo = self._build_memo(gray, depth, t, R)
        if memo is None:
            return True, None

        # First KF -- nothing to score against yet.
        if self.prev_kf is None:
            self._adopt_as_prev_kf(memo)
            return True, None

        # Second frame after a KF acceptance -- it BECOMES the reference (paper
        # Sec. III.A.3: "immediate successor of previous keyframe ... not selected
        # as keyframe"). Reference is purely an anchor for the principal axis,
        # NOT a keyframe -- so we reject it here.
        if self.reference is None:
            self._adopt_as_reference(memo)
            return False, None

        # Regular decision: score current against (reference fixed, prev_kf fixed).
        stats = self._compute_stats(memo)
        if stats.Mc < self.cfg.min_matches:
            score = ScoreDetail(forced=True, n_matches=stats.Mc)
            self._adopt_as_prev_kf(memo)
            self.reference = None        # next frame becomes new reference
            self.N = 0
            return True, score

        score = compute_keyframe_score(stats)
        score.n_matches = stats.Mc
        score.migration = float(stats.Or_ + stats.Oc)

        accept = (score.migration > score.Q) or self.cfg.force_accept_all
        if self.cfg.force_accept_all:
            score.forced = True

        if accept:
            self._adopt_as_prev_kf(memo)
            self.reference = None
            self.N = 0
            return True, score

        self.N += 1
        return False, score

    def _adopt_as_reference(self, memo: _FrameMemo) -> None:
        """Pin this frame as the reference and pre-compute Mr / Dr / Or."""
        self.reference = memo
        self._recompute_ref_cache()

    # ------------------------------------------------------------------
    # State transitions
    # ------------------------------------------------------------------
    def _adopt_as_prev_kf(self, memo: _FrameMemo) -> None:
        self.prev_kf = memo

    def _recompute_ref_cache(self) -> None:
        """Recompute Mr / Dr / Or between current reference and current prev_kf.

        Dr (per paper) is the *total* number of extracted features in the
        reference frame, not the match count.
        """
        if self.reference is None or self.prev_kf is None:
            self.ref_pkmatches = None
            self.Mr_cached = self.Dr_cached = self.Or_cached = 0
            return

        # Dr = total extracted features in reference
        self.Dr_cached = int(len(self.reference.kp_uv))

        idx_r, idx_p = self._match(self.reference.desc, self.prev_kf.desc)
        self.ref_pkmatches = (idx_r, idx_p)
        Mr = len(idx_r)
        self.Mr_cached = Mr
        if Mr == 0:
            self.Or_cached = 0
            return

        # Or: sector migrations between reference and prev_kf. Principal axis
        # is the reference's sightline to the landmark; prev_kf's sightline is
        # checked against it. Reference itself is by construction in sector 1.
        cam_r = self.reference.t
        cam_p = self.prev_kf.t
        Or = 0
        for i, j in zip(idx_r, idx_p):
            wp = self.reference.world_pts[i]
            if not np.isfinite(wp[0]):
                wp = self.prev_kf.world_pts[j]
                if not np.isfinite(wp[0]):
                    continue
            principal = wp - cam_r
            view_p = wp - cam_p
            sec_p = assign_sector(view_p, principal, self.cfg.sector_angle_deg)
            if sec_p != 1:
                Or += 1
        self.Or_cached = Or

    # ------------------------------------------------------------------
    # Stats computation against the live frame
    # ------------------------------------------------------------------
    def _compute_stats(self, current: _FrameMemo) -> KeyframeStats:
        assert self.prev_kf is not None and self.reference is not None

        # Dc = total extracted features in current (paper).
        Dc = int(len(current.kp_uv))

        idx_c, idx_p = self._match(current.desc, self.prev_kf.desc)
        Mc = len(idx_c)

        if Mc == 0:
            return KeyframeStats(Mc=0, Mr=self.Mr_cached, Dc=Dc, Dr=self.Dr_cached,
                                 Or_=self.Or_cached, Oc=0, N=self.N, s=0.0)

        c_pts = current.world_pts[idx_c]
        p_pts = self.prev_kf.world_pts[idx_p]
        valid_c = np.isfinite(c_pts[:, 0])
        valid_p = np.isfinite(p_pts[:, 0])

        # Chamfer between the two matched point clouds (use only mutually valid).
        # Paper Eq. 3: lambda = 1 / |P_2|, i.e. inverse of the current-frame
        # point count -- NOT a free hyperparameter.
        ok = valid_c & valid_p
        n_ok = int(ok.sum())
        if n_ok >= 3:
            d_cham = symmetric_chamfer(c_pts[ok], p_pts[ok])
            s = float(np.exp(-d_cham / n_ok))
        else:
            s = 0.0

        # Oc: sector migrations between current and prev_kf for matched landmarks
        cam_c = current.t
        cam_p = self.prev_kf.t
        cam_r = self.reference.t
        # Build a lookup for "this prev_kf feature was matched to reference at index ir"
        ref_for_p = {}
        if self.ref_pkmatches is not None:
            ir, ip = self.ref_pkmatches
            for a, b in zip(ir, ip):
                ref_for_p[int(b)] = int(a)

        Oc = 0
        for k in range(Mc):
            wp = c_pts[k] if valid_c[k] else p_pts[k] if valid_p[k] else None
            if wp is None:
                continue
            # Principal axis: paper-strict is "reference camera's sightline to
            # the landmark as the reference sees it" (Sec. III.A.3). Use the
            # reference's own 3D point when this landmark is in the ref-match
            # set; otherwise fall back to prev_kf-anchored. Using `wp` (which
            # comes from current/prev_kf depth) for the axis would introduce
            # depth-noise into the principal axis, not paper-correct.
            i_ref = ref_for_p.get(int(idx_p[k]))
            principal = None
            if i_ref is not None:
                ref_wp = self.reference.world_pts[i_ref]
                if np.isfinite(ref_wp[0]):
                    principal = ref_wp - cam_r
            if principal is None:
                principal = wp - cam_p   # fallback: prev_kf-anchored
            view_p = wp - cam_p
            view_c = wp - cam_c
            sec_p = assign_sector(view_p, principal, self.cfg.sector_angle_deg)
            sec_c = assign_sector(view_c, principal, self.cfg.sector_angle_deg)
            if sec_c != sec_p:
                Oc += 1

        return KeyframeStats(
            Mc=Mc, Mr=self.Mr_cached,
            Dc=Dc, Dr=self.Dr_cached,
            Or_=self.Or_cached, Oc=Oc,
            N=self.N, s=s,
        )

    # ------------------------------------------------------------------
    # Glue
    # ------------------------------------------------------------------
    def _build_memo(self, gray: np.ndarray, depth: np.ndarray,
                    t: np.ndarray, R: np.ndarray) -> Optional[_FrameMemo]:
        kps, desc = self.orb.detectAndCompute(gray, None)
        if desc is None or len(kps) == 0:
            return None
        uv = np.array([kp.pt for kp in kps], dtype=np.float32)
        world = backproject(
            uv, depth, self.K_inv,
            np.asarray(t, dtype=np.float32),
            np.asarray(R, dtype=np.float32),
            self.cfg.min_depth, self.cfg.max_depth,
        )
        return _FrameMemo(
            rgb_gray=gray,
            depth=np.asarray(depth, dtype=np.float32),
            t=np.asarray(t, dtype=np.float32),
            R=np.asarray(R, dtype=np.float32),
            kp_uv=uv,
            desc=desc,
            world_pts=world,
        )

    def _match(self, desc_a: np.ndarray, desc_b: np.ndarray
               ) -> tuple[np.ndarray, np.ndarray]:
        """ORB descriptor matching with cross-check; returns (idx_a, idx_b)."""
        if desc_a is None or desc_b is None or len(desc_a) == 0 or len(desc_b) == 0:
            return np.empty(0, int), np.empty(0, int)
        ms = self.matcher.match(desc_a, desc_b)
        if not ms:
            return np.empty(0, int), np.empty(0, int)
        ia = np.array([m.queryIdx for m in ms], dtype=np.int32)
        ib = np.array([m.trainIdx for m in ms], dtype=np.int32)
        return ia, ib

    @staticmethod
    def _to_gray(rgb: np.ndarray) -> np.ndarray:
        if rgb.ndim == 2:
            return rgb if rgb.dtype == np.uint8 else rgb.astype(np.uint8)
        # OpenCV reads BGR; for the matcher it doesn't matter which channel order
        # we collapse from, but we standardize on a clean uint8 gray.
        g = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY) if rgb.shape[2] == 3 else rgb[..., 0]
        return g.astype(np.uint8) if g.dtype != np.uint8 else g


def _empty_memo(gray, depth, t, R) -> _FrameMemo:
    return _FrameMemo(
        rgb_gray=gray, depth=np.asarray(depth, np.float32),
        t=np.asarray(t, np.float32), R=np.asarray(R, np.float32),
        kp_uv=np.empty((0, 2), np.float32),
        desc=np.empty((0, 32), np.uint8),
        world_pts=np.empty((0, 3), np.float32),
    )


# =============================================================================
# Smoke test
# =============================================================================

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    H, W = 240, 320
    fx = fy = 0.5 * W / np.tan(np.deg2rad(70.0) / 2)
    K = np.array([[fx, 0, W / 2], [0, fy, H / 2], [0, 0, 1]], np.float32)
    cfg = NurbsLviConfig(orb_n_features=400, sector_angle_deg=2.0)
    sel = NurbsLviSelector(cfg, K, (H, W))

    def fake_rgb(seed: int) -> np.ndarray:
        return rng.integers(0, 255, (H, W, 3), dtype=np.uint8)

    def fake_depth() -> np.ndarray:
        return np.full((H, W), 3.0, dtype=np.float32)

    yaw = lambda a: np.array([
        [np.cos(a), 0, np.sin(a)],
        [0, 1, 0],
        [-np.sin(a), 0, np.cos(a)],
    ], dtype=np.float32)

    accepted = 0
    for i, ang in enumerate(np.linspace(0, np.pi, 30)):
        t = np.array([0.1 * i, 0.0, 0.0], np.float32)
        R = yaw(ang).astype(np.float32)
        ok, sc = sel.should_accept(fake_depth(), t, R, fake_rgb(i))
        accepted += int(ok)
        if sc is not None and not sc.forced:
            print(f"frame {i:2d} t={t[0]:+.2f} yaw={np.degrees(ang):+6.1f}°  "
                  f"Q={sc.Q:+.3f} mig={sc.migration:+.1f} matches={sc.n_matches}  "
                  f"{'ACCEPT' if ok else 'skip'}")
    print(f"\nTotal accepted: {accepted}/30")
