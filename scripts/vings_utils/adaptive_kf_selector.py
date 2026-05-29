"""
Adaptive-Keyframe-Selector (Momentum-Aware Adaptive Threshold over a hybrid
photometric+SSIM error).

Implements:
  "Adaptive Keyframe Selection for Scalable 3D Scene Reconstruction in
   Dynamic Environments" -- Jha, Zhou, Loianno, arXiv:2510.23928v3, 2025.

What is faithful to the paper
-----------------------------
Algorithm 1 (Hybrid Error, Sec. 3.2):
    Forward-warp I_kf into the current view, driven by D_kf
    (paper Algorithm 1 signature: WarpFrame(Ik, Dk, Posek, Poset, Intrinsics)
    -- the depth map of the LAST KEYFRAME, not of the current frame).
    e_photo  = (1/|M|) * sum_{p in M} |I_t(p) - Î_kf(p)|       # Eq. 1
    e_ssim   = 1 - SSIM(I_t, Î_kf)                              # Eq. 2
    e_t      = alpha * e_photo + beta * e_ssim                   # Eq. 3
    Defaults:  alpha = 0.7, beta = 0.3                           # paper Sec. 3.4

Algorithm 2 (Momentum-Aware Threshold, Sec. 3.3):
    e_t = ComputeHybridError(f_t, f_last_kf)
    E.append(e_t)
    if |E| >= W:                                                 # Eq. 6
        theta = max(theta0, mean(E[-W:]) + k * std(E[-W:]))
    else:                                                        # warm-up, Alg. 2 line 11
        theta = theta0 * (t/W) + theta_init * (1 - t/W)
    if e_t > theta:
        accept f_t
        theta = gamma * theta                                    # Eq. 7
    Defaults:  W = 5, k = 1.5, gamma = 0.95, theta0 = 0.05       # paper Sec. 3.4

Implementation choices not pinned down by the paper
---------------------------------------------------
* Forward-warping with Z-buffer: each KF pixel is back-projected with D_kf and
  splatted into the current view; we paint farthest-first so the nearest depth
  wins per output pixel. The paper says only "project 3D points derived from
  the depth map of Ik into the image plane of It"; the Z-buffer is the standard
  way to resolve multi-source collisions and is consistent with the paper's
  "valid pixels in It that have a valid correspondence in Ik" mask definition.
* Pixel intensities scaled to [0, 1] (float32). Paper does not specify; this
  matches the magnitude of theta0 = 0.05 from the paper's grid search.
* SSIM via skimage on mask-multiplied images (skimage has no masked SSIM).
  Non-overlap region contributes SSIM = 1 (both images = 0) and is thus a
  no-op in the dissimilarity sum.
* theta_init is referenced by Algorithm 2 line 11 but not given a value in
  Sec. 3.4. We default to 0.10 (2x theta0) so the warm-up starts conservative.
* min_overlap_pixels fail-safe: when |M| drops below this, accept and push
  e = 1.0 to the history (matches the paper's "fail-safe trigger" note in
  the discussion of Sec. 5 limitations).

These choices are documented in `docs/ADAPTIVE_KF.md`.

Convention
----------
Same as the other selectors:
    sel = AdaptiveKfSelector(cfg, K, (H, W))
    accept, score = sel.should_accept(depth, t, R, rgb=rgb_uint8)

Poses (t, R) are c2w:  p_world = R @ p_cam + t.
`depth` is (H, W) float32 metres; `rgb` is (H, W, 3) or (H, W) uint8.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    import cv2  # noqa: F401  (used elsewhere in the project; kept for parity)
except ImportError as e:
    raise ImportError("AdaptiveKfSelector requires opencv-python.") from e

# SSIM is optional; we fall back to a photometric-only error if scikit-image
# is missing. The selector logs `e_ssim = 0.0` in that case.
try:
    from skimage.metrics import structural_similarity as _ssim
    _HAS_SSIM = True
except ImportError:
    _HAS_SSIM = False


# =============================================================================
# Config / score
# =============================================================================

@dataclass
class AdaptiveKfConfig:
    # Algorithm 2 -- paper Sec. 3.4 defaults --------------------------------
    theta0: float = 0.05        # base threshold (grid-searched on Bonn val)
    theta_init: float = 0.10    # warm-up initial value (Alg. 2 line 11)
    window_size: int = 5        # W (paper: "small window size W = 5")
    sensitivity: float = 1.5    # k (multiplier on rolling std)
    decay: float = 0.95         # gamma in (0, 1], applied after accept (Eq. 7)

    # Algorithm 1 -- paper Eq. 3 defaults -----------------------------------
    w_photo: float = 0.7        # alpha (paper Eq. 3)
    w_ssim: float = 0.3         # beta  (paper Eq. 3)
    min_overlap_pixels: int = 1000  # fail-safe accept if |M| < this

    # Depth gating (same convention as the other selectors)
    min_depth: float = 0.2
    max_depth: float = 35.0

    # Failsafe / diagnostic
    force_accept_all: bool = False  # log e/theta but always accept


@dataclass
class AdaptiveKfScore:
    e: float = 0.0
    e_photo: float = 0.0
    e_ssim: float = 0.0
    theta: float = 0.0
    n_valid: int = 0
    forced: bool = False     # bootstrap / fail-safe / diagnostic
    warmup: bool = False
    accepted: bool = False


# =============================================================================
# Selector
# =============================================================================

class AdaptiveKfSelector:
    """Momentum-aware adaptive-threshold keyframe selector."""

    def __init__(self, cfg: AdaptiveKfConfig, K: np.ndarray,
                 image_hw: tuple[int, int]):
        self.cfg = cfg
        self.K = np.asarray(K, dtype=np.float32)
        self.K_inv = np.linalg.inv(self.K).astype(np.float32)
        self.H, self.W = image_hw

        # Last accepted KF state
        self.last_kf_gray: Optional[np.ndarray] = None
        self.last_kf_depth: Optional[np.ndarray] = None
        self.last_kf_R: Optional[np.ndarray] = None
        self.last_kf_t: Optional[np.ndarray] = None

        # Error history & current threshold
        self.E: deque[float] = deque(maxlen=cfg.window_size)
        self.theta: float = float(cfg.theta0)
        self.frame_idx: int = 0

        # Precompute pixel grid for warping (one-shot allocation)
        v_idx, u_idx = np.meshgrid(np.arange(self.H), np.arange(self.W),
                                   indexing="ij")
        self._pix_hom = np.stack(
            [u_idx.ravel().astype(np.float32),
             v_idx.ravel().astype(np.float32),
             np.ones(self.H * self.W, dtype=np.float32)],
            axis=0,
        )  # (3, H*W)

    # ------------------------------------------------------------------
    @classmethod
    def from_config(cls, cfg_dict: dict, K: np.ndarray,
                    image_hw: tuple[int, int]) -> "AdaptiveKfSelector":
        fields = set(AdaptiveKfConfig.__dataclass_fields__.keys())
        kwargs = {k: v for k, v in cfg_dict.items() if k in fields}
        return cls(AdaptiveKfConfig(**kwargs), K, image_hw)

    # ------------------------------------------------------------------
    def should_accept(
        self,
        depth: np.ndarray,
        t: np.ndarray,
        R: np.ndarray,
        rgb: Optional[np.ndarray] = None,
        **_: object,
    ) -> tuple[bool, AdaptiveKfScore]:
        self.frame_idx += 1
        score = AdaptiveKfScore(theta=self.theta)

        if rgb is None:
            # No RGB -> we can't compute the hybrid error. Accept and log.
            score.forced = True
            score.accepted = True
            return True, score

        gray = self._to_gray(rgb)
        depth_f = np.asarray(depth, dtype=np.float32)
        t_arr = np.asarray(t, dtype=np.float32).reshape(3)
        R_arr = np.asarray(R, dtype=np.float32).reshape(3, 3)

        # Bootstrap: first frame seeds the KF and is always accepted.
        if self.last_kf_gray is None:
            self._commit(gray, depth_f, R_arr, t_arr)
            score.forced = True
            score.accepted = True
            return True, score

        # ----- Algorithm 1: hybrid error (forward-warp I_kf via D_kf)
        e_photo, e_ssim, n_valid = self._hybrid_error(gray, R_arr, t_arr)
        score.n_valid = n_valid
        score.e_photo = float(e_photo)
        score.e_ssim = float(e_ssim)

        # Fail-safe: rapid motion / mostly-invalid KF depth -> accept and reset.
        if n_valid < self.cfg.min_overlap_pixels:
            score.e = float("inf")
            score.forced = True
            score.accepted = True
            self._commit(gray, depth_f, R_arr, t_arr)
            # Push a representative high error so the window doesn't go stale.
            self.E.append(1.0)
            return True, score

        e = self.cfg.w_photo * e_photo + self.cfg.w_ssim * e_ssim
        score.e = float(e)
        self.E.append(e)

        # ----- Algorithm 2: adaptive threshold
        W = self.cfg.window_size
        if len(self.E) >= W:
            arr = np.asarray(self.E, dtype=np.float64)
            mu = float(arr.mean())
            sigma = float(arr.std())  # paper Eq. 5 uses 1/W, np.std default ddof=0 matches
            theta = max(self.cfg.theta0,
                        mu + self.cfg.sensitivity * sigma)
        else:
            t_ratio = self.frame_idx / float(W)
            theta = (self.cfg.theta0 * t_ratio
                     + self.cfg.theta_init * (1.0 - t_ratio))
            score.warmup = True

        self.theta = float(theta)
        score.theta = self.theta

        accept = (e > self.theta) or self.cfg.force_accept_all
        if self.cfg.force_accept_all:
            score.forced = True

        if accept:
            self.theta = self.cfg.decay * self.theta   # Eq. 7 decay
            self._commit(gray, depth_f, R_arr, t_arr)
            score.accepted = True

        return accept, score

    # ------------------------------------------------------------------
    def _commit(self, gray: np.ndarray, depth: np.ndarray,
                R: np.ndarray, t: np.ndarray) -> None:
        self.last_kf_gray = gray.copy()
        self.last_kf_depth = depth.copy()
        self.last_kf_R = R.copy()
        self.last_kf_t = t.copy()

    # ------------------------------------------------------------------
    def _hybrid_error(self, gray_cur: np.ndarray,
                      R_cur: np.ndarray, t_cur: np.ndarray
                      ) -> tuple[float, float, int]:
        """Paper-faithful forward warp: I_kf is splatted into the current view
        using D_kf. Each KF pixel is back-projected to KF cam frame with D_kf,
        transformed to the current cam, projected with K, and painted with a
        Z-buffer (farthest first, so nearest depth wins per output pixel).
        Returns (e_photo, e_ssim, |M|)."""
        H, W = self.H, self.W

        d_kf = self.last_kf_depth.ravel()
        valid_kf = (np.isfinite(d_kf)
                    & (d_kf > self.cfg.min_depth)
                    & (d_kf < self.cfg.max_depth))
        if not np.any(valid_kf):
            return 0.0, 0.0, 0

        # Relative pose: kf cam -> cur cam (c2w composition).
        #   p_world = R_kf @ p_kf + t_kf
        #   p_cur   = R_cur^T @ (p_world - t_cur)
        #           = R_cur^T @ R_kf @ p_kf + R_cur^T @ (t_kf - t_cur)
        R_kc = R_cur.T @ self.last_kf_R              # (3, 3)
        t_kc = R_cur.T @ (self.last_kf_t - t_cur)    # (3,)

        rays = self.K_inv @ self._pix_hom            # (3, H*W) in KF cam frame
        pts_kf = rays * d_kf[None, :]                # 3D points in KF cam
        pts_cur = R_kc @ pts_kf + t_kc[:, None]      # (3, H*W) in cur cam
        z_cur = pts_cur[2, :]

        ok_z = z_cur > 1e-6
        z_safe = np.where(ok_z, z_cur, 1.0)
        proj = self.K @ pts_cur
        u_cur = proj[0, :] / z_safe
        v_cur = proj[1, :] / z_safe

        in_bounds = ((u_cur >= 0) & (u_cur < W - 1)
                     & (v_cur >= 0) & (v_cur < H - 1))
        src_valid = valid_kf & ok_z & in_bounds
        if not np.any(src_valid):
            return 0.0, 0.0, 0

        # Forward-splat with Z-buffer: paint farthest first so the nearest
        # depth wins per output pixel. argsort is O(N log N) but N = H*W is
        # the only pass we need here.
        idx = np.where(src_valid)[0]
        z_vals = z_cur[idx]
        order = np.argsort(-z_vals)                  # descending z (far first)
        u_int = np.clip(np.round(u_cur[idx][order]).astype(np.int64),
                        0, W - 1)
        v_int = np.clip(np.round(v_cur[idx][order]).astype(np.int64),
                        0, H - 1)

        src_gray = self.last_kf_gray.ravel().astype(np.float32) / 255.0
        g_vals = src_gray[idx][order]

        warped = np.zeros((H, W), dtype=np.float32)
        mask_2d = np.zeros((H, W), dtype=bool)
        warped[v_int, u_int] = g_vals
        mask_2d[v_int, u_int] = True

        n_valid = int(mask_2d.sum())
        if n_valid < 4:
            return 0.0, 0.0, n_valid

        cur_f = gray_cur.astype(np.float32) / 255.0
        diff = np.abs(warped - cur_f)
        e_photo = float(diff[mask_2d].mean())

        if _HAS_SSIM:
            cur_m = cur_f * mask_2d
            warped_m = warped * mask_2d
            s = float(_ssim(cur_m, warped_m, data_range=1.0))
            e_ssim = float(np.clip(1.0 - s, 0.0, 2.0))
        else:
            e_ssim = 0.0

        return e_photo, e_ssim, n_valid

    # ------------------------------------------------------------------
    @staticmethod
    def _to_gray(rgb: np.ndarray) -> np.ndarray:
        if rgb.ndim == 2:
            return rgb if rgb.dtype == np.uint8 else rgb.astype(np.uint8)
        if rgb.shape[2] == 3:
            import cv2 as _cv2
            return _cv2.cvtColor(rgb, _cv2.COLOR_BGR2GRAY).astype(np.uint8)
        return rgb[..., 0].astype(np.uint8)


# =============================================================================
# Smoke test
# =============================================================================

if __name__ == "__main__":
    # Synthetic 64x64 textured scene at z=2m, camera translates along +x.
    # Expect: error rises with motion, threshold ramps from warm-up, after a
    # few frames the selector picks roughly every few frames as KFs.
    rng = np.random.default_rng(0)
    H = W = 64
    fx = fy = 64.0
    cx = cy = 32.0
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

    # A static "wall" of dense texture at constant depth.
    texture = (rng.uniform(0, 255, size=(256, 256)).astype(np.uint8))

    def render(tx: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Render the wall as seen by a camera at (tx, 0, 0) looking +z, plane z=2m."""
        depth = 2.0 * np.ones((H, W), dtype=np.float32)
        v, u = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
        x_cam = (u - cx) / fx * 2.0
        y_cam = (v - cy) / fy * 2.0
        x_world = x_cam + tx
        y_world = y_cam
        tu = np.clip(((x_world + 1.0) * 128).astype(int), 0, 255)
        tv = np.clip(((y_world + 1.0) * 128).astype(int), 0, 255)
        gray = texture[tv, tu]
        rgb = np.stack([gray, gray, gray], axis=-1)
        R = np.eye(3, dtype=np.float32)
        t = np.array([tx, 0.0, 0.0], dtype=np.float32)
        return rgb, depth, t, R

    cfg = AdaptiveKfConfig(theta0=0.01, theta_init=0.05,
                            window_size=5, sensitivity=1.5, decay=0.95)
    sel = AdaptiveKfSelector(cfg, K, (H, W))

    print(f"SSIM available: {_HAS_SSIM}")
    n_accept = 0
    for i, tx in enumerate(np.linspace(0.0, 0.5, 30)):
        rgb, depth, t, R = render(float(tx))
        accept, sc = sel.should_accept(depth, t, R, rgb=rgb)
        n_accept += int(accept)
        flag = "KF" if accept else "  "
        marker = ""
        if sc.warmup:
            marker = "(warmup)"
        elif sc.forced:
            marker = "(forced)"
        print(f"  i={i:2d} tx={tx:+.3f} {flag} e={sc.e:.4f} "
              f"theta={sc.theta:.4f} |M|={sc.n_valid:5d}  {marker}")

    print(f"\nAccepted {n_accept}/30 frames.")
    print("Expect ~1 KF at start (bootstrap) + a handful as motion grows.")
