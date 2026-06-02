"""
TwoGateV2Selector -- Gate B (post-tracker, pre-mapper) keyframe selector, v2.

Drop-in successor to `two_gate_selector.TwoGateSelector` (v1). v1 stays
untouched -- both are selectable via the `frame_selector.kind` switch
(`two_gate` vs. `two_gate_v2`).

What changed vs. v1
-------------------
The GPS-distance check has been removed from B1 and **moved upstream into
Gate A v2 (A3)**, where it can save the ~450 ms tracker cost on stationary
frames instead of running after the tracker has already paid for them.

Concretely:
  * B1 is now **pose-translation only + SSIM-veto**. No `xyz_enu` reads,
    no `gps_d_min_m`, no `gps_noise_floor_m`. The SSIM veto stays
    because it independently catches scale-collapsed tracker poses that
    report motion without scene change (a v1-only concern but a safe
    keep).
  * `TwoGateScore.gps_d_m` is gone.
  * `_commit()` no longer tracks `prev_kf_xyz`.
  * `meta=` is still accepted in `should_accept()` for signature
    compatibility but is no longer read.

B2 (covisibility), B3 (DINO novelty), the adaptive theta, and the budget
cap are all unchanged from v1 -- this file is intentionally minimally
diverged so a side-by-side comparison stays meaningful.

For the rest of the design see the v1 docstring in
`two_gate_selector.py` and `docs/TWO_GATE.md` (still applies to v2 modulo
the B1 change).

Interface
---------
    sel = TwoGateV2Selector.from_config(cfg, K, (H, W))
    accept, score = sel.should_accept(
        depth, t, R, rgb=rgb_uint8, depth_cov=cov, meta=None,
    )
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional, Mapping, Any

import numpy as np

try:
    import cv2  # noqa: F401  (used transitively by SSIM resize)
except ImportError as e:
    raise ImportError("TwoGateV2Selector requires opencv-python.") from e


# =============================================================================
# Config / score
# =============================================================================

@dataclass
class TwoGateV2Config:
    # ---- B1 motion (pose-translation or gps-distance + optional SSIM veto)
    # b1_motion_source picks WHICH distance B1 gates on:
    #   "pose" (default) -> tracker-BA translation (arbitrary SLAM scale, model-only)
    #   "gps"            -> GPS/ENU distance from meta['xyz_enu'] (metric); falls
    #                        back to pose when no GPS is available so it never
    #                        silently passes every frame.
    b1_motion_source: str = "pose"       # "pose" | "gps"
    pose_d_min_m: float = 0.15           # threshold when source == "pose"
    gps_d_min_m: float = 0.5             # threshold [m] when source == "gps"
    visual_change_max_ssim: float = 0.98
    ssim_resize: int = 80                # downsample short side for cheap SSIM
    enable_ssim_veto: bool = True        # off -> pure motion-distance B1

    # ---- B2 covisibility -------------------------------------------------
    covis_thresh: float = 0.85
    n_samples_covis: int = 2048
    min_depth: float = 0.2
    max_depth: float = 60.0

    # ---- B3 DINO content novelty ----------------------------------------
    enable_b3: bool = True
    alpha: float = 0.35
    dino_model: str = "dinov2_vits14"
    dino_image_size: int = 224
    dino_device: str = "cuda"
    dino_max_kfs: int = 10

    # ---- Adaptive threshold ---------------------------------------------
    theta0: float = 0.30
    theta_init: float = 0.35
    window_size: int = 30
    sensitivity: float = 0.5
    decay: float = 0.85

    # ---- Budget ----------------------------------------------------------
    min_spacing: int = 1
    max_per_window: int = 6
    rate_window: int = 30
    force_after: int = 50

    # ---- Logging ---------------------------------------------------------
    verbose: bool = False    # True -> print a per-call decision line to stdout
    log_skips_only: bool = False  # with verbose: only log rejected frames


@dataclass
class TwoGateV2Score:
    b1_pass: bool = False
    pose_d_m: float = 0.0
    gps_d_m: float = 0.0
    visual_ssim: float = 1.0
    b2_pass: bool = False
    covis: float = 1.0
    b3_pass: bool = False
    dino_d_min: float = 0.0
    composite: float = 0.0
    theta: float = 0.0
    forced: bool = False
    budget_blocked: bool = False
    triggered_by: str = ""
    accepted: bool = False


# =============================================================================
# Optional SSIM (skimage) -- same fallback pattern as adaptive_kf_selector
# =============================================================================

try:
    from skimage.metrics import structural_similarity as _ssim
    _HAS_SSIM = True
except ImportError:
    _HAS_SSIM = False


def _cheap_ssim_gray(a: np.ndarray, b: np.ndarray) -> float:
    """SSIM on uint8 grayscale; falls back to 1 - mean|diff|/255 if skimage missing."""
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA)
    if _HAS_SSIM:
        return float(_ssim(a, b, data_range=255))
    return float(1.0 - np.mean(np.abs(a.astype(np.int16) - b.astype(np.int16))) / 255.0)


# =============================================================================
# Selector
# =============================================================================

class TwoGateV2Selector:
    """Hierarchical Gate-B keyframe selector v2 (no GPS in B1)."""

    def __init__(self, cfg: TwoGateV2Config, K: np.ndarray,
                 image_hw: tuple[int, int]):
        self.cfg = cfg
        self.K = np.asarray(K, dtype=np.float32)
        self.K_inv = np.linalg.inv(self.K)
        self.H, self.W = image_hw

        # ---- B2: Mm3dgs covisibility helper, accept-logic neutralised -----
        from vings_utils.mm3dgs_selector import Mm3dgsConfig, Mm3dgsSelector
        self._covis_helper = Mm3dgsSelector(
            Mm3dgsConfig(
                covis_thresh=1.01,         # nothing can be < 1.01 -> never auto-accept
                n_samples=cfg.n_samples_covis,
                min_depth=cfg.min_depth,
                max_depth=cfg.max_depth,
                niqe_window=1,
                force_accept_after=0,
            ),
            K, image_hw,
        )

        # ---- B3: Coko DINO helper, accept-logic neutralised ---------------
        if cfg.enable_b3:
            from vings_utils.coko_slam_selector import CokoSlamConfig, CokoSlamSelector
            self._dino_helper = CokoSlamSelector(
                CokoSlamConfig(
                    alpha=0.0,             # any distance counts as "accept" internally
                    model_name=cfg.dino_model,
                    image_size=cfg.dino_image_size,
                    device=cfg.dino_device,
                    max_kfs=cfg.dino_max_kfs,
                    force_accept_all=True, # diagnostic mode -- we drive accept ourselves
                ),
                K, image_hw,
            )
        else:
            self._dino_helper = None

        # ---- B1 state (no GPS in v2) ------------------------------------
        self.prev_kf_t:   Optional[np.ndarray] = None  # (3,)
        self.prev_kf_R:   Optional[np.ndarray] = None  # (3,3)
        self.prev_kf_xyz: Optional[np.ndarray] = None  # ENU (3,), for gps-based B1
        self.prev_kf_rgb_small: Optional[np.ndarray] = None  # gray, uint8

        # ---- Adaptive theta state ---------------------------------------
        self._score_buf: deque[float] = deque(maxlen=cfg.window_size)
        self._theta = float(cfg.theta_init)

        # ---- Budget state ------------------------------------------------
        self._frames_since_kf = 0
        self._accept_history: deque[int] = deque(maxlen=cfg.rate_window)

        # ---- Logging state -----------------------------------------------
        self._call_idx = -1  # incremented at the top of each should_accept

    @classmethod
    def from_config(cls, cfg_dict: Mapping[str, Any], K: np.ndarray,
                    image_hw: tuple[int, int]) -> "TwoGateV2Selector":
        fields_ = set(TwoGateV2Config.__dataclass_fields__.keys())
        kwargs = {k: v for k, v in cfg_dict.items() if k in fields_}
        return cls(TwoGateV2Config(**kwargs), K, image_hw)

    # ------------------------------------------------------------------
    def should_accept(
        self,
        depth: np.ndarray,
        t: np.ndarray,
        R: np.ndarray,
        rgb: Optional[np.ndarray] = None,
        depth_cov: Optional[np.ndarray] = None,
        meta: Optional[Mapping[str, Any]] = None,  # accepted but unused in v2
        **_: object,
    ) -> tuple[bool, TwoGateV2Score]:
        score = TwoGateV2Score(theta=self._theta)
        meta = meta or {}
        self._frames_since_kf += 1
        self._call_idx += 1

        t = np.asarray(t, dtype=np.float32)
        R = np.asarray(R, dtype=np.float32)

        # ---- Bootstrap ---------------------------------------------------
        if self.prev_kf_t is None:
            self._commit(t, R, rgb, depth, meta)
            score.b1_pass = score.b2_pass = score.b3_pass = True
            score.forced = True
            score.accepted = True
            score.triggered_by = "first"
            self._log(score)
            return True, score

        # ---- B1 motion (pose-translation OR gps-distance + optional SSIM veto)
        # b1_motion_source selects which distance gates B1. "pose" = tracker-BA
        # translation (default); "gps" = metric GPS distance from meta['xyz_enu']
        # (the same A3 signal, but applied here at Gate B / pre-mapper). GPS falls
        # back to pose when meta has no xyz_enu so it never passes everything.
        score.pose_d_m = float(np.linalg.norm(t - self.prev_kf_t))
        if self.cfg.b1_motion_source == "gps":
            gps_xyz = meta.get("xyz_enu")
            if gps_xyz is not None and self.prev_kf_xyz is not None:
                score.gps_d_m = float(np.linalg.norm(
                    np.asarray(gps_xyz, dtype=np.float32) - self.prev_kf_xyz))
                motion_ok = score.gps_d_m >= self.cfg.gps_d_min_m
                b1_reason = "" if motion_ok else "B1_gps_below_min"
            else:
                # no GPS this call -> fall back to pose so B1 still gates
                motion_ok = score.pose_d_m >= self.cfg.pose_d_min_m
                b1_reason = "" if motion_ok else "B1_pose_below_min(no_gps)"
        else:
            motion_ok = score.pose_d_m >= self.cfg.pose_d_min_m
            b1_reason = "" if motion_ok else "B1_pose_below_min"

        if (motion_ok
                and self.cfg.enable_ssim_veto
                and rgb is not None
                and self.prev_kf_rgb_small is not None):
            curr_small = self._small_gray(rgb)
            score.visual_ssim = _cheap_ssim_gray(curr_small, self.prev_kf_rgb_small)
            if score.visual_ssim > self.cfg.visual_change_max_ssim:
                motion_ok = False
                b1_reason = "B1_ssim_veto"
        score.b1_pass = motion_ok

        # ---- B2 covisibility -------------------------------------------
        if self._covis_helper.prev_kf_depth is None:
            score.covis = 0.0
        else:
            score.covis = self._covis_helper._covisibility(t, R)
        score.b2_pass = score.covis < self.cfg.covis_thresh
        b2_novelty = float(np.clip(1.0 - score.covis, 0.0, 1.0))

        # ---- B3 DINO content novelty ------------------------------------
        b3_novelty = 0.0
        if self._dino_helper is not None and rgb is not None:
            import torch
            feat = self._dino_helper._extract(rgb)
            if self._dino_helper.kf_features:
                refs = torch.stack(self._dino_helper.kf_features, dim=0)
                d_min = float(torch.norm(feat[None, :] - refs, dim=1).min().item())
            else:
                d_min = 2.0  # empty FIFO -> max novelty
            score.dino_d_min = d_min
            score.b3_pass = d_min >= self.cfg.alpha
            b3_novelty = float(min(d_min, 2.0) / 2.0)
        elif self._dino_helper is None:
            score.b3_pass = True

        # ---- Composite novelty ------------------------------------------
        if self._dino_helper is not None:
            score.composite = 0.5 * b2_novelty + 0.5 * b3_novelty
        else:
            score.composite = b2_novelty

        # ---- Adaptive theta update (BEFORE the decision) ----------------
        self._score_buf.append(score.composite)
        W = max(int(self.cfg.window_size), 1)
        if len(self._score_buf) >= W:
            arr = np.fromiter(self._score_buf, dtype=np.float32)
            self._theta = float(
                max(self.cfg.theta0, arr.mean() + self.cfg.sensitivity * arr.std())
            )
        else:
            ratio = len(self._score_buf) / float(W)
            self._theta = (
                self.cfg.theta0 * ratio + self.cfg.theta_init * (1.0 - ratio)
            )
        score.theta = self._theta

        # ---- Decision ---------------------------------------------------
        accept = score.b1_pass and (score.composite >= self._theta)
        if accept:
            score.triggered_by = "motion+novelty"
        elif not score.b1_pass:
            # Frame died at the motion gate.
            score.triggered_by = b1_reason or "B1_motion"
        else:
            # B1 passed but the scene was not novel enough.
            score.triggered_by = "B2B3_novelty_below_theta"

        if not accept and self._frames_since_kf >= self.cfg.force_after:
            accept = True
            score.forced = True
            score.triggered_by = "force_after"

        # ---- Budget ------------------------------------------------------
        if accept and self._frames_since_kf < self.cfg.min_spacing:
            accept = False
            score.budget_blocked = True
            score.triggered_by = "spacing_blocked"
        if (
            accept
            and self.cfg.max_per_window > 0
            and sum(self._accept_history) >= self.cfg.max_per_window
        ):
            accept = False
            score.budget_blocked = True
            score.triggered_by = "rate_capped"

        self._accept_history.append(1 if accept else 0)

        if accept:
            self._commit(t, R, rgb, depth, meta)
            score.accepted = True
            self._theta = float(self._theta * self.cfg.decay)
            score.theta = self._theta

        self._log(score)
        return accept, score

    # ------------------------------------------------------------------
    def _commit(
        self,
        t: np.ndarray,
        R: np.ndarray,
        rgb: Optional[np.ndarray],
        depth: np.ndarray,
        meta: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Update internal + helper state for a newly accepted keyframe.

        Same direct-write into Mm3dgs helper attributes as v1. prev_kf_xyz is
        tracked only for gps-based B1 (b1_motion_source == "gps").
        """
        self.prev_kf_t = t.copy()
        self.prev_kf_R = R.copy()

        gps_xyz = (meta or {}).get("xyz_enu")
        if gps_xyz is not None:
            self.prev_kf_xyz = np.asarray(gps_xyz, dtype=np.float32).copy()

        self._covis_helper.prev_kf_R = R.copy()
        self._covis_helper.prev_kf_t = t.copy()
        self._covis_helper.prev_kf_depth = np.asarray(depth, dtype=np.float32).copy()

        if self._dino_helper is not None and rgb is not None:
            feat = self._dino_helper._extract(rgb)
            self._dino_helper._commit(feat)

        if rgb is not None:
            self.prev_kf_rgb_small = self._small_gray(rgb)

        self._frames_since_kf = 0

    # ------------------------------------------------------------------
    def _log(self, score: "TwoGateV2Score") -> None:
        """Print a one-line decision trace if cfg.verbose.

        The line names the exact pipeline step that decided the frame's
        fate via `score.triggered_by`:
          first                      -> bootstrap KF (always mapped)
          B1_pose_below_min          -> B1 motion gate: tracker-pose move too small
          B1_gps_below_min           -> B1 motion gate: GPS move too small (b1_motion_source=gps)
          B1_ssim_veto               -> B1 motion gate: image ~identical (noise/scale-collapse)
          B2B3_novelty_below_theta   -> passed B1, but composite covis+DINO novelty < theta
          force_after                -> failsafe accept after starvation
          spacing_blocked / rate_capped -> would accept, blocked by budget cap
          motion+novelty             -> accepted on its own merits
        """
        if not self.cfg.verbose:
            return
        if self.cfg.log_skips_only and score.accepted:
            return
        flag = "MAP " if score.accepted else "skip"
        print(
            f"[TwoGateV2] i={self._call_idx:5d} {flag} step={score.triggered_by:<28s} "
            f"b1={int(score.b1_pass)} b2={int(score.b2_pass)} b3={int(score.b3_pass)} "
            f"pose_d={score.pose_d_m:.2f} gps_d={score.gps_d_m:.2f} ssim={score.visual_ssim:.3f} "
            f"covis={score.covis:.2f} dino={score.dino_d_min:.2f} "
            f"comp={score.composite:.3f} theta={score.theta:.3f} "
            f"since_kf={self._frames_since_kf}",
            flush=True,
        )

    # ------------------------------------------------------------------
    def _small_gray(self, rgb: np.ndarray) -> np.ndarray:
        h, w = rgb.shape[:2]
        short = min(h, w)
        target = max(int(self.cfg.ssim_resize), 16)
        if short > target:
            s = target / float(short)
            rgb = cv2.resize(rgb, (int(w * s), int(h * s)),
                             interpolation=cv2.INTER_AREA)
        if rgb.ndim == 3 and rgb.shape[2] == 3:
            return cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        return rgb if rgb.ndim == 2 else rgb[..., 0]


# =============================================================================
# Smoke test (CPU-only; enable_b3=False so DINO is skipped)
# =============================================================================

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    H, W = 240, 320
    fx = fy = 0.5 * W / np.tan(np.deg2rad(70.0) / 2)
    K = np.array([[fx, 0, W / 2], [0, fy, H / 2], [0, 0, 1]], np.float32)

    cfg = TwoGateV2Config(
        pose_d_min_m=0.10,
        visual_change_max_ssim=0.97,
        ssim_resize=80,
        enable_ssim_veto=True,
        covis_thresh=0.85,
        n_samples_covis=1024,
        min_depth=0.2,
        max_depth=35.0,
        enable_b3=False,                # no DINO in smoketest
        theta0=0.10,
        theta_init=0.10,
        window_size=8,
        sensitivity=0.5,
        decay=0.9,
        min_spacing=1,
        max_per_window=3,
        rate_window=10,
        force_after=20,
    )
    sel = TwoGateV2Selector(cfg, K, (H, W))

    def stripe_rgb(shift: int) -> np.ndarray:
        v, u = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
        intensity = ((np.sin((u + shift) * 0.05) + 1.0) * 127.5).astype(np.uint8)
        return np.stack([intensity, intensity, intensity], axis=-1)

    depth = np.full((H, W), 3.0, dtype=np.float32)
    R_eye = np.eye(3, dtype=np.float32)

    def report(i, label, ok, sc):
        flag = "ACCEPT" if ok else "skip  "
        extra = ""
        if sc.forced:         extra += " (forced)"
        if sc.budget_blocked: extra += " (budget)"
        print(f"i={i:2d} {label:>16s}  b1={int(sc.b1_pass)} b2={int(sc.b2_pass)} "
              f"covis={sc.covis:.2f} pose_d={sc.pose_d_m:.2f} "
              f"ssim={sc.visual_ssim:.3f} comp={sc.composite:.2f} "
              f"theta={sc.theta:.2f} -> {flag}{extra} ({sc.triggered_by})")

    accepts = []
    forced_count = 0
    budget_count = 0

    # Frame 0: bootstrap -> accept.
    ok, sc = sel.should_accept(
        depth, np.zeros(3, np.float32), R_eye,
        rgb=stripe_rgb(0),
    )
    accepts.append(int(ok))
    forced_count += int(sc.forced)
    report(0, "bootstrap", ok, sc)
    assert ok, "Frame 0 (bootstrap) must accept"

    # Frames 1-4: stationary (pose unchanged) -> B1 reject via pose_d.
    for i in range(1, 5):
        ok, sc = sel.should_accept(
            depth, np.zeros(3, np.float32), R_eye,
            rgb=stripe_rgb(0),
        )
        accepts.append(int(ok))
        forced_count += int(sc.forced)
        budget_count += int(sc.budget_blocked)
        report(i, "stationary", ok, sc)
        assert not ok, f"Stationary frame {i} must be rejected by B1"

    # Frames 5-12: lateral translation + distinct stripes -> motion+novelty.
    for i in range(5, 13):
        t_vec = np.array([1.0 * (i - 4), 0.0, 0.0], np.float32)
        rgb = stripe_rgb((i - 4) * 40)
        ok, sc = sel.should_accept(
            depth, t_vec, R_eye, rgb=rgb,
        )
        accepts.append(int(ok))
        forced_count += int(sc.forced)
        budget_count += int(sc.budget_blocked)
        report(i, "translating", ok, sc)

    # Frames 13-60: same pose, no novelty -> force_after eventually fires.
    for i in range(13, 60):
        ok, sc = sel.should_accept(
            depth, np.array([7.9, 0.0, 0.0], np.float32), R_eye,
            rgb=stripe_rgb(320),
        )
        accepts.append(int(ok))
        forced_count += int(sc.forced)
        budget_count += int(sc.budget_blocked)
        if sc.forced or ok:
            report(i, "stationary2", ok, sc)

    n_accept = sum(accepts)
    print(f"\nTotal: {n_accept} accepts out of {len(accepts)}  "
          f"(forced={forced_count}, budget_blocked={budget_count})")
    assert n_accept >= 3, f"Expected at least 3 accepts, got {n_accept}"
    assert forced_count >= 2, (
        f"Expected force_after to fire >=2 times in 47 stationary frames "
        f"(force_after={cfg.force_after}), got {forced_count}"
    )
    print("TwoGateV2Selector smoketest OK")
