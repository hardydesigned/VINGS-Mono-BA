"""
TwoGateSelector -- Gate B (post-tracker, pre-mapper) keyframe selector.

This is the SECOND of two gates in the hierarchical frame-selection design
(see `docs/TWO_GATE.md`). Gate A (`gate_a.py`) runs *before* the tracker;
Gate B runs *after* tracker-KF acceptance and decides whether the tracker-KF
is forwarded to the mapper.

Three layers (B1, B2, B3) + adaptive threshold + budget cap. Composition-
based: B2 reuses `Mm3dgsSelector._covisibility` and B3 reuses the DINOv2
extraction + FIFO from `CokoSlamSelector`. The sub-selectors' internal
accept logic is neutralised by config (covis_thresh=1.01, alpha=0.0,
force_accept_all=True) -- we only call their helpers and read/write their
state attributes.

Layers
------
  B1 motion          GPS-distance from prev_kf >= gps_d_min_m, plus a cheap
                     anti-noise visual SSIM check on a downsampled image
                     pair. Without GPS (or below GPS-noise floor), falls
                     back to pose-translation. The SSIM veto fires when GPS
                     says "moved" but the image is essentially identical
                     -- catches the dataset's documented `local_position`
                     scale bug (CLAUDE.md sec. 3).

  B2 covisibility    Fraction of current-frame samples that reproject into
                     prev_kf. Catches yaw-without-parallax: a pure rotation
                     in hover gives B1-pass (new pose), but covis stays
                     high (no new viewpoints worth mapping).

  B3 DINO novelty    Min L2 distance from the current frame's DINOv2-Small
                     CLS-token feature to the FIFO of recent KF features.
                     Catches texture-poor cases where covis trivially drops
                     (e.g. flying over a featureless field) but 3DGS cannot
                     gain anything from the new view.

Decision
--------
    composite = 0.5 * (1 - covis) + 0.5 * normed_dino   # or only the first
                                                         # if enable_b3=False
    accept = b1_pass AND composite >= adaptive_theta
    if not accept and frames_since_kf >= force_after:
        accept = True; forced = True
    if frames_since_kf < min_spacing or rate-cap hit:
        accept = False; budget_blocked = True

Adaptive theta
--------------
Same as `adaptive_kf_selector.py` (Eq. 6/7):
    theta = max(theta0, mean(score_buf) + k * std(score_buf))   # full
    theta = theta0 * (r) + theta_init * (1 - r), r = |buf| / W  # warm-up
    on accept:  theta *= decay

Budget cap
----------
- `min_spacing`: hard minimum frames between accepts (avoids two-in-a-row).
- `max_per_window` over `rate_window`: limits KF density in fast-flight
  segments.

Interface
---------
Same as every other selector + a new `meta` kwarg (backward-compatible via
`**_` in all existing selectors):

    sel = TwoGateSelector.from_config(cfg, K, (H, W))
    accept, score = sel.should_accept(
        depth, t, R, rgb=rgb_uint8, depth_cov=cov, meta={'alt_m':..., 'xyz_enu':..., 't_sec':...}
    )
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional, Mapping, Any

import numpy as np

try:
    import cv2
except ImportError as e:
    raise ImportError("TwoGateSelector requires opencv-python.") from e


# =============================================================================
# Config / score
# =============================================================================

@dataclass
class TwoGateConfig:
    # ---- B1 motion -------------------------------------------------------
    gps_d_min_m: float = 0.5
    gps_noise_floor_m: float = 0.3
    pose_d_min_m: float = 0.15
    visual_change_max_ssim: float = 0.98
    ssim_resize: int = 80                # downsample short side for cheap SSIM

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
class TwoGateScore:
    b1_pass: bool = False
    gps_d_m: float = 0.0
    pose_d_m: float = 0.0
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
        # Resize b to match a.
        b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA)
    if _HAS_SSIM:
        return float(_ssim(a, b, data_range=255))
    return float(1.0 - np.mean(np.abs(a.astype(np.int16) - b.astype(np.int16))) / 255.0)


# =============================================================================
# Selector
# =============================================================================

class TwoGateSelector:
    """Hierarchical Gate-B keyframe selector. Composes Mm3dgs + Coko helpers."""

    def __init__(self, cfg: TwoGateConfig, K: np.ndarray,
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

        # ---- B1 state ----------------------------------------------------
        self.prev_kf_xyz: Optional[np.ndarray] = None  # ENU (3,)
        self.prev_kf_t:   Optional[np.ndarray] = None  # (3,)
        self.prev_kf_R:   Optional[np.ndarray] = None  # (3,3)  for completeness
        self.prev_kf_rgb_small: Optional[np.ndarray] = None  # gray, uint8

        # ---- Adaptive theta state ---------------------------------------
        self._score_buf: deque[float] = deque(maxlen=cfg.window_size)
        self._theta = float(cfg.theta_init)

        # ---- Budget state ------------------------------------------------
        self._frames_since_kf = 0
        # 1 if frame was accepted, 0 if skipped. We pre-decision read.
        self._accept_history: deque[int] = deque(maxlen=cfg.rate_window)

        # ---- Logging state -----------------------------------------------
        self._call_idx = -1  # incremented at the top of each should_accept

    @classmethod
    def from_config(cls, cfg_dict: Mapping[str, Any], K: np.ndarray,
                    image_hw: tuple[int, int]) -> "TwoGateSelector":
        fields_ = set(TwoGateConfig.__dataclass_fields__.keys())
        kwargs = {k: v for k, v in cfg_dict.items() if k in fields_}
        return cls(TwoGateConfig(**kwargs), K, image_hw)

    # ------------------------------------------------------------------
    def should_accept(
        self,
        depth: np.ndarray,
        t: np.ndarray,
        R: np.ndarray,
        rgb: Optional[np.ndarray] = None,
        depth_cov: Optional[np.ndarray] = None,
        meta: Optional[Mapping[str, Any]] = None,
        **_: object,
    ) -> tuple[bool, TwoGateScore]:
        score = TwoGateScore(theta=self._theta)
        meta = meta or {}
        self._frames_since_kf += 1
        self._call_idx += 1

        t = np.asarray(t, dtype=np.float32)
        R = np.asarray(R, dtype=np.float32)

        # ---- Bootstrap ---------------------------------------------------
        if self.prev_kf_t is None:
            self._commit(t, R, rgb, meta, depth)
            score.b1_pass = score.b2_pass = score.b3_pass = True
            score.forced = True
            score.accepted = True
            score.triggered_by = "first"
            self._log(score)
            return True, score

        # ---- B1 motion (GPS-distance with anti-noise visual check) -----
        score.pose_d_m = float(np.linalg.norm(t - self.prev_kf_t))
        gps_xyz = meta.get("xyz_enu") if meta is not None else None
        gps_ok = False
        b1_reason = ""  # why B1 rejected (only meaningful if not gps_ok)
        if gps_xyz is not None and self.prev_kf_xyz is not None:
            score.gps_d_m = float(np.linalg.norm(
                np.asarray(gps_xyz, dtype=np.float32) - self.prev_kf_xyz
            ))
            if score.gps_d_m >= self.cfg.gps_d_min_m:
                if score.gps_d_m < self.cfg.gps_noise_floor_m:
                    # GPS says moved, but within its noise floor.
                    # Trust pose instead.
                    gps_ok = score.pose_d_m >= self.cfg.pose_d_min_m
                    if not gps_ok:
                        b1_reason = "B1_pose_below_min(noise_floor)"
                else:
                    gps_ok = True
            else:
                b1_reason = "B1_gps_below_min"
        else:
            # No GPS -> fall back to tracker-pose displacement.
            gps_ok = score.pose_d_m >= self.cfg.pose_d_min_m
            if not gps_ok:
                b1_reason = "B1_pose_below_min(no_gps)"

        # Visual veto: if motion is claimed but image is essentially
        # identical, it was sensor noise.
        if gps_ok and rgb is not None and self.prev_kf_rgb_small is not None:
            curr_small = self._small_gray(rgb)
            score.visual_ssim = _cheap_ssim_gray(curr_small, self.prev_kf_rgb_small)
            if score.visual_ssim > self.cfg.visual_change_max_ssim:
                gps_ok = False
                b1_reason = "B1_ssim_veto"
        score.b1_pass = gps_ok

        # ---- B2 covisibility -------------------------------------------
        # mm3dgs._covisibility uses self.prev_kf_depth internally; we keep it
        # synced in _commit. On the very first call there is no prev_kf yet,
        # so covis is undefined and we treat the frame as fully novel.
        if self._covis_helper.prev_kf_depth is None:
            score.covis = 0.0
        else:
            score.covis = self._covis_helper._covisibility(t, R)
        score.b2_pass = score.covis < self.cfg.covis_thresh
        b2_novelty = float(np.clip(1.0 - score.covis, 0.0, 1.0))

        # ---- B3 DINO content novelty ------------------------------------
        b3_novelty = 0.0
        if self._dino_helper is not None and rgb is not None:
            # Helper's `_extract` returns (D,) L2-normalised on its device.
            import torch
            feat = self._dino_helper._extract(rgb)
            if self._dino_helper.kf_features:
                refs = torch.stack(self._dino_helper.kf_features, dim=0)
                d_min = float(torch.norm(feat[None, :] - refs, dim=1).min().item())
            else:
                d_min = 2.0  # empty FIFO -> max novelty
            score.dino_d_min = d_min
            score.b3_pass = d_min >= self.cfg.alpha
            # L2 of unit vectors is in [0, 2]; normalise into [0, 1].
            b3_novelty = float(min(d_min, 2.0) / 2.0)
        elif self._dino_helper is None:
            # B3 disabled -> don't count it in the composite.
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

        # Force-after failsafe -- avoids KF starvation in stationary segments.
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
            self._commit(t, R, rgb, meta, depth)
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
        meta: Mapping[str, Any],
        depth: np.ndarray,
    ) -> None:
        """Update internal + helper state for a newly accepted keyframe.

        NB: we write directly into self._covis_helper.prev_kf_R/t -- this
        bypasses Mm3dgsSelector._commit (which also clears its window /
        stalled counter). That is intentional: we don't use the helper's
        own gates. If mm3dgs ever renames those attrs, this will break
        loudly via AttributeError, so the coupling is auditable.
        """
        self.prev_kf_t = t.copy()
        self.prev_kf_R = R.copy()

        gps_xyz = meta.get("xyz_enu") if meta is not None else None
        if gps_xyz is not None:
            self.prev_kf_xyz = np.asarray(gps_xyz, dtype=np.float32).copy()

        # Tell the covis helper its new reference KF.
        self._covis_helper.prev_kf_R = R.copy()
        self._covis_helper.prev_kf_t = t.copy()
        self._covis_helper.prev_kf_depth = np.asarray(depth, dtype=np.float32).copy()

        # Push DINO feature into the helper's FIFO.
        if self._dino_helper is not None and rgb is not None:
            feat = self._dino_helper._extract(rgb)
            self._dino_helper._commit(feat)

        # Cache small grayscale for the next B1 SSIM check.
        if rgb is not None:
            self.prev_kf_rgb_small = self._small_gray(rgb)

        self._frames_since_kf = 0

    # ------------------------------------------------------------------
    def _log(self, score: "TwoGateScore") -> None:
        """Print a one-line decision trace if cfg.verbose.

        The line names the exact pipeline step that decided the frame's
        fate via `score.triggered_by`:
          first                      -> bootstrap KF (always mapped)
          B1_gps_below_min           -> B1 motion gate: GPS move too small
          B1_pose_below_min(...)     -> B1 motion gate: tracker-pose move too small
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
            f"[TwoGate] i={self._call_idx:5d} {flag} step={score.triggered_by:<28s} "
            f"b1={int(score.b1_pass)} b2={int(score.b2_pass)} b3={int(score.b3_pass)} "
            f"gps_d={score.gps_d_m:.2f} pose_d={score.pose_d_m:.2f} ssim={score.visual_ssim:.3f} "
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
# Smoke test  (CPU-only; enable_b3=False so DINO is skipped)
# =============================================================================

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    H, W = 240, 320
    fx = fy = 0.5 * W / np.tan(np.deg2rad(70.0) / 2)
    K = np.array([[fx, 0, W / 2], [0, fy, H / 2], [0, 0, 1]], np.float32)

    cfg = TwoGateConfig(
        gps_d_min_m=0.4,
        gps_noise_floor_m=0.2,
        pose_d_min_m=0.10,
        visual_change_max_ssim=0.97,
        ssim_resize=80,
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
    sel = TwoGateSelector(cfg, K, (H, W))

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
              f"covis={sc.covis:.2f} gps_d={sc.gps_d_m:.2f} pose_d={sc.pose_d_m:.2f} "
              f"ssim={sc.visual_ssim:.3f} comp={sc.composite:.2f} "
              f"theta={sc.theta:.2f} -> {flag}{extra} ({sc.triggered_by})")

    accepts = []
    forced_count = 0
    budget_count = 0

    # Frame 0: bootstrap -> accept.
    ok, sc = sel.should_accept(
        depth, np.zeros(3, np.float32), R_eye,
        rgb=stripe_rgb(0),
        meta={"xyz_enu": np.zeros(3, np.float32)},
    )
    accepts.append(int(ok))
    forced_count += int(sc.forced)
    report(0, "bootstrap", ok, sc)
    assert ok, "Frame 0 (bootstrap) must accept"

    # Frames 1-4: stationary + identical rgb -> B1 SSIM veto.
    for i in range(1, 5):
        ok, sc = sel.should_accept(
            depth, np.zeros(3, np.float32), R_eye,
            rgb=stripe_rgb(0),
            meta={"xyz_enu": np.zeros(3, np.float32)},
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
            meta={"xyz_enu": np.array([1.0 * (i - 4), 0.0, 0.0], np.float32)},
        )
        accepts.append(int(ok))
        forced_count += int(sc.forced)
        budget_count += int(sc.budget_blocked)
        report(i, "translating", ok, sc)

    # Frames 13-60: stationary -> at frame ~13+force_after force_after fires.
    for i in range(13, 60):
        ok, sc = sel.should_accept(
            depth, np.array([7.9, 0.0, 0.0], np.float32), R_eye,
            rgb=stripe_rgb(320),  # same stripe pattern, no novelty
            meta={"xyz_enu": np.array([7.9, 0.0, 0.0], np.float32)},
        )
        accepts.append(int(ok))
        forced_count += int(sc.forced)
        budget_count += int(sc.budget_blocked)
        if sc.forced or ok:
            report(i, "stationary2", ok, sc)

    n_accept = sum(accepts)
    print(f"\nTotal: {n_accept} accepts out of {len(accepts)}  "
          f"(forced={forced_count}, budget_blocked={budget_count})")
    assert n_accept >= 3, f"Expected at least 3 accepts (bootstrap + motion + force_after), got {n_accept}"
    assert forced_count >= 2, (
        f"Expected force_after to fire >=2 times in 47 stationary frames "
        f"(force_after={cfg.force_after}), got {forced_count}"
    )
    print("TwoGateSelector smoketest OK")
