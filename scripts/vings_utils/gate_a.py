"""
Gate A -- Pre-Tracker frame-eligibility filter for VINGS-Mono.

This is the FIRST of two gates in the hierarchical frame-selection design
(see `docs/TWO_GATE.md`). Gate A runs *before* the tracker; rejected frames
skip both `tracker.track()` (~450 ms) and the downstream mapper -- a much
cheaper veto than Gate B (which runs post-tracker).

Two sub-gates, AND-decision with early exit:

  A1 (altitude)       reject if fused barometric/RTK altitude < min_altitude_m
                      AGL (Above Ground Level). Ground is auto-calibrated
                      from the first N altitude samples (quantile, robust to
                      noise) or supplied via `ground_alt_m`. Failure mode it
                      protects against: monocular depth fails near ground,
                      small baseline confuses VINGS init.

  A2 (visual quality) reject if any of:
                        - over-exposed   : mean(gray) > overexp_thresh
                        - under-exposed  : mean(gray) < underexp_thresh
                        - blurry         : Var(Laplacian) < blur_thresh
                        - low-texture    : fraction of pixels with
                                           Sobel-magnitude > T_g  <  thresh
                      The gradient-density check is the "well-exposed
                      asphalt" failsafe -- catches featureless frames that
                      pure exposure-checking lets through.

Fail-open policy
----------------
If `meta['alt_m']` is missing, A1 fails open (does not block). If `rgb` is
missing, A2 fails open. Same convention as nurbs_lvi / coko_slam when
their required inputs are absent.

Interface
---------
    g = GateA.from_config(cfg_dict)
    accept, score = g.should_track(meta, rgb=None)
    if not accept: continue   # in run.py main loop

`meta` is the per-frame dict supplied by `GenericVODataset.__getitem__`:
    {'alt_m': float | None, 'xyz_enu': (3,) | None, 't_sec': float}

`rgb` is (H, W, 3) uint8 BGR (the raw image as cv2.imread returns it).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Mapping, Any

import numpy as np

try:
    import cv2
except ImportError as e:
    raise ImportError("GateA requires opencv-python.") from e

from vings_utils.mm3dgs_selector import _laplacian_var


# =============================================================================
# Config / data classes
# =============================================================================

@dataclass
class GateAConfig:
    # ---- A1 altitude -----------------------------------------------------
    enable_a1: bool = True
    min_altitude_m: float = 8.0
    altitude_calib_frames: int = 30
    altitude_calib_quantile: float = 0.05
    # Override auto-calibration (use for mid-cruise slices where the first
    # frames are NOT at ground level -- otherwise ground_m calibrates too
    # high and A1 rejects every frame).
    ground_alt_m: Optional[float] = None

    # ---- A2 visual quality ----------------------------------------------
    enable_a2: bool = True
    blur_thresh: float = 80.0
    overexp_thresh: float = 240.0
    underexp_thresh: float = 15.0
    grad_density_thresh: float = 0.03
    sobel_T: float = 30.0
    image_resize_for_a2: int = 240  # short-side pixels


@dataclass
class GateAScore:
    alt_m: Optional[float] = None
    ground_m: Optional[float] = None
    agl_m: Optional[float] = None
    lap_var: float = 0.0
    mean_gray: float = 0.0
    grad_density: float = 0.0
    reject_reason: str = ""
    accepted: bool = False


# =============================================================================
# Gate A
# =============================================================================

class GateA:
    """Pre-tracker frame eligibility filter (altitude + visual quality)."""

    def __init__(self, cfg: GateAConfig):
        self.cfg = cfg
        self._calib_buf: list[float] = []
        self.ground_m: Optional[float] = cfg.ground_alt_m

    @classmethod
    def from_config(cls, cfg_dict: Mapping[str, Any]) -> "GateA":
        fields_ = set(GateAConfig.__dataclass_fields__.keys())
        kwargs = {k: v for k, v in cfg_dict.items() if k in fields_}
        return cls(GateAConfig(**kwargs))

    # ------------------------------------------------------------------
    def should_track(
        self,
        meta: Mapping[str, Any],
        rgb: Optional[np.ndarray] = None,
    ) -> tuple[bool, GateAScore]:
        score = GateAScore()

        # ---- A1 altitude ------------------------------------------------
        if self.cfg.enable_a1:
            alt = meta.get("alt_m") if meta is not None else None
            score.alt_m = alt
            if alt is not None and np.isfinite(alt):
                if self.ground_m is None:
                    self._calib_buf.append(float(alt))
                    if len(self._calib_buf) >= self.cfg.altitude_calib_frames:
                        self.ground_m = float(
                            np.quantile(self._calib_buf,
                                        self.cfg.altitude_calib_quantile)
                        )
                if self.ground_m is not None:
                    score.ground_m = self.ground_m
                    score.agl_m = float(alt) - self.ground_m
                    if score.agl_m < self.cfg.min_altitude_m:
                        score.reject_reason = "a1_below_altitude"
                        return False, score
            # alt missing -> fail open (do not block)

        # ---- A2 visual quality -----------------------------------------
        if self.cfg.enable_a2 and rgb is not None:
            small = self._downsample(rgb)
            if small.ndim == 3 and small.shape[2] == 3:
                gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            else:
                gray = small if small.ndim == 2 else small[..., 0]
            if gray.dtype != np.uint8:
                gray = gray.astype(np.uint8)

            score.mean_gray = float(gray.mean())
            score.lap_var = _laplacian_var(small)
            score.grad_density = self._grad_density(gray)

            if score.mean_gray > self.cfg.overexp_thresh:
                score.reject_reason = "a2_over_exposed"
                return False, score
            if score.mean_gray < self.cfg.underexp_thresh:
                score.reject_reason = "a2_under_exposed"
                return False, score
            if score.lap_var < self.cfg.blur_thresh:
                score.reject_reason = "a2_blur"
                return False, score
            if score.grad_density < self.cfg.grad_density_thresh:
                score.reject_reason = "a2_low_texture"
                return False, score

        score.accepted = True
        return True, score

    # ------------------------------------------------------------------
    def _downsample(self, rgb: np.ndarray) -> np.ndarray:
        h, w = rgb.shape[:2]
        short = min(h, w)
        target = max(int(self.cfg.image_resize_for_a2), 32)
        if short <= target:
            return rgb
        s = target / float(short)
        return cv2.resize(rgb, (int(w * s), int(h * s)),
                          interpolation=cv2.INTER_AREA)

    def _grad_density(self, gray_u8: np.ndarray) -> float:
        gx = cv2.Sobel(gray_u8, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray_u8, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.hypot(gx, gy)
        return float((mag > self.cfg.sobel_T).mean())


# =============================================================================
# Smoke test
# =============================================================================

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    H, W = 240, 320

    cfg = GateAConfig(
        min_altitude_m=10.0,
        altitude_calib_frames=10,
        altitude_calib_quantile=0.05,
        blur_thresh=50.0,
        grad_density_thresh=0.03,
        sobel_T=30.0,
        image_resize_for_a2=240,
    )
    g = GateA(cfg)

    def sharp(_i: int) -> np.ndarray:
        return rng.integers(0, 255, (H, W, 3), dtype=np.uint8)

    def blurry(_i: int) -> np.ndarray:
        return cv2.GaussianBlur(sharp(_i), (21, 21), 0)

    def asphalt(_i: int) -> np.ndarray:
        # Well-exposed but featureless -> A2 low-texture should reject.
        return np.full((H, W, 3), 128, dtype=np.uint8)

    def report(label: str, ok: bool, sc: GateAScore) -> None:
        flag = "ACCEPT" if ok else "skip  "
        print(f"{label:>20s}  alt={sc.alt_m}  agl={sc.agl_m}  "
              f"lap={sc.lap_var:7.1f}  mean={sc.mean_gray:6.1f}  "
              f"grad={sc.grad_density:.3f}  reason={sc.reject_reason!r}  {flag}")

    # ---- Phase 1: ground (alt ~= ground_alt). Calib buffer fills.
    # During warm-up (frames 0..calib_frames-2) the buffer is incomplete, so
    # A1 has no ground estimate and accepts. On the call that fills the
    # buffer (i == calib_frames - 1 = 9) ground_m is computed and the SAME
    # call evaluates against it -> rejects if at ground level.
    print("--- Phase 1: ground (alt ~100m, expect calibration then reject) ---")
    cf = cfg.altitude_calib_frames                # 10
    for i in range(15):
        ok, sc = g.should_track(
            {"alt_m": 100.0 + rng.normal(0, 0.3)}, rgb=sharp(i)
        )
        report(f"ground i={i:2d}", ok, sc)
        if i < cf - 1:
            assert ok, f"A1 must not block before calibration (i={i})"
        else:
            assert not ok and sc.reject_reason == "a1_below_altitude", (
                f"A1 must reject from calibration onward at i={i}, "
                f"got reason={sc.reject_reason!r}"
            )

    # ---- Phase 2: cruise altitude (+120m AGL), sharp -> accept.
    print("\n--- Phase 2: cruise, sharp -> accept ---")
    ok, sc = g.should_track({"alt_m": 220.0}, rgb=sharp(99))
    report("cruise sharp", ok, sc)
    assert ok and sc.accepted, "must accept at cruise altitude with sharp rgb"

    # ---- Phase 3: cruise but blurry -> A2 blur reject.
    print("\n--- Phase 3: cruise, blurry -> A2 reject ---")
    ok, sc = g.should_track({"alt_m": 220.0}, rgb=blurry(99))
    report("cruise blurry", ok, sc)
    assert not ok and sc.reject_reason == "a2_blur", (
        f"must reject with reason=a2_blur, got {sc.reject_reason!r}"
    )

    # ---- Phase 4: cruise but featureless -> A2 reject.
    # Uniform-gray asphalt has LapVar=0 (zero gradient) so the blur check
    # rejects it first; in real footage the LapVar would be non-zero but
    # gradient density would still catch it. Either reason is acceptable
    # here -- the point is that A2 stops it.
    print("\n--- Phase 4: cruise, asphalt -> A2 reject ---")
    ok, sc = g.should_track({"alt_m": 220.0}, rgb=asphalt(99))
    report("cruise asphalt", ok, sc)
    assert not ok and sc.reject_reason.startswith("a2_"), (
        f"must reject with an a2_* reason, got {sc.reject_reason!r}"
    )

    # ---- Phase 5: missing altitude -> A1 fails open, A2 sharp -> accept.
    print("\n--- Phase 5: no alt -> A1 fail-open, sharp -> accept ---")
    ok, sc = g.should_track({}, rgb=sharp(99))
    report("no-alt sharp", ok, sc)
    assert ok and sc.accepted

    # ---- Phase 6: no rgb -> A2 fail-open; alt high -> accept.
    print("\n--- Phase 6: no rgb, alt high -> accept ---")
    ok, sc = g.should_track({"alt_m": 220.0}, rgb=None)
    report("no-rgb hi-alt", ok, sc)
    assert ok and sc.accepted

    print("\nGateA smoketest OK")
