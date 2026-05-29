"""
Gate A v2 -- Pre-Tracker frame-eligibility filter for VINGS-Mono.

Drop-in successor to `gate_a.py` (v1). v1 stays untouched -- both can be
selected via `cfg['gate_a']['version']`. v2 adds **A3 (GPS motion)** as a
pre-tracker analog of the MotionFilter optical-flow gate: reject frames
that have not moved far enough since the last A3-passed frame, before
paying for ~450 ms of `tracker.frontend_ba`.

Motivation
----------
On s1000_400f the baseline MotionFilter outperformed many post-tracker
selectors because it filters BEFORE tracking -- redundant frames cost
nothing. The post-tracker GPS check in `two_gate_selector.B1` does the
same job semantically, but only AFTER the tracker has already paid for
those frames. v2 moves the GPS-motion check upstream into Gate A. The
v2 post-tracker counterpart `two_gate_v2_selector.B1` therefore drops
the GPS path entirely (kept as pose-translation-only + SSIM-veto).

Three sub-gates, AND-decision with early exit. Cheapest reject first:

  A3 (GPS motion)     reject if ENU-distance from the previous A3-passed
                      frame < gps_d_min_m. Reference is the LAST frame
                      that passed A3 (not the last KF) -- mirrors
                      MotionFilter's "since the last frame the tracker
                      saw" semantics. Fail-open without GPS (no
                      xyz_enu in meta) and on the bootstrap frame.

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
If `meta['xyz_enu']` is missing, A3 fails open. If `meta['alt_m']` is
missing, A1 fails open. If `rgb` is missing, A2 fails open. Same
convention as nurbs_lvi / coko_slam when their required inputs are absent.

Order rationale
---------------
A3 is the cheapest check (two subtractions + a norm) and -- for steady-
state cruise / hover -- the most aggressive reject, so it runs first. A1
is also O(1) but only fires near ground. A2 does Sobel/Laplacian/resize
on the rgb image and is by far the heaviest, so it runs last.

State semantics for A3
----------------------
`_prev_passed_xyz` is updated **only when A3 itself passes** (not when
the whole gate passes). This makes A3 idempotent w.r.t. A1/A2 outcome
and gives well-defined behaviour during init: if A3 passes but A1
rejects (ground-level), the next frame still measures GPS distance
from the latest known position rather than wedging on bootstrap.

Interface
---------
    g = GateAV2.from_config(cfg_dict)
    accept, score = g.should_track(meta, rgb=None)
    if not accept: continue   # in run.py main loop

`meta` is the per-frame dict supplied by `GenericVODataset.__getitem__`:
    {'alt_m': float | None, 'xyz_enu': (3,) | None, 't_sec': float}

`rgb` is (H, W, 3) uint8 BGR (the raw image as cv2.imread returns it).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Mapping, Any

import numpy as np

try:
    import cv2
except ImportError as e:
    raise ImportError("GateAV2 requires opencv-python.") from e

from vings_utils.mm3dgs_selector import _laplacian_var


# =============================================================================
# Config / data classes
# =============================================================================

@dataclass
class GateAV2Config:
    # ---- A3 GPS motion (pre-tracker analog of MotionFilter) -------------
    # Disabled by default so that "version: v2" with no extra knobs behaves
    # identically to v1 -- explicit opt-in required for the new gate.
    enable_a3: bool = False
    gps_d_min_m: float = 0.5    # min ENU distance from last A3-passed frame

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
class GateAV2Score:
    # A3
    gps_d_m: Optional[float] = None
    # A1
    alt_m: Optional[float] = None
    ground_m: Optional[float] = None
    agl_m: Optional[float] = None
    # A2
    lap_var: float = 0.0
    mean_gray: float = 0.0
    grad_density: float = 0.0
    # decision
    reject_reason: str = ""
    accepted: bool = False


# =============================================================================
# Gate A v2
# =============================================================================

class GateAV2:
    """Pre-tracker frame eligibility filter (GPS motion + altitude + visual)."""

    def __init__(self, cfg: GateAV2Config):
        self.cfg = cfg
        self._calib_buf: list[float] = []
        self.ground_m: Optional[float] = cfg.ground_alt_m
        self._prev_passed_xyz: Optional[np.ndarray] = None  # (3,) ENU

    @classmethod
    def from_config(cls, cfg_dict: Mapping[str, Any]) -> "GateAV2":
        fields_ = set(GateAV2Config.__dataclass_fields__.keys())
        kwargs = {k: v for k, v in cfg_dict.items() if k in fields_}
        return cls(GateAV2Config(**kwargs))

    # ------------------------------------------------------------------
    def should_track(
        self,
        meta: Mapping[str, Any],
        rgb: Optional[np.ndarray] = None,
    ) -> tuple[bool, GateAV2Score]:
        score = GateAV2Score()

        # ---- A3 GPS motion (cheapest reject, runs first) ----------------
        if self.cfg.enable_a3:
            xyz = meta.get("xyz_enu") if meta is not None else None
            if xyz is not None:
                xyz_arr = np.asarray(xyz, dtype=np.float32).reshape(-1)
                if xyz_arr.size >= 3 and np.all(np.isfinite(xyz_arr[:3])):
                    xyz_arr = xyz_arr[:3]
                    if self._prev_passed_xyz is None:
                        # Bootstrap: nothing to measure against -> fail open
                        # and seed the reference.
                        self._prev_passed_xyz = xyz_arr.copy()
                        score.gps_d_m = 0.0
                    else:
                        d = float(np.linalg.norm(xyz_arr - self._prev_passed_xyz))
                        score.gps_d_m = d
                        if d < self.cfg.gps_d_min_m:
                            score.reject_reason = "a3_below_gps_distance"
                            return False, score
                        # A3 passes -> commit reference unconditionally,
                        # regardless of A1/A2 outcome (see module docstring).
                        self._prev_passed_xyz = xyz_arr.copy()
            # xyz missing / non-finite -> fail open

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

    cfg = GateAV2Config(
        enable_a3=True,
        gps_d_min_m=1.0,
        enable_a1=True,
        min_altitude_m=10.0,
        altitude_calib_frames=10,
        altitude_calib_quantile=0.05,
        enable_a2=True,
        blur_thresh=50.0,
        grad_density_thresh=0.03,
        sobel_T=30.0,
        image_resize_for_a2=240,
    )
    g = GateAV2(cfg)

    def sharp(_i: int) -> np.ndarray:
        return rng.integers(0, 255, (H, W, 3), dtype=np.uint8)

    def blurry(_i: int) -> np.ndarray:
        return cv2.GaussianBlur(sharp(_i), (21, 21), 0)

    def asphalt(_i: int) -> np.ndarray:
        return np.full((H, W, 3), 128, dtype=np.uint8)

    def report(label: str, ok: bool, sc: GateAV2Score) -> None:
        flag = "ACCEPT" if ok else "skip  "
        gps = f"{sc.gps_d_m:.2f}" if sc.gps_d_m is not None else "  -- "
        print(f"{label:>22s}  gps_d={gps}  alt={sc.alt_m}  agl={sc.agl_m}  "
              f"lap={sc.lap_var:7.1f}  mean={sc.mean_gray:6.1f}  "
              f"grad={sc.grad_density:.3f}  reason={sc.reject_reason!r}  {flag}")

    # ---- Phase 0: bootstrap A3 (no prev_xyz). Cruise altitude + sharp.
    # First call seeds A3 reference and passes (fail-open + A1 not yet
    # calibrated). Subsequent stationary calls hit A3-reject.
    print("--- Phase 0: A3 bootstrap (cruise alt, sharp, stationary) ---")
    base_xyz = np.array([100.0, 50.0, 80.0], np.float32)
    ok, sc = g.should_track(
        {"alt_m": 220.0, "xyz_enu": base_xyz}, rgb=sharp(0)
    )
    report("bootstrap", ok, sc)
    assert ok, "Bootstrap must pass (A3 fail-open on first xyz)"
    for i in range(1, 5):
        # Tiny GPS perturbation < 1m
        jitter = rng.normal(0, 0.1, 3).astype(np.float32)
        ok, sc = g.should_track(
            {"alt_m": 220.0, "xyz_enu": base_xyz + jitter}, rgb=sharp(i)
        )
        report(f"stationary i={i}", ok, sc)
        assert not ok and sc.reject_reason == "a3_below_gps_distance", (
            f"Frame {i} (stationary) must be A3-rejected, got {sc.reject_reason!r}"
        )

    # ---- Phase 1: ground-level + moving GPS. A3 passes (>=1 m steps),
    # A1 calibration fills, then A1 rejects from frame cf-1 onward.
    print("\n--- Phase 1: ground alt ~100m, moving GPS -> A3 ok, A1 reject after calib ---")
    cf = cfg.altitude_calib_frames
    # Fresh gate for this phase so A1 calibration is well-defined.
    g = GateAV2(cfg)
    for i in range(15):
        ok, sc = g.should_track(
            {"alt_m": 100.0 + rng.normal(0, 0.3),
             "xyz_enu": np.array([i * 2.0, 0.0, 0.0], np.float32)},
            rgb=sharp(i),
        )
        report(f"ground i={i:2d}", ok, sc)
        if i < cf - 1:
            assert ok, f"A1 must not block before calibration (i={i})"
        else:
            assert not ok and sc.reject_reason == "a1_below_altitude", (
                f"A1 must reject from calibration onward at i={i}, "
                f"got {sc.reject_reason!r}"
            )

    # ---- Phase 2: cruise altitude + moving GPS + sharp -> accept.
    print("\n--- Phase 2: cruise, moving, sharp -> accept ---")
    ok, sc = g.should_track(
        {"alt_m": 220.0, "xyz_enu": np.array([1000.0, 0.0, 0.0], np.float32)},
        rgb=sharp(99),
    )
    report("cruise sharp", ok, sc)
    assert ok and sc.accepted, "must accept at cruise + moving + sharp"

    # ---- Phase 3: cruise + moving + blurry -> A2 blur reject.
    print("\n--- Phase 3: cruise, moving, blurry -> A2 reject ---")
    ok, sc = g.should_track(
        {"alt_m": 220.0, "xyz_enu": np.array([1002.0, 0.0, 0.0], np.float32)},
        rgb=blurry(99),
    )
    report("cruise blurry", ok, sc)
    assert not ok and sc.reject_reason == "a2_blur", (
        f"must reject with reason=a2_blur, got {sc.reject_reason!r}"
    )

    # ---- Phase 4: missing GPS -> A3 fails open, sharp + cruise -> accept.
    print("\n--- Phase 4: no xyz_enu -> A3 fail-open, accept ---")
    ok, sc = g.should_track({"alt_m": 220.0}, rgb=sharp(99))
    report("no-gps cruise", ok, sc)
    assert ok and sc.accepted

    # ---- Phase 5: no rgb -> A2 fail-open; alt high + moving -> accept.
    print("\n--- Phase 5: no rgb, alt high, moving -> accept ---")
    ok, sc = g.should_track(
        {"alt_m": 220.0, "xyz_enu": np.array([1010.0, 0.0, 0.0], np.float32)},
        rgb=None,
    )
    report("no-rgb hi-alt", ok, sc)
    assert ok and sc.accepted

    # ---- Phase 6: enable_a3=False reproduces v1 behaviour (stationary OK).
    print("\n--- Phase 6: enable_a3=False -> v1 parity (stationary accepted) ---")
    cfg_v1 = GateAV2Config(
        enable_a3=False, enable_a1=False, enable_a2=False,
    )
    g_v1 = GateAV2(cfg_v1)
    ok, sc = g_v1.should_track(
        {"alt_m": 220.0, "xyz_enu": base_xyz}, rgb=sharp(0)
    )
    report("v1 parity 1", ok, sc)
    assert ok
    ok, sc = g_v1.should_track(
        {"alt_m": 220.0, "xyz_enu": base_xyz}, rgb=sharp(1)
    )
    report("v1 parity 2", ok, sc)
    assert ok, "with all gates disabled, stationary frame must accept"

    print("\nGateAV2 smoketest OK")
