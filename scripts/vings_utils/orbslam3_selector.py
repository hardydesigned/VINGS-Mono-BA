"""
ORB-SLAM3-inspired keyframe selector.

Implements the "New Keyframe Decision" step of ORB-SLAM3's tracking thread.
Reference: UZ-SLAMLab/ORB_SLAM3 master, `src/Tracking.cc::NeedNewKeyFrame()`
(lines 3064-3214 of current HEAD). Algorithm originally from
Mur-Artal & Tardós, ORB-SLAM2, IEEE T-RO 2017, Sec. V.E.

Original rule for Mono (`mSensor == MONOCULAR`, non-IMU), simplified:

    nRefMatches = prev_kf.TrackedMapPoints(nMinObs=3)   # property of ref KF
    matches     = mnMatchesInliers                       # inliers in current
    ratio       = matches / nRefMatches
    thRefRatio  = 0.9        (or 0.4 for nKFs < 2 bootstrap)

    c1a  =  N >= MaxFrames
    c1b  =  N >= MinFrames  AND  mapper_idle
    c1c  =  stereo-only  (gated by `mSensor != MONOCULAR ...`)
    c2   =  matches < ratio * thRefRatio  AND  matches > 15
    c3   =  inertial-only timestamp gate (≥ 0.5 s)
    c4   =  IMU_MONOCULAR-only tracking-weak gate

    accept iff  ((c1a OR c1b OR c1c) AND c2) OR c3 OR c4

VINGS-Mono-Reduktion: c1c = c3 = c4 = false (kein Stereo, kein IMU). Also:

    accept iff  (c1a OR c1b)  AND  c2

VINGS-Adaptions (alle in docs/ORB_SLAM3.md dokumentiert):

  - Mapper-Idle: liegt im Selector-Kontext nicht vor → c1b auf reines
    `N >= min_frames` reduziert (Mapper läuft synchron, ist bei Selector-
    Aufruf immer "idle").

  - Reference-KF: sequentieller `prev_kf` (zuletzt akzeptierter KF) statt
    Covisibility-Graph-K_ref. VINGS pflegt keinen ORB-Covisibility-Graph.

  - nRefMatches: Original ist `mpReferenceKF->TrackedMapPoints(3)`, also die
    Anzahl der hinreichend oft beobachteten Map-Points im Ref-KF — eine
    eigenschaft des Ref-KF, frame-unabhängig, über die KF-Lebensdauer
    nahezu konstant. VINGS hat keine Map-Points → Analog ist
    `baseline_matches`: der BFMatch-cross-check-Count zwischen prev_kf und
    dem **ersten Frame nach dem KF-Commit**. Diese Größe wird einmalig pro
    KF gemessen und bleibt für seine Lebensdauer fix. Damit hat das
    Paper-Threshold 0.9 dieselbe Semantik wie im Original ("10 % Drop vom
    Baseline-Tracking-Niveau"). Fallback wenn `baseline_matches == 0`:
    `min(n_kp_curr, n_kp_prev)` als sichere obere Schranke.

  - Bootstrap-Threshold (Original Z. 3131-3132: `thRefRatio = 0.4` bei
    `nKFs < 2`): nicht repliziert. VINGS-Bootstrap erfolgt über den
    "first frame always accept"-Pfad, der das nicht braucht. Effekt:
    in den ersten 1-2 KFs einer Sequenz ist unsere Implementation
    minimal aggressiver (Trigger bei 10 % Drop statt 60 % Drop).

  - Reloc-Gate (Original Z. 3091-3094: `if mnId < mnLastRelocFrameId
    + MaxFrames && nKFs > MaxFrames → false`): bewusst weggelassen,
    VINGS hat keine Relokalisation.

  - nMinObs-Schwelle (Original Z. 3097-3099: `nMinObs = 3`, bzw. `2`
    bei `nKFs <= 2`): keine direkte Entsprechung. Im Original filtert
    `TrackedMapPoints(nMinObs)` Map-Points danach, ob sie von mindestens
    nMinObs Keyframes beobachtet wurden — VINGS hat keine Map-Points
    mit Beobachtungszählern. `baseline_matches` ist *eine* per-KF
    stabile Größe, nicht *die gleiche*.

  - Mapper-busy-Return-Logik (Original Z. 3191-3209): bei busy Mapper
    im Mono-Pfad gibt das Original `false` zurück, auch wenn
    `(c1a||c1b) ∧ c2` true ist. VINGS-Selector ist synchron — Mapper
    ist bei Aufruf per Konstruktion idle, der Pfad fällt weg.

  - Numerator-Operator: Original `mnMatchesInliers` ist das Ergebnis
    von `SearchByProjection` (Map-Point-Reprojektion ins aktuelle Bild
    mit Inlier-Filterung über RANSAC/PnP). VINGS verwendet
    `BFMatcher(NORM_HAMMING, crossCheck=True)` zwischen den
    ORB-Deskriptor-Sets von `current` und `prev_kf`. Beide Operatoren
    messen "wie viele Features des Referenz-Frames sind noch
    wiederfindbar"; ohne Map-Point-Datenstruktur ist BFMatch die
    direkteste Ersatzoperation.

Score-Struct exponiert die drei Bedingungen einzeln (`c1a`, `c1b`, `c2`)
sowie `baseline_matches` für PhaseTimer-/Diagnose-Auswertung.

Same calling convention as the other selectors:

    sel = OrbSlam3Selector(cfg, K, (H, W))
    accept, score = sel.should_accept(depth, t, R, rgb=rgb_uint8)

`rgb` is required (ORB detector input). Without rgb the selector is a no-op
(always accept) — matches the nurbs_lvi / mm3dgs convention.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    import cv2
except ImportError as e:
    raise ImportError("OrbSlam3Selector requires opencv-python.") from e


# =============================================================================
# Config / data classes
# =============================================================================

@dataclass
class OrbSlam3Config:
    # ORB extractor budget. Same default as nurbs_lvi for comparability.
    orb_n_features: int = 800

    # c2 novelty threshold (Paper "thRefRatio" = 0.9, Mono path).
    # accept iff ratio = n_matches / baseline_matches < this.
    tracked_ratio_thresh: float = 0.9

    # c2 precondition: enough inliers to trust the novelty signal.
    # Paper literally checks `mnMatchesInliers > 15`. Default kept at 50 for
    # parity with previous config / extra defensive quality gate.
    min_tracked: int = 50

    # c1a force-rate / max spacing. Paper "MaxFrames"; IMU-init nutzt fps/4.
    max_frames: int = 30
    # c1b min spacing (vermeidet KF-Bursts). Paper "MinFrames".
    min_frames: int = 1

    # Diagnostic mode: log score but always accept.
    force_accept_all: bool = False

    # Depth bounds are kept for parity with sibling selectors. Not used here.
    min_depth: float = 0.2
    max_depth: float = 35.0


@dataclass
class OrbSlam3Score:
    n_matches: int = 0
    n_kp_prev: int = 0
    n_kp_curr: int = 0
    baseline_matches: int = 0    # nRefMatches-Analog (siehe Modul-Doku)
    ratio: float = 1.0
    frames_since_kf: int = 0
    c1a: bool = False     # spacing: N >= max_frames
    c1b: bool = False     # spacing: N >= min_frames (mapper-idle dropped)
    c2: bool = False      # novelty AND matches > 15 precondition
    triggered_by: str = ""    # "bootstrap" | "novelty" | "force_rate+novelty"
                              # | "baseline_warmup" | "forced_diag"
                              # | "pathological"
    accepted: bool = False


# Internal cache of the last accepted KF.
@dataclass
class _FrameMemo:
    desc: np.ndarray              # (N, 32) ORB descriptors
    n_kp: int                     # total #keypoints
    baseline_matches: int = -1    # nRefMatches-Analog; -1 = not yet set
                                  # (set on first frame after this KF commit)


# =============================================================================
# Selector
# =============================================================================

class OrbSlam3Selector:
    """ORB-SLAM3-style keyframe selector. Same shape as the other selectors."""

    def __init__(self, cfg: OrbSlam3Config, K: np.ndarray, image_hw: tuple[int, int]):
        self.cfg = cfg
        self.K = np.asarray(K, dtype=np.float32)
        self.H, self.W = image_hw

        self.orb = cv2.ORB_create(nfeatures=cfg.orb_n_features)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        self.prev_kf: Optional[_FrameMemo] = None
        self.frames_since_kf: int = 0

    # ------------------------------------------------------------------
    @classmethod
    def from_config(cls, cfg_dict: dict, K: np.ndarray,
                    image_hw: tuple[int, int]) -> "OrbSlam3Selector":
        fields = set(OrbSlam3Config.__dataclass_fields__.keys())
        kwargs = {k: v for k, v in cfg_dict.items() if k in fields}
        return cls(OrbSlam3Config(**kwargs), K, image_hw)

    # ------------------------------------------------------------------
    def should_accept(
        self,
        depth: np.ndarray,
        t: np.ndarray,
        R: np.ndarray,
        rgb: Optional[np.ndarray] = None,
        **_: object,
    ) -> tuple[bool, Optional[OrbSlam3Score]]:
        if rgb is None:
            # No image - cannot run ORB. Be the no-op selector.
            return True, None

        gray = self._to_gray(rgb)
        kps, desc = self.orb.detectAndCompute(gray, None)

        # Pathological frame: no ORB features extractable (extreme blur, dark,
        # featureless). Original tracker would relocalize, not insert a KF.
        # Reject and let force-rate eventually pull us out.
        if desc is None or len(kps) == 0:
            self.frames_since_kf += 1
            score = OrbSlam3Score(
                frames_since_kf=self.frames_since_kf,
                triggered_by="pathological",
                accepted=False,
            )
            if self.cfg.force_accept_all:
                # Diagnose-Modus: trotzdem akzeptieren, sonst hängt der Mapper.
                self._commit(np.empty((0, 32), np.uint8), 0)
                score.triggered_by = "forced_diag"
                score.accepted = True
                return True, score
            return False, score

        n_kp_curr = int(len(kps))

        # First frame: always accept, seed prev_kf, no comparison possible.
        if self.prev_kf is None:
            self._commit(desc, n_kp_curr)
            return True, OrbSlam3Score(
                n_kp_curr=n_kp_curr,
                triggered_by="bootstrap", accepted=True,
            )

        # Frame-to-frame ORB match against the cached prev_kf descriptors.
        n_matches = self._match_count(desc, self.prev_kf.desc)
        n_kp_prev = self.prev_kf.n_kp

        # Baseline-Logik (nRefMatches-Analog).
        #   - Vor dem ersten Folgeframe nach Commit ist baseline_matches = -1.
        #     Diesen Frame nutzen wir, um die Baseline zu messen ("Warm-Up").
        #     Während Warm-Up bleibt c2 zwingend False: ohne Baseline ist die
        #     Ratio-Semantik nicht definiert. Wir akzeptieren NICHT, damit
        #     wir nicht zwei KFs hintereinander erzeugen.
        #   - Sobald baseline gesetzt ist, ratio = n_matches / baseline.
        #   - Degenerate baseline (== 0) → Fallback auf min(n_curr, n_prev)
        #     als sichere obere Schranke.
        warmup = self.prev_kf.baseline_matches < 0
        if warmup:
            self.prev_kf.baseline_matches = n_matches
            baseline = n_matches
            ratio = 1.0
        else:
            baseline = self.prev_kf.baseline_matches
            if baseline <= 0:
                # Defensiv: prev_kf hatte beim Commit-Folgeframe 0 Matches
                # (Tracking-Aussetzer direkt nach KF). min(.) als Schranke.
                baseline = max(min(n_kp_curr, n_kp_prev), 1)
            ratio = float(n_matches) / float(baseline)

        # Spacing side (c1a OR c1b).
        c1a = self.frames_since_kf >= self.cfg.max_frames
        c1b = self.frames_since_kf >= self.cfg.min_frames
        spacing_ok = c1a or c1b

        # Novelty side (c2): ratio-drop AND minimum-inliers precondition.
        # Im Warm-Up-Frame ist c2 per Definition False (ratio = 1.0).
        c2 = (not warmup
              and ratio < self.cfg.tracked_ratio_thresh
              and n_matches >= self.cfg.min_tracked)

        # AND of spacing and novelty — paper-faithful.
        accept = spacing_ok and c2

        trigger = ""
        if accept:
            trigger = "force_rate+novelty" if c1a else "novelty"
        elif warmup:
            trigger = "baseline_warmup"

        if self.cfg.force_accept_all and not accept:
            accept = True
            trigger = "forced_diag"

        score = OrbSlam3Score(
            n_matches=n_matches,
            n_kp_prev=n_kp_prev,
            n_kp_curr=n_kp_curr,
            baseline_matches=baseline,
            ratio=ratio,
            frames_since_kf=self.frames_since_kf,
            c1a=c1a, c1b=c1b, c2=c2,
            triggered_by=trigger,
            accepted=accept,
        )

        if accept:
            self._commit(desc, n_kp_curr)
        else:
            self.frames_since_kf += 1

        return accept, score

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _commit(self, desc: np.ndarray, n_kp: int) -> None:
        # baseline_matches=-1 → wird auf dem ersten Folgeframe gemessen
        self.prev_kf = _FrameMemo(desc=desc, n_kp=n_kp, baseline_matches=-1)
        self.frames_since_kf = 0

    def _match_count(self, desc_a: np.ndarray, desc_b: np.ndarray) -> int:
        if (desc_a is None or desc_b is None
                or len(desc_a) == 0 or len(desc_b) == 0):
            return 0
        ms = self.matcher.match(desc_a, desc_b)
        return len(ms) if ms else 0

    @staticmethod
    def _to_gray(rgb: np.ndarray) -> np.ndarray:
        if rgb.ndim == 2:
            return rgb if rgb.dtype == np.uint8 else rgb.astype(np.uint8)
        g = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY) if rgb.shape[2] == 3 else rgb[..., 0]
        return g.astype(np.uint8) if g.dtype != np.uint8 else g


# =============================================================================
# Smoke test
# =============================================================================

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    H, W = 240, 320
    fx = fy = 0.5 * W / np.tan(np.deg2rad(70.0) / 2)
    K = np.array([[fx, 0, W / 2], [0, fy, H / 2], [0, 0, 1]], np.float32)

    cfg = OrbSlam3Config(
        orb_n_features=400,
        tracked_ratio_thresh=0.9,
        min_tracked=20,
        min_frames=1,
        max_frames=15,
    )
    sel = OrbSlam3Selector(cfg, K, (H, W))

    # Strukturierte Szene mit progressiver Verschiebung: liefert monoton
    # fallenden Match-Ratio.
    base = rng.integers(0, 255, (H, W, 3), dtype=np.uint8)

    def shifted(dx: int) -> np.ndarray:
        out = np.zeros_like(base)
        if dx >= W:
            return rng.integers(0, 255, (H, W, 3), dtype=np.uint8)
        if dx >= 0:
            out[:, dx:, :] = base[:, : W - dx, :]
            out[:, :dx, :] = rng.integers(0, 255, (H, dx, 3), dtype=np.uint8)
        return out

    depth = np.full((H, W), 3.0, dtype=np.float32)
    I = np.eye(3, dtype=np.float32)

    def report(i, accept, sc, label):
        if sc is None:
            print(f"frame {i:2d} {label:>12s}  (no rgb)  {'ACCEPT' if accept else 'skip'}")
            return
        flags = f"c1a={int(sc.c1a)} c1b={int(sc.c1b)} c2={int(sc.c2)}"
        print(f"frame {i:2d} {label:>12s}  N={sc.frames_since_kf:2d} "
              f"matches={sc.n_matches:3d}/baseline={sc.baseline_matches:3d}  "
              f"ratio={sc.ratio:.3f}  {flags}  "
              f"trig={sc.triggered_by or '-':>20s}  "
              f"{'ACCEPT' if accept else 'skip'}")

    accepted = 0

    # Frame 0: bootstrap accept.
    ok, sc = sel.should_accept(depth, np.zeros(3, np.float32), I, base)
    accepted += int(ok); report(0, ok, sc, "bootstrap")

    # Frame 1: Baseline-Warm-Up nach Commit (trig="baseline_warmup").
    # Frames 2-4: stationär, identischer Inhalt -> ratio≈1.0, c2=False, skip.
    for i in range(1, 5):
        ok, sc = sel.should_accept(depth, np.zeros(3, np.float32), I, base)
        accepted += int(ok); report(i, ok, sc, "stationary")

    # Frames 5-19: progressiv verschoben -> ratio fällt, c2 feuert.
    for i in range(5, 20):
        dx = 12 * (i - 4)
        img = shifted(dx)
        ok, sc = sel.should_accept(depth, np.zeros(3, np.float32), I, img)
        accepted += int(ok); report(i, ok, sc, f"shift dx={dx}")

    # Frames 20-29: wieder identischer Inhalt -> c2=False, also auch nach
    # force-rate (c1a) KEIN KF (AND-Logik). Das ist paper-konform: ORB-SLAM
    # erzeugt bei reiner Standkamera keinen Keyframe.
    for i in range(20, 30):
        ok, sc = sel.should_accept(depth, np.zeros(3, np.float32), I, base)
        accepted += int(ok); report(i, ok, sc, "static-2")

    print(f"\nTotal accepted: {accepted}/30  "
          f"(expected: 1 bootstrap + KFs während shift-Block ab Frame 6; "
          f"static-Blöcke 0; nach KF jeweils 1 Warm-Up-Frame zwingend skip)")
