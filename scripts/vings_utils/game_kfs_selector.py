"""
Game-KFS keyframe selector (paper-near variant).

Implements the keyframe selection from Chen S. et al., "Game-KFS: Game-Theory-
Inspired Keyframe Selection for Hybrid Representation Visual SLAM", IEEE RA-L
2025, Section III. Two conceptual agents -- Field Representation Agent (FRA,
Eq. 3) and Discrete Representation Agent (DRA, Eq. 9) -- produce three
sub-scores each; a dynamically smoothed weight lambda_t (Eq. 13-14) trades
them off.

    A_t = beta1 * L_uncert + beta2 * L_render + beta3 * L_covis        (Eq. 3)
    B_t = alpha1 * L_assoc + alpha2 * L_flow  + alpha3 * L_motion      (Eq. 9)

    lam_star = sigmoid(gamma1 * L_assoc + gamma2 * L_render)            (Eq. 13)
    lam_t    = eta * lam_t + (1 - eta) * lam_star                       (Eq. 14, EMA)

    composite = lam_t * A_t + (1 - lam_t) * B_t                         (Eq. 1)
    accept    = composite >= accept_thresh                              (Eq. 2 mit Schwelle)

Konvention: alle sechs Sub-Scores liegen in [0, 1], hoeher = "diesen Frame
eher als KF akzeptieren" (siehe docs/GAME_KFS.md fuer die Begruendung dieser
Polung -- das Paper selbst ist intern inkonsistent zwischen Eq. 10 (Reward)
und Eq. 9/12 (Cost)).

Paper-Nahe-Aenderungen ggu. der ersten Fassung (siehe docs/GAME_KFS.md):
  - L_flow:   echtes Delta-Flow ueber 3 Frames (Eq. 11 verbatim, statt
              einfache Flow-Magnitude). Braucht 2 LK-Passes pro Frame.
  - L_render: PSNR(I_t, warp(prev_kf_gray -> current pose)). Paper-Eq. 7
              vergleicht das echte Render mit dem Bild; ohne Mapper-Zugriff
              ist der naechste-KF-Warp der einzige verfuegbare
              Szenen-Prediktor. Frame-Schaerfe-Fallback bleibt nur wenn
              prev_kf_gray fehlt.
  - L_covis:  symmetrische Jaccard-IoU (Eq. 8 verbatim). Vorher einseitige
              Reprojektion. Braucht prev_kf_depth gecached.
  - L_motion: tanh-Saettigung statt Hard-Clip. Eq. 12 hat im Paper keine
              Saettigung; tanh ist die naechstkleinere [0,1]-konforme
              Variante, die Ordnung im Saettigungsregime bewahrt.
  - Sigmoid:  Eq. 13 literal, kein Recentering-Offset mehr.

Verbleibende Mapper-freie Adaptionen (unaenderbar ohne Mapper-Sync):
  - L_uncert: mean(depth_cov) aus dem DBAF-Tracker statt Var[C] aus dem
    Renderer-Compositing.
  - L_assoc:  ORB+BFMatcher mit RANSAC-Homography statt Tracker-internem
    Inlier/Outlier-Split; Polaritaet invertiert (Paper-Eq. 10 ist Reward,
    aber Eq. 9 addiert sie zu Cost-Termen).
  - Decision: composite >= accept_thresh statt argmin (Paper Eq. 2 ist nie
    sauber definiert -- A_t(d_t=0) wird im Paper nicht angegeben).

Calling convention:

    sel = GameKfsSelector(cfg, K, (H, W))
    accept, score = sel.should_accept(depth, t, R, rgb=rgb_uint8, depth_cov=cov)

`depth_cov` und `rgb` sind optional; fehlende Inputs degradieren auf
neutrale Sub-Scores (siehe Failsafes unten).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    import cv2
except ImportError as e:
    raise ImportError("GameKfsSelector requires opencv-python.") from e

from vings_utils.nurbs_lvi_selector import backproject
from vings_utils.mm3dgs_selector import _laplacian_var, _project_to_kf


# =============================================================================
# Config / score dataclasses
# =============================================================================

@dataclass
class GameKfsConfig:
    # ---- FRA weights (Eq. 3) -- paper defaults ----
    beta_uncert: float = 0.3
    beta_render: float = 0.3
    beta_covis:  float = 0.4

    # ---- DRA weights (Eq. 9) -- paper defaults ----
    alpha_assoc:  float = 0.5
    alpha_flow:   float = 0.3
    alpha_motion: float = 0.2

    # ---- Equilibrium strategy (Eq. 13-14) ----
    gamma_assoc:  float = 1.0
    gamma_render: float = 1.0
    eta:          float = 0.8   # EMA smoothing for lambda
    lambda_init:  float = 0.5

    # ---- Decision threshold ----
    accept_thresh: float = 0.5  # accept iff composite >= thresh

    # ---- DRA detail ----
    orb_n_features: int = 800
    ransac_reproj_thresh: float = 4.0   # px; for findHomography inlier mask
    min_matches: int = 12               # below: tracker-stress force-accept
    flow_ref_px: float = 30.0           # normalises mean(||du||) to [0,1]

    # ---- FRA detail ----
    n_samples: int = 2048               # covis sampling grid
    psnr_target: float = 25.0           # Eq. 7 -- target PSNR (dB) for L_render
    lap_var_ref: float = 500.0          # fallback when prev_kf_gray fehlt
    cov_ref: float = 1.0                # normalises mean(depth_cov) to [0,1]

    # ---- L_motion scaling ----
    trans_ref_m: float = 0.30           # m; tanh-Eingang ||dt|| / trans_ref_m
    omega_rot: float = 0.10             # rad-equivalent weight on ||dR||_F

    # ---- Depth gating ----
    min_depth: float = 0.2
    max_depth: float = 35.0


@dataclass
class GameKfsScore:
    # Sub-scores (alle in [0,1], hoeher = mehr Grund zu akzeptieren)
    L_uncert: float = 0.0
    L_render: float = 0.0
    L_covis:  float = 0.0
    L_assoc:  float = 0.0
    L_flow:   float = 0.0
    L_motion: float = 0.0
    # Aggregated
    A_t: float = 0.0     # FRA composite
    B_t: float = 0.0     # DRA composite
    lambda_t: float = 0.5
    composite: float = 0.0
    # Diagnostics
    n_keypoints: int = 0
    n_matches: int = 0
    n_inliers: int = 0
    psnr: float = 0.0          # measured warp-PSNR (dB), 0 = fallback
    forced: bool = False
    accepted: bool = False


# =============================================================================
# Helpers
# =============================================================================

def _sigmoid(x: float) -> float:
    # numerically-safe scalar sigmoid
    if x >= 0:
        z = float(np.exp(-x))
        return 1.0 / (1.0 + z)
    z = float(np.exp(x))
    return z / (1.0 + z)


def _clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


def _make_uv_samples(H: int, W: int, n: int) -> np.ndarray:
    """Deterministic, aspect-aware pixel-sample grid (mm3dgs-stil)."""
    n = max(64, int(n))
    aspect = W / max(H, 1)
    ny = max(8, int(np.sqrt(n / aspect)))
    nx = max(8, int(n / ny))
    xs = np.linspace(0.5, W - 0.5, nx, dtype=np.float32)
    ys = np.linspace(0.5, H - 0.5, ny, dtype=np.float32)
    uu, vv = np.meshgrid(xs, ys)
    return np.stack([uu.ravel(), vv.ravel()], axis=1)


def _to_gray(rgb: np.ndarray) -> np.ndarray:
    if rgb.ndim == 2:
        out = rgb
    elif rgb.shape[2] == 3:
        out = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    else:
        out = rgb[..., 0]
    return out if out.dtype == np.uint8 else out.astype(np.uint8)


def _bilinear_sample(img: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """Bilinearly sample uint8 image at sub-pixel uv coordinates ((N,2)).

    Returns (N,) float32 array of sampled intensities.
    UV outside the image are clamped to the border; callers should mask
    using `inside`-checks beforehand if that matters.
    """
    H, W = img.shape[:2]
    u = np.clip(uv[:, 0].astype(np.float32), 0.0, np.float32(W - 1))
    v = np.clip(uv[:, 1].astype(np.float32), 0.0, np.float32(H - 1))
    x0 = np.clip(np.floor(u).astype(np.int32), 0, W - 2)
    y0 = np.clip(np.floor(v).astype(np.int32), 0, H - 2)
    x1 = x0 + 1
    y1 = y0 + 1
    fx = u - x0
    fy = v - y0
    a = img[y0, x0].astype(np.float32)
    b = img[y0, x1].astype(np.float32)
    c = img[y1, x0].astype(np.float32)
    d = img[y1, x1].astype(np.float32)
    return (a * (1 - fx) * (1 - fy)
            + b * fx * (1 - fy)
            + c * (1 - fx) * fy
            + d * fx * fy)


def _psnr_from_mse(mse: float) -> float:
    """PSNR in dB for uint8 intensities. Returns 60 dB cap for vanishing MSE."""
    if mse < 1e-3:
        return 60.0
    return 10.0 * float(np.log10(255.0 ** 2 / mse))


# =============================================================================
# Selector
# =============================================================================

class GameKfsSelector:
    """Game-KFS keyframe selector. Same shape as FrameSelector siblings."""

    def __init__(self, cfg: GameKfsConfig, K: np.ndarray,
                 image_hw: tuple[int, int]):
        self.cfg = cfg
        self.K = np.asarray(K, dtype=np.float32)
        self.K_inv = np.linalg.inv(self.K)
        self.H, self.W = image_hw

        self.orb = cv2.ORB_create(nfeatures=cfg.orb_n_features)
        # No crossCheck: we want all matches; RANSAC handles outliers.
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        # Last accepted KF state (anchor for L_motion / L_covis / L_assoc / L_render).
        self.prev_kf_R: Optional[np.ndarray] = None
        self.prev_kf_t: Optional[np.ndarray] = None
        self.prev_kf_gray:  Optional[np.ndarray] = None
        self.prev_kf_depth: Optional[np.ndarray] = None
        # Reference ORB features = last accepted KF's keypoints.
        self.ref_kps_uv: Optional[np.ndarray] = None
        self.ref_desc:    Optional[np.ndarray] = None

        # 3-frame state fuer Delta-Flow (Eq. 11):
        # prev_prev_gray = Bild bei t-2, prev_prev_kps = ORB-Keypoints dort
        # prev_gray      = Bild bei t-1, prev_kps_uv   = ORB-Keypoints dort
        self.prev_prev_gray:    Optional[np.ndarray] = None
        self.prev_prev_kps_uv:  Optional[np.ndarray] = None
        self.prev_gray:         Optional[np.ndarray] = None
        self.prev_kps_uv:       Optional[np.ndarray] = None

        # Lambda EMA state.
        self.lambda_t: float = float(cfg.lambda_init)

        # Cached covis sampling grid.
        self._uv_samples = _make_uv_samples(self.H, self.W, cfg.n_samples)

        # Diagonal scale for L_motion (||I - R||_F <= 2*sqrt(2)).
        self._rot_scale_F = float(np.sqrt(2.0))

    # ------------------------------------------------------------------
    @classmethod
    def from_config(cls, cfg_dict: dict, K: np.ndarray,
                    image_hw: tuple[int, int]) -> "GameKfsSelector":
        fields = set(GameKfsConfig.__dataclass_fields__.keys())
        kwargs = {k: v for k, v in cfg_dict.items() if k in fields}
        return cls(GameKfsConfig(**kwargs), K, image_hw)

    # ------------------------------------------------------------------
    def should_accept(
        self,
        depth: np.ndarray,
        t: np.ndarray,
        R: np.ndarray,
        rgb: Optional[np.ndarray] = None,
        depth_cov: Optional[np.ndarray] = None,
        **_: object,
    ) -> tuple[bool, GameKfsScore]:
        cfg = self.cfg
        score = GameKfsScore(lambda_t=self.lambda_t)

        t = np.asarray(t, dtype=np.float32)
        R = np.asarray(R, dtype=np.float32)

        # ---- Optional ORB pass (needed for L_assoc and the LK init) ----
        gray = None
        curr_kps_uv: np.ndarray = np.empty((0, 2), np.float32)
        curr_desc: Optional[np.ndarray] = None
        if rgb is not None:
            gray = _to_gray(rgb)
            kps, desc = self.orb.detectAndCompute(gray, None)
            if kps is not None and len(kps) > 0:
                curr_kps_uv = np.array([kp.pt for kp in kps], dtype=np.float32)
                curr_desc = desc
        score.n_keypoints = int(len(curr_kps_uv))

        # ---- First frame: seed state, force-accept ----
        if self.prev_kf_R is None:
            score.forced = True
            score.accepted = True
            score.composite = 1.0
            self._commit(R, t, gray, depth, curr_kps_uv, curr_desc)
            self._roll_flow_state(gray, curr_kps_uv)
            return True, score

        # ---- L_motion (always available) ----
        score.L_motion = self._l_motion(t, R)

        # ---- L_assoc (only if we have current + ref descriptors) ----
        n_matches = 0
        n_inliers = 0
        if curr_desc is not None and self.ref_desc is not None \
                and len(curr_desc) > 0 and len(self.ref_desc) > 0:
            score.L_assoc, n_matches, n_inliers = self._l_assoc(
                curr_kps_uv, curr_desc)
        else:
            # No reference -> tracking is degenerate; treat as "needs KF".
            score.L_assoc = 1.0
        score.n_matches = n_matches
        score.n_inliers = n_inliers

        # ---- L_flow (echtes Delta-Flow, Eq. 11) ----
        if gray is not None:
            score.L_flow = self._l_flow_delta(gray)
        else:
            score.L_flow = 0.0  # kein RGB -> kein Flow-Signal

        # ---- L_uncert (mean depth-cov proxy) ----
        score.L_uncert = self._l_uncert(depth_cov)

        # ---- L_render (PSNR-Warp-Proxy, Eq. 7) ----
        score.L_render, score.psnr = self._l_render(gray, depth, t, R)

        # ---- L_covis (symmetrische Jaccard-IoU, Eq. 8) ----
        score.L_covis = self._l_covis(depth, t, R)

        # ---- Aggregate FRA / DRA ----
        score.A_t = (cfg.beta_uncert * score.L_uncert
                     + cfg.beta_render * score.L_render
                     + cfg.beta_covis  * score.L_covis)
        score.B_t = (cfg.alpha_assoc  * score.L_assoc
                     + cfg.alpha_flow * score.L_flow
                     + cfg.alpha_motion * score.L_motion)

        # ---- EMA-smoothed lambda (Eq. 13-14) ----
        # Paper-Intention: lambda gross -> FRA gewinnt. L_assoc im Paper ist
        # "Tracker happy" (high = no DRA crisis -> FRA darf priorisieren).
        # Unsere L_assoc-Polung ist invertiert (high = Tracker stress).
        # Damit Eq. 13 semantisch paper-konform bleibt, verwenden wir hier
        # den ungeflippten L_assoc-Sinn (1 - L_assoc):
        #   high L_assoc (stress) -> niedriger Sigmoid-Input -> niedriges lambda
        #   -> DRA gewinnt (richtig: Tracker-Stress -> DRA muss handeln).
        # B_t und Threshold-Decision arbeiten weiterhin mit dem geflippten Wert.
        z = (cfg.gamma_assoc * (1.0 - score.L_assoc)
             + cfg.gamma_render * score.L_render)
        lam_star = _sigmoid(z)
        self.lambda_t = cfg.eta * self.lambda_t + (1.0 - cfg.eta) * lam_star
        score.lambda_t = float(self.lambda_t)

        # ---- Composite + decision (Eq. 1-2 mit Schwelle) ----
        score.composite = (self.lambda_t * score.A_t
                           + (1.0 - self.lambda_t) * score.B_t)
        accept = bool(score.composite >= cfg.accept_thresh)

        # Force-accept if tracking is degenerate (paper-spirit failsafe).
        if n_matches < cfg.min_matches and self.ref_desc is not None:
            accept = True
            score.forced = True

        # ---- 3-Frame-Flow-Speicher schieben (jeden Frame, nicht nur bei accept) ----
        self._roll_flow_state(gray, curr_kps_uv)

        if accept:
            score.accepted = True
            self._commit(R, t, gray, depth, curr_kps_uv, curr_desc)

        return accept, score

    # ------------------------------------------------------------------
    # Sub-score implementations (all return values in [0, 1],
    # hoeher = mehr Grund den Frame als KF zu akzeptieren).
    # ------------------------------------------------------------------
    def _l_motion(self, t: np.ndarray, R: np.ndarray) -> float:
        """Eq. 12: ||dt|| + omega ||dR||_F. Paper hat keine Saettigung;
        wir verwenden tanh statt hard-clip um Aerial-Skalen nicht abzuschneiden."""
        dt = float(np.linalg.norm(t - self.prev_kf_t))
        dR = float(np.linalg.norm(R - self.prev_kf_R, ord="fro"))
        raw = dt / max(self.cfg.trans_ref_m, 1e-6) \
            + self.cfg.omega_rot * dR / max(self._rot_scale_F, 1e-6)
        return float(np.tanh(raw))

    def _l_assoc(self, curr_kps_uv: np.ndarray,
                 curr_desc: np.ndarray) -> tuple[float, int, int]:
        """Eq. 10: nmatch/nref * exp(-noutlier/ntotal). Paper definiert das
        als Reward (hoch = viele inliers, wenig outlier = Tracker happy = kein
        KF noetig). Wir flippen die Polung auf "Select-Reward" (1 - stability)
        damit alle sechs Sub-Scores dieselbe Richtung haben (siehe Doc).
        """
        raw = self.matcher.knnMatch(curr_desc, self.ref_desc, k=2)
        good = []
        for pair in raw:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < 0.85 * n.distance:
                good.append(m)
        n_total = int(len(curr_desc))
        n_ref = int(len(self.ref_desc))
        n_matches = int(len(good))
        if n_matches < 4:
            # too few matches to RANSAC -> treat as full outlier set
            return _clip01(1.0 - 0.0), n_matches, 0

        src = np.array([curr_kps_uv[m.queryIdx] for m in good], dtype=np.float32)
        dst = np.array([self.ref_kps_uv[m.trainIdx] for m in good], dtype=np.float32)
        try:
            _, mask = cv2.findHomography(src, dst, cv2.RANSAC,
                                         self.cfg.ransac_reproj_thresh)
        except cv2.error:
            mask = None
        n_inliers = 0 if mask is None else int(mask.sum())
        # Paper Eq. 10: n_outliers = "rejected matches" = alles im Matching-
        # Pool, das kein Inlier wurde (Lowe-fail UND RANSAC-fail).
        # Pool-Groesse ist n_total (jedes current feature versucht ein Match).
        # exp(-n_outliers/n_total) wird damit zu exp(n_inliers/n_total - 1),
        # ein smoother Penalty fuer niedrigen Inlier-Anteil.
        n_outliers = max(n_total - n_inliers, 0)
        stability = (n_inliers / max(n_ref, 1)) \
            * float(np.exp(-n_outliers / max(n_total, 1)))
        stability = _clip01(stability)
        return _clip01(1.0 - stability), n_matches, n_inliers

    def _l_flow_delta(self, gray: np.ndarray) -> float:
        """Eq. 11: L_flow = (1/N) Σ ||u_i - u_{i-1}||_2.

        u_i (Flow zum aktuellen Frame) und u_{i-1} (Flow zum vorigen Frame)
        muessen *dieselben Features* tracken. Wir realisieren das mit zwei
        LK-Passes:

            LK1: prev_prev_gray -> prev_gray, startend von prev_prev_kps
                 -> Positionen bei t-1 = kps_at_prev
                 -> u_{t-1} = kps_at_prev - prev_prev_kps
            LK2: prev_gray -> gray, startend von kps_at_prev
                 -> Positionen bei t = kps_at_curr
                 -> u_t = kps_at_curr - kps_at_prev

        Nur Features die in beiden Passes ueberleben (status==1 in beiden)
        gehen in den Mittelwert.
        """
        if (self.prev_prev_gray is None or self.prev_prev_kps_uv is None
                or self.prev_gray is None
                or self.prev_prev_kps_uv.shape[0] < 4):
            return 0.0
        pts0 = self.prev_prev_kps_uv.reshape(-1, 1, 2).astype(np.float32)
        try:
            pts1, status1, _ = cv2.calcOpticalFlowPyrLK(
                self.prev_prev_gray, self.prev_gray, pts0, None,
                winSize=(21, 21), maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
            )
        except cv2.error:
            return 0.0
        if pts1 is None or status1 is None:
            return 0.0
        try:
            pts2, status2, _ = cv2.calcOpticalFlowPyrLK(
                self.prev_gray, gray, pts1, None,
                winSize=(21, 21), maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
            )
        except cv2.error:
            return 0.0
        if pts2 is None or status2 is None:
            return 0.0
        ok = (status1.ravel() == 1) & (status2.ravel() == 1)
        if ok.sum() < 4:
            return 0.0
        u_prev = pts1[ok, 0, :] - pts0[ok, 0, :]
        u_curr = pts2[ok, 0, :] - pts1[ok, 0, :]
        delta_flow = u_curr - u_prev
        mean_delta = float(np.linalg.norm(delta_flow, axis=1).mean())
        return _clip01(mean_delta / max(self.cfg.flow_ref_px, 1e-6))

    def _l_uncert(self, depth_cov: Optional[np.ndarray]) -> float:
        if depth_cov is None:
            return 0.5
        cov = np.asarray(depth_cov, dtype=np.float32)
        if cov.size == 0:
            return 0.5
        finite = cov[np.isfinite(cov)]
        if finite.size == 0:
            return 0.5
        mean_cov = float(finite.mean())
        return _clip01(mean_cov / max(self.cfg.cov_ref, 1e-6))

    def _l_render(self, gray: Optional[np.ndarray],
                  depth: np.ndarray,
                  t: np.ndarray, R: np.ndarray) -> tuple[float, float]:
        """Eq. 7 (PSNR-Variante via prev_kf-Warping).

        Paper: L_render = 1 - PSNR(I_t, I_hat_t) / PSNR_target, wobei I_hat_t
        der Render-Output am aktuellen Pose ist. Ohne Mapper-Zugriff nehmen
        wir den naechstgelegenen Szenen-Prediktor: warp(prev_kf_gray ->
        current pose) via Tiefen-basiertem Backward-Sampling.

        Returns (L_render in [0,1], measured PSNR in dB or 0.0 if fallback).
        """
        # Fallback: Frame-Schaerfe wenn kein KF-Bild oder kein Gray verfuegbar.
        if (gray is None or self.prev_kf_gray is None
                or self.prev_kf_depth is None):
            return self._l_render_fallback(gray, depth), 0.0

        # 1. Backproject current samples -> world.
        uv = self._uv_samples
        world = backproject(uv, depth, self.K_inv, t, R,
                            self.cfg.min_depth, self.cfg.max_depth)
        valid = np.isfinite(world[:, 0])
        if int(valid.sum()) < 16:
            return self._l_render_fallback(gray, depth), 0.0

        world_v = world[valid]
        uv_curr_v = uv[valid]
        # 2. Project into prev_kf camera.
        pts_cam = (world_v - self.prev_kf_t) @ self.prev_kf_R
        z = pts_cam[:, 2]
        in_front = z > 1e-6
        if int(in_front.sum()) < 16:
            return 1.0, 0.0
        pts_cam = pts_cam[in_front]
        z = z[in_front]
        uv_curr_v = uv_curr_v[in_front]
        uv_kf = (pts_cam @ self.K.T)[:, :2] / z[:, None]
        inside = ((uv_kf[:, 0] >= 0) & (uv_kf[:, 0] < self.W - 1)
                  & (uv_kf[:, 1] >= 0) & (uv_kf[:, 1] < self.H - 1))
        if int(inside.sum()) < 16:
            return 1.0, 0.0
        uv_kf_in = uv_kf[inside]
        uv_curr_in = uv_curr_v[inside]

        # 3. Bilinear sample beide Bilder und vergleiche.
        pred = _bilinear_sample(self.prev_kf_gray, uv_kf_in)
        actual = _bilinear_sample(gray, uv_curr_in)
        mse = float(((pred - actual) ** 2).mean())
        psnr = _psnr_from_mse(mse)
        score = _clip01(1.0 - psnr / max(self.cfg.psnr_target, 1e-6))
        return score, psnr

    def _l_render_fallback(self, gray: Optional[np.ndarray],
                           depth: np.ndarray) -> float:
        """LapVar-Schaerfe-Proxy (frueher Default). Bleibt fuer den
        rgb-losen oder ersten-Frame-Fall, wo der PSNR-Warp keinen Sinn macht.
        """
        if gray is not None:
            lv = _laplacian_var(gray)
        else:
            d = np.nan_to_num(depth, nan=0.0)
            d_norm = (np.clip(d / max(self.cfg.max_depth, 1e-6), 0, 1)
                      * 255.0).astype(np.uint8)
            lv = _laplacian_var(d_norm)
        return _clip01(lv / max(self.cfg.lap_var_ref, 1e-6))

    def _l_covis(self, depth: np.ndarray, t: np.ndarray, R: np.ndarray) -> float:
        """Eq. 8: L_covis = 1 - |V_t ∩ V_kf| / |V_t ∪ V_kf|.

        Approximation: Pixel-grid-Samples in beiden Views als Stand-in fuer
        die Gaussian-Supports. Wir backprojecten beide Views in die Welt und
        zaehlen samples die im jeweils anderen View sichtbar sind. Der
        symmetrisch gemittelte Intersection-Schaetzer geht in die Jaccard-IoU.

        Faellt zurueck auf einseitige Reprojektion wenn prev_kf_depth fehlt.
        """
        world_curr = backproject(self._uv_samples, depth, self.K_inv, t, R,
                                 self.cfg.min_depth, self.cfg.max_depth)
        valid_curr = np.isfinite(world_curr[:, 0])
        n_curr = int(valid_curr.sum())

        if n_curr == 0:
            return 1.0  # keine Tiefe -> nicht beurteilbar, favourise accept

        visible_curr_in_kf = _project_to_kf(
            world_curr, self.prev_kf_R, self.prev_kf_t,
            self.K, self.H, self.W)
        n_curr_in_kf = int(visible_curr_in_kf.sum())

        if self.prev_kf_depth is None:
            # einseitiger Fallback (alte Implementierung)
            fwd_covis = n_curr_in_kf / max(n_curr, 1)
            return _clip01(1.0 - fwd_covis)

        world_kf = backproject(self._uv_samples, self.prev_kf_depth,
                               self.K_inv, self.prev_kf_t, self.prev_kf_R,
                               self.cfg.min_depth, self.cfg.max_depth)
        valid_kf = np.isfinite(world_kf[:, 0])
        n_kf = int(valid_kf.sum())
        if n_kf == 0:
            fwd_covis = n_curr_in_kf / max(n_curr, 1)
            return _clip01(1.0 - fwd_covis)

        visible_kf_in_curr = _project_to_kf(world_kf, R, t, self.K, self.H, self.W)
        n_kf_in_curr = int(visible_kf_in_curr.sum())

        # Symmetrische Schaetzung des gemeinsamen Volumens. Beide Werte
        # sind Zwei-Wege-Schaetzer fuer |V_t ∩ V_kf|; das geometrische
        # Mittel ist bei asymmetrischer Frustum-Coverage robuster als das
        # arithmetische (kappen multiplikative Outlier in einer Richtung
        # nicht auf eine Seite). Bei symmetrischen Views identisch.
        intersect = float(np.sqrt(max(n_curr_in_kf, 0) * max(n_kf_in_curr, 0)))
        union = float(n_curr + n_kf) - intersect
        if union <= 0.0:
            return 1.0
        iou = intersect / union
        return _clip01(1.0 - iou)

    # ------------------------------------------------------------------
    def _commit(self, R: np.ndarray, t: np.ndarray,
                gray: Optional[np.ndarray],
                depth: Optional[np.ndarray],
                kps_uv: np.ndarray,
                desc: Optional[np.ndarray]) -> None:
        """Akzeptierte Frame -> alle KF-Caches updaten."""
        self.prev_kf_R = R.copy()
        self.prev_kf_t = t.copy()
        if gray is not None:
            self.prev_kf_gray = gray.copy()
        if depth is not None:
            self.prev_kf_depth = np.asarray(depth, dtype=np.float32).copy()
        if desc is not None and len(desc) > 0:
            self.ref_kps_uv = kps_uv.copy()
            self.ref_desc = desc.copy()

    def _roll_flow_state(self, gray: Optional[np.ndarray],
                         kps_uv: np.ndarray) -> None:
        """3-Frame-Speicher schieben: (t-2, t-1) <- (t-1, t).

        Wird jeden Frame gerufen (auch wenn nicht akzeptiert), damit das
        LK-Signal kontinuierlich bleibt. Wenn kein RGB kam, lassen wir den
        State unveraendert (waere sonst nicht-tracking-konsistent).
        """
        if gray is None:
            return
        self.prev_prev_gray = self.prev_gray
        self.prev_prev_kps_uv = self.prev_kps_uv
        self.prev_gray = gray.copy()
        self.prev_kps_uv = kps_uv.copy() if kps_uv.size > 0 else None


# =============================================================================
# Smoke test
# =============================================================================

if __name__ == "__main__":
    H, W = 240, 320
    fx = fy = 0.5 * W / np.tan(np.deg2rad(70.0) / 2)
    K = np.array([[fx, 0, W / 2], [0, fy, H / 2], [0, 0, 1]], np.float32)
    cfg = GameKfsConfig(
        orb_n_features=600,
        accept_thresh=0.50,
        psnr_target=25.0,
        lap_var_ref=2000.0,
        flow_ref_px=40.0,
        trans_ref_m=1.0,
        min_matches=8,
    )
    sel = GameKfsSelector(cfg, K, (H, W))

    # Strukturierter Hintergrund: rotierter Sinus + harte Kanten, damit ORB
    # stabile Features findet und Match-Counts realistisch sind.
    def make_scene(shift_x: int = 0, blur: bool = False) -> np.ndarray:
        ys, xs = np.indices((H, W), dtype=np.float32)
        pat = (
            127.5 + 80.0 * np.sin(0.18 * (xs - shift_x) + 0.07 * ys)
                  + 60.0 * np.sin(0.31 * (ys + 7.0)
                                  + 0.05 * (xs - shift_x)) * np.cos(0.04 * ys)
        )
        for cx, cy in [(60, 70), (210, 60), (140, 170), (40, 200)]:
            cx2 = cx - shift_x
            pat[cy:cy+18, max(cx2, 0):max(cx2+18, 0)] = 0
            pat[cy+6:cy+12, max(cx2+6, 0):max(cx2+12, 0)] = 255
        img = np.clip(pat, 0, 255).astype(np.uint8)
        rgb = np.dstack([img, img, img])
        if blur:
            rgb = cv2.GaussianBlur(rgb, (21, 21), 0)
        return rgb

    depth = np.full((H, W), 3.0, dtype=np.float32)
    cov = np.full((H, W), 0.05, dtype=np.float32)
    I = np.eye(3, dtype=np.float32)

    def report(idx, accept, sc, tag):
        print(f"f{idx:02d} {tag:>12s} "
              f"L=[unc={sc.L_uncert:.2f} ren={sc.L_render:.2f} cov={sc.L_covis:.2f} "
              f"asc={sc.L_assoc:.2f} flo={sc.L_flow:.2f} mot={sc.L_motion:.2f}] "
              f"A={sc.A_t:.2f} B={sc.B_t:.2f} lam={sc.lambda_t:.2f} "
              f"comp={sc.composite:.2f} "
              f"psnr={sc.psnr:5.1f} "
              f"m={sc.n_matches:3d}/i={sc.n_inliers:3d} "
              f"{'ACCEPT' if accept else 'skip  '}"
              f"{' (forced)' if sc.forced else ''}")

    n_accept = 0
    # Frame 0: seed -- always accepted
    ok, sc = sel.should_accept(depth, np.zeros(3, np.float32), I,
                               rgb=make_scene(0), depth_cov=cov)
    n_accept += int(ok); report(0, ok, sc, "first")

    # Frames 1-4: tiny shifts -> high overlap, should mostly skip
    for i in range(1, 5):
        ok, sc = sel.should_accept(depth, np.zeros(3, np.float32), I,
                                   rgb=make_scene(shift_x=i * 1), depth_cov=cov)
        n_accept += int(ok); report(i, ok, sc, "near-static")

    # Frames 5-9: large translation -> low covis + L_motion saturates;
    # mix in blur to test how L_render damps Sharpness-bait.
    for i in range(5, 10):
        tt = np.array([0.6 * (i - 4), 0.0, 0.0], np.float32)
        blur = (i % 2 == 0)
        ok, sc = sel.should_accept(depth, tt, I,
                                   rgb=make_scene(shift_x=30 * (i - 4),
                                                  blur=blur),
                                   depth_cov=cov)
        n_accept += int(ok); report(i, ok, sc, "translating")

    print(f"\nTotal accepted: {n_accept}/10")
