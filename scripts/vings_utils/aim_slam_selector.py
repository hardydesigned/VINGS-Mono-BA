"""
AIM-SLAM-inspirierter Keyframe-Selector.

Übernimmt das **SIGMA-Modul** aus Jeon, Seo, Lee, Myung, "AIM-SLAM: Dense
Monocular SLAM via Adaptive and Informative Multi-View Keyframe Prioritization
with Foundation Model", arXiv:2603.05097v3 (2026), Sec. III.C.

Drei Stages (alle in [docs/AIM_SLAM.md](../../docs/AIM_SLAM.md) hergeleitet):

  1. Voxel-Overlap O zwischen aktuellem Frame und letztem Map-KF        (Eq. 1)
  2. EKF-Kovarianz-Reduktion -> Information-Gain Γ                      (Eq. 2-3)
  3. Reduced-Chi-Square κ über Hybrid-Residual (Ray + Pixel, Eq. 5)     (Eq. 4)

Decision-Rule (binärer VINGS-Slot, AND-verknüpft):

    accept  ⇔  O < overlap_thresh
            ∧  Γ > gain_thresh
            ∧  (κ > chi_thresh   wenn use_chi_square)

mit Failsafes (erster Frame, O < min_overlap_ratio = Tracker-Stress,
force_accept_all = Diagnose-Modus).

Bewusst NICHT übernommen vom Paper: VGGT-Inferenz, Multi-View-Sim(3)-
Optimierung, Loop-Closure mit DINOv2-Tokens, SL(4)-Submap-Alignment. Wir
übersetzen nur die Selektionslogik, nicht den Foundation-Model-Stack.

Gleiche Schnittstelle wie die anderen Selectoren (siehe selector_factory.py):

    sel = AimSlamSelector(cfg, K, (H, W))
    accept, score = sel.should_accept(depth, t, R, rgb=None, depth_cov=cov)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False


# =============================================================================
# Config / Score dataclasses
# =============================================================================

@dataclass
class AimSlamConfig:
    # ---- Stage 1: Voxel-Overlap (Eq. 1) ----
    voxel_size: float = 0.10            # m / cell
    overlap_thresh: float = 0.70        # skip wenn shared/curr > thresh
    min_overlap_ratio: float = 0.05     # force-accept wenn O < min (Tracker-Stress)
    n_voxel_samples: int = 1024         # Pixel fuer Voxel-Hashing

    # ---- Stage 2: EKF-Information-Gain (Eq. 2-3) ----
    # Threshold ist PRO-Punkt (nats/ray), nicht absolut: Γ/N > thresh.
    # Absolutwert Γ skaliert ~linear mit n_rays_score und ist datensatz-
    # /Intrinsics-abhängig; per-ray ist die robustere Tuning-Achse.
    gain_thresh_per_ray: float = 0.5    # accept wenn Γ/N > thresh
    n_rays_score: int = 256             # Punkte fuer Kovarianz-Reduktion
    pixel_sigma: float = 1.0            # σ_pix Mess-Rausch-Term (px)
    prior_sigma_d: float = 0.10         # σ_d Default wenn depths_cov fehlt (m)
    cov_clip: float = 4.0               # Clip-Range fuer depths_cov vor Skalierung

    # ---- Stage 3: Reduced Chi-Square (Eq. 4 + 5) ----
    use_chi_square: bool = True
    chi_thresh: float = 1.0             # κ > 1 → residuen über rauschen → accept
    chi_dof_offset: int = 0             # M - rank(A); 0 weil pose nicht optimiert (Eq. 4)
    chi_ray_weight: float = 1.0         # optionales Gewicht für Ray-Term
    chi_pix_weight: float = 1.0         # optionales Gewicht für Pixel-Term
    # ORB-Matching liefert pose-unabhängige Korrespondenzen m_{i→j} (Paper-treu).
    # Ohne rgb fällt der Test auf Reprojection-basierte Korrespondenz zurück.
    chi_orb_n_features: int = 800
    chi_min_matches: int = 20           # Fallback wenn ORB zu wenig Matches findet
    chi_max_disparity_px: float = 200.0 # Match-Filter (Pixel-Drift > thresh = outlier)

    # ---- Tiefen-Gate ----
    min_depth: float = 0.2
    max_depth: float = 35.0

    # ---- Diagnose ----
    force_accept_all: bool = False


@dataclass
class AimSlamScore:
    # Stage-Statistiken
    overlap: float = 0.0           # O ∈ [0, 1]
    info_gain: float = 0.0         # Σ Γ ≥ 0 (nats, absolut, fürs Logging)
    info_gain_per_ray: float = 0.0 # Γ / N (nats/ray, Tuning-relevant)
    chi_square: float = 0.0        # κ ≥ 0
    # Diagnostics
    n_voxels_curr: int = 0
    n_voxels_shared: int = 0
    n_rays_used: int = 0
    n_orb_matches: int = 0         # 0 = ORB nicht genutzt (Fallback)
    chi_source: str = "none"       # "orb" | "reproject" | "none"
    forced: bool = False
    accepted: bool = False


# =============================================================================
# Helpers
# =============================================================================

def _voxel_set_from_world_points(
    world_pts: np.ndarray, voxel_size: float
) -> set[tuple[int, int, int]]:
    """Hash (N, 3) Welt-Punkte zu Voxel-IDs."""
    if world_pts.size == 0:
        return set()
    finite = np.isfinite(world_pts).all(axis=1)
    if not finite.any():
        return set()
    ids = np.floor(world_pts[finite] / voxel_size).astype(np.int64)
    # set-Konvertierung: unique tuples
    return set(map(tuple, ids.tolist()))


# =============================================================================
# Selector
# =============================================================================

class AimSlamSelector:
    """SIGMA-Modul als binärer KF-Selector. Same shape as the others."""

    def __init__(self, cfg: AimSlamConfig, K: np.ndarray,
                 image_hw: tuple[int, int]):
        self.cfg = cfg
        self.K = np.asarray(K, dtype=np.float32)
        self.K_inv = np.linalg.inv(self.K).astype(np.float32)
        self.H, self.W = image_hw

        # Anchor: letzter akzeptierter KF
        self._last_R: Optional[np.ndarray] = None
        self._last_t: Optional[np.ndarray] = None
        self._last_depth: Optional[np.ndarray] = None
        self._last_cov: Optional[np.ndarray] = None
        self._last_voxels: set[tuple[int, int, int]] = set()
        self._last_gray: Optional[np.ndarray] = None  # für ORB-Matching

        # Deterministische Subsampling-Grids (anchor-aspekt-aware)
        self._uv_voxel = self._make_uv_grid(cfg.n_voxel_samples)
        self._uv_score = self._make_uv_grid(cfg.n_rays_score)

        # Precomputed homogeneous pixel coords (für vektorisierte
        # Backprojection eines Voxel- oder Score-Sample-Sets)
        self._fx = float(self.K[0, 0])
        self._fy = float(self.K[1, 1])
        self._cx = float(self.K[0, 2])
        self._cy = float(self.K[1, 2])

        # ORB für paper-treue Korrespondenz in Stage 3 (Eq. 5 m_{i→j}).
        # Pose-unabhängige Matches sind die Voraussetzung dafür, dass κ
        # tatsächlich Pose-Konsistenz testet — nicht Tiefen-Konsistenz unter
        # gegebener Pose. Wenn cv2 fehlt, bleibt der Reprojection-Fallback.
        if _HAS_CV2:
            self._orb = cv2.ORB_create(nfeatures=cfg.chi_orb_n_features)
            self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        else:
            self._orb = None
            self._matcher = None

    # ------------------------------------------------------------------
    @classmethod
    def from_config(cls, cfg_dict: dict, K: np.ndarray,
                    image_hw: tuple[int, int]) -> "AimSlamSelector":
        fields = set(AimSlamConfig.__dataclass_fields__.keys())
        kwargs = {k: v for k, v in cfg_dict.items() if k in fields}
        return cls(AimSlamConfig(**kwargs), K, image_hw)

    # ------------------------------------------------------------------
    def _make_uv_grid(self, n: int) -> np.ndarray:
        """Aspect-aware Pixel-Grid (uv float32, shape (N, 2))."""
        n = max(64, int(n))
        aspect = self.W / max(self.H, 1)
        ny = max(8, int(np.sqrt(n / aspect)))
        nx = max(8, int(n / ny))
        xs = np.linspace(0.5, self.W - 0.5, nx, dtype=np.float32)
        ys = np.linspace(0.5, self.H - 0.5, ny, dtype=np.float32)
        uu, vv = np.meshgrid(xs, ys)
        return np.stack([uu.ravel(), vv.ravel()], axis=1)

    # ------------------------------------------------------------------
    def _backproject_to_world(
        self,
        uv: np.ndarray,
        depth: np.ndarray,
        t: np.ndarray,
        R: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """uv (N,2), depth (H,W) → (world_pts (M,3), depth_per_point (M,))
        nur Punkte mit gültiger Tiefe (in [min_depth, max_depth])."""
        if uv.size == 0:
            return np.empty((0, 3), np.float32), np.empty((0,), np.float32)

        u = np.clip(uv[:, 0].astype(np.int64), 0, self.W - 1)
        v = np.clip(uv[:, 1].astype(np.int64), 0, self.H - 1)
        d = depth[v, u].astype(np.float32)
        mask = (d > self.cfg.min_depth) & (d < self.cfg.max_depth) & np.isfinite(d)
        if not mask.any():
            return np.empty((0, 3), np.float32), np.empty((0,), np.float32)

        u_m = uv[mask, 0]
        v_m = uv[mask, 1]
        d_m = d[mask]

        # Camera-frame Punkte: K^-1 [u,v,1] * d
        x_cam = (u_m - self._cx) / self._fx * d_m
        y_cam = (v_m - self._cy) / self._fy * d_m
        z_cam = d_m
        pts_cam = np.stack([x_cam, y_cam, z_cam], axis=1).astype(np.float32)

        # c2w: world = R @ cam + t
        world_pts = (R @ pts_cam.T).T + t.astype(np.float32)
        return world_pts, d_m

    # ------------------------------------------------------------------
    def _project_world_to_cam(
        self,
        world_pts: np.ndarray,
        t: np.ndarray,
        R: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """World → Pixel im Ziel-View (R, t als c2w).
        Returns (pix (M,2), z (M,) in cam, mask_inbounds (M,))."""
        if world_pts.size == 0:
            return (np.empty((0, 2), np.float32),
                    np.empty((0,), np.float32),
                    np.empty((0,), bool))
        # World → Cam: cam = R^T (world - t)
        cam = (world_pts - t.astype(np.float32)) @ R  # = R^T @ (world - t)^T transposed
        z = cam[:, 2]
        valid_z = z > 1e-6
        # avoid div-by-zero; fill non-valid mit 1.0 → wird durch mask gefiltert
        z_safe = np.where(valid_z, z, 1.0)
        u = self._fx * cam[:, 0] / z_safe + self._cx
        v = self._fy * cam[:, 1] / z_safe + self._cy
        in_bounds = (u >= 0) & (u < self.W) & (v >= 0) & (v < self.H) & valid_z
        pix = np.stack([u, v], axis=1).astype(np.float32)
        return pix, z.astype(np.float32), in_bounds

    # ------------------------------------------------------------------
    def _sample_prior_sigma_d(
        self, uv: np.ndarray, cov: Optional[np.ndarray], n_pts: int,
        mask: np.ndarray,
    ) -> np.ndarray:
        """σ_d pro Punkt (m). Aus depths_cov falls vorhanden, sonst Default."""
        if cov is None:
            return np.full((n_pts,), self.cfg.prior_sigma_d, dtype=np.float32)
        u = np.clip(uv[mask, 0].astype(np.int64), 0, self.W - 1)
        v = np.clip(uv[mask, 1].astype(np.int64), 0, self.H - 1)
        c = cov[v, u].astype(np.float32)
        c = np.where(np.isfinite(c), c, self.cfg.prior_sigma_d ** 2)
        c = np.clip(c, 1e-8, self.cfg.cov_clip)
        return np.sqrt(c).astype(np.float32)

    # ------------------------------------------------------------------
    def should_accept(
        self,
        depth: np.ndarray,
        t: np.ndarray,
        R: np.ndarray,
        rgb: Optional[np.ndarray] = None,
        depth_cov: Optional[np.ndarray] = None,
        **_: object,
    ) -> tuple[bool, AimSlamScore]:
        cfg = self.cfg
        score = AimSlamScore()

        t = np.asarray(t, dtype=np.float32)
        R = np.asarray(R, dtype=np.float32)

        # ---- Erster Frame: Seed, accept ----
        if self._last_R is None:
            score.forced = True
            score.accepted = True
            score.overlap = 0.0
            self._commit(R, t, depth, depth_cov, rgb)
            return True, score

        # ---- Diagnose-Modus ----
        if cfg.force_accept_all:
            # Trotzdem alle drei Statistiken berechnen (für Logging)
            self._fill_stats(score, depth, t, R, depth_cov, rgb)
            score.forced = True
            score.accepted = True
            self._commit(R, t, depth, depth_cov, rgb)
            return True, score

        # ---- Stage 1: Voxel-Overlap ----
        curr_world, _ = self._backproject_to_world(self._uv_voxel, depth, t, R)
        curr_voxels = _voxel_set_from_world_points(curr_world, cfg.voxel_size)
        n_curr = len(curr_voxels)
        score.n_voxels_curr = n_curr
        if n_curr == 0:
            # Keine Tiefe -> Tracker-Stress, force-accept
            score.forced = True
            score.accepted = True
            score.overlap = 0.0
            self._commit(R, t, depth, depth_cov)
            return True, score

        shared = curr_voxels & self._last_voxels
        score.n_voxels_shared = len(shared)
        score.overlap = float(len(shared)) / float(n_curr)

        # Failsafe: zu wenig Overlap -> Szenenwechsel/Tracking-Stress
        if score.overlap < cfg.min_overlap_ratio:
            score.forced = True
            score.accepted = True
            self._commit(R, t, depth, depth_cov, rgb)
            return True, score

        # Stage-1-Skip: zu redundant
        if score.overlap > cfg.overlap_thresh:
            return False, score

        # ---- Stage 2: EKF-Information-Gain ----
        score.info_gain, n_used = self._stage2_info_gain(
            depth, t, R, depth_cov)
        score.n_rays_used = n_used
        score.info_gain_per_ray = (
            score.info_gain / n_used if n_used > 0 else 0.0)

        if n_used == 0 or score.info_gain_per_ray < cfg.gain_thresh_per_ray:
            return False, score

        # ---- Stage 3: Reduced-Chi-Square ----
        if cfg.use_chi_square:
            score.chi_square, score.n_orb_matches, score.chi_source = (
                self._stage3_chi_square(depth, t, R, depth_cov, rgb))
            if score.chi_square < cfg.chi_thresh:
                return False, score

        # ---- Accept: state update ----
        score.accepted = True
        self._commit(R, t, depth, depth_cov, rgb)
        return True, score

    # ------------------------------------------------------------------
    def _fill_stats(
        self, score: AimSlamScore, depth: np.ndarray,
        t: np.ndarray, R: np.ndarray, depth_cov: Optional[np.ndarray],
        rgb: Optional[np.ndarray],
    ) -> None:
        """Diagnose-Modus: alle drei Statistiken berechnen, kein Skip."""
        curr_world, _ = self._backproject_to_world(self._uv_voxel, depth, t, R)
        curr_voxels = _voxel_set_from_world_points(curr_world, self.cfg.voxel_size)
        n_curr = len(curr_voxels)
        score.n_voxels_curr = n_curr
        if n_curr > 0:
            shared = curr_voxels & self._last_voxels
            score.n_voxels_shared = len(shared)
            score.overlap = float(len(shared)) / float(n_curr)
        score.info_gain, score.n_rays_used = self._stage2_info_gain(
            depth, t, R, depth_cov)
        score.info_gain_per_ray = (
            score.info_gain / score.n_rays_used
            if score.n_rays_used > 0 else 0.0)
        if self.cfg.use_chi_square:
            score.chi_square, score.n_orb_matches, score.chi_source = (
                self._stage3_chi_square(depth, t, R, depth_cov, rgb))

    # ------------------------------------------------------------------
    def _stage2_info_gain(
        self,
        depth: np.ndarray,
        t: np.ndarray,
        R: np.ndarray,
        depth_cov: Optional[np.ndarray],
    ) -> tuple[float, int]:
        """EKF-Update Eq. 2 + 3: vektorisiert über alle Sample-Punkte.

        Pro Punkt x_k:
          P_k  = σ²_d * I_3              (Prior, 3x3 isotrop)
          J    = ∂π/∂X  (2x3 im Ziel-View)
          R_m  = σ²_pix * I_2            (Mess-Kov, 2x2)
          P_k+ = P_k - P_k Jᵀ (R_m + J P_k Jᵀ)⁻¹ J P_k

        Für isotropes P_k = s² I_3 vereinfacht sich:
          P_k Jᵀ = s² Jᵀ           (3x2)
          J P_k Jᵀ = s² J Jᵀ       (2x2)
          S    = R_m + s² J Jᵀ
          K    = s² Jᵀ S⁻¹         (3x2 Kalman-Gain)
          P_k+ = s² (I_3 - Jᵀ S⁻¹ J s²) ... numerisch stabiler:
          P_k+ = P_k - K J P_k

        Info-Gain pro Punkt: 0.5 * log( det(P_k) / det(P_k+) )
        """
        # Source-Punkte aus dem letzten KF (im Welt-Frame)
        world_pts, _ = self._backproject_to_world(
            self._uv_score, self._last_depth, self._last_t, self._last_R)
        if world_pts.shape[0] == 0:
            return 0.0, 0

        # Reproject in den aktuellen View
        pix, z_cam, in_bounds = self._project_world_to_cam(world_pts, t, R)
        if not in_bounds.any():
            return 0.0, 0

        wp_v = world_pts[in_bounds]
        z = z_cam[in_bounds]
        N = wp_v.shape[0]

        # Prior σ_d aus dem letzten KF (am Source-Pixel) → σ²_d
        # _sample_prior_sigma_d will den Mask-Vektor über uv_score; wir nehmen
        # erst alle uv, dann maskieren auf gültige Tiefen, dann nochmal auf
        # in_bounds. Vereinfachung: wir backprojizieren noch einmal mit dem
        # gleichen Mask-Logic und sammeln σ_d über die same mask.
        # Stattdessen: σ_d aus _last_cov direkt am uv_score-Index lesen.
        # Wir brauchen aber dieselbe Auswahl. Trick: berechne uv_valid = uv
        # für die Punkte, die backprojection überlebt haben.
        # Da _backproject_to_world die Mask intern verwirft, holen wir die σ_d
        # ueber dieselbe Logik nach:
        u_all = np.clip(self._uv_score[:, 0].astype(np.int64), 0, self.W - 1)
        v_all = np.clip(self._uv_score[:, 1].astype(np.int64), 0, self.H - 1)
        d_all = self._last_depth[v_all, u_all].astype(np.float32)
        mask_valid = ((d_all > self.cfg.min_depth)
                      & (d_all < self.cfg.max_depth)
                      & np.isfinite(d_all))
        # σ_d Vektor parallel zu world_pts
        sigma_d_all = self._sample_prior_sigma_d(
            self._uv_score, self._last_cov, int(mask_valid.sum()), mask_valid)
        # weiter maskieren auf in_bounds
        sigma_d = sigma_d_all[in_bounds]
        s2 = (sigma_d ** 2).astype(np.float32)                         # (N,)

        # Pinhole-Jacobian ∂π/∂X im Ziel-View. X_cam = R^T (X_w - t).
        # ∂π/∂X_cam = [[fx/Z, 0, -fx*X/Z²], [0, fy/Z, -fy*Y/Z²]]
        # Die Map X_w → π ist die Komposition; J vs. X_w = ∂π/∂X_cam · R^T.
        # Wir formulieren P_k im *Welt-Frame* (P_w = σ²_d * I_3 — isotrop, also
        # rotationsinvariant), damit gilt:
        #   J_w P_w J_wᵀ = σ²_d * J_cam J_camᵀ   (R drops weil R Rᵀ = I)
        # und wir koennen direkt mit J_cam rechnen.
        cam = (wp_v - t) @ R                                            # (N, 3)
        X, Y, Z = cam[:, 0], cam[:, 1], cam[:, 2]
        fx, fy = self._fx, self._fy
        # J_cam (2x3) pro Punkt; baue Tensor (N, 2, 3)
        J = np.zeros((N, 2, 3), dtype=np.float32)
        J[:, 0, 0] = fx / Z
        J[:, 0, 2] = -fx * X / (Z * Z)
        J[:, 1, 1] = fy / Z
        J[:, 1, 2] = -fy * Y / (Z * Z)

        # S = R_m + s² J Jᵀ ∈ R^{2x2} pro Punkt
        JJt = np.einsum("nij,nkj->nik", J, J)                          # (N, 2, 2)
        Rm = (self.cfg.pixel_sigma ** 2) * np.eye(2, dtype=np.float32)
        S = Rm[None, :, :] + s2[:, None, None] * JJt                    # (N, 2, 2)
        # det(S) > 0 garantiert; 2x2 inverse explizit
        a, b = S[:, 0, 0], S[:, 0, 1]
        c, d_ = S[:, 1, 0], S[:, 1, 1]
        det_S = a * d_ - b * c
        det_S = np.where(np.abs(det_S) < 1e-12, 1e-12, det_S)

        # Info-Gain: 0.5 log( det(P_w) / det(P_w+) )
        # Für isotropes P_w = s² I_3:
        #   det(P_w)  = s^6
        #   P_w+ = P_w - K (J P_w) = (I - s² Jᵀ S⁻¹ J) * s² I_3 ... das ist
        #     nicht mehr isotrop. Aber:
        #   det(P_w+) / det(P_w) = det(I_3 - s² Jᵀ S⁻¹ J)
        #                        = det(I_2 - s² J Jᵀ S⁻¹)      (Matrix-Det-Lemma)
        #                        = det(S⁻¹ R_m)                 (S = R_m + s²JJᵀ)
        #                        = det(R_m) / det(S)
        # =>  0.5 log( det(P_w) / det(P_w+) ) = 0.5 log( det(S) / det(R_m) )
        det_Rm = self.cfg.pixel_sigma ** 4                              # det(σ² I_2) = σ^4
        per_point_gain = 0.5 * np.log(np.maximum(det_S / max(det_Rm, 1e-30), 1.0))
        # log(...) >= 0 weil S = R_m + (psd) ⇒ det(S) >= det(R_m). Clip für
        # numerische Sicherheit.

        return float(per_point_gain.sum()), int(N)

    # ------------------------------------------------------------------
    def _stage3_chi_square(
        self,
        depth: np.ndarray,
        t: np.ndarray,
        R: np.ndarray,
        depth_cov: Optional[np.ndarray],
        rgb: Optional[np.ndarray],
    ) -> tuple[float, int, str]:
        """Reduced-Chi-Square nach Eq. 4 mit Hybrid-Residual aus Eq. 5.

        Pro Korrespondenz (p_k, p_f) ein 5D-Residuum:
            r_ij = [Ψ_ray(X_k|k) − Ψ_ray(T^k_f X_f|f),
                    Ψ_π (K, X_k|k) − Ψ_π (K, T^k_f X_f|f)]  ∈ R^{3+2}

        Korrespondenz-Quelle (paper-treuer Pfad):
          ORB-Features in letztem KF und aktuellem Frame, BF-Crosscheck-Match.
          Damit ist (p_k, p_f) **unabhängig von der DBAF-Pose** — und κ misst
          tatsächlich, ob die Pose-Hypothese mit den Matches konsistent ist
          (Paper-Eq.-4-Semantik).

        Fallback (kein rgb / cv2 / zu wenig Matches):
          Korrespondenz via Reprojektion (DBAF-Pose definiert (p_k, p_f)).
          Der Test misst dann *joint Tiefen-Pose-Konsistenz* — schwächer als
          die paper-treue Form, aber bricht nicht ab.

        Whitening: r_pix / σ_pix, r_ray / (σ_pix / f̄).
        DoF = M − chi_dof_offset (Default 0, weil VINGS-Pose nicht optimiert).

        Returns: (κ, n_orb_matches, source) mit source ∈ {"orb","reproject"}.
        """
        if (self._orb is not None and rgb is not None
                and self._last_gray is not None):
            res = self._stage3_chi_orb(depth, t, R, rgb)
            if res is not None:
                kappa, n_matches = res
                return kappa, n_matches, "orb"
        # Fallback
        return self._stage3_chi_reproject(depth, t, R), 0, "reproject"

    # ------------------------------------------------------------------
    def _stage3_chi_orb(
        self,
        depth: np.ndarray,
        t: np.ndarray,
        R: np.ndarray,
        rgb: np.ndarray,
    ) -> Optional[tuple[float, int]]:
        """Paper-treue Eq. 4 mit ORB-basierten, pose-unabhängigen Matches.

        Returns None, wenn zu wenig Matches oder Tiefen verfügbar → caller
        wechselt auf Reprojection-Fallback.
        """
        cfg = self.cfg

        # ---- ORB Detection + Matching ----
        gray_f = self._to_gray(rgb)
        kp_k, desc_k = self._orb.detectAndCompute(self._last_gray, None)
        kp_f, desc_f = self._orb.detectAndCompute(gray_f, None)
        if desc_k is None or desc_f is None or len(kp_k) < 8 or len(kp_f) < 8:
            return None

        try:
            matches = self._matcher.match(desc_k, desc_f)
        except cv2.error:
            return None
        if len(matches) < cfg.chi_min_matches:
            return None

        # Sub-pixel Koordinaten
        p_k = np.array([kp_k[m.queryIdx].pt for m in matches], dtype=np.float32)
        p_f = np.array([kp_f[m.trainIdx].pt for m in matches], dtype=np.float32)

        # Outlier-Filter: groteske Pixel-Drifts rauswerfen (Mismatches)
        drift = np.linalg.norm(p_f - p_k, axis=1)
        keep = drift < cfg.chi_max_disparity_px
        if keep.sum() < cfg.chi_min_matches:
            return None
        p_k = p_k[keep]
        p_f = p_f[keep]

        # ---- Bilinear Depth Sampling an den Match-Sub-Pixeln ----
        d_k, m_k = self._sample_depth_bilinear(self._last_depth, p_k)
        d_f, m_f = self._sample_depth_bilinear(depth, p_f)
        m = m_k & m_f
        if m.sum() < cfg.chi_min_matches:
            return None
        p_k = p_k[m]; p_f = p_f[m]; d_k = d_k[m]; d_f = d_f[m]

        # ---- Backproject: X_k|k im KF-Cam, X_f|f im aktuellem Cam ----
        X_k = np.stack([
            (p_k[:, 0] - self._cx) / self._fx * d_k,
            (p_k[:, 1] - self._cy) / self._fy * d_k,
            d_k,
        ], axis=1).astype(np.float32)
        X_f = np.stack([
            (p_f[:, 0] - self._cx) / self._fx * d_f,
            (p_f[:, 1] - self._cy) / self._fy * d_f,
            d_f,
        ], axis=1).astype(np.float32)

        # ---- T^k_f X_f|f → KF-Cam-Koord ----
        X_f_world = X_f @ R.T + t
        X_f_in_k = (X_f_world - self._last_t) @ self._last_R

        # ---- Hybrid-Residual + κ (gemeinsame Berechnung mit Reproject) ----
        return self._compute_kappa(X_k, X_f_in_k), int(X_k.shape[0])

    # ------------------------------------------------------------------
    def _stage3_chi_reproject(
        self,
        depth: np.ndarray,
        t: np.ndarray,
        R: np.ndarray,
    ) -> float:
        """Fallback: Reprojection-basierte Korrespondenz (nicht paper-treu;
        misst Tiefen-Konsistenz unter gegebener DBAF-Pose). Bilineares Depth-
        Sampling für Sub-Pixel-Genauigkeit.
        """
        cfg = self.cfg

        # 1. Sample im letzten KF, backproject (nearest-grid ist OK, weil uv
        # selbst auf 0.5-Pixel-Grid liegt → kein NN-Round-Off-Loss)
        uv = self._uv_score
        u0 = np.clip(uv[:, 0].astype(np.int64), 0, self.W - 1)
        v0 = np.clip(uv[:, 1].astype(np.int64), 0, self.H - 1)
        d_k = self._last_depth[v0, u0].astype(np.float32)
        m0 = ((d_k > cfg.min_depth) & (d_k < cfg.max_depth) & np.isfinite(d_k))
        if not m0.any():
            return 0.0
        u_k = uv[m0, 0]; v_k = uv[m0, 1]; d_k = d_k[m0]
        X_k = np.stack([
            (u_k - self._cx) / self._fx * d_k,
            (v_k - self._cy) / self._fy * d_k,
            d_k,
        ], axis=1).astype(np.float32)

        # 2. Reprojektion → erwartetes Pixel im aktuellen Frame
        X_k_world = X_k @ self._last_R.T + self._last_t
        cam_f = (X_k_world - t) @ R
        Z_f = cam_f[:, 2]
        valid_z = Z_f > 1e-3
        Z_safe = np.where(valid_z, Z_f, 1.0)
        u_pred = self._fx * cam_f[:, 0] / Z_safe + self._cx
        v_pred = self._fy * cam_f[:, 1] / Z_safe + self._cy
        in_b = (valid_z & (u_pred >= 0) & (u_pred < self.W)
                & (v_pred >= 0) & (v_pred < self.H))
        if not in_b.any():
            return 0.0
        X_k = X_k[in_b]; u_pred = u_pred[in_b]; v_pred = v_pred[in_b]

        # 3. Bilineares Depth-Sampling an Sub-Pixel-Koordinate
        p_pred = np.stack([u_pred, v_pred], axis=1)
        d_obs, m_obs = self._sample_depth_bilinear(depth, p_pred)
        if not m_obs.any():
            return 0.0
        X_k = X_k[m_obs]; u_pred = u_pred[m_obs]; v_pred = v_pred[m_obs]
        d_obs = d_obs[m_obs]

        # 4. X_f → KF-Cam-Koord
        X_f_cam = np.stack([
            (u_pred - self._cx) / self._fx * d_obs,
            (v_pred - self._cy) / self._fy * d_obs,
            d_obs,
        ], axis=1).astype(np.float32)
        X_f_world = X_f_cam @ R.T + t
        X_f_in_k = (X_f_world - self._last_t) @ self._last_R

        return self._compute_kappa(X_k, X_f_in_k)

    # ------------------------------------------------------------------
    def _compute_kappa(
        self, X_k: np.ndarray, X_f_in_k: np.ndarray
    ) -> float:
        """Hybrid-Residual (Eq. 5) + Reduced-Chi-Square (Eq. 4).

        Erhält zwei (N, 3) Punktewolken im KF-Cam-Koord, baut den 5D-Residuum-
        Vektor pro Punkt und gibt κ = ‖b‖²/(M − dof_offset) zurück.
        """
        cfg = self.cfg
        eps = 1e-9

        n_k = np.linalg.norm(X_k, axis=1, keepdims=True)
        n_f = np.linalg.norm(X_f_in_k, axis=1, keepdims=True)
        valid_norm = (n_k[:, 0] > eps) & (n_f[:, 0] > eps)
        if not valid_norm.any():
            return 0.0
        X_k = X_k[valid_norm]
        X_f_in_k = X_f_in_k[valid_norm]
        n_k = n_k[valid_norm]
        n_f = n_f[valid_norm]

        # Eq. 5 Term 1: Ray-Differenz auf Einheitssphäre
        r_ray = X_k / n_k - X_f_in_k / n_f

        # Eq. 5 Term 2: Pixel-Reprojektions-Differenz (beide via K aus KF)
        Zk = X_k[:, 2]; Zf = X_f_in_k[:, 2]
        valid_pix = (Zk > 1e-3) & (Zf > 1e-3)
        if not valid_pix.any():
            return 0.0
        r_ray = r_ray[valid_pix]
        Xk = X_k[valid_pix]; Xf = X_f_in_k[valid_pix]
        Zk = Zk[valid_pix]; Zf = Zf[valid_pix]
        pix_k = np.stack([
            self._fx * Xk[:, 0] / Zk + self._cx,
            self._fy * Xk[:, 1] / Zk + self._cy,
        ], axis=1)
        pix_f = np.stack([
            self._fx * Xf[:, 0] / Zf + self._cx,
            self._fy * Xf[:, 1] / Zf + self._cy,
        ], axis=1)
        r_pix = pix_k - pix_f

        # Whitening: b = [r_ray/σ_ray, r_pix/σ_pix] mit σ_ray = σ_pix/f̄
        sigma_pix = max(cfg.pixel_sigma, 1e-6)
        f_mean = 0.5 * (self._fx + self._fy)
        sigma_ray = sigma_pix / max(f_mean, 1e-6)
        b_ray = (r_ray / sigma_ray) * cfg.chi_ray_weight
        b_pix = (r_pix / sigma_pix) * cfg.chi_pix_weight

        bb = float(np.sum(b_ray * b_ray) + np.sum(b_pix * b_pix))
        M = int(b_ray.size + b_pix.size)
        dof = max(M - cfg.chi_dof_offset, 1)
        return bb / dof

    # ------------------------------------------------------------------
    def _sample_depth_bilinear(
        self, depth: np.ndarray, uv: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Bilineares Depth-Sampling an Sub-Pixel-Koords. Returns (d, valid_mask).
        valid = alle 4 Nachbarn finite und in [min_depth, max_depth].
        """
        cfg = self.cfg
        u = uv[:, 0]; v = uv[:, 1]
        u0 = np.floor(u).astype(np.int64)
        v0 = np.floor(v).astype(np.int64)
        u1 = u0 + 1; v1 = v0 + 1
        # Out-of-bounds → invalid
        in_bounds = ((u0 >= 0) & (u1 < self.W) & (v0 >= 0) & (v1 < self.H))
        u0c = np.clip(u0, 0, self.W - 1); u1c = np.clip(u1, 0, self.W - 1)
        v0c = np.clip(v0, 0, self.H - 1); v1c = np.clip(v1, 0, self.H - 1)
        d00 = depth[v0c, u0c].astype(np.float32)
        d01 = depth[v0c, u1c].astype(np.float32)
        d10 = depth[v1c, u0c].astype(np.float32)
        d11 = depth[v1c, u1c].astype(np.float32)
        du = (u - u0).astype(np.float32)
        dv = (v - v0).astype(np.float32)
        d = ((1 - du) * (1 - dv) * d00 + du * (1 - dv) * d01
             + (1 - du) * dv * d10 + du * dv * d11)
        valid_d = (np.isfinite(d00) & np.isfinite(d01)
                   & np.isfinite(d10) & np.isfinite(d11)
                   & (d00 > cfg.min_depth) & (d00 < cfg.max_depth)
                   & (d01 > cfg.min_depth) & (d01 < cfg.max_depth)
                   & (d10 > cfg.min_depth) & (d10 < cfg.max_depth)
                   & (d11 > cfg.min_depth) & (d11 < cfg.max_depth))
        return d, (in_bounds & valid_d)

    # ------------------------------------------------------------------
    def _to_gray(self, rgb: np.ndarray) -> np.ndarray:
        """RGB (H, W, 3) uint8 → Graustufen (H, W) uint8. Tolerant gegen
        bereits-Graustufen und float32-Inputs."""
        if rgb.ndim == 2:
            g = rgb
        elif rgb.ndim == 3 and rgb.shape[2] >= 3:
            g = cv2.cvtColor(rgb[..., :3], cv2.COLOR_RGB2GRAY)
        else:
            g = rgb.squeeze()
        if g.dtype != np.uint8:
            g = np.clip(g, 0, 255).astype(np.uint8)
        return g

    # ------------------------------------------------------------------
    def _commit(
        self,
        R: np.ndarray,
        t: np.ndarray,
        depth: np.ndarray,
        cov: Optional[np.ndarray],
        rgb: Optional[np.ndarray] = None,
    ) -> None:
        """Anchor + Voxel-Set + Graustufen-Cache des akzeptierten KFs speichern."""
        self._last_R = R.copy()
        self._last_t = t.copy()
        self._last_depth = depth.astype(np.float32).copy()
        self._last_cov = cov.astype(np.float32).copy() if cov is not None else None
        # Voxel-Set des neuen Anchors (nutzt eigenes Subsample-Grid)
        world_pts, _ = self._backproject_to_world(self._uv_voxel, depth, t, R)
        self._last_voxels = _voxel_set_from_world_points(
            world_pts, self.cfg.voxel_size)
        # Graustufen für ORB-Matching im nächsten Stage-3
        if rgb is not None and _HAS_CV2:
            self._last_gray = self._to_gray(rgb).copy()
        else:
            self._last_gray = None


# =============================================================================
# Smoke test
# =============================================================================

if __name__ == "__main__":
    H, W = 240, 320
    fx = fy = 0.5 * W / np.tan(np.deg2rad(70.0) / 2)
    K = np.array([[fx, 0, W / 2], [0, fy, H / 2], [0, 0, 1]], np.float32)

    cfg = AimSlamConfig(
        voxel_size=0.08,
        overlap_thresh=0.40,         # niedrig genug, damit Bewegung accept triggert
        min_overlap_ratio=0.02,
        gain_thresh_per_ray=2.0,     # synthetic σ_d=0.1 => per-ray ~ 4 nats
        n_rays_score=256,
        pixel_sigma=1.0,
        prior_sigma_d=0.10,
        # Stage 3 ist mit der neuen Hybrid-Residual-Form bei perfekter synthetic-
        # Tiefe ≈ 0 (Pose & Tiefe perfekt konsistent). Trotzdem aktiviert lassen,
        # damit der Smoketest die Berechnung mit ausführt; aber chi_thresh tief
        # genug, damit Bewegungen >> Rauschen triggern.
        use_chi_square=True,
        chi_thresh=0.01,
        min_depth=0.2,
        max_depth=10.0,
    )
    sel = AimSlamSelector(cfg, K, (H, W))

    # Box-Room-Szene wie in frame_selector.py
    pix_h = None
    def fake_depth(t: np.ndarray, R: np.ndarray) -> np.ndarray:
        global pix_h
        if pix_h is None:
            vs, us = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
            pix_h = np.stack(
                [us.ravel(), vs.ravel(), np.ones_like(us.ravel())], axis=1
            ).astype(np.float32)
        K_inv = np.linalg.inv(K).astype(np.float32)
        rays_cam_un = pix_h @ K_inv.T
        rays_cam = rays_cam_un / np.linalg.norm(rays_cam_un, axis=1, keepdims=True)
        rays_world = rays_cam @ R.T
        ts = []
        for axis, lim in [(0, 3.0), (0, -3.0), (1, 3.0), (1, -3.0),
                          (2, 3.0), (2, -3.0)]:
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

    def yaw_R(deg: float) -> np.ndarray:
        a = np.deg2rad(deg)
        c, s = np.cos(a), np.sin(a)
        # standard yaw around y-axis (c2w)
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)

    def report(i: int, ok: bool, sc: AimSlamScore, tag: str) -> None:
        print(
            f"f{i:02d} {tag:>12s}  "
            f"O={sc.overlap:.3f} ({sc.n_voxels_shared:4d}/{sc.n_voxels_curr:4d})  "
            f"Γ/N={sc.info_gain_per_ray:5.2f}  "
            f"κ={sc.chi_square:6.2f}  N={sc.n_rays_used:3d}  "
            f"{'ACCEPT' if ok else 'skip  '}"
            f"{' (forced)' if sc.forced else ''}"
        )

    print("=== AIM-SLAM smoketest: box-room, gradual translation + yaw ===")
    n_acc = 0
    I = np.eye(3, dtype=np.float32)

    # Frame 0: Seed
    t0 = np.zeros(3, dtype=np.float32)
    d0 = fake_depth(t0, I)
    ok, sc = sel.should_accept(d0, t0, I)
    n_acc += int(ok)
    report(0, ok, sc, "seed")

    # Frames 1-4: kleine Translation entlang +x → hoher Overlap
    for i in range(1, 5):
        t = np.array([0.05 * i, 0.0, 0.0], dtype=np.float32)
        ok, sc = sel.should_accept(fake_depth(t, I), t, I)
        n_acc += int(ok)
        report(i, ok, sc, "near-static")

    # Frames 5-9: groessere Translation → Overlap sinkt, Γ steigt
    for i in range(5, 10):
        t = np.array([0.4 * (i - 4), 0.0, 0.0], dtype=np.float32)
        ok, sc = sel.should_accept(fake_depth(t, I), t, I)
        n_acc += int(ok)
        report(i, ok, sc, "translating")

    # Frames 10-14: kombinierte Yaw-Rotation
    for i in range(10, 15):
        t = np.array([2.0, 0.0, 0.0], dtype=np.float32)
        R = yaw_R(8.0 * (i - 9))
        ok, sc = sel.should_accept(fake_depth(t, R), t, R)
        n_acc += int(ok)
        report(i, ok, sc, "yaw+trans")

    print(f"\nTotal accepted: {n_acc}/15")
