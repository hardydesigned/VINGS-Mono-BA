"""
Sweep profiles for gen_configs.py.

A profile is a no-arg function that returns a list of (suffix, patch) tuples:
  - `suffix`: short string appended to the base file name + output dir
  - `patch`: dict that will be deep-merged on top of the base config

Add a new selector by appending a function and registering it in PROFILES at
the bottom of this file. The CLI `python scripts/gen_configs.py --list` shows
the current registry.
"""

from __future__ import annotations

from typing import Callable

# Each entry: (suffix, patch). `patch` is deep-merged on the base config.
Profile = list[tuple[str, dict]]


# ---------------------------------------------------------------------------
# Profile 1: naive frame_skip subsampling, no selector
# ---------------------------------------------------------------------------
def skip_no_filter() -> Profile:
    out = []
    for n in range(1, 11):
        out.append((
            f"nofilter_skip{n}",
            {
                "frame_skip": n,
                "frame_selector": {"kind": "none"},
            },
        ))
    return out


# ---------------------------------------------------------------------------
# Profile 2: every Nth tracker-KF goes to the mapper, no selector
# ---------------------------------------------------------------------------
def mapskip() -> Profile:
    out = []
    for n in range(1, 11):
        out.append((
            f"mapskip{n}",
            {
                "mapper_kf_skip": n,
                "frame_selector": {"kind": "none"},
            },
        ))
    return out


# ---------------------------------------------------------------------------
# Profile 3: VISTA-style geometric-gain selector
# ---------------------------------------------------------------------------
def vista() -> Profile:
    # Three gain-thresholds, kept consistent with the smallcity sweep
    common = {
        "voxel_size": 0.10,
        "max_views_per_voxel": 16,
        "trans_thresh_m": 0.15,
        "rot_thresh_deg": 10.0,
        "n_rays_score": 256,
        "n_rays_integrate": 2048,
        "min_depth": 0.2,
        "max_depth": 35.0,
    }
    out = []
    for tag, g in [("g020", 0.20), ("g030", 0.30), ("g040", 0.40)]:
        out.append((
            f"frameselector_{tag}",
            {"frame_selector": {"kind": "vista", **common, "gain_thresh": g}},
        ))
    return out


# ---------------------------------------------------------------------------
# Profile 4: NURBS-LVI-style adaptive Q-score selector
# ---------------------------------------------------------------------------
def nurbs_lvi() -> Profile:
    # NURBS-LVI nach Wu et al. 2026, Sec III.A. Decision: accept iff Or+Oc > Q,
    # mit Q als adaptivem Schwellwert. Es gibt KEINEN globalen threshold-Knopf
    # mehr -- die Schwelle ist intrinsisch. Der Sweep variiert sector_angle_deg
    # (der Hauptknopf der die Sensitivitaet bestimmt) und liefert eine
    # diag-Variante die nur loggt.
    #
    # Paper-Default ist 15° (LiDAR-VIO, grosse Baselines). Bei VINGS-Mono ist
    # die Inter-Frame-Parallaxe nur 0.5-2°, also kleiner Sektor noetig.
    common = {
        "orb_n_features": 800,
        "chamfer_lambda": 0.5,
        "min_matches": 15,
        "min_depth": 0.2,
        "max_depth": 35.0,
    }
    out = []
    # diag: force-accept-all + Logging der Q-/migration-Werte.
    out.append((
        "nurbs_diag",
        {"frame_selector": {"kind": "nurbs_lvi", **common,
                            "sector_angle_deg": 2.0,
                            "force_accept_all": True}},
    ))
    # sector_angle_deg-Sweep: kleiner = sensibler = mehr KFs.
    for tag, sec in [("sec0_2", 0.2), ("sec0_5", 0.5),
                     ("sec1", 1.0), ("sec2", 2.0), ("sec5", 5.0)]:
        out.append((
            f"nurbs_{tag}",
            {"frame_selector": {"kind": "nurbs_lvi", **common,
                                "sector_angle_deg": sec}},
        ))
    return out


# ---------------------------------------------------------------------------
# Profile 5: Game-KFS — gewichtungs-Sweep um die Paper-Defaults
# ---------------------------------------------------------------------------
def game_kfs_w_sweep() -> Profile:
    """1D-Sweep um die Default-Konfiguration (alpha, beta, eta, accept_thresh).
    Volles Kreuzprodukt waere 81 Configs; statt dessen kippen wir je Achse den
    Default + zwei Auspraegungen. 9 Variants insgesamt (1 default + 8 perturb).
    """
    common = {
        "kind": "game_kfs",
        # Paper defaults (Eq. 3 / Eq. 9)
        "beta_uncert": 0.3, "beta_render": 0.3, "beta_covis": 0.4,
        "alpha_assoc": 0.5, "alpha_flow": 0.3, "alpha_motion": 0.2,
        "gamma_assoc": 1.0, "gamma_render": 1.0,
        "eta": 0.8, "lambda_init": 0.5,
        "accept_thresh": 0.5,
        "orb_n_features": 800, "ransac_reproj_thresh": 4.0,
        "min_matches": 12, "flow_ref_px": 30.0,
        "n_samples": 2048, "lap_var_ref": 500.0, "cov_ref": 1.0,
        "trans_ref_m": 0.30, "omega_rot": 0.10,
        "min_depth": 0.2, "max_depth": 35.0,
    }

    def with_override(**ov) -> dict:
        return {"frame_selector": {**common, **ov}}

    out: Profile = [
        ("gkfs_default",      with_override()),
        # alpha-Varianten (Paper sensitivity Table VIII): DRA-bias
        ("gkfs_alpha_assoc",  with_override(alpha_assoc=0.8, alpha_flow=0.1, alpha_motion=0.1)),
        ("gkfs_alpha_flow",   with_override(alpha_assoc=0.2, alpha_flow=0.5, alpha_motion=0.3)),
        # beta-Varianten: FRA-bias
        ("gkfs_beta_covis",   with_override(beta_uncert=0.1, beta_render=0.1, beta_covis=0.8)),
        ("gkfs_beta_balanced", with_override(beta_uncert=0.4, beta_render=0.4, beta_covis=0.2)),
        # eta-Varianten: EMA-smoothing
        ("gkfs_eta050",       with_override(eta=0.5)),
        ("gkfs_eta095",       with_override(eta=0.95)),
        # accept_thresh-Varianten: Entscheidungs-Schwelle
        ("gkfs_t040",         with_override(accept_thresh=0.40)),
        ("gkfs_t060",         with_override(accept_thresh=0.60)),
    ]
    return out


# ---------------------------------------------------------------------------
# Registry — extend here when adding a new keyframe-selection algorithm.
# ---------------------------------------------------------------------------
PROFILES: dict[str, Callable[[], Profile]] = {
    "skip_no_filter":  skip_no_filter,
    "mapskip":         mapskip,
    "vista":           vista,
    "nurbs_lvi":       nurbs_lvi,
    "game_kfs_w_sweep": game_kfs_w_sweep,
}
