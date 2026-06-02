#!/usr/bin/env python3
"""Generate the s1000_400f sweep configs for amtown03 (frames 1000..1399).

Uses configs/local/amtown03/s1000_400f/amtown03_short_200_template.yaml as the
base. One YAML per variant — grouped into subfolders the run_sweep.sh
`--s1000` mode iterates over.

Groups (folder names per user request):
  baseline         vings_filter only (kind=none, mapper_kf_skip=1, motion-filter ON)
  mapskip          mapper_kf_skip in {1,2,3,5,10,20}, motion-filter ON
  skip_no_filter   frame_skip   in {1,2,3,5,10,20}, motion-filter OFF
  adaptive_kf      paper sens / decay sweep
  aim              kind=aim_slam — overlap & info-gain sweep
  coko             kind=coko_slam — alpha & submap sweep
  game             kind=game_kfs — accept_thresh sweep
  mm3dgs           covis-thresh sweep
  nurbs            kind=nurbs_lvi — sector-angle sweep
  orbslam          kind=orbslam3 — max_frames / ratio sweep
  two_gate         kind=two_gate — v6 params (b2only / theta sweep)
  two_gate_v2      kind=two_gate_v2 — Pre-Tracker A3 GPS-Motion + B1 ohne GPS
  vista            gain & trans sweep

Output goes to configs/local/amtown03/s1000_400f/<group>/amtown03_s1000_400f_<variant>.yaml
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import yaml


REPO = Path(__file__).resolve().parents[1]
TEMPLATE = REPO / "configs/local/amtown03/s1000_400f/amtown03_short_200_template.yaml"
TARGET_ROOT = REPO / "configs/local/amtown03/s1000_400f"
SAVE_DIR = str(REPO / "output/exp_amtown03_s1000_400f") + "/"

NAME_PREFIX = "amtown03_s1000_400f"
SKIP_VALUES = [1, 2, 3, 5, 10, 20]


# ── Selector defaults (paper-conformant, kept in sync with CLAUDE.md) ─────────
SELECTOR_DEFAULTS: dict[str, dict] = {
    "vista": {
        "kind": "vista",
        "voxel_size": 0.10,
        "gain_thresh": 0.30,
        "trans_thresh_m": 0.15,
        "rot_thresh_deg": 10.0,
        "n_rays_score": 256,
        "n_rays_integrate": 2048,
        "min_depth": 0.2,
        "max_depth": 35.0,
    },
    "nurbs_lvi": {
        "kind": "nurbs_lvi",
        "orb_n_features": 800,
        "sector_angle_deg": 2.0,
        "min_matches": 15,
        "force_accept_all": False,
        "min_depth": 0.2,
        "max_depth": 35.0,
    },
    "mm3dgs": {
        "kind": "mm3dgs",
        "covis_thresh": 0.95,
        "niqe_window": 5,
        "n_samples": 2048,
        "min_gap_after_kf": 5,
        "force_accept_after": 0,
        "min_depth": 0.2,
        "max_depth": 35.0,
    },
    "game_kfs": {
        "kind": "game_kfs",
        "beta_uncert": 0.3,
        "beta_render": 0.3,
        "beta_covis": 0.4,
        "alpha_assoc": 0.5,
        "alpha_flow": 0.3,
        "alpha_motion": 0.2,
        "gamma_assoc": 1.0,
        "gamma_render": 1.0,
        "eta": 0.8,
        "accept_thresh": 0.5,
        "orb_n_features": 800,
        "flow_ref_px": 30.0,
        "psnr_target": 25.0,
        "lap_var_ref": 500.0,
        "cov_ref": 1.0,
        "trans_ref_m": 0.30,
        "min_depth": 0.2,
        "max_depth": 35.0,
    },
    "adaptive_kf": {
        "kind": "adaptive_kf",
        "theta0": 0.05,
        "theta_init": 0.10,
        "window_size": 5,
        "sensitivity": 1.5,
        "decay": 0.95,
        "w_photo": 0.7,
        "w_ssim": 0.3,
        "min_overlap_pixels": 1000,
        "force_accept_all": False,
        "min_depth": 0.2,
        "max_depth": 35.0,
    },
    "orbslam3": {
        "kind": "orbslam3",
        "orb_n_features": 800,
        "tracked_ratio_thresh": 0.9,
        "min_tracked": 50,
        "min_frames": 1,
        "max_frames": 30,
        "force_accept_all": False,
        "min_depth": 0.2,
        "max_depth": 35.0,
    },
    "coko_slam": {
        "kind": "coko_slam",
        "alpha": 0.02,
        "distance_metric": "cosine",
        "submap_threshold": 0.05,
        "min_kfs_per_submap": 10,
        "max_kfs": 0,
        "memory_mode": "submap_reset",
        "feature_aggregation": "patch_mean_with_cls",
        "model_name": "dinov2_vits14",
        "image_size": 224,
        "device": "cuda",
        "force_accept_all": False,
    },
    "aim_slam": {
        "kind": "aim_slam",
        "voxel_size": 0.10,
        "overlap_thresh": 0.70,
        "min_overlap_ratio": 0.05,
        "n_voxel_samples": 1024,
        "gain_thresh_per_ray": 0.5,
        "n_rays_score": 256,
        "pixel_sigma": 1.0,
        "prior_sigma_d": 0.10,
        "use_chi_square": True,
        "chi_thresh": 1.0,
        "chi_orb_n_features": 800,
        "chi_min_matches": 20,
        "chi_max_disparity_px": 200.0,
        "force_accept_all": False,
        "min_depth": 0.2,
        "max_depth": 35.0,
    },
    # v6 best params from docs/TWO_GATE_RUNLOG.md (PSNR 17.76 on full amtown03).
    "two_gate": {
        "kind": "two_gate",
        "gps_d_min_m": 0.8,
        "gps_noise_floor_m": 0.4,
        "pose_d_min_m": 0.20,
        "visual_change_max_ssim": 0.985,
        "ssim_resize": 80,
        "covis_thresh": 0.85,
        "n_samples_covis": 2048,
        "min_depth": 0.2,
        "max_depth": 35.0,
        "enable_b3": True,
        "alpha": 0.30,
        "dino_model": "dinov2_vits14",
        "dino_image_size": 224,
        "dino_device": "cuda",
        "dino_max_kfs": 10,
        "theta0": 0.15,
        "theta_init": 0.20,
        "window_size": 30,
        "sensitivity": 0.5,
        "decay": 0.85,
        "min_spacing": 3,
        "max_per_window": 2,
        "rate_window": 30,
        "force_after": 100,
        # Per-Frame-Decision-Log: zeigt im Run-Log an welchem Pipeline-Schritt
        # (B1_*/B2B3_*/force/budget) ein Frame aussortiert wurde.
        "verbose": True,
    },
    # v2: GPS-Motion-Check aus B1 raus + neuer Pre-Tracker-A3-Sub-Gate.
    # Identische B2/B3/Theta/Budget wie two_gate; B1 ohne GPS-Pfad.
    # `_gate_a` wird von patch_selector als cfg['gate_a'] geschrieben.
    "two_gate_v2": {
        "kind": "two_gate_v2",
        # B1: Pose-Translation + SSIM-Veto (kein GPS)
        "pose_d_min_m": 0.20,
        "visual_change_max_ssim": 0.985,
        "ssim_resize": 80,
        "enable_ssim_veto": True,
        # B2: Covisibility
        "covis_thresh": 0.85,
        "n_samples_covis": 2048,
        "min_depth": 0.2,
        "max_depth": 35.0,
        # B3: DINO
        "enable_b3": True,
        "alpha": 0.30,
        "dino_model": "dinov2_vits14",
        "dino_image_size": 224,
        "dino_device": "cuda",
        "dino_max_kfs": 10,
        # Adaptive Theta
        "theta0": 0.15,
        "theta_init": 0.20,
        "window_size": 30,
        "sensitivity": 0.5,
        "decay": 0.85,
        # Budget
        "min_spacing": 3,
        "max_per_window": 2,
        "rate_window": 30,
        "force_after": 100,
        # Per-Frame-Decision-Log (siehe two_gate).
        "verbose": True,
        # Pre-Tracker Gate A v2 (A3 GPS-Motion neu). Geschrieben als
        # cfg['gate_a'] via patch_selector. A1 aus (start_frame=1000 ist
        # bereits Cruise), A2 mit relaxten Aerial-Thresholds.
        "_gate_a": {
            "enabled": True,
            "version": "v2",
            "enable_a3": True,
            "gps_d_min_m": 0.8,
            "enable_a1": False,
            "enable_a2": True,
            "blur_thresh": 50.0,
            "grad_density_thresh": 0.02,
        },
    },
}


# ── Per-selector parameter variants ───────────────────────────────────────────
# Designed for 200-frame amtown03 aerial slice:
# - default = paper-conformant baseline
# - one "loose" variant (more KFs → higher PSNR, slower)
# - one "strict" variant (fewer KFs → faster, lower PSNR)
SELECTOR_VARIANTS: dict[str, dict[str, dict]] = {
    "vista": {
        "g020":     {"gain_thresh": 0.20},                                # looser
        "g040":     {"gain_thresh": 0.40},                                # stricter
        "lowtrans": {"trans_thresh_m": 0.05, "rot_thresh_deg": 5.0},      # slow-flight
    },
    "nurbs_lvi": {
        "sec1":   {"sector_angle_deg": 1.0},   # finer migrations → more KFs
        "sec5":   {"sector_angle_deg": 5.0},   # coarser → fewer KFs
        "orb400": {"orb_n_features": 400},     # cheaper ORB
    },
    "mm3dgs": {
        "c85":     {"covis_thresh": 0.85},     # stricter (only on big drops)
        "c99":     {"covis_thresh": 0.99},     # very loose
        "gap10":   {"min_gap_after_kf": 10},   # paper kf_every spacing
    },
    "game_kfs": {
        "th030": {"accept_thresh": 0.30},      # looser
        "th070": {"accept_thresh": 0.70},      # stricter
        "eta05": {"eta": 0.5},                 # faster λ adaptation
    },
    "adaptive_kf": {
        "sens1": {"sensitivity": 1.0},         # looser → more KFs
        "sens3": {"sensitivity": 3.0},         # stricter
        "decay85": {"decay": 0.85},            # faster decay → faster re-acceptance
    },
    "orbslam3": {
        "max15":   {"max_frames": 15},         # force-rate every 15 frames
        "max60":   {"max_frames": 60},
        "ratio80": {"tracked_ratio_thresh": 0.8},  # stricter novelty
    },
    "coko_slam": {
        "a005":  {"alpha": 0.005},             # very strict
        "a050":  {"alpha": 0.050},             # looser
        "st010": {"submap_threshold": 0.10},   # rarer submap resets
    },
    "aim_slam": {
        "ovl055":  {"overlap_thresh": 0.55},   # accept more (skip only on big overlap)
        "ovl085":  {"overlap_thresh": 0.85},   # accept fewer
        "gain030": {"gain_thresh_per_ray": 0.30},  # weaker info-gain requirement
    },
    "two_gate": {
        "b2only": {"enable_b3": False},                                # DINO off
        "loose":  {"theta0": 0.10, "max_per_window": 4},               # more KFs
        "strict": {"theta0": 0.25, "max_per_window": 1},               # fewer KFs
    },
    # v2: gespiegelte Selector-Varianten + zusätzlich A3-Knöpfe (Pre-Tracker
    # GPS-Threshold + ein A3-off-Vergleich zur Isolation der Pre-Tracker-
    # Komponente). _gate_a-Overrides werden von patch_selector in
    # cfg['gate_a'] gemerged (nicht in cfg['frame_selector']).
    "two_gate_v2": {
        "b2only":    {"enable_b3": False},
        "loose":     {"theta0": 0.10, "max_per_window": 4},
        "strict":    {"theta0": 0.25, "max_per_window": 1},
        "a3_loose":  {"_gate_a": {"gps_d_min_m": 0.4}},                # mehr Frames durch
        "a3_strict": {"_gate_a": {"gps_d_min_m": 1.6}},                # weniger Frames durch
        "a3off":     {"_gate_a": {"enable_a3": False}},                # Isolation: B1 ohne GPS, A3 aus
        # B1 (Gate B, pre-mapper) gatet auf GPS-Distanz statt Tracker-Pose:
        "b1gps":         {"b1_motion_source": "gps", "gps_d_min_m": 0.8},
        "a3_loose_b1gps": {"b1_motion_source": "gps", "gps_d_min_m": 0.8,
                           "_gate_a": {"gps_d_min_m": 0.4}},
    },
}


# Folder-name short-form per user request
SELECTOR_FOLDER = {
    "vista":       "vista",
    "nurbs_lvi":   "nurbs",
    "mm3dgs":      "mm3dgs",
    "game_kfs":    "game",
    "adaptive_kf": "adaptive_kf",
    "orbslam3":    "orbslam",
    "coko_slam":   "coko",
    "aim_slam":    "aim",
    "two_gate":    "two_gate",
    "two_gate_v2": "two_gate_v2",
}


def load_template() -> dict:
    with open(TEMPLATE) as f:
        return yaml.full_load(f)


def patch_baseline(cfg: dict) -> dict:
    """VINGS-internal motion filter ON, no selector, every tracker-KF mapped."""
    c = copy.deepcopy(cfg)
    c["frame_selector"] = {"kind": "none"}
    c["mapper_kf_skip"] = 1
    c["frontend"]["filter_thresh"] = 2.4        # Stage 1 motion-filter ON
    c["frontend"]["keyframe_thresh"] = 0.0      # Stage 2 distance-gate untouched
    c.pop("frame_skip", None)
    return c


def patch_mapskip(cfg: dict, n: int) -> dict:
    c = copy.deepcopy(cfg)
    c["frame_selector"] = {"kind": "none"}
    c["mapper_kf_skip"] = n
    c["frontend"]["filter_thresh"] = 2.4
    c["frontend"]["keyframe_thresh"] = 0.0
    c.pop("frame_skip", None)
    return c


def patch_skip_no_filter(cfg: dict, n: int) -> dict:
    c = copy.deepcopy(cfg)
    c["frame_selector"] = {"kind": "none"}
    c["mapper_kf_skip"] = 1
    c["frame_skip"] = n
    c["frontend"]["filter_thresh"] = -1.0       # motion-filter OFF
    c["frontend"]["keyframe_thresh"] = 0.0
    return c


def patch_selector(cfg: dict, sel: str, override: dict | None = None) -> dict:
    c = copy.deepcopy(cfg)
    fs = copy.deepcopy(SELECTOR_DEFAULTS[sel])
    # `_gate_a` ist kein frame_selector-Feld -- es wird in cfg['gate_a']
    # geschrieben. Nur Selectors die einen Pre-Tracker-Filter brauchen
    # (z.B. two_gate_v2) setzen es.
    gate_a_cfg = fs.pop("_gate_a", None)
    if override:
        ov = copy.deepcopy(override)
        ga_override = ov.pop("_gate_a", None)
        fs.update(ov)
        if ga_override is not None:
            if gate_a_cfg is None:
                gate_a_cfg = {}
            gate_a_cfg.update(ga_override)
    c["frame_selector"] = fs
    if gate_a_cfg is not None:
        c["gate_a"] = gate_a_cfg
    c["mapper_kf_skip"] = 1
    c["frontend"]["filter_thresh"] = -1.0       # plugin selector replaces Stage 1
    c["frontend"]["keyframe_thresh"] = 0.0
    c.pop("frame_skip", None)
    return c


def write_yaml(path: Path, cfg: dict, header: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(header.rstrip() + "\n\n")
        yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)


def main() -> None:
    if not TEMPLATE.exists():
        print(f"[FEHLER] Template fehlt: {TEMPLATE}", file=sys.stderr)
        sys.exit(1)

    base = load_template()
    # Force the shared output dir for every variant; per-config save_dir
    # would scatter results otherwise.
    base["output"]["save_dir"] = SAVE_DIR
    # SelektionsUNABHAENGIGE Eval fuer JEDE Variante: Sim(3)-ATE + Held-out-
    # Novel-View-PSNR an fixen Frame-Positionen (jeder 10.) gegen GT. Macht den
    # Selektor-Vergleich fair (gleiches Eval-Set, Novel-View statt train-view).
    base["fair_eval"] = {
        "enabled": True,
        "eval_stride": 10,
        "gt_poses_file": "dji_poses_all_w2c.txt",
        "save_renders": True,
    }

    plan: list[tuple[str, str, dict]] = []

    # 1) baseline: VINGS internal motion-filter only
    plan.append(("baseline", "vings_filter", patch_baseline(base)))

    # 2) mapskip
    for n in SKIP_VALUES:
        plan.append(("mapskip", f"mapskip_{n}", patch_mapskip(base, n)))

    # 3) skip_no_filter
    for n in SKIP_VALUES:
        plan.append(("skip_no_filter", f"nofilter_skip_{n}",
                     patch_skip_no_filter(base, n)))

    # 4) selectors (default first, then variants)
    for sel, short in SELECTOR_FOLDER.items():
        plan.append((short, short, patch_selector(base, sel)))
        for suffix, override in SELECTOR_VARIANTS.get(sel, {}).items():
            plan.append((short, f"{short}_{suffix}",
                         patch_selector(base, sel, override)))

    print(f"\n=== amtown03 s1000_400f — {len(plan)} configs ===")
    for group, variant, cfg in plan:
        path = TARGET_ROOT / group / f"{NAME_PREFIX}_{variant}.yaml"
        header = (
            f"# Auto-generated by scripts/gen_s1000_400f_configs.py — DO NOT EDIT.\n"
            f"# dataset = amtown03 (frames 1000..1399)\n"
            f"# group   = {group}\n"
            f"# variant = {variant}\n"
        )
        write_yaml(path, cfg, header)
        print(f"  {path.relative_to(REPO)}")


if __name__ == "__main__":
    main()
