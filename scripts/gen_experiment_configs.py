#!/usr/bin/env python3
"""
Generate the full sweep of experiment configs for amtown03 (full) and AGZ
(frames 0-10000). One config per variant:

  baseline:    vings_filter_only (kind=none, mapper_kf_skip=1, motion-filter ON)
  mapskip:     mapper_kf_skip in {1,2,3,5,10,20,100}, motion-filter ON
  nofilter:    frame_skip in {1,2,3,5,10,20,100}, motion-filter OFF
  selectors:   vista, nurbs_lvi, mm3dgs, game_kfs, adaptive_kf, orbslam3,
               coko_slam, aim_slam (kind=<selector>, mapper_kf_skip=1)

Outputs live under configs/local/<dataset>/exp/<group>/<dataset>_<variant>.yaml
and are auto-overwritten on each run.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import yaml


REPO = Path(__file__).resolve().parents[1]
CKPT = REPO / "ckpts" / "droid.pth"

SKIP_VALUES = [1, 2, 3, 5, 10, 20, 100]

SELECTOR_DEFAULTS = {
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
        "chamfer_lambda": 0.5,
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
        "eta": 0.8,
        "accept_thresh": 0.5,
        "orb_n_features": 800,
        "flow_ref_px": 30.0,
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
        "window_size": 30,
        "sensitivity": 2.0,
        "decay": 0.85,
        "w_photo": 0.85,
        "w_ssim": 0.15,
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
        "alpha": 0.4,
        "model_name": "dinov2_vits14",
        "image_size": 224,
        "device": "cuda",
        "max_kfs": 10,
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
        "force_accept_all": False,
        "min_depth": 0.2,
        "max_depth": 35.0,
    },
}

SELECTOR_ORDER = list(SELECTOR_DEFAULTS.keys())


# Per-selector parameter variants. Each entry overrides the default dict.
# Naming convention: short suffix that hints at the changed knob.
#   - "loose"   = more KFs accepted (looser gate)
#   - "strict"  = fewer KFs accepted
#   - "lowtrans"/"slow" = motion-threshold lowered for AGZ-style slow flight
# Rationale per selector lives next to the dict.
SELECTOR_VARIANTS: dict[str, dict[str, dict]] = {
    # VISTA: accept iff voxel-gain >= gain_thresh.
    # gain↓ ⇒ more KFs; trans_thresh↓ helps slow flight (AGZ hover) pass pre-filter.
    "vista": {
        "g020":     {"gain_thresh": 0.20},
        "g040":     {"gain_thresh": 0.40},
        "lowtrans": {"trans_thresh_m": 0.05, "rot_thresh_deg": 5.0},
    },
    # NURBS-LVI: doc says paper default 15° is too coarse for VINGS; default 2°.
    # Try finer (1°) and coarser (5°). chamfer_lambda weights depth vs ORB matches.
    "nurbs_lvi": {
        "sec1":     {"sector_angle_deg": 1.0},
        "sec5":     {"sector_angle_deg": 5.0},
        "chamfer1": {"chamfer_lambda": 1.0},
    },
    # MM3DGS: accept iff covis < covis_thresh. ↑ ⇒ more KFs.
    "mm3dgs": {
        "c85": {"covis_thresh": 0.85},
        "c99": {"covis_thresh": 0.99},
    },
    # Game-KFS: accept iff composite ≥ accept_thresh. ↓ ⇒ more KFs.
    "game_kfs": {
        "th030": {"accept_thresh": 0.30},
        "th070": {"accept_thresh": 0.70},
    },
    # Adaptive-KF: θ = max(θ₀, μ + k·σ). sensitivity=k ↓ ⇒ θ closer to mean ⇒ more KFs.
    "adaptive_kf": {
        "sens1": {"sensitivity": 1.0},
        "sens3": {"sensitivity": 3.0},
        "decay070": {"decay": 0.70},
    },
    # ORB-SLAM3 force-rate: max_frames ↓ ⇒ accept more often (force-burst).
    "orbslam3": {
        "max10":  {"max_frames": 10},
        "max60":  {"max_frames": 60},
        "ratio80": {"tracked_ratio_thresh": 0.8},
    },
    # Coko-SLAM: accept iff min DINO L2-dist ≥ alpha. ↓ ⇒ more KFs.
    "coko_slam": {
        "a025": {"alpha": 0.25},
        "a055": {"alpha": 0.55},
        "win20": {"max_kfs": 20},
    },
    # AIM-SLAM: skip iff voxel-overlap > overlap_thresh. ↑ ⇒ accept more.
    # gain_thresh_per_ray ↓ also accepts more.
    "aim_slam": {
        "ovl055":  {"overlap_thresh": 0.55},
        "ovl085":  {"overlap_thresh": 0.85},
        "gain030": {"gain_thresh_per_ray": 0.30},
    },
}


def base_amtown03(out_root: Path) -> dict:
    """Base config for amtown03 FULL sequence (frames 0..6199)."""
    return {
        "adc_args": {"accum_thresh": 0.98},
        "dataset": {
            "image_dir": "images_all",
            "image_ext": "*.jpg",
            "start_frame": 0,
            "max_frames": 6199,
            "module": "datasets.generic_vo",
            "rgb_strip": 1,
            "root": "/home/philipp/Dokumente/datasets/amtown03/",
        },
        "debug_mode": False,
        "device": {"mapper": "cuda:0", "tracker": "cuda:0"},
        "frame_selector": {"kind": "none"},
        "mapper_kf_skip": 1,
        "ply_checkpoint_every_kf": 100,
        "frontend": {
            "active_window": 12,
            "beta": 0.3,
            "buffer": 80,
            "c2i": "None",
            "far_threshold": 0.02,
            "filter_thresh": 2.4,
            "frontend_nms": 1,
            "frontend_radius": 2,
            "frontend_thresh": 16.0,
            "frontend_window": 5,
            "image_size": [240, 288],
            "inac_range": 3,
            "keyframe_thresh": 0.0,
            "mask_threshold": -1.0,
            "max_factors": 48,
            "save_buffer_size": 6300,
            "upsample": False,
            "show_plot": False,
            "skip_edge": [-4, -5, -6],
            "translation_threshold": 0.5,
            "warm_up": 8,
            "weight": str(CKPT),
        },
        "intrinsic": {
            "H": 1024, "W": 1224, "bg": 0,
            "cu": 520.89, "cv": 586.09,
            "fu": 726.64, "fv": 726.86,
            "resolution_scale": 1.0,
        },
        "middleware": {"cov_times": 1000, "max_cov": 3e3, "max_depth": 200},
        "mode": "vo",
        "output": {"save_dir": str(out_root / "exp_amtown03_full") + "/"},
        "storage_manager": {
            "distance_threshold": 3.0,
            "cpu_prune_opacity_threshold": 0.10,
        },
        "training_args": {
            "iters": 50,
            "loss_weights": {
                "rgb_loss": 1.0, "depth_loss": 0.5, "normal_loss": 0.1,
                "alpha_loss": 0.1, "dist_loss": 0.0,
            },
            "lr": {
                "_opacity_lr": 0.05, "_rgb_lr": 0.0005,
                "_rotation_lr": 0.001, "_scaling_lr": 0.001, "_xyz_lr": 0.0001,
            },
            "num_keyframe": 8,
        },
        "use_dynamic": False,
        "use_metric": False,
        "use_pose_refine": False,
        "use_sky": False,
        "use_storage_manager": True,
        "use_vis": False,
        "use_wandb": False,
        "vis": {
            "bev_intrinsic_dict": {
                "H": 758, "W": 1372,
                "cu": 378.5, "cv": 685.5, "fu": 656.4, "fv": 656.4,
            },
            "bev_w2c": [
                [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]
            ],
            "pose_scale": 1.0,
        },
    }


def base_agz(out_root: Path) -> dict:
    """Base config for AGZ frames 0..10000.

    Assumes that scripts/prepare_agz.py has produced
    ~/Dokumente/datasets/agz/agz_0_10000/rectified/{images, intrinsic.txt}
    at half-resolution (1080x1920 -> 540x960). The runner script extracts
    these on first invocation if the dir is missing.
    """
    return {
        "adc_args": {"accum_thresh": 0.98},
        "dataset": {
            "image_dir": "images",
            "image_ext": "*.jpg",
            "module": "datasets.generic_vo",
            "rgb_strip": 1,
            "root": "/home/philipp/Dokumente/datasets/agz/agz_0_10000/rectified/",
            "start_frame": 0,
            "max_frames": 10000,
        },
        "debug_mode": False,
        "device": {"mapper": "cuda:0", "tracker": "cuda:0"},
        "frame_selector": {"kind": "none"},
        "mapper_kf_skip": 1,
        "ply_checkpoint_every_kf": 100,
        "frontend": {
            "active_window": 12,
            "beta": 0.3,
            "buffer": 80,
            "c2i": "None",
            "far_threshold": 0.02,
            "filter_thresh": 2.4,
            "frontend_nms": 1,
            "frontend_radius": 2,
            "frontend_thresh": 16.0,
            "frontend_window": 5,
            "image_size": [240, 432],
            "inac_range": 3,
            "keyframe_thresh": 0.0,
            "mask_threshold": -1.0,
            "max_factors": 48,
            "save_buffer_size": 10300,
            "show_plot": False,
            "skip_edge": [-4, -5, -6],
            "translation_threshold": 0.5,
            "upsample": False,
            "warm_up": 8,
            "weight": str(CKPT),
        },
        "intrinsic": {
            "H": 540, "W": 960, "bg": 0,
            "cu": 277.566750, "cv": 475.565502,
            "fu": 449.163243, "fv": 446.695054,
            "resolution_scale": 1.0,
        },
        "middleware": {"cov_times": 1000, "max_cov": 3e3, "max_depth": 200},
        "mode": "vo",
        "output": {"save_dir": str(out_root / "exp_agz_0_10000") + "/"},
        "storage_manager": {
            "cpu_prune_opacity_threshold": 0.10,
            "distance_threshold": 5.0,
        },
        "training_args": {
            "iters": 50,
            "loss_weights": {
                "alpha_loss": 0.1, "depth_loss": 0.5, "dist_loss": 0.0,
                "normal_loss": 0.1, "rgb_loss": 1.0,
            },
            "lr": {
                "_opacity_lr": 0.05, "_rgb_lr": 0.0005,
                "_rotation_lr": 0.001, "_scaling_lr": 0.001, "_xyz_lr": 0.0001,
            },
            "num_keyframe": 8,
        },
        "use_dynamic": False,
        "use_metric": False,
        "use_pose_refine": False,
        "use_sky": False,
        "use_storage_manager": True,
        "use_vis": False,
        "use_wandb": False,
        "vis": {
            "bev_intrinsic_dict": {
                "H": 758, "W": 1372,
                "cu": 378.5, "cv": 685.5, "fu": 656.4, "fv": 656.4,
            },
            "bev_w2c": [
                [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]
            ],
            "pose_scale": 1.0,
        },
    }


def write_yaml(path: Path, cfg: dict, header: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(header.rstrip() + "\n\n")
        yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)


def patch_baseline(cfg: dict) -> dict:
    c = copy.deepcopy(cfg)
    c["frame_selector"] = {"kind": "none"}
    c["mapper_kf_skip"] = 1
    c["frontend"]["filter_thresh"] = 2.4
    return c


def patch_mapskip(cfg: dict, n: int) -> dict:
    c = copy.deepcopy(cfg)
    c["frame_selector"] = {"kind": "none"}
    c["mapper_kf_skip"] = n
    c["frontend"]["filter_thresh"] = 2.4  # VINGS internal motion-filter ON
    return c


def patch_nofilter(cfg: dict, n: int) -> dict:
    c = copy.deepcopy(cfg)
    c["frame_selector"] = {"kind": "none"}
    c["mapper_kf_skip"] = 1
    c["frame_skip"] = n
    c["frontend"]["filter_thresh"] = -1.0       # VINGS internal motion-filter OFF
    c["frontend"]["keyframe_thresh"] = 0.0      # distance-gate already off; keep explicit
    return c


def patch_selector(cfg: dict, sel: str, override: dict | None = None) -> dict:
    c = copy.deepcopy(cfg)
    fs = copy.deepcopy(SELECTOR_DEFAULTS[sel])
    if override:
        fs.update(override)
    c["frame_selector"] = fs
    c["mapper_kf_skip"] = 1
    # VINGS native pre-filter OFF when a plugin selector runs -- otherwise
    # Stage 1 (motion-filter) already rejects frames before the plugin sees them
    # and the comparison between selector algorithms gets contaminated.
    c["frontend"]["filter_thresh"] = -1.0
    c["frontend"]["keyframe_thresh"] = 0.0
    return c


def _smokify(cfg: dict, dataset: str, n_frames: int, out_root: Path) -> dict:
    """Shrink a config to a ~n_frames smoke-test (same hyperparams, less data).

    Adjusts max_frames + save-buffers so the run finishes in a few minutes and
    produces enough rgbdnua frames for PSNR/SSIM/LPIPS metric extraction.
    """
    c = copy.deepcopy(cfg)
    c["dataset"]["max_frames"] = n_frames
    # save_buffer_size in frontend is the keyframe-buffer slot count; on smoke
    # it can shrink so RAM stays low.
    c["frontend"]["save_buffer_size"] = max(150, n_frames + 50)
    # output dir gets a _smoke suffix so smoke and full sweeps coexist.
    smoke_out_dir = {"amtown03": "exp_amtown03_smoke",
                     "agz":      "exp_agz_smoke"}[dataset]
    c["output"]["save_dir"] = str(out_root / smoke_out_dir) + "/"
    # PLY-checkpoint every 20 KFs so smoke runs still leave a partial ply.
    c["ply_checkpoint_every_kf"] = 20
    return c


def generate(dataset: str, base_fn, out_root: Path, target_root: Path,
             smoke_frames: int | None = None):
    base = base_fn(out_root)
    name_prefix = {"amtown03": "amtown03_full", "agz": "agz_10k"}[dataset]
    if smoke_frames is not None:
        name_prefix = {"amtown03": "amtown03_smoke", "agz": "agz_smoke"}[dataset]
    exp_subdir = "exp_smoke" if smoke_frames is not None else "exp"
    print(f"\n=== {dataset} [{'smoke' if smoke_frames is not None else 'full'}] ===")

    plan = []
    plan.append(("baseline", "vings_filter", patch_baseline(base)))
    for n in SKIP_VALUES:
        plan.append((f"mapskip", f"mapskip_{n}", patch_mapskip(base, n)))
    for n in SKIP_VALUES:
        plan.append((f"skip_no_filter", f"nofilter_skip_{n}", patch_nofilter(base, n)))
    for sel in SELECTOR_ORDER:
        plan.append((sel, sel, patch_selector(base, sel)))
        for suffix, override in SELECTOR_VARIANTS.get(sel, {}).items():
            plan.append((sel, f"{sel}_{suffix}", patch_selector(base, sel, override)))

    for group, variant, cfg in plan:
        if smoke_frames is not None:
            cfg = _smokify(cfg, dataset, smoke_frames, out_root)
        path = target_root / dataset / exp_subdir / group / f"{name_prefix}_{variant}.yaml"
        header = (
            f"# Auto-generated by scripts/gen_experiment_configs.py — DO NOT EDIT.\n"
            f"# dataset = {dataset}\n"
            f"# group   = {group}\n"
            f"# variant = {variant}\n"
            f"# mode    = {'smoke (' + str(smoke_frames) + ' frames)' if smoke_frames else 'full'}\n"
        )
        write_yaml(path, cfg, header)
        print(f"  {path.relative_to(REPO)}")


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="Generate the smoke variant (100 frames per run) into "
                         "configs/local/<ds>/exp_smoke/ instead of exp/.")
    ap.add_argument("--smoke-frames", type=int, default=100,
                    help="Frames per smoke run (default 100, only used with --smoke).")
    args = ap.parse_args()

    out_root = REPO / "output"
    target_root = REPO / "configs" / "local"
    smoke_n = args.smoke_frames if args.smoke else None
    generate("amtown03", base_amtown03, out_root, target_root, smoke_frames=smoke_n)
    generate("agz", base_agz, out_root, target_root, smoke_frames=smoke_n)


if __name__ == "__main__":
    sys.exit(main())
