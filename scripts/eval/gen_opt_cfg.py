#!/usr/bin/env python3
"""Generiere eine interval1-DROID-Test-Config mit PSNR-Optimierungs-Knoepfen.
Nur Config-Aenderungen (kein Code). Basis = interval1_droid_full.yaml.

Usage:
  python gen_opt_cfg.py OUT --start 3300 --frames 200 --hw 432 512 \
     --iters 100 --numkf 8 --kfskip 8 --rgb 1.0 --depth 0.5 --ssim 0.0 \
     --savedir output/exp_opt/t1/
"""
import argparse, yaml, copy

BASE = "configs/local/interval1/interval1_droid_full.yaml"

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--start", type=int, required=True)
ap.add_argument("--frames", type=int, required=True)
ap.add_argument("--hw", nargs=2, type=int, default=[384, 456])
ap.add_argument("--iters", type=int, default=100)
ap.add_argument("--numkf", type=int, default=8)
ap.add_argument("--kfskip", type=int, default=8)
ap.add_argument("--rgb", type=float, default=1.0)
ap.add_argument("--depth", type=float, default=0.5)
ap.add_argument("--normal", type=float, default=0.1)
ap.add_argument("--ssim", type=float, default=0.0)
ap.add_argument("--accum", type=float, default=None)
ap.add_argument("--prune-op", type=float, default=None, help="storage_manager.cpu_prune_opacity_threshold")
ap.add_argument("--dist-thresh", type=float, default=None, help="storage_manager.distance_threshold")
ap.add_argument("--evalstride", type=int, default=None)
ap.add_argument("--no-ext", action="store_true", help="ext_poses entfernen -> pure DROID-local (nicht-metrisch, sharp)")
ap.add_argument("--seed", action="store_true", help="seed_video_with_ext_pose: true -> BA von GT-Posen aus (GT-konsistente Tiefen)")
ap.add_argument("--selector-from", default=None,
                help="frame_selector/gate_a/gate_b aus dieser Referenz-Config uebernehmen (z.B. Sweep-Gewinner)")
ap.add_argument("--savedir", required=True)
a = ap.parse_args()

c = yaml.full_load(open(BASE))
c["dataset"]["start_frame"] = a.start
c["dataset"]["max_frames"] = a.frames
c["frontend"]["image_size"] = list(a.hw)
c["mapper_kf_skip"] = a.kfskip
c["training_args"]["iters"] = a.iters
c["training_args"]["num_keyframe"] = a.numkf
lw = c["training_args"]["loss_weights"]
lw["rgb_loss"] = a.rgb
lw["depth_loss"] = a.depth
lw["normal_loss"] = a.normal
if "ssim_loss" in lw or a.ssim > 0:
    lw["ssim_loss"] = a.ssim
if a.accum is not None:
    c["adc_args"]["accum_thresh"] = a.accum
if a.prune_op is not None:
    c["storage_manager"]["cpu_prune_opacity_threshold"] = a.prune_op
if a.dist_thresh is not None:
    c["storage_manager"]["distance_threshold"] = a.dist_thresh
if a.evalstride is not None:
    c["fair_eval"]["eval_stride"] = a.evalstride
if a.no_ext and "ext_poses_file" in c["dataset"]:
    del c["dataset"]["ext_poses_file"]
if a.seed:
    c["seed_video_with_ext_pose"] = True
sel_kind = "none"
if a.selector_from:
    ref = yaml.full_load(open(a.selector_from))
    for k in ("frame_selector", "gate_a", "gate_b"):
        if k in ref:
            c[k] = copy.deepcopy(ref[k])
    sel_kind = c.get("frame_selector", {}).get("kind", "none")
c["output"]["save_dir"] = a.savedir
yaml.safe_dump(c, open(a.out, "w"), sort_keys=False)
print(f"[gen] {a.out}: hw={a.hw} iters={a.iters} numkf={a.numkf} kfskip={a.kfskip} "
      f"rgb={a.rgb} depth={a.depth} ssim={a.ssim} start={a.start} frames={a.frames} selector={sel_kind}")
