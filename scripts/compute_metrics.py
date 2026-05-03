#!/root/anaconda3/envs/vings/bin/python
"""
Compute or collect PSNR, SSIM, LPIPS for the three benchmark datasets:
  - Bonn (rgbd_bonn_crowd)
  - Small City (hierarchical_small_city)
  - Urbanscene (urbanscene_polytech)

Methods covered:
  - dynagslam   : computed from color.jpg / color_gt.jpg image pairs
  - s3po        : read from psnr/after_opt/final_result.json
  - dronesplat  : read from metrics.json -> average_metrics
  - vpgs-slam   : read from rendering_metrics_global.json or psnr.csv
  - streamsplat : read from metrics.json
"""

import json
import os
import csv
import glob
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import torch
import lpips

RESULTS = Path("/root/results")
NA = float("nan")

# ─── helpers ──────────────────────────────────────────────────────────────────

_lpips_net = None

def get_lpips():
    global _lpips_net
    if _lpips_net is None:
        _lpips_net = lpips.LPIPS(net="alex", verbose=False)
    return _lpips_net


def img_to_tensor(img):
    """HxWxC uint8 -> 1xCxHxW float in [-1, 1]."""
    t = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
    return t


def compute_lpips(img1: np.ndarray, img2: np.ndarray) -> float:
    net = get_lpips()
    with torch.no_grad():
        val = net(img_to_tensor(img1), img_to_tensor(img2))
    return float(val)


def metrics_from_image_pairs(pairs) -> dict:
    """
    Given a list of (rendered, gt) image path pairs, compute mean PSNR/SSIM/LPIPS.
    """
    psnrs, ssims, lpipss = [], [], []
    for pred_path, gt_path in pairs:
        pred = np.array(Image.open(pred_path).convert("RGB"))
        gt   = np.array(Image.open(gt_path).convert("RGB"))
        if pred.shape != gt.shape:
            from PIL import Image as PILImage
            pred_pil = PILImage.fromarray(pred).resize((gt.shape[1], gt.shape[0]))
            pred = np.array(pred_pil)
        psnrs.append(peak_signal_noise_ratio(gt, pred, data_range=255))
        ssims.append(structural_similarity(gt, pred, channel_axis=2, data_range=255))
        lpipss.append(compute_lpips(pred, gt))
    if not psnrs:
        return {"psnr": NA, "ssim": NA, "lpips": NA}
    return {
        "psnr":  float(np.mean(psnrs)),
        "ssim":  float(np.mean(ssims)),
        "lpips": float(np.mean(lpipss)),
    }


def latest_subdir(parent: Path):
    """Return the lexicographically latest subdirectory (timestamp-named)."""
    dirs = sorted([d for d in parent.iterdir() if d.is_dir()])
    return dirs[-1] if dirs else None


# ─── per-method collectors ────────────────────────────────────────────────────

def collect_dynagslam() -> dict:
    base = RESULTS / "dynagslam"
    mapping = {
        "bonn":       base / "run_bonn_crowd" / "eval_render",
        "small_city": base / "run_small_city_hierarchical" / "eval_render",
        "urbanscene": base / "run_urbanscene3d_polytech" / "eval_render",
    }
    results = {}
    for dataset, frame_root in mapping.items():
        if not frame_root.exists():
            results[dataset] = {"psnr": NA, "ssim": NA, "lpips": NA}
            continue
        pairs = []
        for frame_dir in sorted(frame_root.glob("frame_*")):
            pred = frame_dir / "color.jpg"
            gt   = frame_dir / "color_gt.jpg"
            if pred.exists() and gt.exists():
                pairs.append((pred, gt))
        if not pairs:
            results[dataset] = {"psnr": NA, "ssim": NA, "lpips": NA}
        else:
            print(f"  dynagslam/{dataset}: computing over {len(pairs)} frames …")
            results[dataset] = metrics_from_image_pairs(pairs)
    return results


def collect_s3po() -> dict:
    base = RESULTS / "s3po" / "results"
    mapping = {
        "bonn":       base / "bonn_rgbd_bonn_crowd",
        "small_city": base / "hierarchical_small_city",
        "urbanscene": base / "urbanscene_polytech",
    }
    results = {}
    for dataset, parent in mapping.items():
        run = latest_subdir(parent) if parent.exists() else None
        jpath = run / "psnr" / "after_opt" / "final_result.json" if run else None
        if jpath and jpath.exists():
            d = json.loads(jpath.read_text())
            results[dataset] = {
                "psnr":  d.get("mean_psnr", NA),
                "ssim":  d.get("mean_ssim", NA),
                "lpips": d.get("mean_lpips", NA),
            }
        else:
            results[dataset] = {"psnr": NA, "ssim": NA, "lpips": NA}
    return results


def collect_dronesplat() -> dict:
    base = RESULTS / "dronesplat"
    mapping = {
        "bonn":       None,                        # no bonn run
        "small_city": base / "small_city",
        "urbanscene": base / "urbanscene",
    }
    results = {}
    for dataset, path in mapping.items():
        if path is None or not path.exists():
            results[dataset] = {"psnr": NA, "ssim": NA, "lpips": NA}
            continue
        jpath = path / "metrics.json"
        if jpath.exists():
            entries = json.loads(jpath.read_text())
            avg = next((e["average_metrics"] for e in entries if "average_metrics" in e), None)
            if avg and avg.get("psnr", 0) != 0:
                results[dataset] = {
                    "psnr":  avg.get("psnr", NA),
                    "ssim":  avg.get("ssim", NA),
                    "lpips": avg.get("lpips", NA),
                }
                continue
        results[dataset] = {"psnr": NA, "ssim": NA, "lpips": NA}
    return results


def collect_vpgs_slam() -> dict:
    base = RESULTS / "vpgs-slam"

    def from_global_json(path: Path) -> dict:
        if path.exists():
            d = json.loads(path.read_text())
            return {"psnr": d.get("psnr", NA), "ssim": d.get("ssim", NA), "lpips": d.get("lpips", NA)}
        return {"psnr": NA, "ssim": NA, "lpips": NA}

    def from_psnr_csv(path: Path) -> dict:
        if not path.exists():
            return {"psnr": NA, "ssim": NA, "lpips": NA}
        rows = list(csv.DictReader(path.open()))
        psnrs = [float(r["psnr"]) for r in rows if r.get("psnr")]
        ssims = [float(r["ssim"]) for r in rows if r.get("ssim")]
        return {
            "psnr":  float(np.mean(psnrs)) if psnrs else NA,
            "ssim":  float(np.mean(ssims)) if ssims else NA,
            "lpips": NA,
        }

    return {
        "bonn":       from_global_json(base / "bonn" / "rgbd_bonn_crowd_01" / "rendering_metrics_global.json"),
        "small_city": from_psnr_csv(base / "small_city" / "psnr.csv"),
        "urbanscene": from_psnr_csv(base / "urbanscene" / "polytech" / "psnr.csv"),
    }


def collect_vings() -> dict:
    """
    Extract GT and predicted RGB from the rgbdnua grid image.
    Layout: 2 rows × 4 cols. Top-left = gt_rgb, bottom-left = pred_rgb.
    Each run saves one file per keyframe; we use all available frames.
    """
    base = RESULTS / "vings"
    mapping = {
        "bonn":       "bonn_crowd",
        "small_city": "hierarchical_smallcity",
        "urbanscene": "urbanscene_polytech",
    }
    results = {}
    for dataset, name_fragment in mapping.items():
        run_dirs = sorted([d for d in base.iterdir() if d.is_dir() and name_fragment in d.name])
        if not run_dirs:
            results[dataset] = {"psnr": NA, "ssim": NA, "lpips": NA}
            continue
        run = run_dirs[-1]
        frames = sorted((run / "rgbdnua").glob("FrameId=*.png"))
        if not frames:
            results[dataset] = {"psnr": NA, "ssim": NA, "lpips": NA}
            continue
        pairs = []
        for fpath in frames:
            arr = np.array(Image.open(fpath).convert("RGB"))
            H2, W4 = arr.shape[:2]
            H, W = H2 // 2, W4 // 4
            gt   = arr[0:H,   0:W]
            pred = arr[H:H2,  0:W]
            pairs.append((pred, gt))
        print(f"  vings/{dataset}: computing over {len(pairs)} frame(s) …")
        psnrs, ssims, lpipss = [], [], []
        for pred, gt in pairs:
            psnrs.append(peak_signal_noise_ratio(gt, pred, data_range=255))
            ssims.append(structural_similarity(gt, pred, channel_axis=2, data_range=255))
            lpipss.append(compute_lpips(pred, gt))
        results[dataset] = {
            "psnr":  float(np.mean(psnrs)),
            "ssim":  float(np.mean(ssims)),
            "lpips": float(np.mean(lpipss)),
            "n_frames": len(pairs),
        }
    return results


def collect_streamsplat() -> dict:
    base = RESULTS / "streamsplat"
    mapping = {
        "bonn":       base / "run_bonn_crowd",
        "small_city": base / "run_hier_smallcity",
        "urbanscene": base / "run_urbanscene_polytech",
    }
    results = {}
    for dataset, path in mapping.items():
        jpath = path / "metrics.json"
        if jpath.exists():
            d = json.loads(jpath.read_text())
            results[dataset] = {
                "psnr":  d.get("psnr_mean", NA),
                "ssim":  d.get("ssim_mean", NA),
                "lpips": d.get("lpips_mean") if d.get("lpips_mean") is not None else NA,
            }
        else:
            results[dataset] = {"psnr": NA, "ssim": NA, "lpips": NA}
    return results


# ─── main ─────────────────────────────────────────────────────────────────────

METHODS = {
    "vings":       collect_vings,
    "dynagslam":   collect_dynagslam,
    "s3po":        collect_s3po,
    "dronesplat":  collect_dronesplat,
    "vpgs-slam":   collect_vpgs_slam,
    "streamsplat": collect_streamsplat,
}

DATASETS = ["bonn", "small_city", "urbanscene"]


def fmt(v) -> str:
    return f"{v:.4f}" if not (v != v) else "  N/A "   # nan check


def print_table(all_results: dict):
    col_w = 12
    metric_names = ["PSNR", "SSIM", "LPIPS"]

    for dataset in DATASETS:
        header = f"\n{'─'*60}\n  Dataset: {dataset.upper()}\n{'─'*60}"
        print(header)
        print(f"{'Method':<16} {'PSNR':>{col_w}} {'SSIM':>{col_w}} {'LPIPS':>{col_w}}")
        print(f"{'':─<16} {'':─>{col_w}} {'':─>{col_w}} {'':─>{col_w}}")
        for method, by_dataset in all_results.items():
            m = by_dataset.get(dataset, {})
            print(
                f"{method:<16} "
                f"{fmt(m.get('psnr', NA)):>{col_w}} "
                f"{fmt(m.get('ssim', NA)):>{col_w}} "
                f"{fmt(m.get('lpips', NA)):>{col_w}}"
            )


def save_json(all_results: dict, out: Path):
    clean = {}
    for method, by_dataset in all_results.items():
        clean[method] = {}
        for dataset, metrics in by_dataset.items():
            clean[method][dataset] = {
                k: (None if (v != v) else v) for k, v in metrics.items()
            }
    out.write_text(json.dumps(clean, indent=2))
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    all_results = {}
    for method, fn in METHODS.items():
        print(f"\n[{method}]")
        all_results[method] = fn()

    print_table(all_results)

    out_path = RESULTS / "metrics_summary.json"
    save_json(all_results, out_path)
