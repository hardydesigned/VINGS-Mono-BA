"""
Quick PSNR/SSIM/LPIPS evaluator for a single VINGS-Mono run directory.

Usage:
    python scripts/eval_run_metrics.py /path/to/run_output_dir

Looks for rgbdnua/FrameId=*.png in the given dir, computes per-frame metrics,
prints mean PSNR/SSIM/LPIPS, and writes metrics.json alongside.

Mirrors the metric computation in run_experiment.py:compute_metrics so
results are comparable to sweep_results.csv.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True  # tolerate truncated PNGs (killed runs)
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_dir")
    p.add_argument("--no-lpips", action="store_true",
                   help="Skip LPIPS to avoid loading the alex model.")
    args = p.parse_args()

    run_dir = Path(args.run_dir).resolve()
    rgbdnua_dir = run_dir / "rgbdnua"
    if not rgbdnua_dir.exists():
        # Fallback: maybe the run_dir IS already rgbdnua
        if (run_dir / "FrameId=000000.png").exists():
            rgbdnua_dir = run_dir
        else:
            print(f"No rgbdnua dir at {rgbdnua_dir}", file=sys.stderr)
            sys.exit(1)

    frames = sorted(rgbdnua_dir.glob("FrameId=*.png"))
    if not frames:
        print(f"No FrameId=*.png in {rgbdnua_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(frames)} frames, computing metrics...")

    use_lpips = not args.no_lpips
    net = None
    if use_lpips:
        try:
            import lpips as lpips_lib
            import torch
            net = lpips_lib.LPIPS(net="alex", verbose=False)
        except Exception as e:
            print(f"LPIPS unavailable ({e}); skipping.")
            use_lpips = False

    psnrs, ssims, lpipss = [], [], []
    for fp in frames:
        arr = np.array(Image.open(fp).convert("RGB"))
        H2, W4 = arr.shape[:2]
        H, W = H2 // 2, W4 // 4
        gt   = arr[0:H,  0:W]
        pred = arr[H:H2, 0:W]
        psnrs.append(peak_signal_noise_ratio(gt, pred, data_range=255))
        ssims.append(structural_similarity(gt, pred, channel_axis=2, data_range=255))
        if use_lpips:
            import torch
            def to_t(img):
                return torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
            with torch.no_grad():
                lpipss.append(float(net(to_t(pred), to_t(gt))))

    result = {
        "psnr":     round(float(np.mean(psnrs)), 4),
        "ssim":     round(float(np.mean(ssims)), 4),
        "lpips":    round(float(np.mean(lpipss)), 4) if lpipss else None,
        "n_frames": len(frames),
        "run_dir":  str(run_dir),
    }
    print(f"PSNR  = {result['psnr']}")
    print(f"SSIM  = {result['ssim']}")
    print(f"LPIPS = {result['lpips']}")
    print(f"n     = {result['n_frames']}")

    out_path = run_dir / "metrics.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
