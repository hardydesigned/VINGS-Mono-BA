#!/usr/bin/env python3
"""Merge mehrerer 2DGS-PLYs (im SELBEN Weltframe, z.B. GT-Posen-Chunks).

Da alle Chunks via GT-Posen im selben metrischen Frame liegen, ist der Merge
ein simples Concat der Vertex-Arrays (optional pro Chunk spatial+opacity-clean).

Usage:
  python scripts/eval/merge_plys.py --out merged.ply --opacity-min 0.2 \
      --z-clip 120 in1.ply in2.ply ...
"""
import argparse, numpy as np
from plyfile import PlyData, PlyElement


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--opacity-min", type=float, default=0.0)
    ap.add_argument("--z-clip", type=float, default=0.0,
                    help="optional: |z - global_median_z| <= Wert (Floater-in-Tiefe raus)")
    args = ap.parse_args()

    parts, dtype = [], None
    for p in args.inputs:
        try:
            v = PlyData.read(p)["vertex"].data
        except Exception as e:
            print(f"[merge] skip {p}: {e}"); continue
        keep = np.ones(v.shape[0], dtype=bool)
        if "opacity" in v.dtype.names and args.opacity_min > 0:
            op = 1.0 / (1.0 + np.exp(-v["opacity"].astype(np.float64)))
            keep &= op > args.opacity_min
        parts.append(v[keep]); dtype = v.dtype
        print(f"[merge] {p}: {keep.sum()}/{v.shape[0]}")

    if not parts:
        print("[merge] nichts zu mergen"); return
    allv = np.concatenate(parts)

    if args.z_clip > 0:
        zmed = np.median(allv["z"])
        m = np.abs(allv["z"] - zmed) <= args.z_clip
        print(f"[merge] z-clip um {zmed:.1f}+-{args.z_clip}: {m.sum()}/{allv.shape[0]}")
        allv = allv[m]

    PlyData([PlyElement.describe(allv, "vertex")]).write(args.out)
    print(f"[merge] {allv.shape[0]} Gaussians -> {args.out}")


if __name__ == "__main__":
    main()
