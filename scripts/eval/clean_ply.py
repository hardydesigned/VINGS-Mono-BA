#!/usr/bin/env python3
"""Spatial + opacity cleanup for 2DGS/3DGS PLYs.

Floater-Gaussians aus noisy use_metric:false-Tiefe landen weit weg vom
Szenenkern (z.B. +-1300 m bei einem 80-m-AGL-Flug). Dieser Cleaner behaelt
nur Gaussians, die (a) opacity > thresh haben UND (b) innerhalb von
max_dist der naechsten Kamera-Position der Trajektorie liegen. Alle
PLY-Felder bleiben erhalten (valides 2DGS-PLY).

Usage:
  python scripts/eval/clean_ply.py <ply_in> --traj <tracker_raw_c2w.txt> \
      --out <ply_out> --max-dist 200 --opacity-min 0.3
"""
import argparse, numpy as np
from plyfile import PlyData, PlyElement


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ply_in")
    ap.add_argument("--traj", default=None, help="tracker_raw_c2w.txt fuer Spatial-Crop")
    ap.add_argument("--gt-poses", default=None, help="TUM w2c (t tx ty tz qx qy qz qw) fuer Spatial-Crop")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-dist", type=float, default=200.0, help="max Distanz zur Bahn (Welt-Einheiten)")
    ap.add_argument("--opacity-min", type=float, default=0.3)
    ap.add_argument("--max-scale", type=float, default=0.0,
                    help="optional: groesste Gaussian-Achse exp(scale).max <= Wert in m (0=aus) -- killt Floater-Riesen")
    ap.add_argument("--max-z-spread", type=float, default=0.0,
                    help="optional: |z - median_z| <= dieser Wert (0=aus)")
    args = ap.parse_args()

    ply = PlyData.read(args.ply_in)
    v = ply["vertex"].data
    n0 = v.shape[0]
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float64)

    keep = np.ones(n0, dtype=bool)

    # (0) Endlich-Filter: inf/nan-Koordinaten (extreme Floater) raus.
    keep &= np.isfinite(xyz).all(axis=1)
    print(f"[clean] nach finite-Koordinaten: {keep.sum()}/{n0}")

    if args.max_scale > 0:
        scn = sorted([p for p in v.dtype.names if p.startswith("scale_")])
        if scn:
            S = np.stack([v[c] for c in scn], axis=1).astype(np.float64)
            smax = np.exp(S).max(axis=1)
            keep &= np.isfinite(smax) & (smax <= args.max_scale)
            print(f"[clean] nach scale_max<={args.max_scale}m: {keep.sum()}/{n0}")

    if "opacity" in v.dtype.names and args.opacity_min > 0:
        op = 1.0 / (1.0 + np.exp(-v["opacity"].astype(np.float64)))
        keep &= op > args.opacity_min
        print(f"[clean] nach opacity>{args.opacity_min}: {keep.sum()}/{n0}")

    cams = None
    if args.gt_poses:
        rows = np.loadtxt(args.gt_poses, comments="#")
        if rows.ndim == 1:
            rows = rows[None]
        from scipy.spatial.transform import Rotation as _R
        Rw2c = _R.from_quat(rows[:, 4:8]).as_matrix()
        cams = -np.einsum('nij,nj->ni', Rw2c.transpose(0, 2, 1), rows[:, 1:4]).astype(np.float64)
    elif args.traj:
        rows = np.loadtxt(args.traj, comments="#")
        if rows.ndim == 1:
            rows = rows[None]
        cams = rows[:, [5, 9, 13]].astype(np.float64)  # tx ty tz
    if cams is not None:
        from scipy.spatial import cKDTree
        tree = cKDTree(cams)
        mind = np.full(n0, np.inf)
        fin = np.isfinite(xyz).all(axis=1)
        mind[fin], _ = tree.query(xyz[fin], k=1, workers=-1)   # nearest camera center per gaussian
        keep &= mind <= args.max_dist
        print(f"[clean] nach spatial-crop <= {args.max_dist}: {keep.sum()}/{n0}")

    if args.max_z_spread > 0:
        zmed = np.median(xyz[keep, 2]) if keep.any() else 0.0
        keep &= np.abs(xyz[:, 2] - zmed) <= args.max_z_spread
        print(f"[clean] nach z-spread <= {args.max_z_spread} (med {zmed:.1f}): {keep.sum()}/{n0}")

    kept = v[keep]
    el = PlyElement.describe(kept, "vertex")
    PlyData([el]).write(args.out)
    print(f"[clean] {kept.shape[0]}/{n0} Gaussians behalten ({100*kept.shape[0]/n0:.1f}%) -> {args.out}")


if __name__ == "__main__":
    main()
