#!/usr/bin/env python3
"""Overlap-Höhenfeld-Entkippung für GPS-verankerte Segmente (Naht-Fix in 3D).

Hintergrund (2026-06-04): Mehrere unabhängige DROID-Läufe -> per-Segment GPS-Unwarp
verankert die KAMERAS perfekt (UTM, inkl. Höhe), aber die monokulare TIEFEN-Skala je
Segment löst leicht anders auf -> der Boden landet relativ VERKIPPT (Höhen-Naht ~21 m,
corr(dz,Flugrichtung)≈-0.85 = planar). Fix: im Overlap sehen zwei Segmente denselben
Boden; eine Ebene durch die Höhendifferenz dz(x,y) entkippt das spätere Segment ans
frühere. Nutzt die GPS-verankerte Absoluthöhe als Referenz. Reduziert die Naht 21->~4 m.

Sequentiell: Segment 0 = Anker (GPS-absolut). Jedes folgende wird ans schon platzierte
Höhenfeld der Vorgänger entkippt (z += Ebene(x,y) aus dem Overlap). Konkateniert -> Survey.

Usage:
  python scripts/eval/detilt_chain.py seg0_gps.ply seg1_gps.ply ... \
     --out survey.ply [--cell 6] [--ground-pct 20] [--min-overlap 80]
Reihenfolge = Flugreihenfolge (sortiere die Eingaben entsprechend).
"""
import argparse, numpy as np
from plyfile import PlyData, PlyElement


def ground_grid(xy, z, cell, pct, min_pts=3):
    k = np.floor(xy / cell).astype(np.int64)
    d = {}
    for key, zz in zip(map(tuple, k), z):
        d.setdefault(key, []).append(zz)
    return {c: np.percentile(v, pct) for c, v in d.items() if len(v) >= min_pts}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("plys", nargs="+", help="GPS-unwarpte Segment-PLYs in Flugreihenfolge")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cell", type=float, default=6.0, help="Höhenfeld-Zellgröße (m)")
    ap.add_argument("--ground-pct", type=float, default=20.0, help="Perzentil je Zelle = Boden")
    ap.add_argument("--min-overlap", type=int, default=80, help="min. gemeinsame Zellen für Entkippung")
    a = ap.parse_args()

    placed_xy, placed_g = [], {}                 # akkumuliertes Boden-Höhenfeld der platzierten Segmente
    parts = []
    dtype = None
    for i, p in enumerate(a.plys):
        v = PlyData.read(p)["vertex"].data
        xyz = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
        fin = np.isfinite(xyz).all(1)
        g = ground_grid(xyz[fin, :2], xyz[fin, 2], a.cell, a.ground_pct)
        if i == 0:
            corr = "ANKER"
        else:
            common = sorted(set(g) & set(placed_g))
            if len(common) >= a.min_overlap:
                P = np.array([[c[0] * a.cell, c[1] * a.cell, 1.0] for c in common])
                dz = np.array([placed_g[c] - g[c] for c in common])   # auf dieses Segment zu addieren
                coef, *_ = np.linalg.lstsq(P, dz, rcond=None)
                res = dz - P @ coef; keep = np.abs(res) < 2 * res.std() + 1e-9
                coef, *_ = np.linalg.lstsq(P[keep], dz[keep], rcond=None)
                plane = xyz[:, 0] * coef[0] + xyz[:, 1] * coef[1] + coef[2]
                xyz[:, 2] += plane
                g = {c: gz + (c[0] * a.cell * coef[0] + c[1] * a.cell * coef[1] + coef[2]) for c, gz in g.items()}
                corr = f"detilt {len(common)}ov |dz|{np.abs(dz).mean():.1f}->{np.abs(dz - P @ coef).mean():.1f}m"
            else:
                corr = f"NO-OVERLAP ({len(common)}<{a.min_overlap}) -> GPS-absolut belassen"
        # zurückschreiben + Höhenfeld akkumulieren
        out = v.copy()
        out["x"], out["y"], out["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        parts.append(out); dtype = out.dtype
        for c, gz in g.items():
            placed_g[c] = gz if c not in placed_g else 0.5 * (placed_g[c] + gz)
        print(f"[detilt] seg{i} {p.split('/')[-1]}: {corr}  (+{len(out)} gaussians)")

    allv = np.concatenate(parts).astype(dtype)
    PlyData([PlyElement.describe(allv, "vertex")]).write(a.out)
    print(f"[detilt] {len(allv)} Gaussians -> {a.out}")


if __name__ == "__main__":
    main()
