#!/usr/bin/env python3
"""Trajektorie-verankertes Sim3-Unwarp: macht eine durchgehende (nahtlose, scharfe)
DROID-Map metrisch UND driftfrei, indem statt EINES globalen Sim3 ein oertlich
variierender Sim3 entlang der Trajektorie angewandt wird.

Hintergrund (2026-06-03): Ein durchgehender DROID-Lauf liefert eine nahtlose
scharfe Map in EINEM Frame (kein per-Chunk-Gauge, kein Naht-Problem). Aber DROID
driftet ueber lange Strecken (~24 m RMSE / 600 Frames). Ein einzelner starrer Sim3
kann diese NICHT-starre Drift nicht glattziehen. Loesung: pro DROID-Keyframe einen
lokalen Sim3 (DROID-Fenster -> GT-Fenster) fitten; jeden Gaussian per seiner
naechsten Kameras gewichtet deformieren. Die Drift ist glatt -> benachbarte
Gaussians bekommen aehnliche Korrekturen -> nahtlos, aber driftfrei.

Translation/Scale tragen die Drift (gut konditioniert aus lokalen Zentren);
Rotation wird gegen den GLOBALEN Sim3 regularisiert (lokale near-straight Fenster
sind roll-degeneriert), per Fenster nur leicht nachgezogen.

Usage:
  python scripts/eval/sim3_unwarp.py IN.ply --droid-poses RUNDIR/tracker_raw_c2w.txt \
     --gt-poses .../poses_w2c.txt --out OUT.ply [--window 80] [--knn 4] [--crop-radius 100] [--nadir-clear 5]
"""
import argparse, numpy as np
from plyfile import PlyData, PlyElement
from scipy.spatial.transform import Rotation as Rot, Slerp
from scipy.spatial import cKDTree


def umeyama(src, dst):
    mu_s, mu_d = src.mean(0), dst.mean(0)
    s0, d0 = src - mu_s, dst - mu_d
    U, D, Vt = np.linalg.svd((d0.T @ s0) / len(src))
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    s = np.trace(np.diag(D) @ S) / ((s0 ** 2).sum() / len(src))
    return s, R, mu_d - s * R @ mu_s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ply_in")
    ap.add_argument("--droid-poses", required=True)
    ap.add_argument("--gt-poses", help="TUM w2c Anker-Posen (z.B. GT). Alternativ --gps-csv.")
    ap.add_argument("--gps-csv", help="rtk_positions_raw.csv (easting,northing,alt) als Anker statt GT")
    ap.add_argument("--out", required=True)
    ap.add_argument("--window", type=int, default=80, help="Trajektorie-Fenster (Frames) fuer lokalen Sim3")
    ap.add_argument("--knn", type=int, default=4, help="Anzahl naechster Kameras pro Gaussian fuer Blend")
    ap.add_argument("--crop-radius", type=float, default=100.0)
    ap.add_argument("--nadir-clear", type=float, default=5.0)
    ap.add_argument("--rot-reg", type=float, default=0.5,
                    help="0=lokale Rotation voll, 1=nur globale Rotation (gegen Roll-Degeneration)")
    a = ap.parse_args()

    # --- Posen laden ---
    r = np.loadtxt(a.droid_poses, comments="#")
    td = r[:, 1]; Cd = r[:, [5, 9, 13]]
    if a.gps_csv:
        # GPS-Anker: easting/northing/alt mit FIXEM globalem Ursprung (erste rtk-Zeile)
        # -> alle Segmente landen im selben UTM-Frame -> direkt mergebar.
        gg = np.genfromtxt(a.gps_csv, delimiter=",", names=True)
        tgps = gg["headerstamp"]; o = np.argsort(tgps)
        tgps = tgps[o]; e = gg["easting"][o]; n = gg["northing"][o]; al = gg["alt"][o]
        e0, n0, a0 = e[0], n[0], al[0]
        Cgm = np.stack([np.interp(td, tgps, e) - e0,
                        np.interp(td, tgps, n) - n0,
                        np.interp(td, tgps, al) - a0], axis=1)
    else:
        g = np.loadtxt(a.gt_poses, comments="#"); tg = g[:, 0]
        Rg = Rot.from_quat(g[:, 4:8]).as_matrix().transpose(0, 2, 1)
        Cg = -np.einsum('nij,nj->ni', Rg, g[:, 1:4])
        idx = np.clip(np.searchsorted(tg, td), 0, len(tg) - 1)
        Cgm = Cg[idx]
    N = len(td)

    # --- globaler Sim3 (Rotations-Anker + Diagnose) ---
    sG, RG, tG = umeyama(Cd, Cgm)
    rmseG = np.sqrt((np.linalg.norm((sG * (RG @ Cd.T).T + tG) - Cgm, axis=1) ** 2).mean())

    # --- pro Keyframe lokaler Sim3: Rotation FIX = global (lokale near-straight
    #     Fenster sind roll-degeneriert), nur Scale+Translation lokal (gut
    #     konditioniert, tragen die positionsdominierte Drift). ---
    half = max(a.window // 2, 6)
    loc_s = np.empty(N); loc_t = np.empty((N, 3)); loc_R = np.repeat(RG[None], N, 0)
    p_all = (RG @ Cd.T).T                                   # global-rotierte DROID-Zentren
    for i in range(N):
        lo, hi = max(0, i - half), min(N, i + half + 1)
        p = p_all[lo:hi]; q = Cgm[lo:hi]
        mp, mq = p.mean(0), q.mean(0)
        dp = p - mp; dq = q - mq
        s = (dp * dq).sum() / (dp * dp).sum()
        loc_s[i] = s; loc_t[i] = mq - s * mp
    aligned = loc_s[:, None] * p_all + loc_t
    rmseL = np.sqrt((np.linalg.norm(aligned - Cgm, axis=1) ** 2).mean())
    print(f"[unwarp] N={N}  global-Sim3 RMSE={rmseG:.2f}m  ->  lokal(window={a.window}, R=global) RMSE={rmseL:.2f}m")

    # --- Gaussians laden + per kNN-Kameras gewichtet deformieren ---
    ply = PlyData.read(a.ply_in); v = ply["vertex"].data
    xyz = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
    fin = np.isfinite(xyz).all(1)
    tree = cKDTree(Cd)
    k = min(a.knn, N)
    dist = np.full((len(xyz), k), np.inf); nbr = np.zeros((len(xyz), k), dtype=int)
    dist[fin], nbr[fin] = tree.query(xyz[fin], k=k)
    w = 1.0 / (dist + 1e-6); w /= w.sum(1, keepdims=True)

    # gewichtete Scale + Translation; Rotation per groesstem Gewicht (Blend von R via Mittelung+Reortho)
    s_g = (w * loc_s[nbr]).sum(1)
    out_xyz = np.zeros_like(xyz)
    # Rotation pro Gaussian: gewichtetes Mittel der knn-R (chordal) -> reortho
    Rw = (w[:, :, None, None] * loc_R[nbr]).sum(1)          # (M,3,3)
    U, _, Vt = np.linalg.svd(Rw)
    det = np.linalg.det(U) * np.linalg.det(Vt)
    Uc = U.copy(); Uc[det < 0, :, -1] *= -1
    Rg_each = np.einsum('nij,njk->nik', Uc, Vt)             # naechste Rotationsmatrix
    t_g = (w[:, :, None] * loc_t[nbr]).sum(1)
    out_xyz = s_g[:, None] * np.einsum('nij,nj->ni', Rg_each, xyz) + t_g
    out_xyz[~fin] = xyz[~fin]

    # --- Clean: Footprint-Crop gegen GT-Kameras + Nadir-Filter ---
    cam_z = float(np.median(Cgm[:, 2]))
    tg2 = cKDTree(Cgm)
    d2 = np.full(len(out_xyz), np.inf)
    d2[fin], _ = tg2.query(out_xyz[fin])
    keep = fin & (d2 < a.crop_radius) & (out_xyz[:, 2] < cam_z - a.nadir_clear)

    out = v[keep].copy()
    out["x"], out["y"], out["z"] = out_xyz[keep, 0], out_xyz[keep, 1], out_xyz[keep, 2]
    # scales: log + log(local scale); rot: compose Rg_each
    for sc in [p for p in v.dtype.names if p.startswith("scale_")]:
        out[sc] = (out[sc].astype(np.float64) + np.log(s_g[keep])).astype(out[sc].dtype)
    rn = sorted([p for p in v.dtype.names if p.startswith("rot")])
    if len(rn) == 4:
        q = np.stack([out[rn[0]], out[rn[1]], out[rn[2]], out[rn[3]]], 1).astype(np.float64)
        nrm = np.linalg.norm(q, axis=1, keepdims=True); nrm[nrm == 0] = 1
        qx = (q / nrm)[:, [1, 2, 3, 0]]
        qn = (Rot.from_matrix(Rg_each[keep]) * Rot.from_quat(qx)).as_quat()
        out[rn[0]], out[rn[1]], out[rn[2]], out[rn[3]] = qn[:, 3], qn[:, 0], qn[:, 1], qn[:, 2]
    PlyData([PlyElement.describe(out, "vertex")]).write(a.out)
    print(f"[unwarp] {len(xyz)} -> {keep.sum()} Gaussians (crop+nadir) -> {a.out}")


if __name__ == "__main__":
    main()
