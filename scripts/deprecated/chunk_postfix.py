#!/usr/bin/env python3
"""Per-Chunk Naht-Fix fuer Option C: croppt jeden bereits ins Metrische
transformierten Chunk auf SEINEN eigenen Kamera-Footprint und bewertet die
Chunk-Qualitaet (Sim3-Scale + Align-RMSE + Orientierungs-Streuung).

Hintergrund (Diagnose 2026-06-03): Der globale clean_ply --max-dist crop kann
Chunk-Floater nicht entfernen, weil ueber die 5-km-Bahn IMMER irgendeine
GT-Kamera < max-dist entfernt ist. Floater (DROID-Tiefe-Ausreisser bis 20 km)
muessen pro Chunk gegen die ~150 EIGENEN Kameras gecroppt werden. Zusaetzlich
sind manche Chunks degeneriert (c3300: scale 115 vs 82, RMSE 3.4 m statt 1.7) —
die poisonen den Merge und werden per Quality-Gate aussortiert/markiert.

Quelle der Chunk-Kameras: tracker_raw_c2w.txt (Zeitstempel) -> GT poses_w2c.txt.

Usage:
  python scripts/eval/chunk_postfix.py IN_metric.ply \
     --droid-poses RUNDIR/tracker_raw_c2w.txt --gt-poses .../poses_w2c.txt \
     --out OUT.ply --crop-radius 90 --median-scale 82 --max-rmse 2.5
Exit-Code 2 = Chunk vom Quality-Gate verworfen (keine Ausgabe geschrieben).
"""
import argparse, sys, numpy as np
from plyfile import PlyData, PlyElement
from scipy.spatial.transform import Rotation as Rot
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


def rot_mean(Rs):
    U, _, Vt = np.linalg.svd(Rs.sum(0))
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    return U @ S @ Vt


def load_droid(path):
    r = np.loadtxt(path, comments="#")
    if r.ndim == 1:
        r = r[None]
    t = r[:, 1]
    Rc2w = r[:, [2, 3, 4, 6, 7, 8, 10, 11, 12]].reshape(-1, 3, 3)
    C = r[:, [5, 9, 13]]
    return t, Rc2w, C


def load_gt(path):
    g = np.loadtxt(path, comments="#")
    t = g[:, 0]
    Rc2w = Rot.from_quat(g[:, 4:8]).as_matrix().transpose(0, 2, 1)
    C = -np.einsum('nij,nj->ni', Rc2w, g[:, 1:4])
    return t, Rc2w, C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ply_in")
    ap.add_argument("--droid-poses", required=True)
    ap.add_argument("--gt-poses", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--crop-radius", type=float, default=90.0,
                    help="Gaussian darf max. so weit (m) von der naechsten EIGENEN Chunk-Kamera weg sein")
    ap.add_argument("--nadir-clear", type=float, default=None,
                    help="Nadir-Filter: Gaussians mit z > cam_z_med - nadir_clear verwerfen "
                         "(physikalisch: Drohne schaut nach unten, ueber ihr ist nichts). z=hoch=oben angenommen.")
    ap.add_argument("--agl-target", type=float, default=None,
                    help="Erwartete Flughoehe ueber Grund (m). Chunk verwerfen wenn |AGL-target|/target > agl-tol.")
    ap.add_argument("--agl-tol", type=float, default=0.5)
    ap.add_argument("--median-scale", type=float, default=None,
                    help="Erwartete Konsens-Scale; Chunk wird verworfen wenn |s-med|/med > scale-tol")
    ap.add_argument("--scale-tol", type=float, default=0.20)
    ap.add_argument("--max-rmse", type=float, default=None,
                    help="Chunk verwerfen wenn Sim3-Align-RMSE > max-rmse (m)")
    ap.add_argument("--max-ori-spread", type=float, default=None,
                    help="Chunk verwerfen wenn Orientierungs-Streuung > Wert (deg)")
    ap.add_argument("--transform", action="store_true",
                    help="Eingabe ist nicht-metrische DROID-PLY; Sim3 wird angewendet (ersetzt sim3_transform_ply.py)")
    a = ap.parse_args()

    td, Rd, Cd = load_droid(a.droid_poses)
    tg, Rg, Cg = load_gt(a.gt_poses)
    idx = np.clip(np.searchsorted(tg, td), 0, len(tg) - 1)
    Cgm, Rgm = Cg[idx], Rg[idx]

    s, R, t = umeyama(Cd, Cgm)
    err = np.linalg.norm((s * (R @ Cd.T).T + t) - Cgm, axis=1)
    rmse = float(np.sqrt((err ** 2).mean()))
    Rdev = np.einsum('nij,nkj->nik', Rgm, Rd)
    Rori = rot_mean(Rdev)
    ori = np.array([np.degrees(np.linalg.norm(Rot.from_matrix(Rori.T @ Rdev[i]).as_rotvec()))
                    for i in range(len(Rdev))])
    ori_spread = float(ori.std())

    cam_z = float(np.median(Cgm[:, 2]))

    ply = PlyData.read(a.ply_in)
    v = ply["vertex"].data
    raw = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float64)
    # --transform: Eingabe ist nicht-metrische DROID-PLY -> erst per Sim3 ins Metrische,
    # dann Crop/Nadir/AGL in metrischen Koordinaten messen.
    xyz = (s * (R @ raw.T).T + t) if a.transform else raw
    fin = np.isfinite(xyz).all(1)
    n0 = len(xyz)
    tree = cKDTree(Cgm)
    d2cam = np.full(n0, np.inf)
    d2cam[fin], _ = tree.query(xyz[fin])
    keep = fin & (d2cam < a.crop_radius)
    if a.nadir_clear is not None:
        keep &= xyz[:, 2] < (cam_z - a.nadir_clear)
    agl = cam_z - float(np.median(xyz[keep, 2])) if keep.sum() else float("nan")

    tag = (f"scale={s:.1f} rmse={rmse:.2f}m ori_spread={ori_spread:.2f}deg AGL={agl:.1f}m "
           f"crop{100*keep.sum()/n0:.0f}% n={len(td)}")
    # --- Quality-Gate ---
    bad = []
    if a.median_scale is not None and abs(s - a.median_scale) / a.median_scale > a.scale_tol:
        bad.append(f"scale {s:.1f} != ~{a.median_scale}")
    if a.max_rmse is not None and rmse > a.max_rmse:
        bad.append(f"rmse {rmse:.2f}>{a.max_rmse}")
    if a.max_ori_spread is not None and ori_spread > a.max_ori_spread:
        bad.append(f"ori_spread {ori_spread:.2f}>{a.max_ori_spread}")
    if a.agl_target is not None and (not np.isfinite(agl) or
                                     abs(agl - a.agl_target) / a.agl_target > a.agl_tol):
        bad.append(f"AGL {agl:.1f} != ~{a.agl_target}")
    if bad:
        print(f"[postfix] REJECT {a.ply_in}  {tag}  ({'; '.join(bad)})")
        sys.exit(2)

    out = v[keep].copy()
    out["x"], out["y"], out["z"] = xyz[keep, 0], xyz[keep, 1], xyz[keep, 2]
    if a.transform:
        # scales(log+log s) + rot(R∘q) mitziehen (xyz schon oben metrisch gesetzt)
        for sc in [p for p in v.dtype.names if p.startswith("scale_")]:
            out[sc] = (out[sc].astype(np.float64) + np.log(s)).astype(out[sc].dtype)
        rn = sorted([p for p in v.dtype.names if p.startswith("rot")])
        if len(rn) == 4:
            q = np.stack([out[rn[0]], out[rn[1]], out[rn[2]], out[rn[3]]], axis=1).astype(np.float64)
            nrm = np.linalg.norm(q, axis=1, keepdims=True); nrm[nrm == 0] = 1
            q_xyzw = (q / nrm)[:, [1, 2, 3, 0]]
            qn = (Rot.from_matrix(R) * Rot.from_quat(q_xyzw)).as_quat()
            out[rn[0]], out[rn[1]], out[rn[2]], out[rn[3]] = qn[:, 3], qn[:, 0], qn[:, 1], qn[:, 2]
    PlyData([PlyElement.describe(out, "vertex")]).write(a.out)
    print(f"[postfix] KEEP   {a.ply_in}  {tag}  {n0}->{keep.sum()} -> {a.out}")


if __name__ == "__main__":
    main()
