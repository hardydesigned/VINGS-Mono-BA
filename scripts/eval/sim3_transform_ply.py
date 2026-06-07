#!/usr/bin/env python3
"""Transformiere eine in DROID-local rekonstruierte 2DGS-PLY per Sim3 ins
metrische GT-Welt-Frame (Option C: scharf rekonstruieren, dann metrisch
platzieren).

Sim3 (s, R, t) wird per Umeyama aus den Kamerazentren gefittet: DROID-c2w
(tracker_raw_c2w.txt) -> GT-w2c (poses_w2c.txt), gematcht ueber Zeitstempel.
Dann auf die PLY angewendet:
  xyz'      = s * (R @ xyz) + t
  scale_i'  = scale_i + log(s)      (log-space, gleichmaessige Groessenskalierung)
  rot'      = R ∘ rot               (Surfel-Orientierung mitdrehen)
Farbe (f_dc) und Opacity bleiben unveraendert.

Usage:
  python scripts/eval/sim3_transform_ply.py IN.ply --droid-poses tracker_raw_c2w.txt \
     --gt-poses .../poses_w2c.txt --out OUT.ply
"""
import argparse, numpy as np
from plyfile import PlyData, PlyElement
from scipy.spatial.transform import Rotation as Rot


def umeyama(src, dst):
    """Sim3: s,R,t mapping src->dst (Nx3)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    s0, d0 = src - mu_s, dst - mu_d
    cov = (d0.T @ s0) / src.shape[0]
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    var = (s0 ** 2).sum() / src.shape[0]
    s = np.trace(np.diag(D) @ S) / var
    t = mu_d - s * R @ mu_s
    return s, R, t


def umeyama_ransac(src, dst, thresh=1.0, iters=2000, fixed_scale=None, seed_idx=0):
    """Robustes Sim3 per RANSAC -- wirft DROID-Pose-Ausreisser raus.
    fixed_scale: wenn gesetzt, Skala erzwingen (nur R,t fitten)."""
    n = src.shape[0]
    if n < 4:
        s, R, t = umeyama(src, dst)
        return s, R, t, np.ones(n, bool)
    rng = np.random.default_rng(seed_idx)
    best_in, best = None, None
    for _ in range(iters):
        sel = rng.choice(n, 3, replace=False)
        try:
            s, R, t = umeyama(src[sel], dst[sel])
        except Exception:
            continue
        if fixed_scale is not None:
            s = fixed_scale
            t = dst[sel].mean(0) - s * R @ src[sel].mean(0)
        err = np.linalg.norm((s * (R @ src.T).T + t) - dst, axis=1)
        inl = err < thresh
        if best_in is None or inl.sum() > best_in.sum():
            best_in, best = inl, (s, R, t)
    # Refit auf allen Inliern
    if best_in.sum() >= 3:
        s, R, t = umeyama(src[best_in], dst[best_in])
        if fixed_scale is not None:
            s = fixed_scale
            t = dst[best_in].mean(0) - s * R @ src[best_in].mean(0)
        return s, R, t, best_in
    return best[0], best[1], best[2], best_in


def load_droid_c2w(path):
    rows = np.loadtxt(path, comments="#")
    if rows.ndim == 1:
        rows = rows[None]
    t = rows[:, 1]
    centers = rows[:, [5, 9, 13]]
    return t, centers


def load_gt_centers(path):
    rows = np.loadtxt(path, comments="#")
    t = rows[:, 0]
    tr = rows[:, 1:4]
    q = rows[:, 4:8]  # qx qy qz qw
    R = Rot.from_quat(q).as_matrix()           # w2c
    centers = -np.einsum('nij,nj->ni', R.transpose(0, 2, 1), tr)
    return t, centers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ply_in")
    ap.add_argument("--droid-poses", required=True)
    ap.add_argument("--gt-poses", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    t_d, c_d = load_droid_c2w(a.droid_poses)
    t_g, c_g = load_gt_centers(a.gt_poses)
    # match each droid KF to nearest GT time
    idx = np.searchsorted(t_g, t_d)
    idx = np.clip(idx, 0, len(t_g) - 1)
    c_g_m = c_g[idx]

    s, R, t = umeyama(c_d, c_g_m)
    aligned = (s * (R @ c_d.T).T + t)
    err = np.linalg.norm(aligned - c_g_m, axis=1)
    print(f"[sim3] n_match={len(t_d)} scale={s:.4f} "
          f"align-RMSE={np.sqrt((err**2).mean()):.3f} m  mean={err.mean():.3f}  max={err.max():.3f}")

    ply = PlyData.read(a.ply_in)
    v = ply["vertex"].data
    out = v.copy()
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float64)
    xyz2 = (s * (R @ xyz.T).T + t)
    out["x"], out["y"], out["z"] = xyz2[:, 0], xyz2[:, 1], xyz2[:, 2]

    # scales: log-space, uniform size scale by s
    for sc in [p for p in v.dtype.names if p.startswith("scale_")]:
        out[sc] = (v[sc].astype(np.float64) + np.log(s)).astype(v[sc].dtype)

    # rotations: compose world R onto each gaussian quaternion (PLY [w,x,y,z])
    rn = sorted([p for p in v.dtype.names if p.startswith("rot")])
    if len(rn) == 4:
        q_wxyz = np.stack([v[rn[0]], v[rn[1]], v[rn[2]], v[rn[3]]], axis=1).astype(np.float64)
        n = np.linalg.norm(q_wxyz, axis=1, keepdims=True); n[n == 0] = 1
        q_wxyz /= n
        q_xyzw = q_wxyz[:, [1, 2, 3, 0]]
        q_new = (Rot.from_matrix(R) * Rot.from_quat(q_xyzw)).as_quat()  # xyzw
        out[rn[0]] = q_new[:, 3]; out[rn[1]] = q_new[:, 0]
        out[rn[2]] = q_new[:, 1]; out[rn[3]] = q_new[:, 2]

    PlyData([PlyElement.describe(out, "vertex")]).write(a.out)
    print(f"[sim3] {out.shape[0]} Gaussians -> {a.out}")


if __name__ == "__main__":
    main()
