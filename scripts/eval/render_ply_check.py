#!/usr/bin/env python3
"""Headless PLY + trajectory sanity renderer (no OpenGL).

Lädt eine 2DGS/3DGS-PLY (Gaussian-Centers + f_dc-Farbe) und die rohe
Tracker-Trajektorie und rendert mit matplotlib mehrere Projektionen:
  - Punktwolke BEV (top-down) und Seitenansicht (Höhe), RGB-gefärbt
  - geschätzte Kamerabahn vs. GT-DJI-Bahn (Sim(3)-aligned via Umeyama)
  - Höhenprofil est vs. GT über die Zeit

Zweck: schnell prüfen ob ein Drohnenflug vollständig & driftfrei abgedeckt
ist (Loop geschlossen? Höhe plausibel? Bahn deckt sich mit GT?).

Usage:
  python scripts/eval/render_ply_check.py <run_dir> [--ply PATH] [--out PNG]
"""
import os, sys, argparse, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from plyfile import PlyData

C0 = 0.28209479177387814  # SH band-0 constant


def load_ply(path, opacity_min=0.0, max_points=400_000):
    ply = PlyData.read(path)
    v = ply["vertex"].data
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float64)
    if all(k in v.dtype.names for k in ("f_dc_0", "f_dc_1", "f_dc_2")):
        f_dc = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=1)
        rgb = np.clip(f_dc * C0 + 0.5, 0, 1)
    else:
        rgb = np.full((xyz.shape[0], 3), 0.5)
    if "opacity" in v.dtype.names and opacity_min > 0:
        op = 1.0 / (1.0 + np.exp(-v["opacity"]))  # sigmoid of raw opacity
        keep = op > opacity_min
        xyz, rgb = xyz[keep], rgb[keep]
    if xyz.shape[0] > max_points:
        idx = np.random.default_rng(0).choice(xyz.shape[0], max_points, replace=False)
        xyz, rgb = xyz[idx], rgb[idx]
    return xyz, rgb


def load_traj_c2w(path):
    """tracker_raw_c2w.txt: kf_idx t r00 r01 r02 tx r10 r11 r12 ty r20 r21 r22 tz."""
    rows = np.loadtxt(path, comments="#")
    if rows.ndim == 1:
        rows = rows[None]
    t = rows[:, 1]
    centers = rows[:, [5, 9, 13]]  # tx, ty, tz
    return t, centers


def load_gt_w2c(path):
    """TUM w2c: t tx ty tz qx qy qz qw -> camera centers c = -R^T t."""
    rows = np.loadtxt(path, comments="#")
    t = rows[:, 0]
    tr = rows[:, 1:4]
    q = rows[:, 4:8]  # qx qy qz qw

    def quat_to_R(qx, qy, qz, qw):
        n = np.sqrt(qx*qx+qy*qy+qz*qz+qw*qw) + 1e-12
        qx, qy, qz, qw = qx/n, qy/n, qz/n, qw/n
        return np.array([
            [1-2*(qy*qy+qz*qz), 2*(qx*qy-qz*qw),   2*(qx*qz+qy*qw)],
            [2*(qx*qy+qz*qw),   1-2*(qx*qx+qz*qz), 2*(qy*qz-qx*qw)],
            [2*(qx*qz-qy*qw),   2*(qy*qz+qx*qw),   1-2*(qx*qx+qy*qy)],
        ])
    centers = np.zeros((rows.shape[0], 3))
    for i in range(rows.shape[0]):
        R = quat_to_R(*q[i])
        centers[i] = -R.T @ tr[i]
    return t, centers


def umeyama(src, dst):
    """Sim(3): find s,R,t mapping src->dst (Nx3)."""
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


def match_traj_to_gt(t_est, c_est, t_gt, c_gt):
    """Nearest-time match est KFs to GT, return aligned est + gt centers."""
    gt_idx = np.searchsorted(t_gt, t_est)
    gt_idx = np.clip(gt_idx, 0, len(t_gt) - 1)
    c_gt_m = c_gt[gt_idx]
    return c_est, c_gt_m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--ply", default=None)
    ap.add_argument("--gt", default="/home/philipp/Dokumente/datasets/amtown03/metadata/dji_poses_all_w2c.txt")
    ap.add_argument("--out", default=None)
    ap.add_argument("--opacity-min", type=float, default=0.0)
    args = ap.parse_args()

    rd = args.run_dir
    ply_path = args.ply
    if ply_path is None:
        plys = sorted(glob.glob(os.path.join(rd, "ply", "idx=*_2dgs.ply")),
                      key=lambda p: int(p.split("idx=")[1].split("_")[0]))
        if not plys:
            print("NO PLY found in", os.path.join(rd, "ply")); sys.exit(1)
        ply_path = plys[-1]
    out = args.out or os.path.join(rd, "ply_check.png")

    print(f"[render] PLY = {ply_path}")
    xyz, rgb = load_ply(ply_path, opacity_min=args.opacity_min)
    print(f"[render] {xyz.shape[0]} points  bbox min={xyz.min(0)} max={xyz.max(0)}")

    traj_path = os.path.join(rd, "tracker_raw_c2w.txt")
    have_traj = os.path.exists(traj_path)
    if have_traj:
        t_est, c_est = load_traj_c2w(traj_path)
        print(f"[render] {c_est.shape[0]} tracker KFs")

    have_gt = os.path.exists(args.gt)
    s = R = tt = None
    if have_traj and have_gt:
        t_gt, c_gt = load_gt_w2c(args.gt)
        ce, cg = match_traj_to_gt(t_est, c_est, t_gt, c_gt)
        try:
            s, R, tt = umeyama(ce, cg)
            c_est_al = (s * (R @ ce.T).T + tt)
            err = np.linalg.norm(c_est_al - cg, axis=1)
            print(f"[render] Sim3 scale={s:.4f}  ATE rmse={np.sqrt((err**2).mean()):.3f} m  "
                  f"mean={err.mean():.3f} m  max={err.max():.3f} m")
        except Exception as e:
            print("[render] umeyama failed:", e)
            have_gt = False

    # determine height axis: the one with smallest spread among trajectory = up-ish.
    # For aerial we just show XY (BEV) and XZ.
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))

    ax = axes[0, 0]
    ax.scatter(xyz[:, 0], xyz[:, 1], c=rgb, s=0.5, marker=".", linewidths=0)
    if have_traj:
        ax.plot(c_est[:, 0], c_est[:, 1], "-", color="red", lw=1.2, label="cam path")
    ax.set_title("PLY BEV (X-Y) + tracker path"); ax.set_aspect("equal"); ax.legend()
    ax.set_xlabel("X"); ax.set_ylabel("Y")

    ax = axes[0, 1]
    ax.scatter(xyz[:, 0], xyz[:, 2], c=rgb, s=0.5, marker=".", linewidths=0)
    if have_traj:
        ax.plot(c_est[:, 0], c_est[:, 2], "-", color="red", lw=1.2)
    ax.set_title("PLY side (X-Z) — height check"); ax.set_aspect("equal")
    ax.set_xlabel("X"); ax.set_ylabel("Z")

    ax = axes[1, 0]
    if have_traj and have_gt and s is not None:
        ax.plot(cg[:, 0], cg[:, 1], "-", color="green", lw=1.5, label="GT DJI")
        ax.plot(c_est_al[:, 0], c_est_al[:, 1], "-", color="red", lw=1.0, label="est (Sim3)")
        ax.scatter([cg[0, 0]], [cg[0, 1]], color="green", s=60, marker="o")
        ax.scatter([cg[-1, 0]], [cg[-1, 1]], color="green", s=60, marker="s")
        ax.set_title("Trajectory est vs GT (BEV, Sim3-aligned)")
        ax.set_aspect("equal"); ax.legend()
    else:
        ax.text(0.5, 0.5, "no GT/traj overlay", ha="center")
    ax.set_xlabel("X"); ax.set_ylabel("Y")

    ax = axes[1, 1]
    if have_traj and have_gt and s is not None:
        ax.plot(cg[:, 2], "-", color="green", lw=1.5, label="GT height")
        ax.plot(c_est_al[:, 2], "-", color="red", lw=1.0, label="est height")
        ax.set_title("Height over KF index (Sim3-aligned)"); ax.legend()
        ax.set_xlabel("KF idx"); ax.set_ylabel("Z")
    else:
        ax.text(0.5, 0.5, "no GT/traj overlay", ha="center")

    plt.tight_layout()
    plt.savefig(out, dpi=90)
    print(f"[render] wrote {out}")


if __name__ == "__main__":
    main()
