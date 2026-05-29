#!/usr/bin/env python3
"""
Drift-Diagnose: Tracker-Trajektorie (poses_save aus run.py) gegen RTK-GT.

Usage:
    python scripts/plot_drift_diagnostic.py <run_output_dir> [--out PLOT.png]

Erwartet im run_output_dir:
    - tracker_poses_w2c.txt   (vom Patch in run.py)
    - config.yaml             (Config-Snapshot)
    - keyframelist.txt        (optional, fuer KF-Index-Zuordnung)

Plot:
    1) Top-down XY (ENU): Tracker vs RTK, beide auf Origin verschoben
    2) Hoehe Z ueber KF-Index
    3) Drift-Distance |xyz_track - xyz_gt| ueber KF-Index
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation as R
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def w2c_tq_to_c2w_xyz(tq: np.ndarray) -> np.ndarray:
    """tq = (N,7) [tx ty tz qx qy qz qw] world-to-cam -> c2w camera positions (N,3)."""
    t_w2c = tq[:, :3]
    q     = tq[:, 3:7]
    R_w2c = R.from_quat(q).as_matrix()              # (N,3,3)
    R_c2w = np.transpose(R_w2c, (0, 2, 1))           # (N,3,3)
    t_c2w = -np.einsum("nij,nj->ni", R_c2w, t_w2c)   # (N,3)
    return t_c2w


def lla_to_enu(lat_deg, lon_deg, alt_m, lat0, lon0, alt0):
    """Local tangent plane, Frame-0-zentriert. Klein-Sequenz-OK, no WGS84 ellipsoid corrections."""
    a = 6378137.0
    e2 = 6.69437999014e-3
    sin_lat0 = np.sin(np.deg2rad(lat0))
    cos_lat0 = np.cos(np.deg2rad(lat0))
    N0 = a / np.sqrt(1 - e2 * sin_lat0**2)
    dlat = np.deg2rad(lat_deg - lat0)
    dlon = np.deg2rad(lon_deg - lon0)
    east  = dlon * (N0 + alt0) * cos_lat0
    north = dlat * (N0 + alt0)
    up    = alt_m - alt0
    return np.stack([east, north, up], axis=-1)


def load_rtk_at_camstamps(rtk_csv: Path, cam_t_sec: np.ndarray) -> np.ndarray:
    """Interpoliere RTK lat/lon/alt linear auf Cam-Timestamps. Liefert (N,3) ENU."""
    rtk = np.loadtxt(rtk_csv, comments="#", delimiter=None)
    t   = rtk[:, 0]
    lat = rtk[:, 1]
    lon = rtk[:, 2]
    alt = rtk[:, 3]
    # Sortiert nach Zeit (RTK in amtown03 ist es schon)
    lat_at = np.interp(cam_t_sec, t, lat)
    lon_at = np.interp(cam_t_sec, t, lon)
    alt_at = np.interp(cam_t_sec, t, alt)
    return lla_to_enu(lat_at, lon_at, alt_at, lat_at[0], lon_at[0], alt_at[0])


def umeyama_2d(src: np.ndarray, dst: np.ndarray):
    """Schaetze (R,t,s) so dass s*R @ src.T + t ~ dst.T, nur XY-Komponente.
    Liefert eine Funktion src->aligned."""
    s = src[:, :2] - src[:, :2].mean(0)
    d = dst[:, :2] - dst[:, :2].mean(0)
    H = s.T @ d
    U, S, Vt = np.linalg.svd(H)
    Rm = Vt.T @ U.T
    if np.linalg.det(Rm) < 0:
        Vt[-1] *= -1
        Rm = Vt.T @ U.T
    var_src = (s ** 2).sum() / len(s)
    scale = (S.sum()) / (var_src * len(s)) if var_src > 1e-12 else 1.0
    t_xy = dst[:, :2].mean(0) - scale * (Rm @ src[:, :2].mean(0))

    def apply(p: np.ndarray) -> np.ndarray:
        out = p.copy()
        out[:, :2] = (scale * (Rm @ p[:, :2].T)).T + t_xy
        return out
    return apply, scale


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--out", default=None,
                    help="Plot-Pfad. Default: <run_dir>/drift_diagnostic.png")
    ap.add_argument("--no-umeyama", action="store_true",
                    help="Keine 2D-Similarity-Alignment, nur Origin-shift.")
    args = ap.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    if not run_dir.is_dir():
        print(f"[FEHLER] {run_dir} ist kein Ordner", file=sys.stderr)
        sys.exit(1)

    # Bevorzugt: tracker_raw_c2w.txt (per-KF Live-Log, voll). Fallback: das
    # finale tracker_poses_w2c.txt (Active+Marg, partiell).
    raw_c2w_file = run_dir / "tracker_raw_c2w.txt"
    pose_file    = run_dir / "tracker_poses_w2c.txt"
    cfg_file     = run_dir / "config.yaml"
    if not (raw_c2w_file.exists() or pose_file.exists()):
        print(f"[FEHLER] weder {raw_c2w_file.name} noch {pose_file.name} "
              f"in {run_dir} gefunden.", file=sys.stderr)
        sys.exit(1)
    if not cfg_file.exists():
        print(f"[FEHLER] {cfg_file} nicht gefunden", file=sys.stderr)
        sys.exit(1)

    with open(cfg_file) as f:
        cfg = yaml.safe_load(f)

    src_arr = None
    if raw_c2w_file.exists():
        # Pro-KF c2w-Live-Log: kf_idx t_sec + 12 floats (first 3 rows of 4x4 c2w).
        rows = np.loadtxt(raw_c2w_file, comments="#")
        if rows.ndim == 1:
            rows = rows[None, :]
        # cam-position = c2w[:3, 3] = columns 5, 9, 13 (1-indexed: 2 t_sec + 3 mat rows)
        # Layout: kf_idx t_sec r00 r01 r02 tx r10 r11 r12 ty r20 r21 r22 tz
        kf_t_sec  = rows[:, 1]
        track_xyz = np.stack([rows[:, 5], rows[:, 9], rows[:, 13]], axis=-1).astype(np.float64)
        n = len(track_xyz)
        print(f"Tracker-Posen (raw c2w live-log): {n} KFs")
    else:
        # Fallback: tracker_poses_w2c.txt
        kf_t_sec = None
        src_labels: list[str] = []
        tq_list: list[list[float]] = []
        with open(pose_file) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) >= 9 and not parts[1].lstrip('-').replace('.', '', 1).isdigit():
                    src_labels.append(parts[1])
                    tq_list.append([float(x) for x in parts[2:9]])
                else:
                    src_labels.append("legacy")
                    tq_list.append([float(x) for x in parts[1:8]])
        tq = np.asarray(tq_list, dtype=np.float64)
        if tq.ndim == 1:
            tq = tq[None, :]
        track_xyz = w2c_tq_to_c2w_xyz(tq)
        n = len(track_xyz)
        src_arr = np.asarray(src_labels)
        n_marg = int((src_arr == "marg").sum())
        n_act  = int((src_arr == "act").sum())
        print(f"Tracker-Posen (poses_save+active): {n} ({n_marg} marginalisiert + {n_act} aktiv)")

    # ---- RTK-GT laden ----
    ds_cfg = cfg.get("dataset", {})
    rtk_csv = ds_cfg.get("gps_csv")
    if rtk_csv is None or not Path(rtk_csv).exists():
        print("[WARN] kein gps_csv im Config oder Datei fehlt -- ohne GT-Overlay",
              file=sys.stderr)
        gt_xyz = None
    else:
        # Cam-Timestamps: bevorzugt camstamp_all.txt, weil der start_frame-
        # Offset darauf basiert. Sonst aus camstamp_file.
        cam_stamps = None
        cs_all = Path(rtk_csv).parent / "camstamp_all.txt"
        cs_cfg = ds_cfg.get("camstamp_file")
        if cs_all.exists():
            cam_stamps = np.loadtxt(cs_all, comments="#", usecols=(0,))
            print(f"Cam-Timestamps aus {cs_all.name}: {len(cam_stamps)} Frames")
        elif cs_cfg and Path(cs_cfg).exists():
            cam_stamps = np.loadtxt(cs_cfg, comments="#", usecols=(0,))

        if cam_stamps is None:
            print("[WARN] keine Cam-Timestamps -- skip GT", file=sys.stderr)
            gt_xyz = None
        else:
            if kf_t_sec is not None:
                # Wir haben echte Per-KF-Timestamps -- direkt nutzen statt
                # linspace-Approximation.
                gt_xyz = load_rtk_at_camstamps(Path(rtk_csv), kf_t_sec)
                print(f"GT-RTK: {len(gt_xyz)} Stuetzpunkte (echte KF-Timestamps)")
            else:
                start_frame = int(ds_cfg.get("start_frame", 0))
                max_frames  = int(ds_cfg.get("max_frames", len(cam_stamps) - start_frame))
                end_frame   = min(start_frame + max_frames, len(cam_stamps))
                kf_frame_id = np.linspace(start_frame, end_frame - 1, n).astype(int)
                kf_frame_id = np.clip(kf_frame_id, 0, len(cam_stamps) - 1)
                cam_t = cam_stamps[kf_frame_id]
                gt_xyz = load_rtk_at_camstamps(Path(rtk_csv), cam_t)
                print(f"GT-RTK: {len(gt_xyz)} interpolierte Stuetzpunkte "
                      f"(KFs verteilt auf bag-frames {start_frame}..{end_frame-1})")

    # ---- Beide Spuren auf Origin ----
    track_xyz0 = track_xyz - track_xyz[0]
    if gt_xyz is not None:
        gt_xyz0 = gt_xyz - gt_xyz[0]
    else:
        gt_xyz0 = None

    # ---- 2D-Similarity-Alignment (Tracker -> RTK), optional ----
    track_aligned = track_xyz0
    scale = 1.0
    if gt_xyz0 is not None and not args.no_umeyama:
        apply, scale = umeyama_2d(track_xyz0, gt_xyz0)
        track_aligned = apply(track_xyz0)
        print(f"Similarity-Alignment: scale={scale:.4f}")

    # ---- Drift-Distance pro KF ----
    if gt_xyz0 is not None:
        drift = np.linalg.norm(track_aligned[:, :2] - gt_xyz0[:, :2], axis=1)
    else:
        drift = None

    # ---- Plot ----
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    ax0, ax1, ax2 = ax

    # Panel 0: Top-down XY
    ax0.plot(track_aligned[:, 0], track_aligned[:, 1], "-o", ms=2,
             label=f"Tracker (s={scale:.3f})", color="tab:blue")
    if gt_xyz0 is not None:
        ax0.plot(gt_xyz0[:, 0], gt_xyz0[:, 1], "-o", ms=2,
                 label="RTK-GT", color="tab:orange")
    ax0.scatter([0], [0], marker="*", s=80, color="black", zorder=5,
                label="Start")
    ax0.set_xlabel("East [m]")
    ax0.set_ylabel("North [m]")
    ax0.set_aspect("equal", adjustable="datalim")
    ax0.set_title("Top-Down XY (origin = KF0)")
    ax0.legend()
    ax0.grid(alpha=0.3)

    # Panel 1: Z over KF index
    ax1.plot(track_aligned[:, 2], label="Tracker Z (after align)",
             color="tab:blue")
    if gt_xyz0 is not None:
        ax1.plot(gt_xyz0[:, 2], label="RTK Z", color="tab:orange")
    ax1.set_xlabel("KF index")
    ax1.set_ylabel("Up [m]")
    ax1.set_title("Hoehe ueber KF-Index")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Panel 2: Drift over KF index
    if drift is not None:
        ax2.plot(drift, color="tab:red")
        ax2.fill_between(np.arange(len(drift)), 0, drift, alpha=0.2,
                          color="tab:red")
        ax2.set_xlabel("KF index")
        ax2.set_ylabel("|track - GT|_xy [m]")
        max_d = drift.max()
        med_d = np.median(drift)
        ax2.set_title(f"2D-Drift (median={med_d:.2f}m  max={max_d:.2f}m)")
        ax2.grid(alpha=0.3)
    else:
        ax2.text(0.5, 0.5, "Kein GT vorhanden", ha="center", va="center")
        ax2.set_axis_off()

    plt.suptitle(f"Drift-Diagnose: {run_dir.name}", fontsize=11)
    plt.tight_layout()

    out_path = Path(args.out) if args.out else (run_dir / "drift_diagnostic.png")
    plt.savefig(out_path, dpi=110)
    print(f"Plot -> {out_path}")


if __name__ == "__main__":
    main()
