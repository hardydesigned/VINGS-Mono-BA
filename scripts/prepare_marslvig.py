#!/usr/bin/env python3
"""Convert a raw MARS-LVIG / UAVScenes bag-extraction into the amtown03 VINGS layout.

The raw datasets (output of ~/Dokumente/datasets/extract_bag.py) look like:

    <root>/cam_left/<unix_ts>.jpg          (2448x2048, name = Unix timestamp)
    <root>/lidar/<unix_ts>.bin
    <root>/dji_osdk_ros_*.csv              (one CSV per DJI OSDK topic)

amtown03 (the reference) instead has:

    <root>/images_all/000000.jpg ...       (1224x1024, sequential)
    <root>/metadata/camstamp_all.txt       ("<t_sec> <NNNNNN.jpg>")
    <root>/metadata/rtk.csv  /  gps.csv     ("# t_sec lat lon alt")
    <root>/metadata/dji_poses_all_w2c.txt  (TUM "ts tx ty tz qx qy qz qw", row N = frame N)
    <root>/metadata/dji_poses_all_c2w.txt
    <root>/metadata/intrinsic_half.txt

This script produces that layout so the new datasets plug into the existing
generic_vo loader / fair_eval / sweep tooling exactly like amtown03.

Design choices (kept faithful to amtown03 for comparability):
  * images_all = pure DOWNSCALE to 1224x1024 (NO undistortion). amtown03's
    runtime config uses raw-K/2 intrinsics (i.e. no undistortion was applied
    to its images_all either); matching that keeps the comparison apples-to-
    apples. The mild Plumb-Bob distortion is left uncorrected in both. The
    choice acts on every selector equally -> no bias in the selector ranking.
  * GT poses (dji_poses_all_*) are built from /dji_osdk_ros/local_position
    (xyz) + /dji_osdk_ros/attitude (quaternion body->world), sampled at each
    camera timestamp, with the world origin anchored at frame 0's position
    (exactly what amtown03's dji_poses files do: row 0 translation = 0). The
    known ~10% local_position scale bias is irrelevant to fair_eval's
    Sim(3)-aligned ATE (scale-invariant) and to held-out PSNR.

Usage:
    python scripts/prepare_marslvig.py --dataset AMvalley03
    python scripts/prepare_marslvig.py --dataset HKisland03  --motion-report
    python scripts/prepare_marslvig.py --dataset HKairport03 --force-images
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import cv2


# ── Per-scene camera calibration (raw 2448x2048), from the authoritative
#    UAVScenes/calibration_results.py. OpenCV Plumb-Bob D = [k1,k2,p1,p2,k3].
CALIB: dict[str, dict] = {
    "AMvalley03":  dict(fx=1453.88, fy=1452.85, cx=1182.53, cy=1045.82,
                        D=[-0.052,  0.1168, 0.0015,  0.00013, -0.068564]),
    "HKisland03":  dict(fx=1444.43, fy=1444.34, cx=1177.80, cy=1043.60,
                        D=[-0.053,  0.121,  0.00127, 0.00043, -0.06495]),
    "HKairport03": dict(fx=1451.28, fy=1451.29, cx=1177.50, cy=1043.50,
                        D=[-0.0572, 0.1209, 0.00124, -0.00018, -0.06327]),
}

RAW_W, RAW_H = 2448, 2048
OUT_W, OUT_H = 1224, 1024          # half-res, identical to amtown03 images_all

DATASETS_ROOT = Path.home() / "Dokumente" / "datasets"


# ── CSV helpers ───────────────────────────────────────────────────────────────
def load_csv(path: Path, cols: list[str]) -> dict[str, np.ndarray]:
    """Load named columns from a header CSV into float arrays."""
    if not path.exists():
        sys.exit(f"[prep] FEHLER: fehlt {path}")
    with open(path) as f:
        header = f.readline().strip().split(",")
    idx = {c: header.index(c) for c in cols}
    raw = np.genfromtxt(path, delimiter=",", skip_header=1,
                        usecols=[idx[c] for c in cols], dtype=float)
    if raw.ndim == 1:
        raw = raw[None, :]
    return {c: raw[:, i] for i, c in enumerate(cols)}


def quat_wxyz_to_R(q: np.ndarray) -> np.ndarray:
    """(w,x,y,z) unit quaternion -> 3x3 rotation matrix."""
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)],
    ])


def R_to_quat_xyzw(R: np.ndarray) -> np.ndarray:
    """3x3 rotation -> (x,y,z,w) quaternion (TUM order)."""
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([x, y, z, w])
    return q / np.linalg.norm(q)


# ── Steps ──────────────────────────────────────────────────────────────────────
def list_camstamps(root: Path) -> tuple[list[Path], np.ndarray]:
    """Sorted cam_left files + their Unix-time stamps (parsed from filename)."""
    files = sorted((root / "cam_left").glob("*.jpg"),
                   key=lambda p: float(p.stem))
    if not files:
        sys.exit(f"[prep] FEHLER: keine Bilder in {root/'cam_left'}")
    ts = np.array([float(p.stem) for p in files])
    return files, ts


def write_images(files: list[Path], out_dir: Path, force: bool) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(out_dir.glob("*.jpg"))
    if not force and len(existing) == len(files):
        print(f"[prep] images_all bereits vollständig ({len(existing)}) — skip "
              f"(--force-images zum Neuschreiben)")
        return
    for i, src in enumerate(files):
        img = cv2.imread(str(src), cv2.IMREAD_COLOR)
        if img is None:
            sys.exit(f"[prep] FEHLER: konnte {src} nicht lesen")
        if (img.shape[1], img.shape[0]) != (OUT_W, OUT_H):
            img = cv2.resize(img, (OUT_W, OUT_H), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(out_dir / f"{i:06d}.jpg"), img,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        if (i + 1) % 500 == 0:
            print(f"[prep]   images {i+1}/{len(files)}")
    print(f"[prep] images_all geschrieben: {len(files)} @ {OUT_W}x{OUT_H}")


def write_camstamp(files: list[Path], meta: Path) -> None:
    with open(meta / "camstamp_all.txt", "w") as f:
        for i, p in enumerate(files):
            f.write(f"{p.stem} {i:06d}.jpg\n")
    print(f"[prep] camstamp_all.txt: {len(files)} Zeilen")


def write_gps(root: Path, meta: Path) -> None:
    for src_name, out_name in (("dji_osdk_ros_rtk_position.csv", "rtk.csv"),
                               ("dji_osdk_ros_gps_position.csv", "gps.csv")):
        src = root / src_name
        if not src.exists():
            print(f"[prep] WARN: {src_name} fehlt — {out_name} übersprungen")
            continue
        d = load_csv(src, ["stamp", "latitude", "longitude", "altitude"])
        order = np.argsort(d["stamp"])
        with open(meta / out_name, "w") as f:
            f.write("# t_sec lat lon alt\n")
            for i in order:
                f.write(f"{d['stamp'][i]:.6f} {d['latitude'][i]:.9f} "
                        f"{d['longitude'][i]:.9f} {d['altitude'][i]:.4f}\n")
        print(f"[prep] {out_name}: {len(order)} Zeilen")


def write_poses(root: Path, meta: Path, cam_ts: np.ndarray) -> None:
    lp = load_csv(root / "dji_osdk_ros_local_position.csv",
                  ["stamp", "point.x", "point.y", "point.z"])
    at = load_csv(root / "dji_osdk_ros_attitude.csv",
                  ["stamp", "quaternion.w", "quaternion.x",
                   "quaternion.y", "quaternion.z"])
    # sort sources by time
    lo = np.argsort(lp["stamp"]); lt = lp["stamp"][lo]
    lxyz = np.stack([lp["point.x"][lo], lp["point.y"][lo], lp["point.z"][lo]], 1)
    ao = np.argsort(at["stamp"]); at_t = at["stamp"][ao]
    aq = np.stack([at["quaternion.w"][ao], at["quaternion.x"][ao],
                   at["quaternion.y"][ao], at["quaternion.z"][ao]], 1)

    # interpolate position per-axis; nearest-neighbour quaternion (~100 Hz src)
    pos = np.stack([np.interp(cam_ts, lt, lxyz[:, k]) for k in range(3)], 1)
    p0 = pos[0].copy()
    qa_idx = np.clip(np.searchsorted(at_t, cam_ts), 0, len(at_t) - 1)
    # pick the closer of the two bracketing attitude samples
    left = np.clip(qa_idx - 1, 0, len(at_t) - 1)
    use_left = np.abs(at_t[left] - cam_ts) < np.abs(at_t[qa_idx] - cam_ts)
    qa_idx = np.where(use_left, left, qa_idx)

    fw = open(meta / "dji_poses_all_w2c.txt", "w")
    fc = open(meta / "dji_poses_all_c2w.txt", "w")
    for i, ts in enumerate(cam_ts):
        R = quat_wxyz_to_R(aq[qa_idx[i]])        # body(cam) -> world
        c = pos[i] - p0                          # camera centre, world origin @ frame 0
        # c2w = [R | c]
        q_c2w = R_to_quat_xyzw(R)
        fc.write(f"{ts:.9f} {c[0]:.6f} {c[1]:.6f} {c[2]:.6f} "
                 f"{q_c2w[0]:.6f} {q_c2w[1]:.6f} {q_c2w[2]:.6f} {q_c2w[3]:.6f}\n")
        # w2c = [R^T | -R^T c]
        Rt = R.T
        t = -Rt @ c
        q_w2c = R_to_quat_xyzw(Rt)
        fw.write(f"{ts:.9f} {t[0]:.6f} {t[1]:.6f} {t[2]:.6f} "
                 f"{q_w2c[0]:.6f} {q_w2c[1]:.6f} {q_w2c[2]:.6f} {q_w2c[3]:.6f}\n")
    fw.close(); fc.close()
    print(f"[prep] dji_poses_all_{{w2c,c2w}}.txt: {len(cam_ts)} Zeilen "
          f"(Welt-Ursprung @ Frame 0)")


def write_intrinsic(meta: Path, k: dict) -> None:
    """Raw-K/2 (downscale-only) as a 3x3, matching the chosen images_all res."""
    fx, fy = k["fx"] / 2, k["fy"] / 2
    cx, cy = k["cx"] / 2, k["cy"] / 2
    with open(meta / "intrinsic_half.txt", "w") as f:
        f.write(f"{fx:.6f} 0.000000 {cx:.6f}\n")
        f.write(f"0.000000 {fy:.6f} {cy:.6f}\n")
        f.write("0.000000 0.000000 1.000000\n")
    # config (generic_vo) convention: fu=fy, fv=fx, cu=cy, cv=cx
    print(f"[prep] intrinsic_half.txt (raw/2). Config-Block-Werte: "
          f"H={OUT_H} W={OUT_W} fu={fy:.2f} fv={fx:.2f} cu={cy:.2f} cv={cx:.2f}")


def motion_report(root: Path, cam_ts: np.ndarray, block: int = 100) -> None:
    """Print per-block mean horizontal speed + |yaw rate| to sanity-check the
    chosen flight/hover/turn windows (frame indices == sequential after rename)."""
    vel = load_csv(root / "dji_osdk_ros_velocity.csv",
                   ["stamp", "vector.x", "vector.y"])
    ang = load_csv(root / "dji_osdk_ros_angular_velocity_fused.csv",
                   ["stamp", "vector.z"])
    vo = np.argsort(vel["stamp"])
    speed_src = np.hypot(vel["vector.x"][vo], vel["vector.y"][vo])
    speed = np.interp(cam_ts, vel["stamp"][vo], speed_src)
    yo = np.argsort(ang["stamp"])
    yaw = np.abs(np.interp(cam_ts, ang["stamp"][yo], ang["vector.z"][yo]))
    n = len(cam_ts)
    print(f"\n[prep] motion report ({n} frames, block={block}): "
          f"speed[m/s] | |yaw|[rad/s]")
    for s in range(0, n, block):
        e = min(s + block, n)
        print(f"  f{s:5d}-{e-1:5d}  v={speed[s:e].mean():5.2f}  "
              f"yaw={yaw[s:e].mean():.3f}  vmax={speed[s:e].max():5.2f}  "
              f"yawmax={yaw[s:e].max():.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=sorted(CALIB))
    ap.add_argument("--root", default=None,
                    help="dataset root (default ~/Dokumente/datasets/<dataset>)")
    ap.add_argument("--force-images", action="store_true",
                    help="re-write images_all even if already complete")
    ap.add_argument("--motion-report", action="store_true",
                    help="print per-100-frame speed/yaw blocks and exit-after-prep")
    args = ap.parse_args()

    root = Path(args.root) if args.root else DATASETS_ROOT / args.dataset
    if not root.exists():
        sys.exit(f"[prep] FEHLER: root fehlt: {root}")
    meta = root / "metadata"
    meta.mkdir(parents=True, exist_ok=True)
    k = CALIB[args.dataset]

    print(f"[prep] === {args.dataset} === root={root}")
    files, cam_ts = list_camstamps(root)
    print(f"[prep] {len(files)} Frames, t=[{cam_ts[0]:.3f} .. {cam_ts[-1]:.3f}] "
          f"({cam_ts[-1]-cam_ts[0]:.1f}s)")

    write_images(files, root / "images_all", args.force_images)
    write_camstamp(files, meta)
    write_gps(root, meta)
    write_poses(root, meta, cam_ts)
    write_intrinsic(meta, k)

    if args.motion_report:
        motion_report(root, cam_ts)

    print(f"[prep] fertig: {args.dataset}")


if __name__ == "__main__":
    main()
