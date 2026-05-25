"""
Build per-frame ext_poses_file (TUM-style w2c) for AGZ slices.

Combines AGZ's OnboardGPS.csv (per-frame lat/lon/alt) with
OnboardPose.csv (IMU-rate attitude quaternion) to produce a TUM-format
[ts, tx, ty, tz, qx, qy, qz, qw] world-to-camera pose per frame.

ENU origin = first frame in slice. Body (=camera) attitude from
OnboardPose by nearest-timestamp match.

Note: assumes camera body-frame == optical-frame (Nadir downward).
If PLY orientation is wrong by a fixed rotation, apply a camera-to-body
extrinsic via --cam_to_body argument (qx qy qz qw).

Usage:
    python scripts/build_agz_ext_poses.py \
        --gps  "/tmp/AGZ/Log Files/OnboardGPS.csv" \
        --pose "/tmp/AGZ/Log Files/OnboardPose.csv" \
        --start 1 --count 1500 \
        --out  ~/Dokumente/datasets/agz/agz_0_1500/agz_poses_w2c.txt
"""
import argparse
import numpy as np
from scipy.spatial.transform import Rotation as R


def parse_csv(path, n_cols, skiprows=1):
    rows = []
    with open(path) as f:
        for i, line in enumerate(f):
            if i < skiprows:
                continue
            parts = line.strip().split(',')
            if len(parts) < n_cols:
                continue
            try:
                row = [float(p.strip()) for p in parts[:n_cols]]
            except ValueError:
                continue
            rows.append(row)
    return np.array(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gps', required=True)
    ap.add_argument('--pose', required=True)
    ap.add_argument('--start', type=int, default=1,
                    help='1-indexed imgid for first frame in slice (matches prepare_agz.py)')
    ap.add_argument('--count', type=int, required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    gps = parse_csv(args.gps, n_cols=5)
    pose = parse_csv(args.pose, n_cols=18)
    print(f'GPS rows: {len(gps)}  Pose rows: {len(pose)}')

    # GPS: cols [ts, imgid, lat, lon, alt]
    # Pose: cols 0=ts, 14/15/16/17 = Attitude_w/x/y/z
    pose_t = pose[:, 0]
    pose_q_wxyz = pose[:, 14:18]  # (N, 4) [w, x, y, z]

    R_earth = 6378137.0
    # Use first slice frame as ENU origin.
    slice0 = args.start - 1  # 0-indexed into GPS
    if slice0 < 0 or slice0 + args.count > len(gps):
        raise ValueError(f'Slice [{slice0}, {slice0+args.count}) out of range [0, {len(gps)})')

    lat0_rad = np.deg2rad(gps[slice0, 2])
    lon0_rad = np.deg2rad(gps[slice0, 3])
    alt0 = gps[slice0, 4]

    out_rows = []
    n_pose_missing = 0
    for i in range(args.count):
        gps_row = gps[slice0 + i]
        ts, imgid, lat, lon, alt = gps_row[:5]

        # ENU c2w translation
        e = (np.deg2rad(lon) - lon0_rad) * R_earth * np.cos(lat0_rad)
        n = (np.deg2rad(lat) - lat0_rad) * R_earth
        u = alt - alt0
        t_c2w = np.array([e, n, u], dtype=np.float64)

        # Nearest pose by timestamp
        idx = int(np.argmin(np.abs(pose_t - ts)))
        if abs(pose_t[idx] - ts) > 1e6:  # > 1 sec mismatch -> skip
            n_pose_missing += 1
            qw, qx, qy, qz = 1.0, 0.0, 0.0, 0.0
        else:
            qw, qx, qy, qz = pose_q_wxyz[idx]

        # Attitude quaternion as body-to-world rotation
        # scipy quat order: [x, y, z, w]
        R_c2w = R.from_quat([qx, qy, qz, qw]).as_matrix()
        # w2c
        R_w2c = R_c2w.T
        t_w2c = -R_w2c @ t_c2w
        q_w2c = R.from_matrix(R_w2c).as_quat()  # [qx, qy, qz, qw]

        out_rows.append([ts / 1e6, t_w2c[0], t_w2c[1], t_w2c[2],
                         q_w2c[0], q_w2c[1], q_w2c[2], q_w2c[3]])

    out = np.array(out_rows)
    print(f'Output: {out.shape}, pose mismatches: {n_pose_missing}')
    print(f'  ENU c2w span: '
          f'{((np.array([row[1:4] for row in out_rows]) * 0).std(0))}')
    # Compute c2w trajectory diagnostics
    c2w_xyz = []
    for row in out_rows:
        tq = np.array(row[1:])
        t_w2c = tq[:3]
        q_w2c = tq[3:]
        Rm = R.from_quat(q_w2c).as_matrix()
        c = -Rm.T @ t_w2c
        c2w_xyz.append(c)
    c2w_xyz = np.array(c2w_xyz)
    print(f'  c2w xyz span (m): {c2w_xyz.max(0) - c2w_xyz.min(0)}')
    print(f'  c2w path length (m): {np.linalg.norm(np.diff(c2w_xyz, axis=0), axis=1).sum():.2f}')
    print(f'  c2w first->last: {np.linalg.norm(c2w_xyz[-1] - c2w_xyz[0]):.2f}')

    np.savetxt(args.out, out, fmt='%.9f')
    print(f'Wrote {len(out)} ext poses -> {args.out}')


if __name__ == '__main__':
    main()
