#!/usr/bin/env python3
"""Konvertiert UAVScenes interval1_AMtown03 -> VINGS-kompatible Inputs.

UAVScenes liefert pro Frame eine GT-Pose (T4x4, c2w), Intrinsik (P3x3),
und LiDAR. Dieses Skript erzeugt:
  - <out>/poses_w2c.txt   : ext_poses / fair_eval-GT (TUM-w2c: t tx ty tz qx qy qz qw)
  - <out>/camstamp.txt    : <t_sec> <imagename> pro Frame (SortedImageID-Reihenfolge)
  - <out>/intrinsic.txt   : native K + W,H (zum Eintragen in die Config)

Bilder bleiben in interval1_CAM/ (Loader skaliert via frontend.image_size).

Usage:
  python scripts/prepare_uavscenes_interval.py \
      --root ~/Dokumente/datasets/interval1_AMtown03 --out <same>/vings
"""
import os, json, argparse
import numpy as np
from scipy.spatial.transform import Rotation as R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or os.path.join(args.root, "vings")
    os.makedirs(out, exist_ok=True)

    info = json.load(open(os.path.join(args.root, "sampleinfos_interpolated.json")))
    # nach SortedImageID sortieren (= chronologisch = sortierte Dateinamen)
    info = sorted(info, key=lambda x: x["SortedImageID"])

    poses_lines, cam_lines = [], []
    K = np.array(info[0]["P3x3"]).reshape(3, 3)
    W, H = info[0]["Width"], info[0]["Height"]

    for it in info:
        name = it["OriginalImageName"]           # z.B. 1658131847.149322787.jpg
        t = float(os.path.splitext(name)[0])
        T = np.array(it["T4x4"], dtype=np.float64).reshape(4, 4)  # c2w
        Rc2w, tc2w = T[:3, :3], T[:3, 3]
        Rw2c = Rc2w.T
        tw2c = -Rw2c @ tc2w
        q = R.from_matrix(Rw2c).as_quat()        # [x,y,z,w]
        poses_lines.append(f"{t:.9f} {tw2c[0]:.6f} {tw2c[1]:.6f} {tw2c[2]:.6f} "
                           f"{q[0]:.9f} {q[1]:.9f} {q[2]:.9f} {q[3]:.9f}")
        cam_lines.append(f"{t:.9f} {name}")

    with open(os.path.join(out, "poses_w2c.txt"), "w") as f:
        f.write("\n".join(poses_lines) + "\n")
    with open(os.path.join(out, "camstamp.txt"), "w") as f:
        f.write("\n".join(cam_lines) + "\n")
    with open(os.path.join(out, "intrinsic.txt"), "w") as f:
        f.write(f"# native UAVScenes intrinsic (rectified, distortion=0)\n")
        f.write(f"fu {K[0,0]:.4f}\nfv {K[1,1]:.4f}\ncu {K[0,2]:.4f}\ncv {K[1,2]:.4f}\n")
        f.write(f"W {W}\nH {H}\n")

    print(f"[prep] {len(info)} frames -> {out}")
    print(f"[prep] K: fu={K[0,0]:.1f} fv={K[1,1]:.1f} cu={K[0,2]:.1f} cv={K[1,2]:.1f}  WxH={W}x{H}")
    print(f"[prep] poses_w2c.txt, camstamp.txt, intrinsic.txt geschrieben")


if __name__ == "__main__":
    main()
