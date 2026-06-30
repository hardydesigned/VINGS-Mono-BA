#!/usr/bin/env python3
"""Standalone-Detektor auf annotierten Frames -> detections_per_frame.csv.

Fuer die 2D-Detektionsguete-Eval (mAP/IoU) auf Sequenzen, bei denen ein voller
VINGS-Lauf fuer 3D nicht lohnt (z.B. nur DJI-Posen). Der In-Run-Detektor ist
nichts anderes als YOLO auf den Keyframe-Bildern -- hier wenden wir denselben
Detektor (gleiches Modell/imgsz/conf) direkt auf JEDEN annotierten Frame an
(volle GT-Coverage statt nur eines Slices, kein VRAM-/SLAM-Overhead).

Schreibt detections_per_frame.csv im object_eval-Schema (3D-Spalten = NaN, da
kein Lauf) + det_camstamp.txt (ts<->stem), sodass `object_eval.py --part a`
direkt darauf laeuft.

    python scripts/eval/detect_on_frames.py \
      --cam-dir   /home/philipp/Dokumente/datasets/AMvalley03/cam_left \
      --label-dir /home/philipp/Dokumente/datasets/uavscenes/AMvalley03_labels \
      --out       output/det2d_AMvalley03 \
      --model yolo26n-obb --ckpt ckpts/yolo26n-obb.pt --classes 9 10 --imgsz 1280
    python scripts/eval/object_eval.py --part a --rundir output/det2d_AMvalley03 \
      --label-dir /home/philipp/Dokumente/datasets/uavscenes/AMvalley03_labels \
      --camstamp output/det2d_AMvalley03/det_camstamp.txt
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'vings_utils'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cam-dir', required=True,
                    help='Voll-Res CAM-Bilder, Timestamp-benannt (z.B. cam_left)')
    ap.add_argument('--label-dir', required=True,
                    help='UAVScenes-Label-Dir (waehlt die auszuwertenden Frames)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--cam-ext', default='.jpg')
    ap.add_argument('--kind', default='yolo', choices=['yolo', 'rtdetr'])
    ap.add_argument('--model', default='yolo26n-obb')
    ap.add_argument('--ckpt', default='ckpts/yolo26n-obb.pt',
                    help='Gewichts-Pfad; leer "" = auto-download (z.B. rtdetr-l)')
    ap.add_argument('--classes', type=int, nargs='*', default=[9, 10])
    ap.add_argument('--conf', type=float, default=0.20)
    ap.add_argument('--iou', type=float, default=0.7)
    ap.add_argument('--imgsz', type=int, default=1280)
    ap.add_argument('--max-det', type=int, default=100)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--limit', type=int, default=0, help='nur erste N Frames (Test)')
    a = ap.parse_args()

    from detector_factory import make_object_detector
    det_cfg = {'object_detector': {
        'kind': a.kind, 'model': a.model, 'ckpt_path': (a.ckpt or None),
        'conf': a.conf, 'iou': a.iou, 'imgsz': a.imgsz,
        'classes': (a.classes if a.classes else None),
        'max_det': a.max_det, 'device': a.device}}
    detector = make_object_detector(det_cfg, a.device)

    # auszuwertende Frames = Label-Stems, die ein CAM-Bild haben
    label_pngs = [p for p in sorted(glob.glob(
        os.path.join(a.label_dir, '**', '*.png'), recursive=True))
        if '_color' not in p]
    stems = [os.path.splitext(os.path.basename(p))[0] for p in label_pngs]
    if a.limit:
        stems = stems[:a.limit]
    os.makedirs(a.out, exist_ok=True)

    rows, cam_lines, n_img, n_det = [], [], 0, 0
    for kf, stem in enumerate(stems):
        cam = os.path.join(a.cam_dir, stem + a.cam_ext)
        if not os.path.isfile(cam):
            continue
        rgb = np.array(Image.open(cam).convert('RGB'))
        dets = detector.detect(rgb)
        t = float(stem)
        cam_lines.append(f"{stem} {stem}{a.cam_ext}")
        for d in dets:
            x1, y1, x2, y2 = d.bbox_xyxy
            rows.append((int(kf), t, d.cls_name, int(d.cls_id), float(d.conf),
                         float(x1), float(y1), float(x2), float(y2)))
            n_det += 1
        n_img += 1
        if n_img % 100 == 0:
            print(f"  {n_img}/{len(stems)} frames, {n_det} dets", flush=True)

    det_csv = os.path.join(a.out, 'detections_per_frame.csv')
    with open(det_csv, 'w') as f:
        f.write("frame_idx,t_sec,kf,object_id,class,cls_id,conf,"
                "x1,y1,x2,y2,depth,wx,wy,wz\n")
        for (kf, t, cn, cid, cf, x1, y1, x2, y2) in rows:
            f.write(f"{kf},{t:.6f},{kf},-1,{cn},{cid},{cf:.4f},"
                    f"{x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f},nan,nan,nan,nan\n")
    with open(os.path.join(a.out, 'det_camstamp.txt'), 'w') as f:
        f.write('\n'.join(cam_lines) + '\n')
    print(f"[detect_on_frames] {n_img} Bilder, {n_det} Detektionen -> {det_csv}")


if __name__ == '__main__':
    main()
