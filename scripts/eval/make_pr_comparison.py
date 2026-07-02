#!/usr/bin/env python3
"""Multi-Detektor Precision-Recall-Kurven (ein Plot) fuer den OD-Vergleich.

Fuer jeden Detektor: detections_per_frame.csv + UAVScenes-Masken (AMtown03) ->
klassenagnostisches Matching bei IoU>=0.5 -> kumulative Precision/Recall ->
eine Kurve pro Detektor mit AP@.5 in der Legende. Reused object_eval-Helfer.

    python scripts/eval/make_pr_comparison.py --out thesis/img/modulb_pr_curve.png
"""
from __future__ import annotations
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import object_eval as oe

DS = '/home/philipp/Dokumente/datasets'
LABELS = f'{DS}/uavscenes/interval1_amtown03_labels'
W, H = 2448, 2048

# (Anzeigename, run-dir, camstamp)  -- alle auf AMtown03 detektiert
DETECTORS = [
    ('YOLOv8s-VisDrone', 'output/odcmp_visdrone', 'output/odcmp_visdrone/det_camstamp.txt'),
    ('YOLO26n-OBB (DOTA)', 'output/det2d_AMtown03', 'output/det2d_AMtown03/det_camstamp.txt'),
    ('RT-DETR-l (COCO)', 'output/odcmp_rtdetr', 'output/odcmp_rtdetr/det_camstamp.txt'),
    ('YOLOv8n (COCO)', 'output/odcmp_coco', 'output/odcmp_coco/det_camstamp.txt'),
]


_GT_CACHE = {}   # stem -> [normierte GT-Boxen]  (einmal laden, alle Detektoren teilen)


def _gt_for(stem):
    if stem not in _GT_CACHE:
        lp = oe.find_label_for_ts(LABELS, stem)
        _GT_CACHE[stem] = (None if lp is None else
                           [oe._norm_box(ins['bbox'], W, H)
                            for ins in oe.gt_instances(lp, 30)])
    return _GT_CACHE[stem]


def pr_curve(rundir, camstamp, iou_thr=0.5):
    """-> (recall[], precision[], AP@.5) klassenagnostisch 'vehicle'."""
    dets_by_t = oe.load_detections_csv(os.path.join(rundir, 'detections_per_frame.csv'))
    _, stem2ts = oe.load_camstamp(camstamp)
    ts2stem = {round(t, 3): s for s, t in stem2ts.items()}
    matches, n_gt = [], 0
    for tkey, dets in dets_by_t.items():
        stem = ts2stem.get(tkey)
        if stem is None:
            continue
        gts = _gt_for(stem)
        if gts is None:
            continue
        n_gt += len(gts)
        ds = sorted(dets, key=lambda d: -d['conf'])
        used = [False] * len(gts)
        for d in ds:
            db = oe._norm_box(d['bbox'], W, H)
            best_iou, best_j = 0.0, -1
            for j, g in enumerate(gts):
                if used[j]:
                    continue
                i = oe.iou_xyxy(db, g)
                if i > best_iou:
                    best_iou, best_j = i, j
            tp = best_iou >= iou_thr and best_j >= 0
            if tp:
                used[best_j] = True
            matches.append((d['conf'], tp))
    matches.sort(key=lambda m: -m[0])
    tp = np.array([1 if m[1] else 0 for m in matches], float)
    fp = 1.0 - tp
    ctp, cfp = np.cumsum(tp), np.cumsum(fp)
    rec = ctp / max(n_gt, 1)
    prec = ctp / np.maximum(ctp + cfp, 1e-9)
    ap, _, _ = oe.average_precision(matches, n_gt)
    return rec, prec, ap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='thesis/img/modulb_pr_curve.png')
    a = ap.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6, 4.5))
    for name, rundir, cs in DETECTORS:
        if not os.path.isfile(os.path.join(rundir, 'detections_per_frame.csv')):
            print(f'skip {name} (keine CSV)'); continue
        rec, prec, apv = pr_curve(rundir, cs)
        plt.plot(rec, prec, lw=1.8, label=f'{name}  (AP@.5={apv:.3f})')
        print(f'{name}: AP@.5={apv:.3f}  ({len(rec)} Detektionen)')
    plt.title('Precision-Recall der Detektoren (AMtown03, IoU≥0.5, "vehicle")')
    plt.xlabel('Recall'); plt.ylabel('Precision')
    plt.xlim(0, 1); plt.ylim(0, 1.02); plt.grid(alpha=0.3)
    plt.legend(loc='upper right', fontsize=8)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    plt.tight_layout(); plt.savefig(a.out, dpi=150)
    print(f'-> {a.out}')


if __name__ == '__main__':
    main()
