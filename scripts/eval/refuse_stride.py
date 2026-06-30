#!/usr/bin/env python3
"""object_detect_stride-Sweep ohne Neu-Läufe.

Der stride entkoppelt nur, auf welchem n-ten Tracker-KF detektiert wird -- das
Tracking (Posen/Tiefe) bleibt identisch. Daher lässt sich ein stride-n-Lauf
exakt aus dem stride-1-Detektionslog simulieren: nur Detektionen auf KFs mit
kf % n == 0 behalten und neu fusionieren (3D-NN wie object_tracker).

Schreibt pro stride ein objects_droid.csv (nur Position -- yaw/size stehen nicht
im Detektionslog), auf das dann `object_eval.py --part b` läuft.

    python scripts/eval/refuse_stride.py --rundir <run> --strides 1 2 3 5 10
"""
from __future__ import annotations
import argparse
import csv
import os
import numpy as np


def fuse(dets, assoc_r, min_hits):
    """NN-Fusion von (xyz, conf, cls) -> Objekte (xyz=weighted-median, n)."""
    tracks = []
    for p, conf, cls in dets:
        best, bd = None, assoc_r
        for tr in tracks:
            if tr['cls'] != cls:
                continue
            d = float(np.linalg.norm(tr['c'] - p))
            if d < bd:
                best, bd = tr, d
        if best is None:
            best = {'cls': cls, 'pts': [], 'confs': [], 'c': p}
            tracks.append(best)
        best['pts'].append(p); best['confs'].append(conf)
        best['c'] = np.mean(best['pts'], axis=0)
    objs = []
    for tr in tracks:
        if len(tr['pts']) < min_hits:
            continue
        P = np.asarray(tr['pts']); w = np.asarray(tr['confs'])
        if w.sum() <= 0:
            w = np.ones_like(w)
        xyz = []
        for k in range(3):
            o = np.argsort(P[:, k]); cw = np.cumsum(w[o])
            xyz.append(float(P[o][min(int(np.searchsorted(cw, 0.5 * w.sum())), len(P) - 1), k]))
        objs.append({'cls': tr['cls'], 'xyz': xyz, 'n': len(tr['pts']),
                     'conf': float(max(tr['confs']))})
    objs.sort(key=lambda o: (-o['n'], -o['conf']))
    return objs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rundir', required=True)
    ap.add_argument('--strides', type=int, nargs='+', default=[1, 2, 3, 5, 10])
    ap.add_argument('--assoc', type=float, default=3.0)
    ap.add_argument('--min-hits', type=int, default=2)
    a = ap.parse_args()

    rows = []
    with open(os.path.join(a.rundir, 'detections_per_frame.csv')) as f:
        for r in csv.DictReader(f):
            if r['wx'] in ('nan', '', 'NaN'):
                continue
            try:
                p = np.array([float(r['wx']), float(r['wy']), float(r['wz'])])
            except ValueError:
                continue
            rows.append((int(r['kf']), p, float(r['conf']), r['class']))
    kfs = sorted({r[0] for r in rows})
    print(f"{len(rows)} lokalisierte Detektionen ueber {len(kfs)} KFs")

    for n in a.strides:
        keep_kf = set(kfs[::n])
        dets = [(p, conf, cls) for (kf, p, conf, cls) in rows if kf in keep_kf]
        objs = fuse(dets, a.assoc, a.min_hits)
        out = os.path.join(a.rundir, f'stride_sweep/n{n}')
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, 'objects_droid.csv'), 'w') as f:
            f.write("object_id,class,cls_id,conf,n_detections,x,y,z,"
                    "qw,qx,qy,qz,sx,sy,sz\n")
            for i, o in enumerate(objs):
                x, y, z = o['xyz']
                f.write(f"{i},{o['cls']},0,{o['conf']:.4f},{o['n']},"
                        f"{x:.6f},{y:.6f},{z:.6f},1,0,0,0,3,3,3\n")
        # detections_per_frame fuer object_eval-Zeitfenster mitkopieren
        import shutil
        shutil.copy(os.path.join(a.rundir, 'detections_per_frame.csv'),
                    os.path.join(out, 'detections_per_frame.csv'))
        print(f"  stride n={n:2d}: {len(keep_kf)} KFs -> {len(dets)} dets -> "
              f"{len(objs)} Objekte -> {out}/objects_droid.csv")


if __name__ == '__main__':
    main()
