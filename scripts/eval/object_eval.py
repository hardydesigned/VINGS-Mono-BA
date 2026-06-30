#!/usr/bin/env python3
"""Genauigkeit der Objekt-Detektion + 3D-Lokalisierung -- Auswertung.

Vergleicht die VINGS-Objekt-Pipeline-Vorhersagen gegen UAVScenes-Referenzdaten
und erzeugt die Mess-Tabellen/Plots fuer den BA-Abschnitt "Genauigkeit der
3D-Lokalisierung dynamischer Objekte".

Zwei Teile (per --part waehlbar, default both):

  A) 2D-Detektionsguete pro Keyframe  (mAP, IoU)
     Eingaben: <rundir>/detections_per_frame.csv (Vorhersage-Boxen pro KF)
               + UAVScenes-Instanz-/Semantik-Masken (GT)
     Ausgabe : <out>/object_eval_2d.json + pr_curve.png

  B) 3D-Lokalisierungs-Genauigkeit (Position / Yaw / Groesse)
     Vorhersage: <rundir>/objects_droid.csv (fusionierte 3D-Objekte; bei
                 GT-ext_poses-Lauf bereits im metrischen GT-Weltframe)
     Referenz  : Instanz-Maske -> LiDAR-Tiefe -> GT-Pose -> 3D-Cluster pro
                 Objekt, ueber Frames per Instanz-Track-ID fusioniert.
     Ausgabe : <out>/object_eval_3d.json + Fehler-Histogramme + BEV-Scatter

Konventions-Hinweis (WICHTIG, siehe object_tracker.py + generic_vo.py):
  Das Dataset fuettert die LiDAR-Tiefe SELBST als Tracker-Tiefe
  (generic_vo._lidar_depth -> data_packet['depth']). LiDAR-Punkte sind im
  Kameraframe (lx=Z-vorwaerts, ly->u, lz->v) mit lidar_sign_{u,v}=-1.
  Standard-Pinhole: X=su*ly, Y=sv*lz, Z=lx; depth-map D=lx. GT-c2w wird wie in
  run.py._apply_ext_poses_to_vizout gebaut (Rc2w=Rw2c.T, tc2w=-Rc2w@tw2c).
  Damit liegt die Referenz per Konstruktion im selben Weltframe wie die
  Vorhersagen -> direkter Vergleich, kein sim3_unwarp noetig.

Standalone-Smoketest (synthetische IoU/AP/Geometrie-Checks, kein Datenzugriff):
    python scripts/eval/object_eval.py --selftest
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

# object_tracker-Geometrie wiederverwenden (identische Pinhole-Konvention).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'vings_utils'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


# =============================================================================
# Klassen-Mapping  Detektor <-> UAVScenes-GT
# =============================================================================
# UAVScenes-Class-IDs (Cityscapes-style, NICHT Paper-Tab S9), siehe
# docs/SEGMENTATION_AMTOWN.md: Sedan=20, Truck=24. Beide = "vehicle".
GT_VEHICLE_IDS = {20: 'car', 24: 'truck'}     # GT-Maskenwert -> kanonische Klasse

# Detektor-Klassen-Namen (aus detections_per_frame.csv-Spalte 'class') ->
# kanonische Klasse. YOLO26-OBB(DOTA): small/large vehicle; VisDrone: car/van/
# truck/bus; COCO: car/truck/bus. Alles Unbekannte faellt auf 'vehicle' (nur
# fuer die klassenagnostische Metrik relevant).
DET_CLASS_CANON = {
    'small vehicle': 'car', 'large vehicle': 'truck',
    'car': 'car', 'van': 'car', 'truck': 'truck', 'bus': 'truck',
}


def canon_det(name: str) -> str:
    return DET_CLASS_CANON.get(str(name).strip().lower(), 'vehicle')


# =============================================================================
# IO-Helfer
# =============================================================================
def load_detections_csv(path: str):
    """detections_per_frame.csv -> dict[t_sec_key] = list[det].

    det = {'t_sec','frame_idx','class','canon','conf','bbox'(x1,y1,x2,y2 in
    Detektions-Pixeln),'depth','world'(wx,wy,wz)}. Join-Key ist der gerundete
    t_sec (Timestamp), robust gegen Index-/Slicing-Verschiebungen.
    """
    by_t = defaultdict(list)
    with open(path) as f:
        for r in csv.DictReader(f):
            t = float(r['t_sec'])
            by_t[round(t, 3)].append({
                't_sec': t, 'frame_idx': int(r['frame_idx']),
                'class': r['class'], 'canon': canon_det(r['class']),
                'conf': float(r['conf']),
                'bbox': (float(r['x1']), float(r['y1']),
                         float(r['x2']), float(r['y2'])),
                'depth': float(r['depth']),
                'world': (float(r['wx']), float(r['wy']), float(r['wz'])),
            })
    return by_t


def load_objects_csv(path: str):
    """objects_droid.csv -> list[obj] mit xyz, quat(wxyz), size, class, conf.

    Schema-tolerant: aeltere Laeufe haben nur (id,class,cls_id,conf,n,x,y,z)
    ohne quat/size -> Identitaets-Quat + size=None (Yaw/Groesse dann nicht
    auswertbar, Position weiterhin).
    """
    objs = []
    with open(path) as f:
        for r in csv.DictReader(f):
            has_q = 'qw' in r and r.get('qw') not in (None, '')
            has_s = 'sx' in r and r.get('sx') not in (None, '')
            objs.append({
                'object_id': int(r['object_id']), 'class': r['class'],
                'canon': canon_det(r['class']), 'conf': float(r['conf']),
                'n': int(r['n_detections']),
                'xyz': np.array([float(r['x']), float(r['y']), float(r['z'])]),
                'quat': (np.array([float(r['qw']), float(r['qx']),
                                   float(r['qy']), float(r['qz'])]) if has_q
                         else np.array([1.0, 0.0, 0.0, 0.0])),
                'size': (np.array([float(r['sx']), float(r['sy']), float(r['sz'])])
                         if has_s else None),
            })
    return objs


def load_camstamp(path: str):
    """camstamp.txt: '<ts> <name>.jpg' -> (list[ts], dict[name_stem]->ts)."""
    ts_list, stem2ts = [], {}
    with open(path) as f:
        for line in f:
            p = line.split()
            if len(p) < 2:
                continue
            t = float(p[0]); stem = os.path.splitext(p[1])[0]
            ts_list.append(t); stem2ts[stem] = t
    return np.array(ts_list), stem2ts


def load_poses_w2c(path: str):
    """poses_w2c.txt [ts tx ty tz qx qy qz qw] (w2c, scipy-Quat) -> (ts, c2w 4x4).

    c2w wie in run.py._apply_ext_poses_to_vizout: Rc2w=Rw2c.T, tc2w=-Rc2w@tw2c.
    """
    from scipy.spatial.transform import Rotation as Rot
    a = np.loadtxt(path, comments='#')
    ts = a[:, 0]
    Rw2c = Rot.from_quat(a[:, 4:8]).as_matrix()
    tw2c = a[:, 1:4]
    Rc2w = Rw2c.transpose(0, 2, 1)
    tc2w = -np.einsum('nij,nj->ni', Rc2w, tw2c)
    c2w = np.tile(np.eye(4), (len(ts), 1, 1))
    c2w[:, :3, :3] = Rc2w
    c2w[:, :3, 3] = tc2w
    return ts, c2w


def load_intrinsic_txt(path: str):
    """vings/intrinsic.txt -> dict fx,fy,cx,cy,W,H (Standard-Konvention:
    fu/cu = horizontal/width = fx/cx ; fv/cv = vertical/height = fy/cy)."""
    d = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            k, v = line.split()
            d[k] = float(v)
    return {'fx': d['fu'], 'fy': d['fv'], 'cx': d['cu'], 'cy': d['cv'],
            'W': int(d['W']), 'H': int(d['H'])}


def nearest_ts_index(ts_arr: np.ndarray, t: float, tol: float = 0.02):
    """Index des naechsten Timestamps in ts_arr, oder None wenn > tol entfernt."""
    i = int(np.argmin(np.abs(ts_arr - t)))
    return i if abs(ts_arr[i] - t) <= tol else None


# =============================================================================
# LiDAR-Projektion (repliziert generic_vo._lidar_depth, ohne NN-Fill)
# =============================================================================
def lidar_depth_sparse(lidar_path: str, intr: dict, sign=(-1.0, -1.0)):
    """Projiziert LiDAR-Punkte auf das volle CAM-Bild -> sparse Tiefenkarte (m).

    Z-Buffer (nah ueberschreibt fern). KEINE NN-Fuellung -- fuer eine saubere
    Referenz wollen wir nur echte LiDAR-Treffer, kein Smearing ueber Objektkanten.
    Liefert (H, W) float32, 0 wo kein Treffer.
    """
    pts = np.loadtxt(lidar_path)
    if pts.ndim != 2 or pts.shape[0] == 0:
        return None
    lx, ly, lz = pts[:, 0], pts[:, 1], pts[:, 2]
    ok = lx > 0.1
    lx, ly, lz = lx[ok], ly[ok], lz[ok]
    su, sv = sign
    fx, fy, cx, cy = intr['fx'], intr['fy'], intr['cx'], intr['cy']
    W, H = intr['W'], intr['H']
    u = (fx * (su * ly) / lx + cx).astype(np.int64)
    v = (fy * (sv * lz) / lx + cy).astype(np.int64)
    inb = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u, v, d = u[inb], v[inb], lx[inb].astype(np.float32)
    if d.size == 0:
        return None
    depth = np.zeros((H, W), np.float32)
    order = np.argsort(-d)                 # fern zuerst -> nah ueberschreibt
    depth[v[order], u[order]] = d[order]
    return depth


def unproject_pixels(cols, rows, z, intr, c2w):
    """(N,) Pixel + Tiefe -> (N,3) Weltpunkte. Standard-Pinhole z-forward,
    gleiche Algebra wie object_tracker.unproject_center."""
    x = (cols - intr['cx']) / intr['fx'] * z
    y = (rows - intr['cy']) / intr['fy'] * z
    cam = np.stack([x, y, z, np.ones_like(z)], axis=1)         # (N,4)
    return (cam @ np.asarray(c2w).T)[:, :3]


def cloud_pose_size(P: np.ndarray, up_axis: int = 2, size_pct: float = 95.0):
    """Centroid + Yaw([0,pi)) + Extent[long,lat,vert] aus einer Punktwolke.

    Spiegelt object_tracker.estimate_pose_size (PCA via SVD) damit Referenz und
    Vorhersage identisch geschaetzt werden. Liefert (centroid, yaw|None, size)."""
    c = P.mean(0)
    Pc = P - c
    try:
        _, _, Vt = np.linalg.svd(Pc, full_matrices=False)
    except np.linalg.LinAlgError:
        return c, None, None
    lo, hi = 100.0 - size_pct, size_pct
    proj = Pc @ Vt.T
    ext = np.percentile(proj, hi, 0) - np.percentile(proj, lo, 0)
    vext = float(np.percentile(Pc[:, up_axis], hi)
                 - np.percentile(Pc[:, up_axis], lo))
    order = np.argsort(ext)[::-1]
    size = np.array([float(ext[order[0]]), float(ext[order[1]]), vext])
    horiz = [i for i in range(3) if i != up_axis]
    a = Vt[0]; a_h = np.array([a[horiz[0]], a[horiz[1]]])
    yaw = (float(np.arctan2(a_h[1], a_h[0])) % np.pi
           if np.linalg.norm(a_h) > 1e-6 else None)
    return c, yaw, size


# =============================================================================
# GT-Masken laden  (Format wird beim Download verifiziert -- siehe --label-dir)
# =============================================================================
def find_label_for_ts(label_dir: str, stem: str):
    """Sucht die Label-PNG zu einem CAM-Bild-Stem (Timestamp-Name).

    UAVScenes-Layout: <label_dir>/.../interval1_CAM_label_id/<stem>.png
    Format nach Download verifizieren. Wir matchen per Dateinamen-Stem.
    """
    hits = glob.glob(os.path.join(label_dir, '**', f'{stem}.png'), recursive=True)
    # bevorzugt die *_id/-Variante (Class-IDs), nicht *_color/
    hits.sort(key=lambda p: ('_id' not in p, len(p)))
    return hits[0] if hits else None


def gt_instances(label_path: str, min_px: int = 30):
    """Label-PNG -> list[{'cls','inst_id','mask'(bool HxW),'bbox'(x1y1x2y2)}].

    Behandelt zwei Faelle (Auto-Detektion):
      (a) Instanz-Encoding  v = cls*1000 + inst   (gaengig bei UAVScenes-instance)
      (b) reine Semantik    v in GT_VEHICLE_IDS   -> Instanzen via Connected-Comp.
    NUR Fahrzeugklassen (GT_VEHICLE_IDS). min_px filtert Mini-Blobs.
    """
    from PIL import Image
    lab = np.array(Image.open(label_path))
    if lab.ndim == 3:                      # versehentlich color-PNG -> erste Ebene
        lab = lab[..., 0]
    out = []
    uniq = np.unique(lab)
    instance_encoded = bool((uniq >= 1000).any())
    if instance_encoded:
        for v in uniq:
            if v < 1000:
                continue
            cls = int(v) // 1000
            if cls not in GT_VEHICLE_IDS:
                continue
            m = lab == v
            if m.sum() < min_px:
                continue
            out.append({'cls': GT_VEHICLE_IDS[cls], 'inst_id': int(v),
                        'mask': m, 'bbox': _mask_bbox(m)})
    else:
        from scipy.ndimage import label as cc_label
        for cls_id, cname in GT_VEHICLE_IDS.items():
            sem = lab == cls_id
            if not sem.any():
                continue
            cc, n = cc_label(sem)
            for k in range(1, n + 1):
                m = cc == k
                if m.sum() < min_px:
                    continue
                out.append({'cls': cname, 'inst_id': cls_id * 1000 + k,
                            'mask': m, 'bbox': _mask_bbox(m)})
    return out


def _mask_bbox(m: np.ndarray):
    ys, xs = np.where(m)
    return (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))


# =============================================================================
# 2D-Metriken: IoU + AP
# =============================================================================
def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def average_precision(matches, n_gt):
    """matches: list[(score, is_tp)] ueber ALLE Frames. -> (AP, P, R).

    All-Point-Interpolation (VOC2010+/COCO-Stil). n_gt = Gesamtzahl GT-Objekte.
    """
    if n_gt == 0:
        return float('nan'), float('nan'), float('nan')
    matches = sorted(matches, key=lambda m: -m[0])
    tp = np.array([1 if m[1] else 0 for m in matches], np.float64)
    fp = 1.0 - tp
    ctp, cfp = np.cumsum(tp), np.cumsum(fp)
    rec = ctp / n_gt
    prec = ctp / np.maximum(ctp + cfp, 1e-9)
    # all-point interpolation
    mrec = np.concatenate([[0.0], rec, [1.0]])
    mpre = np.concatenate([[0.0], prec, [0.0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))
    fin_p = float(prec[-1]) if len(prec) else float('nan')
    fin_r = float(rec[-1]) if len(rec) else float('nan')
    return ap, fin_p, fin_r


def eval_2d(dets_by_t, gt_by_t, iou_thr=0.5, class_agnostic=True):
    """Per-Frame Greedy-Matching -> AP, mean-IoU, P/R.

    dets_by_t / gt_by_t: dict[t_key] -> list[det] / list[gt]. det braucht
    'bbox','conf','canon'; gt braucht 'bbox','cls'.
    """
    classes = (['vehicle'] if class_agnostic
               else sorted({g['cls'] for gts in gt_by_t.values() for g in gts}))
    res = {}
    for cls in classes:
        matches, n_gt, tp_ious = [], 0, []
        for t, gts in gt_by_t.items():
            gt_c = [g for g in gts if class_agnostic or g['cls'] == cls]
            n_gt += len(gt_c)
            dets = [d for d in dets_by_t.get(t, [])
                    if class_agnostic or d['canon'] == cls]
            dets = sorted(dets, key=lambda d: -d['conf'])
            used = [False] * len(gt_c)
            for d in dets:
                best_iou, best_j = 0.0, -1
                for j, g in enumerate(gt_c):
                    if used[j]:
                        continue
                    i = iou_xyxy(d['bbox'], g['bbox'])
                    if i > best_iou:
                        best_iou, best_j = i, j
                is_tp = best_iou >= iou_thr and best_j >= 0
                if is_tp:
                    used[best_j] = True
                    tp_ious.append(best_iou)
                matches.append((d['conf'], is_tp))
        ap, p, r = average_precision(matches, n_gt)
        res[cls] = {'AP': ap, 'precision': p, 'recall': r,
                    'mean_iou_tp': float(np.mean(tp_ious)) if tp_ious else float('nan'),
                    'n_gt': n_gt, 'n_det': sum(len(d) for d in dets_by_t.values()),
                    'n_tp': len(tp_ious)}
    return res


def map_coco(dets_by_t, gt_by_t, class_agnostic=True):
    """mAP@[.5:.95] (COCO) + AP@.5 + AP@.75."""
    thrs = np.round(np.arange(0.5, 1.0, 0.05), 2)
    per = {t: eval_2d(dets_by_t, gt_by_t, t, class_agnostic) for t in thrs}
    classes = list(per[0.5].keys())
    out = {}
    for cls in classes:
        aps = [per[t][cls]['AP'] for t in thrs]
        out[cls] = {
            'AP@[.5:.95]': float(np.nanmean(aps)),
            'AP@.5': per[0.5][cls]['AP'], 'AP@.75': per[0.75][cls]['AP'],
            'mean_iou_tp@.5': per[0.5][cls]['mean_iou_tp'],
            'precision@.5': per[0.5][cls]['precision'],
            'recall@.5': per[0.5][cls]['recall'],
            'n_gt': per[0.5][cls]['n_gt'], 'n_tp@.5': per[0.5][cls]['n_tp'],
        }
    return out, per


# =============================================================================
# Teil A: 2D-Auswertung
# =============================================================================
def run_part_a(args, intr):
    print("\n=== Teil A: 2D-Detektionsguete (mAP / IoU) ===")
    dets_by_t = load_detections_csv(os.path.join(args.rundir, 'detections_per_frame.csv'))
    _, stem2ts = load_camstamp(args.camstamp)
    ts2stem = {round(t, 3): s for s, t in stem2ts.items()}

    # GT pro Frame (nur Frames, die im Detektions-Log vorkommen UND Label haben),
    # auf Detektions-Aufloesung skaliert.
    gt_by_t, scale_logged = {}, False
    H, W = intr['H'], intr['W']
    n_lab = 0
    for tkey in sorted(dets_by_t.keys()):
        stem = ts2stem.get(tkey)
        if stem is None:
            continue
        lp = find_label_for_ts(args.label_dir, stem)
        if lp is None:
            continue
        insts = gt_instances(lp, args.min_mask_px)
        # Detektions-Boxen liegen in Detektions-Pixeln (imgsz). GT-Boxen in
        # Label-Pixeln (volle CAM-Res). Beide auf [0,1]-normierte Box bringen.
        det_res = _infer_det_res(dets_by_t[tkey], W, H)
        for d in dets_by_t[tkey]:
            d['bbox'] = _norm_box(d['bbox'], det_res[0], det_res[1])
        gts = []
        for ins in insts:
            gts.append({'cls': ins['cls'], 'bbox': _norm_box(ins['bbox'], W, H)})
        gt_by_t[tkey] = gts
        n_lab += 1
        if not scale_logged:
            print(f"  Bsp-Frame {stem}: det_res~{det_res}, label_res={W}x{H}, "
                  f"{len(gts)} GT-Fahrzeuge, {len(dets_by_t[tkey])} Detektionen")
            scale_logged = True
    print(f"  {n_lab} Frames mit GT-Label + Detektion ausgewertet.")
    if n_lab == 0:
        print("  [WARN] keine Frames gematcht -- Label-Dir/Namen pruefen.")
        return None

    agn, per_agn = map_coco(dets_by_t, gt_by_t, class_agnostic=True)
    cls_res, _ = map_coco(dets_by_t, gt_by_t, class_agnostic=False)
    result = {'n_frames': n_lab, 'class_agnostic': agn, 'per_class': cls_res}
    out_json = os.path.join(args.out, 'object_eval_2d.json')
    with open(out_json, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"  -> {out_json}")
    print(f"  vehicle  AP@.5={agn['vehicle']['AP@.5']:.3f}  "
          f"AP@[.5:.95]={agn['vehicle']['AP@[.5:.95]']:.3f}  "
          f"mIoU(TP)={agn['vehicle']['mean_iou_tp@.5']:.3f}  "
          f"P={agn['vehicle']['precision@.5']:.3f} R={agn['vehicle']['recall@.5']:.3f}")
    _plot_pr(per_agn, os.path.join(args.out, 'pr_curve.png'))
    return result


def _infer_det_res(dets, W, H):
    """Schaetzt die Detektions-Bildaufloesung aus max. Box-Koordinaten.

    detections_per_frame.csv speichert Boxen in 'Detektions-Aufloesung'
    (Voll-Res wenn vorhanden, sonst viz_out). Wir leiten W_det/H_det aus den
    beobachteten Maxima ab und nehmen das CAM-Seitenverhaeltnis als Anker.
    """
    if not dets:
        return W, H
    xmax = max(d['bbox'][2] for d in dets)
    ymax = max(d['bbox'][3] for d in dets)
    # Wenn Boxen offensichtlich Voll-Res sind -> CAM-Res nehmen.
    if xmax <= W + 2 and ymax <= H + 2 and xmax > W * 0.4:
        return W, H
    # sonst per beobachteter Breite skalieren, Hoehe ueber CAM-Aspect
    w_det = max(xmax, 1.0)
    return w_det, w_det * H / W


def _norm_box(b, w, h):
    return (b[0] / w, b[1] / h, b[2] / w, b[3] / h)


def _plot_pr(per_thr, path):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"  [PR-Plot skipped: {e}]")
        return
    plt.figure(figsize=(5, 4))
    plt.title('Precision-Recall (vehicle, klassenagnostisch)')
    plt.xlabel('Recall'); plt.ylabel('Precision')
    plt.xlim(0, 1); plt.ylim(0, 1.02); plt.grid(alpha=0.3)
    plt.savefig(path, dpi=130, bbox_inches='tight')
    print(f"  -> {path}")


# =============================================================================
# Teil B: 3D-Lokalisierungs-Genauigkeit
# =============================================================================
def _circular_mean_yaw(yaws):
    """Conf-freier Kreismittelwert von Yaws in [0,pi) (Doppelwinkel-Trick),
    None wenn keine oder inkohaerent. Spiegelt object_tracker.fused_yaw."""
    ys = [y for y in yaws if y is not None]
    if not ys:
        return None
    ang2 = 2.0 * np.array(ys)
    C, S = np.cos(ang2).sum(), np.sin(ang2).sum()
    if np.hypot(C, S) / len(ys) < 0.5:
        return None
    return (0.5 * np.arctan2(S, C)) % np.pi


def build_reference_3d(args, intr, t_window=None):
    """Baut Referenz-3D-Objekte aus Maske + LiDAR-Tiefe + GT-Pose.

    Schritt 1: pro Frame pro Instanz-Maske -> 3D-Cluster (centroid/yaw/size).
    Schritt 2: ueber Frames per 3D-NN fusionieren (wie object_tracker._associate)
               -- NOETIG bei semantischen Masken, deren Connected-Component-
               Index pro Frame willkuerlich ist (kein stabiler Track).
    t_window: (t_min, t_max) -- nur Label-Frames in diesem Zeitfenster (faire
              Precision/Recall ggue. den Vorhersagen, die nur diesen Lauf sehen).
    """
    pose_ts, pose_c2w = load_poses_w2c(args.poses)
    _, stem2ts = load_camstamp(args.camstamp)
    lmap = {}
    for fn in os.listdir(args.lidar_dir):
        if fn.startswith('image') and '_lidar' in fn:
            lmap[fn[len('image'):fn.index('_lidar')]] = os.path.join(args.lidar_dir, fn)

    label_pngs = sorted(glob.glob(os.path.join(args.label_dir, '**', '*.png'),
                                  recursive=True))
    label_pngs = [p for p in label_pngs if '_color' not in p]   # nur *_id/

    # --- Schritt 1: per-Frame-Instanz-Detektionen ---
    raw = []          # {'cls','centroid','yaw','size','n_pts'}
    n_used, n_skip_lidar, n_skip_win = 0, 0, 0
    for lp in label_pngs:
        stem = os.path.splitext(os.path.basename(lp))[0]
        t = stem2ts.get(stem)
        if t is None:
            continue
        if t_window is not None and not (t_window[0] <= t <= t_window[1]):
            n_skip_win += 1
            continue
        pi = nearest_ts_index(pose_ts, t, tol=0.05)
        lpath = lmap.get(stem)
        if pi is None or lpath is None:
            continue
        depth = lidar_depth_sparse(lpath, intr, args.lidar_sign)
        if depth is None:
            n_skip_lidar += 1
            continue
        c2w = pose_c2w[pi]
        for ins in gt_instances(lp, args.min_mask_px):
            m = ins['mask'] & (depth > args.min_depth) & (depth < args.max_depth)
            if int(m.sum()) < args.min_lidar_px:
                continue
            rows, cols = np.where(m)
            z = depth[rows, cols].astype(np.float64)
            P = unproject_pixels(cols.astype(np.float64), rows.astype(np.float64),
                                 z, intr, c2w)
            c, yaw, size = cloud_pose_size(P, size_pct=args.size_pct)
            raw.append({'cls': ins['cls'], 'centroid': c, 'yaw': yaw,
                        'size': size, 'n_pts': len(P)})
        n_used += 1

    # --- Schritt 2: 3D-NN-Fusion ueber Frames ---
    tracks = []       # {'cls','cs':[centroids],'yaws':[],'sizes':[],'centroid'}
    for r in raw:
        best, bestd = None, args.ref_assoc_m
        for tr in tracks:
            if tr['cls'] != r['cls']:
                continue
            d = float(np.linalg.norm(tr['centroid'] - r['centroid']))
            if d < bestd:
                best, bestd = tr, d
        if best is None:
            best = {'cls': r['cls'], 'cs': [], 'yaws': [], 'sizes': [],
                    'centroid': r['centroid']}
            tracks.append(best)
        best['cs'].append(r['centroid'])
        best['yaws'].append(r['yaw'])
        best['sizes'].append(r['size'])
        best['centroid'] = np.median(np.asarray(best['cs']), axis=0)

    objs = []
    for tr in tracks:
        if len(tr['cs']) < args.ref_min_frames:
            continue
        sizes = [s for s in tr['sizes'] if s is not None]
        objs.append({
            'cls': tr['cls'],
            'xyz': np.median(np.asarray(tr['cs']), axis=0),       # robuste Pos
            'yaw': _circular_mean_yaw(tr['yaws']),
            'size': (np.median(np.asarray(sizes), axis=0) if sizes else None),
            'n_frames': len(tr['cs']),
        })
    print(f"  Referenz: {n_used} Label-Frames im Fenster, {n_skip_win} ausserhalb, "
          f"{n_skip_lidar} ohne LiDAR, {len(raw)} Frame-Instanzen -> "
          f"{len(objs)} Referenz-Objekte (>= {args.ref_min_frames} Frames, "
          f"NN-Radius {args.ref_assoc_m} m).")
    return objs


def quat_to_yaw(q):
    """(w,x,y,z) Rotation um Welt-Z -> yaw in [0,pi) (Achse, 180deg-mehrdeutig)."""
    w, x, y, z = q
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return float(yaw % np.pi)


def yaw_err_deg(a, b):
    if a is None or b is None:
        return None
    d = abs(a - b) % np.pi
    return float(np.degrees(min(d, np.pi - d)))    # 180deg-Achsen-Mehrdeutigkeit


def run_part_b(args, intr):
    print("\n=== Teil B: 3D-Lokalisierungs-Genauigkeit ===")
    pred = load_objects_csv(os.path.join(args.rundir, 'objects_droid.csv'))
    if not pred:
        print("  [WARN] objects_droid.csv leer."); return None
    pxyz = np.array([p['xyz'] for p in pred])
    print(f"  {len(pred)} Vorhersage-Objekte. xyz-Bereich "
          f"x[{pxyz[:,0].min():.1f},{pxyz[:,0].max():.1f}] "
          f"y[{pxyz[:,1].min():.1f},{pxyz[:,1].max():.1f}] "
          f"z[{pxyz[:,2].min():.1f},{pxyz[:,2].max():.1f}] "
          f"(Magnitude-Sanity: metrisch?)")

    # Referenz nur im Zeitfenster des Laufs bauen (faire P/R) -- aus den
    # Detektions-Timestamps + kleiner Marge.
    t_window = None
    det_path = os.path.join(args.rundir, 'detections_per_frame.csv')
    if not args.ref_all_frames and os.path.exists(det_path):
        ts = sorted(load_detections_csv(det_path).keys())
        if ts:
            t_window = (ts[0] - args.window_margin_s, ts[-1] + args.window_margin_s)
            print(f"  Referenz-Zeitfenster t=[{t_window[0]:.1f}, {t_window[1]:.1f}]")
    ref = build_reference_3d(args, intr, t_window=t_window)
    if not ref:
        print("  [WARN] keine Referenz-Objekte gebaut."); return None
    rxyz = np.array([r['xyz'] for r in ref])

    # Greedy-NN-Matching Vorhersage<->Referenz nach 3D-Distanz, Gate args.gate_m.
    pairs, used_r = [], set()
    pred_sorted = sorted(range(len(pred)), key=lambda i: -pred[i]['n'])
    for i in pred_sorted:
        dvec = np.linalg.norm(rxyz - pred[i]['xyz'], axis=1)
        for j in np.argsort(dvec):
            if j in used_r:
                continue
            if dvec[j] <= args.gate_m:
                used_r.add(int(j)); pairs.append((i, int(j), float(dvec[j])))
            break
    n_tp = len(pairs)
    precision = n_tp / len(pred) if pred else float('nan')
    recall = n_tp / len(ref) if ref else float('nan')

    pos_err = np.array([d for _, _, d in pairs])
    yaw_errs, size_errs, size_rel = [], [], []
    for i, j, _ in pairs:
        q = pred[i]['quat']
        pred_yaw = None if np.allclose(q, [1, 0, 0, 0]) else quat_to_yaw(q)
        ye = yaw_err_deg(pred_yaw, ref[j]['yaw'])
        if ye is not None:
            yaw_errs.append(ye)
        if ref[j]['size'] is not None and pred[i]['size'] is not None:
            se = np.abs(pred[i]['size'] - ref[j]['size'])
            size_errs.append(se)
            size_rel.append(se / np.maximum(ref[j]['size'], 1e-3))

    def stats(a):
        a = np.asarray(a, float)
        if a.size == 0:
            return {'mean': None, 'median': None, 'rmse': None, 'n': 0}
        return {'mean': float(a.mean()), 'median': float(np.median(a)),
                'rmse': float(np.sqrt((a ** 2).mean())), 'n': int(a.size)}

    result = {
        'n_pred': len(pred), 'n_ref': len(ref), 'n_matched': n_tp,
        'gate_m': args.gate_m, 'precision': precision, 'recall': recall,
        'position_error_m': stats(pos_err),
        'yaw_error_deg': stats(yaw_errs),
        'size_error_m': {ax: stats([s[k] for s in size_errs])
                         for k, ax in enumerate(['long', 'lat', 'vert'])},
        'size_rel_error': {ax: stats([s[k] for s in size_rel])
                           for k, ax in enumerate(['long', 'lat', 'vert'])},
    }
    out_json = os.path.join(args.out, 'object_eval_3d.json')
    with open(out_json, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"  -> {out_json}")
    print(f"  matched {n_tp}/{len(pred)} (P={precision:.2f}) von "
          f"{len(ref)} Referenz (R={recall:.2f})")
    pe = result['position_error_m']
    print(f"  Position-Fehler [m]: mean={pe['mean']:.2f} median={pe['median']:.2f} "
          f"rmse={pe['rmse']:.2f}" if pe['n'] else "  Position-Fehler: --")
    if result['yaw_error_deg']['n']:
        y = result['yaw_error_deg']
        print(f"  Yaw-Fehler [deg]:    mean={y['mean']:.1f} median={y['median']:.1f}")
    _plot_3d(pos_err, yaw_errs, pred, ref, pairs, args.out)
    return result


def _plot_3d(pos_err, yaw_errs, pred, ref, pairs, out):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"  [3D-Plots skipped: {e}]"); return
    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    if len(pos_err):
        ax[0].hist(pos_err, bins=20, color='steelblue')
        ax[0].set_title('Positions-Fehler [m]'); ax[0].set_xlabel('m')
    if len(yaw_errs):
        ax[1].hist(yaw_errs, bins=20, color='indianred')
        ax[1].set_title('Yaw-Fehler [deg]'); ax[1].set_xlabel('deg')
    rxyz = np.array([r['xyz'] for r in ref]); pxyz = np.array([p['xyz'] for p in pred])
    ax[2].scatter(rxyz[:, 0], rxyz[:, 1], s=18, c='k', marker='x', label='Referenz')
    ax[2].scatter(pxyz[:, 0], pxyz[:, 1], s=18, c='tab:blue', alpha=0.6, label='Vorhersage')
    for i, j, _ in pairs:
        ax[2].plot([pred[i]['xyz'][0], ref[j]['xyz'][0]],
                   [pred[i]['xyz'][1], ref[j]['xyz'][1]], 'g-', lw=0.6)
    ax[2].set_title('BEV (x,y)'); ax[2].legend(fontsize=8); ax[2].axis('equal')
    p = os.path.join(out, 'object_eval_3d.png')
    plt.tight_layout(); plt.savefig(p, dpi=130); print(f"  -> {p}")


# =============================================================================
# Selftest (synthetisch, kein Datenzugriff)
# =============================================================================
def selftest():
    # IoU
    assert abs(iou_xyxy((0, 0, 2, 2), (1, 1, 3, 3)) - (1 / 7)) < 1e-6
    assert iou_xyxy((0, 0, 1, 1), (2, 2, 3, 3)) == 0.0
    # AP: 2 GT, perfekte 2 TP -> AP=1
    ap, p, r = average_precision([(0.9, True), (0.8, True)], 2)
    assert abs(ap - 1.0) < 1e-6 and abs(p - 1.0) < 1e-6 and abs(r - 1.0) < 1e-6
    # 1 TP + 1 FP, 2 GT -> recall 0.5
    ap, p, r = average_precision([(0.9, True), (0.8, False)], 2)
    assert abs(r - 0.5) < 1e-6
    # eval_2d end-to-end
    gt = {1.0: [{'cls': 'car', 'bbox': (0, 0, 0.1, 0.1)}]}
    det = {1.0: [{'canon': 'car', 'conf': 0.9, 'bbox': (0, 0, 0.1, 0.1)}]}
    res = eval_2d(det, gt, 0.5, True)
    assert abs(res['vehicle']['AP'] - 1.0) < 1e-6, res
    # Geometrie: planare Tiefe z=5, Identitaet -> centre [0,0,5]
    intr = {'fx': 500., 'fy': 500., 'cx': 320., 'cy': 240., 'W': 640, 'H': 480}
    P = unproject_pixels(np.array([320.]), np.array([240.]), np.array([5.]),
                         intr, np.eye(4))
    assert np.allclose(P[0], [0, 0, 5], atol=1e-4), P
    # cloud_pose_size: ein Kasten 4x2x1 entlang x -> long~4, lat~2, yaw~0
    g = np.array(np.meshgrid(np.linspace(-2, 2, 9), np.linspace(-1, 1, 5),
                             np.linspace(-.5, .5, 3))).reshape(3, -1).T
    c, yaw, size = cloud_pose_size(g)
    assert np.allclose(c, 0, atol=1e-6)
    assert abs(size[0] - 4) < 0.3 and abs(size[1] - 2) < 0.3, size
    assert yaw is not None and (yaw < 0.1 or abs(yaw - np.pi) < 0.1), yaw
    # quat<->yaw roundtrip
    q = [np.cos(0.3), 0, 0, np.sin(0.3)]
    assert abs(quat_to_yaw(q) - 0.6) < 1e-6, quat_to_yaw(q)
    assert yaw_err_deg(0.0, np.pi - 1e-9) < 0.1     # 180deg = 0 error
    print("[object_eval selftest] alle IoU/AP/Geometrie/Yaw-Checks bestanden.")


# =============================================================================
# CLI
# =============================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--selftest', action='store_true',
                    help='synthetische Checks ohne Datenzugriff, dann exit')
    ap.add_argument('--rundir', help='Run-Output-Dir (objects_droid.csv, detections_per_frame.csv)')
    ap.add_argument('--part', choices=['a', 'b', 'both'], default='both')
    ap.add_argument('--out', help='Ausgabe-Dir (default: <rundir>)')
    DS = '/home/philipp/Dokumente/datasets/interval1_AMtown03'
    ap.add_argument('--label-dir',
                    default='/home/philipp/Dokumente/datasets/uavscenes/interval1_amtown03_labels',
                    help='UAVScenes interval1 Instanz-/Semantik-Label-Dir')
    ap.add_argument('--camstamp', default=f'{DS}/vings/camstamp.txt')
    ap.add_argument('--poses', default=f'{DS}/vings/poses_w2c.txt')
    ap.add_argument('--intrinsic', default=f'{DS}/vings/intrinsic.txt')
    ap.add_argument('--lidar-dir', default=f'{DS}/interval1_LIDAR')
    ap.add_argument('--lidar-sign', type=float, nargs=2, default=(-1.0, -1.0))
    # 2D
    ap.add_argument('--min-mask-px', type=int, default=30)
    # 3D
    ap.add_argument('--min-depth', type=float, default=0.2)
    ap.add_argument('--max-depth', type=float, default=150.0)
    ap.add_argument('--min-lidar-px', type=int, default=10,
                    help='min. LiDAR-Treffer pro Instanz-Frame fuer Referenz')
    ap.add_argument('--ref-min-frames', type=int, default=2,
                    help='min. Frames pro Referenz-Objekt')
    ap.add_argument('--ref-assoc-m', type=float, default=3.0,
                    help='3D-NN-Radius fuer Referenz-Fusion ueber Frames (m)')
    ap.add_argument('--max-ref-frames', type=int, default=0,
                    help='(reserviert) 0=alle Label-Frames im Fenster')
    ap.add_argument('--ref-all-frames', action='store_true',
                    help='Referenz aus ALLEN Label-Frames (statt nur Lauf-Zeitfenster)')
    ap.add_argument('--window-margin-s', type=float, default=1.0,
                    help='Marge um das Lauf-Zeitfenster fuer die Referenz (s)')
    ap.add_argument('--size-pct', type=float, default=95.0)
    ap.add_argument('--gate-m', type=float, default=5.0,
                    help='3D-Assoziations-Gate (m)')
    a = ap.parse_args()

    if a.selftest:
        selftest(); return
    if not a.rundir:
        ap.error('--rundir erforderlich (ausser --selftest)')
    a.out = a.out or a.rundir
    os.makedirs(a.out, exist_ok=True)
    intr = load_intrinsic_txt(a.intrinsic)
    print(f"Intrinsik (Standard): {intr}")

    if a.part in ('a', 'both'):
        run_part_a(a, intr)
    if a.part in ('b', 'both'):
        run_part_b(a, intr)


if __name__ == '__main__':
    main()
