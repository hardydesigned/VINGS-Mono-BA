"""
Online 3D object localisation + fusion.

Consumes per-keyframe detections (from a detector behind `ObjectDetectorBase`)
together with the keyframe's depth map, c2w pose and intrinsics, and maintains a
running set of object *tracks* in the gauge-free DROID world frame. At run end
it writes one fused 3D position per object.

Pipeline per keyframe (called from `scripts/run.py`, right after segmentation):

    tracker.update(dets, depth, pose_c2w, intrinsic, f_idx, rgb_u8=...)

and once at the end:

    tracker.finalize(save_dir)
      -> objects_droid.csv          (id, class, conf, n_hits, x, y, z)
      -> object_markers_droid.ply   (class-coloured markers, SAME frame as the map PLY)
      -> object_overlay.mp4         (2D boxes per mapped KF; optional)

Coordinate convention -- IMPORTANT
-----------------------------------
Object points must land on the reconstructed map PLY. The mapper builds that map
via `gaussian/tf.py`, which uses the `viz_out['intrinsic']` dict where (see
`middleware_utils.judge_and_package`):  fu = f_y, fv = f_x, cu = c_y, cv = c_x.
So the standard pinhole back-projection in tf.py is:

    X_cam = (col - cv) / fv * z       # = (col - cx) / fx * z
    Y_cam = (row - cu) / fu * z       # = (row - cy) / fy * z
    Z_cam = z
    p_world = (c2w @ [X, Y, Z, 1])[:3]

We replicate exactly that (NOT the selector K, whose naming convention may
differ) so markers are consistent with the map. bbox centres come in OpenCV
order (col = x, row = y), the same order ultralytics returns.

Metric / GPS coordinates come later: the markers live in the same DROID frame as
the map PLY, so `scripts/eval/sim3_unwarp.py --gps-csv ...` transforms them 1:1.

Standalone smoketest (back-projection axis sanity check):

    python scripts/vings_utils/object_tracker.py
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np


# =============================================================================
# Geometry helpers
# =============================================================================

def unproject_center(col: float, row: float, z: float,
                     intr: dict, c2w: np.ndarray) -> np.ndarray:
    """Back-project one pixel (col, row) at depth z to a DROID-world point.

    `intr` is the viz_out intrinsic dict (fu=fy, fv=fx, cu=cy, cv=cx); see the
    module docstring. Mirrors gaussian/tf.py pixel->cam->world exactly.
    Intrinsic values may be torch tensors (cuda) -> coerce to float.
    """
    fu, fv = float(intr['fu']), float(intr['fv'])
    cu, cv = float(intr['cu']), float(intr['cv'])
    x_cam = (col - cv) / fv * z
    y_cam = (row - cu) / fu * z
    p_cam = np.array([x_cam, y_cam, z, 1.0], dtype=np.float64)
    return (np.asarray(c2w, dtype=np.float64) @ p_cam)[:3]


def sample_box_depth(depth: np.ndarray, bbox_xyxy, box_shrink: float,
                     depth_percentile: float, min_d: float, max_d: float,
                     min_valid_px: int) -> float | None:
    """Robust scalar depth for a bbox.

    Shrinks the box to its central window, keeps only valid depths
    (finite, in (min_d, max_d)), and returns the `depth_percentile`-th
    percentile (a nearer value, so the object's depth wins over background
    behind it). Falls back to the full box, then gives up (None) if too few
    valid pixels.
    """
    H, W = depth.shape
    x1, y1, x2, y2 = bbox_xyxy
    cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
    hw, hh = 0.5 * (x2 - x1) * box_shrink, 0.5 * (y2 - y1) * box_shrink

    def _gather(half_w, half_h):
        c0 = max(0, int(round(cx - half_w))); c1 = min(W, int(round(cx + half_w)) + 1)
        r0 = max(0, int(round(cy - half_h))); r1 = min(H, int(round(cy + half_h)) + 1)
        if c1 <= c0 or r1 <= r0:
            return np.empty(0, np.float32)
        patch = depth[r0:r1, c0:c1].reshape(-1)
        return patch[np.isfinite(patch) & (patch > min_d) & (patch < max_d)]

    valid = _gather(hw, hh)
    if valid.size < min_valid_px:                       # retry on the full box
        valid = _gather(0.5 * (x2 - x1), 0.5 * (y2 - y1))
    if valid.size < min_valid_px:
        return None
    return float(np.percentile(valid, depth_percentile))


def _weighted_median(vals: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(vals)
    v, w = vals[order], weights[order]
    cw = np.cumsum(w)
    cutoff = 0.5 * w.sum()
    return float(v[min(int(np.searchsorted(cw, cutoff)), len(v) - 1)])


# =============================================================================
# Track
# =============================================================================

@dataclass
class _Track:
    cls_id: int
    cls_name: str
    tid: int = -1                                 # stable track id (for det-log linkage)
    pts: list = field(default_factory=list)      # list of (3,) world points
    confs: list = field(default_factory=list)
    cls_ids: list = field(default_factory=list)  # for class-agnostic majority vote
    _sum: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def add(self, p: np.ndarray, conf: float, cls_id: int, cls_name: str):
        if not self.confs or conf > max(self.confs):  # keep most-confident label
            self.cls_id, self.cls_name = cls_id, cls_name
        self.pts.append(p)
        self.confs.append(conf)
        self.cls_ids.append(cls_id)
        self._sum += p

    @property
    def centroid(self) -> np.ndarray:
        return self._sum / max(1, len(self.pts))

    @property
    def conf(self) -> float:
        return max(self.confs) if self.confs else 0.0

    @property
    def n_hits(self) -> int:
        return len(self.pts)

    def fused_position(self) -> np.ndarray:
        pts = np.asarray(self.pts)
        w = np.asarray(self.confs, dtype=np.float64)
        if w.sum() <= 0:
            w = np.ones_like(w)
        return np.array([_weighted_median(pts[:, k], w) for k in range(3)])

    def majority_cls(self) -> tuple[int, str]:
        ids, counts = np.unique(self.cls_ids, return_counts=True)
        return int(ids[counts.argmax()]), self.cls_name


# =============================================================================
# Tracker
# =============================================================================

class ObjectTracker:
    """Incremental nearest-neighbour fusion of per-keyframe detections."""

    def __init__(self, cfg: dict, save_dir: str | None = None):
        det = (cfg.get('object_detector') or {})
        trk = (cfg.get('object_tracker') or {})
        out = (cfg.get('object_output') or {})

        # depth-sampling (shared defaults with the detector block)
        self.box_shrink = float(det.get('box_shrink', 0.5))
        self.depth_percentile = float(det.get('depth_percentile', 30.0))
        self.min_valid_px = int(det.get('min_valid_px', 10))
        self.min_depth = float(det.get('min_depth', 0.2))
        self.max_depth = float(det.get('max_depth', 60.0))

        # fusion
        self.assoc_radius = float(trk.get('assoc_radius', 0.05))   # DROID-frame units
        self.min_hits = int(trk.get('min_hits', 3))
        self.class_agnostic = bool(trk.get('class_agnostic', False))
        self.marker_radius = float(trk.get('marker_radius', self.assoc_radius))

        # output toggles
        self.want_csv = bool(out.get('csv', True))
        self.want_ply = bool(out.get('markers_ply', True))
        self.want_video = bool(out.get('overlay_video', True))
        self.overlay_fps = int(out.get('overlay_fps', 10))
        self.overlay_max_w = int(out.get('overlay_max_w', 1600))  # downscale for video size
        self.want_det_csv = bool(out.get('detections_csv', True))  # per-frame detection log

        self.save_dir = save_dir or (cfg.get('output') or {}).get('save_dir')
        self.overlay_dir = (os.path.join(self.save_dir, 'object_overlays')
                            if self.save_dir else None)

        self.tracks: list[_Track] = []
        self._n_dets = 0
        self._next_tid = 0
        self._kf_count = 0
        self._overlay_paths: list[str] = []
        self._det_log: list[dict] = []           # one row per detection per KF (with time)

    # ------------------------------------------------------------------
    def update(self, dets, depth, pose_c2w, intrinsic, f_idx, rgb_u8=None,
               det_hw=None, t_sec=None):
        """Fold one keyframe's detections into the running tracks.

        Args:
            dets: list[Detection] from the detector, boxes in detection-image
                pixel coords.
            depth: (H, W) numpy depth map (DROID units; 0/invalid filtered).
            pose_c2w: (4, 4) camera-to-world (DROID frame).
            intrinsic: viz_out intrinsic dict {fu,fv,cu,cv,H,W} at depth res.
            f_idx: global frame index (for overlay filenames + det-log).
            rgb_u8: optional (H, W, 3) uint8 RGB for the overlay image
                (detection resolution).
            det_hw: (H_det, W_det) of the detection image. If it differs from
                the depth map, bbox coords are scaled to depth resolution for
                geometry (the overlay keeps the original detection-res boxes).
            t_sec: optional timestamp of this keyframe (Unix epoch). Logged in
                the per-frame detection CSV so detections carry the time axis.
        """
        depth = np.asarray(depth, dtype=np.float32)
        c2w = np.asarray(pose_c2w, dtype=np.float64)
        Hd, Wd = depth.shape
        if det_hw is not None and (int(det_hw[0]) != Hd or int(det_hw[1]) != Wd):
            sx, sy = Wd / float(det_hw[1]), Hd / float(det_hw[0])
        else:
            sx = sy = 1.0
        kf = self._kf_count
        self._kf_count += 1
        kept = []
        for d in dets:
            self._n_dets += 1
            x1, y1, x2, y2 = d.bbox_xyxy
            box_d = (x1 * sx, y1 * sy, x2 * sx, y2 * sy)   # -> depth resolution
            z = sample_box_depth(depth, box_d, self.box_shrink,
                                 self.depth_percentile, self.min_depth,
                                 self.max_depth, self.min_valid_px)
            row = {'frame_idx': int(f_idx), 't_sec': (float(t_sec) if t_sec is not None else float('nan')),
                   'kf': kf, 'class': d.cls_name, 'cls_id': int(d.cls_id),
                   'conf': float(d.conf), 'x1': float(x1), 'y1': float(y1),
                   'x2': float(x2), 'y2': float(y2),
                   'depth': float('nan'), 'wx': float('nan'),
                   'wy': float('nan'), 'wz': float('nan'), 'tid': -1}
            if z is not None:
                col = 0.5 * (box_d[0] + box_d[2])
                rw = 0.5 * (box_d[1] + box_d[3])
                p = unproject_center(col, rw, z, intrinsic, c2w)
                tr = self._associate(p, d.conf, d.cls_id, d.cls_name)
                row.update(depth=float(z), wx=float(p[0]), wy=float(p[1]),
                           wz=float(p[2]), tid=tr.tid)
                kept.append((d, z))
            self._det_log.append(row)

        if self.want_video and rgb_u8 is not None:
            self._save_overlay(rgb_u8, kept, f_idx)

    def _associate(self, p, conf, cls_id, cls_name) -> _Track:
        best, best_d = None, self.assoc_radius
        for tr in self.tracks:
            if not self.class_agnostic and tr.cls_id != cls_id:
                continue
            dist = float(np.linalg.norm(tr.centroid - p))
            if dist < best_d:
                best, best_d = tr, dist
        if best is None:
            best = _Track(cls_id=cls_id, cls_name=cls_name, tid=self._next_tid)
            self._next_tid += 1
            self.tracks.append(best)
        best.add(p, conf, cls_id, cls_name)
        return best

    # ------------------------------------------------------------------
    def _save_overlay(self, rgb_u8, kept, f_idx):
        try:
            import cv2
            from vings_utils.detector_base import class_color
        except ImportError:
            try:
                import cv2
                from detector_base import class_color
            except Exception:
                return
        except Exception:
            return
        if self.overlay_dir is None:
            return
        os.makedirs(self.overlay_dir, exist_ok=True)
        img = np.ascontiguousarray(rgb_u8[..., ::-1])  # RGB -> BGR
        h, w = img.shape[:2]
        th = max(1, int(round(2 * w / 1280)))           # scale line/text with res
        for d, z in kept:
            x1, y1, x2, y2 = [int(v) for v in d.bbox_xyxy]
            col = class_color(d.cls_id)[::-1]
            cv2.rectangle(img, (x1, y1), (x2, y2), col, th)
            cv2.putText(img, f"{d.cls_name} {d.conf:.2f} d={z:.1f}",
                        (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5 * th, col, th)
        if w > self.overlay_max_w:                       # downscale for video size
            s = self.overlay_max_w / float(w)
            img = cv2.resize(img, (self.overlay_max_w, int(round(h * s))))
        path = os.path.join(self.overlay_dir, f"idx={int(f_idx):06d}.png")
        cv2.imwrite(path, img)
        self._overlay_paths.append(path)

    # ------------------------------------------------------------------
    def finalize(self, save_dir: str | None = None):
        """Write CSV / marker-PLY / overlay-video. Returns the kept objects."""
        save_dir = save_dir or self.save_dir
        objects = []
        for tr in self.tracks:
            if tr.n_hits < self.min_hits:
                continue
            cls_id, cls_name = (tr.majority_cls() if self.class_agnostic
                                else (tr.cls_id, tr.cls_name))
            objects.append({
                'class': cls_name, 'cls_id': cls_id, 'conf': tr.conf,
                'n_hits': tr.n_hits, 'xyz': tr.fused_position(), 'tid': tr.tid,
            })
        objects.sort(key=lambda o: (-o['n_hits'], -o['conf']))
        for i, o in enumerate(objects):
            o['object_id'] = i
        tid_to_obj = {o['tid']: o['object_id'] for o in objects}

        print(f"[object_tracker] {self._n_dets} detections -> "
              f"{len(self.tracks)} tracks -> {len(objects)} objects "
              f"(>= {self.min_hits} hits)")

        if save_dir is None:
            return objects
        if self.want_csv:
            self._write_csv(objects, os.path.join(save_dir, 'objects_droid.csv'))
        if self.want_det_csv and self._det_log:
            self._write_detections_csv(
                tid_to_obj, os.path.join(save_dir, 'detections_per_frame.csv'))
        if self.want_ply and objects:
            self._write_ply(objects, os.path.join(save_dir, 'object_markers_droid.ply'))
        if self.want_video and self._overlay_paths:
            self._write_video(os.path.join(save_dir, 'object_overlay.mp4'))
        return objects

    def _write_detections_csv(self, tid_to_obj, path):
        """One row per detection per keyframe -- the raw temporal trace.

        Sorted by frame, carries t_sec (time axis) and object_id (which fused
        object this detection belongs to, -1 if its track was filtered out).
        """
        rows = sorted(self._det_log, key=lambda r: (r['frame_idx'], -r['conf']))
        with open(path, 'w') as f:
            f.write("frame_idx,t_sec,kf,object_id,class,cls_id,conf,"
                    "x1,y1,x2,y2,depth,wx,wy,wz\n")
            for r in rows:
                oid = tid_to_obj.get(r['tid'], -1)
                f.write(f"{r['frame_idx']},{r['t_sec']:.6f},{r['kf']},{oid},"
                        f"{r['class']},{r['cls_id']},{r['conf']:.4f},"
                        f"{r['x1']:.1f},{r['y1']:.1f},{r['x2']:.1f},{r['y2']:.1f},"
                        f"{r['depth']:.4f},{r['wx']:.6f},{r['wy']:.6f},{r['wz']:.6f}\n")
        n_loc = sum(1 for r in rows if r['tid'] >= 0)
        print(f"[object_tracker] detections_per_frame.csv -> {path} "
              f"({len(rows)} detections, {n_loc} localised)")

    def _write_csv(self, objects, path):
        with open(path, 'w') as f:
            f.write("object_id,class,cls_id,conf,n_detections,x,y,z\n")
            for o in objects:
                x, y, z = o['xyz']
                f.write(f"{o['object_id']},{o['class']},{o['cls_id']},"
                        f"{o['conf']:.4f},{o['n_hits']},{x:.6f},{y:.6f},{z:.6f}\n")
        print(f"[object_tracker] objects_droid.csv -> {path}")

    def _write_ply(self, objects, path):
        """Write markers as a **2DGS Gaussian-splat PLY** so splat viewers
        (e.g. superspl.at) render them like the map PLY. Each object is a small
        sphere of class-coloured splats with randomised disk orientations so the
        blob reads as solid from any angle. Schema matches
        gaussian/vis_utils.construct_list_of_attributes('2dgs').
        """
        from plyfile import PlyData, PlyElement
        try:
            from vings_utils.detector_base import class_color
        except ImportError:
            from detector_base import class_color

        C0 = 0.28209479177387814          # SH-DC normalisation (== vis_utils)
        n_per = 80
        rng = np.random.default_rng(0)
        pts, cols = [], []
        for o in objects:
            c = np.asarray(o['xyz'], dtype=np.float32)
            r, g, b = class_color(o['cls_id'])
            dirs = rng.normal(size=(n_per, 3))
            dirs /= (np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-9)
            pts.append(c[None, :] + self.marker_radius * dirs)
            cols.append(np.tile([r, g, b], (n_per, 1)))
        xyz = np.concatenate(pts).astype(np.float32)
        rgb = (np.concatenate(cols).astype(np.float32)) / 255.0
        N = xyz.shape[0]

        normals = np.zeros((N, 3), np.float32)
        f_dc = (rgb - 0.5) / C0                              # RGB -> SH DC
        f_rest = np.zeros((N, 45), np.float32)
        opacity = np.full((N, 1), 6.0, np.float32)           # raw logit; sigmoid->~1
        sigma = max(0.15, self.marker_radius / 5.0)
        scale = np.full((N, 2), np.log(sigma), np.float32)   # raw log-scale (2dgs: 2 dims)
        quat = rng.normal(size=(N, 4)).astype(np.float32)    # random disk orientations
        quat /= (np.linalg.norm(quat, axis=1, keepdims=True) + 1e-9)

        attrs = np.concatenate(
            [xyz, normals, f_dc, f_rest, opacity, scale, quat], axis=1)
        names = (['x', 'y', 'z', 'nx', 'ny', 'nz']
                 + [f'f_dc_{i}' for i in range(3)]
                 + [f'f_rest_{i}' for i in range(45)]
                 + ['opacity', 'scale_0', 'scale_1']
                 + [f'rot_{i}' for i in range(4)])
        elements = np.empty(N, dtype=[(n, 'f4') for n in names])
        elements[:] = list(map(tuple, attrs))
        PlyData([PlyElement.describe(elements, 'vertex')]).write(path)
        print(f"[object_tracker] object_markers_droid.ply -> {path} "
              f"({len(objects)} markers, {N} 2dgs splats)")

    def _write_video(self, path):
        try:
            import cv2
        except Exception as e:
            print(f"[object_tracker] video skipped: {e}")
            return
        paths = sorted(self._overlay_paths)
        first = cv2.imread(paths[0])
        if first is None:
            return
        h, w = first.shape[:2]
        vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'),
                             self.overlay_fps, (w, h))
        for p in paths:
            frame = cv2.imread(p)
            if frame is not None:
                vw.write(frame)
        vw.release()
        print(f"[object_tracker] object_overlay.mp4 -> {path} ({len(paths)} frames)")


# =============================================================================
# Smoketest: back-projection axis sanity check
# =============================================================================

if __name__ == "__main__":
    # Synthetic planar depth at z = 5 m, identity pose, square centred intrinsics.
    H, W = 480, 640
    fx = fy = 500.0
    cx, cy = 320.0, 240.0
    intr = {'fu': fy, 'fv': fx, 'cu': cy, 'cv': cx, 'H': H, 'W': W}
    depth = np.full((H, W), 5.0, dtype=np.float32)
    c2w = np.eye(4)

    # 1) centre pixel -> [0, 0, 5]
    p_mid = unproject_center(cx, cy, 5.0, intr, c2w)
    assert np.allclose(p_mid, [0, 0, 5], atol=1e-4), p_mid

    # 2) shift right (larger column) -> +X
    p_right = unproject_center(cx + 100, cy, 5.0, intr, c2w)
    assert p_right[0] > 0.9 and abs(p_right[1]) < 1e-4, p_right

    # 3) shift down (larger row) -> +Y
    p_down = unproject_center(cx, cy + 100, 5.0, intr, c2w)
    assert p_down[1] > 0.9 and abs(p_down[0]) < 1e-4, p_down

    # 4) full pipeline: a fake "car" box near the centre, fused over 3 frames.
    class _D:
        def __init__(s, box, cid, name, conf):
            s.bbox_xyxy, s.cls_id, s.cls_name, s.conf = box, cid, name, conf
        @property
        def center(s):
            x1, y1, x2, y2 = s.bbox_xyxy
            return 0.5 * (x1 + x2), 0.5 * (y1 + y2)

    cfg = {'object_tracker': {'assoc_radius': 0.5, 'min_hits': 3},
           'object_output': {'csv': False, 'markers_ply': False, 'overlay_video': False}}
    trk = ObjectTracker(cfg, save_dir=None)
    box = (cx - 20, cy - 20, cx + 20, cy + 20)
    for _ in range(3):
        trk.update([_D(box, 2, 'car', 0.9)], depth, c2w, intr, f_idx=0)
    objs = trk.finalize()
    assert len(objs) == 1 and objs[0]['class'] == 'car', objs
    assert np.allclose(objs[0]['xyz'], [0, 0, 5], atol=1e-3), objs[0]['xyz']

    print("[object_tracker smoketest] all axis + fusion checks passed.")
    print(f"  centre={p_mid}  right={p_right}  down={p_down}")
    print(f"  fused car @ {objs[0]['xyz']}  (n_hits={objs[0]['n_hits']})")
