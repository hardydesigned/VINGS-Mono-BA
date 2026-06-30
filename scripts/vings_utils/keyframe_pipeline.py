"""Keyframe-Select + Mapper-Vorbereitung pro Tracker-Keyframe.

Aus scripts/run.py ausgelagert. `KeyframePipeline` buendelt die Logik, die im
Run-Loop *nach* dem Tracker und *vor* (bzw. um) dem Mapper-Aufruf laeuft:

- ``log_raw_tracker_pose`` : Append-Only-Drift-Log der rohen DROID-BA-Pose
  (pre-override) -- diagnostische Tracker-Historie.
- ``decide_mapping``       : FrameSelector-Entscheidung (``should_accept``) bzw.
  ``mapper_kf_skip``-Modulo-Fallback -> ``(do_map, fs_score)``.
- ``run_detection``        : Online-Objektdetektion auf jedem N-ten Tracker-KF
  (``object_detect_stride``), 3D-Lokalisierung via ``object_tracker.update`` und
  Objekt-/Frame-Push ans Stream-Frontend.
- ``prepare_for_mapper``   : Metric3D-Depth-Swap (statt DROID-DBA-Depth) +
  Dynamic-Object-Segmentation, beide direkt auf ``viz_out`` bevor der Mapper
  laeuft.

Der Pipeline werden ihre Kollaboratoren (FrameSelector, Detektor, ObjectTracker,
DynamicModel, Stream, PhaseTimer) im Konstruktor uebergeben; sie haelt selbst
keinen Tracker/Mapper-State.
"""

import os

import numpy as np
import torch


class KeyframePipeline:
    def __init__(self, cfg, *, frame_selector, object_detector, object_tracker,
                 dynamic_model, metric_depth_cache, object_detect_stride,
                 timer, dataset, stream):
        self.cfg = cfg
        self.frame_selector = frame_selector
        self.object_detector = object_detector
        self.object_tracker = object_tracker
        self.dynamic_model = dynamic_model
        self.metric_depth_cache = metric_depth_cache
        self.object_detect_stride = object_detect_stride
        self.timer = timer
        self.dataset = dataset
        self.stream = stream

    def log_raw_tracker_pose(self, viz_out, data_packet, idx, n_keyframes):
        """Pre-Override Drift-Log: schreibt die ROHE Tracker-BA-Pose des neuesten
        KFs pro Iteration in einen Append-Only-Log. Liefert die vollstaendige
        Tracker-Historie (statt nur Active-Window-Snapshot am Ende), nuetzlich
        fuer Drift-Diagnose."""
        if viz_out is None:
            return
        try:
            raw_pose_c2w = viz_out['poses'][-1].detach().cpu().numpy()  # (4,4) c2w
            log_dir = self.cfg['output']['save_dir']
            log_path = os.path.join(log_dir, 'tracker_raw_c2w.txt')
            write_header = not os.path.exists(log_path)
            with open(log_path, 'a') as _lf:
                if write_header:
                    _lf.write('# kf_idx t_sec r00 r01 r02 tx r10 r11 r12 ty '
                              'r20 r21 r22 tz  (4x4 c2w, raw DROID-BA, pre-override)\n')
                flat = raw_pose_c2w[:3].flatten()  # 12 numbers
                t_sec = float(data_packet.get('t_sec', idx))
                _lf.write(f"{n_keyframes} {t_sec:.6f} "
                          + " ".join(f"{v:.6f}" for v in flat) + "\n")
        except Exception:
            pass  # best-effort logging; don't fail the run

    def decide_mapping(self, viz_out, data_packet, idx, n_keyframes, mapper_kf_skip):
        """KF-Filter: entweder FrameSelector (wenn konfiguriert) oder naives
        Modulo-Subsampling. Init-KF wird in beiden Faellen gemappt (Selector
        akzeptiert ersten Frame; modulo-Pfad: (n_keyframes-1)%N==0). Liefert
        ``(do_map, fs_score)``; ``fs_score`` ist ``None`` im Modulo-Pfad."""
        fs_score = None
        if self.frame_selector is not None:
            with self.timer.time('frame_select'):
                depth_np = viz_out['depths'][-1, :, :, 0].detach().cpu().numpy()
                t_np     = viz_out['poses'][-1, :3, 3].detach().cpu().numpy()
                R_np     = viz_out['poses'][-1, :3, :3].detach().cpu().numpy()
                # rgb optional (nur fuer feature-basierte Selektoren wie nurbs_lvi);
                # viz_out['images'][-1] ist HxWx3 in [0,1] float -> uint8 BGR.
                rgb_np = None
                if 'images' in viz_out:
                    img = viz_out['images'][-1].detach().cpu().numpy()
                    rgb_np = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
                # depth_cov optional (nur game_kfs nutzt es; andere ignorieren es ueber **_).
                cov_np = None
                if 'depths_cov' in viz_out:
                    cov_np = viz_out['depths_cov'][-1, :, :, 0].detach().cpu().numpy()
                meta_b = {
                    'alt_m':   data_packet.get('alt_m'),
                    'xyz_enu': data_packet.get('xyz_enu'),
                    't_sec':   data_packet.get('t_sec', float(idx)),
                }
                accept, fs_score = self.frame_selector.should_accept(
                    depth_np, t_np, R_np, rgb=rgb_np, depth_cov=cov_np,
                    meta=meta_b)
            do_map = bool(accept)
        else:
            do_map = (mapper_kf_skip <= 1) or ((n_keyframes - 1) % mapper_kf_skip == 0)
        return do_map, fs_score

    def run_detection(self, viz_out, data_packet, n_keyframes):
        """Online object detection + 3D-localisation. Runs on every Nth tracker
        keyframe (object_detect_stride), DECOUPLED from do_map -- the FrameSelector
        filters mapper KFs hard, so tying detection to it loses most objects. On
        non-mapped KFs viz_out['depths'] is the raw DROID-BA depth (the Metric3D
        swap is mapper-only), which is exactly what object_tracker.unproject
        expects (DROID frame); the ext-pose override already applies here too.
        Streamed objects carry oriented pose (quat) + size so the frontend draws
        3D models."""
        _frame_pushed_idx = -1   # set by the detect block if it pushes a (boxed) frame
        run_detect = (self.object_detector is not None
                      and 'images' in viz_out
                      and ((n_keyframes - 1) % self.object_detect_stride == 0))
        if run_detect:
            try:
                with self.timer.time('detect'):
                    f_idx = int(viz_out['viz_out_idx_to_f_idx'][-1])
                    # Detect on the FULL-RES original frame -- the
                    # 240x288 viz_out image is far too small for
                    # aerial objects. Geometry is scaled back to the
                    # depth resolution inside the tracker (det_hw).
                    det_rgb = None
                    try:
                        import cv2 as _cv2
                        fp = self.dataset.rgbinfo_dict['filepath'][f_idx]
                        bgr = _cv2.imread(fp)
                        if bgr is not None:
                            det_rgb = np.ascontiguousarray(bgr[..., ::-1])  # BGR->RGB
                    except Exception:
                        det_rgb = None
                    if det_rgb is None:  # fallback: low-res viz image
                        img = viz_out['images'][-1].detach().cpu().numpy()
                        det_rgb = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
                    dets = self.object_detector.detect(det_rgb)
                    depth_np = viz_out['depths'][-1, :, :, 0].detach().cpu().numpy()
                    pose_c2w = viz_out['poses'][-1].detach().cpu().numpy()
                    # True time of this keyframe's frame: the
                    # camstamp (real Unix epoch, enables GPS/RTK
                    # correlation later), not rgbinfo['timestamp']
                    # which is just the frame index.
                    try:
                        _cts = getattr(self.dataset, '_cam_t_sec', None)
                        if _cts is not None and f_idx < len(_cts):
                            kf_t_sec = float(_cts[f_idx])
                        else:
                            kf_t_sec = float(self.dataset.rgbinfo_dict['timestamp'][f_idx])
                    except Exception:
                        kf_t_sec = float(data_packet.get('t_sec', f_idx))
                    self.object_tracker.update(
                        dets, depth_np, pose_c2w, viz_out['intrinsic'],
                        f_idx, rgb_u8=det_rgb, det_hw=det_rgb.shape[:2],
                        t_sec=kf_t_sec)
                    if self.stream.enabled:
                        # Object markers (tagged with the canonical glTF
                        # model key inside push_objects).
                        self.stream.push_objects(self.object_tracker)
                        # Push THIS frame with the detection boxes drawn in,
                        # so the camera card shows labelled boxes that match
                        # the exact detected frame. Marks the frame as pushed
                        # to skip the generic (box-free) push below.
                        self.stream.push_frame(self.dataset, f_idx, dets=dets)
                        _frame_pushed_idx = f_idx
            except Exception as _e:
                print(f"[detect] keyframe skipped: {_e}")
        # stream the original RGB keyframe for the viewer's camera card
        # (own stride, decoupled from mapper/detector). Best-effort. Skip if
        # the detect block already pushed this frame (with boxes drawn).
        if self.stream.enabled and 'images' in viz_out:
            _gen_idx = int(viz_out['viz_out_idx_to_f_idx'][-1])
            _fstride = max(1, int(self.stream.stream_cfg.get('frame_stride', 2)))
            if _gen_idx != _frame_pushed_idx and (n_keyframes - 1) % _fstride == 0:
                try:
                    self.stream.push_frame(self.dataset, _gen_idx)
                except Exception:
                    pass

    def prepare_for_mapper(self, viz_out):
        """Mapper-Vorbereitung direkt vor mapper.run(): Metric3D-Depth-Swap +
        Dynamic-Object-Segmentation, beide in-place auf viz_out."""
        # Replace DROID-DBA depths in viz_out with cached Metric3D depths
        # (matched by tracker.video.tstamp). Keep sky pixels (rgb==0) at depth=0
        # so VINGS' sky_mask path still works. depths_cov is tightened where
        # overwritten -> the weighted_l1 in get_loss() (weight=1/cov) trusts the prior.
        if self.cfg.get('use_metric', False) and self.cfg.get('use_metric_for_mapper', True) \
                and len(self.metric_depth_cache) > 0 and 'depths' in viz_out:
            ts_tensor = viz_out['viz_out_idx_to_f_idx']
            ts_list = ts_tensor.detach().cpu().tolist() if hasattr(ts_tensor, 'detach') else list(ts_tensor)
            for i, ts in enumerate(ts_list):
                ts_key = float(ts)
                if ts_key in self.metric_depth_cache:
                    m_d = self.metric_depth_cache[ts_key].to(viz_out['depths'].device)
                    if m_d.shape[:2] == viz_out['depths'].shape[1:3]:
                        img_i = viz_out['images'][i]
                        sky_mask = (img_i.sum(dim=-1) == 0)
                        viz_out['depths'][i, :, :, 0] = torch.where(sky_mask, torch.zeros_like(m_d), m_d)
                        if 'depths_cov' in viz_out:
                            # weighted_l1 in get_loss uses weight=1/cov.
                            # 0.01 -> weight 100 dominiert RGB-Loss.
                            # cfg['metric_cov'] (Default 1.0) -> weight ~1.
                            _cov = float(self.cfg.get('metric_cov', 1.0))
                            viz_out['depths_cov'][i, :, :, 0] = _cov
        # Segment each keyframe once (expensive) and stash the masks on viz_out;
        # the mapper turns them into per-iter dynamic masks inside the (cheap)
        # training loop.
        if self.dynamic_model is not None and 'images' in viz_out:
            with self.timer.time('segment'):
                viz_out['sam_anns'] = [
                    self.dynamic_model.get_anns_raw(viz_out['images'][i])
                    for i in range(viz_out['images'].shape[0])
                ]
