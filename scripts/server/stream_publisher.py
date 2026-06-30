"""Live-Streaming der Gaussians + Objekte ans Web-Frontend.

Aus scripts/run.py ausgelagert. `StreamPublisher` kapselt den
WebSocket-`SplatStreamServer`, den GPS-verankerten Geo-Referencer und die
gesamte Push-Logik (frozen/active-Gaussian-Delta, Kamera-Frames mit
Detektions-Overlay, Objekt-Marker, Loop-Closure-Resync). Der Run-Loop haelt
eine Instanz und ruft nur noch die ``push_*``-Methoden; das ``every_kf`` /
``frame_stride``-Scheduling bleibt im Loop (siehe run.py).

Alle Push-Methoden sind best-effort: bei deaktiviertem Stream (kein Server)
oder einem Encode-Fehler kehren sie geraeuschlos zurueck, damit der Run nie an
der Visualisierung scheitert.
"""

import os
import base64

import numpy as np
import torch


class StreamPublisher:
    def __init__(self, cfg):
        self.cfg = cfg
        self.stream_cfg = (cfg.get('stream') or {})
        self.stream_server = None
        self._streamed_kf_ids: set = set()
        self._active_sig: dict = {}    # kf_id -> change-signature of last-sent active group
        self._stream_epoch = 0
        self._geo = None               # LiveGeoReferencer (GPS-anchored map projection)
        if self.stream_cfg.get('enabled', False):
            try:
                from server.stream_server import SplatStreamServer
                self.stream_server = SplatStreamServer(
                    host=self.stream_cfg.get('host', '0.0.0.0'),
                    port=int(self.stream_cfg.get('port', 8765)),
                    max_queue=int(self.stream_cfg.get('max_queue', 16)))
                self.stream_server.start()
                # Live geo-referencing: project the gauge-free DROID map onto real satellite imagery
                if self.stream_cfg.get('geo', True):
                    self._init_geo_referencer()
            except Exception as _e:
                print(f"[stream] disabled (init failed): {_e}")
                self.stream_server = None

    @property
    def enabled(self):
        return self.stream_server is not None

    def stop(self):
        """WebSocket-Stream-Server stoppen (daemon-Thread; harmlos wenn aus)."""
        if self.stream_server is None:
            return
        try:
            self.stream_server.stop()
        except Exception as _e:
            print(f"[stream] stop failed: {_e}")

    def resync(self):
        """Epoch++ + resync: Frontend leert die Szene, naechster Push re-streamt
        das gesamte frozen-Set neu. Nach Loop-Closure (frozen Gaussians global
        transformiert -> alle bereits gestreamten frozen-Daten sind veraltet)."""
        if self.stream_server is None:
            return
        self._stream_epoch += 1
        self._streamed_kf_ids.clear()
        self._active_sig.clear()
        self.stream_server.push({'type': 'resync', 'epoch': self._stream_epoch})

    def _init_geo_referencer(self):
        """Set up the GPS-anchored live map projection (best-effort)."""
        try:
            gps_csv = (self.cfg.get('dataset') or {}).get('gps_csv')
            if not gps_csv or not os.path.isfile(gps_csv):
                print("[geo] no gps_csv -> map projection off (raw DROID frame)")
                return
            lat_col = int(self.cfg['dataset'].get('gps_lat_col', 1))
            lon_col = int(self.cfg['dataset'].get('gps_lon_col', 2))
            row0 = np.loadtxt(gps_csv, comments='#')[0]
            from server.geo_frame import LiveGeoReferencer
            static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      'static')
            self._geo = LiveGeoReferencer(
                gps_lat0=float(row0[lat_col]), gps_lon0=float(row0[lon_col]),
                static_dir=static_dir,
                min_kfs=int(self.stream_cfg.get('geo_min_kfs', 10)),
                min_span_m=float(self.stream_cfg.get('geo_min_span_m', 40.0)),
                zoom=int(self.stream_cfg.get('geo_zoom', 18)))
            print("[geo] live map projection armed (GPS-anchored)")
        except Exception as _e:
            print(f"[geo] init failed: {_e}")
            self._geo = None

    def add_keyframe_geo(self, viz_out, data_packet):
        """Feed one keyframe's DROID pose + GPS-ENU to the geo-referencer, then
        push the refreshed DROID->three matrix (+satellite) to the frontend."""
        if self._geo is None or self.stream_server is None:
            return
        try:
            xyz_enu = data_packet.get('xyz_enu')
            if xyz_enu is None:
                return
            pose = viz_out['poses'][-1].detach().cpu().numpy()   # (4,4) c2w
            C_droid = pose[:3, 3]
            fwd_droid = pose[:3, 2]                               # camera +z = look dir
            self._geo.add_keyframe(C_droid, fwd_droid, np.asarray(xyz_enu, np.float64))
            msg = self._geo.geo_message(self._stream_epoch)
            if msg is not None:
                self.stream_server.push(msg)
        except Exception as _e:
            print(f"[geo] keyframe push skipped: {_e}")

    def push_gaussians(self, mapper, storage_manager, idx):
        """Push the current Gaussian state to the WebSocket frontend (best-effort).

        Delta strategy keyed on `_globalkf_id` (stable per Gaussian, survives
        prune): newly-frozen KF groups on the StorageManager CPU side are sent
        once as `append_frozen`; the small, still-optimised Mapper GPU set is
        always re-sent in full as `replace_active`. Without a StorageManager
        everything is "active" -> a single `replace_all` snapshot.
        """
        if self.stream_server is None:
            return
        try:
            from server.splat_encode import (encode_splat_from_mapper,
                                              encode_splat_from_storage)
            eps = float(self.stream_cfg.get('flat_scale_eps', 1e-3))
            max_active = int(self.stream_cfg.get('max_active_splats', 200000))
            sm = storage_manager

            # Server sends the COMPLETE current set every push (new frozen KF
            # groups appended + the full live mapper set as replace_active; a
            # single replace_all before the first convey). The "load it into the
            # map gradually" happens purely frontend-side (progressive reveal in
            # viewer.html) -- the wire stays simple and complete.
            if sm is not None and sm._xyz.shape[0] > 0:
                kf_ids = sm._globalkf_id
                present = set(int(k) for k in torch.unique(kf_ids).tolist())
                for kid in sorted(present - self._streamed_kf_ids):
                    blob = encode_splat_from_storage(sm, kf_ids == kid, eps)
                    if blob:
                        self.stream_server.push({
                            'type': 'append_frozen', 'epoch': self._stream_epoch,
                            'kf_id': kid, 'data': blob})
                    self._streamed_kf_ids.add(kid)
                # Active set as a per-KF-group DELTA: only re-encode groups whose
                # gaussians actually moved since the last push (the optimisation
                # window), retract groups that left the mapper. Keeps the wire
                # small + reliable so the live splats track the video instead of
                # the old fat replace_active payload getting dropped.
                self._push_active_delta(mapper, eps)
            else:
                allblob = encode_splat_from_mapper(mapper, max_active, eps)
                self.stream_server.push({
                    'type': 'replace_all', 'epoch': self._stream_epoch,
                    'data': allblob})
        except Exception as _e:
            print(f"[stream] gaussian push skipped: {_e}")

    def _push_active_delta(self, mapper, eps):
        """Stream the live mapper set as a per-``_globalkf_id`` group delta.

        For each active group: compute a cheap change-signature (count + rounded
        aggregate of positions/opacity). Re-encode + send (``replace_active_group``)
        only groups whose signature changed since the last push -- converged groups
        are skipped. Groups that left the mapper (frozen or pruned) are retracted
        (``remove_active_group``). Both message types are non-droppable + small, so
        the recent splats arrive reliably and follow the camera, unlike the old
        single fat ``replace_active`` that got dropped under backpressure.
        """
        from server.splat_encode import encode_splat_from_mapper
        gid = getattr(mapper, '_globalkf_id', None)
        if gid is None or gid.numel() == 0:
            for kid in list(self._active_sig):          # nothing active -> retract all
                self.stream_server.push({'type': 'remove_active_group',
                                         'epoch': self._stream_epoch, 'kf_id': int(kid)})
            self._active_sig.clear()
            return
        xyz = mapper.get_property('_xyz').detach()
        op = mapper.get_property('_opacity').detach()
        present = set()
        for k in torch.unique(gid).tolist():
            kid = int(k)
            present.add(kid)
            m = (gid == k)
            n = int(m.sum().item())
            if n == 0:
                continue
            # signature: rounded aggregates -> stable once a group converges, but
            # changes every push while the group is still being optimised.
            sig = (n,
                   round(float(xyz[m, 0].sum().item()), 2),
                   round(float(xyz[m, 1].sum().item()), 2),
                   round(float(xyz[m, 2].sum().item()), 2),
                   round(float(op[m].sum().item()), 3))
            if self._active_sig.get(kid) == sig:
                continue                                # unchanged -> skip (delta)
            blob = encode_splat_from_mapper(mapper, flat_scale_eps=eps, mask=m)
            if blob:
                self.stream_server.push({'type': 'replace_active_group',
                                         'epoch': self._stream_epoch,
                                         'kf_id': kid, 'data': blob})
                self._active_sig[kid] = sig
        for kid in list(self._active_sig):              # retract groups that left active
            if kid not in present:
                self.stream_server.push({'type': 'remove_active_group',
                                         'epoch': self._stream_epoch, 'kf_id': int(kid)})
                del self._active_sig[kid]

    def push_objects(self, object_tracker):
        """Push the current object-track snapshot to the frontend. Each object is
        tagged with the canonical frontend model key (car/van/truck/bus) so the
        viewer matches a glTF asset authoritatively instead of guessing from the
        raw detector class string."""
        if self.stream_server is None:
            return
        from vings_utils.detector_base import canonical_model_key
        objs = object_tracker.snapshot()
        for _o in objs:
            _mk = canonical_model_key(_o.get('class'))
            if _mk:
                _o['model'] = _mk
        self.stream_server.push({
            'type': 'objects', 'epoch': self._stream_epoch,
            'objects': objs})

    def push_frame(self, dataset, f_idx, dets=None):
        """Push the original RGB keyframe (downscaled JPEG) to the frontend so the
        viewer can show the live camera image next to the 3D map. Best-effort.

        When ``dets`` (a list of ``Detection`` for THIS frame) is given and object
        detection is on, the boxes + class labels are drawn straight into the JPEG
        server-side -- the viewer's camera card needs no change and the overlay is
        guaranteed to match exactly the frame it was detected on (boxes are in the
        full-res pixel space of this same image, scaled by the resize factor).
        """
        if self.stream_server is None:
            return
        if not bool(self.stream_cfg.get('send_frames', True)):
            return
        try:
            import cv2 as _cv2
            fp = dataset.rgbinfo_dict['filepath'][f_idx]
            bgr = _cv2.imread(fp)
            if bgr is None:
                return
            max_px = int(self.stream_cfg.get('frame_max_px', 384))
            h, w = bgr.shape[:2]
            s = max_px / float(max(h, w))
            if s < 1.0:
                bgr = _cv2.resize(bgr, (max(1, int(w * s)), max(1, int(h * s))),
                                  interpolation=_cv2.INTER_AREA)
            else:
                s = 1.0
            # Draw detection overlays (boxes + labels) into the downscaled frame.
            if dets and bool(self.stream_cfg.get('draw_detections', True)):
                from vings_utils.detector_base import class_color
                fh = bgr.shape[0]
                th = max(1, int(round(fh / 240.0)))          # box thickness scales with frame
                fsc = max(0.35, fh / 600.0)                  # label font scale
                for d in dets:
                    x1, y1, x2, y2 = (float(v) * s for v in d.bbox_xyxy)
                    r, g, b = class_color(int(d.cls_id))
                    col = (int(b), int(g), int(r))           # RGB -> BGR for cv2
                    p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
                    _cv2.rectangle(bgr, p1, p2, col, th)
                    label = f"{d.cls_name} {d.conf:.2f}"
                    (tw, tht), _bl = _cv2.getTextSize(label, _cv2.FONT_HERSHEY_SIMPLEX, fsc, 1)
                    ly = max(0, int(y1) - 4)
                    _cv2.rectangle(bgr, (int(x1), ly - tht - 4),
                                   (int(x1) + tw + 2, ly + 2), col, -1)
                    _cv2.putText(bgr, label, (int(x1) + 1, ly - 2),
                                 _cv2.FONT_HERSHEY_SIMPLEX, fsc, (0, 0, 0), 1, _cv2.LINE_AA)
            q = int(self.stream_cfg.get('frame_jpeg_quality', 70))
            ok, enc = _cv2.imencode('.jpg', bgr, [int(_cv2.IMWRITE_JPEG_QUALITY), q])
            if not ok:
                return
            b64 = base64.b64encode(enc.tobytes()).decode('ascii')
            self.stream_server.push({
                'type': 'frame', 'epoch': self._stream_epoch,
                'idx': int(f_idx), 'w': int(bgr.shape[1]), 'h': int(bgr.shape[0]),
                'jpeg': b64})
        except Exception as _e:
            print(f"[stream] frame push skipped: {_e}")
