import numpy as np
import shutil
import torch
from lietorch import SE3
import os
from frontend.dbaf import DBAFusion
from gaussian.gaussian_model import GaussianModel
from gaussian.vis_utils import save_ply, save_ply_streaming, vis_map, vis_bev
import argparse
parser = argparse.ArgumentParser(description="Add config path.")
parser.add_argument("config")
parser.add_argument("--prefix", default='')
args = parser.parse_args()
config_path = args.config
from gaussian.general_utils import load_config, get_name
config = load_config(config_path)
import importlib
get_dataset = importlib.import_module(config["dataset"]["module"]).get_dataset
from vings_utils.middleware_utils import judge_and_package, retrieve_to_tracker, datapacket_to_nerfslam
from vings_utils.frame_selector import FrameSelector
from vings_utils.selector_factory import make_frame_selector
from vings_utils.gate_a import GateA
from vings_utils.gate_a_v2 import GateAV2
from storage.storage_manage import StorageManager
from loop.loop_model import LoopModel
from eval.fair_eval import run_fair_eval
from metric.metric_model import Metric_Model
import time
import json
import statistics
from tqdm import tqdm
if config['mode'] == 'vo_nerfslam': from frontend_vo.vio_slam import VioSLAM


class PhaseTimer:
    """Sammelt Sub-Timer mit cuda.synchronize() vor/nach time.time()."""
    def __init__(self, sync=True):
        self.records = {}
        self.sync = sync and torch.cuda.is_available()

    def time(self, name):
        return _PhaseCtx(self, name)

    def add(self, name, dt):
        self.records.setdefault(name, []).append(dt)

    def last(self, name):
        rec = self.records.get(name)
        return rec[-1] if rec else 0.0

    def patch(self, obj, attr, name):
        orig = getattr(obj, attr)
        timer = self
        def wrapper(*args, **kwargs):
            with timer.time(name):
                return orig(*args, **kwargs)
        setattr(obj, attr, wrapper)

    def patch_callable(self, obj, attr, name):
        orig = getattr(obj, attr)
        proxy = _CallableProxy(orig, self, name)
        setattr(obj, attr, proxy)

    def summary(self, total_wall=None):
        if not self.records:
            print("(no timing records)")
            return
        rows = []
        for name, vals in self.records.items():
            n = len(vals)
            tot = sum(vals)
            mean = tot / n
            med = statistics.median(vals)
            p95 = sorted(vals)[max(0, int(0.95 * n) - 1)] if n > 0 else 0.0
            rows.append((name, n, tot, mean, med, p95))
        rows.sort(key=lambda r: -r[2])
        denom = total_wall if total_wall else max(r[2] for r in rows)
        print(f"{'phase':<28} {'n':>6} {'total[s]':>10} {'mean[ms]':>10} "
              f"{'med[ms]':>10} {'p95[ms]':>10} {'%':>7}")
        print("-" * 86)
        for name, n, tot, mean, med, p95 in rows:
            pct = 100.0 * tot / denom if denom > 0 else 0.0
            print(f"{name:<28} {n:>6} {tot:>10.2f} {mean*1000:>10.1f} "
                  f"{med*1000:>10.1f} {p95*1000:>10.1f} {pct:>6.1f}%")


class _PhaseCtx:
    def __init__(self, timer, name):
        self.timer = timer
        self.name = name

    def __enter__(self):
        if self.timer.sync:
            torch.cuda.synchronize()
        self.t0 = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.timer.sync:
            torch.cuda.synchronize()
        self.timer.add(self.name, time.time() - self.t0)
        return False


class _CallableProxy:
    """Transparent-Proxy: leitet Attribut-Zugriffe ans Original weiter,
    misst aber jeden __call__ in einer Phase."""
    __slots__ = ('_target', '_timer', '_name')

    def __init__(self, target, timer, name):
        object.__setattr__(self, '_target', target)
        object.__setattr__(self, '_timer', timer)
        object.__setattr__(self, '_name', name)

    def __call__(self, *args, **kwargs):
        with self._timer.time(self._name):
            return self._target(*args, **kwargs)

    def __getattr__(self, attr):
        return getattr(self._target, attr)

    def __setattr__(self, attr, value):
        setattr(self._target, attr, value)


class Runner:
    def __init__(self, cfg):
        self.cfg = cfg
        self.dataset  = get_dataset(cfg)
        cfg['frontend']['c2i'] = self.dataset.c2i # (4, 4), ndarray
        
        if self.cfg['mode'] == 'vio' or self.cfg['mode'] == 'vo':
            self.tracker = DBAFusion(cfg)
        elif self.cfg['mode'] == 'vo_nerfslam':     
            self.tracker = VioSLAM(cfg)
        else: assert False, "Error \"mode\" in config file."
        
        if 'phone' not in cfg['dataset']['module']: self.tracker.dataset_length = len(self.dataset)
        
        self.mapper = GaussianModel(cfg)
        
        if cfg.get('use_loop', False):
            self.looper = LoopModel(cfg)
        else:
            self.looper = None
        
        self.metric_predictor = None
        if 'use_metric' in cfg.keys() and cfg['use_metric'] and not cfg['dataset'].get('lidar_dir'):
            self.metric_predictor = Metric_Model(cfg)
        # Cache Metric3D depths indexed by cam-timestamp so the mapper can
        # consume the direct metric prior instead of DROID-DBA's noisy output.
        # Enable via cfg['use_metric_for_mapper'] (default true if use_metric).
        self.metric_depth_cache = {}
        
        if 'use_storage_manager' in cfg.keys() and cfg['use_storage_manager']:
            self.use_storage_manager = True
            self.storage_manager = StorageManager(cfg)
            if cfg['dataset']['module'] != 'phone':
                self.storage_manager.dataset_length = self.dataset.rgbinfo_dict['timestamp'][-1] - self.dataset.rgbinfo_dict['timestamp'][0]
        else:
            self.use_storage_manager = False

        # Profiling: hierarchische Sub-Timer mit cuda.synchronize().
        self.timer = PhaseTimer(sync=True)
        self._install_phase_patches()

        intr = cfg['intrinsic']
        K_full = np.array([[intr['fu'], 0.0,        intr['cu']],
                           [0.0,        intr['fv'], intr['cv']],
                           [0.0,        0.0,        1.0]], dtype=np.float32)
        H_full, W_full = int(intr['H']), int(intr['W'])
        fe_img_size = cfg.get('frontend', {}).get('image_size')
        if fe_img_size and len(fe_img_size) == 2:
            H_low, W_low = int(fe_img_size[0]), int(fe_img_size[1])
            sx = W_low / float(W_full)
            sy = H_low / float(H_full)
            K = K_full.copy()
            K[0, 0] *= sx; K[0, 2] *= sx
            K[1, 1] *= sy; K[1, 2] *= sy
            image_hw = (H_low, W_low)
        else:
            K = K_full
            image_hw = (H_full, W_full)
        self.frame_selector = make_frame_selector(cfg, K, image_hw)

        # Dynamic-object masking (optional, off by default). When use_dynamic is
        # set AND a segmentation backend is configured, each keyframe is segmented
        # before the mapper runs; high-error segments are dropped from the mapping
        self.dynamic_model = None
        if cfg.get('use_dynamic') and (cfg.get('segmentation') or {}).get('kind') not in (None, 'none'):
            from dynamic.dynamic_utils import DynamicModel
            mapper_device = cfg.get('device', {}).get('mapper', 'cuda')
            self.dynamic_model = DynamicModel(cfg, mapper_device)

        # Object detection + online 3D-localisation (optional, off by default).
        self.object_detector = None
        self.object_tracker = None
        if cfg.get('detect_objects') and (cfg.get('object_detector') or {}).get('kind') not in (None, 'none'):
            from vings_utils.detector_factory import make_object_detector
            from vings_utils.object_tracker import ObjectTracker
            mapper_device = cfg.get('device', {}).get('mapper', 'cuda')
            self.object_detector = make_object_detector(cfg, mapper_device)
            self.object_tracker = ObjectTracker(cfg, cfg['output']['save_dir'])
        # Detection runs on every Nth *tracker* keyframe, decoupled from the FrameSelector/mapper decision
        self.object_detect_stride = max(1, int(cfg.get('object_detect_stride', 3)))

        # Live-Streaming of Gaussians 
        self.stream_server = None
        self.stream_cfg = (cfg.get('stream') or {})
        self._streamed_kf_ids: set = set()
        self._active_sig: dict = {}    # kf_id -> change-signature of last-sent active group
        self._stream_epoch = 0
        self._geo = None              # LiveGeoReferencer (GPS-anchored map projection)
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

        # Gate A: Pre-Track-Filter
        ga_cfg = (cfg.get('gate_a') or {})
        if ga_cfg.get('enabled', False):
            ga_version = str(ga_cfg.get('version', 'v1')).lower()
            if ga_version == 'v2':
                self.gate_a = GateAV2.from_config(ga_cfg)
            elif ga_version in ('v1', ''):
                self.gate_a = GateA.from_config(ga_cfg)
            else:
                raise ValueError(
                    f"Unknown gate_a.version={ga_version!r}. Known: v1, v2")
            # Warn if A1 is enabled but the loader has no GPS source.
            if ga_cfg.get('enable_a1', True) and getattr(self.dataset, '_gps_t', None) is None:
                print('WARN: gate_a.enable_a1=true but loader has no GPS source -- A1 dormant')
            # Warn if A3 is enabled (v2 only) but loader has no GPS source.
            if (ga_version == 'v2'
                    and ga_cfg.get('enable_a3', False)
                    and getattr(self.dataset, '_gps_t', None) is None):
                print('WARN: gate_a.enable_a3=true but loader has no GPS source '
                      '-- A3 fail-open (every frame passes)')
        else:
            self.gate_a = None

    def _rgb_tensor_to_uint8_bgr(self, rgb_tensor):
        """Convert tracker rgb tensor (1, 3, H, W) on cuda, RGB float [0,1]/255 ->
        (H, W, 3) uint8 BGR numpy on CPU. Used by GateA for cv2-style operations."""
        # The loader's rgb is uint8 [0,255] cast to float (cv2.resize keeps dtype);
        # tensor(...).permute uses the raw values, so the tensor is float-valued
        # in [0,255]. See GenericVODataset.__getitem__.
        img = rgb_tensor[0].detach().cpu().numpy()           # (3, H, W) float
        img = np.transpose(img, (1, 2, 0))                   # (H, W, 3) RGB
        img = np.clip(img, 0.0, 255.0).astype(np.uint8)
        return img[..., ::-1].copy()                         # -> BGR contig

    def _install_phase_patches(self):
        """Wrappt interne Methoden, sodass jeder Aufruf seine Phase befuellt."""
        # Tracker (frontend.dbaf.DBAFusion):
        #   filterx.track  -> Motion Filter (Optical-Flow Gate)
        #   frontend()     -> Local Bundle Adjustment + Distance/KF-Check
        if hasattr(self.tracker, 'filterx'):
            self.timer.patch(self.tracker.filterx, 'track', 'track.motion_filter')
        if hasattr(self.tracker, 'frontend'):
            # frontend ist ein Callable-Objekt mit Attributen (new_frame_added,
            # all_imu, ...) -- Proxy verwenden, damit Attribute erreichbar bleiben.
            self.timer.patch_callable(self.tracker, 'frontend', 'track.frontend_ba')

        # Mapper (gaussian.gaussian_base.run_only_mapping ruft diese auf):
        #   add_new_frame   -> Render + Prune + PointCloud-Extraction pro KF
        #   train_once_pose -> optionale Pose-Refinement-Iterationen
        #   train_once      -> Haupt-Training-Loop (~train_iters Iterationen)
        for attr, phase in [
            ('add_new_frame', 'map.add_new_frame'),
            ('train_once_pose', 'map.pose_refine'),
            ('train_once', 'map.train_loop'),
        ]:
            if hasattr(self.mapper, attr):
                self.timer.patch(self.mapper, attr, phase)

    # TODO
    def _apply_ext_poses_to_vizout(self, viz_out):
        """Ersetze viz_out['poses'] durch RTK c2w aus dataset.ext_poses, und
        skaliere viz_out['depths'] um den per-window Skalenfaktor.

        Rationale: judge_and_package_v3 liefert Posen aus DROID-DBA's lokalem
        Koordinatensystem -- auf Nadir-Aerial bricht der Maßstab (35× Shrink
        auf amtown03 dokumentiert). Override an dieser Stelle ist sauber:
        - keine Beruehrung von video.poses oder video.poses_save (kein BA-Risiko)
        - Mapper sieht konsistente RTK-Posen + RTK-skalierte Depths
        - alle KFs im selben aktiven Window haben einheitlichen Frame -> kein Mix

        Skala kommt aus dem Verhaeltnis konsekutiver Distanzen RTK / DROID-DBA
        im aktiven Window (Median ueber Paare). EMA-glaettung ueber Aufrufe
        gegen Burst-Rauschen. Bei n=1 oder degeneriertem droid-Window (alle
        Distanzen ~0) fallback auf letzte cached Skala.
        """
        if not (hasattr(self.dataset, 'ext_poses') and self.dataset.ext_poses is not None):
            return viz_out
        if viz_out is None or 'poses' not in viz_out:
            return viz_out

        tstamps = viz_out['viz_out_idx_to_f_idx']
        n = tstamps.shape[0]
        if n == 0:
            return viz_out

        from scipy.spatial.transform import Rotation as _R

        # 1) Gather RTK w2c [tx,ty,tz,qx,qy,qz,qw] for each KF in window.
        ext_arr = self.dataset.ext_poses
        rtk_tq = np.zeros((n, 7), dtype=np.float32)
        any_missing = False
        for i in range(n):
            fi = int(tstamps[i].item())
            if not (0 <= fi < len(ext_arr)):
                any_missing = True
                break
            rtk_tq[i] = ext_arr[fi]
        if any_missing:
            return viz_out  # bail; don't half-override

        # 2) Convert RTK w2c -> c2w 4x4 numpy.
        Rw2c = _R.from_quat(rtk_tq[:, 3:7]).as_matrix()        # (n,3,3)
        tw2c = rtk_tq[:, 0:3]                                  # (n,3)
        Rc2w = Rw2c.transpose(0, 2, 1)                         # (n,3,3)
        tc2w = -np.einsum('nij,nj->ni', Rc2w, tw2c)            # (n,3)
        rtk_c2w = np.zeros((n, 4, 4), dtype=np.float32)
        rtk_c2w[:, :3, :3] = Rc2w
        rtk_c2w[:, :3, 3]  = tc2w
        rtk_c2w[:, 3,  3]  = 1.0

        # 3) Scale from RTK/DROID consecutive-distance ratio over the ENTIRE
        # known trajectory so far (cumulative path lengths), not just per-pair.
        # Per-pair is unstable when DROID drifts and produces near-zero motion
        # while RTK reports normal motion -> ratio explodes (saw scale=460
        # crash gaussian rasterizer). Cumulative ratio is robust to local
        # zero-motion segments.
        #
        # Strategy: collect (d_rtk, d_droid) pairs over all calls into running
        # sums. Discard pairs where either distance is near zero. The scale =
        # sum(d_rtk) / sum(d_droid) is the global Procrustes-like scale.
        droid_xyz = viz_out['poses'][:, :3, 3].detach().cpu().numpy()  # (n,3)
        if not hasattr(self, '_ext_pose_dist_sums'):
            self._ext_pose_dist_sums = [0.0, 0.0]  # [sum_d_rtk, sum_d_droid]
        if n >= 2:
            d_rtk   = np.linalg.norm(np.diff(tc2w,     axis=0), axis=1)
            d_droid = np.linalg.norm(np.diff(droid_xyz, axis=0), axis=1)
            # Require meaningful motion in BOTH frames (filter hovering / drift).
            mask = (d_droid > 5e-3) & (d_rtk > 0.05)
            if mask.sum() > 0:
                # Use the LAST pair only -- avoids double-counting across calls
                # since adjacent calls share most KFs. The last pair is the
                # newest one not yet integrated.
                if mask[-1]:
                    self._ext_pose_dist_sums[0] += float(d_rtk[-1])
                    self._ext_pose_dist_sums[1] += float(d_droid[-1])

        sum_rtk, sum_droid = self._ext_pose_dist_sums
        if sum_droid > 0.01:  # need enough cumulative motion
            scale = sum_rtk / sum_droid
        else:
            scale = getattr(self, '_ext_pose_cached_scale', 1.0)
        # Hard clamp -- amtown03 measured ratio is ~327x (cf. drift_diagnostic
        # plot v5/v9), Bell412/smaller scenes are deutlich darunter. Wir
        # erlauben grosszuegig bis 1000x, damit Aerial-Nadir nicht aufs
        # alte 100x-Limit clipt (das hat depths um Faktor 3 unter-skaliert).
        scale = float(np.clip(scale, 0.1, 1000.0))
        self._ext_pose_cached_scale = scale
        scale_est = scale  # for logging compatibility
        hist = self._ext_pose_dist_sums  # for log formatting

        # 4) Apply.
        device = viz_out['poses'].device
        dtype  = viz_out['poses'].dtype
        viz_out['poses'] = torch.from_numpy(rtk_c2w).to(device=device, dtype=dtype)
        viz_out['depths'] = viz_out['depths'] * scale
        if 'depths_cov' in viz_out and viz_out['depths_cov'] is not None:
            viz_out['depths_cov'] = viz_out['depths_cov'] * (scale * scale)

        if not getattr(self, '_ext_pose_logged_apply_first', False):
            print(f"[ext_pose] first viz_out apply: n_kfs={n} scale={scale:.4f} "
                  f"rtk_xyz_first={tc2w[0]} droid_xyz_first={droid_xyz[0]}")
            self._ext_pose_logged_apply_first = True
        # Log scale every K calls so we can spot spikes.
        c = getattr(self, '_ext_pose_call_count', 0) + 1
        self._ext_pose_call_count = c
        if c % 100 == 1 or scale > 200.0:
            est_str = f"{scale_est:.3f}" if scale_est is not None else "None"
            print(f"[ext_pose] call={c} n={n} scale_est={est_str} "
                  f"rolling={scale:.4f} hist_len={len(hist)}")

        return viz_out

    # TODO
    def _seed_video_poses_with_ext(self):
        """Schreibe RTK-Posen + rescale disparity in video.poses / video.disps,
        damit die naechste BA-Iteration von einer RTK-verankerten Skala
        startet. Greift nur wenn dataset.ext_poses_file gesetzt UND
        cfg['seed_video_with_ext_pose'] == True.

        Mechanik:
          1) Bestimme aktuelle Scale-Schaetzung (gleicher Algorithmus wie
             _apply_ext_poses_to_vizout: kumulatives sum_rtk / sum_droid).
          2) Fuer jede aktive KF k im Active-Window (0..counter):
             - Setze video.poses[k] auf RTK c2w-pose (in TUM-tq w2c-Format)
             - Skaliere video.disps[k] um 1/scale (so dass Tiefe der RTK-Skala
               entspricht; disparity = 1/depth).

        Caveat: BA wird in den naechsten Aufrufen die Posen wieder optimieren
        und ggf. zurueck-driften. Aber jedes track() startet von der
        RTK-Seed -- der Scale kann nicht mehr unbeschraenkt collapsen.
        """
        if not self.cfg.get('seed_video_with_ext_pose', False):
            return
        if not (hasattr(self.dataset, 'ext_poses')
                and self.dataset.ext_poses is not None):
            return

        video = (self.tracker.video if hasattr(self.tracker, 'video')
                 else self.tracker.frontend.video)
        counter = video.counter.value if hasattr(video, 'counter') else 0
        if counter < 2:
            return  # zu frueh fuer scale-Schaetzung

        scale = float(getattr(self, '_ext_pose_cached_scale', 1.0))
        if scale <= 0.0 or scale == 1.0:
            # Wir haben noch keine valide Scale-Messung -- skippe.
            return

        # Per-KF tstamp im DROID-Buffer ist der Dataset-Frame-Idx.
        tstamps = video.tstamp[:counter].detach().cpu().numpy().astype(np.int64)
        n_ext = len(self.dataset.ext_poses)
        # Build TUM-tq w2c poses (the format video.poses expects).
        seed = np.zeros((counter, 7), dtype=np.float32)
        seed[:, -1] = 1.0  # qw
        valid_mask = np.zeros(counter, dtype=bool)
        for k in range(counter):
            f = int(tstamps[k])
            if 0 <= f < n_ext:
                seed[k] = self.dataset.ext_poses[f]
                valid_mask[k] = True
        if not valid_mask.any():
            return

        device = video.poses.device
        dtype  = video.poses.dtype
        # Write seed where valid.
        seed_t = torch.from_numpy(seed[valid_mask]).to(device=device, dtype=dtype)
        idx_t  = torch.from_numpy(np.nonzero(valid_mask)[0]).to(device=device)
        video.poses[idx_t] = seed_t

        # Rescale disparity in proportion. disparity = 1/depth. RTK-scaled
        # depth = DROID_depth * scale -> RTK_disp = DROID_disp / scale.
        if hasattr(video, 'disps'):
            video.disps[:counter] = video.disps[:counter] / float(scale)

        if not getattr(self, '_seed_logged_first', False):
            print(f"[seed] erste video.poses-Seed: counter={counter} "
                  f"scale={scale:.2f} valid={int(valid_mask.sum())}/{counter}")
            self._seed_logged_first = True

    def _write_profiling(self, n_keyframes, n_mapped, n_processed, wall_t0,
                          frame_skip, mapper_kf_skip, last_idx, partial):
        """Atomic profiling.json dump. Survives SIGKILL/OOM mid-run."""
        try:
            out_path = os.path.join(self.cfg['output']['save_dir'], 'profiling.json')
            tmp_path = out_path + '.tmp'
            payload = {
                'wall_total_s': time.time() - wall_t0,
                'n_keyframes': n_keyframes,
                'n_mapped': n_mapped,
                'n_processed': n_processed,
                'n_frames': len(self.dataset),
                'last_idx': last_idx,
                'frame_skip': frame_skip,
                'mapper_kf_skip': mapper_kf_skip,
                'partial': partial,
                'records': self.timer.records,
            }
            with open(tmp_path, 'w') as f:
                json.dump(payload, f)
            os.replace(tmp_path, out_path)
        except Exception as e:
            print(f"profiling.json write failed: {e}")

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
                                      'server', 'static')
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

    def _geo_add_keyframe(self, viz_out, data_packet):
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

    def _stream_push_gaussians(self, idx):
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
            import torch
            from server.splat_encode import (encode_splat_from_mapper,
                                              encode_splat_from_storage)
            eps = float(self.stream_cfg.get('flat_scale_eps', 1e-3))
            max_active = int(self.stream_cfg.get('max_active_splats', 200000))
            sm = self.storage_manager if self.use_storage_manager else None

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
                self._push_active_delta(eps)
            else:
                allblob = encode_splat_from_mapper(self.mapper, max_active, eps)
                self.stream_server.push({
                    'type': 'replace_all', 'epoch': self._stream_epoch,
                    'data': allblob})
        except Exception as _e:
            print(f"[stream] gaussian push skipped: {_e}")

    def _push_active_delta(self, eps):
        """Stream the live mapper set as a per-``_globalkf_id`` group delta.

        For each active group: compute a cheap change-signature (count + rounded
        aggregate of positions/opacity). Re-encode + send (``replace_active_group``)
        only groups whose signature changed since the last push -- converged groups
        are skipped. Groups that left the mapper (frozen or pruned) are retracted
        (``remove_active_group``). Both message types are non-droppable + small, so
        the recent splats arrive reliably and follow the camera, unlike the old
        single fat ``replace_active`` that got dropped under backpressure.
        """
        import torch
        from server.splat_encode import encode_splat_from_mapper
        gid = getattr(self.mapper, '_globalkf_id', None)
        if gid is None or gid.numel() == 0:
            for kid in list(self._active_sig):          # nothing active -> retract all
                self.stream_server.push({'type': 'remove_active_group',
                                         'epoch': self._stream_epoch, 'kf_id': int(kid)})
            self._active_sig.clear()
            return
        xyz = self.mapper.get_property('_xyz').detach()
        op = self.mapper.get_property('_opacity').detach()
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
            blob = encode_splat_from_mapper(self.mapper, flat_scale_eps=eps, mask=m)
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

    def _stream_push_frame(self, f_idx, dets=None):
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
            import base64
            import cv2 as _cv2
            fp = self.dataset.rgbinfo_dict['filepath'][f_idx]
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

    def run(self):
        # Load imu data.
        self.tracker.frontend.all_imu   = self.dataset.preload_imu()
        self.tracker.frontend.all_stamp = self.dataset.preload_camtimestamp()

        # Stage C: GNSS fuer den DBA-Fusion-GPSFactor laden (nur wenn der Loader es
        # anbietet UND dataset.gnss_file gesetzt ist; sonst bleibt all_gnss=[] -> GNSS aus).
        if hasattr(self.dataset, 'preload_gnss'):
            _gnss = self.dataset.preload_gnss()
            if _gnss is not None and len(_gnss) > 0:
                self.tracker.frontend.all_gnss = _gnss
                # Body->GNSS-Antenne-Hebel (tbg) ist sonst None -> GPSFactor crasht.
                # Spike: Antenne ~= Body -> Null-Hebel.
                self.tracker.frontend.video.tbg = np.zeros(3)

        # Profiling-Counter und Print-Throttle.
        n_keyframes = 0
        n_mapped = 0
        n_processed = 0
        last_idx = -1
        log_every = int(self.cfg.get('profiling', {}).get('log_every', 1))
        snapshot_every = int(self.cfg.get('profiling', {}).get('snapshot_every_kf', 10))
        frame_skip = int(self.cfg.get('frame_skip', 1))
        mapper_kf_skip = int(self.cfg.get('mapper_kf_skip', 1))
        wall_t0 = time.time()

        # Run Tracking.
        n_gate_a_rejected = 0
        for idx in tqdm(range(len(self.dataset))):
            # Skip frames before all processing
            if frame_skip > 1 and idx % frame_skip != 0:
                continue

            data_packet = self.dataset[idx]

            # Gate A: pre-tracker eligibility filter. We need rgb (cheap blur /
            # exposure / gradient density), so we load data_packet first. The
            # GPS-altitude check still saves the ~450 ms tracker.frontend_ba.
            if self.gate_a is not None:
                meta_a = {
                    'alt_m':   data_packet.get('alt_m'),
                    'xyz_enu': data_packet.get('xyz_enu'),
                    't_sec':   data_packet.get('t_sec', float(idx)),
                }
                rgb_for_a = self._rgb_tensor_to_uint8_bgr(data_packet['rgb'])
                with self.timer.time('gate_a'):
                    ok_a, score_a = self.gate_a.should_track(meta_a, rgb=rgb_for_a)
                if not ok_a:
                    n_gate_a_rejected += 1
                    if (n_gate_a_rejected % 50) == 1:
                        print(f'[GateA] reject idx={idx} reason={score_a.reject_reason} '
                              f'alt={score_a.alt_m} agl={score_a.agl_m} '
                              f'lap={score_a.lap_var:.1f} mean={score_a.mean_gray:.1f} '
                              f'grad={score_a.grad_density:.3f}')
                    continue
            n_processed += 1

            if 'use_mobile' in self.cfg.keys() and self.cfg['use_mobile']:
                self.tracker.frontend.all_imu   = self.dataset.preload_imu()
                self.tracker.frontend.all_stamp = self.dataset.preload_camtimestamp()

            t_metric = 0.0
            if 'use_metric' in self.cfg.keys() and self.cfg['use_metric']:
                if 'depth' not in data_packet.keys() or data_packet['depth'] is None:
                    with self.timer.time('metric'):
                        data_packet['depth'] = self.metric_predictor.predict(data_packet['rgb'][0])
                    t_metric = self.timer.last('metric')
                # Cache for mapper injection (key = cam timestamp matched against
                # tracker.video.tstamp in viz_out_idx_to_f_idx).
                if data_packet.get('depth') is not None:
                    self.metric_depth_cache[float(data_packet['timestamp'])] = data_packet['depth'].detach().clone()

            with self.timer.time('track.total'):
                self.tracker.track(data_packet if not self.cfg['mode']=='vo_nerfslam' else datapacket_to_nerfslam(data_packet, idx))
                # Per-iteration RTK-Seed in video.poses/disps fuer die NAECHSTE
                # BA-Iteration (anti-scale-collapse-Prior). No-op wenn das
                # Feature im Config nicht aktiviert.
                self._seed_video_poses_with_ext()
            t_track = self.timer.last('track.total')
            t_mf = self.timer.last('track.motion_filter')
            t_fe = self.timer.last('track.frontend_ba')

            # Pose-Override aus externer Pose-Source (z.B. DJI RTK).
            # Wenn der Tracker count_save inkrementiert hat (1+ neue KFs gefreezed),
            # iteriere ueber alle neuen Save-Slots [_last_pose_override_idx, count_save)
            # und schreibe pro Slot k die ext_pose fuer Frame tstamp_save[k]. Die
            # frueher verwendete data_packet['pose']-Variante hat fuer alle Slots
            # die *aktuelle* Frame-Pose geschrieben -- falsch wenn der gefreezte KF
            # eigentlich N Frames zurueck liegt. tstamp_save[k] gibt den Original-
            # Frame-Index fuer Slot k zurueck.
            # Legacy poses_save override (one-shot per marginalized slot)
            # left in for completeness of the history-buffer. The actual
            # mapper override happens AFTER judge_and_package via
            # _apply_ext_poses_to_vizout below -- that's where it lands in
            # viz_out['poses'] which the mapper consumes.
            if (hasattr(self.dataset, 'ext_poses')
                    and self.dataset.ext_poses is not None
                    and self.cfg['mode'] != 'vo_nerfslam'):
                video = self.tracker.video if hasattr(self.tracker, 'video') else self.tracker.frontend.video
                if hasattr(video, 'count_save'):
                    cs = int(video.count_save)
                    last_k = getattr(self, '_last_pose_override_idx', 0)
                    if cs > last_k:
                        for k in range(last_k, cs):
                            frame_idx = int(video.tstamp_save[k].item())
                            if 0 <= frame_idx < len(self.dataset.ext_poses):
                                video.poses_save[k] = torch.as_tensor(
                                    self.dataset.ext_poses[frame_idx], dtype=torch.float32)
                        self._last_pose_override_idx = cs

            # empty_cache() ist Cleanup, kein Algorithmus -- ausserhalb der Phasen-Timer.
            torch.cuda.empty_cache()

            with self.timer.time('judge_pkg'):
                viz_out = judge_and_package(self.tracker, data_packet['intrinsic'])
            t_pkg = self.timer.last('judge_pkg')

            # Pre-Override Drift-Log: schreibt die ROHE Tracker-BA-Pose des
            # neuesten KFs pro Iteration in einen Append-Only-Log. Liefert die
            # vollstaendige Tracker-Historie (statt nur Active-Window-Snapshot
            # am Ende), nuetzlich fuer Drift-Diagnose.
            if viz_out is not None:
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

            # Pose-Override an viz_out: ersetzt DROID-DBA-Posen durch RTK c2w
            # (aus ext_poses_file) UND skaliert depths/depths_cov um den per-window
            # geschaetzten Skalenfaktor, damit Gaussians an metrisch korrekten
            # Positionen platziert werden. Greift nur wenn dataset.ext_poses_file
            # gesetzt ist; legt viz_out unveraendert bei jeder anderen Konfig.
            if viz_out is not None:
                with self.timer.time('ext_pose_apply'):
                    viz_out = self._apply_ext_poses_to_vizout(viz_out)

            t_map = 0.0
            t_prep = 0.0
            t_train = 0.0
            kf_flag = 'N'
            if viz_out is not None and (self.cfg['mode'] in ['vo', 'vo_nerfslam'] or self.tracker.video.imu_enabled):
                kf_flag = 'Y'
                n_keyframes += 1
                # Live GPS-anchored map projection: feed this KF's DROID pose +
                # GPS-ENU and refresh the DROID->three matrix on the frontend.
                self._geo_add_keyframe(viz_out, data_packet)
                # Stash the camera intrinsic the mapper sees, for post-run
                # fair-eval rendering (same {fu,fv,cu,cv,H,W} render() expects).
                if 'intrinsic' in viz_out:
                    self._last_map_intrinsic = viz_out['intrinsic']
                # KF-Filter: entweder VISTA-Selector (wenn konfiguriert) oder
                # naives Modulo-Subsampling. Init-KF wird in beiden Faellen gemappt
                # (Selector akzeptiert ersten Frame; modulo-Pfad: (n_keyframes-1)%N==0).
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
                # Online object detection + 3D-localisation. Runs on every Nth
                # tracker keyframe (object_detect_stride), DECOUPLED from do_map --
                # the FrameSelector filters mapper KFs hard, so tying detection to
                # it loses most objects. On non-mapped KFs viz_out['depths'] is the
                # raw DROID-BA depth (the Metric3D swap is mapper-only), which is
                # exactly what object_tracker.unproject expects (DROID frame); the
                # ext-pose override above already applies here too. Streamed objects
                # carry oriented pose (quat) + size so the frontend draws 3D models.
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
                            if self.stream_server is not None:
                                # Tag each object with the canonical frontend model
                                # key (car/van/truck/bus) so the viewer matches a
                                # glTF asset authoritatively instead of guessing
                                # from the raw detector class string.
                                from vings_utils.detector_base import canonical_model_key
                                objs = self.object_tracker.snapshot()
                                for _o in objs:
                                    _mk = canonical_model_key(_o.get('class'))
                                    if _mk:
                                        _o['model'] = _mk
                                self.stream_server.push({
                                    'type': 'objects', 'epoch': self._stream_epoch,
                                    'objects': objs})
                                # Push THIS frame with the detection boxes drawn in,
                                # so the camera card shows labelled boxes that match
                                # the exact detected frame. Marks the frame as pushed
                                # to skip the generic (box-free) push below.
                                self._stream_push_frame(f_idx, dets=dets)
                                _frame_pushed_idx = f_idx
                    except Exception as _e:
                        print(f"[detect] keyframe skipped: {_e}")
                # stream the original RGB keyframe for the viewer's camera card
                # (own stride, decoupled from mapper/detector). Best-effort. Skip if
                # the detect block already pushed this frame (with boxes drawn).
                if self.stream_server is not None and 'images' in viz_out:
                    _gen_idx = int(viz_out['viz_out_idx_to_f_idx'][-1])
                    _fstride = max(1, int(self.stream_cfg.get('frame_stride', 2)))
                    if _gen_idx != _frame_pushed_idx and (n_keyframes - 1) % _fstride == 0:
                        try:
                            self._stream_push_frame(_gen_idx)
                        except Exception:
                            pass
                if do_map:
                    n_mapped += 1
                    # Snapshot der Sub-Phase-Counts: nur Inkremente dieses Calls anzeigen.
                    _n_prep_before = len(self.timer.records.get('map.add_new_frame', []))
                    _n_train_before = len(self.timer.records.get('map.train_loop', []))
                    # Replace DROID-DBA depths in viz_out with cached Metric3D
                    # depths (matched by tracker.video.tstamp). Keep sky pixels
                    # (rgb==0) at depth=0 so VINGS' sky_mask path still works.
                    # depths_cov is tightened where overwritten -> the
                    # weighted_l1 in get_loss() (weight=1/cov) trusts the prior.
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
                    # Segment each keyframe once (expensive) and stash the masks
                    # on viz_out; the mapper turns them into per-iter dynamic
                    # masks inside the (cheap) training loop.
                    if self.dynamic_model is not None and 'images' in viz_out:
                        with self.timer.time('segment'):
                            viz_out['sam_anns'] = [
                                self.dynamic_model.get_anns_raw(viz_out['images'][i])
                                for i in range(viz_out['images'].shape[0])
                            ]
                    with self.timer.time('map.total'):
                        new_viz_out = self.mapper.run(viz_out, True)
                    t_map = self.timer.last('map.total')
                    if len(self.timer.records.get('map.add_new_frame', [])) > _n_prep_before:
                        t_prep = self.timer.last('map.add_new_frame')
                    if len(self.timer.records.get('map.train_loop', [])) > _n_train_before:
                        t_train = self.timer.last('map.train_loop')
                else:
                    kf_flag = 'S'  # KF vom Tracker, aber Mapper geskippt

                if 'use_loop' in list(self.cfg.keys()) and self.cfg['use_loop']:
                    if viz_out["global_kf_id"][-1] > 10 and viz_out["global_kf_id"][-1] % 3 == 0:
                        with self.timer.time('loop'):
                            self.looper.run(self.mapper, self.tracker, viz_out, idx)
                        # Loop-Closure transformiert frozen Gaussians global ->
                        # alle bereits gestreamten frozen-Daten sind veraltet.
                        # Epoch++ + resync: Frontend leert die Szene, naechster
                        # Push re-streamt das gesamte frozen-Set neu.
                        if self.stream_server is not None:
                            self._stream_epoch += 1
                            self._streamed_kf_ids.clear()
                            self._active_sig.clear()
                            self.stream_server.push({'type': 'resync',
                                                     'epoch': self._stream_epoch})

                if self.use_storage_manager and (idx+1) % 10 == 0:
                    with self.timer.time('storage'):
                        self.storage_manager.run(self.tracker, self.mapper, viz_out)
                    torch.cuda.empty_cache()

                # Live-Stream der Gaussians zum Frontend. Nach dem Storage-Run,
                # damit das frozen-Set (CPU) aktuell ist. Alle stream.every_kf
                # gemappten KFs; non-blocking (drop-oldest-Queue im Server).
                if self.stream_server is not None and do_map and n_mapped > 0:
                    stream_every = int(self.stream_cfg.get('every_kf', 1))
                    if stream_every <= 1 or n_mapped % stream_every == 0:
                        with self.timer.time('stream'):
                            self._stream_push_gaussians(idx)

                # Periodischer PLY-Checkpoint (Crash-Schutz). Wenn der Run vom
                # vram-watchdog/OOMd SIGKILLed wird, hat man wenigstens einen
                # partiellen PLY-Snapshot. Default = 0 (aus).
                ply_ckpt = int(self.cfg.get('ply_checkpoint_every_kf', 0))
                if ply_ckpt > 0 and do_map and n_mapped > 0 and n_mapped % ply_ckpt == 0:
                    try:
                        torch.cuda.empty_cache()
                        sm = self.storage_manager if self.use_storage_manager else None
                        with self.timer.time('save_ply_ckpt'):
                            save_ply_streaming(self.mapper, sm, idx, save_mode='2dgs')
                        print(f"[PLY-CKPT] idx={idx} n_mapped={n_mapped} written.")
                    except Exception as e:
                        print(f"[PLY-CKPT] failed at idx={idx}: {e}")

                if self.cfg['use_vis'] and (idx+1) % 1 == 0:
                    with self.timer.time('vis'):
                        if not self.cfg['use_storage_manager'] or self.storage_manager._xyz.shape[0]==0:
                            vis_map(self.tracker, self.mapper)
                            vis_bev(self.tracker, self.mapper)
                        else:
                            self.storage_manager.vis_map_storage(self.tracker, self.mapper)
                            self.storage_manager.vis_bev_storage(self.tracker, self.mapper)

            t_total = t_metric + t_track + t_pkg + t_map
            if log_every <= 1 or idx % log_every == 0:
                # Selector-Score anhaengen, wenn vorhanden -- hilft beim
                # threshold_Q-Tuning bzw. zur Diagnose, warum Frames rejected werden.
                fs_str = ""
                fs_score_local = locals().get('fs_score', None)
                if fs_score_local is not None and hasattr(fs_score_local, 'Q'):
                    # NURBS-LVI: Q ist Schwelle, migration (=Or+Oc) ist die Evidenz.
                    fs_str = (f" fs(mig={fs_score_local.migration:+.1f}"
                              f" Q={fs_score_local.Q:+.2f}"
                              f" phi={fs_score_local.phi:+.2f}"
                              f" m={fs_score_local.n_matches})")
                elif isinstance(fs_score_local, (int, float)):
                    fs_str = f" fs(g={fs_score_local:.3f})"
                elif fs_score_local is not None and hasattr(fs_score_local, 'triggered_by'):
                    # two_gate / two_gate_v2: emit the exact pipeline step that
                    # decided this frame, frame-indexed, so notebooks can plot
                    # per-frame WHY a frame was dropped (B1_*/B2B3_*/force/budget).
                    fs_str = f" fs(step={fs_score_local.triggered_by})"
                print(f"[{idx:5d}] kf={kf_flag} metric={t_metric:.3f} "
                      f"track={t_track:.3f}(mf={t_mf:.3f} fe={t_fe:.3f}) "
                      f"pkg={t_pkg:.3f} "
                      f"map={t_map:.3f}(prep={t_prep:.3f} train={t_train:.3f}) "
                      f"total={t_total:.3f}{fs_str}")

            last_idx = idx
            if snapshot_every > 0 and n_keyframes > 0 and n_keyframes % snapshot_every == 0:
                self._write_profiling(n_keyframes, n_mapped, n_processed, wall_t0,
                                       frame_skip, mapper_kf_skip, last_idx, partial=True)

        # PLY-Save: chunk-weise schreiben, um Peak-RAM zu minimieren.
        # save_ply_streaming iteriert über StorageManager (CPU) und Mapper (GPU)
        # getrennt in Chunks von 500k Gaussians (~120 MB/Chunk statt ~5 GB auf einmal).
        n_cpu = self.storage_manager._xyz.shape[0] if self.use_storage_manager else 0
        n_gpu = self.mapper._xyz.shape[0]
        if n_cpu + n_gpu > 0:
            sm = self.storage_manager if self.use_storage_manager else None
            with self.timer.time('save_ply'):
                save_ply_streaming(self.mapper, sm, len(self.dataset) - 1, save_mode='2dgs')

        # Fuse + write the online object detections (objects_droid.csv,
        # object_markers_droid.ply, object_overlay.mp4). Markers live in the
        # same DROID frame as the map PLY just written above.
        if self.object_tracker is not None:
            try:
                self.object_tracker.finalize(self.cfg['output']['save_dir'])
            except Exception as _e:
                print(f"[object_tracker] finalize failed: {_e}")

        # WebSocket-Stream-Server stoppen (daemon-Thread; harmlos wenn aus).
        if self.stream_server is not None:
            try:
                self.stream_server.stop()
            except Exception as _e:
                print(f"[stream] stop failed: {_e}")

        # Faire, selektionsunabhaengige Eval (Sim(3)-ATE + Held-out-Novel-View-
        # PSNR an FIXEN Frame-Positionen aus der finalen Map). Gated ueber
        # cfg['fair_eval']['enabled']; Mapper ist hier noch GPU-resident.
        if (self.cfg.get('fair_eval', {}) or {}).get('enabled', False):
            try:
                video = (self.tracker.video if hasattr(self.tracker, 'video')
                         else self.tracker.frontend.video)
                intr = getattr(self, '_last_map_intrinsic', None)
                if intr is None:
                    print('[fair_eval] no map intrinsic captured (no KF mapped?); skipping.')
                else:
                    with self.timer.time('fair_eval'):
                        run_fair_eval(self.mapper, video, self.cfg, intr,
                                      self.cfg['output']['save_dir'])
            except Exception as _e:
                import traceback
                print(f"[fair_eval] failed: {_e}")
                traceback.print_exc()

        # Dump finale Tracker-Posen (w2c in TUM-tq) fuer Drift-Diagnose.
        # Kombiniert MARGINALISIERTE KFs (poses_save[:count_save]) + ACTIVE-Window
        # (poses[:counter.value]) -- die zweite Quelle ist wichtig bei kurzen
        # Sequenzen wo viele KFs noch nicht marginalisiert wurden.
        try:
            video = (self.tracker.video if hasattr(self.tracker, 'video')
                     else self.tracker.frontend.video)
            chunks = []
            # 1) Marginalisierte (append-only history)
            poses_save = video.poses_save.detach().cpu().numpy()
            n_save = int(getattr(video, 'count_save', 0))
            n_save = max(0, min(n_save, poses_save.shape[0]))
            if n_save > 0:
                chunks.append(('marg', poses_save[:n_save]))
            # 2) Active-Window-Posen (aktuelle BA-Schaetzungen)
            poses_act = video.poses.detach().cpu().numpy()
            counter_val = getattr(video, 'counter', None)
            n_act = int(counter_val.value) if counter_val is not None else 0
            n_act = max(0, min(n_act, poses_act.shape[0]))
            if n_act > 0:
                chunks.append(('act', poses_act[:n_act]))
            if chunks:
                rows = []
                idx = 0
                for src, arr in chunks:
                    for tq in arr:
                        rows.append([idx, src] + [float(x) for x in tq])
                        idx += 1
                # Write with mixed int/str/float: do it manually since np.savetxt struggles.
                out_path = os.path.join(self.cfg['output']['save_dir'],
                                        'tracker_poses_w2c.txt')
                with open(out_path, 'w') as f:
                    f.write('# idx src tx ty tz qx qy qz qw  (w2c, VINGS)\n')
                    for r in rows:
                        f.write(f"{r[0]} {r[1]} " + " ".join(f"{v:.6f}" for v in r[2:]) + "\n")
                print(f"tracker_poses_w2c.txt geschrieben "
                      f"({n_save} marginalisiert + {n_act} aktiv).")
        except Exception as _e:
            print(f"[WARN] tracker_poses_w2c dump failed: {_e}")

        wall_total = time.time() - wall_t0
        print(f"\n=== Profiling Summary ({n_keyframes} KFs, {n_mapped} mapped "
              f"/ {n_processed} processed / {len(self.dataset)} dataset, "
              f"frame_skip={frame_skip}, mapper_kf_skip={mapper_kf_skip}, "
              f"wall={wall_total:.1f}s) ===")
        self.timer.summary(total_wall=wall_total)

        self._write_profiling(n_keyframes, n_mapped, n_processed, wall_t0,
                              frame_skip, mapper_kf_skip, last_idx, partial=False)
        print(f"profiling.json -> {os.path.join(self.cfg['output']['save_dir'], 'profiling.json')}")
            

if __name__ == '__main__':
    
    config_basename = os.path.basename(config_path)
    if config_basename.endswith('.yaml'):
        config_basename = config_basename[:-5]
    config['output']['save_dir'] = os.path.join(config['output']['save_dir'], get_name(config)+'-{}-'.format(config_basename)+args.prefix)
    os.makedirs(config['output']['save_dir']+'/droid_c2w', exist_ok=True)
    os.makedirs(config['output']['save_dir']+'/rgbdnua', exist_ok=True)
    os.makedirs(config['output']['save_dir']+'/ply', exist_ok=True)
    if 'debug_mode' in list(config.keys()) and config['debug_mode']:
        os.makedirs(config['output']['save_dir']+'/debug_dict', exist_ok=True)
    shutil.copy(config_path, config['output']['save_dir']+'/config.yaml')
    
    runner = Runner(config)
    torch.backends.cudnn.benchmark = True
    
    runner.run()
    
    