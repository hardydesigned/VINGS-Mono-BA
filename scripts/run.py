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
from vings_utils.phase_timer import PhaseTimer, write_profiling_json
from vings_utils.ext_pose_override import ExtPoseOverrider
from vings_utils.keyframe_pipeline import KeyframePipeline
from vings_utils.run_finalize import finalize_run
from server.stream_publisher import StreamPublisher
from storage.storage_manage import StorageManager
from loop.loop_model import LoopModel
from metric.metric_model import Metric_Model
import time
from tqdm import tqdm
if config['mode'] == 'vo_nerfslam': from frontend_vo.vio_slam import VioSLAM


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
        self.stream_cfg = (cfg.get('stream') or {})
        self.stream = StreamPublisher(cfg)

        # Externe-Pose-Override (RTK/DJI) + Per-Keyframe-Pipeline (Select /
        # Detection / Mapper-Prep). Beide no-op-freundlich, wenn das jeweilige
        # Feature im Config aus ist.
        self.ext_pose = ExtPoseOverrider(cfg, self.dataset)
        self.kf_pipeline = KeyframePipeline(
            cfg, frame_selector=self.frame_selector,
            object_detector=self.object_detector,
            object_tracker=self.object_tracker,
            dynamic_model=self.dynamic_model,
            metric_depth_cache=self.metric_depth_cache,
            object_detect_stride=self.object_detect_stride,
            timer=self.timer, dataset=self.dataset, stream=self.stream)

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
                self.ext_pose.seed_video_poses(self.tracker)
            t_track = self.timer.last('track.total')
            t_mf = self.timer.last('track.motion_filter')
            t_fe = self.timer.last('track.frontend_ba')

            # Legacy poses_save override (one-shot per marginalized slot) --
            # haelt den History-Buffer konsistent; der eigentliche Mapper-Override
            # passiert unten via ext_pose.apply_to_vizout. No-op ohne ext_poses.
            self.ext_pose.override_poses_save(self.tracker)

            torch.cuda.empty_cache()

            with self.timer.time('judge_pkg'):
                viz_out = judge_and_package(self.tracker, data_packet['intrinsic'])
            t_pkg = self.timer.last('judge_pkg')

            # Pre-Override Drift-Log der rohen Tracker-BA-Pose (vor ext-pose-Override).
            self.kf_pipeline.log_raw_tracker_pose(viz_out, data_packet, idx, n_keyframes)

            # Pose-Override an viz_out: ersetzt DROID-DBA-Posen durch RTK c2w
            # (aus ext_poses_file) UND skaliert depths/depths_cov um den per-window
            # geschaetzten Skalenfaktor, damit Gaussians an metrisch korrekten
            # Positionen platziert werden. Greift nur wenn dataset.ext_poses_file
            # gesetzt ist; legt viz_out unveraendert bei jeder anderen Konfig.
            if viz_out is not None:
                with self.timer.time('ext_pose_apply'):
                    viz_out = self.ext_pose.apply_to_vizout(viz_out)

            t_map = 0.0
            t_prep = 0.0
            t_train = 0.0
            kf_flag = 'N'
            fs_score = None
            if viz_out is not None and (self.cfg['mode'] in ['vo', 'vo_nerfslam'] or self.tracker.video.imu_enabled):
                kf_flag = 'Y'
                n_keyframes += 1
                # Live GPS-anchored map projection: feed this KF's DROID pose +
                # GPS-ENU and refresh the DROID->three matrix on the frontend.
                self.stream.add_keyframe_geo(viz_out, data_packet)
                # Stash the camera intrinsic the mapper sees, for post-run
                # fair-eval rendering (same {fu,fv,cu,cv,H,W} render() expects).
                if 'intrinsic' in viz_out:
                    self._last_map_intrinsic = viz_out['intrinsic']
                # KF-Filter: FrameSelector (wenn konfiguriert) oder naives
                # mapper_kf_skip-Modulo. Init-KF wird in beiden Faellen gemappt.
                do_map, fs_score = self.kf_pipeline.decide_mapping(
                    viz_out, data_packet, idx, n_keyframes, mapper_kf_skip)
                # Online-Objektdetektion + 3D-Lokalisierung (eigener Stride,
                # entkoppelt von do_map) inkl. Objekt-/Frame-Push ans Frontend.
                self.kf_pipeline.run_detection(viz_out, data_packet, n_keyframes)
                if do_map:
                    n_mapped += 1
                    # Snapshot der Sub-Phase-Counts: nur Inkremente dieses Calls anzeigen.
                    _n_prep_before = len(self.timer.records.get('map.add_new_frame', []))
                    _n_train_before = len(self.timer.records.get('map.train_loop', []))
                    # Mapper-Vorbereitung: Metric3D-Depth-Swap + Dynamic-Seg auf viz_out.
                    self.kf_pipeline.prepare_for_mapper(viz_out)
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
                        self.stream.resync()

                if self.use_storage_manager and (idx+1) % 10 == 0:
                    with self.timer.time('storage'):
                        self.storage_manager.run(self.tracker, self.mapper, viz_out)
                    torch.cuda.empty_cache()

                # Live-Stream der Gaussians zum Frontend. Nach dem Storage-Run,
                # damit das frozen-Set (CPU) aktuell ist. Alle stream.every_kf
                # gemappten KFs; non-blocking (drop-oldest-Queue im Server).
                if self.stream.enabled and do_map and n_mapped > 0:
                    stream_every = int(self.stream_cfg.get('every_kf', 1))
                    if stream_every <= 1 or n_mapped % stream_every == 0:
                        sm = self.storage_manager if self.use_storage_manager else None
                        with self.timer.time('stream'):
                            self.stream.push_gaussians(self.mapper, sm, idx)

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
                fs_score_local = fs_score
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
                write_profiling_json(self.timer, self.cfg,
                                     n_keyframes=n_keyframes, n_mapped=n_mapped,
                                     n_processed=n_processed, n_frames=len(self.dataset),
                                     last_idx=last_idx, frame_skip=frame_skip,
                                     mapper_kf_skip=mapper_kf_skip, wall_t0=wall_t0,
                                     partial=True)

        # Post-Run-Finalisierung: PLY-Save, Objekt-Fusion, Stream-Stop, fair-eval,
        # Tracker-Pose-Dump und Profiling-Summary (siehe vings_utils/run_finalize.py).
        finalize_run(self, n_keyframes=n_keyframes, n_mapped=n_mapped,
                     n_processed=n_processed, last_idx=last_idx,
                     frame_skip=frame_skip, mapper_kf_skip=mapper_kf_skip,
                     wall_t0=wall_t0)


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