import numpy as np
import shutil
import torch
from lietorch import SE3
import os
from frontend.dbaf import DBAFusion
from gaussian.gaussian_model import GaussianModel
from gaussian.vis_utils import save_ply, vis_map, vis_bev
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
from storage.storage_manage import StorageManager
from loop.loop_model import LoopModel
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
        """Wrappt obj.attr so, dass jeder Aufruf in Phase 'name' geht.
        Funktioniert nur fuer reine Methoden -- nicht fuer Callable-Objekte
        die zusaetzlich Attribute tragen (siehe patch_callable)."""
        orig = getattr(obj, attr)
        timer = self
        def wrapper(*args, **kwargs):
            with timer.time(name):
                return orig(*args, **kwargs)
        setattr(obj, attr, wrapper)

    def patch_callable(self, obj, attr, name):
        """Wrappt ein Callable-Objekt (mit eigenen Attributen) via Proxy.
        Attribut-Zugriff/Set wird transparent ans Original weitergegeben."""
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


class _MergedForSave:
    """Leichter Proxy fuer save_ply: konkateniert Mapper-GPU- und
    StorageManager-CPU-Gaussians auf der CPU, ohne Optimizer/nn.Parameter.
    Vermeidet OOM beim finalen Save grosser Szenen — save_ply lifet nur
    den kleinen RGB-Tensor kurz auf die GPU.
    """
    def __init__(self, mapper, sm):
        cat = lambda a, b: torch.cat([a.detach().cpu(), b.detach().cpu()], dim=0)
        self._xyz         = cat(mapper._xyz,         sm._xyz)
        self._rgb         = cat(mapper._rgb,         sm._rgb)
        self._scaling     = cat(mapper._scaling,     sm._scaling)
        self._rotation    = cat(mapper._rotation,    sm._rotation)
        self._opacity     = cat(mapper._opacity,     sm._opacity)
        self._globalkf_id = cat(mapper._globalkf_id, sm._globalkf_id)
        self.cfg            = mapper.cfg
        self.tfer           = mapper.tfer
        self.activate_dict  = mapper.activate_dict

    def get_property(self, name):
        if name == '_xyz':      return self._xyz
        if name == '_rgb':      return self._rgb
        if name == '_opacity':  return self.activate_dict['_opacity'](self._opacity)
        if name == '_rotation': return self.activate_dict['_rotation'](self._rotation)
        if name == '_scaling':  return self.activate_dict['_scaling'](self._scaling)
        if name == '_zeros':    return torch.zeros_like(self._xyz[:, :2])
        raise ValueError(f"Invalid property name: {name}")


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
        
        if 'use_metric' in cfg.keys() and cfg['use_metric']:
            self.metric_predictor = Metric_Model(cfg) 
        
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

        # Profiling-Counter und Print-Throttle.
        n_keyframes = 0
        n_processed = 0
        log_every = int(self.cfg.get('profiling', {}).get('log_every', 1))
        # Naive temporal sampling: nimm nur jedes N-te Eingangs-Frame.
        frame_skip = int(self.cfg.get('frame_skip', 1))
        wall_t0 = time.time()

        # Run Tracking.
        for idx in tqdm(range(len(self.dataset))):
            if frame_skip > 1 and idx % frame_skip != 0:
                continue
            n_processed += 1

            data_packet = self.dataset[idx]

            if 'use_mobile' in self.cfg.keys() and self.cfg['use_mobile']:
                self.tracker.frontend.all_imu   = self.dataset.preload_imu()
                self.tracker.frontend.all_stamp = self.dataset.preload_camtimestamp()

            t_metric = 0.0
            if 'use_metric' in self.cfg.keys() and self.cfg['use_metric']:
                if 'depth' not in data_packet.keys() or data_packet['depth'] is None:
                    with self.timer.time('metric'):
                        data_packet['depth'] = self.metric_predictor.predict(data_packet['rgb'][0])
                    t_metric = self.timer.last('metric')

            with self.timer.time('track.total'):
                self.tracker.track(data_packet if not self.cfg['mode']=='vo_nerfslam' else datapacket_to_nerfslam(data_packet, idx))
            t_track = self.timer.last('track.total')
            t_mf = self.timer.last('track.motion_filter')
            t_fe = self.timer.last('track.frontend_ba')

            # empty_cache() ist Cleanup, kein Algorithmus -- ausserhalb der Phasen-Timer.
            torch.cuda.empty_cache()

            with self.timer.time('judge_pkg'):
                viz_out = judge_and_package(self.tracker, data_packet['intrinsic'])
            t_pkg = self.timer.last('judge_pkg')

            t_map = 0.0
            t_prep = 0.0
            t_train = 0.0
            kf_flag = 'N'
            if viz_out is not None and (self.cfg['mode'] in ['vo', 'vo_nerfslam'] or self.tracker.video.imu_enabled):
                kf_flag = 'Y'
                n_keyframes += 1
                # Snapshot der Sub-Phase-Counts: nur Inkremente dieses Calls anzeigen.
                _n_prep_before = len(self.timer.records.get('map.add_new_frame', []))
                _n_train_before = len(self.timer.records.get('map.train_loop', []))
                with self.timer.time('map.total'):
                    new_viz_out = self.mapper.run(viz_out, True)
                t_map = self.timer.last('map.total')
                if len(self.timer.records.get('map.add_new_frame', [])) > _n_prep_before:
                    t_prep = self.timer.last('map.add_new_frame')
                if len(self.timer.records.get('map.train_loop', [])) > _n_train_before:
                    t_train = self.timer.last('map.train_loop')

                if 'use_loop' in list(self.cfg.keys()) and self.cfg['use_loop']:
                    if viz_out["global_kf_id"][-1] > 10 and viz_out["global_kf_id"][-1] % 3 == 0:
                        with self.timer.time('loop'):
                            self.looper.run(self.mapper, self.tracker, viz_out, idx)

                if self.use_storage_manager and (idx+1) % 10 == 0:
                    with self.timer.time('storage'):
                        self.storage_manager.run(self.tracker, self.mapper, viz_out)
                    torch.cuda.empty_cache()

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
                print(f"[{idx:5d}] kf={kf_flag} metric={t_metric:.3f} "
                      f"track={t_track:.3f}(mf={t_mf:.3f} fe={t_fe:.3f}) "
                      f"pkg={t_pkg:.3f} "
                      f"map={t_map:.3f}(prep={t_prep:.3f} train={t_train:.3f}) "
                      f"total={t_total:.3f}")

        # Storage-Manager-Bug-Fix: CPU-Gaussians muessen mit in die finale PLY.
        # Statt cpu2gpu() (laedt alles + Optimizer-State auf GPU -> OOM-Risiko)
        # bauen wir einen leichten Proxy mit CPU-konkatenierten Tensoren.
        # save_ply zieht nur das kleine RGB-Stueck kurz auf die GPU.
        if self.use_storage_manager and self.storage_manager._xyz.shape[0] > 0:
            print(f"Merging {self.storage_manager._xyz.shape[0]} CPU + "
                  f"{self.mapper._xyz.shape[0]} GPU Gaussians for PLY save "
                  f"(CPU-side, no bulk GPU upload).")
            save_target = _MergedForSave(self.mapper, self.storage_manager)
        else:
            save_target = self.mapper

        # save_ply ausserhalb der Loop: bei frame_skip>1 wuerde der letzte
        # idx (len-1) sonst vom continue verworfen und nichts gespeichert.
        if save_target._xyz.shape[0] > 0:
            with self.timer.time('save_ply'):
                save_ply(save_target, len(self.dataset) - 1, save_mode='2dgs')

        wall_total = time.time() - wall_t0
        print(f"\n=== Profiling Summary ({n_keyframes} KFs / {n_processed} processed "
              f"/ {len(self.dataset)} dataset, frame_skip={frame_skip}, "
              f"wall={wall_total:.1f}s) ===")
        self.timer.summary(total_wall=wall_total)

        try:
            out_path = os.path.join(self.cfg['output']['save_dir'], 'profiling.json')
            with open(out_path, 'w') as f:
                json.dump({
                    'wall_total_s': wall_total,
                    'n_keyframes': n_keyframes,
                    'n_processed': n_processed,
                    'n_frames': len(self.dataset),
                    'frame_skip': frame_skip,
                    'records': self.timer.records,
                }, f)
            print(f"profiling.json -> {out_path}")
        except Exception as e:
            print(f"profiling.json write failed: {e}")
            

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
    
    