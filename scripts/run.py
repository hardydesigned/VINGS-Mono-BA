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
        
        if 'use_metric' in cfg.keys() and cfg['use_metric']:
            self.metric_predictor = Metric_Model(cfg) 
        
        if 'use_storage_manager' in cfg.keys() and cfg['use_storage_manager']:
            self.use_storage_manager = True
            self.storage_manager = StorageManager(cfg)
            if cfg['dataset']['module'] != 'phone':
                self.storage_manager.dataset_length = self.dataset.rgbinfo_dict['timestamp'][-1] - self.dataset.rgbinfo_dict['timestamp'][0] 
        else:
            self.use_storage_manager = False

    def run(self):
        # Load imu data.
        self.tracker.frontend.all_imu   = self.dataset.preload_imu()
        self.tracker.frontend.all_stamp = self.dataset.preload_camtimestamp()
        
        mapper_run_times = 0
        
        # Run Tracking.
        for idx in tqdm(range(len(self.dataset))):
            
            data_packet = self.dataset[idx]
            
            if 'use_mobile' in self.cfg.keys() and self.cfg['use_mobile']:
                self.tracker.frontend.all_imu   = self.dataset.preload_imu()
                self.tracker.frontend.all_stamp = self.dataset.preload_camtimestamp()
            
            if 'use_metric' in self.cfg.keys() and self.cfg['use_metric']:
                if 'depth' not in data_packet.keys() or data_packet['depth'] is None:
                    data_packet['depth'] = self.metric_predictor.predict(data_packet['rgb'][0])
            
            # torch.set_grad_enabled(False)
            self.tracker.track(data_packet if not self.cfg['mode']=='vo_nerfslam' else datapacket_to_nerfslam(data_packet, idx))
            # torch.set_grad_enabled(True)
            
            torch.cuda.empty_cache()
            # Judge whether new keyframe is added and package keyframe dict.
            viz_out = judge_and_package(self.tracker, data_packet['intrinsic'])
            
            if viz_out is not None and (self.cfg['mode'] in ['vo', 'vo_nerfslam'] or self.tracker.video.imu_enabled):
                # Save and check.
                new_viz_out = self.mapper.run(viz_out, True)
                
                if 'use_loop' in list(self.cfg.keys()) and self.cfg['use_loop']:
                    if viz_out["global_kf_id"][-1] > 10 and viz_out["global_kf_id"][-1] % 3 == 0:
                        self.looper.run(self.mapper, self.tracker, viz_out, idx)

                if self.use_storage_manager and (idx+1) % 10 == 0:
                    self.storage_manager.run(self.tracker, self.mapper, viz_out)
                    torch.cuda.empty_cache()
                
                if self.cfg['use_vis'] and (idx+1) % 1 == 0:
                    if not self.cfg['use_storage_manager'] or self.storage_manager._xyz.shape[0]==0:
                        vis_map(self.tracker, self.mapper)
                        vis_bev(self.tracker, self.mapper) 
                    else:
                        self.storage_manager.vis_map_storage(self.tracker, self.mapper)    
                        self.storage_manager.vis_bev_storage(self.tracker, self.mapper)    
            
            if (idx == len(self.dataset) - 1) and self.mapper._xyz.shape[0] > 0:
            # if ((idx+1) % 100 == 0 or (idx == len(self.dataset) - 1)) and self.mapper._xyz.shape[0] > 0:
                save_ply(self.mapper, idx, save_mode='2dgs')
                # save_ply(self.mapper, idx, save_mode='pth')
            

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
    
    