import numpy as np
import torch
import os
from frontend.dbaf import DBAFusion
import argparse
parser = argparse.ArgumentParser(description="Add config path.")
parser.add_argument("config")
parser.add_argument("--prefix", default='')
args = parser.parse_args()
config_path = args.config
from gaussian.general_utils import load_config, get_name
config = load_config(config_path)
config['frontend']['show_plot'] = True

import shutil
import importlib
get_dataset = importlib.import_module(config["dataset"]["module"]).get_dataset
from vings_utils.middleware_utils import judge_and_package

import open3d as o3d
from lietorch import SE3
from gaussian.vis_utils import check_pcd_with_poses
from metric.metric_model import Metric_Model
from metric.depth_factory import make_depth_model

class Runner:
    def __init__(self, cfg):
        self.cfg = cfg
        self.dataset  = get_dataset(cfg)
        cfg['frontend']['c2i'] = self.dataset.c2i # (4, 4), ndarray
        self.tracker = DBAFusion(cfg)
        if 'use_metric' in cfg.keys() and cfg['use_metric']:
            self.metric_predictor = make_depth_model(cfg)
            
            
    def run(self):
        # Load imu data.
        self.tracker.frontend.all_imu   = self.dataset.preload_imu()
        self.tracker.frontend.all_stamp = self.dataset.preload_camtimestamp()
        print(self.tracker.frontend.all_imu[:5,:])
        # Run Tracking.
        for idx in range(0, min(len(self.dataset), 30000)):
            data_packet = self.dataset[idx]
            
            if 'use_metric' in self.cfg.keys() and self.cfg['use_metric']:
                # data_packet['depth'] = self.metric_predictor.predict(data_packet['rgb'][0])
                # print('depth.shape: ', data_packet['depth'].shape)
                pass
            
            self.tracker.track(data_packet)
            
            # Save poses.
            if (idx+1) % 500 == 0 or idx == len(self.dataset)-1:
                tstamp_save = self.tracker.frontend.video.tstamp_save[:self.tracker.frontend.video.count_save].cpu().tolist()
                poses_save  = self.tracker.frontend.video.poses_save[:self.tracker.frontend.video.count_save].cpu()
                
                for iiiddxx in range(len(tstamp_save)):
                    timestamp = tstamp_save[iiiddxx]
                    c2w = SE3(poses_save[iiiddxx]).inv().matrix().cpu().numpy()
                    np.savetxt(f"{self.cfg['output']['save_dir']}/droid_c2w/{timestamp}.txt", c2w)
            
            
            # TTD 2024/12/06
            if 'debug_mode' in list(self.cfg.keys()) and self.cfg['debug_mode']:
                viz_out = judge_and_package(self.tracker, data_packet['intrinsic'])
                if viz_out is not None:
                    start_timestamp = viz_out['viz_out_idx_to_f_idx'][0].item()
                    end_timestamp   = viz_out['viz_out_idx_to_f_idx'][-1].item()
                    # save_name = os.path.join(config['output']['save_dir'], 'vizout_dict', '{}to{}.pth'.format(round(start_timestamp, 4), round(end_timestamp, 4))) 
                    save_name = os.path.join(config['output']['save_dir'], 'vizout_dict', '{}to{}.pth'.format(str(float(start_timestamp)).zfill(12), str(float(end_timestamp)).zfill(12))) 
                    
                    torch.save(viz_out, save_name)
            
            # torch.cuda.empty_cache()
            # Judge whether new keyframe is added and package keyframe dict.
            # viz_out = judge_and_package(self.tracker, data_packet['intrinsic'])
            
            # if viz_out is not None and idx % 10 == 0:
            #    self.save_c2ws(idx, viz_out)
            # if viz_out is not None and self.tracker.video.imu_enabled:
            #    save_dir = '/data/wuke/DATA/2024/DroidOutput/KITTI/2011_10_03_drive0027_sync/process_dict/'
            #    torch.save(viz_out, f"{save_dir}/{str(idx).zfill(6)}.pth")

        # Save Poses' ply.
        SAVE_POSES = True
        if SAVE_POSES:
            c2ws      = SE3(self.tracker.video.poses_save[:self.tracker.video.count_save]).inv().matrix().detach().cpu()
            poses_ply = check_pcd_with_poses(None, c2ws)
            o3d.io.write_point_cloud(f"{self.cfg['output']['save_dir']}/ply/all_c2ws.ply", poses_ply)
        
        
    def save_c2ws(self, global_idx, viz_out):
        torch.save(viz_out['viz_out_idx_to_f_idx'], f"/data/wuke/workspace/VINGS-Mono/debug/timestamps_{global_idx}.pth")
        torch.save(viz_out['poses'], f"/data/wuke/workspace/VINGS-Mono/debug/c2ws_{global_idx}.pth")
        torch.save(viz_out['depths_cov'], f"/data/wuke/workspace/VINGS-Mono/debug/depths_cov_up_{global_idx}.pth")
        torch.save(viz_out['depths'], f"/data/wuke/workspace/VINGS-Mono/debug/depths_{global_idx}.pth")

        
if __name__ == '__main__':
    config['output']['save_dir'] = os.path.join(config['output']['save_dir'], get_name(config)+'-{}-'.format(config_path.split('/')[-1].strip('.yaml'))+args.prefix)
    os.makedirs(config['output']['save_dir']+'/ply', exist_ok=True)
    os.makedirs(config['output']['save_dir']+'/droid_c2w', exist_ok=True)
    if 'debug_mode' in list(config.keys()) and config['debug_mode']:
        os.makedirs(config['output']['save_dir']+'/vizout_dict', exist_ok=True)
    shutil.copy(config_path, config['output']['save_dir']+'/config.yaml')
    runner = Runner(config)
    torch.backends.cudnn.benchmark = True
    runner.run()