import numpy as np
import shutil
import torch
import os
from frontend.dbaf import DBAFusion
from gaussian.gaussian_model import GaussianModel
from gaussian.vis_utils import save_ply
import argparse
parser = argparse.ArgumentParser(description="Add config path.")
parser.add_argument("config")
args = parser.parse_args()
config_path = args.config
from gaussian.general_utils import load_config, get_name
config = load_config(config_path)
import importlib
get_dataset = importlib.import_module(config["dataset"]["module"]).get_dataset
from vings_utils.middleware_utils import judge_and_package
import torch.multiprocessing as mp
from queue import Queue
import time
import gc

from metric.metric_model import Metric_Model
from metric.depth_factory import make_depth_model
from storage.storage_manage import StorageManager

# TTD 2024/12/29
class FakeVideo:
    def __init__(self):
        pass
    
class FakeTracker:
    '''
    Use for debug storage manager.
    '''
    def __init__(self, cfg):
        self.cfg = cfg
        # It seems like I only need to to use one video object.
        self.video = FakeVideo
        self.video.poses_save = torch.empty(0)

    
def tracking(cfg, tracker2mapper_queue):
    torch.backends.cudnn.benchmark = True
    dataset  = get_dataset(cfg)
    cfg['frontend']['c2i'] = dataset.c2i
    tracker = DBAFusion(cfg)
    tracker.frontend.all_imu   = dataset.preload_imu()
    tracker.frontend.all_stamp = dataset.preload_camtimestamp()
    
    if 'use_metric' in cfg.keys() and cfg['use_metric']:
        metric_predictor = make_depth_model(cfg)
        
        
    idx = 0
    while True:
        if tracker2mapper_queue.qsize() < 5:
            # print(f"Tracker IDX: {idx}.")
            
            # For new frame.
            # dataset.preload_rgbinfo()
            # dataset.preload_imu()
            # dataset.preload_camtimestamp()
            data_packet = dataset[idx] # __getitem__
            if 'use_mobile' in cfg.keys() and cfg['use_mobile']:
                tracker.frontend.all_imu   = dataset.preload_imu()
                tracker.frontend.all_stamp = dataset.preload_camtimestamp()
            
            if 'use_metric' in cfg.keys() and cfg['use_metric']:
               data_packet['depth'] = metric_predictor.predict(data_packet['rgb'][0])
            
            tracker.track(data_packet)
            torch.cuda.empty_cache()
            # Judge whether new keyframe is added and package keyframe dict.
            viz_out = judge_and_package(tracker, data_packet['intrinsic'])
            # TTD 2024/12/30
            # We only need tracker.video.poses_save[:viz_out['global_kf_id'][-1]]
            if viz_out is not None and (tracker.cfg['mode']=='vo' or tracker.video.imu_enabled):
                if idx > 50:
                    viz_out["tracker"] = {}
                    viz_out["tracker"]["poses_save"] = tracker.video.poses_save[:viz_out['global_kf_id'][-1]]
                tracker2mapper_queue.put(viz_out)
            idx += 1
            torch.cuda.empty_cache()
        else:
            time.sleep(0.1)


def mapping(cfg, tracker2mapper_queue):
    mapper = GaussianModel(cfg)
    fake_tracker    = FakeTracker(cfg)
    storage_manager = StorageManager(cfg)
    storage_manager.dataset_length = None # TODO: fix this.
    mapping_idx = 0
    while True:
        # Run mapping.
        if tracker2mapper_queue.qsize() > 0:
            viz_out = tracker2mapper_queue.get()
            mapper.run(viz_out)
            mapping_idx += 1
            torch.cuda.empty_cache()
            if "use_storage_manager" in cfg.keys() and cfg["use_storage_manager"]:
                if "tracker" in viz_out.keys() and mapping_idx % 4 == 0:
                    fake_tracker.video.poses_save = viz_out["tracker"]["poses_save"]
                    storage_manager.run(fake_tracker, mapper, viz_out)
                    torch.cuda.empty_cache()
            if mapping_idx % 10 == 0:
                print(f"Mapping IDX: {mapping_idx}.")
        else:
            time.sleep(0.1) 
        

    
        
if __name__ == '__main__':
    config['output']['save_dir'] = os.path.join(config['output']['save_dir'], get_name(config))
    os.makedirs(config['output']['save_dir']+'/droid_c2w', exist_ok=True)
    os.makedirs(config['output']['save_dir']+'/rgbdnua', exist_ok=True)
    os.makedirs(config['output']['save_dir']+'/ply', exist_ok=True)
    if 'debug_mode' in list(config.keys()) and config['debug_mode']:
        os.makedirs(config['output']['save_dir']+'/debug_dict', exist_ok=True)
    shutil.copy(config_path, config['output']['save_dir']+'/config.yaml')
    
    # ======================================================================
    torch.backends.cudnn.benchmark = True
    
    mp.set_start_method('spawn', force=True)
        
    tracker2mapper_queue = mp.Queue()

    processes = []    
    for rank in range(2):
        if rank == 0:   p = mp.Process(target=tracking, args=(config, tracker2mapper_queue))
        elif rank == 1: p = mp.Process(target=mapping, args=(config, tracker2mapper_queue))
        processes.append(p)
        p.start()
    for p in processes:
        p.join()