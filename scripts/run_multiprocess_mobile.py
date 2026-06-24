import numpy as np
import shutil
import torch
import os
from frontend.dbaf import DBAFusion
from gaussian.gaussian_model import GaussianModel
from gaussian.vis_utils import save_ply
import argparse
import asyncio
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
from server.server import WebsocketServer
from metric.metric_model import Metric_Model
from metric.depth_factory import make_depth_model

from gaussian.vis_utils import vis_map

import cv2

    #todo
def tracking(cfg, tracker2mapper_queue, server2tracker_queue = None):
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
            # data_packet = dataset[idx] # __getitem__

            if 'use_mobile' in cfg.keys() and cfg['use_mobile']:
                # todo
                print("use_mobile")
                while True:
                    if server2tracker_queue.qsize()>0:
                        data_packet = dataset.load_rgb(server2tracker_queue.get(),idx)
                        tracker.frontend.all_imu   = dataset.preload_imu()
                        tracker.frontend.all_stamp = dataset.preload_camtimestamp()
                        break
                    else:
                        print("tracker waiting for server")
                        time.sleep(0.1)   
            else:
                data_packet = dataset[idx]
            
            if 'use_metric' in cfg.keys() and cfg['use_metric']:
               data_packet['depth'] = metric_predictor.predict(data_packet['rgb'][0])
            
            tracker.track(data_packet)
            torch.cuda.empty_cache()
            # Judge whether new keyframe is added and package keyframe dict.
            viz_out = judge_and_package(tracker, data_packet['intrinsic'])
            if viz_out is not None and (tracker.cfg['mode']=='vo' or tracker.video.imu_enabled):
                tracker2mapper_queue.put(viz_out)
                
            idx += 1
            torch.cuda.empty_cache()
        else:
            print("mapper queue full")
            time.sleep(0.1)


def mapping(cfg, tracker2mapper_queue,mapper2server_queue):
    mapper = GaussianModel(cfg)
    cnt=0
    while True:
        # Run mapping.
        
        if tracker2mapper_queue.qsize() > 0:
            viz_out = tracker2mapper_queue.get()
            print("mapper run")
            mapper.run(viz_out)
            print("mapper runned")
            torch.cuda.empty_cache()
            
            rendered_map     = vis_map(None, mapper,True) # numpy, (3,H,W) or (H,W,3)
            
            raw_H, raw_W     = rendered_map.shape[0], rendered_map.shape[1] 
            new_H, new_W     = raw_H//2, raw_W//2
            
            rendered_map     = cv2.resize(rendered_map, (new_W, new_H))
            # print('run_mul/lin102: ', mapper.vis_rgbdnua.shape, mapper.vis_rgbdnua.max())
            
            raw_rgbdnua = mapper.vis_rgbdnua
            new_rgbdnua = np.concatenate((raw_rgbdnua[:, :616], np.concatenate((raw_rgbdnua[:344, 616:616*2], raw_rgbdnua[-344:, 616*2:616*3]), axis=0)), axis=1)
            
            rendered_rgbdnua = cv2.resize(new_rgbdnua, (new_W, new_H))
            
            rendered_image   = np.concatenate([rendered_map, rendered_rgbdnua], axis=1)
            
            # print("shape!!!!:",rendered_image.shape)
            mapper2server_queue.put(rendered_image)
            cnt+=1
            if 'use_mobile' in cfg.keys() and cfg['use_mobile']:
                # if (viz_out['global_kf_id'][-1]+1) % 20 == 0:
                if((cnt+1)%1000==0):
                    save_ply(mapper, viz_out['global_kf_id'][-1], save_mode='3dgs')
        
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
    server2tracker_queue = mp.Queue()
    mapper2server_queue  = mp.Queue()
    
    websocketServer  = WebsocketServer(server2tracker_queue = server2tracker_queue,mapper2server_queue=mapper2server_queue)
    
    processes = []    
    for rank in range(2):
        if rank == 0:   p = mp.Process(target=tracking, args=(config, tracker2mapper_queue,server2tracker_queue))
        elif rank == 1: p = mp.Process(target=mapping, args=(config, tracker2mapper_queue,mapper2server_queue))
        processes.append(p)
        p.start()
        
    asyncio.run(websocketServer.run())
    for p in processes:
        p.join()