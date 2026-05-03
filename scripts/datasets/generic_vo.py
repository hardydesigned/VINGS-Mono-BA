import numpy as np
import os
import glob
import torch
import cv2
from tqdm import tqdm


class GenericVODataset:
    """
    Minimal VO-only dataset loader. Reads sorted images from a configurable directory.

    Config keys used (all under 'dataset'):
      root       - dataset root directory
      image_dir  - subdirectory for images relative to root (default 'color').
                   Set to '' or null to read directly from root.
      image_ext  - comma-separated glob extensions, e.g. '*.png' or '*.jpg,*.JPG'
                   (default: tries *.png then *.jpg then *.JPG)

    Intrinsics convention (same as all other VINGS loaders):
      fu = fy  (focal length along the HEIGHT/row direction)
      fv = fx  (focal length along the WIDTH/col  direction)
      cu = cy  (principal point along HEIGHT)
      cv = cx  (principal point along WIDTH)
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.dataset_dir = cfg['dataset']['root']

        image_subdir = cfg['dataset'].get('image_dir', 'color')
        if image_subdir:
            image_base = os.path.join(self.dataset_dir, image_subdir)
        else:
            image_base = self.dataset_dir

        raw_ext = cfg['dataset'].get('image_ext', '')
        if raw_ext:
            exts = [e.strip() for e in raw_ext.split(',')]
        else:
            exts = ['*.png', '*.jpg', '*.JPG', '*.jpeg', '*.JPEG']

        rgb_files = []
        for ext in exts:
            rgb_files += glob.glob(os.path.join(image_base, ext))
        rgb_files = sorted(set(rgb_files))

        if not rgb_files:
            raise FileNotFoundError(
                f"GenericVODataset: no images found in '{image_base}' "
                f"with extensions {exts}"
            )

        max_frames = cfg['dataset'].get('max_frames', None)
        if max_frames is not None:
            rgb_files = rgb_files[:int(max_frames)]

        self.rgbinfo_dict = {
            'timestamp': list(range(len(rgb_files))),
            'filepath':  rgb_files,
        }
        self.c2i       = np.eye(4)
        self.intrinsic = None
        self.tqdm      = tqdm(total=len(rgb_files))

    def __len__(self):
        return len(self.rgbinfo_dict['timestamp'])

    def preload_camtimestamp(self):
        return np.array(self.rgbinfo_dict['timestamp']).reshape(-1, 1)

    def preload_imu(self):
        all_imu = np.zeros((len(self.rgbinfo_dict['timestamp']), 7))
        all_imu[:, 0] = np.array(self.rgbinfo_dict['timestamp'])
        return all_imu

    def __getitem__(self, idx):
        resized_h = int(self.cfg['frontend']['image_size'][0])
        resized_w = int(self.cfg['frontend']['image_size'][1])

        rgb_raw = cv2.imread(self.rgbinfo_dict['filepath'][idx])
        if rgb_raw is None:
            raise IOError(f"Cannot read image: {self.rgbinfo_dict['filepath'][idx]}")
        rgb = (
            torch.tensor(cv2.resize(rgb_raw, (resized_w, resized_h)))[..., [2, 1, 0]]
        ).permute(2, 0, 1).unsqueeze(0).to(self.cfg['device']['tracker'])

        u_scale = resized_h / self.cfg['intrinsic']['H']
        v_scale = resized_w / self.cfg['intrinsic']['W']
        intrinsic = torch.tensor(
            [
                self.cfg['intrinsic']['fv'] * v_scale,
                self.cfg['intrinsic']['fu'] * u_scale,
                self.cfg['intrinsic']['cv'] * v_scale,
                self.cfg['intrinsic']['cu'] * u_scale,
            ],
            dtype=torch.float32,
            device=self.cfg['device']['tracker'],
        )

        self.tqdm.update(1)
        return {
            'timestamp': self.rgbinfo_dict['timestamp'][idx],
            'rgb':       rgb,
            'intrinsic': intrinsic,
        }


def get_dataset(config):
    return GenericVODataset(config)
