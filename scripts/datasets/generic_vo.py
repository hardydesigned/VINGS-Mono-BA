import numpy as np
import os
import glob
import torch
import cv2
from typing import Optional
from tqdm import tqdm


def _latlonalt_to_enu(lat_deg: np.ndarray, lon_deg: np.ndarray,
                      alt_m: np.ndarray) -> np.ndarray:
    """Equirectangular lat/lon/alt -> ENU(meters) around the first sample.

    For spans of a few kilometres the equirectangular error is sub-metre,
    well below GPS-noise. Returned shape: (N, 3) with columns [east, north, up].
    """
    lat0, lon0, alt0 = float(lat_deg[0]), float(lon_deg[0]), float(alt_m[0])
    R_earth = 6378137.0
    lat_rad0 = np.deg2rad(lat0)
    east  = np.deg2rad(lon_deg - lon0) * R_earth * np.cos(lat_rad0)
    north = np.deg2rad(lat_deg - lat0) * R_earth
    up    = alt_m - alt0
    return np.stack([east, north, up], axis=1)


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

        # Optional: externe Pose-Source (TUM-Format: ts tx ty tz qx qy qz qw).
        # Wenn gesetzt, gibt __getitem__ pro Frame eine 'pose' im data_packet zurueck,
        # die in run.py auf tracker.video.poses_save[count_save-1] geschrieben wird.
        self.ext_poses = None
        ext_path = cfg['dataset'].get('ext_poses_file')
        if ext_path:
            arr = np.loadtxt(ext_path, comments='#')
            # Format: [ts, tx, ty, tz, qx, qy, qz, qw] -- speichern 7-DoF [tx,ty,tz,qx,qy,qz,qw]
            self.ext_poses = arr[:, 1:].astype(np.float32)
            print(f'GenericVODataset: loaded {len(self.ext_poses)} ext poses from {ext_path}')

        # Optional: GPS / RTK pro Frame. Wird vom TwoGateSelector (B1)
        # und vom GateA (A1) genutzt. CSV-Layout: t_sec lat lon alt [...].
        # ENU-Ursprung = erste Zeile; bei wenigen km Span sind die
        # equirectangular-Naeherungsfehler << GPS-Rauschen.
        self._gps_t: Optional[np.ndarray]    = None  # type: ignore[name-defined]
        self._gps_alt: Optional[np.ndarray]  = None  # type: ignore[name-defined]
        self._gps_xyz_enu: Optional[np.ndarray] = None  # type: ignore[name-defined]
        gps_path = cfg['dataset'].get('gps_csv')
        if gps_path:
            raw = np.loadtxt(gps_path, comments='#')
            t_col = int(cfg['dataset'].get('gps_t_col', 0))
            lat_col = int(cfg['dataset'].get('gps_lat_col', 1))
            lon_col = int(cfg['dataset'].get('gps_lon_col', 2))
            alt_col = int(cfg['dataset'].get('gps_alt_col', 3))
            self._gps_t = raw[:, t_col].astype(np.float64)
            self._gps_alt = raw[:, alt_col].astype(np.float32)
            self._gps_xyz_enu = _latlonalt_to_enu(
                raw[:, lat_col], raw[:, lon_col], raw[:, alt_col]
            ).astype(np.float32)
            print(f'GenericVODataset: loaded {len(self._gps_t)} GPS rows from {gps_path}')

        # Optional: per-frame Unix-epoch timestamps (overrides the default
        # frame-index timestamps). Required for GPS lookup to work, since
        # the GPS CSV uses real Unix time. Format: "t_sec filename" per line
        # (whitespace-separated). Only column 0 is read.
        self._cam_t_sec: Optional[np.ndarray] = None  # type: ignore[name-defined]
        camstamp_path = cfg['dataset'].get('camstamp_file')
        if camstamp_path:
            cs = np.loadtxt(camstamp_path, comments='#', usecols=(0,))
            self._cam_t_sec = np.asarray(cs, dtype=np.float64)
            print(f'GenericVODataset: loaded {len(self._cam_t_sec)} cam timestamps from {camstamp_path}')

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

        start_frame = int(cfg['dataset'].get('start_frame', 0))
        if start_frame > 0:
            rgb_files = rgb_files[start_frame:]
            if self._cam_t_sec is not None and start_frame < len(self._cam_t_sec):
                self._cam_t_sec = self._cam_t_sec[start_frame:]
        max_frames = cfg['dataset'].get('max_frames', None)
        if max_frames is not None:
            rgb_files = rgb_files[:int(max_frames)]
            if self._cam_t_sec is not None:
                self._cam_t_sec = self._cam_t_sec[:int(max_frames)]

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
        out = {
            'timestamp': self.rgbinfo_dict['timestamp'][idx],
            'rgb':       rgb,
            'intrinsic': intrinsic,
        }
        if self.ext_poses is not None and idx < len(self.ext_poses):
            out['pose'] = torch.tensor(self.ext_poses[idx], dtype=torch.float32)

        # Per-frame Unix-epoch timestamp + GPS lookup (used by GateA / TwoGate).
        # Fall back to the frame-index timestamp if camstamp_file is not set.
        if self._cam_t_sec is not None and idx < len(self._cam_t_sec):
            t_sec = float(self._cam_t_sec[idx])
        else:
            t_sec = float(self.rgbinfo_dict['timestamp'][idx])
        out['t_sec'] = t_sec

        if self._gps_t is not None and self._gps_xyz_enu is not None:
            j = int(np.argmin(np.abs(self._gps_t - t_sec)))
            out['alt_m']   = float(self._gps_alt[j])
            out['xyz_enu'] = self._gps_xyz_enu[j].copy()
        return out


def get_dataset(config):
    return GenericVODataset(config)
