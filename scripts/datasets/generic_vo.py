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


def _lla_to_ecef(lat_deg: np.ndarray, lon_deg: np.ndarray,
                 alt_m: np.ndarray) -> np.ndarray:
    """WGS84 geodetic lat/lon/alt -> ECEF (meters). Shape (N,3) [X,Y,Z].

    Matches the ECEF convention of frontend/geoFunc/trans.py (cart2geod/Cen),
    so positions feed the DBA-Fusion gtsam.GPSFactor directly (ten0 + Cen(ten0)).
    """
    a = 6378137.0
    f = 1.0 / 298.257223563
    e2 = f * (2.0 - f)
    lat = np.deg2rad(np.asarray(lat_deg, dtype=np.float64))
    lon = np.deg2rad(np.asarray(lon_deg, dtype=np.float64))
    h = np.asarray(alt_m, dtype=np.float64)
    sin_lat = np.sin(lat)
    N = a / np.sqrt(1.0 - e2 * sin_lat * sin_lat)
    X = (N + h) * np.cos(lat) * np.cos(lon)
    Y = (N + h) * np.cos(lat) * np.sin(lon)
    Z = (N * (1.0 - e2) + h) * sin_lat
    return np.stack([X, Y, Z], axis=1)


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
            # ext_poses (z.B. dji_poses) MUSS gleich gesliced werden, sonst sind die
            # Frame-Posen um start_frame verschoben (ext_poses[idx] wird slice-relativ
            # indexiert). Spiegelt das _cam_t_sec-Slicing oben.
            if self.ext_poses is not None and start_frame < len(self.ext_poses):
                self.ext_poses = self.ext_poses[start_frame:]
        max_frames = cfg['dataset'].get('max_frames', None)
        if max_frames is not None:
            rgb_files = rgb_files[:int(max_frames)]
            if self._cam_t_sec is not None:
                self._cam_t_sec = self._cam_t_sec[:int(max_frames)]
            if self.ext_poses is not None:
                self.ext_poses = self.ext_poses[:int(max_frames)]

        self.rgbinfo_dict = {
            'timestamp': list(range(len(rgb_files))),
            'filepath':  rgb_files,
        }

        # Optional: per-frame LiDAR-Tiefe (UAVScenes interval). Projiziert die
        # LiDAR-Punkte (lx=Tiefe, ly->u, lz->v) auf die resized Bildebene und
        # liefert eine DICHTE metrische Tiefe (NN-gefuellt) als data_packet['depth'].
        self.lidar_files = None
        lidar_dir = cfg['dataset'].get('lidar_dir')
        if lidar_dir:
            if not os.path.isabs(lidar_dir):
                lidar_dir = os.path.join(self.dataset_dir, lidar_dir)
            lmap = {}
            for fn in os.listdir(lidar_dir):
                if fn.startswith('image') and '_lidar' in fn:
                    lmap[fn[len('image'):fn.index('_lidar')]] = os.path.join(lidar_dir, fn)
            self.lidar_files = [
                lmap.get(os.path.splitext(os.path.basename(f))[0]) for f in rgb_files
            ]
            n_ok = sum(x is not None for x in self.lidar_files)
            self.lidar_sign = (float(cfg['dataset'].get('lidar_sign_u', 1.0)),
                               float(cfg['dataset'].get('lidar_sign_v', 1.0)))
            print(f'GenericVODataset: LiDAR-Tiefe aktiv, {n_ok}/{len(rgb_files)} Frames gematcht, signs={self.lidar_sign}')
        # mode: vio braucht ECHTE Unix-Zeit als Frame-tstamp (muss zur IMU passen);
        # im VO-Default bleiben es Frame-Indizes (IMU wird ignoriert).
        if cfg.get('mode') == 'vio':
            if self._cam_t_sec is None:
                raise ValueError("mode: vio braucht dataset.camstamp_file (Unix-Zeit pro Frame)")
            n_frames = len(rgb_files)
            if len(self._cam_t_sec) < n_frames:
                raise ValueError(
                    f"camstamp ({len(self._cam_t_sec)}) < frames ({n_frames}) -- kann VIO-tstamps nicht setzen")
            self.rgbinfo_dict['timestamp'] = [float(t) for t in self._cam_t_sec[:n_frames]]

        # Cam->IMU-Extrinsik (4x4). Aus dataset.c2i (verschachtelte Liste) wenn gesetzt --
        # noetig fuer mode: vio (Tbc = Pose3(c2i)). Default Identitaet (VO ignoriert Tbc).
        _c2i_cfg = cfg['dataset'].get('c2i')
        if _c2i_cfg is not None:
            self.c2i = np.asarray(_c2i_cfg, dtype=np.float64).reshape(4, 4)
        else:
            self.c2i = np.eye(4)
        self.intrinsic = None
        self.tqdm      = tqdm(total=len(rgb_files))

    def __len__(self):
        return len(self.rgbinfo_dict['timestamp'])

    def _lidar_depth(self, idx, resized_h, resized_w):
        """Projiziert die LiDAR-Punkte des Frames auf (resized_h, resized_w) und
        liefert eine DICHTE metrische Tiefe (NN-Fill). None wenn kein LiDAR."""
        path = self.lidar_files[idx]
        if path is None:
            return None
        try:
            pts = np.loadtxt(path)
        except Exception:
            return None
        if pts.ndim != 2 or pts.shape[0] == 0:
            return None
        lx, ly, lz = pts[:, 0], pts[:, 1], pts[:, 2]
        v_ok = lx > 0.1
        lx, ly, lz = lx[v_ok], ly[v_ok], lz[v_ok]
        su, sv = self.lidar_sign
        Wn = float(self.cfg['intrinsic']['W']); Hn = float(self.cfg['intrinsic']['H'])
        fu = self.cfg['intrinsic']['fu'] * (resized_w / Wn); cu = self.cfg['intrinsic']['cu'] * (resized_w / Wn)
        fv = self.cfg['intrinsic']['fv'] * (resized_h / Hn); cv = self.cfg['intrinsic']['cv'] * (resized_h / Hn)
        u = (fu * (su * ly) / lx + cu).astype(np.int64)
        v = (fv * (sv * lz) / lx + cv).astype(np.int64)
        inb = (u >= 0) & (u < resized_w) & (v >= 0) & (v < resized_h)
        u, v, d = u[inb], v[inb], lx[inb].astype(np.float32)
        if d.shape[0] == 0:
            return None
        depth = np.zeros((resized_h, resized_w), dtype=np.float32)
        order = np.argsort(-d)              # fern zuerst -> nah ueberschreibt (z-buffer)
        depth[v[order], u[order]] = d[order]
        mask = depth > 0
        if not mask.all():
            from scipy.ndimage import distance_transform_edt
            nn = distance_transform_edt(~mask, return_distances=False, return_indices=True)
            depth = depth[tuple(nn)]        # naechster gueltiger Tiefenwert -> dichte Karte
        return torch.tensor(depth, dtype=torch.float32, device=self.cfg['device']['tracker'])

    def preload_camtimestamp(self):
        return np.array(self.rgbinfo_dict['timestamp']).reshape(-1, 1)

    def preload_imu(self):
        # mode: vio -- echte IMU laden. Format imu_dji.txt: [t gx gy gz ax ay az] mit
        # gyro in deg/s, accel in m/s^2 -- exakt was dbaf_frontend erwartet (gyro wird
        # dort /180*pi konvertiert, accel direkt). t in derselben Unix-Epoch wie camstamp.
        # Einmal geladen, danach gecached (preload_imu wird in run.py einmal aufgerufen).
        imu_file = self.cfg['dataset'].get('imu_file')
        if imu_file:
            if getattr(self, '_imu_cache', None) is None:
                arr = np.loadtxt(imu_file, comments='#').astype(np.float64)
                arr[:, 0] -= float(self.cfg['dataset'].get('imu_delay', 0.0))
                self._imu_cache = arr
                print(f'GenericVODataset: loaded {len(arr)} IMU rows from {imu_file}')
            return self._imu_cache
        # VO-Fallback: Null-IMU mit Frame-Index-Zeit (wird im VO-Modus ohnehin ignoriert)
        all_imu = np.zeros((len(self.rgbinfo_dict['timestamp']), 7))
        all_imu[:, 0] = np.array(self.rgbinfo_dict['timestamp'])
        return all_imu

    def preload_gnss(self):
        """GNSS fuer den DBA-Fusion-GPSFactor (Stage C). Liefert (N,4) [t, X,Y,Z]_ECEF
        oder [] wenn kein gnss_file gesetzt (dann bleibt GNSS aus -- Verhalten wie bisher).

        rtk.csv ist 5 Hz, Kamera 10 Hz; der Frontend-Sync-Gate (dbaf_frontend) akzeptiert
        ein GNSS-Sample nur < 0.01 s vom Frame. Deshalb hier ECEF auf die Kamera-Zeitstempel
        INTERPOLIEREN -> an jedem Frame liegt ein GNSS-Sample. ten0/Cen-Konvention via
        _lla_to_ecef passend zu frontend/geoFunc/trans.py.
        """
        gnss_file = self.cfg['dataset'].get('gnss_file')
        if not gnss_file or self._cam_t_sec is None:
            return []
        raw = np.loadtxt(gnss_file, comments='#')
        t_col   = int(self.cfg['dataset'].get('gps_t_col', 0))
        lat_col = int(self.cfg['dataset'].get('gps_lat_col', 1))
        lon_col = int(self.cfg['dataset'].get('gps_lon_col', 2))
        alt_col = int(self.cfg['dataset'].get('gps_alt_col', 3))
        gt = raw[:, t_col].astype(np.float64)
        ecef = _lla_to_ecef(raw[:, lat_col], raw[:, lon_col], raw[:, alt_col])  # (M,3)
        cam_t = np.asarray(self._cam_t_sec, dtype=np.float64)
        # nur Frames innerhalb der GNSS-Zeitabdeckung (sonst np.interp extrapoliert flach)
        X = np.interp(cam_t, gt, ecef[:, 0])
        Y = np.interp(cam_t, gt, ecef[:, 1])
        Z = np.interp(cam_t, gt, ecef[:, 2])
        out = np.column_stack([cam_t, X, Y, Z]).astype(np.float64)
        print(f'GenericVODataset: built {len(out)} GNSS rows (ECEF, interp@cam) from {gnss_file}')
        return out

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

        # LiDAR-Tiefe (falls aktiv) -> run.py:535 ueberspringt dann Metric3D.
        if self.lidar_files is not None:
            d = self._lidar_depth(idx, resized_h, resized_w)
            if d is not None:
                out['depth'] = d
        return out


def get_dataset(config):
    return GenericVODataset(config)
