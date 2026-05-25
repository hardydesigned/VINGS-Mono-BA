# Pose-Override-Pipeline

Mechanismus um VINGS-Mono mit **externen 6-DoF-Posen** zu füttern statt die
DROID-Tracker-eigenen zu nutzen. Implementiert Mai 2026.

## Wozu

Wenn der Datensatz schon stabile Posen mitbringt (DJI-RTK + IMU-fusion bei MARS,
oder ein vorab-laufendes SfM wie DJI Terra), kann man die anstelle des
DROID-Tracker-Outputs verwenden. Vorteile:
- **Halbierter GPU-Verbrauch** (bei MARS-HKairport v22: 4.6 GB statt 8.5 GB) —
  weil der interne BA stabiler arbeitet
- **Reduzierter lateraler Blur** durch weniger Pose-Drift in der Map
- Für Multi-Frame-3D-Reconstruction (z.B. Dynamic-Object-Tracking) sind GT-Posen
  fast immer besser

Caveat: der Mechanismus ist ein **soft-override**. Das interne DROID-Active-Window-BA
optimiert die Pose nach dem Override noch mit; bei reichlicher Textur (AMtown)
schmälert das den Effekt.

## Code-Patches

Drei Stellen wurden modifiziert:

### 1. `scripts/datasets/generic_vo.py` — Loader

In `__init__`:
```python
self.ext_poses = None
ext_path = cfg['dataset'].get('ext_poses_file')
if ext_path:
    arr = np.loadtxt(ext_path, comments='#')
    # arr columns: [ts, tx, ty, tz, qx, qy, qz, qw]
    # we keep last 7 = the 7-DoF VINGS poses_save format
    self.ext_poses = arr[:, 1:].astype(np.float32)
```

In `__getitem__`:
```python
out = {'timestamp': ..., 'rgb': ..., 'intrinsic': ...}
if self.ext_poses is not None and idx < len(self.ext_poses):
    out['pose'] = torch.tensor(self.ext_poses[idx], dtype=torch.float32)
return out
```

### 2. `scripts/run.py` — Hook nach `tracker.track()`

```python
if 'pose' in data_packet and data_packet['pose'] is not None and self.cfg['mode'] != 'vo_nerfslam':
    video = self.tracker.video if hasattr(self.tracker, 'video') else self.tracker.frontend.video
    if hasattr(video, 'count_save') and video.count_save > getattr(self, '_last_pose_override_idx', -1):
        ext_pose = data_packet['pose'].cpu() if isinstance(data_packet['pose'], torch.Tensor) else torch.tensor(...)
        video.poses_save[video.count_save - 1] = ext_pose
        self._last_pose_override_idx = video.count_save
```

Schreibt die externe Pose in den `poses_save`-Buffer **nach** jedem akzeptierten
Tracker-KF, sodass der Mapper sie sieht.

### 3. Config

```yaml
dataset:
  module: datasets.generic_vo
  root: ...
  ext_poses_file: /path/to/poses_w2c.txt   # TUM-format
```

## Format der externen Pose-Datei

TUM-style mit 7-DoF SE3:
```
# header (optional, # ignored)
1698219148.057266 -0.0042 -0.0011 0.6614 0.9954 0.0030 0.0859 0.0588
^timestamp        ^tx     ^ty     ^tz    ^qx    ^qy    ^qz    ^qw
```

**Konvention: world-to-camera (w2c).** Wichtig — VINGS `poses_save` ist intern
w2c (siehe `SE3(poses_save).inv() = c2w` im Code). Falls deine Posen c2w sind
(z.B. DJI-Drohnen-Body-Pose), zuerst invertieren:

```python
T_w2c = np.linalg.inv(T_c2w)
t = T_w2c[:3, 3]
q = R.from_matrix(T_w2c[:3, :3]).as_quat()  # xyzw
```

Die Reihenfolge der Zeilen muss der Reihenfolge der dataset-frames entsprechen
(nicht der bag-frames). D.h. wenn `start_frame=200, max_frames=500`, dann sind
zeilen 0..499 die für dataset-indices 0..499 (= bag-frames 200..699).

## Wie DJI-Posen aus MARS-LVIG extrahieren

Wir nutzen `/dji_osdk_ros/attitude` (Quaternion, 100 Hz) + `/dji_osdk_ros/local_position`
(xyz, 50 Hz) und interpolieren auf Cam-Frame-Zeitstempel:

```python
slerp = Slerp(att_ts, R.from_quat(att_quats))
quat_at_cam = slerp(cam_ts).as_quat()
pos_at_cam = np.interp_per_axis(cam_ts, pos_ts, pos_xyz)
```

Vollständiges Beispiel siehe `/tmp/mars_dji_to_tum.py` (HKairport) und
`/tmp/amtown_extract.py` (AMtown).

## ⚠ local_position-Scale-Bug

`/dji_osdk_ros/local_position` hat **10% Scale-Verzerrung** gegenüber RTK +
IMU-Velocity-Integration (gemessen auf 200s subset @ AMtown03). Für saubere
Pose-Sources daher RTK-basiert rekonstruieren:

```python
# 1. RTK lat/lon → ENU mit Local-Tangent-Plane an Frame-0-Position
# 2. RTK-Yaw + IMU-Roll/Pitch (via attitude) → 6-DoF
# 3. SLERP auf Cam-Timestamps
```

Nicht umgesetzt; aktuell sind alle pose-override-Runs mit dem 10%-off
`local_position`. Daher ist Pose-Override nur **leicht** besser als pure VO
(+0.5 dB statt erwartet +2-3 dB).

## Resultate (MARS HKairport_GNSS03, 500 Frames)

| | pure VO | Pose-Override c2w | Pose-Override w2c |
|---|---|---|---|
| PSNR | 20.89 | 21.26 | **21.40** |
| SSIM | 0.61 | 0.625 | **0.631** |
| LPIPS | 0.43 | 0.427 | 0.428 |
| Peak-GPU | 8464 MiB | 4597 MiB | 4983 MiB |

w2c-Konvention ist korrekt (+0.14 dB ggü. c2w). GPU-Verbrauch halbiert sich
deutlich, weil der DROID-BA weniger arbeiten muss.

## Bei AMtown03 weniger wirksam

AMtown ist texturreich → DROID-BA hat reichlich Features und überschreibt die
externen Posen via active_window-BA innerhalb weniger Iterations. Pose-Override
hilft hier nur marginal (+0.4 dB bei 1000f). Für stärkeren Effekt müsste man
`use_pose_refine: false` UND den BA-Update sperren — das ist ein
größerer Refactor in den `submodules/dbaf` CUDA-Kerneln.
