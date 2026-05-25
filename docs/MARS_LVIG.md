# MARS-LVIG (HKairport_GNSS03, AMtown03)

Erkenntnisse aus den MARS-LVIG-Runs (HKairport + AMtown). Quelle: MARS-LVIG-Paper
(Li et al., IJRR 2024) + UAVScenes-Paper (Wang et al. 2025) + eigene Messungen.

## Hardware-Setup

DJI M300 RTK mit:
- **Hikvision CA-050-11UC** RGB-Cam, 2448×2048 BGR @ 10 Hz, global shutter
- **Livox Avia** LiDAR + interne BMI088-IMU @ 200 Hz (Output in **g-Einheiten**,
  muss mit 9.803 multipliziert werden für m/s²)
- **DJI Flight-Controller IMU** @ 400 Hz (Output direkt in m/s², gravity-included)
- u-blox ZED-F9P GNSS + RTK + simpleRTK2B

DJI L1 LiDAR ist auch montiert aber **encrypted** → per-frame nicht zugänglich
(UAVScenes-Paper bestätigt das).

## Bag-Topics

| Topic | Rate | Konvention |
|---|---|---|
| `/left_camera/image/compressed` | 10 Hz | JPEG, BGR |
| `/livox/imu` | 200 Hz | accel in **g** → mit 9.803 multiplizieren |
| `/dji_osdk_ros/imu` | 400 Hz | accel direkt m/s² |
| `/dji_osdk_ros/attitude` | 100 Hz | Quaternion (body→world) |
| `/dji_osdk_ros/local_position` | 50 Hz | UTM-like xyz, **10% Scale-Verzerrung** (siehe unten) |
| `/dji_osdk_ros/rtk_position` | 5 Hz | RTK GPS lat/lon/alt, cm-präzise |
| `/dji_osdk_ros/gps_position` | 50 Hz | Standard GPS ~1.5m Genauigkeit |
| `/dji_osdk_ros/velocity` | 50 Hz | Geschwindigkeit (NED) |
| `/dji_osdk_ros/rtk_yaw` | 5 Hz | RTK-heading int16 |

## Kalibrierung

**Camera Intrinsics** sind **pro Region** unterschiedlich (chessboard recalibration
pro Szene). Werte stehen auf `mun-frl-vil-dataset.readthedocs.io/sensor_calibration.html`
+ im UAVScenes-Drive im Calibration-Folder.

### HKairport_GNSS03 (HK_GNSS.yaml)
- K (raw 2448×2048): fx=1444.43 fy=1444.34 cx=1179.50 cy=1044.90
- D (Plumb-Bob k1,k2,p1,p2,k3): -0.0560, 0.1180, 0.00122, 0.00064, -0.0627
- T_cam_lidar siehe yaml (Camera-Lidar-Extrinsik)

### AMtown03 (AMtown.yaml)
- K (raw 2448×2048): fx=1453.72 fy=1453.28 cx=1172.18 cy=1041.78
- D: -0.1210, 0.1113, 0.0016, 0.00013, -0.06235
- Stärkere Distortion als HKairport

**Distortion-Modell**: OpenCV Plumb-Bob (`cv2.getOptimalNewCameraMatrix`), NICHT
Kannala-Brandt wie bei dataset5/Quarry. Für AMtown:
```python
new_K, _ = cv2.getOptimalNewCameraMatrix(K, D, (2448, 2048), alpha=0.0)
map1, map2 = cv2.initUndistortRectifyMap(K, D, np.eye(3), new_K, (2448, 2048), cv2.CV_16SC2)
```

**KEINE published Cam-IMU-Extrinsik** für DJI body → camera. Das ist der
Hauptblocker für VIO.

## Der local_position-10%-Scale-Bug

**Wichtige Erkenntnis** aus IMU/GPS-Sanity-Check (`/tmp/amtown_imu_gps_sanity.py`):

Über die Frames 2900-4899 (200s subset):
- DJI velocity-Integration: **504 m horizontal**
- RTK-ENU diff: **502 m horizontal** ← Ground Truth
- DJI local_position diff: **574 m horizontal** ← 10% länger!

Implikation: `/dji_osdk_ros/local_position` ist um ~10% verzerrt gegenüber RTK +
IMU-velocity-Integration. RTK + IMU passen perfekt. Wenn man DJI-Posen als Pose-
Override für VINGS nutzt, immer **RTK-basiert** rekonstruieren, nicht aus
`local_position`.

Implementiert in den `dji_poses_*` Files in `~/Dokumente/datasets/{amtown03,mars_hkairport_gnss03}/metadata/` —
diese basieren aktuell auf local_position und sind deshalb 10% off. RTK-rebuild
ist als TODO offen.

## VIO funktioniert NICHT bei MARS-LVIG

Mehrere Versuche, alle eingebrochen:
- v14 (HKairport, Livox-IMU, no Metric3D): NaN-crash bei Frame 250
- v14b (HKairport, Livox-IMU + Metric3D): rc=0 aber PSNR=13.88, Map kaputt
- v20 (HKairport, DJI-IMU): PSNR=15.19, Map kaputt
- v_amtown_vio (AMtown, DJI-IMU): rc=1, `video.cur_ii = None` in `__rollup`

**Strukturelle Gründe** (auch im UAVScenes-Paper Sec. 3.1 bestätigt):
> "ground-facing flight causes LiDAR degeneration, leading to unsatisfactory
> reconstruction results"
- Aerial-Nadir-Flug → LiDAR sieht nur flachen Boden → Tiefen-Degenerate
- MARS publiziert **keine Cam-IMU-Extrinsik** → wir mussten T_body_cam raten
  (`T_lidar_cam` oder Identity + 20cm offset). Falscher Offset → IMU-Preintegration
  driftet Posen weg → Mapper bekommt inkonsistente Constraints
- IMU-Noise-Parameter sind Xsens-Werte (aus dataset5 geklaut), nicht BMI088-spezifisch

**Workarounds die nicht halten**:
- Verschiedene `T_body_cam`-Annahmen probiert: alle → Drift
- IMU-Skala fixieren (Livox×9.803): notwendig aber nicht hinreichend
- Verschiedene IMU-Noise-Werte: keine signifikante Wirkung

**Saubere Lösung** (nicht umgesetzt): VINS-Fusion oder OpenVINS auf einem MARS-Bag
mit `extrinsic 2` Online-Estimation laufen lassen → die gelernte T_BC + IMU-Bias
als Init für VINGS nutzen. Etwa 2-3 Stunden Setup-Aufwand.

## Run-Resultate HKairport_GNSS03 (siehe DATASET_CHOICE.md für Vergleich mit UAVScenes-Paper)

| Run | Frames | Mode | PSNR | SSIM | LPIPS | Status |
|---|---|---|---|---|---|---|
| v0 | 300 (2400-2700) | VO half-res | 19.29 | 0.49 | 0.47 | OK |
| v5 | 200 (2400-2600) | VO paperloss | 20.93 | 0.58 | 0.39 | OK |
| **v10** | **200** | **VO paperloss + use_metric=false + native PNG** | **23.68** | **0.74** | **0.26** | **best kurz** |
| v13b | 500 | + storage_mgr (gefixt) | 20.89 | 0.61 | 0.43 | OK |
| v22 | 500 | + Pose-Override (DJI w2c) | 21.40 | 0.63 | 0.43 | leicht besser als pure VO |
| v17 | 1500 | conservative knobs | 19.35 | 0.55 | 0.48 | rc=137 @ 1292 |
| v19 | 800 | balanced | 20.06 | 0.58 | 0.45 | OK (longest stable) |

**UAVScenes 3DGS auf HKairport** (mit DJI-Terra-GT-Posen) erreicht PSNR=20.92 →
unser v13b (20.89) matched das **ohne** externe GT-Posen, v22 (21.40) übertrifft.

## Run-Resultate AMtown03

Texturreicher als HKairport (Vegetation+Buildings+Roads) → Tracker driftet
schneller. Drohne fliegt 10 m/s @ 80 m Höhe.

| Run | Frames | start | mapper_kf_skip | PSNR | SSIM | LPIPS |
|---|---|---|---|---|---|---|
| 200f | 200 | subset[200] | 1 | **23.40** | **0.78** | **0.18** |
| 500f-init0 | 500 | subset[0] | 1 | 15.12 | 0.38 | 0.63 |
| 500f-skip5 | 500 | subset[0] | 5 | 12.47 | 0.25 | 0.79 |
| **500f-v2** | **500** | **subset[200]** | **1** | **22.83** | **0.75** | **0.23** |
| 1000f | 1000 | bag[2900] | 2 | 17.52 | 0.47 | 0.47 |
| 1000f-skip1 | 1000 | bag[3100] | 1 | 18.25 | 0.50 | 0.45 |
| 1000f-pose | 1000 | + DJI-RTK Override | 1 | 17.86 | 0.50 | 0.47 |
| 1800f | 1800 | bag[3100] | 3 | 14.04 | 0.30 | 0.59 (rc=137) |

**Wichtige Befunde:**
- **Init-Phase ist entscheidend**: Sequenz-Start (subset[0]) → schlechte
  Initialisierung → PSNR=15. Mit subset[200] (= 20s in den Cruise rein) → PSNR=23.
- **Drift kicks in nach ~500 Frames** Cruise — bei 800 m Cruise-Strecke akkumulieren
  sich Pose-Errors.
- **Pose-Override hilft nur minimal** (+0.5 dB) weil DJI local_position 10% off ist
  → würde mit RTK-basierten Posen besser werden.

## VRAM-Wand bei VINGS

Bei **native intrinsic 2448×2048** ist die harte Grenze ~150 mapped Frames bevor
die GPU bei ~9.5 GiB den Watchdog (≤8 GB) triggert. Mit:
- mapper_kf_skip=3 + num_keyframe=4 + storage_mgr aktiv: ~1300 Sequence-Frames
- mapper_kf_skip=1: ~500 Sequence-Frames

Half-Res (1224×1024) ist 4x weniger Render-Pixels → 2-3x mehr Frames möglich,
aber PSNR-Drop ~3-4 dB durch Resampling + JPEG-Quantisierung.

## Wo welche Daten liegen

```
~/Dokumente/datasets/amtown03/
├── images_all/                  # alle 6199 Bag-Frames, half-res JPEG
├── metadata/
│   ├── camstamp_all.txt          # bag-frame timestamps
│   ├── dji_poses_all_c2w.txt     # 6-DoF c2w, TUM format (10% scale-off!)
│   ├── dji_poses_all_w2c.txt     # inverse, für VINGS poses_save
│   ├── imu_dji.txt               # 400 Hz DJI IMU
│   ├── imu_livox.txt             # 200 Hz Livox IMU (×9.803 skaliert)
│   ├── gps.csv                   # 50 Hz Standard GPS
│   ├── rtk.csv                   # 5 Hz RTK cm-genau
│   ├── rtk_yaw.csv               # 5 Hz RTK heading
│   ├── velocity.csv              # 50 Hz NED velocity
│   └── intrinsic_half.txt        # Post-Rectify K (1224×1024)
└── amtown03_mask_overlay_v2.mp4  # GT + UAVScenes-Class-Mask Vis-Video
```

`mars_hkairport_gnss03/` hat ähnliche Struktur (HKairport_GNSS03 sequence).

## UAVScenes (Wang et al. 2025) Notes

Bauen auf MARS-LVIG auf:
- Sie haben **DJI-Terra-SfM** für 6-DoF-Poses gemacht weil MARS nur 4-DoF RTK+yaw
  publiziert
- Frame-wise Semantic-Annotations für 20 Sequenzen (16 static + 2 dynamic classes
  = Sedan + Truck, plus 1 Background)
- **Class-IDs in den id-PNG-Files sind CITYSCAPES-style** (0-25+), nicht Paper Tab S9 (0-18):
  - 20 = Sedan
  - 24 = Truck
  - 13 = Vegetation/GreenField (häufigste Klasse)
  - 3 = PavedRoad
- Wir nutzen das `interval5_CAM_label.zip` (1.4 GB, jeder 5te Frame, 1120 für AMtown03)

Download via HF: `https://huggingface.co/datasets/sijieaaa/UAVScenes/resolve/main/interval5_CAM_label.zip`
