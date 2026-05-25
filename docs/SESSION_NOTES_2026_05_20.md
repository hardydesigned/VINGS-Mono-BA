# Session-Notes 2026-05-20 — Aerial-Dataset-Sweep

Kompakte Chronologie der Befunde aus der Session. Detaillierte Erklärungen in
den thematischen Docs (MARS_LVIG.md, POSE_OVERRIDE.md, ...).

## Datensätze die wir durch hatten

| Datensatz | Status | PSNR Best | Befund |
|---|---|---|---|
| Matrix City (small_city aerial block_3) | ❌ | 23.55 | Photogrammetrie-Grid, kein echter SLAM |
| MARS HKairport_GNSS03 | ✅ | **23.68** (v10) | Aerial-Nadir, use_metric=false key |
| MARS AMtown03 | ✅ | **23.40** (200f) | Texturreich, drift bei > 500f |
| MUN-FRL Quarry (DJI M600) | ⚠ | 20.71 | Fisheye gelöst via M600-KB-Koeffs |
| MUN-FRL Bell412 (dataset5) | ⚠ | n/a (frühere Session) | Gleiches Fisheye-Setup |
| MegaNeRF Building | ✅ (Ref) | n/a | Goldstandard für Aerial-VINGS |

## Codenull-Patches die persistent geblieben sind

### `scripts/storage/storage_manage.py` — Bug-Fix
Off-by-one + Shape-Mismatch beim KF-Rollback gefixt. Lange Sequenzen crashen
nicht mehr bei Frame ~489. Siehe `STORAGE_MANAGER_FIX.md`.

### `scripts/datasets/generic_vo.py` — Pose-Override-Loader
Liest optional `ext_poses_file` (TUM-format, w2c) und gibt `pose` pro Frame im
`data_packet` zurück. Siehe `POSE_OVERRIDE.md`.

### `scripts/run.py` — Pose-Override-Hook
Nach jedem `tracker.track()` wird `data_packet['pose']` in
`tracker.video.poses_save[count_save-1]` geschrieben. Auch
`KEEP_RGBDNUA=1`-Env-var support für `cleanup`.

## Die Top-Erkenntnisse

1. **`use_metric: false` ist GAME-CHANGER bei Aerial-Nadir** (+2.75 dB MARS).
   Metric3D ist nicht für flache aerial Szenen ausgelegt. DroidNet-Tiefen aus
   dem internen BA reichen + sind sogar besser.

2. **DJI `local_position` hat 10% Scale-Verzerrung** vs RTK + IMU-Integration.
   Bedeutet alle pose-overrides die auf local_position basieren sind suboptimal.
   Würde mit RTK-basiert rekonstruierten Posen besser werden (TODO).

3. **VINGS' `use_dynamic` ist dead code** — nicht im run.py-Mainloop aufgerufen.
   Wer dynamic-removal will, braucht externe Masken via Loader-Patch.

4. **UAVScenes-Class-IDs sind Cityscapes-style**, NICHT die 0-18 vom Paper Tab S9.
   Sedan=20, Truck=24. Bei AMtown03: 774 von 1120 Frames haben Sedans.

5. **VIO ist strukturell tot auf MARS-LVIG** weil:
   - Keine published Cam-IMU-Extrinsik
   - LiDAR-Degeneration bei Nadir-Flug (UAVScenes-Paper bestätigt)
   - Alle vier VIO-Versuche eingebrochen (NaN-crash, PSNR=13.88, PSNR=15.19,
     rc=1 `video.cur_ii=None`)

6. **VRAM-Wand bei ~150 mapped Frames** mit native intrinsic 2448×2048. Storage-
   Manager hilft aber die kumulierten Gaussians sprengen die GPU.

7. **Init-Phase ist entscheidend**: AMtown03 mit `start_frame=0` → PSNR 15,
   mit `start_frame=200` → PSNR 23. DROID-Tracker braucht "Anlauf".

8. **Quarry DJI M600 hat eigene KB-Koeffs**, NICHT identisch zu Bell412. Wer das
   übersieht, bekommt 50-Pixel Principal-Point-Verschiebung (siehe
   `QUARRY_DJI_M600.md`).

9. **MARS-HKairport: PSNR matched UAVScenes-3DGS-State-of-the-Art** ohne dass
   wir externe GT-Posen brauchen. v13b (PSNR 20.89, SSIM 0.61) vs UAVScenes-3DGS
   (PSNR 20.92, SSIM 0.52). v22 mit Pose-Override sogar +0.5 dB.

## Tools die wir gebaut haben

| Tool | Zweck |
|---|---|
| `/tmp/amtown_extract.py` | MARS AMtown03 Bag → frames + DJI-Posen + IMU |
| `/tmp/amtown_extract_rest.py` | DJI-IMU + GPS + RTK + Velocity extract |
| `/tmp/amtown_imu_gps_sanity.py` | IMU/GPS-Konsistenz-Check (zeigte 10%-Bug auf) |
| `/tmp/amtown_mask_video_v2.py` | UAVScenes-Class-Overlay-Video (GT + colored mask) |
| `/tmp/amtown_500f_video.py` | GT \| Pred \| Pred+Seg 3-col video |
| `/tmp/mars_dji_to_tum.py` | DJI-Posen → TUM-Format c2w+w2c |
| `/tmp/mars_subset_poses_w2c.py` | Pose-Subset für Pose-Override-Configs |

Sie sind in `/tmp/` und sollten dauerhaft in `scripts/datasets/` o.ä. verschoben
werden wenn nochmal gebraucht.

## TODO (offen)

- RTK-basierte Posen rebuilden (statt local_position) — würde +2-3 dB bei
  Pose-Override geben
- VINS-Fusion Pre-Run für saubere Cam-IMU-Extrinsik (VIO-Voraussetzung)
- Three.js Dynamic-Reco-Pipeline aufbauen (siehe SEGMENTATION_AMTOWN.md Sec. 4)
- Instance-Annotations holen aus `interval1_CAM_label.zip` (6.9 GB) für Track-IDs

## Files die rum liegen + können weg

Siehe `DATASET_CHOICE.md` Sec. "Was rumliegt + brauchen wir noch". Hauptkandidat:
- `HKairport_GNSS03.bag` (9.1 GB)
- `mars_hkairport_gnss03/images_1900_3400_full/` (13 GB)
- `mars_vio_*` (~5 GB)
- alte `amtown03/images/` (918 MB, ersetzt durch `images_all/`)
- `amtown03/vio_dji/` (Wrapper-Symlinks für gecrashed VIO)
