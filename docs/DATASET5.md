# dataset5 (Bell412) — Recherche, Setup, Abbruch

Bell412-Helikopter-Aerial-Dataset, extrahiert aus ROS-bag. Versuch am 2026-05-19, das
Dataset für VINGS-Mono-Sweeps zu nutzen — **erfolgreich gedroppt**, weil der Algorithmus
inkompatibel mit dem Sensor-Setup ist. Diese Notiz dokumentiert was wir erarbeitet haben
und warum es nicht läuft, damit niemand das Rad ein zweites Mal aufmacht.

## Quelle

```
/media/philipp/USB_STICK/downloads/dataset5/extracted/
├── camera_image_color/          4328 .jpg, 1440×1080 — main cam (downward fisheye)
├── front_camera_image_color/    5395 .jpg, 720×540   — front cam (rotor + mast im Bild)
├── imu_data/data.csv            193097 IMU-Zeilen (ROS sensor_msgs/Imu, ~400 Hz)
├── imu_data_stamped/, imu_mag/, imu_time_ref*/        IMU-Hilfsdaten
├── fix/data.csv                 4538 GPS-Fixes (sensor_msgs/NavSatFix, GPS-Lat/Lon/Alt)
├── nmea_sentence/data.csv       NMEA-Strings
├── scan/, time_ref_scan/        Laserscan-Zeitreferenzen
└── velodyne_points/             Velodyne-PCDs (Punktwolken)
```

## Kalibration (vom User geliefert)

KANNALA_BRANDT-Fisheye-Modell für die 1440×1080-Cam:

```
model_type: KANNALA_BRANDT
mu:  829.224   mv:  829.454       # fx, fy
u0:  833.937   v0:  562.509       # cx, cy
k2: -0.0764245
k3:  0.0322856
k4: -0.0445168
k5:  0.0163317
```

IMU-Noise (für späteren VIO-Versuch):
```
acc_n: 0.08    gyr_n: 0.004
acc_w: 0.00004 gyr_w: 0.0001
g_norm: 9.803
```

Für die front-Cam (720×540) gibt es **keine separate Kalibration**. Annahme im Versuch:
gleicher Sensor, halbiert → K skaliert mit 0.5 (mu=414.612, v0=281.2545, …). Belastbarkeit
ungewiss, weil das Bild offensichtlich einen anderen Look hat (FOV-Eindruck, Rotor sichtbar).

## Was wir erarbeitet haben (bleibt im Repo)

1. **`scripts/prepare_dataset5.py`** — kopiert beide Cams in sequenziell durchnumerierte
   Output-Ordner und sampelt every-N. Output unter `dataset5/extracted/`:
   - `front_images_renamed/` (5395 files, `000000.jpg`–`005394.jpg`)
   - `images_renamed/`       (4328 files, `000000.jpg`–`004327.jpg`)
   - `every500/`             20 Stichproben (`front_NNNNNN.jpg` + `image_NNNNNN.jpg`)
   - `every100/`             98 Stichproben (separat per Inline-Python erzeugt)
2. **`configs/local/dataset5/`** — drei VINGS-Configs (alle mit `mapper_kf_skip=2`),
   liefen aber alle ins VRAM-Limit:
   - `dataset5_front_500_1100_mapskip2.yaml`     (front, 600 Frames)
   - `dataset5_front_3200_3800_mapskip2.yaml`    (front, 600 Frames)
   - `dataset5_image_3400_4200_mapskip2.yaml`    (main,  800 Frames)
3. **`scripts/run_dataset5.sh`** — sequenzieller Chain-Runner für die drei Configs.

## Warum es nicht läuft

Alle Runs sterben mit `rc=137` (SIGKILL) nach 25–71 Frames. Peak-RSS war jeweils nur
~1.88 GB, also weit unter System-RAM-OOM. **Peak-GPU 9.4–9.8 GB / 10 GB (RTX 3080)** —
der User hat einen eigenen VRAM-Watchdog der den Prozess SIGKILLed wenn VRAM überläuft.

Ursachen für die VRAM-Explosion:

1. **Fisheye nicht entzerrt.** KANNALA_BRANDT mit k2..k5 ≠ 0 (~180° FOV bei der main-Cam,
   sichtbare schwarze Vignettierung). VINGS' Loader nutzt nur das Pinhole-Modell
   (`fu/fv/cu/cv`). Tracker schätzt komplett falsche Tiefen → Mapper platziert Gaussians
   an Random-3D-Stellen → hoher Photo-Loss → Splits → unbegrenztes Gaussian-Wachstum.
2. **front-Cam: Helikopter-Rotor + Mast permanent im oberen Bilddrittel.** Dynamische
   Objekte, falsche Parallax-Hinweise für den Tracker.
3. **main-Cam: nach unten gerichtetes Fisheye über fast featureless Boden** (Wiese aus
   Höhe). DROID-SLAM-Tracking braucht Textur — fehlt.

Bestätigt durch:
- `journalctl -u earlyoom`: mem avail zum Kill-Zeitpunkt = **75 %** → earlyoom triggert nicht
- `journalctl -u systemd-oomd`: keine Kill-Events
- `nvidia-smi`-Peak-Zahlen der gestorbenen Runs (94–98 % VRAM)

## Diagnose-Sackgassen (falsche Spuren, die wir verfolgt haben)

Damit ein nächster Lauf das nicht wiederholt:

| Hypothese | Realität |
|---|---|
| Kernel-OOM-Killer | `dmesg` zeigte nichts. Kein Killed-Process-Eintrag. |
| Cursor Extension Host frisst RAM | Half (war bei 7.5 GB). Nach Restart 13 GB frei → Run starb trotzdem. |
| `vm.overcommit_memory=2` strict | `__vm_enough_memory` Denial im journal war aber für `claude`-Process (Node V8 reserved 137 GB virtual). Auf 0 gestellt — Run starb weiter. |
| earlyoom (`-m 10 -s 5`) | mem avail 75 % zum Kill-Zeitpunkt, weit über 10 % Threshold. |
| systemd-oomd cgroup-pressure | `oomctl` zeigt 0 % Pressure, keine Kill-Logs. |

Der echte Killer ist der **VRAM-Watchdog des Users**. Symptom: rc=137 ohne Traceback,
ohne dmesg/journal-Eintrag, aber peak GPU 94 %+.

## Wenn man es doch nochmal versuchen wollte

Realistischer Aufwand wäre **Tagesprojekt**, nicht Quick-Test. Voraussetzungen:

1. **KB-Rectify-Pipeline** schreiben (Vorbild: `scripts/prepare_ntu_viral.py`). `cv2.fisheye`
   API kann KANNALA_BRANDT, allerdings nur mit 4 Distortion-Params. KB hat hier k2/k3/k4/k5
   — entspricht OpenCV-Kn-Notation; Mapping checken.
2. Rectifizierte Bilder + neues K (mit `cv2.fisheye.estimateNewCameraMatrixForUndistortRectify`)
   in `*_rectified/` ablegen. Configs auf das neue K updaten.
3. Für VIO-Modus (falls Tracking sonst nicht hält):
   - `metadata/imu.txt`: `imu_data/data.csv` parsen → `[t_sec ax ay az gx gy gz]` pro Zeile
   - `metadata/camstamp.txt`: pro Bild `[t_sec filename]` aus Original-Timestamp im Dateinamen
   - `metadata/c2i.txt`: 4×4 Cam→IMU-Extrinsic. **Fehlt** — müsste aus `/tf_static` des
     ROS-bags extrahiert oder geschätzt werden. Default-Identity ist riskant.
   - Neuer Dataset-Loader analog zu `scripts/datasets/kitti_sync.py`.
   - Config: `mode: 'vio'`, `dataset.imu_delay`, pinhole `intrinsic`.

Selbst mit VIO bleibt das Problem mit dem Helikopter-Rotor in der front-Cam und der
texturarmen Main-Cam-Geometrie. Plus IMU-Tuning für 400 Hz vs. dem was VINGS erwartet.

## Stattdessen bewährte Datasets

Stand 2026-05-19 tatsächlich auf der Box:

| Dataset | Pfad | Config |
|---|---|---|
| smallcity | `/media/philipp/USB_STICK/datasets/smallcity/` | `configs/local/smallcity_full.yaml` (+ `smallcity/` Sweeps) |
| bonn RGB-D | `/media/philipp/USB_STICK/datasets/bonn/rgbd_bonn_dataset/` | `configs/local/bonn_crowd.yaml` |
| tum RGB-D | `/media/philipp/USB_STICK/datasets/tum/rgbd_dataset_freiburg3_walking_xyz/` | – (kein Config im Repo) |
| urbanscene | `/media/philipp/USB_STICK/datasets/urbanscene/` | `configs/local/urbanscene_polytech.yaml` |

**Nicht (mehr) da**, obwohl Configs existieren: AGZ (`configs/local/agz/*`, `agz_full.yaml`),
NTU-VIRAL eee_03 (`configs/local/ntu_eee_03/*`), KITTI, Waymo. Alte AGZ-Sweep-Outputs
liegen noch unter `/media/philipp/USB_STICK/ba/vings/agz_full/` — Source-Bilder fehlen.
