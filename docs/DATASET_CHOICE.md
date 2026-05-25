# Dataset-Choice Recap

Stand 2026-05-20. Welche Aerial-/Urban-Datasets fuer VINGS-Mono getestet, was klappt,
was nicht, und welche stehen noch aus. Kern-Befund: VINGS-Mono braucht **echtes RGB
+ kontinuierliche Trajektorie + nicht-zu-extreme Fisheye**. Photogrammetrie-Datasets
(viele Blickwinkel ohne sequentielle Aufnahme) und Stereo-Grayscale (kein Metric3D)
fallen damit raus.

## TL;DR

| Datensatz | Status | Hauptproblem |
|---|---|---|
| **MegaNeRF (Building, Rubble) – Nadir** | ✅ super | nur ~100-Bild-Sequenzen, jede startet woanders |
| **AGZ / Zurich Urban MAV** | ⏳ nächster Versuch | sehr viele Frames → braucht aggressiveren Selector |
| **MARS-LVIG (HKairport_GNSS03, AMtown03)** | ✅ Best-Run getestet | PSNR 23.7 / 23.4 mit use_metric=false; VIO strukturell tot |
| **NTU VIRAL** | ❌ tot | Stereo-Grayscale → Metric3D-Tiefentracker funktioniert nicht |
| **MUN-FRL Quarry (Drohne)** | ⚠ teilweise | Fisheye (rectifizierbar mit korrekten M600-KB-Koeffs), PSNR moderat |
| **MUN-FRL Bell412 (dataset5)** | ⚠ teilweise | gleiches Fisheye-Setup (eigene Koeffs ≠ M600), super Daten an sich |
| **Matrix City** | ❌ ungeeignet | Photogrammetrie-Renderings, keine konsistente SLAM-Sequenz |
| **Residence (UrbanScene3D)** | ❌ ungeeignet | gleiches Photogrammetrie-Problem |
| **TartanAir** | 📋 noch ungetestet | synthetisch, aber Konfigs sind da |
| **smallcity** | ✅ Test-Baseline | synthetisch, läuft seit Anfang als Smoke-Test |

---

## Was funktioniert: Nadir-Aerial mit kontinuierlicher Trajektorie

**Kern-Erkenntnis** aus den Versuchen: MegaNeRF Building (Nadir, lawnmower-Pattern
über einem Bauwerk) rekonstruiert super sauber. Sobald die Kamera nach unten zeigt
und kontinuierlich fliegt, ist VINGS happy:
- Metric3D liefert stabile Tiefen (Boden-/Dach-Ebene gut)
- DROID-Tracking hat reichlich Parallaxe (kein Pure-Rotation-Problem)
- 2D-Gaussians passen perfekt auf Bodenflächen

Aber der Datensatz ist **photogrammetrie-strukturiert**: pro „Sequenz" nur ~100 Bilder,
und die nächste Sequenz startet räumlich woanders. Daher kein konsistenter Map-Build
über mehrere Sequenzen hinweg.

---

## Pro Datensatz

### MegaNeRF – Building / Rubble  ✅ Reko super, ❌ keine SLAM-Continuity
Aerial-Nadir-Survey für NeRF-Training, ~100 Frames pro „Block". Reko-Qualität auf
einem Block ist exzellent (klassischer Use-Case für VINGS-Mono), aber Blocks setzen
nicht aneinander an → kein Multi-Block-Run möglich. Bleibt der Goldstandard für
„VINGS funktioniert grundsätzlich".

- Config: `configs/local/meganerf_building_pixsfm.yaml`, `meganerf_row0.yaml`
- Daten lokal: `~/Dokumente/datasets/building-pixsfm/`

### AGZ – Zurich Urban MAV  ⏳ nächster Versuch
Echtes RGB (GoPro Hero 4 @ 1920×1080), urban, 5-15 m Höhe, ~81k Frames über 2 km.
Letzter Versuch scheiterte am **schieren Frame-Volumen**: VINGS hat fast jeden Frame
als Tracker-KF akzeptiert, Mapper ist mitgelaufen, Output explodiert. **Plan:**
aggressiverer Frame-Selector (z.B. `nurbs_lvi` mit hohem Q, oder `coko_slam` mit
großem α) + ggf. `start_frame`-Skip um die hover-lastigen Anfangs-Sekunden zu
überspringen.

- Konfigs: `configs/local/agz_full.yaml`, `agz_sample.yaml`, `agz_sample_vista.yaml`
- Download: https://rpg.ifi.uzh.ch/zurichmavdataset.html (Sample 200 MB, Full 28 GB)
- Sensoren: RGB + GPS + IMU + Barometer
- Plattform: Fotokite MAV
- Lizenz: keine Einschränkungen
- Caveat: Rolling Shutter (GoPro 30 ms readout) bei schnellen Bewegungen

### MARS-LVIG (HKairport_GNSS03 + AMtown03)  ✅ getestet, super geeignet
DJI M300 RTK mit Hikvision RGB-Cam 2448×2048 BGR @ 10 Hz, Livox-IMU 200 Hz,
DJI-IMU 400 Hz, RTK-GPS. RGB ist **echtes Farbbild** (war anfangs Verdacht
Stereo-Grayscale wie NTU; das ist falsch). 23 Sequenzen verfügbar.

Getestete Sequenzen + Best-PSNR:
- **HKairport_GNSS03 (200f best)**: PSNR=23.68, SSIM=0.74, LPIPS=0.26 mit
  paperloss + use_metric=false + native PNG + Pose-Override.
- **AMtown03 (200f best)**: PSNR=23.40, SSIM=0.78, LPIPS=0.18 — gleiches Pattern.
- Längere Sequenzen (500+): PSNR fällt monoton wegen Tracker-Drift (siehe
  `docs/MARS_LVIG.md` für Details).
- **VIO ist strukturell unbrauchbar** wegen fehlender Cam-IMU-Extrinsik + LiDAR-
  Degeneration (UAVScenes-Paper bestätigt).

Daten + Calibration: `docs/MARS_LVIG.md`, Run-Patterns: `docs/RUN_CONFIG_PATTERNS.md`,
Pose-Override-Pipeline: `docs/POSE_OVERRIDE.md`, Segmentation+Three.js-Konzept:
`docs/SEGMENTATION_AMTOWN.md`.

Download-Trick (Drive-Quota umgehen): direkt von HuggingFace ziehen:
`https://huggingface.co/datasets/sijieaaa/UAVScenes/resolve/main/interval5_CAM_LIDAR.zip`
(28 GB für alle 20 Sequenzen) oder einzelne `.bag`-Files via Browser mit Login.

### NTU VIRAL  ❌ tot
Stereo-**Grayscale** (mono8) Fisheye, 752×480, indoor. Drei harte Blocker:
1. **Metric3D** ist auf RGB+ImageNet-Norm trainiert → Grayscale → schlechte Tiefen
   → Gaussian-Explosion.
2. **Fisheye-Rectification** erzeugt schwarze Ecken die Metric3D als Sky/Empty
   misinterpretiert.
3. **Indoor-GPS unbrauchbar**, Leica TS liefert nur Positionen nicht Orientierungen.

Nicht weiterverfolgen.

### MUN-FRL Dataset5 (Bell412)  ⚠ Fisheye limitiert die PSNR
Hubschrauber-Survey über Quarry. FLIR BFS-U3-16S2M-BD 1440×1080 RGB, KB-Distortion.
Run läuft, aber das **Fisheye lässt sich mit den offiziellen KB-Koeffizienten nicht
gut genug rectifizieren** – nach `cv2.fisheye.undistortImage` mit balance=0 bleiben
sichtbare Distortions am Rand. PSNR Plateau bei ~17 dB.

Offene Frage: gibt es bessere Calibration-Daten (z.B. ein präziseres Distortion-Modell
als KB4, oder eine recalibration), die mehr aus diesem an sich super RGB-Datensatz
rausholen würde?

- Daten lokal: `~/Dokumente/datasets/dataset5/extracted/`
- Configs: `configs/local/dataset5/`
- Rectify-Script: `scripts/rectify_dataset5_{front,nadir}.py`

### MUN-FRL Quarry (DJI M600 Drohne)  ⚠ Fisheye + Hover-lastig, aber lösbar
DJI M600 trägt FLIR-BFS-Payload mit **eigener KB-Kalibrierung** (NICHT identisch
zu Bell412, wie initial angenommen — der Principal-Point ist ~50 px verschoben).
Offizielle M600-Werte aus `mun-frl-vil-dataset.readthedocs.io/sensor_calibration.html`:
mu=854.383 mv=853.286 u0=780.325 v0=520.691, D=[-0.07938, 0.02228, -0.03852, 0.01347].
4627 Frames @ 20 Hz aus `flight_dataset2.bag` extrahiert.

Effekt der korrekten Kalibrierung (Quarry 100 Frames, sonst gleiche Config):
PSNR 19.13 → 19.37, SSIM 0.399 → 0.444 (+11 %), LPIPS 0.785 → 0.688 (−12 %).
Der Boost ist primär strukturell (SSIM/LPIPS), nicht photometrisch — passt zur
Hypothese dass das Tracking durch den Principal-Point-Offset systematisch off war.

Spezifika dieser Drohne:
- **Hover-Pre/Post-Sequenzen**: Frames 0-1400 + 3800-4627 sind statisches Hovering
  (pix-diff ~2.3, |w|<1.5 deg/s). Nur Frames 1500-3700 sind aktiver Flug.
- **Pure-Rotation-Phasen** (Frames 2200-2900): |w| bis 28 deg/s → keine
  Parallaxe → DROID-Tracking kommt nicht klar.
- Beste Subsequenz: 3400-3700 (transl-dominant), aber selbst nach Rectification nur
  PSNR ~17-19. Das Fisheye-Problem aus dataset5 vererbt sich hier 1:1.

- Daten lokal: `~/Dokumente/datasets/quarry/{images, images_rect, metadata}/`
- Config: `configs/local/quarry/quarry_3400_3700_rect.yaml`
- Calibration: identisch zu dataset5 nadir (`scripts/rectify_dataset5_nadir.py`)

### Matrix City  ❌ Photogrammetrie-Renderings
Synthetisches Urban-Dataset (Unreal-Engine). Was bei näherem Hinschauen klar wurde
(`scripts/datasets/generic_vo.py` + JSON-Analyse 2026-05-20):
- `aerial/train/block_X` (185-256 Frames pro Block) ist ein **Survey-Grid**: 37 m
  Schritte in Reihen mit 350 m Sprüngen zwischen Reihen
- `aerial/test/block_X_test` ist ein NeRF-Holdout (jeden N-ten Frame des Trainings)
  → noch sprunghafter
- `street/{test,train}` ist ähnlich subsampled

Ich hatte das übersehen und block_3 trotzdem laufen lassen — die 256 Frames in
einer einzigen `block_3.tar`-Datei sind tatsächlich render-order kontinuierlich
(max consecutive pix-diff 46), aber zwischen verschiedenen Blocks gibt es keine
sinnvolle Sequenz. PSNR-Benchmark 23.5 dB ist ok, aber das ist eben kein echter
SLAM-Datensatz.

- Daten lokal: `~/Dokumente/datasets/matrixcity/block_3/` (256 Frames, behalten)
- Config: `configs/local/matrixcity/matrixcity_aerial_block3_train.yaml`

### Residence / UrbanScene3D  ❌ Photogrammetrie
Analog zu Matrix City — multiple Viewpoints aus verschiedenen Höhen/Winkeln,
optimiert für Photogrammetrie/Multi-View-Stereo. Kein konsistentes
SLAM-Bewegungsprofil. `seg1/seg2/seg3` sind manuelle Splits, aber die
zugrundeliegende Aufnahme bleibt unstructured.

- Daten lokal: `~/Dokumente/datasets/Residence/{photos, seg1, seg2, seg3}/`
- Configs: `configs/local/residence_{A,seg1,seg2,seg3}.yaml`
- Loader: `scripts/datasets/urbanscene3d.py`

### TartanAir  📋 ungetestet
Synthetisch, drone-flying-through-environment, kontinuierliche Trajektorien,
RGB + Depth + Pose. Gleiches Trainings-Set wie DroidNet — VINGS sollte hier
besonders gut performen. Bisher noch nicht real getestet.

- Configs vorbereitet: `configs/local/tartanair_oldtown_P000.yaml`, `tartanair_soulcity_P000.yaml`
- Loader: `scripts/datasets/tartanair.py`
- Würde sich anbieten als Sanity-Check, ob VINGS bei "perfekten" synthetischen
  Bedingungen die Reko-Qualität schafft, die wir bei MegaNeRF auch sehen.

### smallcity  ✅ Smoke-Test-Baseline
Synthetisch, läuft seit Anfang als Sweep-Baseline (200/800/1k/full = 5822 Frames).
Konsistente kontinuierliche Trajektorie. Profiling-Zahlen aus `KEYFRAME.md`
stammen von hier.

---

## Weitere Loader im Repo, ungenutzt für die BA

| Loader | Datensatz | Bewertung |
|---|---|---|
| `bonn.py` | Bonn RGB-D Dynamic | Indoor, RGB-D – nicht aerial |
| `bundlefusion.py` | BundleFusion | Indoor RGB-D |
| `kintinuous.py` | Kintinuous | Indoor RGB-D |
| `kitti_sync.py` | KITTI Odometry | Bodenfahrzeug, stereo |
| `kitti360_unsync.py` | KITTI-360 | Bodenfahrzeug, stereo |
| `replica.py` | Replica | Indoor synth RGB-D |
| `scannetv1.py` | ScanNet | Indoor RGB-D |
| `tumrgbd.py` | TUM RGB-D | Indoor RGB-D |
| `waymo.py` | Waymo | Bodenfahrzeug |
| `weilai.py` | Weilai (?)  | unklar |
| `realsense_vio.py` | Live RealSense | für eigene Aufnahmen |
| `phone*.py`, `mobile*.py` | Mobile Phone | für eigene Aufnahmen |

Aerial-Bereich ist dünn — die ganzen Loader sind primär für Indoor/Ground-Vehicle
da. Für die BA-Frage „KF-Selektion auf Aerial" ist die enge Auswahl aus dem TL;DR
oben tatsächlich der Stand.

---

## Weitere Aerial-Datasets aus der Literatur

Kurze Einordnung: was sind die "üblichen Verdächtigen" in der Aerial-Vision-Welt,
und passen sie auf das VINGS-Mono-Problem?

VINGS-Mono braucht: **echtes RGB**, **kontinuierliche Kamera-Trajektorie** (= aufeinander
folgende Frames mit kleinem Bewegungs-Delta, keine Survey-Grids/Photogrammetrie-Splits),
**moderate Fisheye-Verzerrung** (oder Pinhole), und sinnvolle Größe (>100 Frames
Sequenz). Daran scheitern die meisten "Aerial-Datasets" weil sie für ganz andere
Tasks gebaut wurden — Object-Detection, Multi-Object-Tracking, Semantic Segmentation,
oder NeRF/Photogrammetrie-Surveys.

### VisDrone2018 (Zhu et al., arXiv:1804.07437)  ❌ Detection/Tracking, kein SLAM
263 Video-Clips + 10.209 Bilder von DJI Mavic/Phantom 3/4 über 14 chinesischen Städten,
bis 3840×2160. **Aufgaben:** Object-Detection in Images/Videos, Single- und
Multi-Object-Tracking. Annotationen sind ausschließlich Bounding-Boxes für 10
Objekt-Kategorien (Fußgänger, Autos, Busse, …) plus Occlusion/Truncation/Visibility.
- **Real angeschaut: nur ~10 Bilder pro Sequenz** — viel zu wenig für VINGS-Mono.
  Selbst wenn der Tracker bei 10 Frames startet, hat er keine Multi-View-Constraints
  für sauberes Bundle Adjustment, und der Mapper schafft auf 10 Frames keinen
  brauchbaren Gaussian-Build. Die Statistik im Paper (179k Frames in 263 Clips =
  durchschn. 681 pro Clip) bezieht sich vermutlich auf die internen Video-Annotations-
  Splits; in der Praxis sind die einzeln verfügbaren Sequenzen extrem kurz.
- Zusätzlich: **keine Posen, keine Depth-GT, keine Calibrationsdaten**.
- Drone-Style ist tracking-fokussiert (Kamera folgt einem Objekt) → ungeeignet für
  Reko-fokussierten Mapping-Lauf.
- Fazit: nicht weiter verfolgen

### UAVDT (Du et al., arXiv:1804.00518)  ❌ Detection/Tracking, kein SLAM
100 Videos, ~80k Frames von einem DJI Inspire 2 an chinesischen Verkehrs-Hotspots
(Highways, Kreuzungen). Resolution nur **1080×540**, BoundingBox-Annotationen für
3 Vehicle-Klassen plus Wetter/Höhen/View-Angle-Attribute.
- Selbe Limitationen wie VisDrone: kein GT für Posen oder Depth, niedrigere
  Auflösung
- Höhen- und View-Annotationen wären für VINGS-KF-Selector-Tests nicht relevant
- Fazit: würde ich nicht angehen

### Aerial-Tracking-Benchmark (Taufique et al., RIT, 2021)  ❌ Tracker-Benchmark, kein neues Dataset
Bewertet 10 Deep-Tracker auf 4 existierenden Aerial-Datasets (OTB-Aerial-Subset,
UAV123, UAV20L, DTB70). Stellt **selber kein neues Dataset** bereit — referenziert
nur tracking-fokussierte Datasets.
- Die referenzierten Datasets (UAV123 etc.) haben Bounding-Box-GT für Single-Object-
  Tracking, kein Pose/Depth
- Fazit: für VINGS nicht direkt verwertbar; das Paper ist eher ein
  "Was-funktioniert-bei-aerial-tracking"-Übersicht

### UAVid (Lyu et al., 2018)  ⚠️ semantic-only, aber gute Roh-Sequenzen
30 4K-Video-Sequenzen aus Drohnen-Oblique-View (~50 m Flughöhe), Semantic-
Segmentation-Annotationen mit 8 Klassen (Building/Road/Tree/Low-Veg/Static-Car/
Moving-Car/Human/Clutter). Pro Sequenz 10 Frames im 5-Sekunden-Abstand annotiert.
- **Pro:** 4K oblique-Aerial-RGB, Drone-Style, kontinuierliche Video-Sequenzen
  (nicht photogrammetrie-grid), Daten sind öffentlich
- **Contra:** Annotationen sind Semantic-Maps, **keine Posen, keine Depth-GT** →
  für VINGS müssten wir per SfM (Colmap) selbst Posen ziehen und auf Reko-Qualität
  hoffen
- Wäre eher ein "wir nehmen die ungelabelten Roh-Videos und schauen ob VINGS
  läuft" — analog zu AGZ, mit dem Caveat dass die annotierten Frames nur als
  Test-Bilder für Semantic-Use-Cases nützlich wären
- Fazit: niedrig priorisiert, AGZ ist näher dran

### AGS / WHU-OMVS (Aerial 3DGS Paper)  ✅ direkt unsere Domäne — anschauen
Das Paper ist eine 3DGS-basierte Large-Scale-Aerial-Surface-Rekonstruktions-Methode
und vergleicht auf:
- **WHU-OMVS**: 268 oblique-Aerial-Images, 3712×5504, 10 cm Ground Resolution,
  Flughöhe 550 m, ~850×700 m Areal. **MIT Depth-GT** (für Eval) und 5 Viewpoints
  pro Stelle (1 Nadir + 4 Oblique). Stammt vermutlich aus dem WHU MVS Benchmark
  (Wuhan University)
- Mill-19 + UrbanScene3D nutzen sie nur für Rendering-Qualität, nicht Geometrie
  (gleiche Photogrammetrie-Limitierung die wir bei Matrix City/Residence hatten)

WHU-OMVS ist die interessanteste Anbindung: aerial multi-view mit echter
**Depth-GT** für quantitative SLAM-Eval. **Wichtig vor dem Download zu prüfen**:
ob die 268 Bilder eine kontinuierliche Trajektorie sind (= 5 Viewpoints pro Stelle,
also ~53 Stellen → Trajektorie über die Stellen) oder ein klassisches Photogrammetrie-
Grid (= Aufnahme über mehrere Höhenebenen ohne sequentielle Reihenfolge). Wenn ersteres
brauchbar → kann das WHU-OMVS direkt als zweiter aerial-RGB-Benchmark neben AGZ
laufen. Wenn letzteres → fällt in die Matrix-City-Kategorie.

- **Priorität: hoch** (zusammen mit AGZ-Retry und MARS-LVIG)

### BlendedMVS (Yao et al., CVPR 2020)  ⚠️ MVS-Training-Set, kein SLAM
17k+ Bilder aus 113 Szenen, generiert durch Mesh-Rekonstruktion → Rendering →
Blending mit Original-Lichtinformation. Dense-Depth-Maps als GT. Ursprünglich zum
**Training von MVSNet/R-MVSNet/Point-MVSNet** gebaut, nicht für SLAM-Eval.
- Die 113 Szenen sind voneinander unabhängig (photogrammetrie-strukturiert pro
  Szene) → kein langer SLAM-Trajektorien-Lauf möglich
- **Pro:** ist als VINGS-Pretraining-Datenquelle interessant — könnte fürs
  Metric3D-Finetuning oder als alternatives Trainingsset für den DROID-Tracker
  relevant werden, falls ihr in der BA in diese Richtung wollt
- Für die KF-Selector-Eval (= unser Hauptfokus) **nicht direkt verwendbar**
- Fazit: notieren als potentielle Pretraining-Quelle, nicht als SLAM-Benchmark

---

## Zusammenfassung der Paper-Bewertung

| Paper / Dataset | RGB? | Pose/Depth-GT | Continuous-Seq? | Aerial? | Priorität |
|---|---|---|---|---|---|
| VisDrone2018 | ✓ | ✗ (nur BBox) | tracking-style | ✓ | ❌ |
| UAVDT | ✓ | ✗ (nur BBox) | tracking-style | ✓ | ❌ |
| Aerial-Tracker-Bench (RIT) | — | ✗ | — | — | ❌ kein eigenes Set |
| UAVid | ✓ | ✗ (nur Semantic) | ✓ (5s-Abstand) | ✓ oblique | ⚠ niedrig |
| **WHU-OMVS (AGS)** | ✓ | ✓ Depth-GT | zu prüfen | ✓ MVS-aerial | ✅ **hoch** |
| BlendedMVS | ✓ | ✓ Depth-GT | ✗ (pro Szene grid) | ⚠ gemischt | ⚠ als Training-Set |

Aus den fünf Papern springt klar **WHU-OMVS** heraus als nächster sinnvoller Test-
Kandidat. Der Rest ist entweder für andere Tasks gebaut (Detection/Tracking/Semantic)
oder photogrammetrie-strukturiert (BlendedMVS-Szenen).

---

## Was als nächstes machen

1. **AGZ retry** mit aggressivem Frame-Selector (Coko-SLAM α=0.5+ oder NURBS-LVI
   mit hohem Q). Das ist der naheliegendste echte Aerial-RGB-Datensatz.
2. **MARS-LVIG manuell ziehen** (eine Sequenz reicht zum Probieren), Calibration-
   YAML prüfen — wenn RGB und nicht zu krass Fisheye, sollte das gut laufen.
3. **TartanAir P000** als synth-Sanity-Check, ob die VINGS-Pipeline auf perfekten
   Bedingungen die erwarteten PSNRs liefert (>30).
4. **Dataset5 / Quarry Fisheye recalibration** wenn jemand bessere
   Distortion-Koeffs liefert — das wäre der weitestreichende Hebel weil die
   Datensätze an sich sehr gut sind.

Was rausfällt: Matrix City, Residence/UrbanScene3D, MegaNeRF (für Multi-Block-
Continuity), NTU VIRAL.
