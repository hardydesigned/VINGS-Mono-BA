# Quarry (DJI M600 Drohne) — Calibration + Rectification

Erkenntnisse aus den Quarry-Runs auf der MUN-FRL DJI-M600-Drohne (Dataset
`flight_dataset2.bag` von ravindujhc/MUN-FRL).

## Sensor

DJI M600 hexacopter mit **FLIR BFS-U3-16S2M-BD** Camera (1440×1080, BGR @ 20 Hz),
Xsens MTi-30 IMU, Velodyne LiDAR. Quarry-Sequenz ist Aerial-Nadir-Flug über
Steinbruch.

## Kalibrierung — separate Werte gegenüber Bell412!

**Wichtige Erkenntnis**: Die MUN-FRL-Doku publiziert **separate KB-Koeffs** für
Bell412 (dataset5) und DJI M600 (Quarry). Sie sind **NICHT identisch**, obwohl
beide das gleiche FLIR-Modul tragen:

| | Bell412 (dataset5) | DJI M600 (Quarry) | Δ |
|---|---|---|---|
| fx (mu) | 829.224 | **854.383024** | +3.0% |
| fy (mv) | 829.454 | **853.285954** | +2.9% |
| cx (u0) | 833.937 | **780.324522** | **−53 px** |
| cy (v0) | 562.509 | **520.690672** | **−42 px** |
| k1 | -0.0764245 | -0.07937700 | leicht |
| k2 | 0.0322856 | 0.02228435 | leicht |
| k3 | -0.0445168 | -0.03852023 | leicht |
| k4 | 0.0163317 | 0.01346873 | leicht |

Der größte Unterschied ist der **Principal-Point-Offset von ~50 Pixel**. Wer
Bell412-Werte auf M600-Bilder anwendet, bekommt sichtbar verschobene Rectification.

Quelle: `mun-frl-vil-dataset.readthedocs.io/sensor_calibration.html` Sec. "Camera
Intrinsic Calibration", separate Blöcke für "For Bell412 Datasets" und "For DJI
M600 Datasets".

## Rectification-Script

Nutze **Kannala-Brandt** (`cv2.fisheye.*`), NICHT Plumb-Bob (im Gegensatz zu MARS):

```python
K = np.array([[854.383024, 0, 780.324522],
              [0, 853.285954, 520.690672],
              [0, 0, 1]])
D = np.array([-0.07937700, 0.02228435, -0.03852023, 0.01346873])
new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
    K, D, (1440, 1080), np.eye(3), balance=0.0)  # balance=0: max valid pixels
map1, map2 = cv2.fisheye.initUndistortRectifyMap(
    K, D, np.eye(3), new_K, (1440, 1080), cv2.CV_16SC2)
```

Post-Rect K: fx=704.017, fy=703.113, cx=834.319, cy=513.728.

## Rand-Distortion → Crop

Trotz korrektem KB-Rectification bleiben an den Bildrändern Residuen:
- **Linker Rand**: ~130 px violetter Saum (chromatische Aberration nicht modelliert)
- **Unten**: ~50 px schwarzer Saum
- Oben/rechts: ~30 px

Empfohlener Crop: **130/30/30/50** (left/right/top/bottom) → resultiert in 1280×1000.
Post-Crop K mit Principal-Point-Shift:
- cx_new = cx_rect - 130 = 704.319
- cy_new = cy_rect - 30 = 483.728

## Run-Progression (alle 100 Frames, Quarry frame 3400-3500)

| Variante | PSNR | SSIM | LPIPS | Bemerkung |
|---|---|---|---|---|
| Roh-Fisheye + Bell412-K (falsch) | 19.13 | 0.399 | 0.785 | unverzerrt-Annahme |
| Rect + Bell412-KB (falsch) | 19.37 | 0.444 | 0.688 | falsche Koeffs |
| Rect + M600-KB (korrekt) | 19.69 | 0.463 | 0.660 | KB-Fix |
| Rect + Crop (130/30/30/50) | **19.69** | **0.463** | **0.660** | gleiche metric, sauberes Bild |
| Rect + M600 + no_metric=false | 20.71 | 0.609 | 0.520 | paperloss + DroidNet-depth |

**Hauptbefund**: KB-Korrektur + Edge-Crop verbessern visuell deutlich (klare
Asphalt-Strukturen statt Streifen), PSNR-Boost vor allem in SSIM (+0.06).

Quarry ist visuell ein "schwerer" Datensatz: texturarmer Steinboden + Schatten +
große Yaw-Rotationen. PSNR-Plateau bei ~20 ist hier physikalisch.

## Spezifika der Sequenz

- 4627 Frames @ 20 Hz, 231s Total-Length
- **Frames 0-1400**: Pre-flight Hover/Ground-Stand → unbrauchbar für SLAM
- **Frames 1500-3700**: aktiver Flug
- **Frames 3800-4627**: Hover/Landing → unbrauchbar
- **Frames 2200-2900**: hohe Rotation (|w|=10-28 deg/s) → schlecht für Tracker (Pure-Rotation kein Parallax)
- **Best Subset 3400-3700**: transl-dominant, |w| med 6.4 deg/s, pix-diff med 40

## Files

```
~/Dokumente/datasets/quarry/
├── images/                  # 4627 raw fisheye JPEGs
├── images_rect/             # 4627 KB-rectified (volle Auflösung)
├── images_rect_cropped/     # cropped Variante (130/30/30/50)
├── metadata/
│   ├── c2i.txt
│   ├── camstamp.txt
│   └── imu.txt
└── quarry_3400_3700_preview.mp4
```

Config: `configs/local/quarry/quarry_3400_3700_rect_cropped_nometric.yaml`
