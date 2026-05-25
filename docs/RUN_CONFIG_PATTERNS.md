# Run-Config Patterns — was funktioniert wann

Empirische Sammlung aus den MARS/Quarry/Matrix-City/MegaNeRF-Sweeps Mai 2026.
Diese Hebel haben den größten Einfluss auf PSNR/Stabilität.

## Die wichtigsten 5 Knöpfe

### 1. `use_metric: false` für Aerial-Nadir-Szenen ★

**Hammer-Befund**: VINGS auf MARS-HKairport mit Metric3D AN = PSNR 20.93, mit
Metric3D AUS = PSNR **23.68**. +2.75 dB durch einen Boolean.

**Warum**: Metric3D wurde auf gemischten Real-World-Daten trainiert (indoor +
outdoor + driving). Bei aerial-flach-Szenen mit minimaler Tiefenvariation liefert
es noisy/falsche absolute Tiefen die im Mapper mit `depths_cov=0.01` als harte
Constraints reingehen → Gaussians werden auf falsche 3D-Positionen platziert →
lateraler Blur.

**DroidNet** dagegen ist auf TartanAir trainiert (synthetisches Drone-Flying) —
exakt unsere Domäne. Wenn man `use_metric: false` setzt, nutzt der Mapper die
DroidNet-Disparity die durch das interne BA optimiert wird.

**Wann nicht**: Bei Indoor/Ground-Vehicle-Szenen kann Metric3D weiterhelfen.
MegaNeRF-Building (Aerial-Nadir) deaktiviert es laut config auch.

### 2. Paper-Loss-Weights (Outdoor)

```yaml
training_args:
  loss_weights:
    rgb_loss:    1.0
    depth_loss:  0.5    # statt 1.0 — weicheres depth-constraint
    normal_loss: 0.1
    alpha_loss:  0.1    # statt 1.0 — weniger restrictive opacity
    dist_loss:   0.0
```

Quelle: Paper Eq. 12 + reproduziert in `configs/local/dataset5/dataset5_front_long_paperloss.yaml`.
Effekt auf MARS-HKairport-500fr: 19.74 → 20.93 (+1.2 dB).

### 3. `mapper_kf_skip` — der VRAM-vs-Quality Trade-off

| Wert | Effekt | wann |
|---|---|---|
| 1 | jeder Tracker-KF → Mapper. Max-Quality, max-VRAM. | < 500 Frames, kurze Sequenz |
| 2-3 | balanced | 500-1000 Frames |
| 5 | wie Paper-Default (sparse mapping) | langes Outdoor (> 1000 Frames) oder VRAM-eng |

Drift-Pattern: bei langen Sequenzen ist Skip=1 nicht besser als Skip=2-3, weil
der Mapper überfittet. AMtown03 1000f: Skip=1 → PSNR 18.25, Skip=2 → PSNR 17.52.
HKairport 200f: Skip=1 → 23.68.

### 4. `image_size` (Frontend Tracker-Resolution)

Drone-aerial 4:3 → empirisch:
- **240×288** (1.20 aspect) — VRAM-konservativ, OK bei half-res
- **288×360** — mehr Pixel für Tracker, bessere Pose-Estimation, etwas mehr VRAM
- **384×456** — wie MegaNeRF-Config, native-res-tauglich
- 384×512 — MegaNeRF original (4608×3456)

Höhere Auflösung bessert Pose-Estimation, kostet aber proportional Tracker-VRAM
(ConvGRU + Cost-Volume).

### 5. `intrinsic` Native vs Half-Res

| | Native (2448×2048) | Half-Res (1224×1024) |
|---|---|---|
| PSNR | Baseline | -3 bis -4 dB |
| GPU peak | ~9-10 GB | ~5-6 GB |
| max stable Sequenz-Frames | ~150 (VRAM wand) | ~500+ |
| Mapper-Render-Cost | proportional zu W×H | ¼ davon |

**Mit JPEG-Quantisierung**: zusätzliche -1-2 dB durch double-encoding (Bag→JPEG
→ extraction → JPEG nochmal). PNG-Extract empfohlen für native runs.

## VRAM-Watchdog ~8 GB

Bei dieser GPU (RTX-Klasse, 8 GB Watchdog laut Memory) hart bei **~150 mapped
Frames mit native intrinsic**. Strategien:

| Symptom | Hebel |
|---|---|
| rc=137 nach n_frames=150 | mapper_kf_skip von 1 → 3-5 |
| Convey-spikes triggern Watchdog | distance_threshold 3.0 NICHT 2.0 (zu aggressive offload) |
| native res 2448 zu groß | half-res 1224 → reicht für 500+ Frames |
| Pose-Override aktiv | spart ~3-4 GB VRAM (siehe POSE_OVERRIDE.md) |

## Init-Phase-Trap

**Wichtige Erkenntnis** aus AMtown03-Runs: ein 500-Frame-Run der bei
`start_frame=0` startet → PSNR 15. Der gleiche 500-Frame-Run mit `start_frame=200`
→ PSNR **23**.

Erklärung: der DROID-Tracker braucht ein paar dutzend Frames Warmup mit
konsistenten Features. Wenn die Sequenz-Anfang zu schnelle Drohnen-Bewegung oder
zu komplexe Szene hat, initialisiert er schlecht und driftet die ganze Map.

**Workaround**: 50-100 Frames Skip am Sequenz-Anfang. Bei stationärem Hover am
Anfang sogar mehr (z.B. MARS HKairport startet mit Take-off — Frames 0-1400 sind
Hover/Vertikal-Take-off, erst ab 1500 brauchbar).

## Drift bei langen Sequenzen

PSNR fällt monoton mit Sequenz-Länge sobald > 500 Frames:
- 200f: 23.4
- 500f: 22.8
- 1000f: 18.2
- 1800f: 14.0

Ursachen:
- Akkumulierte Pose-Drift (1-2% pro 100 Frames bei pure VO)
- Mapper bekommt mit jedem KF mehr widersprüchliche Constraints
- Active-Window-BA (12 Frames) kompensiert nur kurzfristige Drift

**Mitigationen**:
- Pose-Override mit GT-Posen (DJI-RTK) — half bei HKairport +0.5 dB, bei AMtown nur +0.4 dB
- Storage-Manager + aggressive CPU-Offload — verhindert OOM, nicht Drift
- Loop-Closure: VINGS hat ein `loop/`-Modul aber nicht in den Standard-Configs

## Dataset-spezifische Patterns

### MARS-LVIG (HKairport, AMtown)
- `use_metric: false` ★
- paperloss
- mapper_kf_skip=1 für < 500 frames, =2-3 für längere
- native intrinsic für max-PSNR, half-res für VRAM-eng
- start_frame ≥ 100-200 (Take-off skippen)

### Quarry / dataset5 (Bell412)
- `use_metric: true` (Helikopter-Sensor moderater Höhe → Metric3D hilft)
- KB-rectified Bilder als Input (siehe QUARRY_DJI_M600.md)
- separate Calibration für Bell412 vs DJI M600

### MegaNeRF Building/Rubble
- `use_metric: false`
- mapper_kf_skip=1
- buffer=80, distance_threshold=30 (kurze Sequenz, kein Offload nötig)
- Geht super weil Posen via PixSfM mitgegeben werden... wait, eigentlich nicht.
  MegaNeRF-Loader hat `c2i = np.eye(4)`, Posen werden vom DROID geschätzt.
  Es funktioniert trotz Self-Estimation weil die Bilder hochaufgelöst (4608×3456)
  + saubere Nadir-Aufnahme.

### Matrix City
Photogrammetrie-strukturiert, nicht SLAM-tauglich (Survey-Grid mit 37m
Schritten). Aber wenn man einen einzelnen Block nimmt, ist es kontinuierlich
genug → PSNR 23.5 bei 256 Frames mit Standard-Config möglich.
