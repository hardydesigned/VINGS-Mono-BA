# VIO + GPS auf amtown03 (Stage C) — Session-Erkenntnisse 2026-05-31/06-01

Diese Doku fasst die komplette VIO/GPS-Untersuchung auf amtown03 zusammen: vom
Full-Sweep über die VIO-Aktivierung, GPS-Gewichtung, Extrinsik-Optimierung bis zum
direkten Verwenden der DJI-Onboard-Posen.

---

## In einfachen Worten (TL;DR)

Wir wollten wissen, ob sich die **Posen-Drift** der Vollsequenz-Rekonstruktion mit
**IMU (VIO)** und **GPS** verbessern lässt. Kurzfassung:

1. **Reines VO ist erstaunlich gut.** Über ein 400-Frame-Cruise-Stück: ATE 12.76 m.
2. **VIO allein macht es *schlechter*** (ATE 38 m) — weil die Kamera-IMU-Kalibrierung
   nicht exakt stimmt und dadurch die Schwerkraft falsch herausgerechnet wird. Die
   Trajektorie bläht sich dann um das ~20-fache auf.
3. **VIO mit *stark gewichtetem* GPS gewinnt klar** (ATE 6.14 m, ~2× besser als VO).
   GPS zwingt die Positionen auf die Metrik und überschreibt die kaputte IMU-Integration.
4. **Noch mehr GPS-Gewicht bringt nichts mehr** — ab cm-Genauigkeit ist es gesättigt.
5. Der verbleibende Fehler ist **Orientierung** (nicht Position) und hängt an der
   Kamera-IMU-Extrinsik. Die beste Extrinsik (Gravity-aligned + Yaw 270°) hebt psnr_ho
   von 11.6 auf **12.05** — danach **Plateau** (Roll/Pitch holen nichts mehr).
6. **DJI-Onboard-Posen direkt verwenden ist KEINE Alternative** (psnr_ho 6.83): die
   Position stimmt (ATE 0), aber die Onboard-Orientierung ist für Gaussian-Splatting
   zu verrauscht. Held-out-Render belohnt Selbst-Konsistenz über absolute Genauigkeit.
7. **Map-Geometrie vs. Schärfe ist ein fundamentaler Trade-off** (Abschnitt 7d/7e):
   DROID-Tiefe rendert scharf, ist aber ein degenerierter Strang; Metric3D macht sie
   flach/korrekt, aber unschärfer. Selbst Metric3D *als BA-Prior* hilft nicht — die
   photometrische BA kollabiert die Geometrie zurück zum Strang. „Flach UND scharf"
   bräuchte LiDAR-Tiefe oder mehr Parallaxe. **GPS/VIO fixen die Trajektorie, nicht die
   Tiefen-Degeneration, die die Map ruiniert.**

Wichtig fürs Verständnis: **GPS fixt die Position, die Extrinsik fixt die Orientierung.**

**Bestes Rezept:** VIO + hartes GPS + Gravity-aligned-y270-Extrinsik → **ATE 6.14 m
(2× besser als VO), psnr_ho 12.05** (400-Frame-Cruise-Slice).

---

## 1. Datensatz-Provenance: amtown03 = MARS-LVIG „AMtown"

- amtown03 stammt aus **MARS-LVIG** (HKU-MARS); **UAVScenes** (ICCV 2025,
  `github.com/sijieaaa/UAVScenes`) baut darauf auf. Sensoren: **Hikvision-Kamera**
  + **Livox Avia LiDAR/IMU** (rigides Payload, **kein Gimbal**).
- Unsere `images_all` (6199 Frames) **sind die Hikvision-Kamera bei halber Auflösung** —
  die Config-Intrinsik matcht exakt die publizierte AMtown-Kalibrierung
  (fu 726.64 = 1453.28/2, cu 520.89 = 1041.78/2, cv 586.09 = 1172.18/2).
- **Wir haben nur einen Teilstand**: `images_all/` + `metadata/`. Es fehlen LiDAR-
  Punktwolken, Semantik-Labels, die offiziellen Calib-Dateien und der originale
  Trim-Bildordner (vermutlich auf dem ausgefallenen USB-Stick, siehe Memory).
- **Zwei Timelines im Datensatz:**

  | | Frames | Zeit-Offset | IMU | GT |
  |---|--:|---|---|---|
  | `camstamp_all` (full_6199f) | 6199 | 0 s | `imu_dji.txt` (400 Hz, Z-up) | RTK + dji_poses |
  | `camstamp` (MARS-LVIG-Trim) | 2000 | +290 s | `imu_livox.txt` (208 Hz, **X-down**) | — |

  Die Livox-Daten decken nur das Fenster **images_all[2896:4904]** ab (Livox-Start
  +290 s nach Sequenzbeginn). Für den VIO-Spike haben wir genau dieses Fenster
  (`start_frame=2916, max_frames=400`) genutzt — die Hikvision-Frames stecken in
  `images_all`, kein Re-Download nötig.

**Publizierte AMtown cam↔LiDAR-Extrinsik** (aus `calibration_results.py`, von GitHub):
```
camera_ext_R = [ 0.00298088, -0.999728,   -0.0231416,
                -0.00504636,  0.0231263,  -0.99972,
                 0.999983,    0.00309683, -0.00497605 ]   # ~90°-Permutation
camera_ext_t = [ 0.0025563,   0.0567484,  -0.0512149 ]    # ~5-7 cm Hebel
```
Das ist cam↔LiDAR. Die Livox-Avia-IMU↔LiDAR-Rotation wird als ~Identität approximiert
(Avia-IMU ist grob LiDAR-aligned) — das ist eine der Fehlerquellen, siehe Abschnitt 6.

---

## 2. Der full_6199f-Sweep: Ergebnisse + VRAM-Wand

Voller amtown03-Sweep (6199 Frames, alle Selektoren, faire Metriken). **67 Runs:
20 OK, 47 FAIL.** Auswertung in `scripts/analyze_sweep_full_6199f_fair.ipynb`.

- **Nur die 20 OK-Runs sind vergleichbar** (`psnr_ho`/`ate_rmse_m`). Die 47 FAIL haben
  KEINE fairen Metriken (vor/während fair_eval gestorben).
- **42 der 47 FAILs sterben MITTEN im Hauptlauf** an der VRAM-Wand (rc=137, 8–9.8 GB,
  verteilt über Frame 223…4890) — nicht im fair_eval. Nur **4** Runs liefen komplett
  durch und crashten erst im fair_eval-Render.
- Survivorship-Bias: fast nur die dünnen `two_gate_v2`-Varianten überleben; map-dichte
  Selektoren (vista, aim, coko, nurbs…) fehlen komplett.
- Alle Zahlen sind schwach (psnr_ho 6–13 dB, ATE 208–253 m), und der fair_eval-**Scale
  schwankt 1.2×–84×** run-zu-run → monokularer Maßstab ist bei Nadir-Aerial praktisch
  unbeobachtbar. **Genau das motiviert VIO/GPS.**
- Lektion: `frontend.save_buffer_size` MUSS ≥ `max_frames` sein (sonst IndexError exakt
  bei Frame == save_buffer_size, unabhängig vom Selektor). Konvention: max_frames×1.3.

---

## 3. VIO: läuft, aber `gtsam.GPSFactor` ist an VIO geschweißt

- Der `gtsam.GPSFactor` (`depth_video.py`) sitzt im gtsam-Faktorgraph-Zweig von
  `ba_raw`, der **nur bei `imu_enabled=True` (VIO)** läuft. Im VO-Modus nimmt `ba_raw`
  den reinen Torch-`droid_backends.ba`-Pfad — kein Graph, kein GPS. „GPSFactor
  aktivieren" ⟺ „VIO aktivieren".
- **VIO-Init-Gate:** `dbaf_frontend.init_VI()` aktiviert IMU nur, wenn die IMU-Anregung
  `var_g ≥ 0.25` (sonst stiller Visual-Only-Fallback). Livox var_g=0.76 (komfortabel),
  DJI var_g=0.29 (knapp — auf glatten Segmenten fällt DJI evtl. auf Visual-Only zurück).
- VIO initialisiert auf amtown03 sauber (beide IMUs, kein NaN). Format `imu_*.txt` =
  `[t gx gy gz ax ay az]`, gyro deg/s (Code rechnet `/180·π`), accel m/s².

---

## 4. Warum VIO *ohne* GPS scheitert: der Gravity-Leak

**Ursachenkette (im Code verifiziert):**
1. `VisualIMUAlignment` rechnet die Körper-Orientierung aus den visuellen Posen **über
   die Extrinsik**: `wTbs = wTcs · Tbc⁻¹`. Die Cam-IMU-Extrinsik `Tbc` bestimmt also
   direkt, wie „aufrecht" der Körper laut IMU steht.
2. Die Preintegration setzt Gravitation als **−Z, 9.807 m/s²** (`MakeSharedU(GRAVITY)`,
   `multi_sensor.py`).
3. **Falsche `Tbc` → falsche Orientierung → falsche Gravity-Richtung → Schwerkraft wird
   in der falschen Richtung abgezogen → Restgravitation bleibt.**
4. Diese ~g integriert **quadratisch** in die Position.

**Größenordnung passt:** unkompensierte Gravitation über 40 s = `0.5·9.8·40²` = **7848 m**
Drift gegen echten Pfad 433 m = **~18×** ≈ gemessene **26×** (fair-Scale 0.038).

**Belege:** (a) Größenordnung 18× ≈ 26×; (b) Maßstabs-Flip: bei Frame 14 (nach
GNSS-Reskalierung) ist die Bahn 4.5× zu *klein*, bei Frame 400 26× zu *groß* → sie
divergiert während des Laufs; (c) Code-Kette `wTbs = wTcs·Tbc⁻¹`.

**Der Kern:** Bei Aerial-Cruise sind echte Bewegungs-Beschleunigungen winzig (~0.1–1 m/s²),
Gravitation ist 9.8 m/s². Schon **5° Extrinsik-Fehler** lecken `sin(5°)·9.8 = 0.85 m/s²`
→ über 40 s = 684 m. Die Extrinsik muss hier auf **~1° genau** stimmen.

---

## 5. Hartes GPS gewinnt — und Stage-C-Verdrahtung

**Stage C verdrahtet GPS in den DBA-Fusion-GPSFactor:**
- `generic_vo._lla_to_ecef` (WGS84, passt zu `trans.cart2geod`) + `preload_gnss`
  (rtk.csv → ECEF, **interpoliert auf Kamera-Zeit**, weil der Frontend-Sync-Gate nur
  GNSS < 0.01 s vom Frame akzeptiert; rtk ist 5 Hz, cam 10 Hz).
- `run.py` setzt `frontend.all_gnss` + `video.tbg = zeros(3)`.
- `dbaf_frontend.init_GNSS`: `ten0`-Fallback auf erste gnss_position (kein GT-File) +
  Baseline-Fenster **10→30 Frames** (sonst nie >10 m bei Cruise → „Baseline too short").
- Config-Flag: `dataset.gnss_file`. GPSFactor-Gewicht via `frontend.gnss_sigma` /
  `frontend.gnss_robust` (→ `depth_video._gnss_noise()`).

**Ergebnis (400-Frame-Cruise, gleiches Fenster, fair gegen DJI-GT):**

| Run | ATE rmse ↓ | fair-Scale | psnr_ho | GPS-Gewicht |
|---|--:|--:|--:|---|
| VO-Baseline | 12.76 m | 293.6 | 13.61 | — |
| VIO (soft GPS) | 35.73 m | 0.038 | 8.98 | Cauchy + 1 m |
| **VIO + hartes GPS** | **6.14 m** | 0.847 | 11.76 | cm-Sigma, robust aus |
| VIO + extra-hartes GPS | 6.14 m | 0.847 | 11.79 | mm-Sigma, robust aus |

- **Hartes GPS halbiert die ATE gegenüber VO** (6.14 vs 12.76 m) und fixt den Maßstab.
- **Warum soft GPS versagt:** der Original-GPSFactor ist `Cauchy(0.08)` + 1-m-Sigmas;
  bei der 26×-Bahn sehen die GPS-Residuen wie Ausreißer aus → weggedämpft.
- **GPS-Gewicht saturiert ab cm** (hard = xhard, identisch) — kein Hebel mehr.

---

## 6. Die „0.847" ist GT-Verzerrung, nicht VINGS-Fehler

Pfadlänge im Fenster: **RTK 470.5 m**, **DJI-GT 418.6 m** → **DJI/RTK = 0.89**. Die
DJI-Posen sind ~**11 % kürzer** als RTK = der **DJI-`local_position`-10%-Scale-Bug**
(siehe CLAUDE.md). Unsere Bahn ist an RTK geankert (korrekt metrisch), fair_eval misst
aber gegen die verzerrte DJI-GT → die 0.847 spiegelt v.a. die GT-Distortion. Heißt:
die GPS-geankerte Bahn ist **näher an echt-metrisch als 0.847 suggeriert**. Der
VO-vs-VIO-Vergleich bleibt fair (gleiche GT) → VIO+hartGPS 6.14 m ist echt 2× besser.

---

## 7. Extrinsik-Optimierung — konvergiert (Plateau bei Yaw 270°)

Da GPS die Position erledigt, bestimmt die Extrinsik nur noch die **Orientierung** →
die psnr-Lücke zu VO. Mit hartem GPS diskriminiert **nur `psnr_ho`** die Extrinsiken
(ATE/scale sind GPS-gepinnt, über alle Kandidaten identisch 6.16 / 0.847). Run-zu-Run-
Rauschen ~0.2–0.3 psnr → Unterschiede < 0.3 sind nicht signifikant.

**Methode — Gravity-Selbstkalibrierung:** mittlerer Accelerometer-Vektor im Level-Cruise
= Gravitationsrichtung im IMU-Frame → Roll/Pitch direkt; Yaw als freier DOF gesweept.
Mittlerer Livox-Accel im Fenster = `[-9.76, -1.39, -0.32]` → **gravity-down ≈ IMU +X**
(Livox-IMU ist X-down, nicht Z-up). `R_align` = Rotation, die cam-Z (0,0,1, nadir) auf
gravity-down dreht; Kandidat = `R_yaw(θ) @ R_align`.

**Ergebnisse** (alle mit hartem GPS, psnr_ho):

| Kandidat | psnr_ho | Anmerkung |
|---|--:|---|
| pubR (Richtungs-Flip) | 11.17 | → **Rᵀ ist die richtige Richtung** |
| pubRT (publiziert, Rᵀ) | 11.59 | publizierte MARS-LVIG-Kalibrierung |
| grav y0/y90/y180 | FAIL / 10.1 / 10.8 | degeneriert (gnss_init-scale 69–75!) |
| **grav y270** | **12.08** | **bester** (gnss-scale 3.55, sauber) |
| grav y240/y255/y285/y300/y315 | 11.8/11.8/11.8/11.4/11.1 | glatter Peak um 270° |
| y270 + roll/pitch ±5° | 11.8–12.05 | **Plateau** — kein Gewinn |

**Yaw ist der entscheidende DOF.** y270 (gravity-aligned + 270° Yaw) **schlägt die
publizierte Extrinsik** um ~0.5 psnr (12.08 vs 11.59). Roll/Pitch um y270 plateauen bei
~12.0 → konvergiert. Die Gravity-Selbstkalibrierung mit richtigem Yaw ist also leicht
besser als die publizierte cam↔LiDAR-Extrinsik (die IMU≈LiDAR approximiert).

Generatoren: `/tmp/gen_extr.py` (grobe Kandidaten), `/tmp/gen_extr_fine.py` (Yaw-Fein),
`/tmp/gen_extr_rp.py` (Roll/Pitch). Treiber `/tmp/extr_sweep.sh` (sammelt psnr in
`output/extr_sweep_results.csv`, löscht schweren Output). **WICHTIG bei Sweeps:**
`ply_checkpoint_every_kf: 99999` (keine Zwischen-PLYs, sonst Disk voll); schlechte
Extrinsik → größere Map → `rc=137` VRAM-Wand im fair_eval.

---

## 7b. DJI-Posen direkt verwenden — funktioniert, hilft aber nicht

Test: die DJI-Onboard-6-DoF-Posen direkt als Trajektorie einspeisen (statt VO/VIO zu
schätzen), via `dataset.ext_poses_file`. Der bestehende `_apply_ext_poses_to_vizout`
(run.py:617) ersetzt die Mapper-Posen + skaliert die Depths; mit
`seed_video_with_ext_pose: true` werden auch die Tracker-Posen geseedet.

**Ergebnis (gleiches Fenster):**

| Run | ATE | psnr_ho | train-PSNR |
|---|--:|--:|--:|
| VO | 12.76 | 13.61 | 21.96 |
| VIO + hartGPS + y270 | 6.14 | 12.08 | ~14.8 |
| **DJI-Posen direkt** | 0.00* | **6.83** | 11.14 |

\* ATE=0 ist **zirkulär** (est-Posen = GT). Das Scale-Matching konvergierte sauber von
98× (DROID-Depths ~0 am Anfang) auf ~1.0.

**Befund:** Position exakt (ATE=0), aber **Render-Qualität schlecht** (psnr_ho 6.83,
train 11.14 — beide weit unter VO/VIO+GPS). Die Onboard-AHRS-**Orientierung** ist für
Gaussian-Splatting zu verrauscht (braucht ~Sub-Grad; Wind/Vibration/Kompass-Drift).
Die ATE prüft nur Translation → kann die Orientierungs-Ungenauigkeit nicht sehen.
**Held-out-PSNR belohnt Selbst-Konsistenz** (VOs Map+Posen gemeinsam optimiert) über
absolute-aber-verrauschte Genauigkeit (DJI). *Restunsicherheit:* könnte teils ein
Quaternion-Konventions-Effekt im ext_pose-Render-Pfad sein (nicht abschließend geklärt;
train-PSNR 11.14 „schlecht aber nicht null" spricht eher für Rauschen als Verdrehung).

---

## 7c. Eigene DJI-Drohne: 6-DoF-Posen sind direkt loggbar

Die `dji_poses` (Position + Quaternion pro Frame) sind **Onboard-Telemetrie**, nicht
aufwendig berechnet. Mit eigener DJI+RTK-Drohne bekommt man sie direkt:
- **Position:** RTK lat/lon/alt → ENU (cm-genau; `local_position` meiden, ~10%-Scale).
- **Orientierung:** Flightcontroller-AHRS (IMU+Kompass) → Attitude-Quaternion.
- **Gimbal-Winkel** (falls Gimbal-Kamera) — DJI loggt sie, damit Kamera-Orientierung
  rekonstruierbar.
Quelle: DJI-Flightlogs (`.DAT`, via DatCon) oder DJI-SDK-Telemetrie.

**Relevanz:** Eigene Flüge liefern damit beides, was wir hier aus dem fremden Datensatz
zogen — die 6-DoF-GT (fair_eval) UND den RTK-Positions-Anker für den Stage-C-GPSFactor
(der hart gewichtet VIO auf 6.14 m drückte). Man wäre nicht auf MARS-LVIG angewiesen.
Einzige zusätzliche Kalibrierung: die Cam-IMU/Body-Extrinsik (fest bei starrer Kamera;
aus Gimbal-Winkeln bei Gimbal-Kamera).

**Hinweis (bestätigt):** Die `dji_poses`-Orientierung variiert stark (roll ±20°,
pitch ±27°, yaw 310° Spannweite) — die Kamera ist nadir, aber das **rigide Payload kippt
mit der schwankenden Drohne** (Wind/Bewegung). Kein Gimbal → feste Extrinsik (Abschnitt 1).

---

## 7d. Map-Geometrie: DROID-Tiefe strängt, Metric3D macht sie flach

Die 400-Frame-PLYs sahen als „extrem schmaler langer Strang ohne Bilder" aus. Ursache
ist **NICHT** ein Pose-Bild-Mismatch (400 Posen ↔ 400 Bilder, per Frame-Index aligned),
sondern **monokulare Tiefen-Degeneration auf flacher Nadir-Aerial**:

| PLY | Kern-bbox (1-99 %) | Vertikal | Befund |
|---|---|--:|---|
| VO (DROID) | 64 × 7 × 984 m | 984 m | **Strang** — 98.7 % der Gaussians in 0.5 m, Scale-Collapse (293×) |
| Winner VIO+hartGPS (DROID) | 983 × 792 × 1020 m | 1020 m | echter Maßstab, aber Tiefen-Floater |
| DJI-poses (DROID) | 358 × 155 × 513 m | 17728 m voll | verrauschte Orientierung → Extrem-Ausreißer |
| **Winner + use_metric** | **415 × 158 × 29 m** | **29 m** | **flache echte Szene ✓** |

**Mechanismus:** Bei schwacher Parallaxe (flacher Boden, Nadir) kann DROID-BA die Tiefe
nicht festlegen → Gaussians werden **entlang des Sehstrahls** verschmiert (die ~1000 m
in der Tiefenrichtung). Bei VO kommt der Scale-Collapse dazu (Kamera bewegt sich im
SLAM-Frame nur ~0.5 m) → alle Sehstrahlen überlagern sich zu **einem** Strang.

**Wichtigster Befund:** **GPS/VIO fixen die *Trajektorie* (ATE), nicht die *Tiefen-
Degeneration*, die die *Map* ruiniert.** Selbst der Winner (metrisch korrekte Posen)
hat 1020 m Tiefen-Floater. Das ist die strukturelle Schwäche von Mono-Nadir-Aerial.

**Fix:** `use_metric: true` (Metric3D) — per-Bild metrische Tiefe ohne Parallaxe →
Boden flach bei ~80 m. Vertikal-Ausdehnung **1020 m → 29 m**. **Trade-off:** geometrisch
korrekt, aber psnr_ho FÄLLT (9.37 vs 12.08) — Metric3D ist per-Pixel verrauschter, die
self-konsistente DROID-Tiefe rendert schärfer. Also: **`use_metric:true` zum Anschauen /
für echte 3D-Geometrie, `use_metric:false` für die PSNR-Metrik** (zwei verschiedene Ziele;
deckt sich mit der CLAUDE.md-Aerial-Notiz).

**Floater-Filter:** `scripts/filter_ply_floaters.py` (Distanz/Opacity/Größe) — entfernt
die Rest-Ausreisser. Auf der Metric-PLY: 5.8 % raus → bbox 869×617×260 → **521×261×43 m**.

---

## 7e. „Metric + Bundle-Adjustment" — geht, aber die BA kollabiert die Geometrie zurück

Frage: Kann man Metric3D *mit* der BA kombinieren (BA verfeinert photometrisch, Maßstab
bleibt metrisch → flach UND scharf)? Code-Befund: **Metric3D fließt schon in die BA** —
`DBAFusion.track` → `MotionFilter.track` → `video.append(...,depth,...)` → `depth_video.py:186`
`disps_sens[i]=1/depth`, und `droid_backends.ba(poses, disps, intrinsics, disps_sens, ...)`
ankert die BA an diese Tiefe (weicher Prior). ABER der Knopf **`use_metric_for_mapper`
(Default true)** überschreibt den Mapper-Input mit der **rohen** Metric3D-Tiefe statt der
BA-verfeinerten (run.py:669).

**Experiment** (`use_metric:true` + **`use_metric_for_mapper:false`** → Mapper nimmt
BA-verfeinerte Tiefe), Quality-Setup (skip1/iters100):

| Variante | train-PSNR | Geometrie (vertikal, gefiltert) |
|---|--:|---|
| roh-metric (`for_mapper=true`) | 14.62 | **flach (33 m)** ✓ |
| BA-metric (`for_mapper=false`) | **15.67** (schärfste!) | **Strang (1006 m)** ✗ |
| DROID (kein metric) | ~14.8 | Strang |

**Befund:** Die BA-verfeinerte Variante ist die **schärfste überhaupt** (15.67), aber die
Geometrie **kollabiert zurück zum Strang** (1006 m). Der weiche `disps_sens`-Prior wird vom
**photometrischen BA-Ziel überstimmt** — das *will* auf flacher Nadir-Aerial (schwache
Parallaxe) die degenerierte Lösung, weil sie den Bildfehler minimiert. **Das ist die
fundamentale Spannung, jetzt experimentell bewiesen:** Photometrische BA bevorzugt den
Kollaps; höhere PSNR belohnt Selbst-Konsistenz, nicht echte 3D-Geometrie.

**Flach UND scharf gibt es hier nicht gratis.** Wege dazu:
1. **Stärkerer Metric-Anker** — `disps_sens`-Prior-Gewicht im `droid_backends.ba`-C++-Kernel
   hochziehen (nicht per Config tunebar, Rebuild nötig).
2. **Bessere Tiefe/Parallaxe** — LiDAR (MARS-LVIG hat es, nicht heruntergeladen) oder ein
   dynamischeres Segment (mehr Parallaxe → BA kollabiert nicht).

**Best-aussehende PLY** bleibt der **roh-metric Quality-Run**
(`output/winner_metric_quality_filtered.ply`, flach 477×217×33 m, leicht unscharf).
Configs: `winner_metric_quality.yaml` (roh-metric, dicht), `winner_metric_ba.yaml`
(BA-Prior). Qualitäts-Hebel: `mapper_kf_skip:1`, `iters:100`, `num_keyframe:16`.

---

## 8. Code-Änderungen dieser Session (alle mode-gated, VO/Sweep unberührt)

| Datei | Änderung |
|---|---|
| `scripts/datasets/generic_vo.py` | `c2i` aus `dataset.c2i`; `preload_imu()` lädt `dataset.imu_file` (gecached); VIO setzt Frame-tstamp = echte Unix-Zeit; `_lla_to_ecef` + `preload_gnss` (ECEF, interp@cam) |
| `scripts/run.py` | `frontend.all_gnss = dataset.preload_gnss()` + `video.tbg = zeros(3)` (nur wenn gnss vorhanden) |
| `scripts/frontend/dbaf_frontend.py` | `init_GNSS`: `ten0`-Fallback + Baseline-Fenster 30; Diagnose-prints `[vio_spike]` (var_g, imu_enabled, GNSS init) |
| `scripts/frontend/depth_video.py` | `gnss_sigma`/`gnss_robust` aus Config → `_gnss_noise()`; 3 GPSFactor-Noise-Blöcke ersetzt |
| `scripts/eval/fair_eval.py` | `_est_keys_to_slice`: mappt VIO-Unix-tstamps via camstamp auf Slice-Index (sonst ATE für VIO übersprungen); `collect_est_w2c_tq` keyt nach float |
| `scripts/datasets/generic_vo.py` | `ext_poses` nach `start_frame`/`max_frames` slicen (sonst Posen auf Slice um start_frame verschoben) — für DJI-Posen via `dataset.ext_poses_file` |
| `scripts/filter_ply_floaters.py` (NEU) | Floater-Filter für 2DGS-PLYs (Distanz/Opacity/Größe), entfernt Nadir-Aerial-Tiefen-Ausreißer |

---

## 9. Configs + Reproduktion

`configs/local/amtown03/vio_spike/`:
- `amtown03_vio_spike_vo_baseline.yaml` — VO-Baseline
- `amtown03_vio_spike_livox.yaml` / `_dji.yaml` — VIO (Livox/DJI-IMU)
- `amtown03_vio_spike_livox_gps.yaml` — VIO + soft GPS
- `amtown03_vio_spike_livox_gps_hard.yaml` — VIO + hartes GPS (gnss_sigma [0.05,0.05,0.10], robust false)
- `amtown03_vio_spike_livox_gps_xhard.yaml` — mm-Sigma
- `extr_sweep/extr_*.yaml` — Extrinsik-Kandidaten (Generator `/tmp/gen_extr.py`)

Run: `python scripts/run_experiment.py <config>` (vings-env, positional arg).
Alle nutzen `start_frame=2916, max_frames=400` (Livox-Fenster), `fair_eval.enabled=true`,
GT `dji_poses_all_w2c.txt`, eval_stride 20.

---

## 10. Offene Punkte / nächste Hebel

1. **Extrinsik:** echten Avia-IMU↔LiDAR-Offset besorgen (statt IMU≈LiDAR); Gravity-
   Selbstkalibrierung verfeinern; auf dynamischerem Segment testen (besseres
   Signal-zu-Gravity-Verhältnis).
2. **Orientierung → psnr:** das ist die Restlücke zu VO; hängt an der Extrinsik.
3. **Volle Sequenz:** der VIO+hartGPS-Vorteil wurde nur auf 400 Frames gezeigt — der
   eigentliche Drift-Test wäre die ganze Sequenz (wofür full_6199f gedacht war).
4. **DJI-GT ist verzerrt** (~11 %); für saubere absolute Zahlen wäre eine RTK-basierte
   GT besser, aber RTK ist der GPS-Anker → zirkulär. DJI-GT bleibt als unabhängige
   (wenn auch verzerrte) Referenz für relative Vergleiche brauchbar.
