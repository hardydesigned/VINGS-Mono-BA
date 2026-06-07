# interval1_AMtown03 — kompletter LiDAR-Survey (driftfreie PLY)

## Worum geht's (in einfach)

Ziel: aus dem kompletten ~4,9-km-Drohnenflug (UAVScenes `interval1_AMtown03`)
**eine durchgehende, scharfe, nicht-weggedriftete 3D-Szene** als PLY bauen.

Mit dem normalen VINGS-Pfad (`mode:vo`, Metric3D-Tiefe) ging das **nicht**:
- Tracking driftet über 4,9 km komplett weg (ATE 219 m).
- Metric3D gibt auf flacher Nadir-Luftaufnahme unscharfe, schmale Tiefenstreifen
  mit Lücken → Floater und Löcher in der Szene.
- Der volle Lauf am Stück sprengt den VRAM (rc=137, Peak ~9,7 GB) bei ~Frame 600.

Die Lösung sind vier Bausteine. Kurz: **GT-Posen** fürs Tracking, **LiDAR-Tiefe**
für Skala+Abdeckung+Schärfe, **Chunking+Merge** für den VRAM. Weil die Posen
global-metrisch sind, passen die Chunks beim Zusammenfügen exakt aufeinander.

Endergebnis: `output/exp_interval1_lidarchunks/survey_lidar_complete.ply`
(9,12 M Gaussians, 1171 × 883 m, driftfrei). Echter Render-Check:
`survey_lidar_complete_views.png` (Nadir-Renders zeigen Bäume, Felder, Gebäude,
asphaltierte Flächen — siehe „Render-Check" unten).

---

## Die vier Bausteine

### 1. Tracking driftfrei → GT-Posen statt VO

UAVScenes liefert Ground-Truth-Posen mit. Konvertierung ins VINGS-Format:

```bash
python scripts/prepare_uavscenes_interval.py
# UAVScenes T4x4 (cam-to-world) -> poses_w2c.txt (TUM w2c),
#   camstamp.txt, intrinsic.txt  -> interval1_AMtown03/vings/
```

In der Config:
```yaml
dataset:
  ext_poses_file: .../interval1_AMtown03/vings/poses_w2c.txt
```
Die GT-Posen überschreiben im Mapper-Pfad die getrackten Posen
(`_apply_ext_poses_to_vizout`, `run.py`). → ATE **0,07 m** statt 219 m.

### 2. Tiefe/Skala/Abdeckung → LiDAR-Tiefe statt Metric3D

DROID kann auf Nadir-Luftbild keine Skala etablieren (Floater); Metric3D ist
unscharf und gibt nur schmale Streifen. Stattdessen die echte Livox-Tiefe:

- `scripts/datasets/generic_vo.py::_lidar_depth()` — projiziert die LiDAR-Punkte
  ins Bild (LiDAR-Frame: **X = Tiefe**, Y/Z = Bildebene; `depth = lx`,
  `u = fu·ly/lx + cu`, `v = fv·lz/lx + cv`), z-buffert (fern zuerst, nah
  überschreibt) und füllt Lücken per `distance_transform_edt` (Nearest-Neighbor)
  → dichte Tiefenkarte in Bildauflösung.
- `scripts/run.py` lädt das **Metric3D-Modell nicht**, wenn `dataset.lidar_dir`
  gesetzt ist (spart ~1,5 GB GPU); Loader-Tiefe ersetzt die Vorhersage am
  Injektionspunkt (`if 'depth' not in data_packet … None`).

Config:
```yaml
dataset:
  lidar_dir: .../interval1_AMtown03/interval1_LIDAR
  lidar_sign_u: -1.0          # Bildachsen-Vorzeichen (Kalibrierung)
  lidar_sign_v: -1.0
use_metric: true              # bleibt true, wird aber durch LiDAR-Tiefe bypassed
```
→ Voller FOV (~250 m breite Streifen), scharf, flach (z-Spread ~23 m statt
verschmiert). GPU pro Chunk nur ~1,8 GB.

### 3. VRAM-Wand → Chunking in GT-Frame

Voller Lauf (5300 Frames) crasht am VRAM-Watchdog. Lösung: in Chunks aufteilen.
Weil **alle Posen im selben metrischen Welt-Frame** liegen, sind die Teil-PLYs
**trivial mergebar** (kein Re-Alignment).

```bash
bash scripts/run_interval1_lidar_chunks.sh
# 9 Chunks à 600 Frames (skip 8), Starts 300 900 1500 … 5100
# (Hover 0–300 übersprungen), seriell, je eigener Run.
```

### 4. Merge

```bash
python scripts/eval/merge_plys.py \
  --out survey_lidar_complete.ply --opacity-min 0.2 <9 chunk-plys>
```
Konkateniert (gleicher Frame) + Opacity-Filter gegen Floater.
→ **9,12 M Gaussians, kompletter Survey, lückenlos.**

---

## Render-Check (sieht man wirklich was?)

`scripts/eval/render_ply_check.py` macht nur einen **matplotlib-Scatter der
Gaussian-Zentren** → grauer Nebel, egal wie gut die Map ist. Das ist KEIN
Qualitätsurteil über die PLY.

Für einen echten Eindruck: `scripts/eval/render_ply_views.py` rastert die PLY
mit dem **echten diff_surfel-Rasterizer** aus den Nadir-Flugposen (pro Ansicht
spatialer Crop ~250 m Umkreis gegen OOM):

```bash
python scripts/eval/render_ply_views.py \
  output/exp_interval1_lidarchunks/survey_lidar_complete.ply \
  --frames 600,1500,2700,3900,4500,5100 --res 480 576
# -> survey_lidar_complete_views.png
```
Ergebnis: erkennbare Bäume, Felder, Gebäude, asphaltierte Flächen — natürliche
Farben, scharfe Texturen. Alternativ die PLY direkt in SuperSplat (Browser),
MeshLab oder CloudCompare öffnen (Standard-2DGS-PLY).

---

## Dateien dieser Pipeline

| Datei | Rolle |
|---|---|
| `scripts/prepare_uavscenes_interval.py` | UAVScenes T4x4 → VINGS-Posen/camstamp/intrinsic |
| `scripts/datasets/generic_vo.py` (`_lidar_depth`) | LiDAR→dichte Tiefenkarte |
| `scripts/run.py` (Metric3D-Skip bei `lidar_dir`) | spart GPU, Loader-Tiefe-Bypass |
| `configs/local/interval1/interval1_lidar_full.yaml` | Basis-Config (GT-Posen + LiDAR) |
| `scripts/run_interval1_lidar_chunks.sh` | 9 Chunks + Merge |
| `scripts/eval/merge_plys.py` | Chunk-PLYs zusammenführen |
| `scripts/eval/render_ply_views.py` | echter Nadir-Render-Check |
| **`output/exp_interval1_lidarchunks/survey_lidar_complete.ply`** | **Endergebnis** |

## Ohne LiDAR: geschätzte Tiefe + Posen-Stütze (2026-06-02)

Frage: klappt der Survey auch **ohne LiDAR**, nur mit geschätzter Tiefe, gestützt
von GPS/Posen? Antwort: **Ja — aber nur mit DROID-BA-Tiefe, nicht mit Metric3D.**

Einzel-Chunk-Vergleich (Frame 2700–3300, identische GT-Posen-Stütze):

| Tiefe | PSNR | SSIM | Render |
|---|---|---|---|
| **LiDAR** (Referenz) | — | — | kristallklar |
| **DROID-BA + Posen-Skala** (`use_metric:false`) | 14.0 | 0.48 | scharf — Gebäude/Bäume/Plantage erkennbar |
| Metric3D monokular (`use_metric:true`, kein LiDAR) | 12.2 | 0.17 | matschig, radiale Schlieren |

**Warum DROID schlägt Metric3D:** monokulare metrische Tiefe (Metric3D) ist auf
flachem Nadir skalen-inkonsistent über das Bild → Gaussians smearen radial. Die
DROID-BA-Tiefe kommt aus **Multi-View-Triangulation** (geometrisch konsistent),
ihr fehlt nur der absolute Maßstab — und **den liefern die Posen/GPS**:
`_apply_ext_poses_to_vizout` skaliert `viz_out['depths'] *= sum_rtk/sum_droid`
(kumulatives Pfadlängen-Verhältnis RTK/DROID). Genau hier stützt GPS die Tiefe.
Deckt sich mit CLAUDE.md-Aerial-Befund #1 (DROID-Tiefe > Metric3D bei Nadir).

Voller Survey (9 Chunks, gleiche Mechanik):
```bash
bash scripts/run_interval1_droid_chunks.sh
# Base configs/local/interval1/interval1_droid_full.yaml (use_metric:false)
# -> output/exp_interval1_droidchunks/survey_droid_complete.ply (12.0M Gaussians)
```
Ergebnis: kompletter Flug abgedeckt, scharfe Strukturen über alle Beine; nur die
Post-Hover-Startregion (Frame ~600, c300-Chunk nur idx=400) und das Flugende
(~5100) sind weicher. Render-Check identisch via `render_ply_views.py`.

Damit generalisiert das Chunking auf realistische Setups **ohne LiDAR**: DROID-BA
liefert die Geometrie, GPS/Posen den Maßstab und die Driftfreiheit.

## PSNR erhöhen: Auflösung + Train-Iterationen (2026-06-02)

Frage: bringt mehr **Auflösung** und mehr **Train-Iterationen** pro Chunk PSNR?
Antwort: **Ja — aber Auflösung ist VRAM-gebunden, nicht qualitätsgebunden.**

Einzel-Chunk Frame 2700 (DROID-Tiefe, gleiche Posen-Stütze):

| Auflösung | iters | Frames | PSNR | Peak-GPU | Status |
|---|---|---|---|---|---|
| 240×288 | 50 | 600 | 14.03 | ~1.5 GB | OK |
| 384×456 | 100 | 300 | 14.97 | ~2.5 GB | OK |
| 480×576 | 100 | 600 | 18.7 (KF 1) | 9584 MB | **rc=137 OOM bei KF 1** |

Erkenntnisse:
- **Höhere Auflösung hebt PSNR klar** (erste-KF-PSNR springt von ~14 auf 18.7 bei
  480×576). Mehr Pixel → mehr Gaussians → feinere Textur.
- **Der Engpass ist VRAM, und er skaliert mit der akkumulierten KF-Zahl im Chunk**
  (Storage-Manager hält ein GPU-Working-Set). 480×576 = 4× Pixel/KF → bei 600
  Frames sofort OOM, bei 384×456/300 nur 2.5 GB. **Also: Auflösung hoch ⇒
  Chunk-Länge runter.** Bei 384/300 ist noch viel Luft (2.5/9.6 GB) → 480×576
  passt mit kurzen Chunks (~200–300 Frames).
- Train-`iters` 50→100 ist billig (kostet kaum VRAM) und hilft der Konvergenz.
- Praktische Rezept-Achse: **(image_size, chunk_len)** gemeinsam wählen, damit
  Peak-GPU < ~8 GB bleibt. Mehr, kürzere Chunks mergen genauso trivial (GT-Frame).

## Floater wegfiltern (2026-06-02)

Beobachtung: tolle Boden-Reko, aber viele **weit entfernte Floater-Gaussians**.
Der Merge-`opacity-min 0.2` fängt sie NICHT — Floater haben **hohe Opacity**
(Median 0.98). Diagnose auf dem 12M-Survey:

- **x/y/z reichen bis ±inf**, p99(x)=26151 m (Median −166 m) — Extrem-Ausreißer.
- **Scale ist der echte Diskriminator:** gute Bodengaussians `exp(scale).max`
  Median 0.52 m, aber p99 = 640 m, max = 18 km. **6.2 % haben scale > 2 m** —
  das sind die Floater-Riesen.
- Boden bei z ≈ −64 m; `|z − med| ≤ 60 m` behält 94 %, `≤ 30 m` 69 %.

→ Die wirksamen Filter sind **Endlich-Koords + Scale-Max + Spatial-Crop + z-Band**,
NICHT Opacity. `scripts/eval/clean_ply.py` erweitert:

```bash
python scripts/eval/clean_ply.py survey_droid_complete.ply \
  --gt-poses .../vings/poses_w2c.txt \
  --max-dist 120 --max-scale 2.0 --opacity-min 0.5 --max-z-spread 60 \
  --out survey_droid_cleaned.ply
```
Filter-Kaskade (Survey-Beispiel, 12.0M → 10.3M, 85.8 % behalten):
1. **finite** (inf/nan-Koords raus) 12.01M → 11.82M
2. **--max-scale 2.0** (Riesen-Gaussians) → 11.26M
3. **--opacity-min 0.5** → 11.08M
4. **--gt-poses + --max-dist 120** (Spatial-Crop zur Flugbahn) → 10.91M
5. **--max-z-spread 60** (vertikale Floater) → 10.31M

Neue Flags: `--max-scale` (größte Gaussian-Achse in m), `--gt-poses` (TUM-w2c als
Crop-Bahn, rechnet Kamerazentren `c = −Rᵀt`), plus impliziter Endlich-Filter.
Ergebnis: `output/exp_interval1_droidchunks/survey_droid_cleaned.ply`. Boden bleibt
voll erhalten, Floater weg. Knöpfe nach Geschmack: `--max-scale 1.0` aggressiver,
`--max-z-spread 30` enger (bei flachem Gelände), `--max-dist` an Flughöhe koppeln
(~1.5× AGL deckt den Nadir-Swath).

## Option C: scharfe non-metrische Chunks + Sim3 — und das Naht-Problem (2026-06-03)

**Einfach gesagt:** Variante A (metrisch, GT-Posen + ×82-Tiefenskala) ist *nahtlos
aber weich* — alle Frames sitzen im selben GT-Welt-Frame, also passen die Chunks
zusammen, aber die ×82 hochskalierte verrauschte DROID-Tiefe verwischt das Bild
(PSNR ~14–18). Variante **C** rekonstruiert jeden Chunk **non-metrisch** im reinen
DROID-Frame (scharf, PSNR ~21–23) und schiebt ihn *danach* per Sim3 (Umeyama,
DROID-Kameras → GT-Kameras) ins metrische Welt-Frame. Problem: jeder Chunk wird
**unabhängig** ausgerichtet → die Chunks passen nicht zusammen (sichtbare Nähte,
Höhensprünge).

**Warum die Naht entsteht (Diagnose an c3000/c3150/c3300, Frames 3000–3450):**

1. **Skala ist echt pro Chunk verschieden** (c3000/c3150 ≈ 82, c3300 = 115). Jeder
   Chunk ist ein eigener DROID-Lauf mit eigener monokularer Gauge. Das ist *korrekt*
   — der per-Chunk-Sim3 skaliert jeden eigenständig. RANSAC oder eine erzwungene
   Global-Skala *verschlimmern* es (fixed-scale 83.5 auf c3300 → RMSE 16.8 m).
2. **Tiefen-Floater dominieren.** Der globale `clean_ply --max-dist` greift nicht:
   über die 5-km-Bahn ist *immer* irgendeine GT-Kamera < max-dist entfernt, also
   überlebt jeder Floater. Floater müssen pro Chunk gegen die **~150 eigenen**
   Kameras gecroppt werden.
3. **Nadir-Tiefen-Müll.** c3150 hatte **61 %** seiner Gaussians *über* der Drohne
   (z bis +44 m bei Kamera-z 0.5) — physikalisch unmöglich für eine nach unten
   schauende Kamera. Vor dem Filter zeigte das eine AGL (Flughöhe über Grund) von
   nur 3.6 m statt ~40 m; nach dem Nadir-Filter 30 m.
4. **Degenerierte Chunks** (c3300: scale 115, RMSE 3.44 m, AGL 58 m statt 40)
   poisonen den Merge. Sie sind an Scale/RMSE/AGL erkennbar und müssen raus.
5. **Rest-Kippung (ungelöst, fundamental).** Eine fast geradlinige Nadir-Bahn
   constrained den Sim3-Roll/Tilt schlecht (Position-Umeyama: ~1° Rest-Tilt × 80 m
   Tiefe → Meter-Höhenfehler, der mit Abstand zur Flugachse wächst). Eine
   Orientierungs-basierte Korrektur scheiterte: zwischen DROID-Cam- und GT-Cam-Frame
   liegt ein **inkonsistenter** Konvent-Offset (R_conv der zwei Chunks differieren
   um 137°), und Roll ist aus near-straight Nadir-Posen schlicht nicht stabil
   schätzbar. Echte Nahtfreiheit braucht ein **gemeinsames Pose-Graph-/BA über alle
   Chunk-Sim3** (= Code, kein Config-Hebel).

**Was der Fix kann (`scripts/eval/chunk_postfix.py`):** Footprint-Crop (1) +
Nadir-Filter (3) + Quality-Gate auf Scale/RMSE/AGL (2,4). Damit verschwinden
Floater, Müll-Tiefen und Gift-Chunks; die zwei guten Chunks stoßen am Übergang
sauber aneinander (beide ~−25…−35 m an der Naht). Übrig bleibt nur die
Rest-Kippung (5). Vorher/Nachher: `output/exp_interval1_optC/seam_bev.png`
(roh, Floater bis −20000) vs. `seam_nadir.png` (bereinigt).

```bash
# Orchestrator (überlappende Chunks, damit verworfene keine Lücke lassen):
CHUNK=150 STEP=100 bash scripts/run_interval1_optC.sh        # voller Survey
bash scripts/run_interval1_optC.sh 3000 3150 3300            # nur 3 Chunks (Test)
# Gate-Knöpfe per env: MEDSCALE=82 MAXRMSE=2.5 AGL=40 AGLTOL=0.4 CROPR=90 NADIR=5
```

`chunk_postfix.py` faltet Sim3-Transform + Crop + Gate in einen Schritt
(`--transform` für rohe DROID-PLY). Exit-Code 2 = Chunk vom Gate verworfen.

### Update 2026-06-03 (nachmittags): Chaining + VRAM-Taming

Zwei Fortschritte machen Option C deutlich tragfähiger:

1. **Sequentielles Chaining gegen die Rest-Kippung** (`scripts/eval/chain_chunks.py`).
   Chunk 0 = Anker an GT; jeder folgende Chunk holt seine **Rotation aus den
   Kamera-Orientierungen am Overlap** (`R = mean_f[ welt_orient_vorg(f) ∘ Rd_i(f)ᵀ ]`)
   statt aus Positionen — letztere sind auf near-straight Nadir-Bahnen roll-degeneriert
   (Position-Umeyama lieferte Δrot=171° = fast Flip). Orientierungs-basiert: well-conditioned,
   korrigierte real **20° relative Rotation** zwischen c3000/c3075-Gauges + 33 % mehr
   xy-Overlap. Scale/Translation bei fixem R aus Overlap(an Vorgänger)+GT(Rest), kein Drift.
   **Rest:** ~7 m Höhen-Disagreement in Overlaps bleibt (Chunks haben echte
   Tiefen-Skala 82 vs 70, AGL 40 vs 36 m) — im Top-Down-Nadir-View aber kaum sichtbar.
2. **VRAM-Reliabilität: `iters 150 + num_keyframe 4`** statt 200/8. Der
   Densifikations-/Raster-Spike riss mit numkf8 ~50 % der Chunks über den 8-GB-Watchdog
   (Peak 8.7–9.8 GB, rc=137); mit numkf4 Peak **3.3 GB** und **PSNR 23.6** (sogar besser).
   `accum_thresh` war schon 0.98, der Spike kam von zu vielen gleichzeitigen KF-Views.

Pipeline jetzt komplett im Orchestrator: Run → `chunk_postfix` (Crop+Nadir+Gate+Sim3) →
`chain_chunks` (Rotations-Chaining) → Merge → Clean.

**Fazit / Empfehlung:** Variante A (`survey_droid_complete.ply`, weich aber nahtlos)
bleibt die sichere Wahl für eine garantiert nahtlose Szene. Variante C ist jetzt aber
ein ernsthafter Kandidat für **scharf + weitgehend nahtlos**: Gate wirft Gift-Chunks,
Chaining fixt Rotation/xy, VRAM ist gezähmt. Der harte Rest (~7 m Höhe in Overlaps)
bräuchte ein volles gemeinsames Sim3-BA über alle Chunks (Tiefen+Posen gekoppelt).

### Update 2026-06-03 (abends): Variante D — durchgehender Lauf + Unwarp (löst alles)

**Die per-Chunk-Sim3 war der ganze Fehler.** Separate DROID-Läufe = unabhängige
Gauges = Nähte. Stattdessen (User-Idee): **EIN durchgehender non-metrischer Lauf**
(Tracker resettet nie) → die Map lebt in **einem** Frame → **nahtlos by construction**
*und* scharf (Posen+Tiefen aus derselben BA, keine ×82-Skalierung). VRAM hält der
**Storage-Manager** (`distance_threshold: 0.1`, nicht 3.0; `use_storage_manager: true`):
600 Frames → Peak **6.0 GB**, PSNR **22.8**, 557 KFs, eine zusammenhängende Stadt.

Dann **metrisch + driftfrei** per *einem* Lauf von **örtlich variierendem Sim3**
(`scripts/eval/sim3_unwarp.py`): globale Rotation überall fix (lokale near-straight
Fenster sind roll-degeneriert — local-R machte RMSE 24→57 m schlechter), nur lokale
**Scale+Translation** pro Trajektorie-Fenster (window 80), verankert an den GT/GPS-
Kamerazentren → Kamera-RMSE **23.7 m → 1.8 m**. Die DROID-Drift (glatte Verbiegung,
keine Naht) wird so glattgezogen, ohne Re-Run.

```bash
# 1. Durchgehender Lauf (reiner DROID-Frame, Storage-Manager):
python scripts/eval/gen_opt_cfg.py cfg.yaml --savedir out/ --start 3000 --frames 600 \
   --hw 240 288 --iters 150 --numkf 4 --kfskip 1 --prune-op 0.5 --dist-thresh 0.1 --no-ext
python scripts/run_experiment.py cfg.yaml
# 2. Örtlich variierender Sim3 -> metrisch + driftfrei:
python scripts/eval/sim3_unwarp.py RUNDIR/ply/idx=N_2dgs.ply \
   --droid-poses RUNDIR/tracker_raw_c2w.txt --gt-poses .../poses_w2c.txt \
   --out survey.ply --window 80 --knn 4 --crop-radius 100 --nadir-clear 5
```

### Update 2026-06-04: Voll-Survey — Segment-Limit + Overlap-Entkippung

**VRAM-Hartgrenze des durchgehenden Laufs:** Die GPU wächst mit der GESAMT-Map-Größe
(Storage-Manager lagert auf CPU aus, aber das GPU-residente Aktiv-Set wächst). Stirbt
rc=137 ~9.7 GB bei ~500–750 KFs, egal numkf(2/4)/dist(0.05/0.1)/kfskip(1/2). Todespunkt
regionsabhängig: GPS-Höhenprofil zeigt dichte Regionen sind NUR Takeoff(f0–1000) +
Landung(f5000+); der Cruise (f1000–5000, 80% des Flugs, 12 m/s, 80 m) ist dünn. Ein
gestorbener Lauf speichert trotzdem seine letzte periodische ply (nutzbare Teil-Map).

**Voller Survey braucht also mehrere Segmente** (je ~500 Frames, überlappend). Aber:
unabhängige DROID-Läufe haben verschiedene monokulare TIEFEN-Skala → GPS verankert die
KAMERAS (RMSE 1.7–2.3 m) aber NICHT die Karten-Tiefe → relative Boden-Kippung ~21 m
zwischen Segmenten (`corr(dz, Flugrichtung)≈−0.85` = planar). **Fix (GPS-Höhe-Idee):**
Im Overlap sehen beide denselben Boden; eine Ebene durch `dz(x,y)` entkippt das spätere
Segment ans frühere (`scripts/eval/detilt_chain.py`). **Naht 21 m → ~4 m.** Top-down eh
unsichtbar (z-Buffer), jetzt auch in 3D stimmig.

**Voll-Survey-Pipeline:**
```bash
# pro Segment (start 1000,1400,...,5000 step 400, je 500 Frames, Cruise):
gen_opt_cfg --no-ext --dist-thresh 0.1 --numkf 4 --iters 150 --kfskip 1 --start S --frames 500
run_experiment ; sim3_unwarp.py PLY --gps-csv rtk_positions_raw.csv --out segN_gps.ply
# dann über alle Segmente (Flugreihenfolge):
detilt_chain.py seg0_gps.ply seg1_gps.ply ... --out survey.ply
```
`sim3_unwarp --gps-csv` nutzt easting/northing/alt mit FIXEM globalem Ursprung → alle
Segmente im selben UTM-Frame. Demo (2 Segmente): `cruise2seg_level.ply`.

Ergebnis 600f: `output/exp_interval1_optC/cont600_unwarp.ply` (nahtlos+scharf+driftfrei+
metrisch). **Für mono+IMU+GPS-Deployment:** GT-Posen in `sim3_unwarp` durch GPS
(`rtk_positions_raw.csv`) ersetzen — gleiche Pipeline. **Verifiziert 2026-06-03:** GPS-Anker
liefert RMSE **1.75 m** (GT: 1.82 m) — identisch. `rtk_positions_raw.csv` hat schon
`easting,northing,alt` (UTM, metrisch); auf KF-Zeiten interpolieren, Ursprung = erste
GPS-Position, als Zentren in eine TUM-Datei (`tr=-center, quat=I` → `Cg=center`).
Ergebnis `cont600_gps_unwarp.ply` — komplette Lösung OHNE LiDAR/GT (nur mono-Video für
die Map, GPS für den metrischen Anker). Die per-Chunk-Skripte
(`chunk_postfix`/`chain_chunks`/`run_interval1_optC.sh`) sind damit für den Voll-Survey
überholt. Offen: durchgehender Lauf über alle 5599 Frames (Storage-Manager VRAM ok,
aber CPU-RAM-Watchdog <1.5 GB beobachten, da ausgelagerte Gaussians wachsen).

### Update 2026-06-04 (II): Lücken gefüllt + GPS-Boden-Leveling ersetzt detilt_chain

Zwei Verbesserungen am mono+GPS-Voll-Survey (`output/exp_interval1_survey/`):

**1. Gestorbene Segmente nachgefüllt.** Drei Segmente (Start 1000/1600/4600) waren
beim ersten Voll-Survey am VRAM-Spike gestorben — und zwar VOR dem ersten
ply-Checkpoint (`ply_checkpoint_every_kf: 50`, KF<50) → leeres ply-dir, echte
Lücken. Ränder (Takeoff f1000, Landung f4600) erzeugen einen KF pro Frame →
Densification-Spike sofort. Fix `scripts/run_interval1_gaps.sh`:
- `num_keyframe 4→2` (halbiert gleichzeitig optimierte KF-Views = kleinerer Spike;
  dominanter VRAM-Hebel),
- `ply_checkpoint_every_kf 20` (selbst ein früh sterbender Lauf hinterlässt eine
  brauchbare Teil-ply mit Nachbar-Overlap),
- Retry-Schleife (der rc=137-Tod ist stochastisch; s1000 starb Versuch 1, überlebte
  Versuch 2 voll). → alle drei Lücken gefüllt.

**2. `detilt_gps.py` (GPS-Boden-Leveling) statt `detilt_chain.py` (sequentielles Ketten).**
*Diagnose:* `sim3_unwarp` verankert die **Kameras** perfekt an GPS — die Kamerahöhe
`cam_alt` im PLY-Frame ist über ALLE Segmente konstant ~80 m (Cruise). Aber die
monokulare **Tiefen**-Skala je Segment löst falsch auf, also landet der **Boden**
pro Segment auf einer ganz anderen Höhe (gemessen −13 .. +72 m statt konstant). Das
— nicht das Warpen — ist die Quelle der „großen Unterschiede". `detilt_chain`
(Ketten an den jeweils vorigen) akkumulierte Fehler → 9–14 m Restnaht an seg5/6/7.

*Lösung (User-Idee):* Boden je Segment robust über den **dichtesten Gaussian-z-Layer**
schätzen (Histogramm-Mode pro grober (x,y)-Zelle → die dünn besetzten Häuser/hohen
Strukturen fallen raus, robust gegen die ~50 m vertikale Tiefen-Schmiere), dann ALLE
Segmente auf **eine gemeinsame GPS-Referenzebene** leveln (`z += ref − g_seg`).
Koplanar, kein Akkumulieren, **braucht keinen Overlap** (löst auch Lücken-/Boundary-
Segmente). Die Streuung der Zell-Böden (MAD) ist die Reliabilität: Schmier-Segmente
(s1900/s2200/s2800/s3400, MAD 14–21 m) bekommen nur einen Shift, keinen Tilt aus dem
Müll-Boden. `--detilt` (zusätzl. horizontaler Tilt der zuverlässigen Segmente) bringt
fast nichts → shift-only nutzen.

```bash
python scripts/eval/detilt_gps.py output/exp_interval1_survey/s*_gps.ply \
   --out survey_raw.ply --cell 25 --clip-scale 2.0 --zband 120
# Spatial-Crop: GT-Posen liegen in ANDEREM Frame als die GPS-Survey -> gegen eine
# aus rtk-csv gebaute GPS-Bahn croppen (gps_traj_w2c.txt, quat=I, tr=-center), NICHT
# gegen poses_w2c.txt (sonst werden ~60% fälschlich weggeschnitten).
python scripts/eval/clean_ply.py survey_raw.ply --gt-poses gps_traj_w2c.txt \
   --max-dist 160 --max-scale 1.5 --opacity-min 0.4 --max-z-spread 60 --out survey_complete.ply
```

*Ergebnis:* Boden-within-±5m **12% → 35%**, vertikaler z-Spread p90 **134 → 30 m**
(`compare_level.png`). Deliverable `survey_gpslevel_clean.ply` (18.1M Gaussians, 13
Segmente inkl. Lücken). Beides im Orchestrator `scripts/run_interval1_survey.sh`
verdrahtet. **Restliche vertikale Dicke** = segment-interne Schmiere des 240×288-
DROID-BA-Tiefenrauschens (Rekonstruktionslimit; Leveling kann das nicht fixen — nur
die Schmier-Segmente bei höherer Auflösung neu fahren). Für echtes Gelände/Gebirge
(nicht-flacher Boden) wäre die Flachheits-Annahme zu hinterfragen; hier (flache Stadt,
konstante Cruise-Höhe) ist sie korrekt.

## Offene Hebel (optional)

- **Voll-Survey in hoher Auflösung**: 480×576 + iters 100 mit kurzen Chunks
  (~200–250 Frames, damit Peak-GPU < 8 GB) über den ganzen Flug → schärfste
  Variante. Mehr Chunks, längere Laufzeit, aber VRAM-sicher und trivial mergebar.
- **GPS-only statt GT-Posen**: die ext_poses durch die verrauschten RTK/GPS-Posen
  aus `AMtown03.bag` ersetzen (echtes Deployment ohne Ground-Truth) — ehrlichste
  Validierung der Posen-Stütze.
- **Start/Ende glätten**: überlappende Chunks (z.B. 200-Frame-Overlap) gegen die
  weicheren Post-Hover-Start- (~600) und Flugende-Regionen (~5100).
- **Floater-Filter feinjustieren**: `--max-scale 1.0`, `--max-z-spread 30`,
  `--max-dist` an Flughöhe koppeln (siehe Floater-Abschnitt).
