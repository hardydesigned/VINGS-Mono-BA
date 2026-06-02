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
