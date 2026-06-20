# Objekterkennung und Segmentierung in VINGS-Mono-BA

*Quellmaterial-Report für das Bachelorarbeits-Kapitel. Kombiniert ausformulierten
Fließtext mit kompakten Fakten- und Referenztabellen. Alle Code-Verweise als
`Datei:Zeile`; alle Zahlen gegen den Quellcode verifiziert (Stand 2026-06-20).*

---

## 1. Einleitung und Motivation

VINGS-Mono ist ein Online-Monokular-SLAM-System, das parallel zum Tracking eine
3D-Szene aus 2D-Gaussian-Splats rekonstruiert. Der Datenfluss ist zweigeteilt: Ein
**Tracker** (DROID-basiertes Bundle-Adjustment, `submodules/dbaf`) schätzt für jeden
Frame die Kamerapose, ein **Mapper** (`scripts/gaussian/`) trainiert daraus auf
ausgewählten Keyframes die Gaussian-Map. Auf diese Pipeline setzen zwei
weiterführende, fachlich orthogonale Module auf, die im Rahmen dieser Arbeit
hinzugefügt wurden:

- **Objekterkennung mit Online-3D-Lokalisierung** beantwortet die Frage *„Was steht
  in der Szene und wo?"*. Pro Keyframe werden 2D-Bounding-Boxes detektiert, über die
  Tracker-Tiefe in die rekonstruierte Welt zurückprojiziert und über die Zeit zu
  stabilen 3D-Objekt-Tracks fusioniert.

- **Segmentierung zur Dynamik-Maskierung** beantwortet die Frage *„Was darf nicht
  gemappt werden?"*. Bewegte Objekte (fahrende Autos, Fußgänger) verletzen die
  Statik-Annahme der Gaussian-Map und erzeugen Geister-Artefakte. Das Modul
  segmentiert jeden Keyframe, erkennt die bewegten Segmente und schließt deren Pixel
  aus dem Mapping-Loss aus.

Beide Module teilen drei Designprinzipien, die das gesamte Kapitel durchziehen:

1. **Backend-Agnostik über Registry-Factories.** Sowohl der Detektor (YOLO ↔ RT-DETR)
   als auch das Segmentierungs-Backend (FastSAM ↔ SAM2 ↔ SAM3) sind über eine
   `@register_*`-Factory austauschbar — ein Wort in der Config, kein Code-Umbau. Dieses
   Muster ist bewusst von den Frame-Selektoren des Projekts (`selector_factory.py`)
   übernommen.
2. **Best-effort, bricht den Run nie.** Beide Module sind optional (Master-Gates
   `detect_objects` bzw. `use_dynamic`) und in `try/except` gekapselt; ein Fehler
   führt zu einem `print` und wird übersprungen, statt den SLAM-Lauf zu beenden.
3. **VRAM-Disziplin.** Schwere Inferenz läuft genau einmal pro Keyframe, Ergebnisse
   werden auf die CPU zurückgegeben, und die Modelle sind klein/lazy-geladen — relevant
   für die 8-GB-VRAM-Decke, an der das Projekt arbeitet.

| Modul | Hauptdoku | Master-Gate | Kern-Code |
|---|---|---|---|
| Objekterkennung | `docs/OBJECT_DETECTION.md` | `detect_objects: true` | `scripts/vings_utils/object_tracker.py`, `*_detector.py` |
| Dynamik-Segmentierung | `docs/SEGMENTATION_BACKEND.md` | `use_dynamic: true` | `scripts/dynamic/dynamic_utils.py` |

---

## 2. Objekterkennung und Online-3D-Lokalisierung

### 2.1 Die Pipeline

Die Verarbeitung eines Keyframes durchläuft fünf Stufen — von der 2D-Box bis zum
fusionierten 3D-Objekt:

```
RGB-Keyframe (Full-Res-Originaldatei)
  → detector.detect(rgb)                 → [Detection(bbox, cls, conf), ...]
  → pro Box: robustes Tiefen-Perzentil   (aus viz_out['depths'], Tracker-Tiefe)
  → unproject(center, depth, K, c2w)     → world_xyz im DROID-Welt-Frame
  → ObjectTracker.update(...)            → NN-Assoziation an bestehende Tracks
Run-Ende:
  → ObjectTracker.finalize(save_dir)     → CSV + PLY + Video
```

Der Detektor liefert pro Frame eine Liste von `Detection`-Objekten
(`detector_base.py:84-100`): jede trägt `bbox_xyxy` (Pixel, OpenCV-Konvention
col=x/row=y), `cls_id`, `cls_name` und `conf`.

Weil die Detektion auf dem Full-Res-Originalbild läuft, die Geometrie aber auf der
kleinen Tracker-Tiefenkarte (≈240×288), skaliert `ObjectTracker.update()` die Box
zuerst auf die Tiefenauflösung (`object_tracker.py:342-352`):

```python
sx, sy = Wd / float(det_hw[1]), Hd / float(det_hw[0])
box_d = (x1 * sx, y1 * sy, x2 * sx, y2 * sy)   # -> depth resolution
```

Die Tiefe pro Box wird **robust** gesampelt (`sample_box_depth`,
`object_tracker.py:72-101`): Die Box wird auf ein zentrales Fenster geschrumpft
(`box_shrink`), ungültige Tiefen (nicht-finit, außerhalb `[min_d, max_d]`) verworfen
und das `depth_percentile`-te Perzentil genommen. Mit dem Default 30 gewinnt der
**nähere** Wert — so bestimmt das Objekt und nicht der Boden dahinter die Tiefe.

Die Rückprojektion (`unproject_center`, `object_tracker.py:56-69`) ist ein
Standard-Pinhole-Unproject gefolgt von der Kamera-zu-Welt-Transformation:

```python
x_cam = (col - cv) / fv * z
y_cam = (row - cu) / fu * z
p_cam = np.array([x_cam, y_cam, z, 1.0])
return (c2w @ p_cam)[:3]
```

### 2.2 Modelle und Backends

Es sind zwei Detektor-Backends registriert, beide über das `ultralytics`-Paket:

| kind | Klasse | Modelle | Default-Gewicht | Import |
|---|---|---|---|---|
| `yolo` | `YoloDetector` (`yolo_detector.py`) | `yolov8n` (schnell) … `yolov8x` (genau) | `ckpts/yolov8n.pt` | `from ultralytics import YOLO` |
| `rtdetr` | `RtdetrDetector` (`rtdetr_detector.py`) | `rtdetr-l`, `rtdetr-x` | `ckpts/rtdetr-l.pt` | `from ultralytics import RTDETR` |

RT-DETR ist ein Transformer-basierter Detektor und als „drop-in alternative" mit
identischem Output-Contract konzipiert — der Wechsel ist eine Ein-Zeilen-Config-Änderung
(`kind: rtdetr`). Beide Backends laden ihr Modell lazy (erst beim ersten `detect()`),
flippen RGB→BGR für die ultralytics-Numpy-Konvention und delegieren das Parsen an
`boxes_to_detections(results[0].boxes, results[0].names)`.

**Gewichts-Auflösung** (`_resolve_weights`, `yolo_detector.py:71-77`): Existiert die
repo-relative `.pt`-Datei, wird sie geladen; fehlt sie, wird der bloße Modellname an
ultralytics gereicht → Auto-Download. Dadurch ist jedes kompatible Gewicht drop-in
ladbar.

**COCO vs. domänenspezifische Gewichte.** Default ist COCO (80 Klassen,
`COCO_CLASSES` in `detector_base.py:34-48`); ein Klassenfilter wie `[2,5,7]` behält
nur car/bus/truck. Für Aerial-Daten existiert eine VisDrone-Config
(`interval1_objects_visdrone.yaml`) mit `model: yolov8s` und
`ckpt_path: ckpts/yolov8s-visdrone.pt`. Entscheidend für korrekte Labels: Das Parsen
benutzt bewusst `results[0].names` (das modell-eigene id→name-Mapping), **nicht** die
hardcodierte COCO-Tabelle. Der Docstring formuliert es explizit: *„using it (not a
hardcoded COCO table) keeps labels correct for non-COCO models like VisDrone
(car=3, not 2)."* (`detector_base.py:103-124`).

### 2.3 3D-Lokalisierung im Detail: Orientierung, Größe, Fusion

Über die reine Position hinaus schätzt das Modul **Orientierung und Größe** pro Objekt
per PCA (`estimate_pose_size`, `object_tracker.py:104-172`). Dazu werden alle gültigen
Tiefenpixel der geschrumpften Box zu einer Mini-Punktwolke entprojiziert, zentriert und
per SVD analysiert. Der Yaw ergibt sich aus der auf die Horizontalebene projizierten
Hauptachse (`atan2`), die Größe aus robusten 95/5-Perzentil-Extents
`[long, lateral, vertical]`. Liefert die PCA zu wenige Pixel (`min_pca_px`) oder eine
fast vertikale Hauptachse, wird `None` zurückgegeben.

Eine bewusste Einschränkung ist die **180°-Yaw-Ambiguität**: Eine PCA-Achse ist
vorzeichenfrei, das Heading lässt sich daher nur auf `[0,π)` bestimmen — vorne/hinten
ist geometrisch nicht rekonstruierbar (`object_tracker.py:114-118`).

Die **Mapper-Pinhole-Konvention** ist hier kritisch und im Code explizit dokumentiert:
`unproject_center` repliziert exakt die Intrinsik-Benennung des Mappers
(`fu=f_y, fv=f_x, cu=c_y, cv=c_x`, `object_tracker.py:20-37`) und **nicht** das K der
Frame-Selektoren. Nur so liegen die Objekt-Marker deckungsgleich mit der Map-PLY, die
der Mapper über dieselbe Konvention baut.

Die **Online-Fusion** (`_associate`, `object_tracker.py:379-392`) ist ein
Nearest-Neighbor-Matching an bestehende Tracks gleicher Klasse, wenn die Distanz zum
Centroid unter `assoc_radius` liegt; sonst entsteht ein neuer Track. Jeder Track
akkumuliert Punkte, Confidences, Klassen-IDs, Yaws und Größen und reduziert sie beim
Finalisieren zu robusten Schätzern:

| Größe | Fusions-Methode | Code |
|---|---|---|
| Position | confidence-gewichteter Median | `fused_position`, `:222-227` |
| Yaw | Doppelwinkel-Zirkularmittel (Lift auf 2·yaw, mitteln, halbieren); `None` bei Inkohärenz `< 0.5` | `fused_yaw`, `:233-253` |
| Größe | Achsen-Median, geclampt auf min. 0.2 | `fused_size`, `:262-267` |

### 2.4 Designentscheidungen

**Detektions-Takt vom Mapper entkoppelt (`object_detect_stride`).** Dies ist die
zentrale, Bachelorarbeits-relevante Entscheidung. Die Detektion läuft auf *jedem N-ten
Tracker-Keyframe*, **nicht** gekoppelt an die `do_map`-Entscheidung des FrameSelectors
(`run.py:848-855`). Der Grund: Der FrameSelector ist gerade das Forschungsobjekt der
Arbeit und filtert die *gemappten* Keyframes hart — eine an `do_map` gekoppelte
Detektion verlöre die meisten Objekte. Auf nicht-gemappten Keyframes ist
`viz_out['depths']` die rohe DROID-BA-Tiefe (der Metric3D-Swap ist mapper-only), genau
die Konvention, die `unproject_center` erwartet. Default-Stride ist 3; bei langen
Läufen wird 5–10 empfohlen.

**Detektion auf Full-Res, Geometrie auf Depth-Res.** Das 240×288-`viz_out`-Bild ist
„far too small for aerial objects"; daher detektiert das Modul auf dem Full-Res-Original
(`cv2.imread` des Dateipfads) und skaliert die Box anschließend zur Tiefenauflösung
zurück (`run.py:863-878`).

**Echter Unix-Zeitstempel.** Pro Detektion wird der echte Cam-Zeitstempel
(`dataset._cam_t_sec`, Unix-Epoch) bevorzugt statt des bloßen Frame-Index, damit später
eine GPS/RTK-Korrelation möglich ist (`run.py:882-893`).

**Best-effort-Kapselung.** Der gesamte Detektionsblock und auch `finalize()` stehen in
`try/except` (`run.py:860-903`, `1054-1058`).

### 2.5 Output-Artefakte

Am Run-Ende schreibt `finalize()` (`object_tracker.py:468-503`) vier Artefakte in den
`save_dir` (nach Filterung auf `n_hits ≥ min_hits` und Sortierung nach
`(−n_hits, −conf)`):

| Datei | Inhalt | Schema / Writer |
|---|---|---|
| `objects_droid.csv` | ein fusioniertes Objekt pro Zeile | `object_id,class,cls_id,conf,n_detections,x,y,z,qw,qx,qy,qz,sx,sy,sz` (`:525-537`) |
| `detections_per_frame.csv` | eine Zeile pro Detektion pro KF (zeitliche Spur) | `frame_idx,t_sec,kf,object_id,class,cls_id,conf,x1,y1,x2,y2,depth,wx,wy,wz` (`:505-523`) |
| `object_markers_droid.ply` | klassen-gefärbte 3D-Marker im 2DGS-Splat-Format | je Objekt 80 Splats als Kugel, Schema identisch zur Map-PLY (`:539-587`) |
| `object_overlay.mp4` | Sanity-Check-Video (Box + Label + Tiefe pro KF) | `_save_overlay`/`_write_video` (`:411-441`, `:589-607`) |

Da die Marker-PLY das **gleiche Schema wie die Map-PLY** verwendet
(`construct_list_of_attributes('2dgs')`), öffnet sie z. B. in superspl.at direkt neben
der Map. Zusätzlich liefert `snapshot()` (`:444-465`) die aktuell fusionierten Objekte
JSON-serialisierbar für den Live-Stream (siehe Abschnitt 4).

### 2.6 Konfigurationsparameter

| Param | Default | Bedeutung |
|---|---|---|
| `detect_objects` | — | Master-Gate (bool) |
| `object_detect_stride` | 3 | jeder N-te Tracker-KF (`max(1,·)`) |
| `object_detector.kind` | — | `yolo` / `rtdetr` / `none` |
| `object_detector.model` | `yolov8n` / `rtdetr-l` | Modellname |
| `object_detector.ckpt_path` | `ckpts/yolov8n.pt` | Gewicht; fehlt → Auto-Download |
| `object_detector.conf` | 0.35 | Confidence-Threshold |
| `object_detector.iou` | 0.7 | NMS-IoU |
| `object_detector.imgsz` | 640 (Aerial: 1280) | Inferenz-Auflösung |
| `object_detector.device` | cuda (Config: cpu) | VRAM-schonend |
| `object_detector.classes` | None | zu behaltende IDs; None = alle |
| `object_detector.max_det` | 100 | max. Detektionen/Frame |
| `object_detector.box_shrink` | 0.5 | zentrales Box-Fenster fürs Tiefen-Sampling |
| `object_detector.depth_percentile` | 30.0 | näheres Perzentil gewinnt über Boden |
| `object_detector.min_valid_px` | 10 | min. gültige Tiefenpixel pro Box |
| `object_detector.min_depth` / `max_depth` | 0.2 / 60.0 | Tiefenschranken |
| `object_detector.min_pca_px` | 30 | min. Pixel für vertrauenswürdigen Yaw/Size |
| `object_detector.size_percentile` | 95.0 | robuste 95/5-Extent-Spanne |
| `object_tracker.assoc_radius` | 0.05 | NN-Radius, **gauge-frei** (DROID-Einheit!) |
| `object_tracker.min_hits` | 3 | min. Sichtungen, sonst rausgefiltert |
| `object_tracker.class_agnostic` | False | klassenübergreifend mergen + Mehrheits-Voting |
| `object_tracker.marker_radius` | = assoc_radius | PLY-Marker-Radius |
| `object_output.{csv,markers_ply,overlay_video,detections_csv}` | True | Artefakt-Schalter |
| `object_output.overlay_fps` / `overlay_max_w` | 10 / 1600 | Video-Parameter |

> **Gauge-Hinweis:** `assoc_radius` ist nur bei metrischer Welt (GT-`ext_poses` + LiDAR,
> z. B. interval1) ein Meter-Wert (`3.0`). Bei reinem VO ist die Welt gauge-frei → klein
> wählen (`0.05`).

### 2.7 Bekannte Limitierungen

1. **COCO-YOLO ist auf Nadir-Aerial schwach.** Bei Top-Down-Flug sehen COCO-Modelle
   winzige Objekte und vergeben falsche Klassen (umbrella/tv/…). Workaround:
   `classes:[2,5,7]` + `imgsz:1280` + niedriger `conf`. Für ernsthafte Aerial-Detektion
   muss `ckpt_path` auf ein VisDrone-/DOTA-trainiertes `.pt` zeigen (drop-in). Auf
   oblique/Boden-Daten ist COCO direkt brauchbar.
2. **Bewegte Objekte** erscheinen an mehreren Weltpositionen → mehrere/verschmierte
   Tracks (statische Annahme). Ein Filter über die Dynamik-Maske ist denkbar.
3. **`assoc_radius` ist gauge-frei** — metrisch sauberes Re-Merge erst mit dem späteren
   Sim3-Schritt (Abschnitt 4.5).
4. **VRAM** — Detektor default `device: cpu`, lazy-load, kleines Modell, Läufe seriell.

---

## 3. Segmentierung und Dynamik-Maskierung

### 3.1 Zwei-Phasen-Architektur

Die Kernidee, wörtlich aus dem Modul-Header (`dynamic_utils.py:1-18`), trennt das
Problem in zwei entkoppelte Hälften:

1. **Segmentierung** — ein austauschbares Backend (FastSAM/SAM2/SAM3) zerlegt einen
   RGB-Frame klassen-agnostisch in K Instanz-Masken („everything").
2. **Dynamik-Detektion** — `compute_dynamic_mask()` flaggt diejenigen Segmente, deren
   Pixel einen überproportional hohen photometrischen Fehler zwischen gerendertem und
   echtem Bild tragen. Diese Pixel bilden die Dynamik-Maske und fallen aus dem
   Mapping-Loss.

Der Charme dieser Trennung: Das Segmentierungsmodell braucht kein semantisches Wissen
darüber, was „dynamisch" ist — diese Entscheidung trifft allein der photometrische
Fehler. Ein geparktes Auto erzeugt keinen hohen Fehler, ein fahrendes schon.

### 3.2 Pipeline im Run-Loop

```
run.py (pro Keyframe-Batch viz_out):
  DynamicModel.get_anns_raw(rgb)  ──► viz_out['sam_anns'] = [(K,H,W) bool, ...]   # SAM 1× pro KF
        ▼
mapper.run(viz_out)  →  train_once_gaussian(batch=viz_out):
  pro Trainings-Iter, gewähltes KF curr_id:
    compute_dynamic_mask(batch['sam_anns'][curr_id], gt_rgb, pred_rgb)            # billig
        → gt_dict['dynamic_mask']  (H,W) bool, True = dynamisch
        ▼
get_loss(cfg, pred_dict, gt_dict):
  valid_mask &= ~dynamic_mask     # dynamische Pixel raus aus rgb/normal/depth/dist
```

Entscheidend für die Performance: Die **teure SAM-Inferenz läuft genau einmal pro
Keyframe** (`run.py:944-949`, PhaseTimer-Phase `segment`), nicht pro Trainings-Iteration.
Die Masken reisen über den bestehenden `viz_out → batch`-Kanal wie die Bilder selbst.
`compute_dynamic_mask` ist dagegen reine Elementwise-Arithmetik über die gecachten
Segmente und damit pro Iteration unkritisch.

Die Maske wird im Mapper auf dem **Base-Render** berechnet, *bevor* das Sky-Model in
den Render gefused wird (`gaussian_base.py:371-381`), und greift in `get_loss`
(`loss_utils.py:118-119`):

```python
if gt_dict.get('dynamic_mask') is not None:
    valid_mask = torch.bitwise_and(valid_mask, ~gt_dict['dynamic_mask'])
```

`valid_mask` geht anschließend in alle photometrischen/geometrischen Terme (rgb_loss,
normal_loss, depth_loss, dist_loss, disp). Der `alpha_loss` nutzt dagegen weiterhin nur
die `sky_mask` — und bewusst **nicht** die Dynamik-Maske: Dynamische Pixel sollen nicht
als Himmel behandelt werden, sonst würden bewegte Objekte fälschlich Transparenz lernen
(Code-Kommentar `loss_utils.py:115-117`).

### 3.3 Modelle und Backends

| kind | Modell | Status | Idee | Gewicht |
|---|---|---|---|---|
| `fastsam` | FastSAM-x / -s (YOLOv8-seg) | live | prompt-frei „everything", schnell | `ckpts/FastSAM-x.pt`, Auto-Download |
| `sam2` | SAM2.1 (t/s/b/l) | live, **empfohlen** | echtes SAM2.1, „segment everything" | `ckpts/sam2.1_b.pt` (~162 MB) |
| `sam3` | SAM3 (Concept) | **Code da, Weights gated** | text-getrieben („car/person/…") | `ckpts/sam3.pt`, nicht frei ladbar |

Alle drei laufen über `ultralytics`. FastSAM (`fastsam_backend.py`) und SAM2
(`sam2_backend.py`) sind nahezu identische Wrapper; der Hauptunterschied ist die native
Inferenz-Auflösung (FastSAM 512, SAM2 1024 → mehr VRAM, aber bessere Qualität). Beide
geben die Masken auf der **CPU** zurück, um den Mapper-VRAM frei zu halten.

SAM3 (`sam3_backend.py`) ist konzeptionell anders: *concept-driven / text-grounded*.
Man übergibt eine Klassenliste (Default `[car, truck, bus, person, bicycle,
motorcycle]`) und erhält direkt Masken für diese Konzepte. Der high-Error-Filter
komponiert *obendrauf* — auch SAM3-Klassenmasken durchlaufen `compute_dynamic_mask`,
sodass geparkte von fahrenden Autos getrennt werden. **Blocker:** Die SAM3-Gewichte
sind nicht im ultralytics-Asset-Index, Metas HF-Repo `facebook/sam3` ist `gated`
(HTTP 401) und liefert das transformers- statt des ultralytics-`.pt`-Formats;
`_resolve_weights()` wirft daher einen klaren `FileNotFoundError`.

### 3.4 Der Error-Proxy (Dynamik-Detektion)

Der Kern ist `compute_dynamic_mask` (`dynamic_utils.py:45-89`). Der photometrische
Fehler pro Pixel ist das Produkt aus L1 und (1−SSIM), gemittelt über die Farbkanäle:

```python
rgb_l1 = torch.abs(pred_rgb - gt_rgb).mean(dim=0)
rgb_ssim = 1.0 - ssim_img(pred_rgb, gt_rgb).mean(dim=0)
multi_loss = rgb_l1 * rgb_ssim
thr = torch.quantile(multi_loss, loss_quantile)
high_loss_mask = multi_loss > thr
```

Die `ssim_img`-Funktion ist dieselbe Gauß-Fenster-SSIM wie im Trainings-Loss
(`loss_utils.py:60-78`). Die Schwelle ist **pro Frame adaptiv**: Die obersten 10 %
Fehler-Pixel (`loss_quantile = 0.9`) gelten als „high-loss", kein fester Wert.

Ein Segment wird genau dann als dynamisch markiert, wenn ein **Doppelkriterium** (AND)
erfüllt ist (`dynamic_utils.py:75-84`):

- `rate = (#high-loss-Pixel im Segment) / (#Pixel im Segment) > dyn_high_rate (0.2)`, **und**
- `mean(Fehler über Segment) > dyn_mean_loss (0.002)`.

Das zweite, absolute Kriterium ist ein Floor, der verhindert, dass ein insgesamt
fehlerarmes Segment allein durch die relative Quantil-Definition geflaggt wird. Die
Ausgabe ist die Vereinigung (`any(dim=0)`) aller dynamischen Segmente als `(H,W)`-Bool.

Sichere Defaults: K=0 oder `raw_ann is None` → leere Maske (kein Masking); leere
Segmente werden übersprungen; keine dynamischen Segmente → leere Maske. Der Smoketest
verifiziert diese K=0-Härtung explizit.

### 3.5 Designentscheidungen

**War früher Dead Code.** Die ursprüngliche `DynamicModel`-Klasse nutzte FastSAM, war
aber nirgends im Run-Loop aufgerufen und hatte einen hardcodierten Pfad
(`/data/wuke/workspace/FastSAM/`). Die jetzige Version (live seit 2026-06-05) ersetzt
sie vollständig und ist tatsächlich in `run.py`/`gaussian_base.py`/`loss_utils.py`
verdrahtet.

**Loss-basiert statt semantisch.** Die Detektion ist klassen-agnostisch — high-Error-
Regionen gelten als dynamisch, ohne „car"/„person"-Labels. Vorteil: kein
domänenspezifisches Detektionsmodell nötig. Nachteil: siehe 3.7.

**Swappable Registry-Factory** identisch zum Selektor-Pattern (`segmentation_base.py`
ABC + `segmentation_factory.py`); Mapper/Loss/Run-Loop bleiben bei einem Backend-Wechsel
unberührt.

**Maske auf Base-Render vor Sky-Fuse**, und `alpha_loss` nutzt bewusst nicht die
Dynamik-Maske (siehe 3.2).

### 3.6 Konfigurationsparameter

| Param | Default | Bedeutung |
|---|---|---|
| `use_dynamic` | false | Master-Schalter (zusammen mit `segmentation.kind` ≠ none) |
| `segmentation.kind` | none | `fastsam` / `sam2` / `sam3` |
| `segmentation.model` | `FastSAM-x` / `sam2.1_b` | Modellvariante (VRAM↔Qualität) |
| `segmentation.ckpt_path` | `ckpts/...` | Gewicht; fehlt → Auto-Download (außer SAM3) |
| `segmentation.device` | cuda | Config-Override gewinnt |
| `segmentation.conf` | 0.4 | Confidence-Threshold |
| `segmentation.iou` | 0.9 | NMS-IoU |
| `segmentation.imgsz` | 512 (FastSAM) / 1024 (SAM2) | Inferenz-Auflösung |
| `segmentation.min_area_px` | 0 | Masken < X px verwerfen |
| `segmentation.classes` | (SAM3) car/truck/… | Text-Konzepte für SAM3 |
| `segmentation.dyn_loss_quantile` | 0.9 | Error-Quantil für „high loss" |
| `segmentation.dyn_high_rate` | 0.2 | Segment dynamisch wenn > 20 % high-loss … |
| `segmentation.dyn_mean_loss` | 0.002 | … UND mittlerer Fehler über diesem Floor |

Beispiel-Configs: `configs/local/dynamic/amtown03_s1000_400f_dynamic_{fastsam,sam2}.yaml`
(Vollläufe) und `..._smoke.yaml` (Smoketests).

### 3.7 Bekannte Limitierungen

1. **Kritisch auf Aerial: Der Error-Proxy flaggt das Falsche.** Auf Nadir-Drohnenszenen
   ist der höchste photometrische Fehler nicht das kleine, langsam bewegte Auto, sondern
   schlecht rekonstruierte Dächer/Strukturen. Dem klassen-agnostischen Heuristik fehlt
   das semantische Wissen, dass nur Fahrzeuge dynamisch sind → auf Aerial ist der
   Online-SAM-Pfad unzuverlässig (daher der externe-Masken-Weg, 3.8).
2. **SAM3 nicht lauffähig** ohne manuelle Gewichts-Beschaffung (gated, transformers-Format).
3. **SAM generalisiert nicht auf Nadir** ohne Fine-Tuning; UAVScenes-GT ist für
   AMtown03/HKairport der bessere Stack.
4. **Render-qualitäts-abhängig.** Da der Proxy `pred_rgb` (aktueller Map-Render) braucht,
   ist die Detektion in frühen Iterationen verrauscht; Quantil-Schwelle und
   `dyn_mean_loss`-Floor sind die einzigen Stabilisatoren.

### 3.8 Alternativweg: externe semantische Masken (UAVScenes-GT)

Für die offiziellen MARS-LVIG-Sequenzen existieren per-Frame-GT-Annotationen
(UAVScenes, 19 Klassen). Für AMtown03 sind 1120 Frames gelabelt, darunter Sedan und
Truck als Dynamik-Klassen. Ein **kritischer Befund** dieser Arbeit: Die id-PNGs
verwenden Cityscapes-style-IDs, nicht die Paper-Tabelle — verifiziert per
Farb-/Pixelverteilung: **Sedan = 20** (in 774/1120 = 69 % der Frames), **Truck = 24**.
Die zuvor angenommenen Paper-IDs (Sedan=17/Truck=18) lieferten 352/0 und waren falsch.

| | SAM-online (`use_dynamic`) | Externe UAVScenes-GT |
|---|---|---|
| Quelle | Live-Inferenz pro KF | vorab-gelabelte GT-PNGs |
| Klassen | class-agnostic (Error) bzw. Text (SAM3) | echte Semantik (Sedan/Truck) |
| Qualität auf Aerial | schlecht (flaggt Dächer) | sauberer als jedes Inferenzmodell |
| Verfügbarkeit | überall (kein GT nötig) | nur offizielle MARS-Sequenzen |
| Integration | automatisch via `viz_out['sam_anns']` | bräuchte Mask-Loader + Loss-Patch (Konzept) |

---

## 4. Einbettung und Visualisierung (Kontext)

### 4.1 Reihenfolge im Run-Loop

Beide Module sind in den Tracker→Mapper-Loop eingehängt; die Reihenfolge pro
Tracker-Keyframe (`run.py`, ab ≈809) ist:

1. `_geo_add_keyframe` — DROID-Pose + GPS-ENU an den Geo-Referencer (nur Streaming-Pfad).
2. **FrameSelector-Entscheidung** (`do_map`) — Plugin-Selector oder Modulo.
3. **Objekterkennung** — Gate `(detector ≠ None) ∧ ('images' in viz_out) ∧
   ((n_keyframes−1) % stride == 0)`; sitzt **nach** der `do_map`-Entscheidung, aber
   **vor** `if do_map:` (Entkopplung, siehe 2.4).
4. Live-RGB-Frame-Push für die Kamera-Karte.
5. **`if do_map:`** — Metric3D-Depth-Swap, **Segmentierung** (`viz_out['sam_anns']`),
   dann `mapper.run(viz_out, True)`.
6. Loop-Closure → ggf. Epoch++ und `resync`-Push.
7. Gaussian-Stream-Push nach dem Storage-Run.

### 4.2 Live-Streaming

Der `SplatStreamServer` (`scripts/server/stream_server.py`) ist ein
Daemon-Thread mit eigener asyncio-Loop; der Run-Loop füttert eine `queue.Queue` mit
**drop-oldest**-Strategie (droppbar nur `replace_active|replace_all|objects|geo`, nie
`append_frozen|resync`). Über denselben Port wird auch die `viewer.html` ausgeliefert
(Static-Server mit Traversal-Guard).

Das Wire-Protokoll ist zweigeteilt: **binär** = `.splat`
(`[uint32-LE header_len][JSON header][.splat bytes]`, 32 B/Gaussian, Quaternion
(w,x,y,z); Serializer `splat_encode.py`) und **Text-JSON** für die `objects`-Message
mit `{object_id, class, conf, xyz, quat:[w,x,y,z], size:[sx,sy,sz]}` (aus
`object_tracker.snapshot()`). Ein frozen/active-Split überträgt die fertigen KFs
inkrementell (Key `_globalkf_id`), die aktiven jedes Mal als full-replace; bei
Loop-Closure verwirft das Frontend Nachrichten mit veralteter Epoch.

### 4.3 Frontend-Rendering der Objekte

Das three.js-Frontend (`viewer.html`) rendert pro Klasse echte glTF-Modelle über eine
`MODEL_REGISTRY` (`car→car.glb, truck→truck.glb, bus→bus.glb, van→car.glb`,
überschreibbar via `static/models/registry.json`). Jedes Objekt wird an `xyz`
positioniert, per `quat` orientiert und uniform in seine `size`-Box skaliert. Fehlt ein
`.glb`, greift ein orientierter Wireframe-Box-Fallback mit gleicher Pose/Größe. Die
Gaussians selbst sind als Disks (InstancedMesh) oder EWA-Splats (ShaderMaterial)
darstellbar.

### 4.4 Geo-Referenzierung

Für die Kartendarstellung wird die DROID-Welt auf einen Esri-Satelliten projiziert.
Bewusst **kein** Umeyama-Fit (der bei nahezu geradem GPS-Flug degeneriert), sondern ein
physik-fundierter Frame aus *up* (= −mean(camera forward), Nadir-Cam),
*heading* (DROID-Chord rotiert auf GPS-Chord), *scale* (GPS-/DROID-Along-Track) und
Zentroid-Match (`geo_frame.py:build_geo_frame`). Ein dokumentierter Fallstrick ist die
Händigkeit: `right = fwd × up` ist Pflicht, sonst spiegelt die Szene. Das Verzeichnis
`od-experiments/` enthält die Offline-Validierung dieser Projektion, die später in den
Live-Pfad übernommen wurde.

### 4.5 Nachgelagerte Metrik / GPS

Da die Objekt-Marker im selben DROID-Frame wie Map und `tracker_raw_c2w.txt` liegen,
lassen sie sich mit der bestehenden Pipeline metrisch machen:

```bash
python scripts/eval/sim3_unwarp.py \
  --droid-poses .../tracker_raw_c2w.txt --gps-csv .../rtk_positions_raw.csv \
  .../object_markers_droid.ply --out object_markers_gps.ply
```

**Offene Punkte / Vorbehalte:** Ein dünnes Apply-Skript für die CSV-Positionen
(lat/lon/UTM) ist geplant, aber noch nicht implementiert. Außerdem erreicht
`ext_poses_file` den Mapper-Pfad nicht zuverlässig (dokumentierter ~35×-Skalenfehler bei
DROID-Drift) — relevant, falls das Kapitel metrische Objektgenauigkeit behauptet.

---

## 5. Reproduzierbarkeit

**Standalone-Smoketests** (jeweils mit synthetischem Frame bei fehlendem Bild-Argument):

```bash
python scripts/vings_utils/yolo_detector.py [bild.jpg]    # YOLO: load + detect + parse
python scripts/vings_utils/rtdetr_detector.py [bild.jpg]  # RT-DETR
python scripts/vings_utils/object_tracker.py              # Achsen-Sanity + Fusions-Check
python scripts/vings_utils/fastsam_backend.py             # FastSAM: #Masken + Overlay
python scripts/vings_utils/sam2_backend.py                # SAM2.1: #Masken + Overlay
python scripts/vings_utils/sam3_backend.py                # SAM3: zeigt Weights-Blocker
python scripts/dynamic/dynamic_utils.py                   # Segment + Dynamik + K=0-Härtung
```

Der `object_tracker.py`-Test ist besonders nützlich: Er prüft die Achsenkonvention
(Mitte→`[0,0,5]`, rechts→`+X`, unten→`+Y`) und fängt eine `[u,v]`-Vertauschung sofort.

**Voller Lauf** (seriell, RAM/VRAM-Watchdog beachten):

```bash
python scripts/run_experiment.py configs/local/object_detect/interval1_objects.yaml
python scripts/run_experiment.py configs/local/dynamic/amtown03_s1000_400f_dynamic_sam2.yaml
```

Profiling-Phasen im PhaseTimer-Summary: `detect` (Objekterkennung) und `segment`
(Segmentierung).

---

## 6. Datei- und Referenzanhang

### Objekterkennung

| Datei | Rolle |
|---|---|
| `docs/OBJECT_DETECTION.md` | Hauptdoku |
| `scripts/vings_utils/detector_base.py` | `Detection`, `COCO_CLASSES`, `boxes_to_detections` |
| `scripts/vings_utils/detector_factory.py` | Registry + `make_object_detector` |
| `scripts/vings_utils/yolo_detector.py` | YOLO-Backend (`kind: yolo`) |
| `scripts/vings_utils/rtdetr_detector.py` | RT-DETR-Backend (`kind: rtdetr`) |
| `scripts/vings_utils/object_tracker.py` | Unprojection, PCA-Pose/Size, Fusion, Writer, `snapshot` |
| `scripts/run.py:215-235, 848-903, 1051-1058` | Init, Detektionsblock, Finalize |
| `configs/local/object_detect/` | Beispiel-Configs (COCO + VisDrone) |

### Segmentierung

| Datei | Rolle |
|---|---|
| `docs/SEGMENTATION_BACKEND.md` | Hauptdoku (swappable Backend) |
| `docs/SEGMENTATION_AMTOWN.md` | externe UAVScenes-Masken (Alternative) |
| `scripts/vings_utils/segmentation_base.py` | ABC + `to_uint8_rgb` (Swap-Vertrag) |
| `scripts/vings_utils/segmentation_factory.py` | Registry + `make_segmentation_backend` |
| `scripts/vings_utils/{fastsam,sam2,sam3}_backend.py` | die drei Backends |
| `scripts/dynamic/dynamic_utils.py` | `DynamicModel` + `compute_dynamic_mask` |
| `scripts/gaussian/loss_utils.py:60-119` | `ssim_img`, `get_loss` (`valid_mask &= ~dynamic_mask`) |
| `scripts/gaussian/gaussian_base.py:371-381` | baut `gt_dict['dynamic_mask']` |
| `scripts/run.py:209-213, 941-949` | Init `DynamicModel`, Segmentierung pro KF |
| `configs/local/dynamic/` | Beispiel-Configs (fastsam/sam2 × full/smoke) |

### Einbettung & Visualisierung

| Datei | Rolle |
|---|---|
| `docs/STREAMING.md` | Konzept-Doku Live-Stream |
| `scripts/server/stream_server.py` | WebSocket-Daemon (drop-oldest, Ein-Port) |
| `scripts/server/splat_encode.py` | `.splat`-Serializer (32 B/Gaussian) |
| `scripts/server/geo_frame.py` | `LiveGeoReferencer`, `build_geo_frame`, Esri-Fetch |
| `scripts/server/replay_run.py` | SLAM-freier E2E-Test des Geo+OD-Streams |
| `scripts/server/static/viewer.html` | Frontend: Splats, MODEL_REGISTRY, glTF/Box |
| `od-experiments/` | Offline-Validierung der Geo-Projektion |
| `scripts/eval/sim3_unwarp.py` | nachgelagerte Metrik/GPS-Transformation |
