# Objekterkennung + Online-3D-Lokalisierung

## Worum geht's (in einfachen Worten)

Während VINGS läuft, schauen wir uns jeden Keyframe an und fragen einen
Objektdetektor (YOLO oder RT-DETR): *Was ist auf diesem Bild — Autos, Personen,
LKW?* Für jedes gefundene Objekt nehmen wir die **geschätzte Tiefe** an der
Stelle der Box und die **Kamerapose** und rechnen aus, **wo das Objekt in der
3D-Welt steht**. Dasselbe Auto sehen wir über mehrere Keyframes — diese
Beobachtungen fassen wir zu **einem** Objekt mit einer 3D-Position zusammen.

Am Ende des Laufs liegen drei Dateien im `save_dir`:

| Datei | Was drin ist |
|---|---|
| `objects_droid.csv` | eine Zeile pro **fusioniertem Objekt**: ID, Klasse, Confidence, Anzahl Sichtungen, x/y/z |
| `detections_per_frame.csv` | eine Zeile pro **Detektion pro Keyframe** — die zeitliche Spur: `frame_idx, t_sec, kf, object_id, class, conf, bbox, depth, world xyz`. `object_id` verknüpft die Detektion mit ihrem fusionierten Objekt (−1 = Track unter `min_hits` rausgefiltert). `t_sec` ist der Frame-Zeitstempel (Unix). |
| `object_markers_droid.ply` | klassen-gefärbte 3D-Marker im **2DGS-Gaussian-Splat-Format** (gleiches Schema wie die Map-PLY) → öffnet in superspl.at / jedem Splat-Viewer **neben der Map** |
| `object_overlay.mp4` | die Keyframes mit eingezeichneten Boxen + Klasse + Tiefe (Sanity-Check) |

Die Koordinaten sind im **DROID-Weltframe** (gauge-frei, dieselbe Welt wie die
Map-PLY). Die Umrechnung in **echte GPS-/Meter-Koordinaten kommt später** — sie
ist nur ein Sim3-Transform und wird unten erklärt.

## Anschalten

```yaml
detect_objects: true          # Master-Gate
object_detector:
  kind: yolo                  # yolo | rtdetr | none
  model: yolov8n
  classes: [2, 5, 7]          # COCO car/bus/truck; null = alle
  device: cpu                 # VRAM-schonend
object_tracker:
  assoc_radius: 0.05          # DROID-Frame-Einheiten (kein Meter!)
  min_hits: 3
object_output:
  csv: true
  markers_ply: true
  overlay_video: true
```

Beispiel-Config: `configs/local/object_detect/interval1_objects.yaml`.
Lauf: `python scripts/run_experiment.py configs/local/object_detect/interval1_objects.yaml`
(seriell — RAM/VRAM-Watchdog beachten).

Standalone-Smoketests:
```bash
python scripts/vings_utils/yolo_detector.py [bild.jpg]
python scripts/vings_utils/rtdetr_detector.py [bild.jpg]
python scripts/vings_utils/object_tracker.py        # Achsen- + Fusions-Unit-Check
```

## Wie es im Code hängt

```
RGB-Keyframe (viz_out['images'][-1])
  → detector.detect(rgb)            → [Detection(bbox, cls, conf), ...]
  → pro Box: robustes Tiefen-Perzentil aus viz_out['depths'][-1]
  → unproject(center, depth, intrinsic, c2w)   → world_xyz (DROID-Frame)
  → ObjectTracker.update(...)       → NN-Assoziation an bestehende Tracks
Run-Ende:
  → ObjectTracker.finalize(save_dir) → CSV + PLY + Video
```

Eingehängt in `scripts/run.py` direkt **nach** dem Segment-Block und **vor**
`mapper.run()` (gleiche Stelle, an der schon segmentiert wird — dort liegen
`images`, `depths`, `poses`, `intrinsic` für den neuesten KF vor).

Module (Registry-Factory wie bei Selektoren/Segmentierung):

| Datei | Inhalt |
|---|---|
| `scripts/vings_utils/detector_base.py` | `ObjectDetectorBase`, `Detection`, COCO-Klassen + Farben, `boxes_to_detections` |
| `scripts/vings_utils/yolo_detector.py` | `YoloDetector` (`@register_detector("yolo")`) |
| `scripts/vings_utils/rtdetr_detector.py` | `RtdetrDetector` (`@register_detector("rtdetr")`) |
| `scripts/vings_utils/detector_factory.py` | `make_object_detector(cfg, device)` |
| `scripts/vings_utils/object_tracker.py` | Unprojection + Online-Fusion + Writer |

## Details

### Tiefen-Sampling pro Box
`sample_box_depth` schrumpft die Box auf das zentrale Fenster (`box_shrink`),
verwirft ungültige Tiefen (0/NaN/außerhalb `[min_depth, max_depth]`) und nimmt
das `depth_percentile`-te Perzentil (default 30 — ein **näherer** Wert, damit
das Objekt gewinnt und nicht der Boden dahinter). Zu wenige gültige Pixel →
Detektion wird verworfen.

### Koordinaten-Konvention (wichtig)
Die Marker müssen auf der Map-PLY liegen. Der Mapper baut die Map über
`gaussian/tf.py` mit dem `viz_out['intrinsic']`-Dict, wo **`fu=f_y, fv=f_x,
cu=c_y, cv=c_x`** ist. `object_tracker.unproject_center` repliziert genau diese
Standard-Pinhole-Rückprojektion:

```
X_cam = (col - cv) / fv * z      # = (col - cx) / fx * z
Y_cam = (row - cu) / fu * z      # = (row - cy) / fy * z
Z_cam = z
p_world = (c2w @ [X, Y, Z, 1])[:3]
```

Box-Zentren kommen in OpenCV-Reihenfolge `(col=x, row=y)` — wie ultralytics sie
liefert. Der Unit-Check in `object_tracker.py` (`__main__`) fängt eine
`[u,v]`-Vertauschung sofort (Mitte→`[0,0,5]`, rechts→`+X`, unten→`+Y`).

Bewusst **nicht** das K der Selektoren (`run.py:185`) verwendet — dessen
fu/fv-Benennung kann von der Mapper-Konvention abweichen; nur die
Mapper-Konvention garantiert, dass Marker und Map deckungsgleich sind.

### Online-Fusion
`ObjectTracker._associate` ordnet jede Detektion per nearest-neighbor einem
Track **gleicher Klasse** zu, wenn der Abstand zum Track-Zentroid <
`assoc_radius` ist; sonst neuer Track. `assoc_radius` ist im **gauge-freien
DROID-Frame** (kein Meter!) — szenenabhängig wählen. Die fusionierte Position
ist der conf-gewichtete Median aller Track-Punkte; Tracks mit < `min_hits`
Sichtungen fallen raus (Einmal-False-Positives). `class_agnostic: true` mergt
klassenübergreifend und entscheidet die Klasse per Mehrheits-Voting.

## Nächster Schritt: Metrik / GPS

Die Marker liegen im selben DROID-Frame wie die Map-PLY und
`tracker_raw_c2w.txt`. Damit transformiert die **bestehende** Sim3-Pipeline sie
1:1 wie die Map:

```bash
python scripts/eval/sim3_unwarp.py \
  --droid-poses output/exp_interval1_objects/.../tracker_raw_c2w.txt \
  --gps-csv     /home/philipp/Dokumente/datasets/interval1_AMtown03/rtk_positions_raw.csv \
  output/exp_interval1_objects/.../object_markers_droid.ply \
  --out         object_markers_gps.ply
```

Für die CSV-Positionen (lat/lon/UTM) ist ein dünnes Apply-Skript geplant, das
denselben per-KF lokalen Sim3 auf die `objects_droid.csv`-Punkte anwendet —
**noch nicht implementiert** (bewusst aus dem ersten Schritt herausgehalten).

## Bekannte Grenzen
- **COCO-YOLO ist auf Nadir-Aerial schwach.** interval1 ist ein Top-Down-
  Flug; ein COCO-trainiertes YOLO/RT-DETR sieht winzige Objekte und vergibt bei
  hoher Auflösung viele falsche Klassen (umbrella/tv/...). Mit `classes: [2,5,7]`
  (car/bus/truck) + `imgsz: 1280` + niedriger `conf` kommen vereinzelt echte
  Autos durch. Für ernsthafte Aerial-Detektion: `ckpt_path` auf ein
  **VisDrone-/DOTA-trainiertes** ultralytics-`.pt` zeigen lassen — drop-in, kein
  Code-Change (die Factory lädt jedes kompatible YOLO/RT-DETR-Gewicht). Auf
  oblique/Boden-Daten ist COCO direkt brauchbar.
- **Bewegte Objekte** (fahrende Autos) erscheinen an mehreren Weltpositionen →
  mehrere/verschmierte Tracks. Aktuell: statische Objekte angenommen. Späterer
  Filter über das Dynamic-Masking (`use_dynamic`) denkbar.
- **`assoc_radius` ist gauge-frei** — metrisch sauberes Re-Merge erst mit dem
  späteren Sim3-Schritt.
- **VRAM** — Detektor default auf `device: cpu`; lazy-load; kleines Modell
  (yolov8n/rtdetr-l). Läufe seriell.
