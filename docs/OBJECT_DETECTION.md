# Objekterkennung + Online-3D-Lokalisierung

## Worum geht's (in einfachen Worten)

Während VINGS läuft, schauen wir uns jeden Keyframe an und fragen einen
Objektdetektor (YOLO oder RT-DETR): *Was ist auf diesem Bild — Autos, Personen,
LKW?* Für jedes gefundene Objekt nehmen wir die **geschätzte Tiefe** an der
Stelle der Box und die **Kamerapose** und rechnen aus, **wo das Objekt in der
3D-Welt steht**. Dasselbe Auto sehen wir über mehrere Keyframes — diese
Beobachtungen fassen wir zu **einem** Objekt mit einer 3D-Position zusammen.

Default ist **YOLO26-OBB** (orientierte Boxen, DOTA-v1-trainiert): statt eines
achsenparallelen Rechtecks liefert es eine **gedrehte** Box, deren Winkel die
Fahrzeug-Achse ist. Daraus wird direkt das **Heading** des Objekts in der Karte
— ohne sich auf die (bei Nadir-Aerial oft dünne) Tiefen-PCA verlassen zu müssen.
Auf interval1 verifiziert: Fahrzeuge mit conf bis 0.81, Box-Winkel entlang der
Straße (siehe Smoketest unten).

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
object_detect_stride: 3       # auf jedem N-ten *Tracker*-KF detektieren (Default 3)
object_detector:
  kind: yolo                  # yolo | rtdetr | none
  model: yolo26n-obb          # OBB-Default (DOTA); axis-aligned: yolov8s-visdrone
  classes: [9, 10]            # DOTA-v1 large/small vehicle; null = alle 15
  device: cpu                 # VRAM-schonend
  min_pca_px: 30              # PCA-Fallback für Yaw/Größe wenn kein OBB-Winkel da
  size_percentile: 95.0       # robuste Extent-Spanne (95/5) für die 3D-Größe
object_tracker:
  assoc_radius: 0.05          # DROID-Frame-Einheiten (kein Meter!)
  min_hits: 3
object_output:
  csv: true
  markers_ply: true
  overlay_video: true
```

### Detektions-Takt: jeder N-te Tracker-KF (entkoppelt vom Mapper)
`object_detect_stride` (Default 3) steuert, auf **jedem N-ten Tracker-Keyframe**
detektiert wird — **bewusst entkoppelt** vom FrameSelector/Mapper. Grund: der
FrameSelector filtert die *gemappten* KFs hart (das ist der Sinn der BA), sodass
eine an `do_map` gekoppelte Detektion die meisten Objekte verlöre. Auf
nicht-gemappten KFs nutzt die Lokalisierung die rohe DROID-BA-Tiefe (der
Metric3D-Swap ist mapper-only) — genau die Konvention, die `unproject_center`
erwartet, also landen die Marker weiter im selben Frame wie die Map. **Kosten:**
mehr YOLO/RT-DETR-Pässe als früher → `object_detect_stride` bei langen Läufen
höher setzen (5–10). Der Block ist best-effort (`try/except`), bricht den Run nie.

### Orientierung + Größe (für gestreamte 3D-Modelle)
**Yaw — OBB-Pfad (Default):** Liefert der Detektor einen Box-Winkel (`Detection.angle`,
nur YOLO26-OBB/YOLO11-OBB), rechnet `obb_yaw_world` ihn in einen **Welt-Yaw** um:
es entprojiziert das Box-Zentrum und einen um wenige Pixel entlang der Bild-Long-Axis
versetzten Punkt bei **gleicher Tiefe** und nimmt das Heading der Welt-Differenz —
korrekt auch bei schrägem Blick (Tilt kommt über `c2w`), und **unabhängig** davon,
wie dicht die Box-Tiefe ist. Das ersetzt den PCA-Yaw.

**Yaw — Fallback + Größe:** `estimate_pose_size` schätzt aus der entprojizierten
Tiefen-Punktwolke per PCA einen **Yaw** (Rotation um die Welt-Hoch-Achse, auf
`[0, π)` kanonisiert — 180°-Ambiguität bleibt, vorne/hinten ist geometrisch nicht
bestimmbar; **OBB ist hier genauso mod π**) und eine **3D-Größe**
`[long, lateral, vertical]` (robuste 95/5-Extents). Bei axis-aligned-Detektoren
(yolov8/visdrone/rtdetr, `angle=None`) ist der PCA-Yaw die einzige Quelle. Über die Sichtungen
fusioniert `_Track`: Yaw via Doppelwinkel-Zirkularmittel (conf-gewichtet, Kohärenz-
Schwelle → sonst Identität), Größe via Achsen-Median. `snapshot()` liefert daraus
`quat:[w,x,y,z]` + `size`, die der Live-Stream ans Frontend gibt (siehe
`docs/STREAMING.md`, „3D-Modell-Modus"). `objects_droid.csv` trägt die zusätzlichen
Spalten `qw,qx,qy,qz,sx,sy,sz`.

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

Eingehängt in `scripts/run.py` auf der **Tracker-KF-Ebene** (nach der
`do_map`-Entscheidung, **vor** `if do_map:`), gated über `object_detect_stride`
— also **unabhängig** davon, ob der Mapper auf diesen KF läuft. `images`, `depths`,
`poses`, `intrinsic` liegen auf jedem Tracker-KF vor (Pose-Override greift ebenfalls
schon hier). Die Segmentierung fürs Dynamic-Masking bleibt davon getrennt **im**
`if do_map:`-Block (nur Mapper-Loss).

Module (Registry-Factory wie bei Selektoren/Segmentierung):

| Datei | Inhalt |
|---|---|
| `scripts/vings_utils/detector_base.py` | `ObjectDetectorBase`, `Detection` (+ `angle`), COCO-Klassen + Farben, `boxes_to_detections`, `obb_to_detections` |
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

## Evaluation (Genauigkeit) — `scripts/eval/object_eval.py`

Misst Detektions- und 3D-Lokalisierungsgüte gegen UAVScenes-Referenzdaten
(für den BA-Abschnitt „Genauigkeit der 3D-Lokalisierung dynamischer Objekte").
Zwei Teile, `--part a|b|both`:

- **Teil A — 2D (mAP/IoU).** `detections_per_frame.csv` (Boxen, Voll-Res) vs.
  UAVScenes-Semantikmasken (`interval5_CAM_label`, Sedan=20/Truck=24). 2D-Instanzen
  via Connected-Components (UAVScenes liefert öffentlich **keine** Instanz-IDs,
  nur Semantik, nur jeder 5. Frame). COCO-Style AP@.5 / AP@[.5:.95] + mean-IoU,
  primär klassenagnostisch „vehicle". Join über `t_sec`-Timestamp.
- **Teil B — 3D (Pos/Yaw/Größe).** `objects_droid.csv` vs. LiDAR-Pseudo-GT, über
  Frames per 3D-NN fusioniert, NN-Match (Gate 5 m) → Positions- (m), Yaw- (deg,
  mod π) und Größen-Fehler. Liegt der Lauf im metrischen GT-Frame (interval1 =
  `ext_poses` + LiDAR), ist **kein** `sim3_unwarp` nötig. **Zwei Referenzquellen
  (`--ref-source`):**
  - `cam_mask` — 2D-Semantikmaske → sparse LiDAR-Tiefe → GT-Pose (jeder Maskenpixel).
  - `lidar_label` (**empfohlen, sauberer**) — direkt **gelabelte LiDAR-Punkte**
    (`interval5_LIDAR_label.zip`, per-Punkt-Klasse zeilengleich zur LiDAR-XYZ-Datei),
    pro Frame **3D-Clustering** (trennt Instanzen, da nur semantisch) → GT-Pose.
    Dichter & rauschärmer — die 2D-Maskenprojektion verfälscht v.a. Yaw/Größe.

```bash
# GT laden (einmalig): interval5_CAM_label.zip (1.5G, 2D-Masken) + interval5_LIDAR_label.zip (101M, 3D-Ref)
#   -> ~/Dokumente/datasets/uavscenes/{interval1_amtown03_labels, amtown03_lidar_labels}/
python scripts/run_experiment.py configs/local/object_detect/interval1_objects_eval.yaml   # OBB-Lauf
python scripts/eval/object_eval.py --rundir output/exp_interval1_objects_eval/<ts>/ \
       --ref-source lidar_label          # Teil A+B; object_eval_2d.json / object_eval_3d.json (+Plots)
python scripts/eval/object_eval.py --selftest   # synthetische IoU/AP/Geometrie-Checks
```

**Ergebnisse interval1_AMtown03** (YOLO26-OBB, `lidar_label`-Ref, 5/10 Referenz
gematcht): **Position median 3.3 m**, **Yaw median 14.8°**, **Größe long 0.89 m
(23 %)/lat 0.75 m (33 %)**, vert schlecht (Mono-Nadir löst Höhe nicht auf). 2D:
OBB AP@.5≈0.27 / mIoU(TP)≈0.80 — niedriger Recall (DOTA-nano schwach auf winzigen
Nadir-Autos; aerial-spezialisiertes `yolov8s-visdrone` erreicht AP@.5≈0.71, hat
aber kein Heading). **Referenz ist geschätzt** (LiDAR-dicht, kein Vermessungs-GT)
— in der Arbeit so deklarieren. Precision/Recall gegen die spärliche interval5-GT
sind unsicher; aussagekräftig ist der **Lage-/Yaw-/Größenfehler auf den Matches**.

**2D auf weiteren Sequenzen ohne Voll-Lauf** (`scripts/eval/detect_on_frames.py`):
wendet denselben Detektor standalone auf alle annotierten cam_left-Frames an →
`detections_per_frame.csv` (+`det_camstamp.txt`) → `object_eval.py --part a`.
Genutzt für AMvalley03/HKairport03/HKisland03 (dort nur 2D — DJI-Posen + .bin-LiDAR
machen 3D unzuverlässig).

## Bekannte Grenzen
- **Domäne.** Default ist jetzt **YOLO26-OBB (DOTA-v1)** — die richtige Wahl für
  Nadir-Aerial, da DOTA Overhead-Domäne ist und der OBB-Winkel das Heading
  liefert (interval1-verifiziert, Fahrzeuge conf bis 0.81). DOTA produziert auf
  dieser Domäne aber auch Klassen-Rauschen (vereinzelt plane/ship/helicopter) —
  `classes: [9, 10]` (large/small vehicle) filtert das weg. Bei abweichender
  GSD: kurzer Finetune auf gelabelten interval1/amtown03-Frames (analog VisDrone).
  Für rein achsenparallele Detektion bleibt `model: yolov8s-visdrone` (oder COCO
  `yolov8n`, `classes: [2,5,7]`) ein drop-in — die Factory lädt jedes kompatible
  Gewicht, `detect()` erkennt OBB- vs. Box-Ausgabe automatisch.
- **Heading ist mod π.** Auch der OBB-Yaw kennt kein vorne/hinten (eine Achse ist
  vorzeichenfrei). Auflösbar nur über Bewegungs-Heading (z.B. ein 2D-Tracker wie
  ByteTrack mit Velocity-Vektor) — aktuell nicht implementiert.
- **Bewegte Objekte** (fahrende Autos) erscheinen an mehreren Weltpositionen →
  mehrere/verschmierte Tracks. Aktuell: statische Objekte angenommen. Späterer
  Filter über das Dynamic-Masking (`use_dynamic`) denkbar.
- **`assoc_radius` ist gauge-frei** — metrisch sauberes Re-Merge erst mit dem
  späteren Sim3-Schritt.
- **VRAM** — Detektor default auf `device: cpu`; lazy-load; kleines Modell
  (yolov8n/rtdetr-l). Läufe seriell.
