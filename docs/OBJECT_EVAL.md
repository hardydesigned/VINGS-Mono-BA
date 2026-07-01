# Genauigkeit der Objekt-Detektion + 3D-Lokalisierung — Methodik & Ergebnisse

## Worum geht es (in einfachen Worten)

VINGS erkennt in jedem Keyframe Fahrzeuge (2D-Box) und trägt sie als **3D-Objekt
in die Rekonstruktion ein** (Position, Ausrichtung, Größe). Dieser Abschnitt misst
zwei Dinge:

1. **Erkennt der Detektor die Fahrzeuge zuverlässig?** → 2D-Metriken **mAP** und
   **IoU** (wie gut sitzt die erkannte Box auf dem echten Fahrzeug im Bild).
2. **Wird das Objekt am richtigen Ort in 3D eingezeichnet?** → **Positionsfehler
   (m)**, **Ausrichtungsfehler (Grad)** und **Größenfehler (m)** gegenüber einer
   Referenz, die möglichst nah an der Realität ist.

Die zentrale Frage der Arbeit ist Punkt 2: **„Wie nah ist das eingezeichnete
Objekt an der Wirklichkeit?"** Antwort (bester Fall): **~3 m Position, ~15° Yaw**.

Kurz zum „Woher kommt die Wahrheit": es gibt keine amtlichen 3D-Fahrzeug-
Annotationen. Wir bauen eine **Referenz aus LiDAR + GT-Kameraposen**: die
gelabelten LiDAR-Punkte eines Fahrzeugs, in die Welt projiziert, ergeben seinen
echten Ort/Größe/Ausrichtung. Das ist dicht und sauber, aber selbst *geschätzt*
(kein Vermessungs-GT) — das muss man so deklarieren.

---

## Datengrundlage

| Rolle | Quelle | Inhalt |
|---|---|---|
| Sequenz + LiDAR + GT-Posen | `interval1_AMtown03` (MARS-LVIG) | 5599 Frames, per-Frame LiDAR (`.txt` xyz), GT-Posen (UAVScenes, ATE 0,07 m) |
| 2D-GT (Masken) | `interval5_CAM_label.zip` (HF, 1,5 GB) | Semantik-Masken, jeder 5. Frame, Cityscapes-IDs **Sedan=20, Truck=24** |
| 3D-GT (LiDAR-Labels) | `interval5_LIDAR_label.zip` (HF, 101 MB) | **per-Punkt-Klasse**, zeilengleich zur LiDAR-XYZ-Datei |

Wichtig: `interval1`-Label-Zips existieren **nicht** auf HF — die `interval5`-
Labels (jeder 5. Frame) sind dieselben Kamera-Frames wie in interval1 bei Stride 5,
gematcht wird über den **Timestamp** (Dateiname), nicht über den Index.

Die Sekundärsequenzen `AMvalley03 / HKairport03 / HKisland03` haben Labels + LiDAR,
aber nur **DJI-Posen** (~10 % Scale-Verzerrung) → dort nur 2D ausgewertet, kein 3D
(DJI-Posen würden die Referenz selbst verfälschen).

---

## Was die Pipeline ausgibt (Vorhersage)

Ein Lauf mit `detect_objects: true` schreibt:
- `detections_per_frame.csv` — pro Detektion pro KF: 2D-Box, conf, class, Tiefe,
  3D-Weltposition. (Basis für **2D**.)
- `objects_droid.csv` — pro fusioniertem Objekt: 3D-Position, Quaternion (Yaw),
  Größe. (Basis für **3D**.)

Weil der Lauf mit **GT-`ext_poses` + LiDAR** läuft, liegt `objects_droid.csv`
bereits im **metrischen GT-Weltframe** — die LiDAR-Referenz (gleiche GT-Posen)
ist direkt vergleichbar, **kein `sim3_unwarp` nötig**.

---

## Methodik 2D (mAP / IoU) — `object_eval.py --part a`

1. **GT-Boxen** aus den Semantik-Masken: pro Fahrzeugklasse (20/24) werden
   zusammenhängende Regionen (Connected-Components) zu Instanzen → Bounding-Box.
   (UAVScenes liefert öffentlich keine Instanz-IDs → CC ist die Näherung;
   Schwachpunkt = dicht geparkte Fahrzeuge verschmelzen.)
2. **Zuordnung Frame↔GT** über den Timestamp; nur Frames werten, die Detektion
   **und** Label haben. Detektions- und GT-Boxen werden auf `[0,1]` normiert
   (gleiches FOV, unterschiedliche Auflösung erlaubt via `--det-res`).
3. **Matching pro Frame** (greedy, höchste IoU zuerst): eine Detektion ist TP,
   wenn `IoU ≥ 0.5` mit einer noch freien GT-Box.
4. **AP** = Fläche unter der Precision-Recall-Kurve (All-Point-Interpolation,
   COCO-Stil), berichtet als **AP@.5** und **AP@[.5:.95]** (Mittel über
   IoU-Schwellen 0.5…0.95). **mIoU(TP)** = mittlere IoU der Treffer.
5. Primär **klassenagnostisch „vehicle"** (Detektor-Taxonomie DOTA/VisDrone/COCO
   ≠ GT-Taxonomie), optional per Klasse.

Ohne Voll-Lauf (nur 2D): `detect_on_frames.py` wendet denselben Detektor
standalone auf **alle** annotierten Frames an → `detections_per_frame.csv`.

---

## Methodik 3D (Position / Yaw / Größe) — `object_eval.py --part b`

**Referenz-Aufbau** (`--ref-source lidar_label`, empfohlen):
1. Pro annotiertem Frame: LiDAR-Punkte mit Fahrzeug-Label (20/24) → über die
   GT-Pose in die Welt transformiert (`X=−ly, Y=−lz, Z=lx`, `p_w = c2w·X_cam`;
   identische Pinhole-Konvention wie `object_tracker`).
2. **3D-Clustering** pro Klasse und Frame (BFS über Radius-Graph, 2 m) trennt
   mehrere Fahrzeuge (Labels sind nur semantisch, nicht instanzweise).
3. **Fusion über Frames** per 3D-Nächster-Nachbar (Radius 3 m, wie der
   `object_tracker`): jedes Cluster → Referenz-Objekt mit Position (Median der
   Frame-Centroide), Yaw (Kreismittel), Größe (Median-Extent via PCA).
4. Nur im **Zeitfenster des Laufs** (faire Precision/Recall).

(Alternative `--ref-source cam_mask`: 2D-Maske → sparse LiDAR-Tiefe → Pose. Gleiche
Position, aber verrauschter Yaw/Größe → nur als Gegenprobe.)

**Vergleich Vorhersage ↔ Referenz:**
- **Match**: 3D-Nächster-Nachbar, Gate 5 m.
- **Positionsfehler** = euklid. Abstand der Zentren (mean/median/rmse).
- **Yaw-Fehler** = Winkeldifferenz **mod π** (Fahrzeug-180°-Mehrdeutigkeit).
- **Größenfehler** = per-Achse |Δ| von [Länge, Breite, Höhe] (abs. + relativ).
- **Recall** = Anteil der Referenzfahrzeuge, die einen Treffer im Gate haben
  (= „wie viele echte Fahrzeuge wurden platziert").

**Wichtig zur Einordnung:** Die Referenz ist geschätzt und nur jeder 5. Frame
annotiert → **Precision/Recall gegen diese spärliche GT sind unsicher**.
Belastbar ist der **Lage-/Yaw-/Größenfehler auf den Matches**.

---

## Architektur-Fakten (relevant für die Selector-Frage)

- Objekt-Detektion läuft in `run.py:317` (`run_detection`) **vor** dem
  Mapping-Gate (`if do_map:`). Sie ist damit **vom Frame-Selector entkoppelt** —
  post-Tracker-Selektoren (nurbs/vista/…) ändern nur das Mapping/PSNR, nicht die
  Objekte.
- Auf interval1 sind **Posen = GT** und **Tiefe = LiDAR**, beide *frame-lokal*.
  Deshalb hängt die Objektlokalisierung nicht von der Tracking-Dichte ab; ein
  Tracker-Frame-Gate (gate_a) oder ein größerer Detektions-Stride reduzieren nur
  die **Abdeckung** (Recall), nicht die **Genauigkeit**.

---

## Ergebnisse

### Kernergebnis — 3D-Lokalisierung (interval1_AMtown03, YOLO26-OBB, LiDAR-Ref)
| Metrik | median | mean | rmse |
|---|---|---|---|
| **Position** | **3,28 m** | 3,09 | 3,14 |
| **Yaw** (mod π) | **14,8°** | 30,0 | — |
| Größe Länge | 0,89 m (23 %) | — | — |
| Größe Breite | 0,75 m (33 %) | — | — |
| Größe Höhe | unbrauchbar (Nadir-Mono löst Höhe nicht auf) | — | — |

5 von 10 Referenzfahrzeugen im Zeitfenster gematcht (Recall 0,50).

### OD-Modell-Vergleich — 2D (AMtown03)
| Detektor | AP@.5 | AP@[.5:.95] | mIoU | P | R |
|---|---|---|---|---|---|
| **yolov8s-visdrone** | **0,610** | **0,444** | 0,87 | 0,64 | 0,66 |
| yolo26n-obb (DOTA) | 0,354 | 0,189 | 0,78 | 0,74 | 0,39 |
| rtdetr-l (COCO) | 0,332 | 0,238 | 0,85 | 0,17 | 0,58 |
| yolov8n (COCO) | 0,171 | 0,117 | 0,85 | 0,63 | 0,21 |

### OD-Modell-Vergleich — 3D (interval1)
| Detektor | Recall | Pos median | Yaw median |
|---|---|---|---|
| yolo26n-obb | 0,50 | **3,28 m** | **14,8°** |
| yolov8s-visdrone | 0,70 | 4,36 m | 43,8° |
| rtdetr-l | **0,90** | 4,15 m | 28,9° |

→ Kein Einzelsieger: **VisDrone** bestes 2D + guter Recall; **rtdetr** max Recall
(aber P=0,17, viele FP); **OBB** klar bester Yaw (echtes Heading). PSNR aller
Läufe ~19,6 ⇒ Detektor ändert das Mapping nicht.

### 2D auf weiteren Szenen
| Szene | AP@.5 | mIoU | P | R |
|---|---|---|---|---|
| AMtown03 | 0,354 | 0,78 | 0,74 | 0,39 |
| AMvalley03 | 0,469 | 0,74 | 0,86 | 0,52 |
| HKairport03 | 0,553 | 0,79 | 0,77 | 0,60 |
| HKisland03 | 0,031 | 0,62 | 0,11 | 0,12 |

### Frame-Selector & Detektions-Stride
- **Post-Tracker-Selektor:** kein Einfluss (entkoppelt, s.o.).
- **`object_detect_stride` n=1→10:** Objekte 140→26, Recall 0,50→0,30, Position
  bleibt ~3 m. → Recall/Compute-Trade-off, **kein Genauigkeits-Trade-off**.

---

## Grenzen / ehrliche Einordnung
- Referenz ist **geschätzt** (LiDAR-dicht, kein Vermessungs-GT), nur jeder 5.
  Frame → P/R unsicher; Lage-/Yaw-/Größenfehler auf Matches sind belastbar.
- 2D-Instanzen via Connected-Components → dicht geparkte Fahrzeuge verschmelzen.
- Objekthöhe aus Nadir-Mono-Tiefe nicht auflösbar (Größe Höhe unbrauchbar).

---

## Reproduktion
```bash
# GT einmalig laden (HF): interval5_CAM_label.zip (1,5G) + interval5_LIDAR_label.zip (101M)
#   -> ~/Dokumente/datasets/uavscenes/{interval1_amtown03_labels, amtown03_lidar_labels}/

# 3D + 2D (Hauptergebnis):
python scripts/run_experiment.py configs/local/object_detect/interval1_objects_eval.yaml
python scripts/eval/object_eval.py --rundir output/exp_interval1_objects_eval/<ts>/ --ref-source lidar_label

# OD-Modell-Vergleich 2D (standalone, alle Frames):
python scripts/eval/detect_on_frames.py --cam-dir <cam> --label-dir <labels> --out <out> \
       --kind yolo --model yolov8s-visdrone --ckpt ckpts/yolov8s-visdrone.pt --classes 3 4 5 8
python scripts/eval/object_eval.py --part a --rundir <out> --label-dir <labels> \
       --camstamp <out>/det_camstamp.txt --det-res 2448 2048

# OD-Modell-Vergleich 3D:  configs/local/object_detect/interval1_objects_eval_{visdrone,rtdetr}.yaml
# Stride-Sweep (ohne Neu-Lauf):  python scripts/eval/refuse_stride.py --rundir <run> --strides 1 2 3 5 10
# Sanity:  python scripts/eval/object_eval.py --selftest
```

Artefakte pro Lauf: `object_eval_2d.json` (+`pr_curve.png`), `object_eval_3d.json`
(+`object_eval_3d.png`: Fehler-Histogramme + BEV-Scatter), `overlay_gt_vs_det.png`
(GT grün vs. Detektion rot). Konsolidiert:
`output/exp_interval1_objects_eval/EVAL_SUMMARY.md`.
