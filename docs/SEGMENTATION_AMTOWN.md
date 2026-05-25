# Semantic Segmentation + Dynamic Objects

Erkenntnisse zur Dynamic-Object-Erkennung in VINGS-Mono + zur UAVScenes-GT-Labels-
Integration für AMtown03.

## VINGS' built-in `use_dynamic` ist DEAD CODE

`scripts/dynamic/dynamic_utils.py` enthält eine `DynamicModel`-Klasse die FastSAM
nutzt — aber **sie ist nirgends im run.py-Mainloop aufgerufen**. Der Code wird nur
über `__main__`-Block standalone laufen lassen.

Konkret:
- `use_dynamic: true` in der Config setzt nur Visualisierungs-Flags (`vis_utils.py:155`)
- Hardcoded-Pfad `/data/wuke/workspace/FastSAM/` — lokal nicht installiert
- Pipeline: pre-compute SAM-Annotations OFFLINE → save → load via `get_anns_load`
- Detection ist **loss-basiert**, class-agnostic (high-error-regions = bewegliche
  Objekte). Keine "car", "person" labels.

**Bedeutet**: Wer einen "VINGS-eingebauten dynamic remover" erwartet, bekommt
nichts. Für echte dynamic-removal beim Training braucht's Custom-Code oder externe
Masken die vor dem Mapper appliziert werden.

## Alternative: UAVScenes-Labels nutzen

UAVScenes (Wang et al. 2025) hat MARS-LVIG mit **per-frame Semantic-Annotations**
nachgelabelt. Für AMtown03 sind 1120 Frames (interval5 = jeder 5te) annotiert mit
19 Klassen, darunter Sedan + Truck als Dynamic-Klassen.

### Download

```bash
curl -L -o /tmp/uavs.zip \
  "https://huggingface.co/datasets/sijieaaa/UAVScenes/resolve/main/interval5_CAM_label.zip"
# 1.4 GB, enthält alle 20 MARS-LVIG-Sequenzen
unzip -q /tmp/uavs.zip "interval5_CAM_label/interval5_AMtown03/*" -d /home/philipp/Dokumente/datasets/uavscenes/amtown03_labels/
```

Folder-Struktur:
```
interval5_AMtown03/
├── interval5_CAM_label_color/     # RGB-color-rendering der Masken (für Vis)
│   └── 1658131847.149322787.png
└── interval5_CAM_label_id/        # uint8 PNG mit Class-IDs
    └── 1658131847.149322787.png
```

Filename ist der **Bag-Timestamp** der originalen Frame.

### Class-ID-Mapping — KRITISCHER BEFUND

Paper Tab. S9 listet 19 Klassen mit IDs 0-18 (Sedan=17, Truck=18). **In den
echten id-PNG-Files wird aber Cityscapes-style-IDs (0-25+) verwendet**, nicht
die Paper-IDs. Verifiziert durch Color+Pixel-Distribution-Analyse:

| ID-File | Klasse | BGR Avg-Color | Vorkommen AMtown03 |
|---|---|---|---|
| 0 | Background | [0, 0, 0] | weit |
| 1 | Roof / Gebäude | [32, 11, 119] | viel |
| 2 | DirtRoad / Lehmweg | [180, 165, 180] | mittel |
| 3 | PavedRoad / Asphalt | [128, 64, 128] | mittel |
| 5, 6 | minor (Bridge, Container?) | versch. | wenig |
| 13 | **Vegetation/GreenField** | [35, 142, 107] | **dominant** |
| 14 | TransparentRoof | [140, 180, 210] | wenig |
| 15 | Traffic-Sign | [0, 220, 220] | wenig |
| 16 | Pole | [153, 153, 153] | wenig |
| 17 | (unklar) | [90, 0, 0] | wenig |
| 19 | PavedWalk | [232, 35, 244] | mittel |
| **20** | **SEDAN** | **[142, 0, 0]** | **774/1120 frames** |
| **24** | **TRUCK** | **[70, 0, 0]** | **262/1120 frames** |
| 25 | (Person?) | versch. | wenig |

**Beachte**: 774/1120 = **69% der Frames haben Sedans**. Vorher hatten wir
fälschlich Sedan=17, Truck=18 angenommen (Paper-Tab) und kamen auf 352/0 — das
war komplett falsch.

Die `_label_color`-PNGs sind die direkt visualisierbare RGB-Rendering davon.

## Verfügbare Tools / Workflows

### Mask-Overlay-Video erstellen
```bash
python /tmp/amtown_mask_video_v2.py
# Output: ~/Dokumente/datasets/amtown03/amtown03_mask_overlay_v2.mp4
# Format: GT + class-mask semi-transparent overlay + class-legend, 230 MB, 1120 frames @ 10 fps
# Vehicle-pixel-count pro Frame als Text-Overlay
```

### Pred-vs-GT-Overlay-Video (mit Bounding-Boxes für Vehicles)
```bash
python /tmp/amtown_500f_video.py
# 3 columns: GT | VINGS-Pred | Pred + Seg-Overlay
# Sedan/Truck-Detection als Cyan-Bounding-Boxes mit Class-Label
# Output: amtown03_500f_pred_overlay.mp4
```

### Welche Sequenz hat die meisten Vehicles?
Top-Vehicle-Frames in AMtown03 sind um Bag-Index 3160 (Vehicle-rich Cruise).
Frame 632 in den Labels (filename 1658132131.138881355.png) hat 32685 Sedan-Pixel.

## Three.js Dynamic-Reconstruction-Pipeline (Konzept, nicht implementiert)

User-Vorschlag: statische VINGS-Reko + dynamische Autos als Three.js-Objekte
über die Map gelegt. Pipeline-Skizze:

### 1. Statische Reko mit Dynamic-Mask
VINGS-Mono trainieren mit **dynamic_mask appliziert**: Sedan/Truck-Pixel im
RGB-Loss ausgeschlossen → saubere Background-Gaussians ohne "verschmierte" Autos.
Würde erfordern: Mask-Loader in dataset + im Mapper-Loss `mask * (gt - pred)`.

### 2. Dynamic-Object-3D-Extraction
Pro Frame:
```python
# pixel (u,v) in vehicle_mask → 3D via depth + cam-pose
depth_at_pixel = render_depth[v, u]   # vom VINGS-Mapper
cam_coord = depth_at_pixel * K_inv @ [u, v, 1]
world_coord = T_c2w @ [cam_coord, 1]
```
→ pro Auto: 3D-Punkt-Cluster → Centroid + Bounding-Box.

### 3. Inter-Frame Track-IDs
UAVScenes hat Instance-Annotations (laut Paper Tab. 2 = 280k Instances mit
Track-IDs), aber **nicht in dem `interval5_CAM_label.zip`** das wir haben —
vermutlich im `interval1_CAM_label.zip` (6.93 GB). Alternativ:
- **SORT** auf den Semantic-Masken: `cv2.connectedComponentsWithStats(idmap == 20)`
  → BBox-pro-Auto → IoU-Match zwischen Frames → Track-IDs
- Funktioniert gut bei langsamem Drohnen-Movement, ~2h Aufwand
- **DeepSORT** mit Appearance-Features wenn Occlusion-Robustness nötig

### 4. Three.js-Render
- Statische Map: Gaussian-Splat oder Mesh (export von VINGS via PLY → Three.js-Loader)
- Dynamische Autos: 3D-Box oder generic car-mesh mit interpolierter Trajektorie
- Kamera-Animation = DJI-RTK-Pose-Sequenz

### Caveats
- VINGS-Depth ist noisy → 3D-Position pro Frame jittert → Kalman-Filter-Smoothing
- VINGS rendert depth bei jeder Mapper-Iteration; muss aus dem Output extrahiert
  werden (aktuell nicht direkt gespeichert, müsste Loader-Patch)
- Auto-Detection-Rate hängt von Mask-Quality ab; UAVScenes-GT ist sauberer als
  jedes Inference-Modell auf Drohnen-Bildern

## Alternative für Realtime / eigene Drohnen-Aufnahmen

UAVScenes-Labels sind nur für die offiziellen MARS-Sequenzen verfügbar. Für eigene
Drohnen-Aufnahmen müsste man:
- **Mask2Former-Cityscapes** (HuggingFace transformers) — kennt "car", "person",
  "truck", ~50ms/frame auf RTX 4080
- **YOLO11-seg** — schneller, gut für Vehicle-Detection
- **SAM2** für instance-segmentation mit Track-ID

Diese laufen aber NICHT auf MARS-Aerial-Nadir-Domain ohne Fine-Tuning. UAVScenes-
GT ist deshalb für AMtown03/HKairport-Tests immer der bessere Stack.
