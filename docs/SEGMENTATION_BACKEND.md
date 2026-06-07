# Segmentierung & Dynamic-Object-Masking

## Worum geht's (in einfach)

Auf Stadt-/Straßenszenen bewegen sich Dinge: Autos, Fußgänger. Wenn der Mapper
versucht, ein bewegtes Auto als feste 3D-Fläche zu rekonstruieren, schmiert er
die Szene voll mit Geistern. Idee: **bewegte Objekte erkennen und beim Mapping
ignorieren.**

Wie? In zwei Schritten:

1. **Segmentieren** — ein Bildmodell (FastSAM) zerlegt jedes Keyframe in viele
   einzelne Objekt-Masken („das ist ein Ding, das ist noch ein Ding").
2. **Dynamische rausfiltern** — wir vergleichen das gerenderte Bild (was die
   Karte vorhersagt) mit dem echten Kamerabild. Wo der Fehler bei einem Segment
   besonders hoch ist, ist das Objekt vermutlich an einer anderen Stelle als die
   Karte denkt → **dynamisch** → seine Pixel fliegen aus dem Loss raus.

Das Segmentierungsmodell ist **austauschbar**: FastSAM, SAM2.1 oder SAM3 — ein
Wort in der Config (`segmentation.kind`), kein Code-Umbau.

| kind | Modell | Status | Idee |
|---|---|---|---|
| `fastsam` | FastSAM-x/-s (YOLOv8-seg) | live | prompt-frei „everything", schnell |
| `sam2` | SAM2.1 (t/s/b/l) via ultralytics | live | echtes SAM2.1, qualitativ über FastSAM |
| `sam3` | SAM3 (Concept) via ultralytics | **Code da, Weights gated** | text-getrieben: segmentiert direkt „car/person/…" |

## Einschalten (FastSAM)

```yaml
use_dynamic: true
segmentation:
  kind: fastsam            # none | fastsam | sam2 | sam3
  model: FastSAM-x         # FastSAM-x (genau) | FastSAM-s (schnell)
  ckpt_path: ckpts/FastSAM-x.pt
  conf: 0.4
  iou: 0.9
  imgsz: 512
  device: cuda:0
  min_area_px: 0           # Masken kleiner als X Pixel verwerfen (0 = alle behalten)
  dyn_loss_quantile: 0.9   # Pixel oberhalb dieses Error-Quantils = "high loss"
  dyn_high_rate: 0.2       # Segment dynamisch wenn >20% high-loss-Pixel ...
  dyn_mean_loss: 0.002     # ... UND mittlerer Error > diesem Floor
```

Beispiel: `configs/local/dynamic/amtown03_s1000_400f_dynamic_fastsam.yaml`.

## Einschalten (SAM2.1 — empfohlen, besser als FastSAM)

```yaml
use_dynamic: true
segmentation:
  kind: sam2               # echtes SAM2.1 via ultralytics (gleiche API wie FastSAM)
  model: sam2.1_b          # sam2.1_t | _s | _b (base_plus) | _l (large)
  ckpt_path: ckpts/sam2.1_b.pt   # 162 MB, auto-download via ultralytics-Asset
  conf: 0.4
  iou: 0.9
  imgsz: 1024              # SAM2 native; höher als FastSAM (512) -> mehr VRAM
  device: cuda:0
  min_area_px: 0
  dyn_loss_quantile: 0.9
  dyn_high_rate: 0.2
  dyn_mean_loss: 0.002
```

Beispiel: `configs/local/dynamic/amtown03_s1000_400f_dynamic_sam2.yaml`.
Größenwahl ist der VRAM↔Qualität-Hebel: `sam2.1_t/_s` für knappen VRAM,
`sam2.1_l` für beste Masken. Weights laden auto via ultralytics
(`ckpts/sam2.1_b.pt`), oder direkt:
`curl -fL -o ckpts/sam2.1_b.pt https://github.com/ultralytics/assets/releases/download/v8.4.0/sam2.1_b.pt`
(161935802 Bytes — Größe prüfen, der Endpunkt bricht gern mittendrin ab).

Wenn `use_dynamic: false` oder `segmentation.kind: none` (Default), passiert
nichts — der alte Pfad bleibt unverändert.

## Setup (einmalig)

```bash
conda activate vings
pip install ultralytics            # liefert FastSAM
# Gewichte landen unter ckpts/ (Repo-Konvention, wie droid.pth):
python scripts/vings_utils/fastsam_backend.py   # lädt FastSAM-x.pt beim ersten Lauf
```

## Wie es im Code läuft (technisch)

```
run.py (pro Keyframe-Batch viz_out):
  DynamicModel.get_anns_raw(rgb)  ──► viz_out['sam_anns'] = [(K,H,W) bool, ...]   # SAM 1× pro KF
        │  (FastSamBackend.segment → ultralytics FastSAM, Masken auf CPU)
        ▼
mapper.run(viz_out)  →  train_once_gaussian(batch=viz_out):
  pro Trainings-Iter, gewähltes KF curr_id:
    compute_dynamic_mask(batch['sam_anns'][curr_id], gt_rgb, pred_rgb)  # billig
        → gt_dict['dynamic_mask']  (H,W) bool, True = dynamisch
        ▼
get_loss(cfg, pred_dict, gt_dict):
  valid_mask &= ~dynamic_mask     # dynamische Pixel raus aus rgb/normal/depth/dist
```

Teure SAM-Inferenz läuft **genau einmal pro Keyframe** (nicht pro Iteration);
die Masken fließen über den bestehenden `viz_out → batch`-Kanal wie `images`.
`compute_dynamic_mask` ist nur Elementwise-Arithmetik über die gecachten
Segmente und daher pro Iter unkritisch.

**Dynamic-Detection (`scripts/dynamic/dynamic_utils.py::compute_dynamic_mask`):**
Per-Pixel-Fehler `e = L1(pred,gt) · (1 − SSIM(pred,gt))`. Ein Segment gilt als
dynamisch, wenn mehr als `dyn_high_rate` seiner Pixel über dem
`dyn_loss_quantile`-Quantil liegen **und** sein mittlerer Fehler `dyn_mean_loss`
übersteigt. `sky_mask` bleibt unangetastet (dynamische Pixel werden nicht als
Himmel gewertet). Bei 0 Segmenten → leere Maske → kein Masking (sicherer Default).

## SAM3 (Code da, Weights gated)

`scripts/vings_utils/sam3_backend.py` ist ausgebaut — aber **anders** als
FastSAM/SAM2: SAM3 („Segment Anything with Concepts", Meta, Nov 2025) ist
*text-getrieben*. Statt prompt-frei „everything" zu segmentieren und dann
high-Error-Segmente zu filtern, gibt man SAM3 eine Klassen-Liste
(`classes: [car, truck, bus, person, bicycle, motorcycle]`) und bekommt direkt
die Masken dieser Konzepte. Für Dynamic-Masking ist das der natürlichere Weg
(die zurückgegebenen Masken *sind* die Kandidaten); der bestehende
high-Error-Filter komponiert obendrauf (geparktes Auto = kein High-Error,
fahrendes = schon). Inferenz läuft über `SAM3SemanticPredictor` (der einfache
`ultralytics.SAM(...)`-Wrapper kann nur Punkt/Box-Prompts, kein Text).

**Blocker: die Weights sind nicht frei ladbar (Stand 2026-06-05).** SAM3 ist
*kein* ultralytics-Auto-Download-Asset (anders als `sam2.1_*.pt`), und Metas
HF-Repo `facebook/sam3` ist `gated: manual` (HTTP 401 GatedRepo) und liefert
zudem das *transformers*-Format, nicht das ultralytics-`.pt`-Layout. Bis ein
Checkpoint unter `ckpt_path` liegt, wirft `_ensure_model()` einen klaren Fehler.

Zwei Wege an die Weights (Entscheidung nötig):
- **Route A (ultralytics .pt, was das Backend erwartet):** Metas rohen
  SAM3-Checkpoint (`detector.`/`tracker.`-Keys) vom offiziellen
  facebookresearch/sam3-Release ziehen → `ckpts/sam3.pt`.
- **Route B (HF transformers):** Lizenz auf huggingface.co/facebook/sam3
  akzeptieren (manuelle Freigabe), `pip install huggingface_hub`,
  `huggingface-cli login` mit berechtigtem Token, dann laden — und entweder ins
  ultralytics-`.pt` konvertieren oder das Backend auf die transformers-`Sam3`-API
  umstellen (anderer Inferenz-Pfad als der aktuell verdrahtete).

Mapper, Loss und Run-Loop bleiben in jedem Fall unberührt — das ist der Sinn des
`SegmentationBackend`-Vertrags (`scripts/vings_utils/segmentation_base.py`).

## Schnelltests

```bash
python scripts/vings_utils/fastsam_backend.py     # FastSAM: #Masken + Overlay
python scripts/vings_utils/sam2_backend.py        # SAM2.1: #Masken + Overlay
python scripts/vings_utils/sam3_backend.py        # SAM3: zeigt Weights-Blocker (gated)
python scripts/dynamic/dynamic_utils.py           # Segment + Dynamic-Maske + K=0-Härtung
```

## Architektur-Dateien

| Datei | Rolle |
|---|---|
| `scripts/vings_utils/segmentation_base.py` | `SegmentationBackend`-ABC = der Swap-Vertrag |
| `scripts/vings_utils/segmentation_factory.py` | Registry + `make_segmentation_backend(cfg, device)` |
| `scripts/vings_utils/fastsam_backend.py` | FastSAM via ultralytics (`kind: fastsam`) |
| `scripts/vings_utils/sam2_backend.py` | SAM2.1 via ultralytics (`kind: sam2`) |
| `scripts/vings_utils/sam3_backend.py` | SAM3 text-grounded via ultralytics (`kind: sam3`, Weights gated) |
| `scripts/dynamic/dynamic_utils.py` | `DynamicModel` + `compute_dynamic_mask` |
| `scripts/gaussian/loss_utils.py` | `get_loss`: `valid_mask &= ~dynamic_mask` |
| `scripts/gaussian/gaussian_base.py` | `train_once_gaussian`: baut `gt_dict['dynamic_mask']` |
| `scripts/run.py` | instanziiert `DynamicModel`, segmentiert `viz_out` |
