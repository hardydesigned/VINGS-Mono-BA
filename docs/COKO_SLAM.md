# Coko-SLAM Keyframe-Selektor

Referenz: Li M.M.Q., Lajoie P.-Y., Liu J., Beltrame G. — *"Compact Keyframe-Optimized
Multi-Agent Gaussian Splatting SLAM"* (Coko-SLAM), arXiv:2604.00804, April 2026.
Code: <https://github.com/lemonci/coko-slam>.

Diese Datei dokumentiert ausschließlich den **Keyframe-Selektor** aus Sektion 3.1
des Papers. Die anderen beiden Beiträge — Multi-Agent-Loop-Closure ohne Initialposen
(Sektion 3.3) und GaussianSPA-Compaction (Sektion 3.2) — sind für VINGS (single-agent)
nicht relevant und werden hier nicht implementiert.

## 1. In ganz einfachen Worten

Stell dir vor, du fährst mit einer Drohne durch eine Stadt und filmst alles. Du
bekommst pro Sekunde z.B. 30 Bilder. **Die meisten dieser Bilder zeigen aber das
Gleiche** — die Drohne bewegt sich nur langsam, und zwei aufeinanderfolgende
Bilder unterscheiden sich kaum.

Wenn du jetzt aus all den Bildern eine 3D-Karte bauen willst, willst du nicht
alle 30 Bilder pro Sekunde benutzen. Du willst nur die "interessanten" Bilder —
also die, die wirklich etwas Neues zeigen. Die nennt man **Keyframes**.

Die Frage ist: wie entscheidet man, ob ein Bild "neu genug" ist?

Coko-SLAM hat einen einfachen, aber smarten Trick: sie geben jedes Bild durch
ein vortrainiertes neuronales Netz (DINOv2), das aus dem Bild eine Art **Steckbrief**
macht — eine Liste von 384 Zahlen, die das Bild zusammenfasst. Bilder, die ähnlich
aussehen, bekommen ähnliche Steckbriefe; Bilder mit anderen Inhalten bekommen
verschiedene Steckbriefe.

Dann ist die Entscheidung trivial: vergleiche den Steckbrief des neuen Bildes
mit den Steckbriefen aller bisherigen Keyframes. Wenn selbst der ähnlichste
alte Keyframe noch deutlich anders ist als das neue Bild, dann ist das neue
Bild interessant genug — also Keyframe. Sonst überspringen.

In Stichworten: **"wenn nichts in meinem Archiv aussieht wie dieses Bild,
ist es ein guter Kandidat fürs Archiv."**

## 2. Was ist ein Feature-Vektor und warum DINOv2?

Bevor wir zur Mathematik kommen, eine Bridge: was bedeutet eigentlich "Steckbrief"?

Im Machine-Learning-Jargon heißt der Steckbrief **Feature-Vektor** oder
**Embedding**. Es ist tatsächlich nichts anderes als eine Liste von Zahlen
(in unserem Fall 384 Stück), die irgendwie das Bild beschreiben. Diese Beschreibung
ist nicht von Menschen ausgedacht — sie kommt aus einem neuronalen Netz, das
genau darauf trainiert wurde, **ähnliche Bilder ähnlich darzustellen**.

Das ist auch der Unterschied zu einem klassischen Fingerabdruck oder Hash: ein
Hash ist eindeutig (zwei verschiedene Bilder → komplett verschiedene Hashes),
ein Feature-Vektor ist **gradual** (zwei ähnliche Bilder → ähnliche Vektoren).
Genau diese graduelle Eigenschaft brauchen wir, weil wir ja messen wollen,
*wie ähnlich* zwei Bilder sind.

**DINOv2** (D**I**stillation with **NO** labels, version 2) ist ein Vision-Transformer
von Meta, der mittels self-supervised learning auf 142 Millionen Internet-Bildern
trainiert wurde — komplett ohne menschliche Labels. Das Modell hat dabei gelernt,
Bilder so zu kodieren, dass semantisch ähnliche Inhalte nahe beieinanderliegen.

Coko-SLAM benutzt die kleinste Variante, **DINOv2-Small (ViT-S/14)**:
- ~22 Millionen Parameter (klein genug für Edge-Devices)
- 384-dimensionale Embeddings
- Robust gegenüber Domain-Shifts (Outdoor, Drohnen, etc.)
- Schnelle Inferenz (~5 ms pro 224×224-Bild auf einer modernen GPU)

Wie messen wir "Ähnlichkeit" zwischen zwei Steckbriefen? Wir ziehen die Listen
voneinander ab, quadrieren die Differenzen, addieren sie auf, und ziehen die
Wurzel — das ist die **euklidische Distanz** (L2-Norm) zwischen zwei Vektoren.
Kleine Distanz = ähnlich, große Distanz = unähnlich.

## 3. Der Algorithmus, Wort für Wort

Hier der Pseudocode aus Paper-Sektion 3.1 (leicht umformuliert für Klarheit):

```
input:  Stream von RGB-Bildern E_1, E_2, E_3, ...
state:  Liste K von bisher akzeptierten Keyframes (anfangs leer)
param:  Akzeptanz-Schwelle α (z.B. 0.4)

for jedes neue Bild E in stream:
    # Schritt 1: Bild durch DINOv2 jagen
    f_new = ϕ(E)                                    # Vektor in ℝ^384
    f_new = f_new / ||f_new||                       # L2-normalisieren

    # Schritt 2: Bootstrap — wenn keine Keyframes da, akzeptiere
    if K ist leer:
        akzeptiere E, hänge f_new an K an
        continue

    # Schritt 3: minimale Distanz zu allen Keyframes berechnen
    d = min_{f_k in K} ||f_new - f_k||₂

    # Schritt 4: Entscheiden
    if d >= α:
        akzeptiere E, hänge f_new an K an    # Bild ist neu genug → Keyframe
    else:
        überspringe E                         # zu ähnlich zu existierenden KFs
```

Schritt für Schritt erklärt:

1. **Feature-Extraktion**: Wir nehmen das neue Bild $E$, geben es durch DINOv2,
   und bekommen einen Vektor $\phi(E)$ mit 384 Zahlen raus. Damit Distanzen
   sinnvoll vergleichbar sind, normalisieren wir den Vektor auf Länge 1
   (L2-Normalisierung). Danach liegt jeder Feature-Vektor auf der Einheitssphäre
   in $\mathbb{R}^{384}$.

2. **Bootstrap**: Wenn noch nichts im "Gedächtnis" ist, kann man nichts vergleichen.
   Erstes Bild wird also immer akzeptiert.

3. **Minimum-Distanz**: Wir berechnen für jeden gespeicherten Feature-Vektor
   $f_k$ die Distanz $\|f_{\text{new}} - f_k\|_2$. Das gibt uns $n$ Distanzwerte.
   Wir nehmen das **Minimum**: das ist die Distanz zum *ähnlichsten* existierenden
   Keyframe. Wenn dieser Wert klein ist, gibt es schon einen sehr ähnlichen
   KF — kein Bedarf für einen neuen. Wenn er groß ist, ist das neue Bild von
   allen bisherigen verschieden.

4. **Entscheidung**: Vergleich mit Threshold $\alpha$. Bei normalisierten
   Vektoren liegt jede paarweise Distanz im Intervall $[0, 2]$:
   - $d \approx 0$ → identische Bilder
   - $d \approx 1.41$ → orthogonale Features (Cosine-Ähnlichkeit 0)
   - $d \approx 2.0$ → antipodale Features (Cosine-Ähnlichkeit −1, in der Praxis selten)

   Typische sinnvolle $\alpha$-Werte: 0.2 (sehr nachgiebig, viele KFs) bis
   0.6 (sehr streng, wenige KFs). Default in unserer Implementation: $\alpha = 0.4$.

## 4. Symbol-Tabelle

| Symbol | Bedeutung | Wo im Code |
|---|---|---|
| $E$ | aktueller Kandidaten-Frame (RGB-Bild) | `rgb` Parameter von `should_accept()` |
| $\phi(\cdot)$ | DINOv2-Feature-Extractor | `self._extract()` |
| $\phi(E) \in \mathbb{R}^{384}$ | L2-normalisierter Feature-Vektor | `feat` Tensor in `should_accept()` |
| $K = \{K_1, \ldots, K_n\}$ | bisher akzeptierte Keyframes | `self.kf_features: list[Tensor]` |
| $n$ | Anzahl gespeicherter Keyframe-Features | `len(self.kf_features)` |
| $d$ | minimale Feature-Distanz | `score.min_dist` |
| $\alpha$ | Akzeptanz-Schwelle | `cfg.alpha` |
| $N_{\max}$ | Sliding-Window-Größe (FIFO) | `cfg.max_kfs` |

## 5. Was wir übernehmen, adaptieren, weglassen

| Paper-Komponente | In unserer Impl? | Anmerkung |
|---|---|---|
| DINOv2-Small als $\phi$ | übernommen verbatim | via `torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')` |
| L2-Distanz auf normalisierten Features | übernommen verbatim | `\|\phi(E) - \phi(K)\|_2 \geq \alpha` |
| Bootstrap (1. Frame force-accept) | übernommen | identisch |
| Threshold $\alpha$ als Hyperparameter | übernommen | Paper gibt keinen konkreten Wert; wir defaulten auf 0.4 |
| Submap-Konzept (10 KFs pro Submap, dann Reset) | **adaptiert** | VINGS hat keine Submaps; wir verwenden ein FIFO-Window der Länge $N_{\max} = 10$. Das hält den Vergleichshorizont lokal (wie ein Submap-Reset), ohne die Submap-Indirection. |
| Multi-Agent-Loop-Closure (Beitrag 1) | nicht übernommen | nicht relevant für single-agent VINGS |
| GaussianSPA-Compaction (Beitrag 2) | nicht übernommen | nicht im KF-Selector-Slot; eigenes Feature |
| Diagnose-Modus `force_accept_all` | unsere Ergänzung | analog zu `mm3dgs`/`adaptive_kf` — akzeptiert alles, loggt aber Distanzen für α-Kalibrierung |

## 6. Implementierungs-Details

### 6.1 DINOv2 laden

```python
import torch
model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
model = model.to(device).eval()
for p in model.parameters():
    p.requires_grad_(False)
```

Erster Aufruf braucht **Internet** und lädt ~85 MB Gewichte nach
`~/.cache/torch/hub/`. Danach offline-tauglich. Wenn kein Netz da ist und kein
Cache existiert, schlägt der Load fehl — der Selector wirft dann eine klare
`RuntimeError`-Meldung (kein silent fallback).

### 6.2 Bild-Preprocessing

DINOv2 erwartet `(B, 3, H', W')` float32 mit H' und W' als Multiple of 14
(Patch-Size). Wir benutzen 224 × 224 = 16 · 14 × 16 · 14. Eingang ist `(H, W, 3)
uint8 BGR` (so liefert es `viz_out['images'][-1]` nach Konversion in `run.py:265-267`).
Konvertierung Schritt für Schritt:

```python
# 1. BGR → RGB
rgb = rgb_uint8[..., ::-1].copy()
# 2. (H, W, 3) uint8 → (3, H, W) float32 / 255
tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
# 3. ImageNet-normieren
tensor = (tensor - mean) / std        # mean=[0.485,0.456,0.406] std=[0.229,0.224,0.225]
# 4. Batch-Dim hinzu
tensor = tensor.unsqueeze(0).to(device)
# 5. Resize auf 224×224
tensor = F.interpolate(tensor, size=(224, 224), mode='bilinear', align_corners=False)
```

### 6.3 Feature-Extraktion

```python
with torch.no_grad():
    feat = model.forward_features(tensor)['x_norm_clstoken']   # (1, 384)
feat = F.normalize(feat, dim=1).squeeze(0)                     # (384,)
```

`forward_features()` ist das DINOv2-API für strukturierten Output (klares CLS-Token
+ Patch-Tokens). Das `x_norm_clstoken` ist der LayerNorm-normalisierte CLS-Token
und der korrekte Ankerpunkt für Bild-Embeddings.

### 6.4 Vektorisierte Min-Distance

```python
refs = torch.stack(self.kf_features, dim=0)    # (n, 384)
dists = torch.norm(feat[None, :] - refs, dim=1)  # (n,)
d_min = float(dists.min())
```

Kein Python-Loop. Für n = 10 KFs ist das ~10 µs auf GPU — vernachlässigbar
gegen die DINOv2-Inferenz von ~5 ms.

### 6.5 Cosine-Äquivalenz (zur Reviewer-Beruhigung)

Bei L2-normalisierten Vektoren $a, b$ mit $\|a\| = \|b\| = 1$ gilt:

$$\|a - b\|_2^2 = \|a\|^2 + \|b\|^2 - 2\langle a, b\rangle = 2 - 2\cos(a, b)$$

Also ist $\|a-b\|_2 \geq \alpha$ äquivalent zu $\cos(a,b) \leq 1 - \alpha^2/2$.

| $\alpha$ | äquivalente cosine-Schwelle |
|---|---|
| 0.2 | 0.98 |
| 0.3 | 0.955 |
| 0.4 | 0.92 |
| 0.5 | 0.875 |
| 0.6 | 0.82 |

Wir verwenden trotzdem L2 weil das Paper es so notiert.

### 6.6 FIFO mit `max_kfs`

```python
self.kf_features.append(feat.detach())
if self.cfg.max_kfs > 0 and len(self.kf_features) > self.cfg.max_kfs:
    self.kf_features.pop(0)
```

Bei `max_kfs = 0` werden alle Features behalten — Memory ist vernachlässigbar
(384 floats × 4 B × 500 KFs = ~750 KB).

### 6.7 Failsafes

- **rgb is None** → force-accept. Tritt bei rgb-losen Smoketests auf; auch
  konsistent mit `adaptive_kf`-Selector-Verhalten.
- **DINOv2-Load fehlgeschlagen** → `RuntimeError` im Constructor. Kein
  Fallback auf Gradient-Features o.ä., weil das die Paper-Methodik bricht.
- **`force_accept_all = True`** → Diagnose-Modus. Selector berechnet weiterhin
  $d$ (für Logging via `score.min_dist`), akzeptiert aber alles. So kann man
  die Verteilung der Distanzen im PhaseTimer-Output sehen und ein passendes
  $\alpha$ ableiten.

## 7. Tuning-Workflow für α

Schritt für Schritt:

1. **Diagnose-Lauf** mit `force_accept_all: true` auf einer kurzen Sequenz
   (z.B. 200 Frames smallcity). Das akzeptiert alles, loggt aber pro Frame
   den `min_dist`-Wert.

2. **Verteilung lesen** aus dem Log oder via `scripts/analyze_profiling.py`.
   Die Distanzen sind typischerweise bimodal:
   - $d \in [0.05, 0.2]$ — fast-statische Frames (kaum Bewegung)
   - $d \in [0.3, 0.8]$ — echte Szenenwechsel

3. **α wählen** so, dass ~30–50 % der Frames akzeptiert werden:
   - smallcity (langsame Drohne, viel Detail): $\alpha \approx 0.3$
   - AGZ (schnelles Aerial, weniger Frame-zu-Frame-Überlapp): $\alpha \approx 0.5$
   - statisch-rotierende Kamera: ggf. $\alpha > 0.5$, weil DINOv2 weniger
     empfindlich auf Pose-Änderung ist

4. **Verifizieren**: erneuter Lauf mit `force_accept_all: false` und finalem α.
   KF-Rate im PhaseTimer-Summary checken (`frame_select`-Zeile + Mapper-Rate).

### Tradeoff explizit
- **kleines $\alpha$** → viele KFs → bessere Rekonstruktion, aber Mapper
  überlastet (siehe `MAPPING_TRACKING.md`)
- **großes $\alpha$** → wenige KFs → schneller Mapping-Schritt, aber
  Lücken in der Karte

## 8. Sensitivität & Datensatz-Spezifikum

- **Bildauflösung**: DINOv2 resizet intern auf 224 × 224. Originalauflösung
  ist also egal, solange das Resize nicht zu aggressiv ist. Paper merkt an,
  dass auf 512 × 512 Aria die Loop-Closure-Stage schlechter ist als auf
  1200 × 680 Replica — das liegt aber an der Renderqualität für Loop-Closure
  (Beitrag 1, nicht Beitrag 3), **nicht** an DINOv2 selbst. Der KF-Selektor
  ist auflösungs-agnostisch.

- **Lighting & Domain**: DINOv2 ist self-supervised auf ~142 M Internet-Bildern
  trainiert → robust für natürliche Outdoor-Szenen (smallcity, AGZ). Stark
  synthetische Szenen (z.B. Replica) sind out-of-distribution, sollten aber
  trotzdem brauchbare Embeddings liefern.

- **Bewegung**: Der Selektor kennt **weder Translation noch Rotation explizit**.
  Er sieht nur Bild-Inhalt. Bei reiner Kamerarotation ohne Translation kann
  es passieren, dass Frames als zu ähnlich klassifiziert werden, obwohl sich
  die Geometrie ändert (DINOv2 ist relativ orientierungs-robust). Failsafe
  via `mapper_kf_skip` oder größeres $\alpha$.

- **Dynamic Scenes**: Bewegte Objekte (Autos, Personen) verschieben Features
  stärker als statische Szenen. Bei sehr dynamischen Sequenzen evtl. $\alpha$
  anheben, damit nicht jedes leicht versetzte Frame einen neuen KF triggert.

- **GPU-Memory**: DINOv2-S braucht ~200 MB VRAM für das Modell selbst + ein
  paar MB für eine 224×224-Inferenz. Vernachlässigbar gegen den Mapper.

## 9. Code-Pointer + Smoketest

| Datei | Inhalt |
|---|---|
| `scripts/vings_utils/coko_slam_selector.py` | Selector + `CokoSlamConfig` + `CokoSlamScore` + Standalone-Smoketest |
| `scripts/vings_utils/selector_factory.py` | `@register_selector("coko_slam")`-Eintrag |
| `configs/local/smallcity/coko_slam/smallcity_200_coko_slam.yaml` | Beispiel-Config (smallcity) |
| `docs/COKO_SLAM.md` | diese Datei |

Standalone-Test:

```bash
cd /home/philipp/Dokumente/Github/VINGS-Mono-BA
PYTHONPATH=scripts python scripts/vings_utils/coko_slam_selector.py
```

Erwartete Ausgabe:
- 10 Test-Frames mit unterschiedlichen Random-Texturen
- Erstes Frame: `forced ACCEPT`
- Frames mit stark unterschiedlichem Inhalt: `ACCEPT` (typisch 3-6 von 9)
- Frames mit sehr ähnlichem Inhalt: `skip`

Wenn DINOv2 nicht ladbar (offline, keine Cache):
```
skipped: dinov2 not loadable (...)
```

## 10. Beispiel-Config

```yaml
frame_selector:
  kind: coko_slam
  alpha: 0.4               # L2-Distanz-Threshold im Feature-Space
                           # niedriger = mehr KFs, höher = weniger KFs
  model_name: dinov2_vits14
                           # alternative: dinov2_vitb14 (768-dim), vitl14 (1024-dim)
  image_size: 224          # Resize-Target, muss Multiple of 14 sein
  device: cuda             # cpu möglich, aber 10-20× langsamer
  max_kfs: 10              # Sliding-Window-Größe (Paper: 10 KFs/Submap)
                           # 0 = alle KFs behalten
  force_accept_all: false  # true = Diagnose-Modus: alles akzeptieren, Distanzen loggen
```

Vollständige Config-Datei: `configs/local/smallcity/coko_slam/smallcity_200_coko_slam.yaml`.

## 11. Was im BA-Methodenkapitel steht

- **Quelle**: Li et al. 2026, *"Compact Keyframe-Optimized Multi-Agent Gaussian
  Splatting SLAM"*, Sektion 3.1 (Keyframing using Feature Vector). Selbst eine
  Adaption von Thorne et al. 2024 (LiDAR-Keyframe-Selection via Feature-Distanz).

- **Adaption**: Das Paper benutzt das Selektionsverfahren innerhalb von Submaps
  (10 KFs pro Submap, dann Submap-Reset). VINGS hat kein Submap-Konzept; wir
  verwenden stattdessen ein FIFO-Sliding-Window der Größe 10 als äquivalenten
  lokalen Vergleichshorizont. Default-$\alpha = 0.4$ ist eine eigene Wahl,
  weil das Paper keinen konkreten Zahlenwert angibt.

- **Begründung**: Von den drei Coko-SLAM-Beiträgen ist nur (c) — die
  KF-Selektion — single-agent-relevant. (a) Multi-Agent-Loop-Closure ohne
  Initialposen und (b) GaussianSPA-Compaction sind beide Multi-Agent-Server-spezifisch
  und liegen außerhalb des Scope dieser BA.
