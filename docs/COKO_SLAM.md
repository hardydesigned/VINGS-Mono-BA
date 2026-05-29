# Coko-SLAM Keyframe-Selektor

Referenz:
- **Paper**: Li M.M.Q., Lajoie P.-Y., Liu J., Beltrame G. — *"Compact Keyframe-Optimized
  Multi-Agent Gaussian Splatting SLAM"* (Coko-SLAM), arXiv:2604.00804, April 2026.
- **Referenz-Code**: <https://github.com/lemonci/coko-slam>
  - Selektion: `src/entities/agent.py::should_start_mapping` und
    `should_start_new_submap`
  - Feature-Extraktion: `src/entities/loop_detection/feature_extractors.py::DINOFeatureExtractor`
  - Defaults: `configs/ReplicaMultiagent/replica_multiagent.yaml`

Diese Datei dokumentiert ausschließlich den **Keyframe-Selektor** aus Sektion 3.1
des Papers. Die anderen beiden Beiträge — Multi-Agent-Loop-Closure ohne
Initialposen (Sektion 3.3) und GaussianSPA-Compaction (Sektion 3.2) — sind für
VINGS (single-agent) nicht relevant und werden hier nicht implementiert.

> **Status (2026-05-26)**: Diese Implementierung ist auf das Referenz-Repo
> abgeglichen, nicht nur auf den Paper-Text. Wo Repo und Paper-Text
> auseinanderlaufen (Cosine-Distanz statt L2, Patch-Mean-Features statt
> CLS-Token, datengetriebener Submap-Reset statt fix-N), folgen wir dem Repo.
> Siehe Sektion **5. Was wir übernehmen, adaptieren, weglassen** für Details.

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

Zusätzlich gibt es eine **zweite Stufe**: wenn das neue Bild *sogar von dem ersten
Bild der aktuellen Sammlung* sehr verschieden ist (und genug KFs in der Sammlung
sind), startet Coko-SLAM eine ganz neue Sammlung. Das nennen sie "Submap-Reset".

In Stichworten: **"wenn nichts in meinem Archiv aussieht wie dieses Bild, ist
es ein guter Kandidat fürs Archiv — und wenn es so anders ist, dass es nicht
mal mehr zum Ursprung der aktuellen Sammlung passt, fange ich eine neue an."**

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

Wie messen wir "Ähnlichkeit" zwischen zwei Steckbriefen? **Das Repo nutzt
Cosine-Ähnlichkeit** (genauer: `1 − cos(a, b)` = **Cosine-Distanz**) via
FAISS-Inner-Product-Index auf L2-normalisierten Vektoren. Cosine-Distanz misst
nur die *Richtung* der Vektoren; Magnituden werden durch die Normalisierung
auf 1 weggebügelt.

## 3. Der Algorithmus, Wort für Wort

Hier der zweistufige Algorithmus aus dem Repo (`src/entities/agent.py`):

```
input:  Stream von RGB-Bildern E_1, E_2, E_3, ...
state:  K       = Liste der Feature-Vektoren akzeptierter KFs in der
                  aktuellen Submap (anfangs leer)
        anchor  = Feature des ersten KFs der aktuellen Submap
        sub_idx = laufende Submap-Nummer
param:  alpha             — Keyframing-Schwelle (cosine-distance)
        submap_threshold  — Submap-Reset-Schwelle (cosine-distance)
        min_kfs           — minimale KFs pro Submap vor Reset-Eligibility

for jedes neue Bild E in stream:
    f = phi(E) / ||phi(E)||                          # Schritt 1: Feature

    # Schritt 2: erster Frame? -> Submap 0 starten
    if anchor is None:
        K     = [f]
        anchor = f
        sub_idx = 0
        akzeptiere E
        continue

    # Schritt 3: Submap-Reset? (datengetrieben)
    d_anchor = 1 - <f, anchor>
    if d_anchor > submap_threshold and |K| >= min_kfs:
        K       = [f]                                # neues Submap
        anchor  = f
        sub_idx += 1
        akzeptiere E
        continue

    # Schritt 4: In-Submap-Keyframe-Decision
    d_min = min_{f_k in K} (1 - <f, f_k>)
    if d_min > alpha:
        K.append(f)                                   # neues KF in der Submap
        akzeptiere E
    else:
        überspringe E
```

Schritt für Schritt:

1. **Feature-Extraktion**: Wir nehmen das neue Bild $E$, geben es durch DINOv2,
   nehmen den **Mittelwert über `[CLS-Token, Patch-Token-1, ..., Patch-Token-N]`**
   (so macht es das Repo via HuggingFace `last_hidden_state.mean(dim=1)`), und
   bekommen einen Vektor mit 384 Zahlen. Wir normalisieren auf L2-Länge 1,
   damit Cosine-Distanz wohldefiniert ist und beide Stufen denselben
   Wertebereich teilen.

2. **Bootstrap**: Wenn noch nichts im "Gedächtnis" ist, starten wir Submap 0
   mit dem aktuellen Frame als Anker und erstem KF. Force-accept.

3. **Submap-Reset (Stage 1)**: Vergleiche $f$ mit dem Submap-Anker (= erster
   KF der aktuellen Submap). Wenn die Cosine-Distanz $d_{anchor} >$
   `submap_threshold` **UND** die Submap schon $\geq$ `min_kfs` KFs enthält,
   schließe die alte Submap und starte eine neue mit $f$ als Anker und erstem
   KF. Force-accept. **Die „10 KFs pro Submap" aus dem Paper sind die untere
   Schranke `min_kfs`, kein fester Reset-Zeitpunkt.**

4. **In-Submap-KF-Decision (Stage 2)**: Wenn kein Submap-Reset ausgelöst wurde,
   berechne die minimale Cosine-Distanz zu allen $K$ der aktuellen Submap.
   Wenn diese Distanz $> $ `alpha`, akzeptiere; sonst überspringe.

**Wichtig**: Stufe 1 und Stufe 2 sind exklusiv pro Frame — entweder triggert
das Frame einen Submap-Reset (und wird der Seed der neuen Submap), oder es
durchläuft die normale KF-Decision innerhalb der aktuellen Submap.

### 3.1 Repo-Default-Werte (Replica)

Aus `configs/ReplicaMultiagent/replica_multiagent.yaml`:

| Param | Wert | Bedeutung |
|---|---|---|
| `keyframing_threshold` (= `alpha`) | **0.02** | Cosine-Distanz im Submap |
| `submapping_threshold` (= `submap_threshold`) | **0.05** | Cosine-Distanz zum Anker |
| `keyframe_num` (= `min_kfs_per_submap`) | **10** | min. KFs pro Submap |
| Feature-Extractor | `dino` (DINOv2-Small) | via HuggingFace `./dinov2-small` |
| `embed_size` | 384 | DINOv2-Small CLS+Patch-Mean-Dim |

### 3.2 Cosine-Distanz vs L2-Distanz

Bei L2-normalisierten Vektoren $a, b$:

$$\|a-b\|_2^2 = 2 (1 - \cos(a, b)) = 2 \cdot d_{\cos}$$

also $\|a-b\|_2 = \sqrt{2 \cdot d_{\cos}}$. Beide Maße sind monoton, aber
der Threshold ist nicht identisch. Konvertierungstabelle für gleichen
Operating-Point:

| $d_{\cos}$ | $\|a-b\|_2$ | Interpretation |
|---|---|---|
| 0.001 | 0.045 | extrem ähnlich |
| 0.005 | 0.10 | sehr ähnlich |
| 0.01 | 0.14 | identisch bis leicht verschieden |
| 0.02 | 0.20 | leicht verschieden (**Repo `alpha`**) |
| 0.05 | 0.32 | deutlich verschieden (**Repo `submap_threshold`**) |
| 0.1 | 0.45 | erkennbar anders |
| 0.2 | 0.63 | klar verschieden |
| 0.5 | 1.0 | orthogonal-nah |

`distance_metric: cosine` ist der Default (repo-treu). `distance_metric: l2`
existiert für Legacy-Configs, die α im L2-Sinne tuneten (Werte 0.2-0.6).

## 4. Symbol-Tabelle

| Symbol | Bedeutung | Wo im Code |
|---|---|---|
| $E$ | aktueller Kandidaten-Frame (RGB-Bild) | `rgb` Parameter von `should_accept()` |
| $\phi(\cdot)$ | DINOv2-Feature-Extractor + L2-Norm | `self._extract()` |
| $\phi(E) \in \mathbb{R}^{384}$ | L2-normalisierter Feature-Vektor | `feat` in `should_accept()` |
| $K = \{K_1, \ldots, K_n\}$ | KFs der aktuellen Submap | `self.kf_features: list[Tensor]` |
| $A$ | Anker-Feature der aktuellen Submap | `self._submap_anchor: Tensor` |
| $\alpha$ | Keyframing-Schwelle | `cfg.alpha` |
| $\beta$ | Submap-Reset-Schwelle | `cfg.submap_threshold` |
| $N_{\min}$ | min. KFs pro Submap | `cfg.min_kfs_per_submap` |
| $N_{\max}$ | Hard-Cap auf Submap-Größe (0 = aus) | `cfg.max_kfs` |
| $d_{anchor}$ | Distanz zum Submap-Anker | `score.submap_anchor_dist` |
| $d_{\min}$ | min. Distanz zu allen KFs in $K$ | `score.min_dist` |

## 4.1 Verifikations-Matrix (Paper-Text ↔ Repo-Code ↔ unsere Impl)

Jede Zeile ist eine algorithmische Entscheidung, die ich beim Refactor
auf Paper-Treue gegenprüfen wollte. Repo-Spalte zitiert oder verweist auf die
echte Codezeile aus `github.com/lemonci/coko-slam`.

| Aspekt | Paper (Sec. 3.1) | Repo | Unsere Impl |
|---|---|---|---|
| **Feature-Backbone** | "Dino V2-Small [38]" | `DINOFeatureExtractor` mit `AutoModel.from_pretrained` auf `./dinov2-small` (`feature_extractors.py:34`) | `torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')` — selbe Gewichte, hub statt HF |
| **Feature-Aggregation** | nicht spezifiziert | `outputs.last_hidden_state.mean(dim=1)` über `[CLS, patch_1..N]` (`feature_extractors.py:46`) | `torch.cat([cls.unsqueeze(1), patches], dim=1).mean(dim=1)` → mathematisch identisch (default `feature_aggregation="patch_mean_with_cls"`) |
| **L2-Normalisierung** | nicht spezifiziert | `features / features.norm(p=2, dim=1, keepdim=True)` (`feature_extractors.py:47`) | `F.normalize(feat, dim=1)` |
| **Distanz-Maß** | `||ϕ(E)−ϕ(K)||` (L2-Notation) | `faiss.IndexFlatIP` (Inner-Product auf L2-norm Vektoren = cosine sim); `1 - sim` als cosine-dist (`agent.py:160`) | `1.0 - refs @ feat` (mathematisch identisch zu Repo, ohne FAISS-Dep). Default `distance_metric="cosine"`. Legacy `"l2"` für alte Configs. |
| **Stage 2: KF-Akzeptanz** | `d ≥ α` (≥) | `1 - highest_similarity > keyframing_threshold` (strict `>`) (`agent.py:160`) | `d_min > cfg.alpha` (strict `>`, **Repo gefolgt**, weicht von Paper ≥ um Floating-Point-Measure-Zero ab) |
| **Stage 1: Submap-Reset-Trigger** | nicht im Paper-Text als Algorithmus beschrieben, nur "10 KFs per submap" | `(1 - cos(frame, anchor) > submapping_threshold) AND (n_kfs >= keyframe_num)` (`agent.py:135`) | `(anchor_dist > cfg.submap_threshold) AND (len(kf_features) >= cfg.min_kfs_per_submap)` |
| **Submap-Anker** | nicht explizit | `current_submap_feature`, beim Submap-Start auf erste KF-Feature gesetzt (`agent.py:140, 295`) | `self._submap_anchor`, in `_open_submap()` auf seed gesetzt, eingefroren bis nächster Reset |
| **Submap-Memory-Reset** | nicht explizit | `submap_faiss_index.reset(); submap_faiss_index.add(current_frame_feature)` (`agent.py:288-289`) | `self.kf_features.clear(); self.kf_features.append(feat)` (in `_open_submap`) |
| **Bootstrap (erster Frame)** | nicht explizit | `init_map` → `keyframe_ids.append(0)`; vor der Hauptschleife: `submap_faiss_index.add(current_keyframe_feature)` (`agent.py:268-282`) | `if self._submap_anchor is None: self._open_submap(feat)` — force-accept, seed Submap 0 |
| **Reihenfolge der Stages** | nicht explizit | `if start_new_submap: ... elif should_start_mapping(): ...` (`agent.py:286,297`) — exklusiv | Stage 1 returnt early bei Trigger; sonst Stage 2 |
| **Repo-Defaults (Replica)** | keine Zahlen | `keyframing_threshold: 0.02`, `submapping_threshold: 0.05`, `keyframe_num: 10` (`configs/ReplicaMultiagent/replica_multiagent.yaml`) | `alpha=0.02`, `submap_threshold=0.05`, `min_kfs_per_submap=10` |
| **Suchstruktur** | nicht spezifiziert | FAISS `IndexFlatIP` | `torch.stack + @` — für n ≤ 20 KFs/Submap schneller als FAISS-Overhead, identisches Resultat |

**Bewusste Abweichungen vom Paper-Text zugunsten Repo:**

1. `>` statt `≥` in der KF-Akzeptanz: ist Measure-Zero für stetige Features
   (zwei reelle Distanzen sind praktisch nie exakt gleich), pragmatisch
   irrelevant.
2. Cosine statt L2: monoton äquivalent auf L2-norm Vektoren
   (`||a-b||² = 2(1-cos)`), aber der Schwellenwert hat eine andere Skala.
   Wir folgen dem Repo, damit `alpha=0.02` der publizierten Zahl entspricht.
3. Patch-Mean-Features statt nur CLS: nicht im Paper-Text, aber das ist
   was der Repo-Code tatsächlich tut.
4. Submap-Reset ist im Repo *datengetrieben* (zwei Schwellen), nicht fix
   bei N=10. Die "10" im Paper ist im Repo als untere Schranke
   (`keyframe_num`) implementiert. Diese Doppelschwellen-Logik steht
   nicht im Paper-Text, ist aber das, was läuft.

**Bewusste Erweiterungen über Paper+Repo hinaus** (VINGS-spezifisch):

- `distance_metric: l2` — Legacy-Mode für VINGS-Configs, die vor dem
  Repo-Lookup mit L2-Schwellen getuned wurden (α-Werte 0.2-0.6).
- `feature_aggregation: cls` — Legacy-Mode für ältere VINGS-Runs.
- `memory_mode: fifo` — alternative Hard-Cap-Policy (drop-oldest). Nicht
  paper/repo, nur für Ablations-Vergleiche.
- `max_kfs: int` — Hard-Cap als Safety-Net falls eine sehr statische Szene
  den datengetriebenen Reset nie auslöst. Default `0` (= aus).
- `force_accept_all: bool` — Diagnose-Modus zur α-Kalibrierung. Analog zu
  `mm3dgs`/`adaptive_kf`-Selektoren.

## 5. Was wir übernehmen, adaptieren, weglassen

| Komponente (Repo) | In unserer Impl? | Anmerkung |
|---|---|---|
| DINOv2-Small als $\phi$ | übernommen verbatim | via `torch.hub.load` statt HuggingFace; selbe Gewichte |
| Feature = `last_hidden_state.mean(dim=1)` | übernommen | als Mittelwert über `[CLS, patch_1, …, patch_N]` aus `forward_features()` (`feature_aggregation="patch_mean_with_cls"`). **Default**. Alternative `cls` = nur LayerNorm-CLS, legacy. |
| L2-Normalisierung der Features | übernommen verbatim | `F.normalize(dim=1)` |
| Cosine-Distanz (über FAISS Inner-Product) | übernommen | als `1 - feat @ refs.T` (mathematisch äquivalent, ohne FAISS-Dependency). **Default `distance_metric: cosine`**. Legacy `l2` verfügbar. |
| Submap-Reset = Anker-Divergenz AND min-Größe | übernommen verbatim | `cfg.submap_threshold` + `cfg.min_kfs_per_submap` |
| Bootstrap (1. Frame force-accept) | übernommen | identisch |
| Faiss-Backend | nicht übernommen | unnötig für $n \leq 20$ KFs/Submap; torch-stack + matmul ist schneller |
| Multi-Agent-Loop-Closure (Beitrag 1) | nicht übernommen | nicht relevant für single-agent VINGS |
| GaussianSPA-Compaction (Beitrag 2) | nicht übernommen | nicht im KF-Selector-Slot; eigenes Feature |
| Submap-Datenstruktur (Persistenz, Server-Sync) | nicht übernommen | VINGS hat keinen Server; Submap existiert hier nur als Selector-Memory |
| `memory_mode: fifo` | unsere Ergänzung | Drop-Oldest-Sliding-Window. Nicht im Paper / Repo; nur für Ablations-Vergleich. |
| `max_kfs` Hard-Cap | unsere Ergänzung | Safety-Net falls statische Szene den datengetriebenen Reset nie triggert. Default `0` (= aus, vollständig datengetrieben). |
| Diagnose-Modus `force_accept_all` | unsere Ergänzung | analog zu `mm3dgs`/`adaptive_kf` — akzeptiert alles, loggt aber Distanzen für α-Kalibrierung |

## 6. Implementierungs-Details

### 6.1 DINOv2 laden

Wir nutzen `torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")` statt
des HF-Pfads aus dem Repo (`AutoImageProcessor` + `AutoModel`). Die Gewichte sind
identisch (Meta veröffentlicht beide), die Hub-Variante ist schlanker und
braucht kein `transformers`-Paket. Erster Aufruf braucht **Internet** und lädt
~85 MB Gewichte nach `~/.cache/torch/hub/`. Danach offline-tauglich.

### 6.2 Bild-Preprocessing

DINOv2 erwartet `(B, 3, H', W')` float32 mit H' und W' als Multiple of 14
(Patch-Size). Wir benutzen 224 × 224 = 16 · 14 × 16 · 14. Eingang ist `(H, W, 3)
uint8 BGR` (so liefert es `viz_out['images'][-1]` nach Konversion in
`run.py:608-611`). Konvertierung Schritt für Schritt:

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

### 6.3 Feature-Extraktion (Repo-treu)

```python
with torch.no_grad():
    out = model.forward_features(tensor)
    cls = out["x_norm_clstoken"].unsqueeze(1)       # (1, 1, D)
    patches = out["x_norm_patchtokens"]             # (1, N, D)
    feat = torch.cat([cls, patches], dim=1).mean(dim=1)  # (1, D)
feat = F.normalize(feat, dim=1).squeeze(0)          # (D,)
```

Repo-Code (HuggingFace-Variante, äquivalent):
```python
outputs = self.model(**inputs)                       # last_hidden_state: (1, 1+N, D)
features = outputs.last_hidden_state.mean(dim=1)     # (1, D)
features = features / features.norm(p=2, dim=1, keepdim=True)
```

### 6.4 Distanz-Berechnung

Cosine (Default, Repo):
```python
sims = refs @ feat                       # (n,) inner products, refs ist L2-normalisiert
dists = 1.0 - sims                       # cosine distances
```

L2 (Legacy):
```python
dists = torch.norm(feat[None, :] - refs, dim=1)
```

Kein Python-Loop. Für $n \leq 20$ KFs/Submap ist das ~10 µs auf GPU —
vernachlässigbar gegen die DINOv2-Inferenz von ~5 ms.

### 6.5 Submap-Anker-Update

Der Anker $A$ wird in `_open_submap(feat)` gesetzt. Er bleibt **eingefroren**,
solange die Submap besteht. Die Anker-Distanz $d_{anchor}$ wächst monoton mit
der Szenen-Divergenz seit Submap-Start — sobald sie `submap_threshold`
überschreitet und $|K| \geq$ `min_kfs`, schließt die Submap.

### 6.6 Failsafes

- **`rgb is None`** → `RuntimeError`. Coko-SLAM ist ein bildbasiertes Verfahren;
  ohne RGB *darf* der Selector nicht silent durchlassen. (Älteres Verhalten
  „force-accept on missing rgb" wurde entfernt — es maskierte Call-Site-Bugs.)
- **DINOv2-Load fehlgeschlagen** → `RuntimeError` im Constructor. Kein
  Fallback auf andere Features, das würde die Paper-Methodik brechen.
- **`force_accept_all = True`** → Diagnose-Modus. Selector berechnet weiterhin
  $d_{\min}$ (für Logging via `score.min_dist`), akzeptiert aber alles in
  Stufe 2. Stufe 1 (Submap-Reset) bleibt aktiv, weil sie eine andere
  Semantik hat (echte Szenenwechsel-Detektion).
- **`max_kfs > 0` Hard-Cap** → falls eine sehr statische Szene den
  datengetriebenen Reset nie auslöst, kappt `max_kfs` das Submap. Bei
  `memory_mode: submap_reset` wird beim Erreichen ein erzwungenes Submap
  geöffnet; bei `fifo` wird der älteste KF verworfen. Default `0` (= aus).

## 7. Tuning-Workflow für α und `submap_threshold`

Schritt für Schritt:

1. **Diagnose-Lauf** mit `force_accept_all: true` auf einer kurzen Sequenz
   (z.B. 200 Frames smallcity). Das akzeptiert alles in Stufe 2, loggt aber
   pro Frame `score.min_dist` und `score.submap_anchor_dist`.

2. **Verteilungen lesen** via `scripts/analyze_profiling.py`. Erwartete
   Bimodalität (Cosine-Distanz):
   - `min_dist` $\in [0.001, 0.01]$ — quasi-statische Frames
   - `min_dist` $\in [0.02, 0.1]$ — echte Szenenwechsel
   - `submap_anchor_dist` wächst monoton während einer Submap; steigt
     typischerweise von $\sim 0$ auf $\sim 0.05$ über 5-15 KFs

3. **Schwellen wählen**:
   - `alpha` so, dass ~30-50 % der Frames Stufe 2 passieren
     (Repo-Default 0.02 ist ein guter Startpunkt)
   - `submap_threshold` ≈ 2-3 × `alpha`, sodass nur "wirkliche" Szenenwechsel
     ein neues Submap öffnen (Repo: 0.05 = 2.5 × 0.02)
   - `min_kfs_per_submap` = 10 (Repo). Niedriger → mehr Submap-Resets =
     mehr forced-accepts an Boundaries.

4. **Verifizieren**: erneuter Lauf mit `force_accept_all: false` und finalen
   Werten. KF-Rate im PhaseTimer-Summary checken (`frame_select`-Zeile +
   Mapper-Rate).

### Tradeoff explizit
- **kleines $\alpha$, kleines `submap_threshold`** → viele KFs, oft neue
  Submaps → bessere Rekonstruktion, aber Mapper überlastet
- **großes $\alpha$, großes `submap_threshold`** → wenige KFs, lange Submaps
  → schneller Mapping-Schritt, aber Lücken in der Karte

### Legacy-Configs migrieren (L2 → Cosine)

Wenn du alte L2-α-Werte hast (0.2-0.6), konvertiere via
$d_{\cos} = \|a-b\|_2^2 / 2$:

| altes `alpha` (L2) | neues `alpha` (cosine) |
|---|---|
| 0.20 | 0.020 |
| 0.25 | 0.031 |
| 0.40 | 0.080 |
| 0.55 | 0.151 |
| 0.60 | 0.180 |

Alternativ explizit `distance_metric: l2` setzen und die alten Werte
weiternutzen — dann ist α nicht repo-faithful, aber der Lauf reproduzierbar.

## 8. Sensitivität & Datensatz-Spezifikum

- **Bildauflösung**: DINOv2 resizet intern auf 224 × 224. Originalauflösung
  ist also egal, solange das Resize nicht zu aggressiv ist. Paper merkt an,
  dass auf 512 × 512 Aria die Loop-Closure-Stage schlechter ist als auf
  1200 × 680 Replica — das liegt aber an der Renderqualität für Loop-Closure
  (Beitrag 1, nicht Beitrag 3), **nicht** an DINOv2 selbst. Der KF-Selektor
  ist auflösungs-agnostisch.

- **Aspect-Ratio**: Der Resize ist bilinear ohne Pad/Crop. Für 16:9 oder 3:2
  Bilder wird das Bild gestaucht. DINOv2 ist relativ robust, aber bei sehr
  asymmetrischen Aspect-Ratios merklicher Feature-Drift möglich. Falls
  problematisch: pad-to-square als externer Pre-Step.

- **Lighting & Domain**: DINOv2 ist self-supervised auf ~142 M Internet-Bildern
  trainiert → robust für natürliche Outdoor-Szenen (smallcity, AGZ). Stark
  synthetische Szenen (z.B. Replica) sind out-of-distribution, sollten aber
  trotzdem brauchbare Embeddings liefern.

- **Bewegung**: Der Selektor kennt **weder Translation noch Rotation explizit**.
  Er sieht nur Bild-Inhalt. Bei reiner Kamerarotation ohne Translation kann
  es passieren, dass Frames als zu ähnlich klassifiziert werden, obwohl sich
  die Geometrie ändert (DINOv2 ist relativ orientierungs-robust). Failsafe:
  `mapper_kf_skip` oder größeres $\alpha$.

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
| `configs/local/*/exp*/coko_slam/*.yaml` | Auto-generierte Sweep-Configs |
| `docs/COKO_SLAM.md` | diese Datei |

Standalone-Test:

```bash
cd /home/philipp/Dokumente/Github/VINGS-Mono-BA
PYTHONPATH=scripts python scripts/vings_utils/coko_slam_selector.py
```

Erwartete Ausgabe:
- 10 Test-Frames, erstes als `forced ACCEPT [RESET]` (Bootstrap = Submap 0)
- Stark unterschiedliche Patterns: `ACCEPT` (typisch 4-7 von 9)
- Duplikate: `skip`
- Manche Frames können den datengetriebenen Submap-Reset triggern
  (`[RESET]`-Tag in der Ausgabe, `submap_idx` steigt)

Wenn DINOv2 nicht ladbar (offline, kein Cache):
```
skipped: dinov2 not loadable (...)
```

## 10. Beispiel-Config

Repo-treuer Default:

```yaml
frame_selector:
  kind: coko_slam
  alpha: 0.02              # Cosine-Distanz-Schwelle für In-Submap-KF (Repo-Default)
  submap_threshold: 0.05   # Cosine-Distanz-Schwelle für Submap-Reset (Repo-Default)
  min_kfs_per_submap: 10   # min. KFs bevor Reset zulässig (Repo-Default)
  max_kfs: 0               # Hard-Cap auf Submap-Größe (0 = aus, rein datengetrieben)
  memory_mode: submap_reset  # was bei max_kfs-Cap? "submap_reset" | "fifo"
  distance_metric: cosine    # "cosine" (Repo) | "l2" (Legacy)
  feature_aggregation: patch_mean_with_cls  # "patch_mean_with_cls" (Repo) | "cls" (Legacy)
  model_name: dinov2_vits14
  image_size: 224          # muss Multiple of 14 sein (DINOv2-Patch-Size)
  device: cuda
  force_accept_all: false  # true = Diagnose-Modus
```

Legacy-Config (alte L2-Werte, falls Vergleichbarkeit mit alten Runs nötig):

```yaml
frame_selector:
  kind: coko_slam
  alpha: 0.4               # L2-Distanz-Schwelle (alte Semantik)
  distance_metric: l2
  feature_aggregation: cls
  submap_threshold: 1.0    # praktisch aus (L2-Distanz erreicht selten 1.0)
  min_kfs_per_submap: 0
  max_kfs: 10
  memory_mode: submap_reset
  # ... DINOv2-Settings ...
```

## 11. Was im BA-Methodenkapitel steht

- **Quelle**: Li et al. 2026, *"Compact Keyframe-Optimized Multi-Agent Gaussian
  Splatting SLAM"*, Sektion 3.1 (Keyframing using Feature Vector). Selbst eine
  Adaption von Thorne et al. 2024 (LiDAR-Keyframe-Selection via Feature-Distanz).

- **Quelle, technisch**: Referenz-Code aus dem offiziellen Coko-SLAM-Repo
  (`github.com/lemonci/coko-slam`), insbesondere `src/entities/agent.py` und
  `src/entities/loop_detection/feature_extractors.py`. Wo Paper-Text und
  Repo-Code auseinanderlaufen, wurde dem Repo gefolgt (Cosine-Distanz statt
  L2-Distanz, Patch-Mean-Features statt CLS-Token, zweistufige Decision mit
  datengetriebenem Submap-Reset statt fixer "10 KFs pro Submap").

- **Adaption**: Das Repo benutzt das Selektionsverfahren innerhalb eines
  Multi-Agent-Submap-Konstrukts mit Server-Sync. VINGS ist single-agent ohne
  Submap-Persistenz; wir reproduzieren nur die Selector-Semantik (Memory-Reset
  bei Anker-Divergenz). Keine Submap-Datei wird auf Platte geschrieben.

- **Begründung**: Von den drei Coko-SLAM-Beiträgen ist nur (c) — die
  KF-Selektion — single-agent-relevant. (a) Multi-Agent-Loop-Closure ohne
  Initialposen und (b) GaussianSPA-Compaction sind beide
  Multi-Agent-Server-spezifisch und liegen außerhalb des Scope dieser BA.

- **Validierungspfad**: Repo-Defaults (`alpha=0.02`, `submap_threshold=0.05`,
  `min_kfs=10`) auf VINGS-Datensätzen evaluieren; falls KF-Rate zu hoch/niedrig,
  via Diagnose-Workflow (Sektion 7) nachjustieren.
