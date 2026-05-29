# Adaptive-KF-Selector

Momentum-Aware Adaptive Keyframe-Selektor nach

> **"Adaptive Keyframe Selection for Scalable 3D Scene Reconstruction in
> Dynamic Environments"** — Jha, Zhou, Loianno,
> *arXiv:2510.23928v3 [cs.RO]*, 28 Dec 2025.

Diese Doku erklärt: was das Paper spezifiziert, was im offiziellen GitHub-Repo
(`jhakrraman/Adaptive_Keyframe_Selection`) *nicht* implementiert ist, was wir
übernehmen, was wir reproduzieren mussten und wo wir bewusst abweichen.

## Was das Paper spezifiziert

### Algorithmus 1 — Hybrid Error (Sec. 3.2)

Pseudo-Code (Algorithmus 1 im Paper):

```
function ComputeHybridError(I_t, I_k, D_k, Pose_t, Pose_k, K)
    Î_k, M  ← WarpFrame(I_k, D_k, Pose_k, Pose_t, K)        # (1)
    e_photo ← (1/|M|) · Σ_{p∈M} |I_t(p) - Î_k(p)|            # Eq. 1
    e_ssim  ← 1 - SSIM(I_t, Î_k)                              # Eq. 2
    e_t     ← α · e_photo + β · e_ssim                        # Eq. 3
    return e_t
```

Paper-Defaults (Sec. 3.4):
- **α = 0.7**, **β = 0.3** (Eq. 3)

Was das Paper *nicht* festlegt:
- Implementierung von `WarpFrame` (Forward-Splat vs. Backward-Sample, Z-Buffer
  vs. Interpolation, wie M aus OOB / Disokklusion / ungültiger Tiefe gebaut wird)
- Maskierung von SSIM (skimage hat keine maskierte SSIM)
- Pixel-Skalierung ([0, 255] oder [0, 1])

Was klar ist: `WarpFrame` bekommt **`D_k` (Keyframe-Tiefe)**, nicht `D_t`
(current-Tiefe). Paper Sec. 3.2:

> "we warp Ik into the camera view of It... by projecting the 3D points
>  **derived from the depth map of Ik**"

Das ist *Forward-Warping mit `D_k`*.

### Algorithmus 2 — Momentum-Aware Threshold (Sec. 3.3)

```
1: K ← {f_1},  f_last_kf ← f_1
2: E ← [],  θ ← θ_0
3: for t = 2 .. n do
4:     e_t ← ComputeHybridError(f_t, f_last_kf)
5:     append e_t to E
6:     if |E| ≥ W then
7:         μ_t ← mean(last W errors)                          # Eq. 4
8:         σ_t ← std(last W errors)                           # Eq. 5
9:         θ ← max(θ_0,  μ_t + k · σ_t)                       # Eq. 6
10:    else                                                    # warm-up
11:        θ ← θ_0 · (t/W)  +  θ_init · (1 - t/W)
12:    end if
13:    if e_t > θ then
14:        append f_t to K;  f_last_kf ← f_t
15:        θ ← γ · θ                                           # Eq. 7
16:    end if
17: end for
```

Paper-Defaults (Sec. 3.4):
- **θ_0 = 0.05** (Grid-Search auf Bonn-Validation)
- **W = 5** Frames (Paper-Text: *"We employ a **small window size W = 5** for
  the moving average to prioritize high responsiveness to rapid motion, which a
  standard EMA might over-smooth."*)
- **k = 1.5** Sensitivity
- **γ = 0.95** Decay-Faktor
- `θ_init` wird in Algorithmus 2 Z. 11 verwendet, aber in Sec. 3.4 nicht
  spezifiziert. Wir defaulten auf `θ_init = 2·θ_0 = 0.10` (Warm-up startet
  konservativ und entspannt).

## Was im offiziellen GitHub-Repo *nicht* steht

Das Repo (`jhakrraman/Adaptive_Keyframe_Selection`, Stand 2026) enthält
**weder Algorithmus 1 noch Algorithmus 2**:

| Paper-Komponente | Im Repo? |
|---|---|
| Algorithmus 1 (Hybrid-Error, Eq. 1-3) | ❌ |
| Algorithmus 2 (`θ = max(θ_0, μ + k·σ)`, Decay, Warm-up) | ❌ |
| Sliding-Window-Statistik E | ❌ |
| Overlap-Maske M aus Depth-Warping | ❌ |

Das Repo ist eine Sammlung ROS2-Wrapper für CUT3R-Inferenz; die einzige
Selektor-Logik ist eine hardcodierte Pose-Heuristik (`cut3r_package_improved.py:692-717`):

```python
rotation_threshold = 10.0       # Grad
translation_threshold = 0.2     # Meter
# accept iff Δrot > 10° OR Δtrans > 20 cm
```

…plus ein optionaler Whole-Image-SSIM-Vergleich. Das ist **näher an den
Baselines, gegen die das Paper sich vergleicht** (Tab. 5, "inertial-based [11]",
"optimization-based [14]") als am publizierten Verfahren. Konkret heißt das:
wir müssen Algorithmus 1 selbst implementieren, weil das Paper ihn nur
strukturell (Eq. 1-3) spezifiziert, nicht implementiert.

## Unsere Implementierung

### Algorithmus 2 — verbatim

`scripts/vings_utils/adaptive_kf_selector.py` Z. 167-235 reproduziert
Algorithmus 2 zeilenweise:

- Bootstrap (`last_kf_gray is None`) → Accept, seed KF
- Hybrid-Error berechnen (Algorithmus 1, siehe unten)
- Fail-safe: `|M| < min_overlap_pixels` → Accept, push `e=1.0` in E
- Sonst: `E.append(e)`, dann Threshold-Update (Eq. 6 oder Warm-up Z. 11)
- Accept iff `e > θ`; danach `θ ← γ·θ` (Eq. 7)

`np.std(arr)` ohne `ddof` → 1/W-Normierung wie in Paper Eq. 5.

### Algorithmus 1 — Forward-Warp mit `D_k`

Paper-faithful umgesetzt (`_hybrid_error`, Z. 245-322):

```
1. d_kf = self.last_kf_depth.ravel()
   valid_kf = isfinite ∧ min_depth < d_kf < max_depth

2. Rel-Pose KF→Cur:  R_kc = R_cur^T · R_kf
                     t_kc = R_cur^T · (t_kf - t_cur)

3. Für jeden KF-Pixel q mit valid_kf:
       P_kf  = K⁻¹ · [q.u, q.v, 1]ᵀ · D_k(q)         # KF-cam frame
       P_cur = R_kc · P_kf + t_kc                     # cur-cam frame
       (u, v) = π(K · P_cur)                          # projektion in cur-view

4. Forward-Splat mit Z-Buffer:
       Sortiere alle Quell-Pixel absteigend nach z_cur (fernster zuerst).
       warped[v_round, u_round] = I_k(q)
       mask_2d[v_round, u_round] = True
   → Der näheste Treffer pro Output-Pixel überschreibt → korrektes Z-Buffer.

5. e_photo = (1/|M|) · Σ_{p∈M} |warped(p) - I_t(p)| / 255       # Eq. 1
   e_ssim  = 1 - SSIM(I_t * M, warped * M, data_range=1.0)       # Eq. 2
   e       = α · e_photo + β · e_ssim                              # Eq. 3
```

**Z-Buffer durch farthest-first-Sortierung** statt explizitem Buffer: bei
kollidierenden Quell-Pixeln gewinnt der spätere `=`-Write, also der mit
kleinerem `z_cur`. Geometrisch korrekt, kostet einen `argsort` über H·W.

**SSIM mit Mask-Multiplikation** statt korrekt-masked-SSIM: skimage's
`structural_similarity` unterstützt keine Maske. Mask-Zero macht out-of-overlap
identisch in beiden Bildern (SSIM=1, Dissimilarity=0 dort), das verzerrt die
globale SSIM leicht nach oben — Approximation gut genug für den Score.

**Pixel-Skalierung [0, 1]** (float32 / 255). Paper spezifiziert das nicht
explizit; konsistent mit der Größenordnung von `θ_0 = 0.05` aus dem Paper-Grid-
Search.

### Defaults (vollständig paper-faithful)

| Symbol | Bedeutung | Default | Paper |
|---|---|---|---|
| `α` / `w_photo` | Photo-Gewicht (Eq. 3) | 0.7 | 0.7 ✓ |
| `β` / `w_ssim`  | SSIM-Gewicht (Eq. 3) | 0.3 | 0.3 ✓ |
| `W` (`window_size`) | Sliding-Window (Eq. 4-5) | 5 | 5 ✓ |
| `k` (`sensitivity`) | σ-Multiplikator (Eq. 6) | 1.5 | 1.5 ✓ |
| `γ` (`decay`) | Post-Accept Decay (Eq. 7) | 0.95 | 0.95 ✓ |
| `θ_0` | Untere Schwelle (Eq. 6) | 0.05 | 0.05 ✓ |
| `θ_init` | Warm-up-Startwert (Alg. 2 Z. 11) | 0.10 | n/a (nicht spezifiziert) |
| `min_overlap_pixels` | Fail-safe-Schwelle |‎ 1000 | n/a |

**Wichtig**: Die Defaults entsprechen Paper Sec. 3.4. Wer VINGS-spezifische
Werte (z.B. größeres W weil Tracker-KFs spärlich kommen, oder aggressiveren
γ) braucht, muss explizit in der Config überschreiben — siehe
`configs/local/agz/adaptive_kf/` und `configs/local/amtown03/exp/adaptive_kf/`.

## Pose-Konvention

c2w (wie in den anderen Selektoren):

```
p_world = R · p_cam + t
```

Relative Pose `kf cam → cur cam` (für Forward-Warp KF→Cur):

```
R_kc = R_cur^T · R_kf
t_kc = R_cur^T · (t_kf - t_cur)
```

## Failsafes

| Trigger | Verhalten |
|---|---|
| `last_kf_gray is None` (erster Frame) | force-accept, seed KF (gray + depth + pose) |
| `rgb is None` | force-accept (kein Error berechenbar, KF nicht aktualisiert) |
| Alle KF-Depth invalid | gibt `(0, 0, 0)` zurück → `|M| < min_overlap` → fail-safe |
| `|M| < min_overlap_pixels` | force-accept, push synthetic `e = 1.0` in E |
| `force_accept_all = true` | accept jeden Frame, logge e/θ (Diagnose-Modus) |

## Tuning-Workflow

Mit `force_accept_all: true` einmal laufen lassen, dann e-Verteilung
analysieren.

1. Run auf einem Repräsentativ-Dataset (`smallcity_200` oder `ntu_eee_03_200`).
2. `grep "frame_select" run.log` → e-Werte sammeln.
3. Aus der e-Verteilung:
   - `θ₀` ← unteres Quartil (Q25)
   - `k` ← so wählen, dass `μ + k·σ` zu ~25-30% Accept-Rate führt
4. `γ` empirisch: 0.95 ist Paper-Wert; 0.85/0.70 zunehmend aggressiver
   (kürzerer Refractory-Period nach Accept).
5. `W = 5` ist Paper-Wert; bei sehr seltenen Tracker-KFs ggf. W=3 oder
   W=10, mit klarer Begründung im Methodenkapitel.

**Wichtig**: Paper macht Grid-Search auf Bonn-Validation. Für VINGS wäre
vergleichbare Validation sinnvoll, ist aber out-of-scope dieser BA — die
Paper-Defaults sind defensiv genug als Startpunkt.

## Datensatz-spezifische Kalibration

Paper testet auf **kurzen indoor RGB-D-Sequenzen** (7Scenes, NRGBD, Sintel,
Bonn) mit moderaten Baselines und KF-Kompressionsraten von ~90 %.

Bei VINGS-smallcity/NTU outdoor:
- `θ₀ = 0.05` ist möglicherweise zu niedrig (mehr Texturwechsel pro Frame)
- `W = 5` bei 10-Hz-Tracker-KFs deckt nur 0.5 s — bei langsamen Szenen evtl.
  zu reaktiv, also gerne W = 10-15 überschreiben
- Bei rapid motion kommen Fail-Safes (`|M| < min_overlap`) öfter; das ist
  by design

## Was im Methodenkapitel der BA stehen sollte

Die Übernahme ist jetzt **vollständig paper-faithful** (Stand 2025-11-26):

1. **Algorithmus 2** (momentum-aware adaptive threshold) ist verbatim aus dem
   Paper. Eq. (4) Moving-Average, Eq. (5) Std, Eq. (6) Threshold-Update,
   Eq. (7) Post-Accept-Decay, Warm-up-Linearinterpolation aus Algorithmus 2
   Z. 11.
2. **Algorithmus 1** wurde reproduziert, weil das offizielle Repo
   (`jhakrraman/Adaptive_Keyframe_Selection`) ihn nicht enthält. Eq. (1)-(3)
   sind verbatim umgesetzt; `WarpFrame` ist als Forward-Splat mit Z-Buffer
   implementiert (Paper spezifiziert keine konkrete Implementierung, sagt aber
   explizit "project 3D points derived from D_k into the image plane of I_t").
3. **Hyperparameter** `α=0.7, β=0.3, W=5, k=1.5, γ=0.95` entsprechen
   Paper Sec. 3.4. `θ_init = 0.10` ist ein Zusatz-Default (Paper nennt das
   Symbol in Algorithmus 2 Z. 11 aber gibt keinen Wert).
4. Robotersteuerung / Voronoi / aktive Exploration aus ActiveSplat sind nicht
   übernommen (vgl. `docs/ACTIVESPLAT.md`).

### Bewusste Abweichungen vom Paper

- **Pixel-Skalierung [0, 1]** statt [0, 255]; konsistent mit der Größenordnung
  von Paper-θ₀ = 0.05.
- **Mask-Multiplikation für SSIM** statt korrekt-masked-SSIM (skimage
  unterstützt das nicht); minimaler Bias.
- **Forward-Splat mit Z-Buffer per farthest-first-sort** statt OpenGL-style
  Z-Buffer-Array. Mathematisch äquivalent.
- **`min_overlap_pixels`-Fail-safe** zusätzlich; Paper erwähnt nur in Sec. 5
  qualitativ "fail-safe trigger of a new keyframe" bei verschwindender Maske.
- **Fail-safe schreibt `e = 1.0` in E** damit die Statistik nicht stale wird.
  Paper sagt nichts dazu.

## Code-Pointer

| Datei | Inhalt |
|---|---|
| `scripts/vings_utils/adaptive_kf_selector.py` | Selektor + Config + Smoketest |
| `scripts/vings_utils/selector_factory.py` | Registrierung als `kind: adaptive_kf` |
| `scripts/vings_utils/adaptive_kf_selector.py:__main__` | Standalone-Smoke-Test (synthetic moving camera) |

Smoketest:

```bash
PYTHONPATH=scripts python scripts/vings_utils/adaptive_kf_selector.py
```

Erwartete Ausgabe: ~3-8/30 Accepts, Warm-up sichtbar (θ-Interpolation),
nach |E|≥5 Sprung auf `μ + k·σ`, Decay (·0.95) nach Accept.

## Beispiel-Config (Paper-Defaults)

```yaml
frame_selector:
  kind: adaptive_kf
  # Algorithmus 2 (Paper Sec. 3.4)
  theta0: 0.05            # Untere Schwelle (Eq. 6)
  theta_init: 0.10        # Warm-up-Startwert (Alg. 2 Z. 11)
  window_size: 5          # W
  sensitivity: 1.5        # k
  decay: 0.95             # γ (Eq. 7)

  # Algorithmus 1 (Paper Eq. 3)
  w_photo: 0.7            # α
  w_ssim: 0.3             # β
  min_overlap_pixels: 1000

  # Depth-Gate
  min_depth: 0.2
  max_depth: 35.0

  # Diagnose
  force_accept_all: false
```

## Beispiel-Config (VINGS-Aerial mit größerem Window)

Wenn Tracker-KFs spärlich kommen (z.B. AGZ/amtown03 mit ~10-20 KFs/s) und du
ein ruhigeres Statistik-Fenster willst:

```yaml
frame_selector:
  kind: adaptive_kf
  theta0: 0.05
  theta_init: 0.10
  window_size: 30         # explizite Abweichung vom Paper-Default W=5
  sensitivity: 2.0        # konservativer
  decay: 0.85
  w_photo: 0.7
  w_ssim: 0.3
  min_overlap_pixels: 1000
  min_depth: 5.0          # Aerial
  max_depth: 200.0
  force_accept_all: false
```

Begründung für die Abweichung gehört dann ins Methodenkapitel.
