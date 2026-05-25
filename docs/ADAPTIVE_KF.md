# Adaptive-KF-Selector

Momentum-Aware Adaptive Keyframe-Selektor nach **"Adaptive Keyframe Selection
for Scalable 3D Scene Reconstruction in Dynamic Environments"** (ROBOVIS 2026
Submission, anonyme Autoren). Diese Doku erklärt: was im Paper steht, was im
zugehörigen GitHub-Repo (`jhakrraman/Adaptive_Keyframe_Selection`) *fehlt*, und
wie wir die Lücke geschlossen haben.

## Was wir aus dem Paper übernehmen

### Algorithmus 2 — Adaptive Schwelle (verbatim)

```
für jeden Frame t:
    e_t = ComputeHybridError(f_t, f_last_kf)              # Algorithmus 1
    E.append(e_t)
    if |E| ≥ W:                                           # Eq. 6
        θ = max(θ₀,  μ_t + k · σ_t)
    else:                                                 # Warm-up
        θ = θ₀ · (t/W)  +  θ_init · (1 - t/W)
    if e_t > θ:
        accept f_t
        θ ← γ · θ                                         # Eq. 7
```

Drei wirksame Knöpfe:

| Symbol | Bedeutung | Default |
|---|---|---|
| `W`  | Sliding-Window-Länge (Statistik) | 30 |
| `k`  | Sensitivity (Multiplikator auf σ) | 2.0 |
| `γ`  | Decay nach Accept, ∈ (0, 1]      | 0.85 |
| `θ₀` | Untere Schwelle (Grid-Search im Paper) | 0.05 |

### Algorithmus 1 — `ComputeHybridError`

Im Paper-Text nur **qualitativ** beschrieben:

> „assumes the availability of depth data (RGB-D) for the warping module ...
>  in cases of extremely rapid motion where the overlap mask M becomes
>  negligible, the warping-based error may lose its geometric grounding,
>  though this typically results in a 'fail-safe' trigger of a new keyframe."

> Ablation Tab. 5: „Removing the photometric error forces reliance solely
>  on SSIM" / „excluding SSIM makes the process brittle to illumination
>  changes"

Keine Formel, keine Gewichte, kein Pseudocode.

## Wichtig: Was im GitHub-Repo *nicht* steht

Das offizielle Repo (`jhakrraman/Adaptive_Keyframe_Selection`, Stand
ROBOVIS-2026-Submission) enthält **weder Algorithmus 1 noch Algorithmus 2**.

| Paper-Komponente | Im Repo? |
|---|---|
| Algorithmus 1: Photometric + SSIM via Warping | ❌ nicht implementiert |
| Algorithmus 2: `θ = max(θ₀, μ + k·σ)` | ❌ nicht implementiert |
| Sliding-Window-Statistik E | ❌ |
| Decay-Faktor γ | ❌ |
| Warm-Up-Phase | ❌ |
| Overlap-Maske M | ❌ |
| Depth-basiertes photometric warping | ❌ |

Das Repo besteht aus vier nahezu identischen ROS2-Wrappern für CUT3R-Inferenz.
Die einzige Selektor-Logik ist eine hardcodierte Pose-Heuristik:

```python
# cut3r_package_improved.py:692-717
rotation_threshold = 10.0       # Grad
translation_threshold = 0.2     # Meter
# accept iff Δrotation > 10° OR Δtranslation > 20cm
```

…plus ein optionaler Whole-Image-SSIM-Vergleich für „scene_change_detection".
Das ist **näher an den Baselines, gegen die das Paper sich vergleicht**
(„inertial-based [11]", „optimization-based [14]", Tab. 5) als am
publizierten Verfahren.

## Unsere Rekonstruktion von Algorithmus 1

Aus dem Paper-Text rekonstruiert. Standard depth-basiertes Backward-Warping:

```
1.  Für jedes Pixel (u_c, v_c) im current frame mit gültiger Tiefe d_c:
        P_cur   = K⁻¹ · [u_c, v_c, 1]ᵀ · d_c            # back-project current cam
        P_kf    = R_kf^T · R_cur · P_cur                # current → last_kf
                 + R_kf^T · (t_cur - t_kf)
        u_kf, v_kf = π(K · P_kf)                        # project into last_kf

2.  M = { (u_c, v_c) : d_c valid ∧ P_kf.z > 0
                       ∧ (u_kf, v_kf) ∈ image bounds }

3.  I_warped = cv2.remap(I_last_kf, u_kf, v_kf, bilinear)

4.  e_photo  = mean_M |I_warped - I_current| / 255
    e_ssim   = 1 - SSIM(I_current * M, I_warped * M)    # full-image, mask-gefüllt

5.  e = w_photo · e_photo + w_ssim · e_ssim
```

| Parameter | Default | Begründung |
|---|---|---|
| `w_photo` | 0.85 | Standard-Ratio Photo:SSIM in dichten SLAM-Photometric-Losses |
| `w_ssim`  | 0.15 | dito |
| `min_overlap_pixels` | 1000 | unter 1k Pixel `|M|` → Fail-safe-Accept (rapid motion / depth-glitch) |

**Backward Warping** statt Forward Warping, weil:
- Dense, regelmäßige Sampling-Grid auf dem Current-Frame
- Keine Z-Buffer-Konflikte bei Forward-Splat
- `cv2.remap` ist GPU-schnell + bilinear interpoliert sauber

**SSIM mit Mask-Multiplikation** statt korrekt-masked-SSIM, weil:
- `skimage.metrics.structural_similarity` unterstützt keine Masken
- Mask-Zero macht out-of-overlap-Regionen identisch (1-SSIM dort = 0)
- Approximation gut genug für den Score-Mittelwert

## Pose-Konvention

c2w (wie in den anderen Selektoren):

```
p_world = R · p_cam + t
```

Relative Pose `current cam → last_kf cam`:

```
R_rel = R_kf^T · R_cur
t_rel = R_kf^T · (t_cur - t_kf)
```

## Failsafes

| Trigger | Verhalten |
|---|---|
| `last_kf is None` (erster Frame) | force-accept, seed last_kf |
| `rgb is None` | force-accept (kein Error berechenbar) |
| `|M| < min_overlap_pixels` | force-accept, push synth-error=1.0 in `E` |
| `force_accept_all=true` | accept jeden Frame, logge e/θ (Diagnose-Modus) |

## Tuning-Workflow

Wie bei NURBS-LVI: zuerst `force_accept_all: true` setzen, dann die
`e`-Verteilung im Profiling-Log anschauen.

1. Run mit `force_accept_all: true` auf einem Repräsentativ-Dataset
   (`smallcity_200` oder `ntu_eee_03_200`).
2. `grep "frame_select" run.log` → e-Werte sammeln.
3. Aus der e-Verteilung:
   - `θ₀` ← unteres Quartil (Q25) der e-Werte
   - `k` ← so wählen, dass `μ + k·σ` zu ungefähr 25-30% Accept-Rate führt
4. `γ` empirisch: 0.85 ist defensiv, 0.7 aggressiver (mehr KFs nach Acceptance).
5. `W` skaliert mit Frame-Rate: bei 30 FPS sind 30 ≈ 1s Adaption.

**Wichtig**: das Paper macht Grid-Search auf Bonn-Validation (ihre eigene
Test-Suite). Für VINGS wäre eine vergleichbare Validation-Splits sinnvoll, aber
out-of-scope dieser BA — defensive Defaults oben sollten als Startpunkt
reichen.

## Sensitivität: Datensatz-spezifische Kalibration

Das Paper testet auf **kurzen indoor RGB-D-Sequenzen** (7Scenes, NRGBD,
Sintel, Bonn) mit moderaten Baselines. `KFCR` von ~90% (jedes 10. Frame als
KF) ist dort gut.

Bei VINGS-smallcity/NTU mit längeren outdoor-Pfaden:
- Default-θ₀ = 0.05 ist möglicherweise zu niedrig (mehr Texturwechsel pro Frame)
- W = 30 bei 10-Hz-Tracker-KFs deckt 3s ab — ausreichend für die Statistik
- Bei rapid motion kommen Fail-Safes (M < min_overlap) öfter; das ist okay

## Was im Methodenkapitel der BA stehen sollte

Die Übernahme ist **partiell und transparent**:

1. **Algorithmus 2** (momentum-aware adaptive threshold) ist verbatim aus dem
   Paper. Verwendet werden Eq. 6 (`θ = max(θ₀, μ + k·σ)`) und Eq. 7
   (`θ ← γ·θ` Decay).
2. **Algorithmus 1** ist rekonstruiert, weil das Paper keine Formel und das
   offizielle Repo keinen Code liefert. Wir verwenden Standard-RGB-D-Backward-
   Warping mit `cv2.remap`, L1-Photometric, mask-gefülltes SSIM.
3. **Gewichte `w_photo = 0.85, w_ssim = 0.15`** sind aus der Ablation der
   verwandten Arbeiten übernommen (DSO, MonoGS); das Paper macht hierzu
   keine Angabe.
4. Robotersteuerung / Voronoi / aktive Exploration aus ActiveSplat sind nicht
   übernommen (vgl. `docs/ACTIVESPLAT.md`).

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

Erwartete Ausgabe: ~10/30 Accepts, Warm-up sichtbar (θ-Interpolation), nach
Window-Füllung θ-Sprung auf `μ + k·σ`, Decay nach Accept.

## Beispiel-Config

```yaml
frame_selector:
  kind: adaptive_kf
  # Algorithmus 2
  theta0: 0.05            # Untere Schwelle
  theta_init: 0.10        # Warm-up-Startwert
  window_size: 30         # W
  sensitivity: 2.0        # k
  decay: 0.85             # γ

  # Algorithmus 1
  w_photo: 0.85
  w_ssim: 0.15
  min_overlap_pixels: 1000

  # Depth-Gate
  min_depth: 0.2
  max_depth: 35.0

  # Diagnose
  force_accept_all: false
```
