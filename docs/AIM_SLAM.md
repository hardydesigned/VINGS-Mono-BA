# AIM-SLAM-Selector (SIGMA-Modul)

## In einfachen Worten

Stell dir vor, du machst gerade eine Stadtführung mit der Kamera und sollst
**entscheiden, welche Fotos du wirklich behalten willst**, weil Speicher knapp
ist. Drei Fragen helfen dir intuitiv:

1. **„Sehe ich gerade ungefähr dasselbe wie auf dem letzten Foto?"** Wenn ja,
   ist das neue Foto überflüssig — weg damit.
2. **„Würde das neue Foto mir dabei helfen, mein letztes besser zu verstehen?"**
   Wenn es etwa eine schwer einsehbare Ecke aus einem anderen Winkel zeigt:
   behalten, das macht das letzte Foto eindeutiger.
3. **„Stimmt das, was ich auf dem neuen Foto sehe, mit dem überein, was ich
   anhand des letzten *vorhergesagt* hätte?"** Wenn ja, hat sich kaum etwas
   verändert — weg damit. Wenn nein (Bewegung war groß), behalten.

Genau das macht der AIM-SLAM-Selector. Statt Fotos prüft er
**Tracker-Keyframes**, statt Bauchgefühl benutzt er drei Maße:

| Frage | Maß im Code |
|---|---|
| 1. Wie ähnlich? | **Voxel-Overlap** zwischen den 3D-Punkten des aktuellen Frames und denen des letzten Map-KFs |
| 2. Wie viel neue Information? | **Information-Gain** als Reduktion der Unsicherheit (Kovarianz) des letzten KFs, wenn man den neuen Frame dazunimmt |
| 3. Wie überraschend? | **Reduced-Chi-Square** der Reprojektions-Residuen — größer als 1 heißt „überraschender als das Rauschen erlaubt" |

Wir akzeptieren den neuen Frame nur, wenn er **alle drei Bedingungen erfüllt**:
*nicht zu ähnlich* UND *bringt genug neue Information* UND *ist überraschend
genug*. Das spart Mapping-Aufrufe ohne Pose-Qualität zu verlieren.

## Paper

Jeon, J., Seo, D.-U., Lee, E. M., Myung, H.,
„**AIM-SLAM: Dense Monocular SLAM via Adaptive and Informative Multi-View
Keyframe Prioritization with Foundation Model**", arXiv:2603.05097v3, 2026
(KAIST). Sec. III.C beschreibt das **SIGMA-Modul** (Selective Information-
and Geometric-aware Multi-view Adaptation). Im Paper dient SIGMA dazu, ein
Multi-View-Subset für die VGGT-Foundation-Model-Inferenz auszuwählen; wir
übernehmen nur die **Selektionslogik** (nicht VGGT, nicht die Sim(3)-
Optimierung, nicht das Loop-Closure) und übersetzen sie in unseren binären
„Diesen Tracker-KF mappen ja/nein"-Slot (analog zu den anderen Selectoren in
[selector_factory.py](../scripts/vings_utils/selector_factory.py)).

## Algorithmus (Paper, drei Stages)

Sei `I_k` der letzte gemappte Keyframe und `I_f` der aktuelle Tracker-KF
(„current frame").

**Stage 1 — Geometry-based Initial Subset Construction (Eq. 1)**
Voxel-Overlap zwischen `I_k` und einem Kandidaten-View `I_i`:

```
O(I_k, I_i) = | v(I_k) ∩ v(I_i) |
```

wobei `v(I)` die Menge aller Voxel ist, die `I` sieht. Top-N nach `O` bildet
das initiale Kandidaten-Set `W_v`.

**Stage 2 — Information-driven Re-Ranking (Eq. 2 + 3)**
Für jeden Punkt `x_k` aus `I_k` mit Prior-Kovarianz `P_k⁻` updated ein
EKF-Schritt diese Kovarianz, wenn man View `I_j` hinzunimmt:

```
P_k⁺ = P_k⁻ − P_k⁻ J_rᵀ (R + J_r P_k⁻ J_rᵀ)⁻¹ J_r P_k⁻      (Eq. 2)
```

`J_r` ist der Jacobian des Ray-Residuums beim Reprojezieren von `x_k` in
`I_j`, `R` die Mess-Kovarianz.

Information-Gain als Entropie-Reduktion (alle Punkte summiert):

```
Γ(I_k, I_j) = Σ_x 0.5 · log( det(P_k⁻) / det(P_k⁺) )         (Eq. 3)
```

`W_v` wird absteigend nach `Γ` umsortiert.

**Stage 3 — Adaptive Subset Activation (Eq. 4 + Hybrid-Residual Eq. 5)**
Reduced-Chi-Square auf dem **Hybrid-Residual** aus Eq. 5 (Ray + Pixel
gestackt), whitened mit den jeweiligen Mess-Sigmas. Pro adjacent pair
`(I_i, I_j)` ist das Per-Korrespondenz-Residual:

```
r_ij = [ Ψ_ray(X_i|i) − Ψ_ray(T^i_j X_j|j),         (3D, Eq. 5 erster Term)
         Ψ_π (K_i, X_i|i) − Ψ_π (K_i, T^i_j X_j|j) ] (2D, Eq. 5 zweiter Term)
```

mit `Ψ_ray(X) = X / ‖X‖` (Einheitssphäre) und `Ψ_π` als Pinhole-Projektion.
Whitened als `b = [r_ray/σ_ray, r_pix/σ_pix]` mit `σ_ray = σ_pix / f`.

Eq. 4 liefert dann:

```
κ = bᵀb / ( M − rank(A) )                                    (Eq. 4)
```

`κ ≤ 1` ⇒ Optimierung ist stabil, default-3-View-Subset reicht.
`κ > 1`  ⇒ instabil, sukzessive einen weiteren Kandidat-View aus `W_v`
zuschalten und `κ` neu evaluieren.

## Übersetzung auf den binären VINGS-Slot

Der Paper-Slot ist multi-view (welche N Frames füttern wir VGGT?). VINGS hat
einen binären Slot (mappen wir den aktuellen Tracker-KF?). Die Übersetzung:

| Paper | VINGS-Adaption |
|---|---|
| `O(I_k, I_i)` über alle Datenbank-KFs → Top-N | `O(I_f, I_k)` einmalig zwischen aktuellem Frame und letztem Map-KF. **Skip** wenn `O / |v(I_f)| > overlap_thresh` (zu redundant). |
| `Γ(I_k, I_j)` als Reranking-Schlüssel für `W_v` | `Γ(I_k, I_f)` als **Akzeptanz-Schwellwert**: accept wenn `Σ Γ > gain_thresh`. |
| `κ` regelt iterative Subset-Erweiterung | `κ` als **Stabilitäts-Gate**: accept nur wenn `κ > chi_thresh` (Tracker-Residuen größer als Rauschen → echte Bewegung, lohnt sich zu mappen). |
| Messkovarianz `R` aus VGGT-Confidence | `R = σ²_pix · I_2` (fixer Pixel-Rausch-Term). |
| Prior `P_k` aus VGGT-Pointmap-Confidence | `P_k = σ²_d(u,v) · I_3` aus DBAF `depths_cov` (falls vorhanden), sonst `P_k = σ²_default · I_3`. |
| 3×3 invers-Projektion-Jacobian für `J_r` | 2×3 Standard-Pinhole-Jacobian `∂π/∂X` (übliche EKF-VIO-Formulierung); Tiefe dominiert die 3D-Kovarianz, deswegen kein Genauigkeitsverlust. |
| Korrespondenz für Eq. 5 via VGGT-Pointmap-Ray-Matching | **ORB-BFMatch zwischen letztem KF und aktuellem Frame** (paper-treuer Pfad, wenn `rgb` verfügbar). Wichtig: Korrespondenz ist damit **pose-unabhängig** — Voraussetzung dafür, dass κ tatsächlich Pose-Konsistenz testet (nicht Tiefen-Konsistenz unter gegebener Pose). Fallback auf Reprojection-basierte Korrespondenz mit bilinearem Depth-Sampling wenn cv2 fehlt oder ORB zu wenig Matches findet. |
| rank(A) in Eq. 4 aus Sim(3)-Variablen | rank(A) = 0, weil VINGS die Pose nicht optimiert (DBAF-Output ist fix). DoF = M. |

**Decision-Rule (verbatim umgesetzt):**

```python
accept = (overlap < overlap_thresh)
     and (info_gain / N_rays > gain_thresh_per_ray)
     and (chi_square > chi_thresh)      # nur wenn use_chi_square
```

Wir threshholden auf den **per-ray-Wert** `Γ/N`, nicht auf das absolute `Γ`.
Der Absolutwert skaliert linear mit `n_rays_score` und ist datensatz-
/Intrinsics-abhängig; `Γ/N` ist die robustere Tuning-Achse.

mit Failsafes:

- **Erster Frame** → seed, accept.
- **`overlap < min_overlap_ratio`** → Tracker-Stress (Szenenwechsel,
  Re-Initialisierung); force-accept, sonst hängen wir.
- **`force_accept_all: true`** → Diagnose-Modus, loggt nur die drei
  Statistiken.

## Variablen-Tabelle

| Symbol | Bedeutung | Implementierung |
|---|---|---|
| `v(I)` | Menge aller Voxel-IDs `(i,j,k)`, die `I` sieht | `_voxel_set(depth, t, R)` — Subsample N Pixel, Backproject zu Welt, hash via `np.floor(X / voxel_size)` |
| `O` | Voxel-Overlap-Ratio `|v(I_f) ∩ v(I_k)| / |v(I_f)|` | Set-Schnitt + Längen-Division |
| `P_k⁻` | Prior-3D-Kovarianz für Punkt `x_k` | `σ²_d · I_3` mit `σ²_d` aus `depth_cov[u,v]` (clipped) oder Default |
| `J_r` | Reprojektions-Jacobian `∂π/∂X` in der Form `[[f_x/Z, 0, -f_x·X/Z²], [0, f_y/Z, -f_y·Y/Z²]]` | Per-Punkt vektorisiert in `_pinhole_jacobian()` |
| `R` | 2×2 Mess-Kovarianz im Pixel-Raum | `σ²_pix · I_2` (Pixel-Rausch-Term, fix) |
| `P_k⁺` | Posterior nach EKF-Update | `_ekf_update()` vektorisiert über alle Sample-Punkte |
| `Γ` | Summe der Log-Determinanten-Differenzen | `0.5 · Σ log(det(P_k⁻)/det(P_k⁺))` |
| `r_ij` | Hybrid-Residuum aus Eq. 5 (Ray + Pixel gestackt, 5D pro Korrespondenz) | `_stage3_chi_square()`; Korrespondenz via Reprojektion mit DBAF-Pose |
| `b` | Whitened Residuen-Vektor | `[r_ray/σ_ray, r_pix/σ_pix]` mit `σ_ray = σ_pix / f̄` |
| `κ` | Reduced-Chi-Square | `bᵀb / (M − chi_dof_offset)`, `chi_dof_offset = 0` weil keine Pose-Optimierung |

## Sensitivität / Tuning

| Parameter | Wirkung | Wo nachschärfen |
|---|---|---|
| `voxel_size` | Kleiner = feinere Overlap-Diskriminierung, aber mehr Voxel und längere Set-Operationen. 5-15 cm sinnvoll | Standardwert 10 cm; bei sehr feinen Detail-Sequenzen evtl. 5 cm |
| `overlap_thresh` | Senken (z. B. 0.5) = mehr Akzepts (toleranter gegen Redundanz). Erhöhen (z. B. 0.85) = nur sehr unterschiedliche Frames | Erste Sweep-Achse. Default 0.7 |
| `gain_thresh_per_ray` | Skalen-abhängig (nats/ray). Höher = strenger. Hängt direkt von `prior_sigma_d`, `pixel_sigma` und Brennweite ab | Erst `force_accept_all: true` laufen lassen, Median(`Γ/N`) aus dem Log ablesen, Threshold knapp darunter setzen. Box-Room-Smoketest zeigt ~4 nats/ray; Default 0.5 ist konservativ |
| `chi_thresh` | Strikt 1.0 ist Paper-Default. `0.5` lockert (mehr Akzepts), `2.0` strafft | Sequenz-spezifisch; bei lauten Tracker-Posen kann `κ` chronisch hoch sein → höher setzen |
| `pixel_sigma` | Rausch-Annahme im Pixel-Raum. Höher = `κ` sinkt, `Γ` sinkt | 1 px ist der Klassiker für KLT/ORB-Tracker |
| `prior_sigma_d` | Default-Tiefen-Unsicherheit (Meter), wenn `depths_cov` nicht durchgereicht wird | 10 cm für indoor (smallcity), 30 cm für outdoor/luftgestützt |
| `n_rays_score` | Punkte für Stage 2 + 3. Mehr = stabiler, aber linear teurer | 256 ist der Sweet-Spot; <128 wird verrauscht |
| `n_voxel_samples` | Punkte für Stage 1. Mehr = präzisere Overlap-Schätzung | 1024 reicht; bei sehr dichter Tiefe darf hoch |

**Empfohlener Tuning-Workflow:**

1. `force_accept_all: true` setzen, einen kurzen Run (z. B. 200 Frames).
2. In den Logs `O`, `Γ/N`, `κ` ablesen (Median + Spreizung).
3. `force_accept_all: false`, dann `overlap_thresh` ≈ Median(`O`) + 0.1,
   `gain_thresh_per_ray` ≈ Median(`Γ/N`), `chi_thresh` ≈ 1.0.
4. Ziel-Akzeptanz-Rate prüfen; im Sweep die drei Schwellen orthogonal
   verschieben.

## Code-Pointer

| Datei | Inhalt |
|---|---|
| [`scripts/vings_utils/aim_slam_selector.py`](../scripts/vings_utils/aim_slam_selector.py) | `AimSlamConfig`, `AimSlamScore`, `AimSlamSelector` + Smoketest |
| [`scripts/vings_utils/selector_factory.py`](../scripts/vings_utils/selector_factory.py) | Registriert `kind: aim_slam` |
| [`configs/local/smallcity/aim_slam/`](../configs/local/smallcity/aim_slam/) | Beispiel-Configs |

Standalone-Smoketest:

```bash
PYTHONPATH=scripts python scripts/vings_utils/aim_slam_selector.py
```

Erwartung: Box-Room-Szene, 15 Frames, Seed-KF wird akzeptiert, danach gehen
die meisten Frames durch Stage 3 mit `κ ≈ 0` (synthetische Tiefe ist perfekt
konsistent — der Test funktioniert wie erwartet). Bei `O = 0` (komplett
neuer Bereich) greift die `min_overlap_ratio`-Force-Accept-Klausel.
Smoketest nutzt `rgb=None` → Reprojection-Fallback (`chi_source =
"reproject"`); in echten Sequenzen mit rgb wird der ORB-Pfad aktiv.

## Beispiel-Config

```yaml
frame_selector:
  kind: aim_slam
  # Stage 1: Voxel-Overlap
  voxel_size: 0.10
  overlap_thresh: 0.70        # Skip wenn shared/curr > thresh (zu redundant)
  min_overlap_ratio: 0.05     # Failsafe: force-accept bei Szenenwechsel
  n_voxel_samples: 1024
  # Stage 2: EKF-Information-Gain (Eq. 2-3)
  gain_thresh_per_ray: 0.5    # Accept wenn Γ/N > thresh (nats/ray)
  n_rays_score: 256
  pixel_sigma: 1.0            # Mess-Rauschen σ_pix
  prior_sigma_d: 0.10         # σ_d Default wenn depths_cov fehlt
  cov_clip: 4.0               # Clip-Range für depths_cov
  # Stage 3: Reduced-Chi-Square (Eq. 4 + Hybrid-Residual Eq. 5)
  use_chi_square: true
  chi_thresh: 1.0             # κ > 1 → residuen über rauschen → accept
  chi_dof_offset: 0           # M - rank(A); 0 weil keine Pose-Optimierung
  chi_ray_weight: 1.0         # optionales Gewicht für Ray-Term (Eq. 5 1. Term)
  chi_pix_weight: 1.0         # optionales Gewicht für Pixel-Term (Eq. 5 2. Term)
  # ORB-Korrespondenz (paper-treu, pose-unabhängig). Bei rgb=None,
  # fehlendem cv2 oder < chi_min_matches Matches → Reprojection-Fallback.
  chi_orb_n_features: 800
  chi_min_matches: 20
  chi_max_disparity_px: 200.0
  # Tiefen-Gate
  min_depth: 0.2
  max_depth: 35.0
  # Diagnose
  force_accept_all: false
```

## Bewusste Abweichungen vs. Paper (BA-Methoden-Recap)

Was **verbatim** aus dem Paper kommt:

- Voxel-Overlap-Formel (Eq. 1) inkl. der Idee, Keyframe-Visibility statt
  Punkt-Landmarks zu hashen.
- EKF-Kovarianz-Update (Eq. 2) und Log-Determinant-Information-Gain (Eq. 3).
- Reduced-Chi-Square-Stabilitätstest (Eq. 4) **inkl. Hybrid-Residual aus
  Eq. 5** (Ray-Term auf der Einheitssphäre + Pixel-Reprojektions-Term,
  gestackt zu einem 5D-Residuum pro Korrespondenz), und der Schwellwert
  `κ = 1`.

Was **adaptiert** wurde:

- **Slot-Übersetzung**: Multi-View-Reranking → binäres Accept/Skip mit
  drei AND-verknüpften Schwellen. Grund: VINGS hat keinen Multi-View-VGGT-Slot.
- **Mess-Kovarianz**: 2×2 Pixel-Rauschen statt VGGT-getriebener 3×3-Kovarianz.
  Grund: kein Foundation-Model-Output verfügbar; der Pinhole-Jacobian ist
  Standard-EKF-VIO.
- **Prior-Kovarianz**: 3×3 isotrope σ²_d aus DBAF `depths_cov` statt
  VGGT-Confidence. Grund: nutzt die Tracker-seitige Unsicherheit, die wir
  ohnehin haben.
- **Chi-Square-DoF**: `chi_dof_offset = 0` (Default), weil VINGS die Pose
  nicht optimiert — die DBAF-Pose ist gegeben, `rank(A) = 0`, also `dof = M`.
  Im Paper sind das die Sim(3)-Variablen pro Frame-Pair (7 pro Paar). Der
  Offset bleibt als Tuning-Knob konfigurierbar, ist aber bei `M ≫ 7` ohnehin
  numerisch irrelevant.
- **Korrespondenz**: Paper nutzt VGGT-Pointmap-Ray-Matching, um die
  Korrespondenz `(p_i, p_j)` zwischen Frames aufzubauen. Wir haben kein
  Foundation-Model-Matching; stattdessen nutzen wir **ORB-Features +
  BFMatch (Hamming, Crosscheck)** zwischen letztem KF und aktuellem Frame.
  Das ist der entscheidende Paper-Treue-Punkt: die Korrespondenz ist
  **pose-unabhängig**, dadurch testet Eq. 4 tatsächlich, ob die DBAF-
  Pose-Hypothese mit den unabhängig gefundenen Matches konsistent ist —
  und nicht nur die Tiefen-Konsistenz unter gegebener Pose. Fallback auf
  Reprojection-basierte Korrespondenz wenn `rgb` fehlt, cv2 nicht
  importierbar, oder ORB < `chi_min_matches` Matches findet; in diesem
  Fall misst κ einen schwächeren *joint-Tiefen-Pose*-Konsistenz-Proxy
  (in `score.chi_source = "reproject"` markiert).
- **Bilineares Depth-Sampling**: an Sub-Pixel-Match-Koordinaten (vs.
  Nearest-Neighbor) — wichtig für die ORB-Subpixel-Genauigkeit und
  konsistenter mit dem kontinuierlichen Pointmap-Sampling im Paper.

Was **weggelassen** wurde:

- VGGT-Foundation-Model-Inferenz (gehört zur Optimierung, nicht zur
  Selektion).
- Multi-View-Sim(3)-Optimierung (gehört zum Backend, nicht zur Selektion).
- Loop-Closure mit DINOv2-Tokens (VINGS hat eigenes Loop-Modul).
- Iterative `W_v`-Erweiterung mit Re-Evaluierung von `κ` (irrelevant im
  binären Slot).

Für die Methoden-Sektion: AIM-SLAM ist im Vergleich zu VISTA (rein
geometrisch), MM3DGS (Covisibility + Quality), Game-KFS (composite
multi-agent), NURBS-LVI (Feature + Sektor-Migration) und Adaptive-KF
(Hybrid-Error + Momentum) der Selector mit dem **explizitesten
Unsicherheits-Modell**: er ist der einzige, der die *Kovarianz*-Reduktion
quantifiziert statt nur Reward-Heuristiken zu summieren. Das ist sein
Alleinstellungsmerkmal in der Vergleichstabelle.
