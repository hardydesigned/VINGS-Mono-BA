# Game-KFS-Selector

Keyframe-Auswahl nach **Chen S., Yang B., Wang C. et al., „Game-KFS:
Game-Theory-Inspired Keyframe Selection for Hybrid Representation Visual
SLAM", IEEE Robotics and Automation Letters 2025**, Sec. III. Originalpaper-
Framework ist Photo-SLAM (ORB-SLAM3 + 3D-Gaussian-Mapper). Hier in VINGS-Mono
sitzt der Selector im selben Plugin-Slot wie `vista` / `nurbs_lvi` / `mm3dgs`
(siehe `scripts/vings_utils/selector_factory.py`).

Diese Doku beschreibt die **paper-nahe Variante** (zweite Iteration, siehe
„Versionierung" weiter unten).

## Idee

Statt einer einzelnen Heuristik werden zwei konzeptuelle Agenten gegeneinander
gewogen:

| Agent | Was er „möchte" | Sub-Scores |
|---|---|---|
| **FRA** (Field) | Frames, die das Rendering verbessern, Lücken schließen, Unsicherheit senken | `L_uncert`, `L_render`, `L_covis` |
| **DRA** (Discrete) | Frames, die das Feature-Tracking stabil halten und Bewegungs-Inkonsistenzen abfangen | `L_assoc`, `L_flow`, `L_motion` |

Sie werden über einen online adaptierten Gewichtsfaktor λₜ zu einem skalaren
Score kombiniert:

```
A_t   = β1·L_uncert + β2·L_render + β3·L_covis             # Eq. 3
B_t   = α1·L_assoc  + α2·L_flow   + α3·L_motion            # Eq. 9
λ*    = σ(γ1·(1−L_assoc) + γ2·L_render)                     # Eq. 13 (s. unten)
λ_t   = η·λ_t + (1−η)·λ*                                    # Eq. 14
comp  = λ_t·A_t + (1−λ_t)·B_t                              # Eq. 1
accept ⇔ comp ≥ accept_thresh                              # Eq. 2 mit Schwelle
```

**Hinweis zur Sigmoid-Eingabe (Eq. 13):** Paper-literal ist `σ(γ1·L_assoc + γ2·L_render)`,
aber die Paper-Notation hat `L_assoc` als *Reward* (hoch = Tracker happy).
Wir verwenden `L_assoc` als *Stress-Signal* (hoch = Tracker im Stress, siehe
Konvention unten). Damit Eq. 13 semantisch dieselbe Richtung behält wie im
Paper — *hoher Sigmoid-Input = mehr FRA-Gewicht, weil Tracker keinen
Aufmerksamkeitsbedarf hat* — flippen wir den L_assoc-Beitrag zur
λ-Berechnung zurück. B_t und Threshold-Decision verwenden weiter den
geflippten Wert.

## Konvention: alle Sub-Scores als „Select-Reward"

Im Paper sind die sechs Sub-Costs heterogen: manche Reward-artig (`L_assoc`
zählt erfolgreiche Matches → hoch=gut), manche Cost-artig (`L_render = 1 −
PSNR/target` → hoch=schlecht), und Eq. 9 addiert sie alle ungeprüft. Dass eine
einzelne Schwelle gegen ein gemischtes Reward+Cost-Bündel keinen wohldefinierten
Vergleich hat, hat das Paper nicht aufgelöst.

Wir lösen die Inkonsistenz durch eine einheitliche Polung: **alle sechs
Sub-Scores liegen in [0,1], höher = „diesen Frame eher als KF akzeptieren"**.
Konsequenz: `L_assoc` ist gegenüber Eq. 10 invertiert (`1 − stability`), alle
anderen Sub-Scores haben paper-konsistente Richtung.

## Sub-Score-Definitionen (paper-nahe)

| Symbol | Berechnung | Paper-Eq. | Adaption |
|---|---|---|---|
| `L_motion` | `tanh( ‖Δt‖ / trans_ref_m + ω·‖ΔR‖_F / √2 )` | Eq. 12 | tanh-Sättigung statt Paper's unbeschnittenem `‖Δt‖ + ω‖ΔR‖_F`; gleiche Ordnung, [0,1]-konsistent mit den anderen Sub-Scores |
| `L_assoc` | `1 − (n_inliers/n_ref)·exp(−n_outliers/n_total)` mit `n_outliers = n_total − n_inliers` | Eq. 10 | Inliers via ORB+BFMatcher (Lowe 0.85) + RANSAC-Homography; `n_outliers` ist der gesamte Matching-Pool ohne Inliers (Lowe-fail + RANSAC-fail), nicht nur RANSAC-fail; Polaritäts-Flip (s.o.) |
| `L_flow` | `mean‖u_t − u_{t-1}‖₂ / flow_ref_px`, geclippt | Eq. 11 | **Echtes Δflow über 3 Frames via zwei LK-Passes** (s.u.) — misst Bewegungs-Inkonsistenz, nicht Bewegungs-Magnitude |
| `L_uncert` | `mean(depth_cov) / cov_ref`, geclippt | Eq. 4-6 | Tracker-Depth-Cov statt Renderer-Var[C] (kein Mapper-Sync verfügbar) |
| `L_render` | `1 − PSNR(I_t, Î_t) / PSNR_target`, geclippt | Eq. 7 | **Î_t = bilinearer Backward-Warp von prev_kf_gray** mit aktueller Tiefe in die aktuelle Pose. Misst Coverage-Lücke statt Frame-Schärfe |
| `L_covis` | `1 − \|V_t∩V_kf\| / \|V_t∪V_kf\|` | Eq. 8 | **Symmetrische Jaccard-IoU** aus Pixel-Grid-Sampling beider Views als Stand-in für Gaussian-Supports |

### L_flow — 3-Frame-Δflow im Detail

Paper Eq. 11 sagt `u_i` = Flow am aktuellen Frame, `u_{i-1}` = Flow desselben
Features am vorigen Frame. Das misst die *Veränderung* des Flusses, nicht
seinen Betrag — ein gleichmäßiges Schwenken liefert kleines Δflow (stabiles
Tracking), ein abruptes Stopp/Verdecken großes Δflow.

Realisierung mit zwei LK-Passes pro Frame:

1. ORB-Keypoints auf `prev_prev_gray` (Frame t-2) → `kps_t-2`
2. **LK1:** `prev_prev_gray → prev_gray`, startend bei `kps_t-2`
   → `kps_t-1` = Positionen bei t-1
   → `u_{t-1} = kps_t-1 − kps_t-2`
3. **LK2:** `prev_gray → gray`, startend bei `kps_t-1`
   → `kps_t` = Positionen bei t
   → `u_t = kps_t − kps_t-1`
4. `L_flow = mean‖u_t − u_{t-1}‖` über Features mit `status==1` in beiden LK-Passes

Der 3-Frame-Speicher (`prev_prev_gray`, `prev_prev_kps_uv`) wird jeden Frame
geshiftet, auch wenn nicht akzeptiert — sonst bricht die LK-Kontinuität ab.

### L_render — PSNR via prev_kf-Warp im Detail

Paper Eq. 7 vergleicht `I_t` (echtes Bild) mit `Î_t` (Renderer-Output am
aktuellen Pose). Der Renderer ist hier asynchron, also können wir nicht im
Selector-Slot rendern. Stattdessen verwenden wir den **nächstgelegenen
verfügbaren Szenen-Prediktor**: den letzten akzeptierten KF.

Schritte:

1. Cache `prev_kf_gray` und `prev_kf_depth` bei jedem `_commit`.
2. Aktuelle Sample-UV → backproject mit aktueller Tiefe → Welt-Punkte.
3. Welt-Punkte → projizieren in die `prev_kf`-Kamera → UV in prev_kf.
4. Bilinear samplen: `pred = prev_kf_gray(uv_in_kf)`, `actual = gray(uv_current)`.
5. `MSE = mean((pred − actual)²)`, `PSNR = 10·log10(255²/MSE)`.
6. `L_render = clip(1 − PSNR/PSNR_target, 0, 1)`.

Der Fallback (LapVar-Schärfe) bleibt nur aktiv wenn `prev_kf_gray` fehlt
(erster Frame, oder rgb-loser Aufruf).

**Caveat:** prev_kf ist ein einzelner Frame, nicht die gesamte Map. Bei
großen Baselines wird die Warp-Vorhersage strukturell schlechter (= L_render
↑), nicht weil der Mapper fehlerhaft ist, sondern weil der View-Sprung groß
ist. Das ist trotzdem das gewünschte Signal: große View-Sprünge → KF nötig.

### L_covis — Symmetrische IoU im Detail

Paper Eq. 8 ist Jaccard über Gaussian-Supports. Ohne Mapper-Zugriff
approximieren wir `V_t` und `V_kf` durch Pixel-Grid-Samples aus jeweils der
zugehörigen Tiefe:

```
|V_t ∩ V_kf|  ≈  sqrt( (# current-samples sichtbar in prev_kf)
                     · (# prev_kf-samples sichtbar in current) )
|V_t ∪ V_kf|  =  |V_t| + |V_kf| − |V_t ∩ V_kf|
```

Das **geometrische Mittel** beider Richtungs-Schätzer ist robuster als das
arithmetische, weil multiplikative Outlier in einer Richtung (z.B. ein Frame
mit fast keiner gültigen Tiefe) nicht durchschlagen. Bei symmetrischen Views
identisch zur arithmetischen Variante.

Beide Richtungen brauchen die jeweils andere Tiefe; deshalb `prev_kf_depth`
zusätzlich zu `prev_kf_gray` im Cache. Fällt zurück auf einseitige
Reprojektion (alte Variante) wenn `prev_kf_depth` fehlt (erster
post-init-Frame, in dem der Tiefe-Snapshot noch nicht eingespielt war).

## Versionierung

| Aspekt | v1 (alt) | v2 / v2.1 / v2.2 (jetzt) | Paper |
|---|---|---|---|
| `L_flow` | `‖LK-displacement‖` (Flow-Magnitude) | **v2:** `‖u_t − u_{t-1}‖` (Δflow, 3-Frame, 2× LK) | Eq. 11 verbatim |
| `L_render` | `LapVar(gray) / lap_var_ref` (Schärfe) | **v2:** `1 − PSNR(I_t, warp(prev_kf)) / target` | Eq. 7 (PSNR vs Render) |
| `L_covis` (Schätzer-Topologie) | einseitige Reprojektion `1 − n_in_kf/n_curr` | **v2:** symmetrische Jaccard-IoU | Eq. 8 verbatim |
| `L_covis` (Intersect-Schätzer) | — | **v2.2:** geom. Mittel `sqrt(n_curr_in_kf · n_kf_in_curr)` statt arith. Mittel `0.5·(a+b)` | Eq. 8 (Schätzer nicht spezifiziert; geom. ist robuster bei asymm. Frustum-Coverage) |
| `L_motion` | hard-clip `clip(raw, 0, 1)` | **v2:** `tanh(raw)` | Eq. 12 (kein Clip) |
| `L_assoc` `n_outliers` | `n_matches − n_inliers` (nur RANSAC-fail) | **v2.1:** `n_total − n_inliers` (Lowe-fail + RANSAC-fail) | Eq. 10 ("rejected matches" = gesamter Match-Pool ohne Inliers) |
| Sigmoid Eingang | `σ(z − 0.5·(γ1+γ2))` mit Recentering | **v2.1:** `σ(γ1·(1−L_assoc) + γ2·L_render)` — kompensiert L_assoc-Flip | Eq. 13 (literal σ(γ1·L_assoc + γ2·L_render) mit Paper-L_assoc-Polung) |
| `prev_kf`-Cache | nur Pose | **v2:** Pose + gray + depth | — |

Die nicht änderbaren Adaptionen (`L_uncert` über Depth-Cov, `L_assoc` über
eigenes ORB+RANSAC, `accept`-Schwelle statt argmin) bleiben — die hängen am
asynchronen Mapper und am DBAF-Tracker-Interface, nicht an der
Selector-Implementation.

### Konsequenzen der v2/v2.1/v2.2-Änderungen

- **L_covis Geom-Mean (v2.2):** Erster v2-Wurf nahm `intersect = 0.5·(a+b)`
  (arithmetisches Mittel der beiden Richtungs-Schätzer). Bei stark
  asymmetrischer Frustum-Coverage (z.B. ein Frame mit fast keiner gültigen
  Tiefe) ist `0.5·(a+b)` ein optimistischer Schätzer, der die echte
  Intersection überschätzen kann. Geometric mean `sqrt(a·b)` ist robuster
  gegen einseitige Ausreißer; bei symmetrischen Views identisch zur
  arithmetischen Variante. Praktisch klein-aber-richtig.
- **n_outliers-Fix (v2.1):** Erster v2-Wurf zählte `n_outliers = n_matches −
  n_inliers` (nur RANSAC-rejected). Paper liest sich aber so, dass der
  gesamte Matching-Pool ohne Inliers die Outliers ausmacht (Lowe-rejected +
  RANSAC-rejected). Mit `n_outliers = n_total − n_inliers` wird
  `exp(-n_outliers/n_total) = exp(n_inliers/n_total − 1)` — eine glatte
  Strafe für niedrigen Inlier-Anteil. Bei typischen Werten
  (n_total=800, n_inliers=150) gibt das exp(-650/800)≈0.44 statt
  exp(-50/800)≈0.94 (alte Version) — also ~2× strenger.
- **Sigmoid-Polaritäts-Fix (v2.1):** Mit unserer L_assoc-Flip-Konvention
  würde der literale Eq. 13 in die *falsche* Richtung schwingen: hoher
  Tracker-Stress → λ groß → FRA gewinnt, statt umgekehrt. Wir verwenden
  daher im Sigmoid den ungeflippten L_assoc-Sinn (`1 − L_assoc`), was die
  Paper-Intention reproduziert (Tracker happy → FRA-Bias, Tracker stress
  → DRA-Bias). Bei `γ=1` und neutralen Inputs `(L_assoc, L_render)=(0.5,0.5)`
  ergibt das `σ(1.0)≈0.73` — der Paper-Bias Richtung FRA bei beidseitig
  "ok"-Lage, exakt wie im Paper.
- **L_flow auf Δflow:** Gleichmäßige Bewegungen geben kleinen L_flow (kein
  Push zur Akzeptanz, anders als v1). Inkonsistente Bewegungen (Stöße,
  Okklusionen, Tracking-Glitches) treiben L_flow hoch. Paper-konform, aber
  in der Praxis ein gänzlich anderes Signal als v1.
- **L_render auf PSNR-Warp:** Frame-Qualitäts-Bias (v1) ist weg. Stationäre
  Szenen geben niedrigen L_render (Warp matcht das Bild), View-Sprünge
  treiben L_render hoch. Stimmt jetzt semantisch mit dem Paper überein.
- **L_motion tanh:** In Aerial-Szenen mit großen Inter-KF-Translationen
  (smallcity, amtown03) sättigt L_motion bei v1 auf 1.0 und ist als Signal
  unbrauchbar. tanh saturiert weicher; ‖Δt‖/trans_ref_m = 2 → 0.96,
  = 3 → 0.995, behält also Ordnung bis in den hohen Bewegungsbereich.

## Konfigurations-Knöpfe

```yaml
frame_selector:
  kind: game_kfs
  # FRA weights (β1+β2+β3 ≈ 1)
  beta_uncert: 0.3
  beta_render: 0.3
  beta_covis:  0.4
  # DRA weights (α1+α2+α3 ≈ 1)
  alpha_assoc:  0.5
  alpha_flow:   0.3
  alpha_motion: 0.2
  # Lambda-Adaption (Eq. 13-14)
  gamma_assoc:  1.0
  gamma_render: 1.0
  eta:          0.8     # EMA smoothing; 1 = freeze, 0 = nur λ*
  lambda_init:  0.5
  # Decision
  accept_thresh: 0.5    # accept iff composite >= thresh
  # ORB / DRA-Detail
  orb_n_features: 800
  ransac_reproj_thresh: 4.0
  min_matches:   12     # force-accept wenn weniger Matches (Tracker im Stress)
  # Normalisierungs-Skalen (datensatz-abhängig)
  flow_ref_px:   30.0   # mean(||Δu||) -> "viel Bewegungs-Inkonsistenz"
  psnr_target:   25.0   # PSNR-Zielwert (dB), in [0,1]-Normierung
  lap_var_ref:   500.0  # nur Fallback wenn prev_kf_gray fehlt
  cov_ref:       1.0    # mean(depth_cov) -> "sehr unsicher"
  trans_ref_m:   0.30   # ‖Δt‖ -> tanh-Eingang
  omega_rot:     0.10   # Gewicht auf ‖ΔR‖_F
  n_samples:     2048   # Pixel-Grid für L_covis + L_render
  # Tiefe
  min_depth: 0.2
  max_depth: 35.0
```

`flow_ref_px` / `psnr_target` / `trans_ref_m` / `cov_ref` sind die wichtigsten
sequenzspezifischen Knöpfe. Die α/β/γ-Defaults sind robust laut Sensitivity-
Tabelle VIII im Paper, aber das gilt für die *Paper-Sub-Scores*. Mit unseren
Adaptionen (Depth-Cov statt Render-Var, prev_kf-Warp statt echter Render) sind
die α/β möglicherweise neu zu kalibrieren — siehe Sweep.

## Strukturell nicht behebbare Abweichungen vom Paper

Drei Punkte sind auch in der finalen v2.2 **nicht paper-treu** und können
es ohne Architektur-Änderung am VINGS-Mapper-Interface auch nicht werden.
Für die BA-Diskussion explizit zu benennen:

1. **L_uncert misst Tracker-Pose-Cov, nicht Renderer-Color-Variance.**
   Paper Eq. 4-6 integriert die per-Ray-Color-Varianz aus den
   Volume-Rendering-Compositing-Weights — also Map-seitige *epistemische*
   Unsicherheit („wie sicher ist die Map an diesem View?"). Unsere
   `mean(depth_cov)` misst Tracker-seitige *geometrische* Unsicherheit
   („wie zuverlässig ist die Tiefen-Schätzung?"). Die zwei Signale können
   **negativ korreliert** sein: ein neuer, dem Mapper unbekannter View hat
   oft niedrige Tracker-Cov (gute Features, klare Geometrie) aber **hohe**
   Render-Cov (Map hat ihn nie gesehen). Wir messen damit einen ganz
   anderen Aspekt von „Frame-Wichtigkeit für die Map".

2. **L_render warpt nur den letzten KF, nicht die volle Map.** Paper
   `Î_t` ist der Renderer-Output mit **allen** akkumulierten Gaussians;
   unser `Î_t` ist ein bilinearer Backward-Warp vom **einen** letzten
   akzeptierten KF. Konsequenzen:
   - **Loop-Closure-Fall:** Wenn die Kamera nach 100 Frames an einen
     früher gesehenen View zurückkehrt, gäbe Paper-L_render niedrig
     („Map hat den View"), wir geben fälschlich hoch („prev_kf hat ihn
     nicht"). Wir verlieren den Coverage-Vorteil längerer Maps.
   - **Steady-State:** In kontinuierlicher Bewegung dominiert prev_kf
     ohnehin den Renderer-Beitrag → unsere Approximation ist qualitativ
     ähnlich.

3. **L_assoc nutzt eigenes ORB+RANSAC, nicht den DBAF-Tracker-Output.**
   Paper-Notation für `n_match`/`n_outlier` bezieht sich auf ORB-SLAM3's
   internen Match-Pool nach Pose-Estimation. DBAF exponiert keine
   Per-Match-Inlier-Stats, also doppeln wir die Feature-Detection mit
   eigenem ORB + BFMatcher + RANSAC-Homography. Kostet ~10–25 ms pro
   Frame und ist nicht garantiert konsistent mit DBAF's eigenem
   Match-Verständnis.

Zusätzlich gibt es zwei [0,1]-Konformitäts-Anpassungen die im Paper-Text
keine direkte Entsprechung haben:

- **Decision: `composite ≥ accept_thresh` statt `argmin L`** —
  Paper-Eq. 2 ist nie sauber definiert (`L(d_t=0)` wird im Paper nirgends
  spezifiziert). Threshold ist die einzige praktikable Interpretation.
- **L_motion: `tanh(...)` statt `‖Δt‖ + ω‖ΔR‖_F`** — Paper-Formel hat
  keine Sättigung und keine Normalisierung. Ohne Anpassung würde L_motion
  alle anderen [0,1]-Sub-Scores in B_t dominieren. tanh ist ein
  Kompromiss; die Ordnung bleibt erhalten, das Ausmaß-Detail bei
  ‖Δt‖/trans_ref_m > 3 geht verloren.

Diese fünf Punkte zusammen heißen: die Implementation ist eine
**„Game-KFS-inspirierte Composite-Score-Strategie mit dokumentierten
Mapper-frei-Approximationen"**, nicht eine 1:1-Reproduktion des Papers.
Der Decision-Mechanismus (FRA + DRA + EMA-λ + Schwelle) ist paper-treu,
zwei der sechs Sub-Scores messen aber strukturell andere Signale.

## Sweep

`scripts/config_profiles.py` registriert `game_kfs_w_sweep` (9 Variants um
den Default herum: 1 default, 2 α-, 2 β-, 2 η-, 2 accept_thresh-Varianten).
Aufruf:

```bash
python scripts/gen_configs.py --profile game_kfs_w_sweep \
  --base configs/local/ntu_eee_03_200.yaml \
  --out  configs/local/ntu_eee_03/game_kfs_w_sweep
```

## Failsafes

- **First frame**: immer akzeptiert, seedet `prev_kf` (Pose+gray+depth) und
  `ref_kps`.
- **Wenig Matches**: wenn `n_matches < min_matches` (Default 12), wird der
  Frame zwangsweise akzeptiert (`score.forced = True`). Vermeidet
  Tracking-Starvation auf featurearmen Sequenzen. Nicht im Paper, aber
  paper-spirit.
- **`depth_cov` fehlt**: `L_uncert = 0.5` (neutral). Andere Sub-Scores
  funktionieren ohne.
- **`rgb` fehlt**: `L_assoc = 1.0` (Tracker als „im Stress" interpretiert),
  `L_flow = 0.0`, `L_render` aus depth-normalisierter LapVar als Proxy
  (Fallback-Pfad).
- **Erster Frame nach `_commit`**: `prev_kf_depth`-Cache existiert; falls
  doch nicht, fällt `L_covis` auf einseitige Reprojektion zurück.
- **3-Frame-Speicher nicht voll**: `L_flow = 0` für die ersten zwei Frames
  nach Reset (kein Δflow berechenbar).

## Profiling

`PhaseTimer` misst den Selector als `frame_select`-Phase. Erwartete Kosten auf
smallcity_200 / 690×1024:

| Subroutine | v1 | v2 |
|---|---|---|
| ORB detect + match | ~10–25 ms | ~10–25 ms |
| RANSAC Homography | < 1 ms | < 1 ms |
| Sparse-LK | ~2–5 ms (1×) | ~4–10 ms (2×) |
| L_render (LapVar vs PSNR-Warp) | < 1 ms | ~2–4 ms |
| L_covis (einseitig vs symmetrisch) | < 1 ms | ~1–2 ms |
| Aggregation | < 1 ms | < 1 ms |
| **Total** | **~15–30 ms** | **~20–40 ms** |

Damit ist Game-KFS v2 etwas teurer als v1, aber liefert ein paper-treues
Signalbündel. Liegt komfortabel unter den ~1150 ms eines
`map.train_loop`-Calls.
