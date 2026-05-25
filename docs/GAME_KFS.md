# Game-KFS-Selector

Keyframe-Auswahl nach **Chen S., Yang B., Wang C. et al., „Game-KFS:
Game-Theory-Inspired Keyframe Selection for Hybrid Representation Visual
SLAM", IEEE Robotics and Automation Letters 2025**, Sec. III. Originalpaper-
Framework ist Photo-SLAM (ORB-SLAM3 + 3D-Gaussian-Mapper). Hier in VINGS-Mono
sitzt der Selector im selben Plugin-Slot wie `vista` / `nurbs_lvi` / `mm3dgs`
(siehe `scripts/vings_utils/selector_factory.py`).

## Idee

Statt einer einzelnen Heuristik werden zwei konzeptuelle Agenten gegeneinander
gewogen:

| Agent | Was er „möchte" | Sub-Scores |
|---|---|---|
| **FRA** (Field) | Frames, die das Rendering verbessern, Lücken schließen, Unsicherheit senken | `L_uncert`, `L_render`, `L_covis` |
| **DRA** (Discrete) | Frames, die das Feature-Tracking stabil halten und keine Pose-Sprünge verursachen | `L_assoc`, `L_flow`, `L_motion` |

Sie werden über einen online adaptierten Gewichtsfaktor λₜ zu einem skalaren
Score kombiniert:

```
A_t   = β1·L_uncert + β2·L_render + β3·L_covis             # FRA composite
B_t   = α1·L_assoc  + α2·L_flow   + α3·L_motion            # DRA composite
λ*    = σ(γ1·L_assoc + γ2·L_render − offset)               # Eq. 13
λ_t   = η·λ_t + (1−η)·λ*                                    # Eq. 14, EMA
comp  = λ_t·A_t + (1−λ_t)·B_t                              # Eq. 1
accept ⇔ comp ≥ accept_thresh                              # Eq. 2 mit Schwelle
```

## Konvention: alle Sub-Scores als „Select-Reward"

Im Paper sind die sechs Sub-Costs heterogen: manche Reward-artig (`L_assoc`
zählt erfolgreiche Matches → hoch=gut), manche Cost-artig (`L_render = 1 −
PSNR/target` → hoch=schlecht). Damit die argmin-Decision (Eq. 2) konsistent
funktioniert, sind hier **alle sechs Sub-Scores in [0,1] und mit derselben
Polung**: höher = „diesen Frame eher als KF akzeptieren". Das vermeidet die
implizite Doppeldeutigkeit im Paper.

Konsequenz: `L_assoc` wird gegenüber Eq. (10) invertiert (`1 − stability`,
wenig stabile Matches = Tracker im Stress = KF nötig).

## Variablen

| Symbol | Berechnung |
|---|---|
| `L_motion` | `clip((‖Δt‖ / trans_ref_m + ω·‖ΔR‖_F / √2) , 0, 1)` |
| `L_assoc` | `1 − (n_inliers/n_ref) · exp(−n_outliers/n_total)`, Inlier/Outlier-Split via RANSAC-Homography auf den ORB-Matches |
| `L_flow` | `clip(mean‖LK-displacement‖_2 / flow_ref_px, 0, 1)` |
| `L_uncert` | `clip(mean(depth_cov) / cov_ref, 0, 1)` |
| `L_render` | `clip(LapVar(gray) / lap_var_ref, 0, 1)` |
| `L_covis` | `1 − \|V_t∩V_kf\| / \|V_t\|`, einseitige Vorwärts-Pixel-Reprojektion (mm3dgs-Stil) |

`σ(z − offset)` zentriert das Sigmoid bei `0.5·(γ_assoc + γ_render)`, damit
neutrale Eingaben λ ≈ 0.5 ergeben (anders als bei `σ(0) = 0.5`, was nur bei
γ = 0 stimmen würde).

## Adaptionen vs. Original (Photo-SLAM-Pipeline)

| Original | VINGS-Adaption | Grund |
|---|---|---|
| `L_uncert = mean(Var[C])` aus Rasterizer-Compositing-Weights | `mean(depth_cov)` vom DBAF-Tracker | Mapper läuft async; kein Render-Pass im Selector-Slot; `depths_cov` ist die natürlich verfügbare Tracker-seitige Unsicherheit |
| `L_render = 1 − PSNR(I_t, Î_t)/PSNR_target` | `clip(LapVar(gray)/lap_var_ref, 0, 1)` | Sharpness-Proxy ist gratis (gleiche Berechnung wie in `mm3dgs_selector`); kein Render-Pass nötig; misst dasselbe Konstrukt (Frame-Qualität für die Map) |
| `L_covis = 1 − Gaussian-IoU` zwischen Views | Einseitige Pixel-Reprojektion (Anteil current-depth-Samples, die *nicht* in prev_kf projizieren) | Echtes Gaussian-IoU braucht eine Visibility-Query pro Frame in den Mapper; Reprojektion deckt dasselbe Konstrukt ab („wieviel Neues sehe ich vs. prev_kf?") |
| `n_outlier` aus ORB-SLAM3-Tracker-Bookkeeping | Eigene ORB-Detection + BFMatcher (k=2 + Lowe-Ratio 0.85) + RANSAC-Homography für den Inlier-Split | DBAF gibt keine Per-Match-Inlier-Stats nach aussen; ORB+RANSAC ist Standard-VSLAM-Mechanik und reproduzierbar |
| `L_flow` über ORB-SLAM3-internen Optical-Flow | `cv2.calcOpticalFlowPyrLK` auf den ORB-Keypoints des prev_frame im Selector | Keine Modifikation am dbaf-Submodul nötig; `prev_gray` + `prev_kps` werden jeden Frame aktualisiert (nicht nur bei KF-Acceptance), damit das LK-Signal kontinuierlich bleibt |
| `argmin L(d_t)` über `d_t ∈ {0,1}` | `accept ⇔ composite ≥ accept_thresh` (Default 0.5) | Eq. (2) ist nur wohldefiniert wenn `L(0) ≠ 0`; die explizite Schwelle ist äquivalent + sweepbar |

Der **Kern-Mechanismus** (FRA + DRA + EMA-λ + skalare Decision) ist verbatim
aus dem Paper.

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
  # Lambda-Adaption
  gamma_assoc:  1.0
  gamma_render: 1.0
  eta:          0.8     # EMA smoothing; 1 = freeze, 0 = nur λ*
  lambda_init:  0.5
  # Decision
  accept_thresh: 0.5    # accept iff composite >= thresh
  # Normalisierungs-Skalen (datensatz-abhängig)
  flow_ref_px:  30.0    # mean LK displacement -> "viel Bewegung"
  lap_var_ref:  500.0   # LapVar -> "scharf"
  cov_ref:      1.0     # mean(depth_cov) -> "sehr unsicher"
  trans_ref_m:  0.30    # ‖Δt‖ -> "grosse Translation"
  omega_rot:    0.10    # Gewicht auf ‖ΔR‖_F
```

`flow_ref_px` / `lap_var_ref` / `cov_ref` / `trans_ref_m` sind die wichtigsten
Knöpfe, weil sie die Sub-Scores in den Decision-relevanten [0,1]-Bereich
ziehen. Die Paper-Defaults für α/β sind robust (Sensitivity-Tabelle VIII im
Paper), die Skalen aber sind sequenzspezifisch.

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

- **First frame**: immer akzeptiert, seedet `prev_kf`, `ref_kps`, `prev_gray`.
- **Wenig Matches**: wenn `n_matches < min_matches` (Default 12), wird der
  Frame zwangsweise akzeptiert (`score.forced = True`). Vermeidet
  Tracking-Starvation auf featurearmen Sequenzen.
- **`depth_cov` fehlt**: `L_uncert = 0.5` (neutral). Andere Sub-Scores
  funktionieren ohne.
- **`rgb` fehlt**: `L_assoc` = 1.0 (Tracker als „im Stress" interpretiert),
  `L_flow` = 0.0, `L_render` aus depth-normalisierter LapVar als Proxy.

## Profiling

`PhaseTimer` misst den Selector als `frame_select`-Phase. Erwartete Kosten auf
smallcity_200 / 690×1024:

| Subroutine | Größenordnung |
|---|---|
| ORB detect + match | ~10–25 ms |
| RANSAC Homography | < 1 ms |
| Sparse-LK | ~2–5 ms |
| Backproject + Reprojektion (2048 Samples) | < 1 ms |
| LapVar + Aggregation | < 1 ms |
| **Total** | **~15–30 ms** |

Damit ist Game-KFS deutlich teurer als VISTA (~1–2 ms) oder MM3DGS (~3–5 ms),
aber liefert ein reichhaltigeres Signal. Liegt aber komfortabel unter den
~1150 ms eines `map.train_loop`-Calls, sprich: nicht der Flaschenhals.
