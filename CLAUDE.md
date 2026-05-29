# VINGS-Mono-BA

Bachelorarbeit-Fork von VINGS-Mono: ein Online-Mono-SLAM-System, das parallel zum
Tracking eine 3D-Szene aus 2D-Gaussian-Splats rekonstruiert. Aktueller Fokus:
**Mapping-Last reduzieren, ohne Posenqualität aufzugeben**.

## Tracker vs. Mapper

| Hälfte | Wo im Code | Frequenz | Kosten (smallcity_200) |
|---|---|---|---|
| Tracking (Posen) | `submodules/dbaf` + `scripts/frontend/dbaf_frontend.py` | jeder Frame | `track.frontend_ba` ≈ 450 ms/Frame |
| Mapping (Gaussians) | `scripts/gaussian/` | nur Keyframes | `map.train_loop` ≈ 1150 ms/KF |

Mapping ist der teurere Brocken (~46 % Wandzeit). Vollständige Tracking↔Mapping-
Erklärung in `MAPPING_TRACKING.md`.

## Aktuelle Frame-Selection-Pipeline

```
RGB-Frame
  → motion_filter.py   (Stage 1: Optical-Flow-Gate, thresh=2.4)
  → DepthVideo.append()
  → dbaf_frontend.__update()   (Stage 2: distance() vs. keyframe_thresh)
  → judge_and_package_v3()     → viz_out (Batch der lokalen KFs)
  → [neu] FrameSelector        → do_map ja/nein
  → mapper.run(viz_out, True)
```

Stages 1+2 sind die alten VINGS-Filter (im `frontend/`-Submodul), entscheiden ob
ein Frame Tracker-Keyframe wird. Der **neue FrameSelector** sitzt nach dem
Tracker und entscheidet, ob der Mapper auf diesen KF angesetzt wird.

## Frame-Auswahl-Algorithmen (Plugin-basiert)

Der Slot zwischen Tracker-KF und Mapper-Aufruf ist **algorithmusagnostisch**:
`scripts/vings_utils/selector_factory.py:make_frame_selector(cfg, K, image_hw)`
liest `frame_selector.kind` und liefert eine Instanz mit gemeinsamer Schnittstelle
`should_accept(depth, t, R, rgb=None) -> (accept, score)`. Neue Algos
registrieren sich via `@register_selector("name")`-Dekorator.

Aktuell registriert:

| kind | Modul | Idee | Braucht rgb? |
|---|---|---|---|
| `vista` | `frame_selector.py` (`FrameSelector`) | View-Angle-Diversity pro Voxel (VISTA Eq. 1); Pose-Filter vorgeschaltet, Reservoir-Sampling pro Voxel | nein |
| `nurbs_lvi` | `nurbs_lvi_selector.py` (`NurbsLviSelector`) | Adaptive Q-Score nach NURBS-LVI: ORB-Matches + Chamfer + Sektor-Migrationen | ja |
| `mm3dgs` | `mm3dgs_selector.py` (`Mm3dgsSelector`) | Covisibility-Drop (< 95 %) vs. prev_kf + NIQE-Min (Variance-of-Laplacian) im Sliding-Window | ja |
| `game_kfs` | `game_kfs_selector.py` (`GameKfsSelector`) | Game-theory-inspirierte Composite-Cost aus FRA (uncert+render+covis) und DRA (assoc+flow+motion) mit EMA-λ; siehe `docs/GAME_KFS.md` | ja |
| `adaptive_kf` | `adaptive_kf_selector.py` (`AdaptiveKfSelector`) | Momentum-Aware Adaptive Threshold (`θ = max(θ₀, μ+k·σ)` + Decay γ) über Hybrid-Error (Photo + SSIM via Depth-Warping); siehe `docs/ADAPTIVE_KF.md` | ja |
| `orbslam3` | `orbslam3_selector.py` (`OrbSlam3Selector`) | ORB-SLAM3-„New Keyframe Decision": `(c1a ∨ c1b) ∧ c2` — Spacing-OR (force-rate ∨ min-frames) AND Novelty (`ratio < 0.9` mit `ratio = matches / baseline_matches` als nRefMatches-Analog AND `matches ≥ min_tracked`); siehe `docs/ORB_SLAM3.md` | ja |
| `coko_slam` | `coko_slam_selector.py` (`CokoSlamSelector`) | DINOv2 + Cosine-Distanz, zweistufig (Repo-treu): Stage 1 Submap-Reset `d_anchor > submap_threshold ∧ |K| ≥ min_kfs`, Stage 2 In-Submap-KF `min cos_dist > α`; siehe `docs/COKO_SLAM.md` | ja |
| `aim_slam` | `aim_slam_selector.py` (`AimSlamSelector`) | SIGMA-Modul: Voxel-Overlap (Stage 1) + EKF-Kovarianzreduktion Info-Gain (Stage 2) + Reduced-Chi-Square-Stabilität auf Hybrid-Residual Eq. 5 (Stage 3) mit ORB-Korrespondenz (Paper-treu wenn rgb da; sonst Reprojection-Fallback), AND-Decision; siehe `docs/AIM_SLAM.md` | optional |
| `none` | — | Selektor aus, `mapper_kf_skip`-Pfad aktiv | — |

Profiling-Phase: `frame_select` im `PhaseTimer`-Summary. Standalone-Smoketests:
`python scripts/vings_utils/frame_selector.py`,
`python scripts/vings_utils/nurbs_lvi_selector.py`,
`python scripts/vings_utils/mm3dgs_selector.py`,
`python scripts/vings_utils/game_kfs_selector.py`,
`python scripts/vings_utils/adaptive_kf_selector.py`,
`python scripts/vings_utils/orbslam3_selector.py`,
`python scripts/vings_utils/coko_slam_selector.py`,
`python scripts/vings_utils/aim_slam_selector.py`.

Config-Beispiele:

```yaml
frame_selector:
  kind: vista
  voxel_size: 0.10
  gain_thresh: 0.30
  trans_thresh_m: 0.15
  rot_thresh_deg: 10.0
  n_rays_score: 256
  n_rays_integrate: 2048
  min_depth: 0.2
  max_depth: 35.0
```

```yaml
frame_selector:
  kind: nurbs_lvi
  orb_n_features: 800
  sector_angle_deg: 2.0    # Re-skaliert fuer VINGS-Parallaxe (~0.5-2°). Code-Design hat
                           # EINE Migrations-Schwelle bei sector_angle_deg/2, Paper hat
                           # ZWEI bei 15°/30° -- nicht 1:1 mappbar. Siehe NURBS_LVI.md.
  # chamfer_lambda entfernt -- Paper Eq. 3 definiert lambda = 1/|P_2| (kein Hyperparam)
  min_matches: 15          # force-accept wenn Mc < min_matches
  force_accept_all: false  # true = Diagnose-Modus (loggt Q/mig, akzeptiert alles)
  min_depth: 0.2
  max_depth: 35.0
```

```yaml
frame_selector:
  kind: mm3dgs
  covis_thresh: 0.95       # accept wenn Overlap (prev_kf-Depth → current frame) < 95 %
  niqe_window: 5           # Sliding-Window für Quality-Gate (Lap-Var, jeder Frame pusht)
  n_samples: 2048          # Pixel für die Backprojection
  min_gap_after_kf: 5      # Paper `kf_every` — Min-Gap seit letztem KF, wirkt nur mit below_thresh (kein force-accept)
  force_accept_after: 0    # >0 = Quality-Streak-Failsafe (VINGS-spezifisch, kein Reference-Pendant)
  min_depth: 0.2
  max_depth: 35.0
```

```yaml
frame_selector:
  kind: adaptive_kf
  theta0: 0.05             # untere Schwelle (Eq. 6)
  theta_init: 0.10         # Warm-up-Startwert
  window_size: 5           # W (Paper Sec. 3.4)
  sensitivity: 1.5         # k (Paper Sec. 3.4)
  decay: 0.95              # γ (Eq. 7, Paper Sec. 3.4)
  w_photo: 0.7             # α (Paper Eq. 3)
  w_ssim: 0.3              # β (Paper Eq. 3)
  min_overlap_pixels: 1000 # |M|-Fail-safe
  force_accept_all: false  # true = Diagnose-Modus
  min_depth: 0.2
  max_depth: 35.0
```

```yaml
frame_selector:
  kind: orbslam3
  orb_n_features: 800
  tracked_ratio_thresh: 0.9   # c2-Threshold; ratio = matches / baseline_matches
                              # (baseline = BFMatch-Count am ersten Frame nach KF-Commit;
                              # nRefMatches-Analog, frame-unabhängig per KF-Lebensdauer)
  min_tracked: 50             # c2-Precondition (Paper "> 15"; konservativer Default für VINGS)
  min_frames: 1               # c1b: Spacing-Untergrenze (vermeidet Bursts)
  max_frames: 30              # c1a: Spacing-Obergrenze (Paper "MaxFrames")
  force_accept_all: false     # true = Diagnose-Modus
  min_depth: 0.2
  max_depth: 35.0
# Akzeptanz iff (c1a OR c1b) AND c2 — paper-konformes AND zwischen Spacing
# und Novelty. Reine Standkamera erzeugt KEINEN KF, auch wenn max_frames
# überschritten ist (Novelty fehlt). Für harten Mindesttakt: mapper_kf_skip.
# Der erste Frame nach einem KF-Commit ist Baseline-Warm-Up und immer skip.
```

```yaml
frame_selector:
  kind: coko_slam
  # Stage 2: in-submap KF decision (Repo-Default 0.02)
  alpha: 0.02              # cosine-distance threshold im DINOv2-Feature-Space
  distance_metric: cosine  # "cosine" (Repo) | "l2" (Legacy, alte α-Werte 0.2-0.6)
  # Stage 1: data-driven submap reset (Repo-Default 0.05 / 10)
  submap_threshold: 0.05   # cosine-distance zum Submap-Anker
  min_kfs_per_submap: 10   # min. KFs bevor Reset zulässig (Repo `keyframe_num`)
  max_kfs: 0               # Hard-Cap (0 = aus, rein datengetrieben)
  memory_mode: submap_reset  # bei max_kfs-Cap: "submap_reset" | "fifo"
  # DINOv2-Backend
  model_name: dinov2_vits14
  image_size: 224          # muss Multiple of 14 sein (DINOv2-Patch-Size)
  feature_aggregation: patch_mean_with_cls  # Repo: HF `last_hidden_state.mean(dim=1)`
  device: cuda
  force_accept_all: false  # true = Diagnose-Modus (loggt min_dist, akzeptiert alles)
```

```yaml
frame_selector:
  kind: game_kfs           # paper-nahe v2: Δflow / PSNR-Warp / Jaccard-IoU
  beta_uncert: 0.3         # FRA-Gewichte (Eq. 3)
  beta_render: 0.3
  beta_covis: 0.4
  alpha_assoc: 0.5         # DRA-Gewichte (Eq. 9)
  alpha_flow: 0.3
  alpha_motion: 0.2
  gamma_assoc: 1.0         # Sigmoid-Input (Eq. 13, literal)
  gamma_render: 1.0
  eta: 0.8                 # EMA-Smoothing für λₜ (Eq. 14)
  accept_thresh: 0.5       # accept iff λ·A + (1-λ)·B ≥ thresh
  orb_n_features: 800
  flow_ref_px: 30.0        # mean(||Δu||) -> "viel Bewegungs-Inkonsistenz"
  psnr_target: 25.0        # PSNR-Zielwert (dB) für L_render
  lap_var_ref: 500.0       # Fallback nur wenn prev_kf_gray fehlt
  cov_ref: 1.0
  trans_ref_m: 0.30        # tanh-Eingang für L_motion
  min_depth: 0.2
  max_depth: 35.0
```

```yaml
frame_selector:
  kind: aim_slam
  # Stage 1: Voxel-Overlap (Eq. 1)
  voxel_size: 0.10
  overlap_thresh: 0.70        # Skip wenn shared/curr > thresh (zu redundant)
  min_overlap_ratio: 0.05     # Failsafe: force-accept bei Szenenwechsel
  n_voxel_samples: 1024
  # Stage 2: EKF-Information-Gain (Eq. 2-3)
  gain_thresh_per_ray: 0.5    # accept wenn Γ/N > thresh (nats/ray)
  n_rays_score: 256
  pixel_sigma: 1.0            # Mess-Rauschen σ_pix (px)
  prior_sigma_d: 0.10         # σ_d Default wenn depths_cov fehlt (m)
  # Stage 3: Reduced-Chi-Square (Eq. 4 + Hybrid-Residual Eq. 5)
  use_chi_square: true
  chi_thresh: 1.0             # κ > 1 → residuen über rauschen → accept
  chi_orb_n_features: 800     # ORB-Korrespondenz (paper-treu); 0 = Reproject-Fallback
  chi_min_matches: 20         # Untergrenze für ORB-Pfad; sonst Fallback
  chi_max_disparity_px: 200.0 # Match-Outlier-Filter
  force_accept_all: false     # true = Diagnose-Modus (loggt alle 3 Stats)
  min_depth: 0.2
  max_depth: 35.0
```

Formel-Details für NURBS-LVI: `docs/NURBS_LVI.md`. Folgt Wu et al. TMECH 2026
Sec. III.A; Entscheidungsregel ist `Or + Oc > Q` (Q ist die *Schwelle*, nicht
der Score).

Formel-Details für Adaptive-KF: `docs/ADAPTIVE_KF.md`. Folgt Jha et al.
arXiv:2510.23928v3 (Dec 2025), Sec. 3.2-3.4. Algorithmus 2 verbatim
(`θ = max(θ₀, μ + k·σ)` + Decay γ), Algorithmus 1 mit den Paper-Formeln
Eq. (1)-(3) und Defaults `α=0.7, β=0.3, W=5, k=1.5, γ=0.95`. `WarpFrame` ist
als Forward-Splat mit Z-Buffer aus **`D_kf`** implementiert (Paper-Signatur
Algorithmus 1) — das Paper sagt nicht *wie* gewarpt wird, aber explizit
dass `D_k` (Keyframe-Tiefe) gewarpt wird. Reproduziert, weil das offizielle
Repo (`jhakrraman/Adaptive_Keyframe_Selection`) nur eine Pose-Heuristik enthält.

Formel-Details für MM3DGS: `docs/MM3DGS.md`. Folgt Sun et al. IROS 2024
Sec. III.E; Entscheidungsregel ist `covis < 0.95` AND `argmax(lap_var) in window`.
NIQE des Originals durch Variance-of-Laplacian-Proxy ersetzt; Depth-Quelle ist
VINGS-Tracker-Depth statt Map-Rendering.

Formel-Details für Game-KFS: `docs/GAME_KFS.md`. Folgt Chen et al. RA-L 2025
Sec. III; Composite-Cost `L = λₜ·A_t + (1−λₜ)·B_t` mit Sub-Scores aus FRA
(Uncertainty/Render/Covisibility) und DRA (Assoc/Flow/Motion), Entscheidung
via Schwelle `accept_thresh`. **v2 (paper-nahe):** Δflow über 3 Frames
(Eq. 11), PSNR via prev_kf-Warp statt LapVar (Eq. 7), symmetrische Jaccard-
IoU (Eq. 8), tanh-Saturation für L_motion (Eq. 12), Sigmoid literal ohne
Recentering (Eq. 13). Mapper-freie Restadaptionen: L_uncert via depth_cov
(statt Renderer-Var[C]), L_assoc via eigenes ORB+RANSAC.

Formel-Details für ORB-SLAM3: `docs/ORB_SLAM3.md`. Folgt UZ-SLAMLab/
ORB_SLAM3 `Tracking.cc::NeedNewKeyFrame` (Campos et al. T-RO 2021).
Akzeptanz iff `(c1a ∨ c1b) ∧ c2` mit Spacing-Seite `c1a = N ≥ max_frames`,
`c1b = N ≥ min_frames` (Mapper-Idle-Pfad entfällt; Selector wird synchron
vor `mapper.run()` aufgerufen) und Novelty-Seite
`c2 = (ratio < 0.9) ∧ (matches ≥ min_tracked)`. `ratio = matches /
baseline_matches`, wobei `baseline_matches` der BFMatch-Count zwischen
`prev_kf` und dem **ersten Frame nach KF-Commit** ist (nRefMatches-Analog;
Original verwendet `mpReferenceKF->TrackedMapPoints(3)`, das ohne
Map-Punkt-Tracking nicht verfügbar ist). Diese Größe ist über die
KF-Lebensdauer konstant und macht das 0.9-Threshold semantisch identisch
zum Original („10 % Drop vom Tracking-Niveau am KF-Anfang"). Reference-KF
ist sequentieller `prev_kf` statt Covisibility-Graph-K_ref. Stereo-Pfad
(c1c, `bNeedToInsertClose`) sowie Inertial-Pfade (c3, c4) entfallen
— Mono-Only. Erster Frame nach jedem KF-Commit ist Baseline-Warm-Up und
wird zwingend abgelehnt. Historische Drei-OR-Logik vor 2026-05-26 war
ein Implementierungsfehler (siehe Doku-Abschnitt „Historie / Korrektur").

Formel-Details für Coko-SLAM: `docs/COKO_SLAM.md`. Folgt Li et al.
arXiv:2604.00804 (Apr 2026), Sec. 3.1; Entscheidung iff
`min_K ||ϕ(E)−ϕ(K)||₂ ≥ α` mit DINOv2-Small als ϕ und L2-normalisierten
Embeddings. Submap-Reset (Paper: alle 10 KFs) durch FIFO-Sliding-Window
der Länge `max_kfs=10` ersetzt; VINGS hat kein Submap-Konzept, das Window
ist der äquivalente lokale Vergleichshorizont. Diagnose-Modus
`force_accept_all` analog zu `mm3dgs`/`adaptive_kf`/`orbslam3` zur
α-Kalibrierung.

Formel-Details für AIM-SLAM: `docs/AIM_SLAM.md`. Folgt Jeon et al.
arXiv:2603.05097 (2026), Sec. III.C („SIGMA-Modul"); AND-Decision aus drei
Stages: Voxel-Overlap `O = |v(I_f) ∩ v(I_k)| / |v(I_f)|` (Eq. 1, Schwelle
`overlap_thresh`), EKF-Information-Gain `Γ = Σ 0.5·log(det(P_k⁻)/det(P_k⁺))`
(Eq. 2-3, per-ray-Schwelle `gain_thresh_per_ray`), Reduced-Chi-Square
`κ = bᵀb / (M − chi_dof_offset)` (Eq. 4, Schwelle `chi_thresh`) **auf
Hybrid-Residual aus Eq. 5** (Ray-Term auf Einheitssphäre + Pixel-
Reprojektions-Term gestackt). Korrespondenz `m_{k→f}` aus **ORB-BFMatch**
(pose-unabhängig — Voraussetzung für die Paper-Eq.-4-Semantik); Fallback
auf Reprojection-basierte Korrespondenz mit bilinearem Depth-Sampling
wenn rgb/cv2 fehlt oder zu wenig Matches. Paper-Slot ist Multi-View-
Reranking für VGGT-Input; hier auf binären Mapper-Slot übersetzt (siehe
Doku „Übersetzung auf den binären VINGS-Slot"). VGGT selbst, Sim(3)-
Optimierung und DINOv2-Loop-Closure sind bewusst nicht übernommen.

Beispiel-Configs: `configs/local/{frameselector,nurbs_lvi,mm3dgs,game_kfs,adaptive_kf,orbslam3,coko_slam,aim_slam}/` (smallcity) und
`configs/local/ntu_eee_03/{vista,nurbs_lvi,mm3dgs,game_kfs}/` (NTU-VIRAL).

## Wo steht was

| Datei | Inhalt |
|---|---|
| `REPO.md` | Repo-Struktur, Submodule, kritische Dateien |
| `MAPPING_TRACKING.md` | Tracking ↔ Mapping erklärt, frame_skip-Folgen |
| `KEYFRAME.md` | Profiling-Zahlen, Budget-Tabelle für den Selector, mapskip-Ergebnisse |
| `HOW_TO_RUN.md` | Run-Anleitung |
| `COMMANDS.md` | nützliche Aufrufe |
| `README.md` | Originale Repo-README (vorgelagertes Projekt) |

## Verwandte Knöpfe / Skripte

- `frame_skip: N` — naive Eingangs-Subsampling (verschlechtert Tracking, siehe MAPPING_TRACKING.md)
- `mapper_kf_skip: N` — jeden N-ten Tracker-KF an den Mapper geben; nur aktiv wenn FrameSelector aus
- `configs/local/{mapskip,skip_no_filter,vista,nurbs_lvi,mm3dgs,game_kfs}/` — Sweep-Configs pro Algorithmus
- `scripts/gen_configs.py --list` — verfügbare Sweep-Profile (mehr in `scripts/config_profiles.py` registrieren)
- `scripts/analyze_profiling.py` — Auswertung der PhaseTimer-Logs

## NTU-VIRAL-Datensatz vorbereiten

Roh: `~/Dokumente/datasets/eee_03/{eee_03.bag, camera_left.yaml, leica_prism.yaml, ...}`.
Vor dem ersten Run muss der Bag entpackt **und entzerrt** werden:

```bash
pip install rosbags                # einmalig in der vings-env
python scripts/prepare_ntu_viral.py \
  --bag   ~/Dokumente/datasets/eee_03/eee_03.bag \
  --calib ~/Dokumente/datasets/eee_03/camera_left.yaml \
  --leica ~/Dokumente/datasets/eee_03/leica_prism.yaml \
  --out   ~/Dokumente/datasets/eee_03 \
  --extract-pose
```

Output: `rectified/images/000000.jpg` + `rectified/intrinsic.txt` (post-Rectification K).
Bei `alpha=0.0` bleibt das neue K nahe am Rohwert; bei Bedarf in
`configs/local/ntu_eee_03_200.yaml` aktualisieren.

---

## Aerial-Dataset-Erkenntnisse (Mai 2026)

Detailliert in `docs/SESSION_NOTES_2026_05_20.md`. Pro Thema separate Docs.

### Top-Erkenntnisse (TL;DR)

1. **`use_metric: false` bei Aerial-Nadir-Szenen** ist ein +2-3 dB PSNR-Hebel.
   Metric3D auf flachem Boden gibt noisy Tiefen → falsche Gaussian-Platzierung.
   DroidNet-Tiefe aus dem internen BA reicht (TartanAir-trainiert = passende Domäne).
   Siehe `docs/RUN_CONFIG_PATTERNS.md`.

2. **`scripts/dynamic/dynamic_utils.py` ist DEAD CODE** — nicht in run.py-Mainloop
   aufgerufen. `use_dynamic: true` in der Config setzt nur Vis-Flags, kein actual
   dynamic removal. Wer das braucht, muss externe Masken via Loader liefern.
   Siehe `docs/SEGMENTATION_AMTOWN.md`.

3. **DJI `/local_position` hat 10 % Scale-Verzerrung** vs RTK + IMU-Velocity-
   Integration. Pose-Override mit local_position-basierten DJI-Posen ist deshalb
   nur leicht besser als pure VO. Für saubere Posen: RTK-basiert rekonstruieren.
   Siehe `docs/MARS_LVIG.md` Sec. "local_position-Bug".

4. **VIO auf MARS-LVIG ist strukturell unbrauchbar** wegen LiDAR-Degeneration bei
   Nadir-Flug + fehlender published Cam-IMU-Extrinsik. UAVScenes-Paper bestätigt
   den gleichen Befund. Wer VIO will: VINS-Fusion-Pre-Run für Online-Extrinsic-
   Estimation, dann die gelernten Werte in VINGS füttern.

5. **UAVScenes Class-IDs sind Cityscapes-style, NICHT Paper-Tab S9** (0-18).
   In den echten id-PNGs: **Sedan = 20**, **Truck = 24**, Vegetation = 13.
   Für AMtown03 sind 774 von 1120 Frames Sedan-positive (vs vorher 352 mit falschem
   ID-Mapping). Siehe `docs/SEGMENTATION_AMTOWN.md`.

6. **Quarry (DJI M600) hat eigene KB-Kalibrierung**, NICHT identisch zu Bell412.
   50-Pixel Principal-Point-Offset. Werte in `docs/QUARRY_DJI_M600.md`. Falls
   `scripts/rectify_dataset5_nadir.py` auf Quarry läuft = falsche Resultate.

### Persistierende Code-Patches in dieser Session

- **`scripts/storage/storage_manage.py`** Off-by-one + Shape-Mismatch-Bug gefixt
  bei langen Sequenzen. Crashed nicht mehr bei Frame ~489. Siehe
  `docs/STORAGE_MANAGER_FIX.md`.
- **`scripts/datasets/generic_vo.py`** + **`scripts/run.py`** unterstützen jetzt
  `dataset.ext_poses_file: ...` für externe Pose-Sources (z.B. DJI-RTK oder
  Terra-SfM). Halbiert GPU-Verbrauch + reduziert Drift. Format: TUM w2c.
  Siehe `docs/POSE_OVERRIDE.md`.
- **`scripts/run_experiment.py`** unterstützt `KEEP_RGBDNUA=1` env-var um alle
  Render-Frames zu behalten (für Mask-Overlay-Videos).

### Run-Pattern Quick-Reference

- **Kurze Aerial-Sequenz (≤ 500 Frames)**: paperloss + `use_metric: false` +
  `mapper_kf_skip: 1` + native PNG + image_size 384×456. → PSNR 23+ möglich.
- **Lange Sequenz (1000+ Frames)**: + Storage-Manager aktiv + skip 3-5 +
  num_keyframe 4-8. → PSNR plateau ~20, VRAM-Wand bei ~150 mapped frames.
- **Init-Phase wichtig**: `start_frame` ≥ 100-200 frames in die Sequenz hinein,
  Take-off / Hover am Anfang skippen.

### Datasets-Inventory + Empfehlung

| Pfad | Status | Empfehlung |
|---|---|---|
| `amtown03/images_all/` (2.6 GB) | aktive Daten | behalten |
| `amtown03/metadata/` (26 MB) | DJI + IMU + GPS + RTK | behalten |
| `uavscenes/amtown03_labels/` (106 MB) | Class-Masken AMtown03 | behalten |
| `HKairport_GNSS03.bag` (9.1 GB) | Source-Bag MARS HK | löschen (alles raus) |
| `mars_hkairport_gnss03/images_1900_3400_full/` (13 GB) | Long-Run-Frames | löschen (metrics docu) |
| `mars_vio_*` (~5 GB) | VIO-Stack | löschen (crashed) |
| `amtown03/vio_dji/` (12 KB) | VIO-Wrapper | löschen (crashed) |
