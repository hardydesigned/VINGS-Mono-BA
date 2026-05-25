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
| `vista` | `frame_selector.py` (`FrameSelector`) | View-Angle-Diversity pro Voxel (VISTA Eq. 2/3); Pose-Filter vorgeschaltet | nein |
| `nurbs_lvi` | `nurbs_lvi_selector.py` (`NurbsLviSelector`) | Adaptive Q-Score nach NURBS-LVI: ORB-Matches + Chamfer + Sektor-Migrationen | ja |
| `mm3dgs` | `mm3dgs_selector.py` (`Mm3dgsSelector`) | Covisibility-Drop (< 95 %) vs. prev_kf + NIQE-Min (Variance-of-Laplacian) im Sliding-Window | ja |
| `game_kfs` | `game_kfs_selector.py` (`GameKfsSelector`) | Game-theory-inspirierte Composite-Cost aus FRA (uncert+render+covis) und DRA (assoc+flow+motion) mit EMA-λ; siehe `docs/GAME_KFS.md` | ja |
| `adaptive_kf` | `adaptive_kf_selector.py` (`AdaptiveKfSelector`) | Momentum-Aware Adaptive Threshold (`θ = max(θ₀, μ+k·σ)` + Decay γ) über Hybrid-Error (Photo + SSIM via Depth-Warping); siehe `docs/ADAPTIVE_KF.md` | ja |
| `orbslam3` | `orbslam3_selector.py` (`OrbSlam3Selector`) | ORB-SLAM3-„New Keyframe Decision": Drei-Bedingungs-OR (force-rate ∨ tracking-weak ∨ `matches/n_kp_prev < 0.9`); siehe `docs/ORB_SLAM3.md` | ja |
| `coko_slam` | `coko_slam_selector.py` (`CokoSlamSelector`) | DINOv2-Feature-Distanz: accept iff `min ||ϕ(E)−ϕ(K)||₂ ≥ α` über FIFO-Sliding-Window (Paper: 10 KFs/Submap); siehe `docs/COKO_SLAM.md` | ja |
| `aim_slam` | `aim_slam_selector.py` (`AimSlamSelector`) | SIGMA-Modul: Voxel-Overlap (Stage 1) + EKF-Kovarianzreduktion Info-Gain (Stage 2) + Reduced-Chi-Square-Stabilität (Stage 3), AND-Decision; siehe `docs/AIM_SLAM.md` | nein |
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
  sector_angle_deg: 2.0    # Paper-Default 15°; bei VINGS zu grob (siehe NURBS_LVI.md)
  chamfer_lambda: 0.5
  min_matches: 15          # force-accept wenn Mc < min_matches
  force_accept_all: false  # true = Diagnose-Modus (loggt Q/mig, akzeptiert alles)
  min_depth: 0.2
  max_depth: 35.0
```

```yaml
frame_selector:
  kind: mm3dgs
  covis_thresh: 0.95       # accept wenn Overlap mit prev_kf < 95 %
  niqe_window: 5           # Sliding-Window für Quality-Gate (Lap-Var)
  n_samples: 2048          # Pixel für die Backprojection
  force_accept_after: 0    # >0 = Failsafe: Accept wenn N Frames hintereinander covis<thresh aber nicht best
  min_depth: 0.2
  max_depth: 35.0
```

```yaml
frame_selector:
  kind: adaptive_kf
  theta0: 0.05             # untere Schwelle (Eq. 6)
  theta_init: 0.10         # Warm-up-Startwert
  window_size: 30          # W
  sensitivity: 2.0         # k
  decay: 0.85              # γ (Eq. 7)
  w_photo: 0.85            # Hybrid-Error-Gewichte (Algorithmus 1 rekonstruiert)
  w_ssim: 0.15
  min_overlap_pixels: 1000 # |M|-Fail-safe
  force_accept_all: false  # true = Diagnose-Modus
  min_depth: 0.2
  max_depth: 35.0
```

```yaml
frame_selector:
  kind: orbslam3
  orb_n_features: 800
  tracked_ratio_thresh: 0.9   # Paper-Default: accept wenn matches/n_kp_prev < 0.9
  min_tracked: 50             # tracking-weak Notfallpfad (Paper: "< 50 points")
  min_frames: 1               # Spacing-Untergrenze (vermeidet Bursts)
  max_frames: 30              # force-rate Failsafe (1s @ 30fps)
  force_accept_all: false     # true = Diagnose-Modus
  min_depth: 0.2
  max_depth: 35.0
```

```yaml
frame_selector:
  kind: coko_slam
  alpha: 0.4               # L2-Distanz-Threshold im DINOv2-Feature-Space
  model_name: dinov2_vits14
  image_size: 224          # muss Multiple of 14 sein (DINOv2-Patch-Size)
  device: cuda
  max_kfs: 10              # FIFO-Window (Paper: 10 KFs/Submap); 0 = unbounded
  force_accept_all: false  # true = Diagnose-Modus (loggt min_dist, akzeptiert alles)
```

```yaml
frame_selector:
  kind: game_kfs
  beta_uncert: 0.3         # FRA-Gewichte (Eq. 3)
  beta_render: 0.3
  beta_covis: 0.4
  alpha_assoc: 0.5         # DRA-Gewichte (Eq. 9)
  alpha_flow: 0.3
  alpha_motion: 0.2
  eta: 0.8                 # EMA-Smoothing für λₜ (Eq. 14)
  accept_thresh: 0.5       # accept iff λ·A + (1-λ)·B ≥ thresh
  orb_n_features: 800
  flow_ref_px: 30.0        # Sequenz-spezifische Normalisierungen
  lap_var_ref: 500.0
  cov_ref: 1.0
  trans_ref_m: 0.30
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
  # Stage 3: Reduced-Chi-Square (Eq. 4)
  use_chi_square: true
  chi_thresh: 1.0             # κ > 1 → instabil → accept
  force_accept_all: false     # true = Diagnose-Modus (loggt alle 3 Stats)
  min_depth: 0.2
  max_depth: 35.0
```

Formel-Details für NURBS-LVI: `docs/NURBS_LVI.md`. Folgt Wu et al. TMECH 2026
Sec. III.A; Entscheidungsregel ist `Or + Oc > Q` (Q ist die *Schwelle*, nicht
der Score).

Formel-Details für Adaptive-KF: `docs/ADAPTIVE_KF.md`. Folgt ROBOVIS 2026
Submission, Algorithmus 2 verbatim (`θ = max(θ₀, μ + k·σ)` + Decay γ). Algorithmus 1
(Hybrid-Error) ist rekonstruiert weil das offizielle Repo
(`jhakrraman/Adaptive_Keyframe_Selection`) ihn nicht enthält — verwendet
depth-basiertes Backward-Warping mit `cv2.remap`, L1-Photometric + mask-gefülltes SSIM.

Formel-Details für MM3DGS: `docs/MM3DGS.md`. Folgt Sun et al. IROS 2024
Sec. III.E; Entscheidungsregel ist `covis < 0.95` AND `argmax(lap_var) in window`.
NIQE des Originals durch Variance-of-Laplacian-Proxy ersetzt; Depth-Quelle ist
VINGS-Tracker-Depth statt Map-Rendering.

Formel-Details für Game-KFS: `docs/GAME_KFS.md`. Folgt Chen et al. RA-L 2025
Sec. III; Composite-Cost `L = λₜ·A_t + (1−λₜ)·B_t` mit Sub-Scores aus FRA
(Uncertainty/Render/Covisibility) und DRA (Assoc/Flow/Motion), Entscheidung
via Schwelle `accept_thresh`. Mapper-frei: alle FRA-Signale via Tracker-Proxies
(depth_cov, LapVar, Pixel-Reprojection).

Formel-Details für ORB-SLAM3: `docs/ORB_SLAM3.md`. Folgt Campos et al. T-RO
2021 („New Keyframe Decision"); Akzeptanz iff eine von drei Bedingungen
zutrifft: `N ≥ max_frames` (force-rate), `matches < min_tracked`
(tracking-weak) oder `matches/n_kp_prev < 0.9` AND `N ≥ min_frames`
(ratio-drop). Reference-KF ist sequentieller `prev_kf` statt
Covisibility-Graph-K_ref (VINGS hat keinen ORB-Graph).

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
`κ = bᵀb / (M − chi_dof_offset)` (Eq. 4, Schwelle `chi_thresh`). Paper-Slot
ist Multi-View-Reranking für VGGT-Input; hier auf binären Mapper-Slot
übersetzt (siehe Doku „Übersetzung auf den binären VINGS-Slot"). VGGT
selbst, Sim(3)-Optimierung und DINOv2-Loop-Closure sind bewusst nicht
übernommen.

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
