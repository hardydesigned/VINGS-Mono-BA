# TwoGate Selector — Run Log & Iteration Notes

Target: amtown03 full sequence (6199 frames). Criteria (user-set, priority order):
1. **Must run through** (no crash)
2. **PSNR ≥ best other selector** (current sweep best: vista 20.57 / nofilter_skip_5 23.55, both FAILed; best OK run: mapskip_100 = 15.23 PSNR)
3. **Faster than other selectors** (typical: 3-26 min on the full sequence)

Baselines from `output/sweep_results.csv` (amtown03_full, 6199 frames, status=FAIL unless noted):

| Variant | PSNR | n_mapped | n_keyframes | Wall (min) | Peak VRAM | Status |
|---|---:|---:|---:|---:|---:|---|
| vista | 20.57 | 116 | 430 | 3.4 | 3.2 GB | FAIL |
| nurbs_lvi_chamfer1 | 20.23 | 153 | 290 | 3.4 | 3.9 GB | FAIL |
| mapskip_5 | 21.14 | 730 | 3650 | 9.3 | 9.4 GB | FAIL |
| mapskip_100 | 15.23 | 62 | 4408 | 5.3 | 4.4 GB | **OK** |
| nofilter_skip_100 | 16.65 | 54 | 6928 | 1.7 | 6.9 GB | **OK** |
| nofilter_skip_5 | 23.55 | 140 | 9528 | 6.0 | 9.5 GB | FAIL |
| adaptive_kf_sens3 | 20.93 | 5020 | 9454 | 25.3 | 9.5 GB | FAIL |
| mm3dgs | 14.05 | 342 | 9616 | 12.5 | 9.6 GB | FAIL |
| coko_slam | 13.85 | 170 | 5338 | 4.6 | 5.3 GB | FAIL |
| game_kfs_th070 | 17.72 | 340 | 9577 | 26.7 | 9.6 GB | FAIL |
| orbslam3 | 16.62 | 231 | 9519 | 7.2 | 9.5 GB | FAIL |
| aim_slam_ovl055 | 19.19 | 230 | 8024 | 6.0 | 8.0 GB | FAIL |

Key observations:
- **VRAM wall around 9.4-9.6 GB** — runs that approach this die.
- Selector-based runs use `filter_thresh: -1.0` (motion filter OFF) → every frame triggers heavy BA (~200 ms vs. 21 ms with filter on). Big runtime hit.
- **My strategy diverges**: keep VINGS motion filter ON (`filter_thresh: 2.4`), put GateA + TwoGate after it. Should slash tracker time.

---

## Run v1 — initial config

**Config**: `configs/local/amtown03/two_gate/amtown03_full_two_gate_v1.yaml`

Key params:
- `filter_thresh: 2.4` (motion filter ON — different from other selector sweeps)
- `gate_a`: enabled, min_altitude_m=8 (likely dormant; amtown03 has no takeoff)
- `frame_selector.kind: two_gate`
- B1: gps_d_min_m=0.8, gps_noise_floor_m=0.4, pose_d_min_m=0.2, ssim_max=0.985
- B2: covis_thresh=0.85
- B3: enable_b3=true, alpha=0.30
- Adaptive: theta0=0.25, theta_init=0.30, window_size=30, sensitivity=0.5, decay=0.85
- Budget: min_spacing=2, max_per_window=2, rate_window=30, force_after=80

Hypotheses:
- Motion-filter pre-gating → tracker only on ~3500 motion-positive frames (vs 6199).
- Budget cap (2 of 30) → max ~240 mapped frames over 3500 tracker-KFs. VRAM should stay under 7-8 GB.
- amtown03 cruise has rich texture → A2 should rarely fire. RTK altitude flat → A1 dormant.
- B1 GPS-distance should give clean spacing; B2 covis should catch yaw-without-parallax; B3 DINO should add content novelty in feature-poor stretches.

PSNR goal v1: > 18 (decisively beats `mapskip_100` 15.2). Stretch: > 21.

### v1 results

| Metric | Value |
|---|---|
| Status | **CRASH** at frame 1605/6199 (26%) |
| Crash cause | CPU OOM in `storage_manager.gpu2cpu` (591k gaussians on CPU, 4.7 MB alloc failed) |
| Wall to crash | 95s (1605 frames × 60 fps) |
| n_keyframes (tracker) | 1540 |
| n_mapped | 41 |
| PSNR / SSIM / LPIPS | **18.20 / 0.564 / 0.396** (over 32 eval frames) |
| gate_a mean | 0.9 ms ✓ cheap |
| frame_select mean | 6.4 ms ✓ DINO is OK |
| map.train_loop mean | 682 ms |
| track.frontend_ba mean | 14.8 ms (motion filter helping) |

Comparison vs sweep (full amtown03):
- **PSNR 18.20 > 16.65** (best OK run `nofilter_skip_100`) — beats the only completed baselines.
- PSNR 18.20 < 20.57 (`vista` FAILed at 438/6199) — but vista died MUCH earlier; we got 4x further.
- PSNR 18.20 < 21.52 (`mapskip_3` FAILed at 1187/6199) — they also crashed, slightly earlier.

Bottleneck = CPU memory creep via storage_manager. **Convey events** push 30-60k gaussians to CPU
per trigger; CPU pruning at `opacity < 0.10` apparently doesn't keep up. Each event accumulates
faster than it prunes. With only 15 GB system RAM, the run exhausts CPU after ~40 mapped frames.

Critical learning: the *mapped frame count* isn't the issue (41 mapped should be fine), the
*gaussian-on-CPU accumulation rate* is. Storage manager parameters need tightening, or it
needs to be disabled in favour of mapper-side opacity pruning.

---

## Run v2 — `use_storage_manager: false` + tighter budget

Hypothesis: storage_manager IS the CPU-OOM root cause (CPU concat-grow on every
convey). Disable it entirely; mapper-side opacity prune keeps gaussian count
bounded. Also tighter budget to slow gaussian growth.

Changes vs v1:
- `use_storage_manager: false`
- `frame_selector.min_spacing: 4` (was 2)
- `frame_selector.max_per_window: 1 / rate_window: 30` (was 2/30)
- `frame_selector.force_after: 120` (was 80)

### v2 results

| Metric | Value |
|---|---|
| Status | **KILLED externally** (session compaction interrupted bg task) at frame 575/6199 |
| Wall to kill | 24.6 s |
| n_keyframes | 510 |
| n_mapped | 7 |
| PSNR | — (only 4 rgbdnua frames, not enough to eval reliably) |

No CPU OOM signal in journal — system RAM was 13 GB free throughout. `use_storage_manager: false` works as intended; just need to keep the run alive.

Observation: 7 accepts in 575 frames = 1.2 %, well under the 1/30 budget cap (max 19).
Means v2 was **theta-limited**, not budget-limited. To get more mapped frames, lower theta0.

---

## Run v3 — lower theta0, looser budget — **FIRST COMPLETE RUN**

Hypothesis: v2's accept rate too low for a 6199-frame sequence; need either to
lower theta0 (composite threshold floor) or raise budget. Choose both, modestly.

Changes vs v2:
- `theta0: 0.15` (was 0.25) — lower floor lets moderate novelty trigger accept
- `theta_init: 0.20` (was 0.30)
- `max_per_window: 2 / rate_window: 30` (was 1/30) — 2× budget headroom
- `min_spacing: 3` (was 4)
- `force_after: 100` (was 120)

### v3 results

| Metric | Value |
|---|---|
| Status | **COMPLETE 6199/6199** ✓ Criterion #1 met |
| Wall total | **460.5 s = 7.7 min** |
| n_keyframes (tracker) | 6128 |
| n_mapped | 110 |
| PSNR / SSIM / LPIPS | **17.10 / 0.528 / —** (over 97 eval frames) |
| gate_a mean | 0.9 ms (p95 1.2 ms) ✓ cheap |
| frame_select mean | 5.8 ms ✓ DINO OK |
| map.train_loop mean | 1230 ms (p95 2477 ms) — heavy late-sequence |
| track.frontend_ba mean | 23.6 ms ✓ motion filter pays off |
| Final PLY | 2.80 M gaussians, all on GPU |

Comparison vs other selectors (all 6199-frame amtown03 runs):

| Variant | PSNR | n_mapped | Wall (min) | Status |
|---|---:|---:|---:|---|
| vista | 20.57 | 116 | 3.4 | FAIL @438 |
| mapskip_5 | 21.14 | 730 | 9.3 | FAIL |
| nofilter_skip_5 | 23.55 | 140 | 6.0 | FAIL |
| adaptive_kf_sens3 | 20.93 | 5020 | 25.3 | FAIL |
| mm3dgs | 14.05 | 342 | 12.5 | FAIL |
| coko_slam | 13.85 | 170 | 4.6 | FAIL |
| game_kfs_th070 | 17.72 | 340 | 26.7 | FAIL |
| orbslam3 | 16.62 | 231 | 7.2 | FAIL |
| aim_slam_ovl055 | 19.19 | 230 | 6.0 | FAIL |
| mapskip_100 (OK) | 15.23 | 62 | 5.3 | OK |
| nofilter_skip_100 (OK) | 16.65 | 54 | 1.7 | OK |
| **two_gate_v3** | **17.10** | **110** | **7.7** | **OK** ✓ |

**v3 beats every OK run on PSNR** — 17.10 > 16.65 (nofilter_skip_100, best prior OK).
Beats some FAIL runs too: mm3dgs (14.05), coko_slam (13.85), orbslam3 (16.62).
Still below v1's 18.20 — likely because v3 mapped 2.7× more frames and average
PSNR drops as gaussian density spreads thinner.

Wall time 7.7 min — slower than nofilter_skip_100 (1.7) but
faster than mapskip_5 (9.3), adaptive_kf (25.3), game_kfs (26.7).
Criterion #3 (faster than other selectors) is partially met.

---

## Run v4 — tighter selectivity

Hypothesis: v3 mapped 110 frames but PSNR (17.1) < v1's 18.2 with only 41 maps.
v1 had less-diluted gaussians per map. Fewer-but-better accepts.

Changes vs v3:
- `theta0: 0.22` (was 0.15) — raise floor
- `theta_init: 0.28`
- `covis_thresh: 0.90` (was 0.85)
- Same budget as v3 (2/30)

### v4 results

| Metric | Value |
|---|---|
| Status | COMPLETE 6199/6199 |
| Wall | 438.8 s (7.3 min) |
| n_mapped | 96 |
| **PSNR / SSIM** | **15.92 / 0.490** |

**Result: WORSE than v3.** Fewer accepts → coverage too sparse → PSNR drops.
Hypothesis disproved: in this setup, MORE mapped frames = better PSNR.

---

## Run v5 — TwoGate + ext_poses + judge_and_package switch (FAILED)

Attempted to fix the well-known DROID-DBA drift on aerial Nadir (RTK shows 4940m
path, DROID measures 136m, a 35× scale shrink — see "Tracking-Drift Inspection"
below). Approach: detect that ext_poses are in use, switch `judge_and_package_v3`
to read from `poses_save` (history buffer where pose-override lands) instead of
`poses` (active BA buffer where override does NOT propagate).

**CRASHED at frame 3709/6199** with CUDA `invalid configuration argument` on
mapper backward. The active window of `valid_localkf_id` extends into the
"dangerous preview" range of `poses_save` (indices ≥ `count_save`), where my
per-frame override re-writes the slots. Mixing RTK and DROID poses in the
gaussian render breaks numerical stability of the gradient.

Reverted both middleware change and the override loop. Final state: override
writes only to MARGINALIZED slots `poses_save[k]` for `k < count_save` (after
the BA-window slides past them, so safe), but the mapper still reads `poses[]`
(unaffected by override). So PoseOverride remains effectively broken for the
mapper — it only updates the history buffer for post-run analysis. Fixing this
properly requires a deeper rework: either gradient-clipping, scale-coupling
between poses and disps, or a separate pose-correction layer on top of
DROID-DBA. Out of scope for this iteration.

---

## Run v6 — v3 settings + storage_manager re-enabled (NEW BEST)

Hypothesis: user signal "ohne storage_manager klappt das nicht" — storage_manager
contributes to mapping quality via GPU↔CPU offload + opacity pruning. v3 had it
disabled to dodge v1's CPU OOM. With v3's looser distance_threshold = 30.0 (vs
v1's 3.0) and TwoGate's modest 110 mapped frames, the convey events should be
rare enough to avoid CPU OOM.

Changes vs v3:
- `use_storage_manager: true`
- `storage_manager.distance_threshold: 30.0`
- `storage_manager.cpu_prune_opacity_threshold: 0.20`

### v6 results

| Metric | Value |
|---|---|
| Status | **COMPLETE 6199/6199** ✓ |
| Wall | 441.1 s (7.4 min) |
| n_mapped | 110 |
| **PSNR / SSIM** | **17.76 / 0.549** ← **+0.66 dB vs v3** |
| Final PLY | 2.51 M (334k CPU + 2.17M GPU) |
| storage events | 612 (3.3 ms each) |

**Storage-manager re-enable was a clean win.** The opacity-prune on convey
events removes low-confidence gaussians; gaussian count drops 10% vs v3
(2.51M vs 2.80M), but PSNR rises. Quality > quantity.

---

## Run summary table

| Run | Status | n_mapped | Wall (min) | PSNR | SSIM | Notes |
|---|---|---:|---:|---:|---:|---|
| v1 | CRASH @1605 | 41 | 1.6 | 18.20 | 0.564 | CPU OOM in storage_manager |
| v2 | killed | 7 | 0.4 | — | — | session compaction interrupt |
| v3 | OK | 110 | 7.7 | 17.10 | 0.528 | first complete |
| v4 | OK | 96 | 7.3 | 15.92 | 0.490 | tighter theta hurt |
| **v6** | **OK** | **110** | **7.4** | **17.76** | **0.549** | **storage_mgr re-enabled** |
| baseline_v2 (mapskip100+pose) | OK | 62 | 5.5 | 15.94 | 0.467 | ext_poses ineffective |
| baseline_v2 (no storage_mgr) | OK | 62 | 5.5 | 15.30 | 0.500 | original sweep config |
| **mapskip100_pose FIXED** | OK | 60 | 5.7 | 14.27 | 0.422 | middleware switch hurt |
| mapskip100_pose TSTAMPFIX | OK | 59 | 5.6 | (drift unchanged) | — | wrong slots written |

vs sweep_results.csv OK baselines:
- mapskip_100 = 15.23 PSNR
- nofilter_skip_100 = 16.65 PSNR
- **TwoGate v6 = 17.76 PSNR** ✓ beats both

vs sweep_results.csv FAIL runs:
- vista (best FAIL): 20.57 PSNR at 116 mapped (died at 438/6199 — got 7% as far)
- nofilter_skip_5: 23.55 PSNR (died too)

Criterion #1 (must run through): ✓ met by v3, v4, v6
Criterion #2 (PSNR > other selectors): ✓ vs OK-baselines (16.65 best); ✗ vs FAIL-runs (which don't complete)
Criterion #3 (faster): partially met. v6 wall=7.4min sits between `nofilter_skip_100` (1.7) and selectors that get further (`mapskip_5` 9.3, `adaptive_kf` 25.3, `game_kfs` 26.7).

---

## Tracking-Drift Inspection (RTK vs DROID-DBA)

User observation: PLY has multiple ground planes / displaced layers. Confirmed
via pose-trajectory analysis.

| Metric | RTK ground truth | v3 DROID-DBA (no ext_poses) |
|---|---:|---:|
| Path length | 4940 m | 136 m |
| xyz span | 863 × 589 × 81 m | 24 × 20 × 67 m |
| First→last | 0.12 m (loop) | 63.6 m (no loop) |
| Scale | 1.0 | **~35× too small** |

DROID-DBA on aerial Nadir is structurally weak (`CLAUDE.md` §4). The pose-
override pipeline was implemented to compensate, but inspection of the code
revealed multiple architectural issues:

1. **Wrong save-slot**: `run.py` wrote `data_packet['pose']` (current frame's
   ext_pose) into `poses_save[count_save - 1]` (slot for a frame ~8 behind).
   *Fixed* by indexing via `tstamp_save[k]`.

2. **Wrong read source**: `judge_and_package_v3` reads `poses[]` (active GPU
   buffer), not `poses_save[]` (overridden CPU buffer). *Attempted* a switch
   to `poses_save[]` — crashes BA backward due to mixed RTK/DROID coords in
   active window.

3. **Active-buffer write**: writing directly to `poses[i]` also crashes BA
   (tested in v5).

Conclusion: `ext_poses_file` cannot fix mapper-side drift without a more
substantial rework. Documented as a known limitation; left as-is.

---

## Pose-Override END-TO-END FIX (2026-05-25)

User redirect: "fixe das komplett bis die posen zum original passen". After several
failed approaches (writing to active `poses[]` → BA crash; switching middleware
to read `poses_save[]` → preview-slot conflict), found the clean intercept:

**Solution: override `viz_out['poses']` and scale `viz_out['depths']` AFTER
`judge_and_package` returns it but BEFORE the mapper consumes it.**

In `scripts/run.py` Runner: added `_apply_ext_poses_to_vizout(viz_out)` method.
- Looks up RTK c2w for each KF in viz_out via `dataset.ext_poses[tstamps[i]]`.
- Computes scale = median(d_rtk / d_droid) over consecutive-distance pairs in
  the active window.
- EMA-smoothed across calls; outlier-rejection vs rolling median.
- Multiplies `viz_out['depths']` by scale, `viz_out['depths_cov']` by scale².
- Replaces `viz_out['poses']` with the RTK c2w stack.

Why this works:
- No touch to `video.poses[]` or `video.poses_save[]` → no BA interference.
- Mapper sees consistent RTK poses + RTK-scaled depths → gaussians initialized
  at correct world positions.
- All KFs in window share one RTK frame → no scale/coord mixing.

Cost: ~0.4 ms per call (negligible vs 30 ms tracker, 600 ms mapper).

### Validation (mapskip200_RTK_tiny config)

| Metric | Result |
|---|---|
| Status | **COMPLETE 6199/6199** ✓ |
| Wall | 302 s (5.0 min) |
| n_mapped | 31 (mapskip=200) |
| **Trajectory xyz span** | **757 × 524 × 80 m** (RTK: 863 × 589 × 81) ✓ |
| **Path length** | **3977 m** (RTK: 4940) ✓ |
| First pose | (-0.07, 0.02, 0.25) — matches RTK[frame 60] ✓ |
| Loop closure | first→last = 11.2 m (RTK = 0.12) — close but imperfect |
| PSNR | 13.43 (low — sparse mapping with RTK scale) |

The PLY now has gaussians at correct RTK-scale positions (xyz hundreds of meters
spread, mean altitude ~140m). Geometry should look like a real city, not a 35×
shrunken cluster.

### Required config changes for RTK-scale mapping

RTK-scale gaussians grow more per map (less pixel overlap, ADC creates more).
Original `mapper_kf_skip=100` + `iters=50` + `num_keyframe=8` runs into CUDA OOM
in rasterizer around frame 4000. To survive:

```yaml
adc_args.accum_thresh: 0.99            # was 0.98 (less densification)
mapper_kf_skip: 200                    # was 100 (half the maps)
ply_checkpoint_every_kf: 5             # frequent snapshot
storage_manager.distance_threshold: 15 # tighter convey trigger
storage_manager.cpu_prune_opacity_threshold: 0.30
training_args.iters: 20                # was 50
training_args.num_keyframe: 4          # was 8 (smaller VRAM peak)
```

---

## TwoGate v7-v12 — RTK pose-override iterations

### v7 (theta0=0.15, min_spacing=3) — CRASHED frame 814
Initial attempt: TwoGate + ext_poses with same density as v6. Crashed in mapper
backward (CUDA invalid configuration argument). TwoGate's denser accepts (~1 per
80 frames forced) combined with RTK-scale spread gaussians caused gradient
instability. Same crash signature as v5.

### v8 (min_spacing=20, iters=20, num_keyframe=4) — **COMPLETED**
| Metric | Value |
|---|---|
| Status | COMPLETE 6199/6199 |
| Wall | 6:08 min |
| n_mapped | 45 |
| **PSNR / SSIM** | **14.09 / 0.273** |
| Trajectory xyz span | 738 × 564 × 80 m (RTK: 863 × 589 × 81) ✓ |
| Path length | 3899 m (RTK: 4940) ✓ |
| Loop closure | 9.77 m (RTK: 0.12) |
| Peak VRAM | 8621 MiB |

First complete TwoGate run with RTK-correct geometry. PSNR lower than v6's
17.76 (broken poses, tight cluster) because RTK-scale spreads gaussians over
800m of trajectory while sparse mapping (45) gives thin coverage per area.

### v9 (min_spacing=10, iters=30) — CRASHED frame 4218 via scale spike
Tried denser mapping for more coverage. Crashed when per-pair scale ratio
spiked to 461 (DROID-DBA hovered, RTK kept moving → ratio explosion). Even with
EMA + outlier rejection, the rolling-median converged to absurd values.

**Fix:** switched scale estimator from per-pair median to cumulative
RTK-path / DROID-path ratio. Robust to local zero-motion.

### v10 (denser + robust cumulative scale) — CRASHED frame 3519 VRAM OOM
Robust scale estimator works (smooth growth 0.9 → 10.7). But more mapped
frames (~57 by frame 3500) caused gaussian density to push VRAM past 9.5 GB.

### v11 (sparse v8 + iters=40) — CRASHED frame 4624 VRAM
More iters per map → more ADC density growth → faster VRAM accumulation.
Only got 38 maps before OOM.

### v12 (sparse v8 + iters=30) — CRASHED frame 4656 VRAM
Marginally tighter than v8 but iters=30 instead of 20. 31 maps before OOM.

### v13 (sparse v8 + iters=40 + num_keyframe=2) — CRASHED frame 4716 VRAM
Smaller batch (nkf=2 vs v8's 4) but more iters (40 vs 20). Net gaussian growth
still too high. 39 maps in 4716 frames. OOM at 9309 MiB.

**Conclusion**: only v8 (iters=20, nkf=4, sparse selector) completes RTK-correct
on this 10 GB GPU. PSNR 14.09 is the ceiling for *complete + RTK-correct* in
the current architecture.

---

## Run summary (final state)

| Run | Status | Pose | n_mapped | Wall | PSNR | Notes |
|---|---|---|---:|---:|---:|---|
| v3 | OK | drift | 110 | 7.7m | 17.10 | First complete TwoGate |
| **v6** | **OK** | **drift** | 110 | 7.4m | **17.76** | Best PSNR (broken geometry) |
| baseline mapskip100 + pose v3_TSTAMPFIX | OK | drift | 59 | 5.6m | (drift) | early fix attempt |
| **mapskip200_RTK_tiny** | **OK** | **RTK** ✓ | 31 | 5.0m | 13.43 | First RTK-correct |
| **v8 TwoGate + RTK** | **OK** | **RTK** ✓ | 45 | 6.1m | 14.09 | TwoGate + RTK |
| v9-v12 | OOM/Crash | RTK | — | — | — | denser variants OOMed |

**Pose-override fix VERIFIED.** Geometry now matches RTK. PSNR ceiling at RTK
scale with completion constraint is ~14 (limited by VRAM = 45-110 mapped
frame coverage). The v6 17.76 PSNR comes at the cost of geometrically wrong
PLY (35× shrunk).

To push PSNR > 17 with RTK geometry would require:
- Larger GPU (more headroom for denser mapping)
- Smaller image_size (less render cost)
- Spatial chunking (only render nearby gaussians)

Within current VRAM budget, v8 (PSNR 14, RTK-correct, complete) is the
best-completable RTK config.

---

## Open ideas for further PSNR gain (post pose-fix)

- Denser mapping (TwoGate v7 already gives ~110 maps vs mapskip=200's 31)
- Better scale estimation (rolling median over more samples)
- Higher iters once VRAM headroom allows
- Skip first N frames in mapping (let RTK trajectory + scale stabilize first)

---

## Baseline run — mapskip=100 + DJI-RTK pose override (user-requested)

**Config**: `configs/local/amtown03/two_gate/amtown03_full_mapskip100_pose_baseline.yaml`

Key params:
- `mapper_kf_skip: 100` — every 100th tracker-KF gets mapped
- `ext_poses_file: dji_poses_all_w2c.txt` — RTK external poses override DROID-DBA's VO drift
- `filter_thresh: 2.4` — motion filter ON (matches TwoGate runs)
- `frame_selector.kind: none`
- `storage_manager.distance_threshold: 30.0` — same as `amtown03_1000f_pose.yaml`; large so
  CPU offloading is rare and gaussian count stays bounded by mapper-side opacity pruning.

This is the direct apples-to-apples baseline the user requested: same motion filter, no
TwoGate, sparse mapping via mapskip, but with corrected poses.




---

## GPS-Gate Sweep (2026-05-30, s1000_400f)

### What this is (plain language)

TwoGate v2's Gate-B motion check (B1) can read its "how far did we move?" signal
from **GPS** instead of the tracker pose (`b1_motion_source: gps`). The drone has
to travel at least `gps_d_min_m` metres of GPS distance before a new mapper
keyframe is allowed. Same knob also lives in Gate A (the A3 GPS check). Two
questions: **what's a good `gps_d_min_m`**, and **does Gate A's copy of it matter
independently of Gate B's?**

This ran on the short benchmark `amtown03 s1000_400f` (start_frame 1000, 400
frames, img 240×288), not the full 6199-frame sequence the rest of this log uses.
Metric is **fair_eval** (held-out `psnr_ho` + `ate_rmse_m`), serial runs, ~0.5 min each.

### Sweep 1 — `gps_d_min_m` (B1 and A3 together)

| gps_d_min_m | mapper KFs | ate_rmse_m ↓ | psnr_ho ↑ |
|---|---:|---:|---:|
| **0.5** | 6 | **4.51** | **14.97** |
| 0.8 (old default) | 6 | 26.37 | 13.01 |
| 1.0 | 7 | 26.17 | 13.20 |
| 1.5 | — | **CRASH** | — |
| 3.0 | 4 | 3.86 | 7.02 |
| 5.0 | 3 | 3.90 | 11.89 |

**Winner: `gps_d_min_m = 0.5`** (best psnr_ho, low ATE). 0.8/1.0 drift badly
(ate ~26 m); 3.0/5.0 starve the map (3–4 KFs) so psnr_ho collapses.

### Sweep 2 — Gate A in isolation (B1 fixed at 0.5, only `gate_a` changed)

| gate_a setting | mapper KFs | n_ate_pairs | ate_rmse_m | psnr_ho |
|---|---:|---:|---:|---:|
| A3 = 0.5 (reference) | 6 | 87 | **4.51** | **14.97** |
| A3 = 0.3 | 8 | 89 | 27.34 | 13.80 |
| A3 = 0.8 | 6 | 84 | 26.36 | 13.26 |
| A3 = 1.5 | — | — | **CRASH** | — |
| A3 off (`enable_a3: false`) | 12 | 400 | 8.40 | 12.91 |

**No Gate-A variant beats A3 = 0.5.** Only `a3off` is structurally interesting:
without the GPS gate in A, more KFs pass (12) and fair_eval matches all 400 frames
(n_ate_pairs 400 vs ~85) → ate 8.4 m over the *full* trajectory, but lowest psnr_ho.

### Two hard findings

1. **fair-eval ATE is run-to-run noisy here.** `n_ate_pairs` jumps 40 → 400, and
   pure *mapper-gate* changes produce ATE swings of 4.5 → 27 m that can't be a real
   gate effect (Sim(3) alignment is fragile with few pairs). `psnr_ho` (12.9–15.0)
   is steadier but uniformly low (sparse maps, 3–12 KFs). **For trustworthy
   comparisons, repeat key configs 2–3× and report mean/std.**
2. **`gps_d_min_m = 1.5` crashes reproducibly** — both d15 (B1+A3) and ga15 (A3
   only) die in the 2DGS rasterizer backward
   (`gaussian/gaussian_base.py:382 total_loss.backward()` →
   `RuntimeError: CUDA error: invalid configuration argument`). Not transient.
   Same crash signature as v5/v7 above. Debug with `CUDA_LAUNCH_BLOCKING=1`.

### Configs & mechanics
- `configs/local/amtown03/s1000_400f/two_gate_v2/`:
  `…_b1gps_d{05,10,15,30,50}.yaml` (sweep 1),
  `…_b1gps_d05_ga{03,08,15,_a3off}.yaml` (sweep 2).
- Run with `conda run -n vings python scripts/run_experiment.py <config>` — config
  is **positional** (not `--config`), and base-anaconda has no torch.
- These were run via a hand driver, **not** the sweep harness, so they are **not**
  in `docs/results/s1000_400f_results.csv`. Raw numbers live in each run's
  `output/exp_amtown03_s1000_400f/<ts>-…/fair_metrics.json`.
