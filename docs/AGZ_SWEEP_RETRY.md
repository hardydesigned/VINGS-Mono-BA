# AGZ-Sweep Retry-Liste

Sweep gestartet: **2026-05-15 16:17** via `bash scripts/run_agz_full_sweep.sh`
(nohup detached, Log: `/tmp/agz_sweep.log`).

Hier kommen Configs rein, die *vorzeitig* gecrashed sind ohne erwartbaren Grund
(also nicht `mapskip1` / `nofilter_skip1`, die per Design crashen). Schwelle:
**< 150 input frames** = vermutlich Setup-Problem (OOM, Watchdog, IO).

Format:  `- [STATUS] config | last_input_idx=X | last_mapped_frame=Y | n_kfs=Z | wahrscheinliche Ursache`

## Erwartete Crashes (NICHT retryen)

- `agz_full_mapskip1` — jeden KF mappen → Mapper-Last sprengt VRAM/RAM (~200 frames).
- `agz_full_nofilter_skip1` — jeden Frame trackieren+mappen, gleicher Grund.
- `agz_full_nurbs_diag` — `force_accept_all: true` (Diagnose), ähnliches Profil wie mapskip1.
- `agz_adaptive_kf_diag` — `force_accept_all: true` (Diagnose), bewusst kurz.

## Unerwartete Crashes (retryen)

### Systemisches Problem: GPU-OOM bei AGZ-Render-Auflösung

AGZ-Intrinsic ist 1080×1920 (Tracker läuft auf 360×640, aber 2DGS-Rasterizer
rendert in Originalauflösung). Das sprengt 9.64 GiB VRAM. Plain-Retry hilft
nicht — **vor dem Retry intrinsic skalieren** (z.B. /2 → 540×960) oder
`max_views_per_voxel`/`num_keyframe` runter, oder beides.

| Config | rc | last_idx | n_kfs | PSNR | Fehler |
|---|---|---|---|---|---|
| `agz_full_frameselector_g040` | 137 (SIGKILL) | 254 | 78 | 22.30 | OOM-killer / VRAM-Watchdog, GPU peak 9639 MiB |
| `agz_full_frameselector_g030` | 1 (CUDA OOM) | ~298 | 92 | 22.24 | `torch.cuda.OutOfMemoryError` in `diff_surfel_rasterization`, 1014 MiB free |
| `agz_full_frameselector_g020` | 1 (CUDA OOM) | 288 | 88 | 22.27 | gleicher OOM, 5.5 min |

**Vista-Familie komplett OOM bei AGZ-Auflösung** (alle 3 Configs gecrasht zwischen Frame 250–300). Render-Resolution muss runter.


## Notizen

- Wenn ein Config nur als FAIL_rc137 / FAIL_rc-9 erscheint UND last_input_idx ≥ 150: vermutlich VRAM-Watchdog, retry mit weniger parallelem Load.
- Wenn FAIL_rc1 + last_input_idx < 50: Setup-Bug (Config, Dataset-Pfad, Permission). Hier hilft Retry nicht.
- Bei verdächtigem Pattern (3+ Configs in Folge fail) → Sweep stoppen und Debug.
