# Sweep-Ergebnisse

Aggregierte CSVs der amtown03-Selektor-Sweeps. Eine Zeile pro Run, geschrieben
von `scripts/log_sweep_row.py`. Doppelte `variant`-Zeilen (Re-Runs) sind möglich
— die Notebooks deduplizieren auf den jüngsten `timestamp_end`.

| Datei | Slice | Frames | Stand |
|---|---|---|---|
| `s3100_200f_results.csv` | amtown03 3100–3299 | 200 | alle 56 Configs fair, 56/56 OK |
| `s1000_400f_results.csv` | amtown03 1000–1399 | 400 | alle 56 Configs neu (fair); 47 OK, 9 VRAM-FAIL bei 400 Frames |
| `interval1_survey_full_results.csv` | interval1_AMtown03 1000–5100 | 13×500 | mono+GPS-Voll-Survey, 1 Zeile/Segment; 6 OK / 7 rc=137-Teil-ply (alle nutzbar) |

## Faire vs. unfaire Qualitätsspalten

* `psnr` / `ssim` / `lpips` — **train-view, selektionsabhängig** (nur auf den vom
  Selektor zufällig gemappten Frames). NICHT für Selektor-Vergleich geeignet.
* `psnr_ho` / `ssim_ho` / `lpips_ho` — **fair**: Held-out-Novel-View an festen
  Frame-Positionen (jede 10.), gleiches Set für alle Configs.
* `ate_rmse_m` / `ate_mean_m` — Sim(3)-alignierte Trajektorien-Fehler vs. GT
  (DJI). `n_ate_pairs` = Zahl gematchter KFs, `n_tracked` = getrackte Frames.
* `n_eval_ho` — Zahl der Held-out-Eval-Frames (Stride 10).

Methodik: `docs/FAIR_EVAL.md`. Erzeugt von `scripts/eval/fair_eval.py` (pro Run
`fair_metrics.json` + `fair_eval/ho_*.png` im Run-Ordner). Analyse-Notebooks:
`scripts/analyze_sweep_s3100_200f.ipynb`, `scripts/analyze_sweep_s1000_400f.ipynb`.

Hinweis s1000: Voller fairer Sweep gelaufen — 47/56 Runs mit fairen Metriken.
Die 9 FAILs (kein `*_ho`/`ate_*`) crashten an der ~8 GB-VRAM-Wand bei 400 Frames
(dichte Mapper): `aim`, `aim_gain030`, `aim_ovl055`, `aim_ovl085`, `coko_a005`,
`nofilter_skip_1`, `nofilter_skip_2`, `nurbs`, `nurbs_sec1`. Bei s3100 (200
Frames) crasht nichts → 56/56 fair.

Faire Notebooks (`psnr` := held-out `psnr_ho`, gefiltert auf Runs mit fairer
Eval): `scripts/analyze_sweep_s3100_200f_fair.ipynb` (56/56),
`scripts/analyze_sweep_s1000_400f_fair.ipynb` (47/56).

## interval1-Voll-Survey (`interval1_survey_full_results.csv`)

Mono+GPS-Survey über den ganzen Cruise (`scripts/run_interval1_survey.sh`), eine Zeile
pro durchgehendem 500-Frame-Segment (Start 1000…4600, Step 300) **plus eine Aggregat-
Zeile `variant=ALL`** für den Gesamtlauf. Inline von `log_sweep_row.py` während des
Laufs geschrieben; am Ende sauber regeneriert (Segmente + ALL) von
`scripts/eval/measure_survey.py` — auch retroaktiv aus vorhandenen Run-Ordnern. Gleiches
47-Spalten-Schema.

Die `ALL`-Zeile: `duration_min`/`wall_total_s` = **Summe** (serielle Gesamt-Compute-Zeit),
`ply_mb_final` = **finale gemergte survey_complete.ply** (nicht Summe der Segmente),
`psnr`/`ssim`/`lpips` = über `n_metric_frames` **gewichtetes Mittel**, Timings über
`n_mapped`/`n_processed` gewichtet, `peak_vram_mib`/`peak_ram_gb` = **Max**, KF/Frame-
Spalten = Summe, `status` = `<n>OK/<n>rc137`.

**interval1-spezifische Vorbehalte (anders als amtown03!):**
* `psnr_ho`/`ssim_ho`/`lpips_ho` sind **leer/null** — die Segmente laufen non-metrisch
  im reinen DROID-Frame (`--no-ext`); der Held-out-Render nutzt die gedrifteten,
  skalenfreien Tracker-Posen → kein sinnvoller Novel-View. Die Survey-Qualität wird
  NICHT hierüber bewertet, sondern über die geometrische Boden-Kohärenz nach dem
  GPS-Leveling + PLY-Render. Siehe `docs/INTERVAL1_LIDAR_PIPELINE.md`.
* `ate_rmse_m` misst die **Tracker-Drift** (DROID vs GT, 6–31 m), NICHT die Karte — die
  wird per `sim3_unwarp --gps-csv` GPS-verankert (Kamera-RMSE ~1.8 m). Nur bei sauberem
  Exit (status OK) gesetzt; rc=137-Segmente crashen vor `fair_eval`.
* `status=FAIL`/`exit_code=137` = VRAM-Wand (peak ~9.8 GB), aber die Teil-ply nach dem
  letzten Checkpoint ist **nutzbar** (`n_keyframes`/`ply_mb_final` zeigen wie viel) —
  7/13 Segmente sind so partielle, aber verwendete Maps. `train-view psnr` (~18–24 dB)
  ist optimistisch und sagt nichts über die 3D-Geometrie (s2800: psnr 23.6 trotz
  schlechtester Boden-Schmiere).
