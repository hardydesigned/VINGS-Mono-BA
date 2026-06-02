# Sweep-Ergebnisse

Aggregierte CSVs der amtown03-Selektor-Sweeps. Eine Zeile pro Run, geschrieben
von `scripts/log_sweep_row.py`. Doppelte `variant`-Zeilen (Re-Runs) sind möglich
— die Notebooks deduplizieren auf den jüngsten `timestamp_end`.

| Datei | Slice | Frames | Stand |
|---|---|---|---|
| `s3100_200f_results.csv` | amtown03 3100–3299 | 200 | alle 56 Configs fair, 56/56 OK |
| `s1000_400f_results.csv` | amtown03 1000–1399 | 400 | alle 56 Configs neu (fair); 47 OK, 9 VRAM-FAIL bei 400 Frames |

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
