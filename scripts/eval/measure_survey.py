#!/usr/bin/env python3
"""Erzeuge retroaktiv die Sweep-CSV (Schema wie docs/results/s1000_400f_results.csv)
aus bereits gelaufenen interval1-Survey-Segment-Ordnern.

Für jedes Segment wird der ERFOLGREICHE Run-Ordner gewählt (höchste n_mapped aus
profiling.json -- bei mehreren Ordnern, z.B. gestorbener Erst-Lauf + erfolgreicher
Gap-Re-Run). ply-Spalten kommen aus der GPS-unwarpten sN_gps.ply (die rohe Run-ply
wird nach dem Unwarp gelöscht). Nutzt die Parser aus scripts/log_sweep_row.py.

Usage:
  python scripts/eval/measure_survey.py --out-dir output/exp_interval1_survey \
     --csv docs/results/interval1_survey_full_results.csv
"""
import argparse, os, sys, glob, json, csv
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # scripts/
import log_sweep_row as L

# Survey-Summary-CSV: EINE Zeile pro Survey-Lauf (Gesamtdauer, gemergte ply, min/max/mean
# der Metriken). Vergleichshorizont = ganze Surveys (später z.B. je Selektor/Auflösung),
# nicht einzelne Segmente. Für jede Metrik unten werden _min/_max/_mean (über die Segmente)
# erzeugt; psnr/ate haben zusätzlich _wmean (frame-/pair-gewichtet = faires Headline-Mittel).
SUMMARY_METRICS = ["psnr", "ssim", "lpips", "ate_rmse_m",
                   "psnr_ho", "ssim_ho", "lpips_ho",
                   "peak_vram_mib", "peak_ram_gb", "duration_min"]
SUMMARY_HEAD = ["survey_name", "timestamp", "dataset", "selector_kind",
                "n_segments", "n_ok", "n_fail",
                "duration_total_min", "wall_total_s",
                "ply_mb_merged", "n_gaussians_merged",
                "n_keyframes_total", "n_mapped_total", "n_dataset_frames_total"]
SUMMARY_COLUMNS = SUMMARY_HEAD + [f"{m}_{s}" for m in SUMMARY_METRICS for s in ("min", "max", "mean")] \
                  + ["psnr_wmean", "ate_rmse_m_wmean"]


def pick_rundir(seg_dir):
    """Run-Ordner mit höchster n_mapped (= erfolgreichster Lauf)."""
    best, best_n = None, -1
    for rd in glob.glob(os.path.join(seg_dir, "*/")):
        prof = L.parse_profiling_json(Path(rd))
        n = prof.get("n_mapped", -1) if prof else -1
        if not isinstance(n, int):
            n = -1
        if n > best_n:
            best_n, best = n, rd
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="output/exp_interval1_survey")
    ap.add_argument("--csv", default="docs/results/interval1_survey_full_results.csv")
    ap.add_argument("--summary-csv", default="docs/results/survey_summary.csv",
                    help="EINE Zeile pro Survey-Lauf (Gesamtdauer, gemergte ply, min/max/mean)")
    ap.add_argument("--survey-name", default=None,
                    help="Zeilen-Schlüssel in der Summary-CSV (Default: out-dir-Basename)")
    a = ap.parse_args()
    OUT = a.out_dir
    csv_path = Path(a.csv)
    if csv_path.exists():
        csv_path.unlink()      # frisch schreiben (eine Zeile je Segment)

    segs = sorted(glob.glob(os.path.join(OUT, "s*/")),
                  key=lambda d: int(os.path.basename(d.rstrip("/"))[1:]))
    n_ok = 0
    rows = []                                   # gesammelte Segment-Zeilen für die Aggregat-Zeile
    for seg_dir in segs:
        label = os.path.basename(seg_dir.rstrip("/"))          # sNNNN
        start = int(label[1:])
        rd = pick_rundir(seg_dir)
        if rd is None:
            print(f"  {label}: kein Run-Ordner -- skip"); continue
        rd = Path(rd)
        metrics = L.parse_metrics_json(rd)
        prof = L.parse_profiling_json(rd)
        cfg = Path(metrics.get("config", os.path.join(OUT, f"cfg_{label}.yaml")))
        status_raw = str(metrics.get("status", ""))
        exit_code = 0 if status_raw.startswith("OK") else 137
        status = "OK" if status_raw.startswith("OK") else "FAIL"
        # Laufzeit/Zeitstempel aus metrics + Ordner-mtime
        dur_min = metrics.get("laufzeit_min", "")
        end_ts = int(rd.stat().st_mtime)
        start_ts = int(end_ts - float(dur_min) * 60) if dur_min != "" else end_ts

        row = {
            "timestamp_start": start_ts, "timestamp_end": end_ts,
            "dataset": "interval1", "group": "survey", "variant": label,
            "config_path": str(cfg), "save_dir": os.path.join(OUT, label), "out_dir": str(rd),
            "status": status, "exit_code": exit_code,
            "duration_min": round(float(dur_min), 2) if dur_min != "" else "",
            "peak_ram_gb": metrics.get("ram_gb", ""), "peak_vram_mib": metrics.get("gpu_mib", ""),
            "psnr": metrics.get("psnr", ""), "ssim": metrics.get("ssim", ""),
            "lpips": metrics.get("lpips", ""), "n_metric_frames": metrics.get("n_frames", ""),
            "n_keyframes": prof.get("n_keyframes", ""), "n_mapped": prof.get("n_mapped", ""),
            "n_processed": prof.get("n_processed", ""), "n_dataset_frames": prof.get("n_frames", ""),
        }
        row.update(L.parse_config(cfg)); row.pop("cfg_max_frames", None)
        row.update(L.phase_means_from_profiling(prof))
        row.update(L.parse_fair_metrics(rd))
        # ply-Spalten aus der GPS-unwarpten Segment-ply (die rohe wurde gelöscht)
        gps = Path(OUT) / f"{label}_gps.ply"
        if gps.exists():
            row["ply_mb_final"] = round(gps.stat().st_size / 1e6, 2)
            row["ply_count"] = 1
            row["last_ply_kf"] = prof.get("n_keyframes", "")
        else:
            row.update(L.scan_ply(rd))
        log = L.newest_log(rd)
        if log is not None:
            row["log_path"] = str(log)
        L.append_row(csv_path, row); rows.append(row)
        n_ok += 1
        print(f"  {label}: status={status} psnr={row['psnr']} kf={row['n_keyframes']} "
              f"vram={row['peak_vram_mib']} ate={row.get('ate_rmse_m','')} ply={row.get('ply_mb_final','')}MB")

    # --- Aggregat-Zeile (variant=ALL): Gesamtlauf über alle Segmente ---
    if rows:
        L.append_row(csv_path, aggregate_row(rows, OUT))
    print(f"[measure_survey] {n_ok}/{len(segs)} Segmente + 1 Aggregat -> {csv_path}")

    # --- Survey-Summary: EINE Zeile pro Survey-Lauf (de-dupe by survey_name) ---
    if rows:
        name = a.survey_name or os.path.basename(OUT.rstrip("/"))
        srow = summary_row(rows, OUT, name)
        sp = write_summary(a.summary_csv, srow)
        print(f"[measure_survey] Survey-Zeile '{name}' -> {sp}  "
              f"(dur={srow['duration_total_min']}min ply={srow['ply_mb_merged']}MB "
              f"gauss={srow['n_gaussians_merged']} "
              f"psnr min/max/mean={srow['psnr_min']}/{srow['psnr_max']}/{srow['psnr_mean']})")


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _wmean(rows, col, wcol):
    """Gewichteter Mittelwert von col mit Gewicht wcol (überspringt leere)."""
    num = den = 0.0
    for r in rows:
        v, w = _f(r.get(col)), _f(r.get(wcol))
        if v is not None and w:
            num += v * w; den += w
    return round(num / den, 4) if den else ""


def _sum(rows, col):
    s = 0.0; any_ = False
    for r in rows:
        v = _f(r.get(col))
        if v is not None:
            s += v; any_ = True
    return round(s, 2) if any_ else ""


def _max(rows, col):
    vals = [_f(r.get(col)) for r in rows if _f(r.get(col)) is not None]
    return max(vals) if vals else ""


def _stats(rows, col):
    """(min, max, mean) über die Segmente (ungewichtet, leere übersprungen)."""
    vals = [_f(r.get(col)) for r in rows if _f(r.get(col)) is not None]
    if not vals:
        return "", "", ""
    return round(min(vals), 4), round(max(vals), 4), round(sum(vals) / len(vals), 4)


def _ply_vertex_count(path):
    """Liest 'element vertex N' aus dem PLY-Header, ohne die Daten zu laden."""
    if not os.path.exists(path):
        return ""
    try:
        with open(path, "rb") as f:
            for _ in range(60):
                line = f.readline()
                if line.startswith(b"element vertex"):
                    return int(line.split()[2])
                if line.strip() == b"end_header":
                    break
    except (OSError, ValueError, IndexError):
        pass
    return ""


def aggregate_row(rows, out_dir):
    """Eine Zeile für den GESAMTLAUF: Summen (Laufzeit/KF/Frames), Max (VRAM/RAM),
    gewichtete Mittel (PSNR über n_metric_frames; Timings über n_mapped/n_processed),
    finale Survey-ply-Größe."""
    n_fail = sum(1 for r in rows if r.get("status") != "OK")
    survey = os.path.join(out_dir, "survey_complete.ply")
    ply_mb = round(os.path.getsize(survey) / 1e6, 2) if os.path.exists(survey) else ""
    agg = {c: "" for c in L.COLUMNS}
    agg.update({
        "timestamp_start": min(_f(r["timestamp_start"]) for r in rows),
        "timestamp_end":   max(_f(r["timestamp_end"]) for r in rows),
        "dataset": "interval1", "group": "survey_total", "variant": "ALL",
        "config_path": "scripts/run_interval1_survey.sh", "save_dir": out_dir,
        "out_dir": survey if ply_mb != "" else out_dir,
        "status": f"{len(rows)-n_fail}OK/{n_fail}rc137", "exit_code": "",
        "duration_min": _sum(rows, "duration_min"),     # serielle Gesamt-Compute-Zeit
        "wall_total_s":  _sum(rows, "wall_total_s"),
        "peak_ram_gb":   _max(rows, "peak_ram_gb"), "peak_vram_mib": _max(rows, "peak_vram_mib"),
        "n_keyframes": _sum(rows, "n_keyframes"), "n_mapped": _sum(rows, "n_mapped"),
        "n_processed": _sum(rows, "n_processed"), "n_dataset_frames": _sum(rows, "n_dataset_frames"),
        "mapper_kf_skip": rows[0].get("mapper_kf_skip", ""), "frame_skip": rows[0].get("frame_skip", ""),
        "filter_thresh": rows[0].get("filter_thresh", ""), "selector_kind": rows[0].get("selector_kind", ""),
        "ply_mb_final": ply_mb,                          # GESAMTE finale Survey-ply (gemergt+gecleant)
        "ply_count": len(rows), "last_ply_kf": _sum(rows, "n_keyframes"),
        "psnr":  _wmean(rows, "psnr", "n_metric_frames"),    # KF-/Frame-gewichtetes Mittel
        "ssim":  _wmean(rows, "ssim", "n_metric_frames"),
        "lpips": _wmean(rows, "lpips", "n_metric_frames"),
        "n_metric_frames": _sum(rows, "n_metric_frames"),
        "ate_rmse_m": _wmean(rows, "ate_rmse_m", "n_ate_pairs"),
        "ate_mean_m": _wmean(rows, "ate_mean_m", "n_ate_pairs"),
        "n_ate_pairs": _sum(rows, "n_ate_pairs"), "n_tracked": _sum(rows, "n_tracked"),
        "psnr_ho": _wmean(rows, "psnr_ho", "n_eval_ho"), "ssim_ho": _wmean(rows, "ssim_ho", "n_eval_ho"),
        "lpips_ho": _wmean(rows, "lpips_ho", "n_eval_ho"), "n_eval_ho": _sum(rows, "n_eval_ho"),
        "track_total_mean_ms":        _wmean(rows, "track_total_mean_ms", "n_processed"),
        "track_total_p95_ms":         _max(rows, "track_total_p95_ms"),
        "track_motion_filter_mean_ms":_wmean(rows, "track_motion_filter_mean_ms", "n_processed"),
        "track_frontend_ba_mean_ms":  _wmean(rows, "track_frontend_ba_mean_ms", "n_processed"),
        "map_total_mean_ms":          _wmean(rows, "map_total_mean_ms", "n_mapped"),
        "map_total_p95_ms":           _max(rows, "map_total_p95_ms"),
        "map_train_loop_mean_ms":     _wmean(rows, "map_train_loop_mean_ms", "n_mapped"),
        "crash_reason": f"{n_fail}/{len(rows)} Segmente rc=137 (Teil-ply genutzt)" if n_fail else "",
    })
    return agg


def summary_row(rows, out_dir, name):
    """EINE Zeile für den ganzen Survey: Gesamtdauer, gemergte ply (MB+Gauss-Count),
    min/max/mean der Metriken über die Segmente (+ frame-/pair-gewichtetes Headline-Mittel)."""
    n_fail = sum(1 for r in rows if r.get("status") != "OK")
    survey = os.path.join(out_dir, "survey_complete.ply")
    ply_mb = round(os.path.getsize(survey) / 1e6, 2) if os.path.exists(survey) else ""
    row = {c: "" for c in SUMMARY_COLUMNS}
    row.update({
        "survey_name": name,
        "timestamp": int(max(_f(r["timestamp_end"]) for r in rows)),
        "dataset": rows[0].get("dataset", "interval1"),
        "selector_kind": rows[0].get("selector_kind", ""),
        "n_segments": len(rows), "n_ok": len(rows) - n_fail, "n_fail": n_fail,
        "duration_total_min": _sum(rows, "duration_min"),     # serielle Gesamt-Compute-Zeit
        "wall_total_s": _sum(rows, "wall_total_s"),
        "ply_mb_merged": ply_mb, "n_gaussians_merged": _ply_vertex_count(survey),
        "n_keyframes_total": _sum(rows, "n_keyframes"),
        "n_mapped_total": _sum(rows, "n_mapped"),
        "n_dataset_frames_total": _sum(rows, "n_dataset_frames"),
        "psnr_wmean": _wmean(rows, "psnr", "n_metric_frames"),
        "ate_rmse_m_wmean": _wmean(rows, "ate_rmse_m", "n_ate_pairs"),
    })
    for m in SUMMARY_METRICS:
        mn, mx, mean = _stats(rows, m)
        row[f"{m}_min"], row[f"{m}_max"], row[f"{m}_mean"] = mn, mx, mean
    return row


def write_summary(summary_csv, row):
    """Schreibt die Survey-Zeile in die Summary-CSV; ersetzt eine vorhandene Zeile
    mit gleichem survey_name (Re-Run = Update, kein Duplikat)."""
    path = Path(summary_csv)
    existing = []
    if path.exists():
        with open(path, newline="") as f:
            existing = [r for r in csv.DictReader(f) if r.get("survey_name") != row["survey_name"]]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in existing:
            w.writerow(r)
        w.writerow(row)
    return path


if __name__ == "__main__":
    main()
