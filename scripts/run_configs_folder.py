#!/usr/bin/env python3
"""
Run *every* config in a folder via run_experiment.py and log one CSV row per
run, written INCREMENTALLY (flushed after each run) so a crash mid-sweep never
loses the rows already collected.

This is a thin wrapper: all the robust parsing (metrics.json, profiling.json,
fair_metrics.json, ply scan, crash-log salvage) is reused verbatim from
scripts/log_sweep_row.py — this file only owns the loop + a fixed column order.

Usage:
    python scripts/run_configs_folder.py configs/local/validated_selectors/vista
    python scripts/run_configs_folder.py <folder> --csv output/my_results.csv \
        --timeout 3600 --pattern '*.yaml'

The CSV lands in the run's output dir by default (output/<folder-name>_results.csv).
"""

from __future__ import annotations

import argparse
import csv
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import log_sweep_row as L  # reuse all parsers  # noqa: E402

# selector_name first, then exactly the columns requested.
COLUMNS = [
    "selector_name",
    "dataset", "group", "variant", "status", "exit_code", "duration_min",
    "peak_ram_gb", "peak_vram_mib",
    "n_keyframes", "n_mapped", "n_processed", "n_dataset_frames",
    "mapper_kf_skip", "frame_skip", "filter_thresh", "selector_kind",
    "ply_mb_final", "ply_count", "last_ply_kf",
    "psnr", "ssim", "lpips", "n_metric_frames",
    "ate_rmse_m", "ate_mean_m", "n_ate_pairs", "n_tracked",
    "psnr_ho", "ssim_ho", "lpips_ho", "n_eval_ho",
    "wall_total_s",
    "track_total_mean_ms", "track_total_p95_ms",
    "track_motion_filter_mean_ms", "track_frontend_ba_mean_ms",
    "map_total_mean_ms", "map_total_p95_ms", "map_train_loop_mean_ms",
    "frame_select_mean_ms",
    "log_path", "crash_reason",
]


def map_status(rc: int) -> str:
    return {0: "OK", 124: "TIMEOUT", 137: "OOM", 139: "FAIL"}.get(rc, "FAIL")


def append_row(csv_path: Path, row: dict) -> None:
    new = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in COLUMNS})
        f.flush()


def build_row(cfg_path: Path, save_dir: Path, known_before: set[str],
              t0: int, t1: int, rc: int, status: str, dataset: str) -> dict:
    """Assemble a row using log_sweep_row's parsers (crash-robust)."""
    row = {
        "selector_name": "",  # filled below
        "dataset": dataset,
        "group": cfg_path.parent.name,
        "variant": cfg_path.stem,
        "status": status,
        "exit_code": rc,
        "duration_min": round((t1 - t0) / 60.0, 2),
    }
    row.update(L.parse_config(cfg_path))

    out_dir = L.find_new_outdir(save_dir, known_before)
    if out_dir is None:
        row["crash_reason"] = "no output dir created"
        row["selector_name"] = row.get("selector_kind") or cfg_path.stem
        return row

    metrics = L.parse_metrics_json(out_dir)
    prof = L.parse_profiling_json(out_dir)
    salvage = L.scrape_log_on_crash(out_dir) if not prof else {}

    row.update({
        "peak_ram_gb": metrics.get("ram_gb", ""),
        "peak_vram_mib": metrics.get("gpu_mib", ""),
        "psnr": metrics.get("psnr", ""),
        "ssim": metrics.get("ssim", ""),
        "lpips": metrics.get("lpips", ""),
        "n_metric_frames": metrics.get("n_frames", ""),
        "n_keyframes": prof.get("n_keyframes", salvage.get("n_keyframes", "")),
        "n_mapped": prof.get("n_mapped", salvage.get("n_mapped", "")),
        "n_processed": prof.get("n_processed", salvage.get("n_processed", "")),
        "n_dataset_frames": prof.get("n_frames", row.get("cfg_max_frames", "")),
    })
    row.pop("cfg_max_frames", None)
    row.update(L.scan_ply(out_dir))
    row.update(L.phase_means_from_profiling(prof))
    row.update(L.parse_fair_metrics(out_dir))

    log = L.newest_log(out_dir)
    if log is not None:
        row["log_path"] = str(log)
    if "crash_reason" in salvage and not row.get("crash_reason"):
        row["crash_reason"] = salvage["crash_reason"]

    row["selector_name"] = row.get("selector_kind") or cfg_path.stem
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", help="folder of .yaml configs to run")
    ap.add_argument("--csv", default=None,
                    help="output CSV (default: output/<folder-name>_results.csv)")
    ap.add_argument("--pattern", default="*.yaml", help="glob for configs")
    ap.add_argument("--recursive", action="store_true", help="recurse into subfolders")
    ap.add_argument("--timeout", type=int, default=21600, help="per-run wall-clock cap (s)")
    ap.add_argument("--keep", action="store_true",
                    help="keep the run output dir (default: delete it after the CSV row is written)")
    args = ap.parse_args()

    folder = Path(args.folder)
    if not folder.is_dir():
        print(f"[run_folder] not a directory: {folder}", file=sys.stderr)
        return 2

    globber = folder.rglob if args.recursive else folder.glob
    configs = sorted(globber(args.pattern))
    if not configs:
        print(f"[run_folder] no configs matching {args.pattern} in {folder}", file=sys.stderr)
        return 2

    csv_path = Path(args.csv) if args.csv \
        else REPO / "output" / f"{folder.name}_results.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[run_folder] {len(configs)} configs → {csv_path}")
    ok = fail = 0
    for i, cfg in enumerate(configs, 1):
        try:
            d = yaml.safe_load(cfg.read_text()) or {}
        except Exception as e:
            print(f"[run_folder {i}/{len(configs)}] SKIP {cfg} (bad yaml: {e})")
            continue
        save_dir = Path((d.get("output") or {}).get("save_dir", "") or ".")
        dataset = Path((d.get("dataset") or {}).get("root", "") or "").name or "unknown"
        save_dir.mkdir(parents=True, exist_ok=True)
        known_before = {x.name for x in save_dir.iterdir() if x.is_dir()}

        print(f"\n{'='*60}\n [{i}/{len(configs)}] {cfg.name}\n{'='*60}")
        t0 = int(time.time())
        rc = _run(cfg, args.timeout)
        t1 = int(time.time())
        status = map_status(rc)
        print(f"[run_folder] {cfg.stem} rc={rc} → {status} ({(t1-t0)//60}min)")

        row = build_row(cfg, save_dir, known_before, t0, t1, rc, status, dataset)
        append_row(csv_path, row)
        print(f"[run_folder] logged: status={row['status']} psnr={row.get('psnr','')} "
              f"kf={row.get('n_keyframes','')} ply={row.get('ply_count','')}")

        # Free disk: drop the output subdir(s) THIS run created (row is safely in
        # the CSV now). Only dirs absent from the before-snapshot are touched, so
        # earlier runs' artifacts are never removed. The CSV lives in output/, not
        # inside these dirs, so it survives.
        if not args.keep and save_dir.exists():
            for d in sorted(save_dir.iterdir()):
                if d.is_dir() and d.name not in known_before:
                    shutil.rmtree(d, ignore_errors=True)
                    print(f"[run_folder] removed {d}")

        ok += status == "OK"
        fail += status != "OK"

    print(f"\n[run_folder] done — ok={ok} fail={fail} csv={csv_path}")
    return 0


def _run(cfg: Path, timeout: int) -> int:
    """Run run_experiment.py, optionally with a wall-clock cap (TERM then KILL)."""
    cmd = ["python", str(REPO / "scripts" / "run_experiment.py"), str(cfg)]
    if timeout <= 0:
        return subprocess.call(cmd)
    proc = subprocess.Popen(cmd, start_new_session=True)
    try:
        return proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        import os
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
        return 124


if __name__ == "__main__":
    sys.exit(main())
