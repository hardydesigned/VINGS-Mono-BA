#!/usr/bin/env python3
"""
Append one CSV row describing a single experiment run.

Called by scripts/run_sweep.sh after each `run_experiment.py` invocation.
It is robust to crashed/aborted runs — pulls whatever it can find from
metrics.json, profiling.json, the ply/ dir, and the run log.

Usage (all args required):
    log_sweep_row.py \
        --csv     output/sweep_results.csv \
        --dataset amtown03 \
        --group   mapskip \
        --variant mapskip_5 \
        --config  configs/local/amtown03/exp/mapskip/amtown03_full_mapskip_5.yaml \
        --save-dir output/exp_amtown03_full \
        --known-before /tmp/before_dirs.txt \
        --start-ts 1717000000 \
        --end-ts   1717003600 \
        --exit-code 0 \
        --status OK
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
import re
import sys
from pathlib import Path

COLUMNS = [
    "timestamp_start", "timestamp_end",
    "dataset", "group", "variant",
    "config_path", "save_dir", "out_dir",
    "status", "exit_code", "duration_min",
    "peak_ram_gb", "peak_vram_mib",
    "n_keyframes", "n_mapped", "n_processed", "n_dataset_frames",
    "mapper_kf_skip", "frame_skip", "filter_thresh", "selector_kind",
    "ply_mb_final", "ply_count", "last_ply_kf",
    "psnr", "ssim", "lpips", "n_metric_frames",
    # Fair, selection-independent eval (fair_metrics.json): Sim(3)-ATE +
    # held-out novel-view PSNR/SSIM/LPIPS on a FIXED frame set vs GT.
    "ate_rmse_m", "ate_mean_m", "n_ate_pairs", "n_tracked",
    "psnr_ho", "ssim_ho", "lpips_ho", "n_eval_ho",
    # Phase means / p95 from profiling.json[records] (all milliseconds).
    "wall_total_s",
    "track_total_mean_ms", "track_total_p95_ms",
    "track_motion_filter_mean_ms", "track_frontend_ba_mean_ms",
    "map_total_mean_ms", "map_total_p95_ms",
    "map_train_loop_mean_ms",
    "frame_select_mean_ms",
    "log_path", "crash_reason",
]


PHASE_FIELDS: dict[str, str] = {
    # CSV column                       phase name in profiling.json[records]
    "track_total_mean_ms":              "track.total",
    "track_total_p95_ms":               "track.total",
    "track_motion_filter_mean_ms":      "track.motion_filter",
    "track_frontend_ba_mean_ms":        "track.frontend_ba",
    "map_total_mean_ms":                "map.total",
    "map_total_p95_ms":                 "map.total",
    "map_train_loop_mean_ms":           "map.train_loop",
    "frame_select_mean_ms":             "frame_select",
}


def _phase_stats(records: dict, phase: str, stat: str) -> str:
    """Return mean (or p95) of phase samples in milliseconds, blank if missing."""
    vals = records.get(phase) or []
    if not vals:
        return ""
    if stat == "mean":
        return round(1000.0 * sum(vals) / len(vals), 2)
    if stat == "p95":
        srt = sorted(vals)
        idx = max(0, int(0.95 * len(srt)) - 1)
        return round(1000.0 * srt[idx], 2)
    return ""


def phase_means_from_profiling(prof: dict) -> dict:
    """Extract per-phase mean/p95 (ms) from profiling.json into CSV columns."""
    out: dict = {}
    records = (prof or {}).get("records") or {}
    for col, phase in PHASE_FIELDS.items():
        stat = "p95" if col.endswith("_p95_ms") else "mean"
        out[col] = _phase_stats(records, phase, stat)
    if "wall_total_s" not in out:
        wt = (prof or {}).get("wall_total_s")
        out["wall_total_s"] = round(wt, 1) if isinstance(wt, (int, float)) else ""
    return out


def find_new_outdir(save_dir: Path, known_before: set[str]) -> Path | None:
    if not save_dir.exists():
        return None
    after = sorted(d for d in save_dir.iterdir() if d.is_dir())
    for d in after:
        if d.name not in known_before:
            return d
    return after[-1] if after else None


def parse_metrics_json(out_dir: Path) -> dict:
    p = out_dir / "metrics.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def parse_profiling_json(out_dir: Path) -> dict:
    p = out_dir / "profiling.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def parse_fair_metrics(out_dir: Path) -> dict:
    """Read fair_metrics.json (selection-independent eval); blank if absent."""
    p = out_dir / "fair_metrics.json"
    if not p.exists():
        return {}
    try:
        d = json.loads(p.read_text())
    except Exception:
        return {}
    keys = ["ate_rmse_m", "ate_mean_m", "n_ate_pairs", "n_tracked",
            "psnr_ho", "ssim_ho", "lpips_ho", "n_eval_ho"]
    return {k: (d.get(k) if d.get(k) is not None else "") for k in keys}


def parse_config(config_path: Path) -> dict:
    """Extract a few hot fields from the yaml without yaml dep."""
    out = {
        "mapper_kf_skip": "", "frame_skip": "",
        "filter_thresh": "", "selector_kind": "",
        "cfg_max_frames": "",
    }
    if not config_path.exists():
        return out
    try:
        import yaml
        d = yaml.safe_load(config_path.read_text()) or {}
        out["mapper_kf_skip"] = d.get("mapper_kf_skip", 1)
        out["frame_skip"] = d.get("frame_skip", 1)
        out["filter_thresh"] = (d.get("frontend") or {}).get("filter_thresh", "")
        out["selector_kind"] = (d.get("frame_selector") or {}).get("kind", "")
        out["cfg_max_frames"] = (d.get("dataset") or {}).get("max_frames", "")
    except Exception:
        pass
    return out


def scan_ply(out_dir: Path) -> dict:
    """Find ply files even if the run crashed mid-mapping."""
    ply_dir = out_dir / "ply"
    if not ply_dir.exists():
        return {"ply_count": 0, "ply_mb_final": "", "last_ply_kf": ""}
    plys = list(ply_dir.glob("*.ply"))
    if not plys:
        return {"ply_count": 0, "ply_mb_final": "", "last_ply_kf": ""}
    # Highest-size ply (canonical = the run_experiment cleanup pattern), or
    # largest by mtime if cleanup did not run.
    biggest = max(plys, key=lambda p: p.stat().st_size)
    # Try to extract a frame/keyframe index from filename: typical patterns are
    # idx=NNN_2dgs.ply (current VINGS-Mono), kfNNN.ply, *_kf=N*.ply, NNNNNN.ply.
    last_kf = ""
    rx = re.compile(r"(?:idx[_=]|kf[_=]?|frame[_=]?|^)(\d{1,7})", re.IGNORECASE)
    best_n = -1
    for p in plys:
        m = rx.search(p.stem)
        if m:
            n = int(m.group(1))
            if n > best_n:
                best_n = n
                last_kf = str(n)
    return {
        "ply_count": len(plys),
        "ply_mb_final": round(biggest.stat().st_size / 1e6, 2),
        "last_ply_kf": last_kf,
    }


def newest_log(out_dir: Path) -> Path | None:
    logs = list(out_dir.glob("run_*.log"))
    if not logs:
        return None
    return max(logs, key=lambda p: p.stat().st_mtime)


# Ungeankert, damit tqdm-ANSI-Escapes (z.B. "67.64it/s][A...[   59] kf=N")
# nicht ganze Zeilen verschlucken. finditer findet alle Matches pro Zeile.
_KF_LINE_RX = re.compile(rb"\[\s*(\d+)\]\s+kf=([YNS])")


def scrape_log_on_crash(out_dir: Path) -> dict:
    """Salvage progress from the run log when profiling.json is missing.

    Two paths:
    1) "Profiling Summary (KFs mapped processed)" — printed only on clean exit.
    2) Per-frame `[ NNNNN] kf=Y/S/N ...` lines — printed for every processed
       frame, so they survive any crash that happens after the first frame.
       We count them across the WHOLE log (not just tail), because a crash at
       frame 9000 would push the early lines past any tail window.
    """
    out: dict = {}
    log = newest_log(out_dir)
    if log is None:
        return out
    try:
        sz = log.stat().st_size
        # Always read the tail for OOM markers + Profiling Summary detection.
        with open(log, "rb") as f:
            f.seek(max(0, sz - 256 * 1024))
            tail = f.read().decode("utf-8", errors="replace")
    except Exception:
        return out

    # Path 1: clean-exit Profiling Summary
    m = re.search(r"Profiling Summary \((\d+) KFs, (\d+) mapped / (\d+) processed", tail)
    if m:
        out["n_keyframes"] = int(m.group(1))
        out["n_mapped"]    = int(m.group(2))
        out["n_processed"] = int(m.group(3))
    else:
        # Path 2: count per-frame kf-lines across the whole log.
        n_proc = 0
        n_kf = 0
        n_mapped = 0
        last_idx = -1
        try:
            with open(log, "rb") as f:
                for raw in f:
                    for m2 in _KF_LINE_RX.finditer(raw):
                        n_proc += 1
                        idx_i = int(m2.group(1))
                        if idx_i > last_idx:
                            last_idx = idx_i
                        flag = m2.group(2)
                        if flag in (b"Y", b"S"):
                            n_kf += 1
                        if flag == b"Y":
                            n_mapped += 1
        except Exception:
            n_proc = 0
        if n_proc > 0:
            out["n_processed"] = n_proc
            out["n_keyframes"] = n_kf
            out["n_mapped"]    = n_mapped
            out["last_idx"]    = last_idx

    # Crash-reason markers (tail-only is fine — exceptions land at the end).
    if re.search(r"CUDA out of memory|OutOfMemoryError|RuntimeError.*memory", tail):
        out["crash_reason"] = "OOM"
    elif re.search(r"Killed\b|signal 9", tail):
        out["crash_reason"] = "SIGKILL (likely VRAM watchdog)"
    return out


def append_row(csv_path: Path, row: dict) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    new = not csv_path.exists()
    # File lock to be safe if two runners ever race
    with open(csv_path, "a", newline="") as f:
        try:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        except OSError:
            pass
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in COLUMNS})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--group", required=True)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--save-dir", required=True,
                    help="parent dir into which run_experiment.py spawned an output sub-dir")
    ap.add_argument("--known-before", required=True,
                    help="file listing dir names that existed in save-dir BEFORE the run")
    ap.add_argument("--start-ts", type=int, required=True)
    ap.add_argument("--end-ts", type=int, required=True)
    ap.add_argument("--exit-code", type=int, required=True)
    ap.add_argument("--status", required=True,
                    help="OK | FAIL | OOM | TIMEOUT | SKIPPED")
    args = ap.parse_args()

    save_dir = Path(args.save_dir)
    config = Path(args.config)
    known_before = set()
    kb_path = Path(args.known_before)
    if kb_path.exists():
        known_before = {l.strip() for l in kb_path.read_text().splitlines() if l.strip()}

    out_dir = find_new_outdir(save_dir, known_before)

    row = {
        "timestamp_start": args.start_ts,
        "timestamp_end":   args.end_ts,
        "dataset": args.dataset,
        "group":   args.group,
        "variant": args.variant,
        "config_path": str(config),
        "save_dir": str(save_dir),
        "out_dir": str(out_dir) if out_dir else "",
        "status": args.status,
        "exit_code": args.exit_code,
        "duration_min": round((args.end_ts - args.start_ts) / 60.0, 2),
    }
    row.update(parse_config(config))

    if out_dir is None:
        row["crash_reason"] = "no output dir created"
        append_row(Path(args.csv), row)
        print(f"[log_sweep_row] {args.variant}: NO OUTPUT DIR")
        return

    metrics = parse_metrics_json(out_dir)
    prof    = parse_profiling_json(out_dir)
    ply     = scan_ply(out_dir)
    salvage = scrape_log_on_crash(out_dir) if not prof else {}
    phases  = phase_means_from_profiling(prof)

    row.update({
        "peak_ram_gb":      metrics.get("ram_gb", ""),
        "peak_vram_mib":    metrics.get("gpu_mib", ""),
        "psnr":             metrics.get("psnr", ""),
        "ssim":             metrics.get("ssim", ""),
        "lpips":            metrics.get("lpips", ""),
        "n_metric_frames":  metrics.get("n_frames", ""),
        "n_keyframes":      prof.get("n_keyframes", salvage.get("n_keyframes", "")),
        "n_mapped":         prof.get("n_mapped",    salvage.get("n_mapped", "")),
        "n_processed":      prof.get("n_processed", salvage.get("n_processed", "")),
        "n_dataset_frames": prof.get("n_frames", row.get("cfg_max_frames", "")),
    })
    row.pop("cfg_max_frames", None)  # internal field, do not emit
    row.update(ply)
    row.update(phases)
    row.update(parse_fair_metrics(out_dir))

    log = newest_log(out_dir)
    if log is not None:
        row["log_path"] = str(log)

    if "crash_reason" in salvage and not row.get("crash_reason"):
        row["crash_reason"] = salvage["crash_reason"]

    # If the runner said FAIL but PSNR exists (rare: cleanup crashed after metrics),
    # don't overwrite the status — trust the runner's decision.
    append_row(Path(args.csv), row)
    print(f"[log_sweep_row] {args.variant}: status={row['status']} "
          f"psnr={row.get('psnr','')} kf={row.get('n_keyframes','')} "
          f"ply={row.get('ply_count','')} dur={row['duration_min']}min")


if __name__ == "__main__":
    sys.exit(main())
