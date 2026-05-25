#!/usr/bin/env python3
"""Aggregiert profiling.json aus allen Output-Runs.

Liest jede profiling.json unter output/**/ und gibt pro Run und Phase
(track.total, map.total) die Per-Call-Statistik aus: count, min, max,
median, mean, p95, total. Tracking-Werte sind pro processed Frame,
Mapping-Werte sind pro Keyframe.
"""

import argparse
import json
import statistics
from pathlib import Path

PHASES = [
    ("track.total", "Tracking pro Frame"),
    ("track.frontend_ba", "  └─ Frontend BA"),
    ("map.total", "Mapping pro KF"),
    ("map.train_loop", "  └─ Train Loop"),
]


def percentile(values, q):
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * q
    f, c = int(k), min(int(k) + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)


def stats(values):
    if not values:
        return None
    return {
        "n": len(values),
        "min_ms": min(values) * 1000,
        "max_ms": max(values) * 1000,
        "med_ms": statistics.median(values) * 1000,
        "mean_ms": statistics.mean(values) * 1000,
        "p95_ms": percentile(values, 0.95) * 1000,
        "total_s": sum(values),
    }


def fmt_row(label, s):
    if s is None:
        return f"  {label:<26} -- keine Daten --"
    return (
        f"  {label:<26} "
        f"n={s['n']:>4}  "
        f"min={s['min_ms']:>7.1f}  "
        f"med={s['med_ms']:>7.1f}  "
        f"mean={s['mean_ms']:>7.1f}  "
        f"p95={s['p95_ms']:>7.1f}  "
        f"max={s['max_ms']:>7.1f}  "
        f"total={s['total_s']:>6.1f}s"
    )


def analyze_run(profiling_path):
    data = json.load(open(profiling_path))
    records = data.get("records", {})
    return {
        "path": profiling_path,
        "wall_total_s": data.get("wall_total_s"),
        "n_frames": data.get("n_frames"),
        "n_processed": data.get("n_processed"),
        "n_keyframes": data.get("n_keyframes"),
        "frame_skip": data.get("frame_skip"),
        "phases": {p: stats(records.get(p, [])) for p, _ in PHASES},
    }


def print_run(run):
    name = run["path"].parent.name
    print("=" * 110)
    print(f"Run: {name}")
    print(
        f"  wall={run['wall_total_s']:.1f}s  "
        f"frames={run['n_frames']}  processed={run['n_processed']}  "
        f"KFs={run['n_keyframes']}  frame_skip={run['frame_skip']}"
    )
    print(f"  {'Phase':<26} {'n':>4}  {'min':>11}  {'med':>11}  {'mean':>11}  "
          f"{'p95':>11}  {'max':>11}  {'total':>13}")
    for phase, label in PHASES:
        print(fmt_row(label, run["phases"][phase]))
    print()


def print_summary_table(runs):
    print("=" * 110)
    print("Zusammenfassung (Mean pro Call, ms):")
    print("=" * 110)
    header = (
        f"{'Run':<55} {'skip':>5} {'KFs':>5} "
        f"{'track.mean':>11} {'track.med':>10} {'map.mean':>10} {'map.med':>10}"
    )
    print(header)
    print("-" * len(header))
    for run in runs:
        name = run["path"].parent.name
        t = run["phases"]["track.total"]
        m = run["phases"]["map.total"]
        print(
            f"{name:<55} "
            f"{str(run['frame_skip']):>5} "
            f"{str(run['n_keyframes']):>5} "
            f"{(t['mean_ms'] if t else 0):>11.1f} "
            f"{(t['med_ms'] if t else 0):>10.1f} "
            f"{(m['mean_ms'] if m else 0):>10.1f} "
            f"{(m['med_ms'] if m else 0):>10.1f}"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "output",
    )
    args = ap.parse_args()

    paths = sorted(args.output_root.rglob("profiling.json"))
    if not paths:
        print(f"Keine profiling.json unter {args.output_root}")
        return

    runs = [analyze_run(p) for p in paths]
    runs.sort(key=lambda r: r["path"].parent.name)

    for run in runs:
        print_run(run)
    print_summary_table(runs)


if __name__ == "__main__":
    main()
