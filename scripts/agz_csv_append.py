#!/usr/bin/env python3
"""Append eine Zeile an die AGZ-Sweep summary.csv.

Liest den auf USB rsync-ten Run-Ordner zu einer Config und sammelt:
  - last_input_idx       letzte [NNNNN]-Zeile aus run_*.log (Tracker-Frontend-Index)
  - last_mapped_frame    hoechstes FrameId aus rgbdnua/FrameId=*.png
  - n_keyframes          Zeilen in keyframelist.txt
  - n_rgbdnua_frames     rgbdnua-PNGs nach Cleanup (3 wenn voll, sonst <3)
  - psnr/ssim/lpips/laufzeit_min/ram_gb/gpu_mib/ply_mb aus metrics.json
  - ply_checkpoint_count Anzahl PLY-CKPT-Eintraege im run_*.log (Hinweis auf Checkpoint-Phasen
                         vor dem Cleanup, der nur den groessten/letzten PLY behaelt)

Wird vom scripts/run_agz_full_sweep.sh nach jedem Run aufgerufen, BEVOR
das lokale output/<name>/ geloescht wird. Die Quelle ist trotzdem der USB-
Sync-Ordner -- run_experiment.py loescht lokale Daten via DELETE_LOCAL_AFTER_SYNC=1
schon vorher, und die USB-Kopie ist vollstaendig.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from datetime import datetime
from pathlib import Path


IDX_RE = re.compile(r"\[\s*(\d+)\]")
FRAMEID_RE = re.compile(r"FrameId=(\d+)\.png$")
PLY_CKPT_RE = re.compile(r"\[PLY-CKPT\]")


def find_usb_run_dir(usb_root: Path, cfg_name: str) -> Path | None:
    """run_experiment.py rsynct nach $USB_SYNC_DIR/<cfg-basename>/<timestamp>/.
    Wir nehmen den juengsten timestamp-Ordner.
    """
    parent = usb_root / cfg_name
    if not parent.exists():
        return None
    subdirs = sorted([d for d in parent.iterdir() if d.is_dir()],
                     key=lambda d: d.stat().st_mtime)
    return subdirs[-1] if subdirs else None


def parse_last_input_idx(log_path: Path) -> int | None:
    """Letzte [NNNNN]-Zeile aus dem Lauf-Log. tail-aehnlich, Datei kann gross sein."""
    if not log_path.exists():
        return None
    last = None
    try:
        with log_path.open("rb") as f:
            f.seek(0, 2)
            size = f.tell()
            chunk = 65536
            buf = b""
            while size > 0 and last is None:
                read = min(chunk, size)
                size -= read
                f.seek(size)
                buf = f.read(read) + buf
                lines = buf.splitlines()
                for line in reversed(lines):
                    s = line.decode("utf-8", errors="replace")
                    m = IDX_RE.search(s)
                    if m:
                        last = int(m.group(1))
                        break
                buf = lines[0] if lines else b""
    except Exception:
        return None
    return last


def parse_last_mapped_frame(rgbdnua_dir: Path) -> int | None:
    if not rgbdnua_dir.exists():
        return None
    best = -1
    for p in rgbdnua_dir.iterdir():
        m = FRAMEID_RE.search(p.name)
        if m:
            best = max(best, int(m.group(1)))
    return best if best >= 0 else None


def count_keyframes(kf_path: Path) -> int | None:
    if not kf_path.exists():
        return None
    try:
        with kf_path.open() as f:
            return sum(1 for _ in f)
    except Exception:
        return None


def count_rgbdnua(rgbdnua_dir: Path) -> int | None:
    if not rgbdnua_dir.exists():
        return None
    return sum(1 for p in rgbdnua_dir.iterdir() if FRAMEID_RE.search(p.name))


def count_ply_checkpoints(log_path: Path) -> int | None:
    if not log_path.exists():
        return None
    try:
        n = 0
        with log_path.open("rb") as f:
            for line in f:
                if b"[PLY-CKPT]" in line:
                    n += 1
        return n
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Pfad zur ausgefuehrten YAML")
    ap.add_argument("--status", required=True, help="OK / FAIL_rcN / MISSING")
    ap.add_argument("--usb-root", required=True, help="USB_SYNC_DIR")
    ap.add_argument("--csv", required=True, help="Pfad zur summary.csv")
    ap.add_argument("--wall-sec", required=True, type=int,
                    help="Wall-Time des subprocess (Fallback wenn metrics.json fehlt)")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    cfg_name = cfg_path.stem
    usb_root = Path(args.usb_root)

    run_dir = find_usb_run_dir(usb_root, cfg_name)

    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "config": cfg_name,
        "status": args.status,
        "last_input_idx": "",
        "last_mapped_frame": "",
        "n_keyframes": "",
        "n_rgbdnua_frames": "",
        "psnr": "",
        "ssim": "",
        "lpips": "",
        "laufzeit_min": round(args.wall_sec / 60.0, 2),
        "ram_gb": "",
        "gpu_mib": "",
        "ply_mb": "",
        "ply_checkpoint_count": "",
        "usb_run_dir": str(run_dir) if run_dir else "",
    }

    if run_dir is None:
        print(f"[csv_append] kein USB-Run-Dir gefunden fuer {cfg_name}; CSV-Zeile minimal.",
              file=sys.stderr)
    else:
        metrics_path = run_dir / "metrics.json"
        if metrics_path.exists():
            try:
                m = json.loads(metrics_path.read_text())
                for k_src, k_dst in [
                    ("psnr", "psnr"), ("ssim", "ssim"), ("lpips", "lpips"),
                    ("laufzeit_min", "laufzeit_min"),
                    ("ram_gb", "ram_gb"), ("gpu_mib", "gpu_mib"),
                    ("ply_mb", "ply_mb"),
                ]:
                    v = m.get(k_src)
                    if v is not None:
                        row[k_dst] = v
            except Exception as e:
                print(f"[csv_append] metrics.json parse error: {e}", file=sys.stderr)

        # Run-Log: ein *.log liegt im Run-Ordner (run_<timestamp>.log)
        logs = sorted(run_dir.glob("run_*.log"))
        log_path = logs[-1] if logs else None
        if log_path:
            row["last_input_idx"] = parse_last_input_idx(log_path) or ""
            row["ply_checkpoint_count"] = count_ply_checkpoints(log_path) or 0

        rgbdnua = run_dir / "rgbdnua"
        row["last_mapped_frame"] = parse_last_mapped_frame(rgbdnua) or ""
        row["n_rgbdnua_frames"] = count_rgbdnua(rgbdnua) or 0

        kf = run_dir / "keyframelist.txt"
        row["n_keyframes"] = count_keyframes(kf) or 0

    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with csv_path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)

    short = {k: row[k] for k in ("config", "status", "last_input_idx",
                                 "last_mapped_frame", "n_keyframes",
                                 "psnr", "laufzeit_min")}
    print(f"[csv_append] appended: {short}")


if __name__ == "__main__":
    main()
