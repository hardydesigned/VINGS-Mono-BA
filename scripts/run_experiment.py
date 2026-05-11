#!/usr/bin/env python3
"""
Generischer VINGS-Mono Single-Run mit Metrik-Logging und Cleanup.

Logged : PSNR, SSIM, LPIPS, RAM_GB, Laufzeit_min, PLY_MB
Cleanup: löscht droid_c2w/, behält 3 rgbdnua-Bilder (Anfang/Mitte/Ende),
         behält 1 PLY (größte), behält config.yaml + keyframelist.txt

Nutzung:
    conda run -n vings python scripts/run_experiment.py <config> [--prefix PREFIX]
"""

import argparse
import json
import logging
import os
import select
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np
import yaml
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


# ── Logging ────────────────────────────────────────────────────────────────────

def setup_logging(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("vings_experiment")
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s  %(levelname)-7s  %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    # Konsole
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    # Datei (alles inkl. DEBUG)
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


def stream_subprocess(proc, log_path: Path, logger: logging.Logger):
    """Streamt stdout+stderr des Subprozesses gleichzeitig auf Konsole und in log_path."""
    with open(log_path, "ab") as lf:
        while True:
            line = proc.stdout.readline()
            if not line and proc.poll() is not None:
                break
            if line:
                text = line.decode("utf-8", errors="replace").rstrip()
                print(text, flush=True)
                lf.write(line)
                lf.flush()
        # Restliche Ausgabe
        for line in proc.stdout:
            text = line.decode("utf-8", errors="replace").rstrip()
            print(text, flush=True)
            lf.write(line)
    return proc.wait()


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.full_load(f)


def find_new_outdir(parent: Path, known: set) -> Optional[Path]:
    for d in sorted(parent.iterdir()):
        if d.is_dir() and d not in known:
            return d
    return None


def poll_gpu(stop_event: threading.Event, interval: float = 1.0) -> list:
    readings = []
    while not stop_event.is_set():
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                stderr=subprocess.DEVNULL,
            )
            readings.append(int(out.strip().split()[0]))
        except Exception:
            pass
        time.sleep(interval)
    return readings


def compute_metrics(rgbdnua_dir: Path, logger: logging.Logger) -> dict:
    """
    rgbdnua-Grid: 2 Zeilen × 4 Spalten.
    Zeile 0, Spalte 0 = GT RGB  |  Zeile 1, Spalte 0 = Pred RGB
    """
    import torch
    import lpips as lpips_lib

    frames = sorted(rgbdnua_dir.glob("FrameId=*.png"))
    if not frames:
        logger.warning("Keine FrameId=*.png Bilder in %s", rgbdnua_dir)
        return {"psnr": None, "ssim": None, "lpips": None, "n_frames": 0}

    logger.info("Berechne Metriken über %d Frames …", len(frames))
    net = lpips_lib.LPIPS(net="alex", verbose=False)
    psnrs, ssims, lpipss = [], [], []

    for fp in frames:
        arr = np.array(Image.open(fp).convert("RGB"))
        H2, W4 = arr.shape[:2]
        H, W = H2 // 2, W4 // 4
        gt   = arr[0:H,  0:W]
        pred = arr[H:H2, 0:W]

        psnrs.append(peak_signal_noise_ratio(gt, pred, data_range=255))
        ssims.append(structural_similarity(gt, pred, channel_axis=2, data_range=255))

        def to_t(img):
            return torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0

        with torch.no_grad():
            lpipss.append(float(net(to_t(pred), to_t(gt))))

    result = {
        "psnr":     round(float(np.mean(psnrs)), 4),
        "ssim":     round(float(np.mean(ssims)), 4),
        "lpips":    round(float(np.mean(lpipss)), 4),
        "n_frames": len(frames),
    }
    logger.info("PSNR=%.4f  SSIM=%.4f  LPIPS=%.4f", result["psnr"], result["ssim"], result["lpips"])
    return result


def cleanup(out_dir: Path, logger: logging.Logger):
    droid_dir = out_dir / "droid_c2w"
    if droid_dir.exists():
        shutil.rmtree(droid_dir)
        logger.info("droid_c2w/ gelöscht")

    rgbdnua_dir = out_dir / "rgbdnua"
    if rgbdnua_dir.exists():
        frames = sorted(rgbdnua_dir.glob("FrameId=*.png"))
        if len(frames) > 3:
            keep = {0, len(frames) // 2, len(frames) - 1}
            removed = sum(1 for i, fp in enumerate(frames)
                          if i not in keep and not fp.unlink())
            logger.info("rgbdnua: %d Bilder gelöscht, 3 behalten (Anfang/Mitte/Ende)", removed)

    ply_dir = out_dir / "ply"
    if ply_dir.exists():
        plys = sorted(ply_dir.glob("*.ply"), key=lambda p: p.stat().st_size, reverse=True)
        for fp in plys[1:]:
            fp.unlink()
        if len(plys) > 1:
            logger.info("ply: %d Datei(en) gelöscht, 1 behalten (%s)",
                        len(plys) - 1, plys[0].name)


def ply_size_mb(out_dir: Path) -> Optional[float]:
    plys = list((out_dir / "ply").glob("*.ply")) if (out_dir / "ply").exists() else []
    if not plys:
        return None
    return round(max(p.stat().st_size for p in plys) / 1e6, 2)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="VINGS-Mono Experiment Runner")
    parser.add_argument("config")
    parser.add_argument("--prefix", default="")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    if not config_path.exists():
        print(f"[FEHLER] Config nicht gefunden: {config_path}", file=sys.stderr)
        sys.exit(1)

    cfg = load_yaml(config_path)
    save_dir = Path(cfg["output"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    # Logger schon hier aufsetzen, damit auch Fehler beim Start geloggt werden
    run_log = save_dir / f"run_{time.strftime('%Y%m%d_%H%M%S')}.log"
    logger = setup_logging(run_log)

    logger.info("=" * 58)
    logger.info("Config  : %s", config_path.name)
    logger.info("Prefix  : %s", args.prefix or "<leer>")
    logger.info("Log     : %s", run_log)
    logger.info("=" * 58)

    known_dirs = set(d for d in save_dir.iterdir() if d.is_dir())

    script_dir = Path(__file__).parent
    cmd = [sys.executable, "-u", str(script_dir / "run.py"), str(config_path)]
    if args.prefix:
        cmd += ["--prefix", args.prefix]

    logger.debug("Befehl: %s", " ".join(cmd))

    # GPU-Polling
    stop_gpu = threading.Event()
    gpu_readings: list = []

    def _gpu_worker():
        nonlocal gpu_readings
        gpu_readings = poll_gpu(stop_gpu)

    gpu_thread = threading.Thread(target=_gpu_worker, daemon=True)
    gpu_thread.start()

    # Peak-RAM via /usr/bin/time -v
    use_time_v = Path("/usr/bin/time").exists()
    time_tmp = Path("/tmp/_vings_run_time.txt")

    logger.info("Starte run.py …")
    t0 = time.time()

    if use_time_v:
        full_cmd = ["/usr/bin/time", "-v", "--output", str(time_tmp)] + cmd
    else:
        full_cmd = cmd

    proc = subprocess.Popen(
        full_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # stderr → stdout zusammenführen
    )
    ret = stream_subprocess(proc, run_log, logger)
    wall_sec = time.time() - t0

    stop_gpu.set()
    gpu_thread.join()

    logger.info("run.py beendet (rc=%d, %.1f min)", ret, wall_sec / 60)

    # Output-Verzeichnis finden
    out_dir = find_new_outdir(save_dir, known_dirs)
    if out_dir is None:
        dirs = sorted([d for d in save_dir.iterdir() if d.is_dir()])
        out_dir = dirs[-1] if dirs else save_dir
    logger.info("Output-Ordner: %s", out_dir.name)

    # Log vom Exp-Parent in den Run-Ordner verschieben.
    if out_dir != save_dir:
        new_log = out_dir / run_log.name
        for h in list(logger.handlers):
            if isinstance(h, logging.FileHandler):
                h.close()
                logger.removeHandler(h)
        shutil.move(str(run_log), str(new_log))
        fh = logging.FileHandler(new_log)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(
            "%(asctime)s  %(levelname)-7s  %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"))
        logger.addHandler(fh)
        run_log = new_log
        logger.info("Log in Run-Ordner verschoben: %s", run_log)

    # Peak-RAM parsen
    peak_ram_gb = None
    if use_time_v and time_tmp.exists():
        for line in time_tmp.read_text().splitlines():
            if "Maximum resident set size" in line:
                kb = int("".join(filter(str.isdigit, line)))
                peak_ram_gb = round(kb / 1e6, 2)
                break
        time_tmp.unlink(missing_ok=True)
    logger.debug("Peak-RAM: %s GB", peak_ram_gb)

    peak_gpu_mib = max(gpu_readings) if gpu_readings else None
    logger.debug("Peak-GPU: %s MiB", peak_gpu_mib)

    ply_mb = ply_size_mb(out_dir)

    # Metriken
    rgbdnua_dir = out_dir / "rgbdnua"
    metrics = compute_metrics(rgbdnua_dir, logger) if rgbdnua_dir.exists() else \
              {"psnr": None, "ssim": None, "lpips": None, "n_frames": 0}

    # Cleanup
    logger.info("Cleanup …")
    cleanup(out_dir, logger)

    # Ergebnis
    result = {
        "config":       str(config_path),
        "prefix":       args.prefix,
        "status":       "OK" if ret == 0 else f"FAIL(rc={ret})",
        "laufzeit_min": round(wall_sec / 60, 2),
        "ram_gb":       peak_ram_gb,
        "gpu_mib":      peak_gpu_mib,
        "ply_mb":       ply_mb,
        "psnr":         metrics["psnr"],
        "ssim":         metrics["ssim"],
        "lpips":        metrics["lpips"],
        "n_frames":     metrics["n_frames"],
    }

    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(result, indent=2))

    logger.info("=" * 58)
    logger.info("ERGEBNIS")
    logger.info("=" * 58)
    for k, v in result.items():
        logger.info("  %-16s: %s", k, v)
    logger.info("metrics.json → %s", metrics_path)
    logger.info("run.log      → %s", run_log)
    logger.info("=" * 58)

    sys.exit(0 if ret == 0 else 1)


if __name__ == "__main__":
    main()
