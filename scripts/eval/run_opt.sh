#!/bin/bash
# run_opt.sh LABEL -- <gen_opt_cfg args>
# Generiert Config, faehrt den Run, extrahiert PSNR/SSIM/Status/Peak-GPU.
set -u
cd /home/philipp/Dokumente/Github/VINGS-Mono-BA
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
PY=/home/philipp/anaconda3/envs/vings/bin/python
LABEL="$1"; shift
OUTDIR="output/exp_opt/$LABEL"
CFG="output/exp_opt/cfg_$LABEL.yaml"
mkdir -p "$OUTDIR" output/exp_opt
$PY scripts/eval/gen_opt_cfg.py "$CFG" --savedir "$OUTDIR/" "$@" || { echo "GEN FAIL"; exit 1; }
LOG="output/exp_opt/run_$LABEL.log"
$PY scripts/run_experiment.py "$CFG" > "$LOG" 2>&1
PSNR=$(grep -oE "PSNR=[0-9.]+" "$LOG" | tail -1)
SSIM=$(grep -oE "SSIM=[0-9.]+" "$LOG" | tail -1)
STAT=$(grep -oE "status +: [A-Za-z0-9()=]+" "$LOG" | tail -1)
GPU=$(grep -oE "gpu_mib +: [0-9]+" "$LOG" | tail -1)
NF=$(grep -oE "n_frames +: [0-9]+" "$LOG" | tail -1)
echo "RESULT[$LABEL] $PSNR $SSIM | $STAT | $GPU | $NF"
