#!/bin/bash
# Voller AGZ-Survey (agz_0_10000, 10000 Frames) -- Analog zu run_interval1_survey.sh.
# Durchgehender DROID-Lauf in ueberlappenden Segmenten (je 500f, Step 300).
#
# UNTERSCHIED zu interval1: AGZ hat KEIN echtes RTK-GPS (agz_gps.csv ist synthetisch,
# aus den GT-Posen invers-projiziert, ohne easting/northing). Deshalb ankert der
# Sim3-Unwarp an die GT-DJI-Posen (agz_poses_w2c.txt, w2c -> C=-R^T t) via --gt-poses
# statt --gps-csv. Die GT-Posen decken alle 10000 Frames in EINEM globalen Frame ab
# -> alle Segmente landen im selben Frame -> direkt mergebar (wie der UTM-Frame bei GPS).
# (AMTOWN hatte verifiziert: GPS = GT, Kamera-RMSE 1.75 vs 1.82 m -> aequivalent.)
#
# Env: FRAMES STEP S0 S1 NUMKF CKPT MAXTRY CSV MERGE  (MERGE=0 = nur Segmente+Messung)
#      OUT (Save-/ply-Ordner)  SELECTOR_FROM (Referenz-Config -> frame_selector injizieren)
#      SUMMARY_CSV + SURVEY_NAME (1 Zeile/Survey in die gemeinsame Summary-CSV)
# Positional: explizite Start-Frames (sonst S0..S1 in STEP-Schritten).
set -u
cd /home/philipp/Dokumente/Github/VINGS-Mono-BA
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
PY=/home/philipp/anaconda3/envs/vings/bin/python
BASECFG=configs/local/agz/agz_droid_full.yaml
GT=/home/philipp/Dokumente/datasets/agz/agz_0_10000/agz_poses_w2c.txt
OUT=${OUT:-output/exp_agz_survey}
mkdir -p "$OUT"
# Default deckt den ganzen agz_0_10000-Block ab (0..9500 Step 300 = 32 Segmente).
FRAMES=${FRAMES:-500}; STEP=${STEP:-300}; S0=${S0:-0}; S1=${S1:-9500}
NUMKF=${NUMKF:-4}; CKPT=${CKPT:-20}; MAXTRY=${MAXTRY:-2}; MERGE=${MERGE:-1}
CSV=${CSV:-docs/results/agz_survey_full_results.csv}
SELARG=(); [ -n "${SELECTOR_FROM:-}" ] && SELARG=(--selector-from "$SELECTOR_FROM")
if [ "$#" -gt 0 ]; then STARTS=("$@"); else
  STARTS=(); s=$S0; while [ "$s" -le "$S1" ]; do STARTS+=("$s"); s=$((s+STEP)); done
fi

for START in "${STARTS[@]}"; do
  LABEL="s$START"; CFG="$OUT/cfg_$LABEL.yaml"; GPSPLY="$OUT/${LABEL}_gps.ply"
  $PY scripts/eval/gen_opt_cfg.py "$CFG" --base "$BASECFG" --savedir "$OUT/$LABEL/" \
     --start "$START" --frames "$FRAMES" --hw 240 432 --iters 150 --numkf "$NUMKF" \
     --kfskip 1 --prune-op 0.5 --dist-thresh 0.1 --no-ext ${SELARG[@]+"${SELARG[@]}"} || continue
  $PY - "$CFG" "$CKPT" <<'EOF'
import sys, yaml
c=yaml.full_load(open(sys.argv[1])); c["ply_checkpoint_every_kf"]=int(sys.argv[2])
yaml.safe_dump(c,open(sys.argv[1],"w"),sort_keys=False)
EOF
  echo "=========== SEGMENT start=$START (numkf=$NUMKF) ==========="
  KB=$(mktemp); ls -1 "$OUT/$LABEL" 2>/dev/null > "$KB"
  PLY=""; RC=1; TS0=$(date +%s)
  for try in $(seq 1 "$MAXTRY"); do
    $PY scripts/run_experiment.py "$CFG" > "$OUT/run_$START.log" 2>&1; RC=$?
    RD=$(ls -dt "$OUT/$LABEL"/*/ 2>/dev/null | head -1)
    PLY=$(ls "$RD"ply/idx=*_2dgs.ply 2>/dev/null | sort -t= -k2 -n | tail -1)
    [ -n "$PLY" ] && break
    echo "  WARN: kein ply (Versuch $try/$MAXTRY) -- starb vor erstem Checkpoint (KF<$CKPT)"
  done
  TS1=$(date +%s)
  ST="OK"; [ "$RC" -ne 0 ] && ST="FAIL"
  $PY scripts/log_sweep_row.py --csv "$CSV" --dataset agz --group survey \
     --variant "$LABEL" --config "$CFG" --save-dir "$OUT/$LABEL" --known-before "$KB" \
     --start-ts "$TS0" --end-ts "$TS1" --exit-code "$RC" --status "$ST"
  rm -f "$KB"
  if [ -z "$PLY" ]; then echo "  WARN: keine ply start=$START -- übersprungen"; continue; fi
  $PY scripts/eval/sim3_unwarp.py "$PLY" --droid-poses "$RD"tracker_raw_c2w.txt \
     --gt-poses "$GT" --out "$GPSPLY" --window 80 --knn 4 --crop-radius 100 --nadir-clear 5
  rm -f "$RD"ply/idx=*_2dgs.ply
  [ -f "$GPSPLY" ] && echo "  -> $GPSPLY"
done

if [ "$MERGE" != "0" ]; then OUT="$OUT" bash scripts/merge_agz_survey.sh; else echo "MERGE=0 -> kein Merge"; fi
echo "=========== MESSUNG -> $CSV ==========="
MEAS=(--out-dir "$OUT" --csv "$CSV")
[ -n "${SUMMARY_CSV:-}" ] && MEAS+=(--summary-csv "$SUMMARY_CSV")
[ -n "${SURVEY_NAME:-}" ] && MEAS+=(--survey-name "$SURVEY_NAME")
$PY scripts/eval/measure_survey.py "${MEAS[@]}"
