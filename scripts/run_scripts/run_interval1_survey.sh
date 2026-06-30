#!/bin/bash
# Voller interval1-Cruise-Survey aus mono+GPS (KEIN LiDAR, KEIN GT) -- kanonischer,
# robuster, GEMESSENER Lauf. Schreibt pro Segment eine Zeile ins Sweep-CSV (gleiches
# Schema wie docs/results/s1000_400f_results.csv, via scripts/log_sweep_row.py).
#
# Architektur (docs/INTERVAL1_LIDAR_PIPELINE.md):
#   - Cruise (f1000-4600) in überlappenden durchgehenden Segmenten (je 500f, Step 300).
#     Jedes Segment intern nahtlos+scharf (EIN DROID-Frame, Storage-Manager).
#   - Robustheit gg. VRAM-Spike: numkf, Checkpoint alle CKPT KF, Retry-Schleife
#     (rc=137-Tod ist stochastisch; eine Teil-ply nach Checkpoint ist nutzbar).
#   - Pro Segment: sim3_unwarp.py --gps-csv -> metrisch+driftfrei im gemeinsamen UTM-Frame.
#   - Am Ende merge_survey.sh: detilt_gps GPS-Boden-Leveling + clean -> survey_complete.ply.
#
# Env: FRAMES STEP S0 S1 NUMKF CKPT MAXTRY CSV MERGE  (MERGE=0 = nur Segmente+Messung, kein Merge)
#      OUT (Save-/ply-Ordner)  SELECTOR_FROM (Referenz-Config -> frame_selector/gate_a injizieren)
#      SUMMARY_CSV + SURVEY_NAME (1 Zeile/Survey in die gemeinsame Summary-CSV, siehe measure_survey.py)
# Positional: explizite Start-Frames (sonst S0..S1 in STEP-Schritten).
set -u
cd /home/philipp/Dokumente/Github/VINGS-Mono-BA
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
PY=/home/philipp/anaconda3/envs/vings/bin/python
RTK=/home/philipp/Dokumente/datasets/interval1_AMtown03/rtk_positions_raw.csv
OUT=${OUT:-output/exp_interval1_survey}
mkdir -p "$OUT"
FRAMES=${FRAMES:-500}; STEP=${STEP:-300}; S0=${S0:-0}; S1=${S1:-4600}
NUMKF=${NUMKF:-4}; CKPT=${CKPT:-20}; MAXTRY=${MAXTRY:-2}; MERGE=${MERGE:-1}
CSV=${CSV:-docs/results/interval1_survey_full_results.csv}
# optionaler Selektor-Hook: frame_selector/gate_a aus einer Referenz-Config uebernehmen
SELARG=(); [ -n "${SELECTOR_FROM:-}" ] && SELARG=(--selector-from "$SELECTOR_FROM")
if [ "$#" -gt 0 ]; then STARTS=("$@"); else
  STARTS=(); s=$S0; while [ "$s" -le "$S1" ]; do STARTS+=("$s"); s=$((s+STEP)); done
fi

for START in "${STARTS[@]}"; do
  LABEL="s$START"; CFG="$OUT/cfg_$LABEL.yaml"; GPSPLY="$OUT/${LABEL}_gps.ply"
  $PY scripts/eval/gen_opt_cfg.py "$CFG" --savedir "$OUT/$LABEL/" \
     --start "$START" --frames "$FRAMES" --hw 240 288 --iters 150 --numkf "$NUMKF" \
     --kfskip 1 --prune-op 0.5 --dist-thresh 0.1 --no-ext ${SELARG[@]+"${SELARG[@]}"} || continue
  $PY - "$CFG" "$CKPT" <<'EOF'
import sys, yaml
c=yaml.full_load(open(sys.argv[1])); c["ply_checkpoint_every_kf"]=int(sys.argv[2])
yaml.safe_dump(c,open(sys.argv[1],"w"),sort_keys=False)
EOF
  echo "=========== SEGMENT start=$START (numkf=$NUMKF) ==========="
  KB=$(mktemp); ls -1 "$OUT/$LABEL" 2>/dev/null > "$KB"        # known-before für log_sweep_row
  PLY=""; RC=1; TS0=$(date +%s)
  for try in $(seq 1 "$MAXTRY"); do
    $PY scripts/run_experiment.py "$CFG" > "$OUT/run_$START.log" 2>&1; RC=$?
    RD=$(ls -dt "$OUT/$LABEL"/*/ 2>/dev/null | head -1)
    PLY=$(ls "$RD"ply/idx=*_2dgs.ply 2>/dev/null | sort -t= -k2 -n | tail -1)
    [ -n "$PLY" ] && break
    echo "  WARN: kein ply (Versuch $try/$MAXTRY) -- starb vor erstem Checkpoint (KF<$CKPT)"
  done
  TS1=$(date +%s)
  ST="OK"; [ "$RC" -ne 0 ] && ST="FAIL"     # exit_code+ply-Spalten zeigen, ob eine Teil-ply nutzbar ist
  $PY scripts/log_sweep_row.py --csv "$CSV" --dataset interval1 --group survey \
     --variant "$LABEL" --config "$CFG" --save-dir "$OUT/$LABEL" --known-before "$KB" \
     --start-ts "$TS0" --end-ts "$TS1" --exit-code "$RC" --status "$ST"
  rm -f "$KB"
  if [ -z "$PLY" ]; then echo "  WARN: keine ply start=$START -- übersprungen"; continue; fi
  $PY scripts/eval/sim3_unwarp.py "$PLY" --droid-poses "$RD"tracker_raw_c2w.txt \
     --gps-csv "$RTK" --out "$GPSPLY" --window 80 --knn 4 --crop-radius 100 --nadir-clear 5
  rm -f "$RD"ply/idx=*_2dgs.ply        # Disk: rohe Segment-ply weg (gps-Version + Messung reichen)
  [ -f "$GPSPLY" ] && echo "  -> $GPSPLY"
done

if [ "$MERGE" != "0" ]; then OUT="$OUT" bash scripts/merge_survey.sh; else echo "MERGE=0 -> kein Merge"; fi
echo "=========== MESSUNG (sauber, inkl. Aggregat-Zeile ALL) -> $CSV ==========="
# Regeneriert die CSV sauber aus den Run-Ordnern: 1 Zeile/Segment + 1 ALL-Aggregat
# (Gesamtlaufzeit, finale Survey-ply-Größe, gewichtete Mittel). Ersetzt die während
# des Laufs inline geschriebenen Progress-Zeilen. Braucht survey_complete.ply für die
# ply-Größe der Aggregat-Zeile -> nach dem Merge. Optional zusätzlich 1 Zeile/Survey in
# die gemeinsame Summary-CSV (SUMMARY_CSV/SURVEY_NAME, für den Selektor-Vergleich).
MEAS=(--out-dir "$OUT" --csv "$CSV")
[ -n "${SUMMARY_CSV:-}" ] && MEAS+=(--summary-csv "$SUMMARY_CSV")
[ -n "${SURVEY_NAME:-}" ] && MEAS+=(--survey-name "$SURVEY_NAME")
$PY scripts/eval/measure_survey.py "${MEAS[@]}"
