#!/bin/bash
# Option C v2: non-metric (scharf) -> per-Chunk Sim3 ins metrische GT-Frame + Naht-Fix.
#
# Naht-Fix-Pipeline (scripts/eval/chunk_postfix.py, Diagnose docs/INTERVAL1_LIDAR_PIPELINE.md):
#   1. Footprint-Crop  : Gaussian < crop-radius zur NAECHSTEN eigenen Chunk-Kamera
#                        (globaler max-dist crop kann Floater nicht entfernen, weil ueber
#                         die 5-km-Bahn immer IRGENDEINE GT-Kamera nah ist).
#   2. Nadir-Filter    : Gaussians ueber der Drohne (z > cam_z - nadir_clear) verwerfen
#                        (Nadir-Cam schaut nach unten; c3150 war 61% solcher Tiefen-Floater).
#   3. Quality-Gate    : Chunk verwerfen wenn Sim3-Scale, Align-RMSE oder AGL (Flughoehe
#                        ueber Grund) vom Konsens abweichen (degenerierte DROID-Chunks wie
#                        c3300: scale 115 statt 82, AGL 58 statt 40).
# Chunks UEBERLAPPEN (step < len), damit ein verworfener Chunk keine Luecke hinterlaesst.
set -u
cd /home/philipp/Dokumente/Github/VINGS-Mono-BA
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
PY=/home/philipp/anaconda3/envs/vings/bin/python
GTPOSES=/home/philipp/Dokumente/datasets/interval1_AMtown03/vings/poses_w2c.txt
OUT=output/exp_interval1_optC
CHUNK=${CHUNK:-150}          # Frames pro Chunk (VRAM-sicher bei kfskip1/it200/dist0.1)
STEP=${STEP:-100}            # Chunk-Abstand -> 50 Frames Ueberlapp gegen Luecken
# Quality-Gate: rmse + AGL sind die echten Geometrie-Signale. Scale ist nur DROIDs
# willkuerliche Gauge (variiert 61-115 je Region) und wird vom Chaining ueberbrueckt
# -> KEIN scale-band-Gate per Default (war zu streng, warf brauchbare Chunks wie
# c3200 scale61/rmse2.07 raus). Per MEDSCALE=82 wieder aktivierbar.
MEDSCALE=${MEDSCALE:-}; MAXRMSE=${MAXRMSE:-2.5}; AGL=${AGL:-40}; AGLTOL=${AGLTOL:-0.45}
CROPR=${CROPR:-90}; NADIR=${NADIR:-5}
mkdir -p "$OUT"
if [ "$#" -gt 0 ]; then STARTS=("$@"); else
  STARTS=(); s=300; while [ "$s" -le 5400 ]; do STARTS+=("$s"); s=$((s+STEP)); done
fi
# Reliabilitaet: iters150 + numkf4 zaehmt den Densifikations-Spike (Peak ~3.3 GB
# statt 8.7 GB die mit numkf8/it200 ~50% der Chunks am Watchdog killten); PSNR ~23.5.
ITERS=${ITERS:-150}; NUMKF=${NUMKF:-4}
KEEP=(); KEEP_SPEC=()
for START in "${STARTS[@]}"; do
  LABEL="c$START"; CFG="$OUT/cfg_$LABEL.yaml"
  $PY scripts/eval/gen_opt_cfg.py "$CFG" --savedir "$OUT/$LABEL/" \
     --start "$START" --frames "$CHUNK" --hw 240 288 --iters "$ITERS" --numkf "$NUMKF" \
     --kfskip 1 --prune-op 0.5 --dist-thresh 0.1 --no-ext || continue
  echo "=========== OptC-CHUNK start=$START ==========="
  $PY scripts/run_experiment.py "$CFG" > "$OUT/run_$START.log" 2>&1
  RD=$(ls -dt "$OUT/$LABEL"/*/ 2>/dev/null | head -1)
  PLY=$(ls "$RD"ply/idx=*_2dgs.ply 2>/dev/null | sort -t= -k2 -n | tail -1)
  if [ -z "$PLY" ]; then echo "  WARN: keine PLY start=$START (VRAM/Crash?)"; continue; fi
  FIX="$OUT/${LABEL}_fix.ply"
  SCALEARG=(); [ -n "$MEDSCALE" ] && SCALEARG=(--median-scale "$MEDSCALE")
  $PY scripts/eval/chunk_postfix.py "$PLY" --transform \
     --droid-poses "$RD"tracker_raw_c2w.txt --gt-poses "$GTPOSES" --out "$FIX" \
     --crop-radius "$CROPR" --nadir-clear "$NADIR" \
     "${SCALEARG[@]}" --max-rmse "$MAXRMSE" --agl-target "$AGL" --agl-tol "$AGLTOL"
  RC=$?
  rm -f "$RD"ply/idx=*_2dgs.ply      # Disk sparen: rohe non-metric ply weg
  if [ "$RC" -eq 0 ] && [ -f "$FIX" ]; then
    KEEP+=("$FIX"); KEEP_SPEC+=("$LABEL:$FIX:$RD"); echo "  -> KEEP $FIX"
  fi
done

echo "=========== CHAIN (${#KEEP[@]} Chunks) ==========="
[ "${#KEEP[@]}" -eq 0 ] && { echo "Keine Chunks bestanden -- Gate zu streng?"; exit 1; }
# Sequentielles Chaining: Rotation aus Kamera-Orientierungen am Overlap (gut
# konditioniert), Position/Scale gegen Vorgaenger+GT. Fixt relative Kippung/Versatz.
CHOUT=$($PY scripts/eval/chain_chunks.py --chunks "${KEEP_SPEC[@]}" \
   --gt-poses "$GTPOSES" --out-dir "$OUT/chained" --overlap-weight 5 --match-dt 0.06 \
   | tee /dev/stderr | grep "^CHAINED_PLYS=" | cut -d= -f2-)
read -r -a CHAINED <<< "$CHOUT"
[ "${#CHAINED[@]}" -eq 0 ] && CHAINED=("${KEEP[@]}")   # Fallback: ungekettet

echo "=========== MERGE (${#CHAINED[@]} gekettete Chunks) ==========="
$PY scripts/eval/merge_plys.py --out "$OUT/survey_optC_raw.ply" --opacity-min 0.2 "${CHAINED[@]}"
$PY scripts/eval/clean_ply.py "$OUT/survey_optC_raw.ply" \
  --gt-poses "$GTPOSES" --max-dist 120 --max-scale 2.0 --opacity-min 0.4 \
  --out "$OUT/survey_optC_cleaned.ply"
[ -f "$OUT/survey_optC_cleaned.ply" ] && rm -f "$OUT/survey_optC_raw.ply"
df -h . | tail -1
echo "=========== FERTIG -> $OUT/survey_optC_cleaned.ply ==========="
