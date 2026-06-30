#!/bin/bash
# Hochaufloesender (384x456, iters 100) DROID-Tiefe-Survey in VRAM-sicheren
# 300-Frame-Chunks (Peak-GPU ~2.5 GB << 8GB-Watchdog). GT-Posen -> selber Frame
# -> trivial mergebar. Seriell. Am Ende Merge + Floater-Clean.
set -u
cd /home/philipp/Dokumente/Github/VINGS-Mono-BA
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
PY=/home/philipp/anaconda3/envs/vings/bin/python
BASE=configs/local/interval1/interval1_droid384_full.yaml
CHUNKDIR=output/exp_interval1_droid384chunks
GTPOSES=/home/philipp/Dokumente/datasets/interval1_AMtown03/vings/poses_w2c.txt
mkdir -p "$CHUNKDIR"
PLYS=()

CHUNK=300
for START in 300 600 900 1200 1500 1800 2100 2400 2700 3000 3300 3600 3900 4200 4500 4800 5100 5400; do
  cfg="$CHUNKDIR/cfg_$START.yaml"
  cp "$BASE" "$cfg"
  sed -i "s/^  start_frame: .*/  start_frame: $START/" "$cfg"
  sed -i "s/^  max_frames: .*/  max_frames: $CHUNK/" "$cfg"
  $PY - "$cfg" "$START" <<'PYEOF'
import sys, re
cfg, start = sys.argv[1], sys.argv[2]
s = open(cfg).read()
s = re.sub(r"output:\s*\{save_dir:[^}]*\}",
           f"output: {{save_dir: /home/philipp/Dokumente/Github/VINGS-Mono-BA/output/exp_interval1_droid384chunks/c{start}/}}", s)
open(cfg, "w").write(s)
PYEOF
  echo "=========== DROID384-CHUNK start=$START len=$CHUNK ==========="
  $PY scripts/run_experiment.py "$cfg" > "$CHUNKDIR/run_$START.log" 2>&1
  d=$(ls -dt output/exp_interval1_droid384chunks/c$START/*/ 2>/dev/null | head -1)
  ply=$(ls "$d"ply/idx=*_2dgs.ply 2>/dev/null | sort -t= -k2 -n | tail -1)
  if [ -n "$ply" ]; then PLYS+=("$ply"); echo "  chunk $START -> $(basename "$ply")"; else echo "  WARN: keine PLY chunk $START"; fi
done

echo "=========== MERGE (${#PLYS[@]} chunks) ==========="
$PY scripts/eval/merge_plys.py --out "$CHUNKDIR/survey_droid384_raw.ply" --opacity-min 0.2 "${PLYS[@]}"
echo "=========== CLEAN (Floater-Filter) ==========="
$PY scripts/eval/clean_ply.py "$CHUNKDIR/survey_droid384_raw.ply" \
  --gt-poses "$GTPOSES" --max-dist 120 --max-scale 2.0 --opacity-min 0.5 --max-z-spread 60 \
  --out "$CHUNKDIR/survey_droid384_cleaned.ply"
# Disk sparen: Raw-Merge (gross) nach erfolgreichem Clean loeschen
if [ -f "$CHUNKDIR/survey_droid384_cleaned.ply" ]; then
  rm -f "$CHUNKDIR/survey_droid384_raw.ply"
  echo "[disk] raw-merge geloescht"
fi
df -h . | tail -1
echo "=========== FERTIG -> $CHUNKDIR/survey_droid384_cleaned.ply ==========="
