#!/bin/bash
# Voller interval1-Survey mit LiDAR-Tiefe in VRAM-sicheren Chunks (GT-Posen ->
# selber Frame -> trivial mergebar). skip 8, 600-Frame-Chunks ab Frame 300
# (Hover uebersprungen). Seriell. Am Ende Merge -> kompletter scharfer Survey.
set -u
cd /home/philipp/Dokumente/Github/VINGS-Mono-BA
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
PY=/home/philipp/anaconda3/envs/vings/bin/python
BASE=configs/local/interval1/interval1_droid_full.yaml
CHUNKDIR=output/exp_interval1_droidchunks
mkdir -p "$CHUNKDIR"
PLYS=()

CHUNK=600
for START in 300 900 1500 2100 2700 3300 3900 4500 5100; do
  cfg="$CHUNKDIR/cfg_$START.yaml"
  cp "$BASE" "$cfg"
  sed -i "s/^  start_frame: .*/  start_frame: $START/" "$cfg"
  sed -i "s/^  max_frames: .*/  max_frames: $CHUNK/" "$cfg"
  $PY - "$cfg" "$START" <<'PYEOF'
import sys, re
cfg, start = sys.argv[1], sys.argv[2]
s = open(cfg).read()
s = re.sub(r"output:\s*\{save_dir:[^}]*\}",
           f"output: {{save_dir: /home/philipp/Dokumente/Github/VINGS-Mono-BA/output/exp_interval1_droidchunks/c{start}/}}", s)
open(cfg, "w").write(s)
PYEOF
  echo "=========== LIDAR-CHUNK start=$START len=$CHUNK ==========="
  $PY scripts/run_experiment.py "$cfg" > "$CHUNKDIR/run_$START.log" 2>&1
  d=$(ls -dt output/exp_interval1_droidchunks/c$START/*/ 2>/dev/null | head -1)
  ply=$(ls "$d"ply/idx=*_2dgs.ply 2>/dev/null | sort -t= -k2 -n | tail -1)
  if [ -n "$ply" ]; then PLYS+=("$ply"); echo "  chunk $START -> $(basename "$ply")"; else echo "  WARN: keine PLY chunk $START"; fi
done

echo "=========== MERGE (${#PLYS[@]} chunks) ==========="
$PY scripts/eval/merge_plys.py --out "$CHUNKDIR/survey_droid_complete.ply" --opacity-min 0.2 "${PLYS[@]}"
echo "=========== FERTIG -> $CHUNKDIR/survey_droid_complete.ply ==========="
