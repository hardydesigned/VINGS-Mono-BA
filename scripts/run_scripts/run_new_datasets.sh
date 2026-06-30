#!/usr/bin/env bash
# =============================================================================
# Run the full 1:1 amtown03 selector sweep on three additional MARS-LVIG /
# UAVScenes datasets (AMvalley03, HKisland03, HKairport03), each over three
# motion-profile windows (9 slices total, ~57 configs each):
#
#   200f  reiner Flug         (steady cruise, low yaw)
#   400f  Hover -> Flug       (straddles the take-off / cruise transition)
#   500f  schwieriger Inhalt  (turns / unsteady flight / deceleration->hover)
#
# Windows were picked from the DJI velocity / yaw-rate motion reports printed
# by scripts/prepare_marslvig.py --motion-report.
#
# Per slice this:
#   1) ensures the dataset is in the amtown03 layout (prepare_marslvig.py)
#   2) (re)generates the config tree (gen_slice_configs.py)
#   3) runs the whole sweep serially via run_sweep.sh --slice with CLEANUP_RUNS=1
#      (each run's output dir is deleted right after its CSV row is logged)
#   4) removes the slice's exp_ dir (belt-and-suspenders) — the result CSV in
#      output/<DS>_s<START>_<FRAMES>f_results.csv is kept.
#
# Serial only (never parallel) — run_sweep.sh handles VRAM/RAM idle waits and
# is idempotent (resumes; OK rows are skipped). Safe to re-run after a kill.
# =============================================================================
set -u
cd "$(dirname "$0")/.."
REPO="$PWD"

source /home/philipp/anaconda3/etc/profile.d/conda.sh
conda activate vings

# DS:START:FRAMES
SLICES=(
  "AMvalley03:2900:200"  "AMvalley03:800:400"  "AMvalley03:4300:500"
  "HKisland03:2600:200"  "HKisland03:770:400"  "HKisland03:2800:500"
  "HKairport03:1600:200" "HKairport03:670:400" "HKairport03:2800:500"
)

# 1) Prepare datasets into the amtown03 layout (idempotent; the image-resize
#    step self-skips once images_all is complete).
for DS in AMvalley03 HKisland03 HKairport03; do
  if [[ ! -f "$HOME/Dokumente/datasets/$DS/metadata/camstamp_all.txt" ]]; then
    echo "[run-new] preparing $DS ..."
    python scripts/prepare_marslvig.py --dataset "$DS"
  fi
done

# 2) Generate + run each slice; keep only the CSV.
export CLEANUP_RUNS=1
for s in "${SLICES[@]}"; do
  IFS=':' read -r DS START FRAMES <<< "$s"
  SLUG="s${START}_${FRAMES}f"
  echo
  echo "############################################################"
  echo "# SLICE  $DS  $SLUG"
  echo "############################################################"
  python scripts/gen_slice_configs.py --dataset "$DS" --start "$START" --frames "$FRAMES"
  bash scripts/run_sweep.sh --slice "$s"
  rm -rf "$REPO/output/exp_${DS}_${SLUG}"
  echo "[run-new] $DS $SLUG done -> output/${DS}_${SLUG}_results.csv"
done

echo
echo "[run-new] ALL DONE. Result CSVs:"
ls -1 "$REPO"/output/{AMvalley03,HKisland03,HKairport03}_s*_results.csv 2>/dev/null
