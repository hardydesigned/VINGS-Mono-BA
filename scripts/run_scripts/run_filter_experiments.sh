#!/usr/bin/env bash
# Sequenziell alle Filter-Experiment-Configs via run_experiment.py.
# Configs liegen in configs/local/exp/ (generiert via /tmp/gen_filter_configs.py).
# Reihenfolge: erst 800-Frame-Suite (no loop), dann 200-Frame-Suite (storage manager).
# Innerhalb jeder Suite: nofilter (kein skip), dann nofilter+skip 2..10.

set -u
cd "$(dirname "$0")/.."

source /home/philipp/anaconda3/etc/profile.d/conda.sh
conda activate vings

CONFIGS=(
  configs/local/skip_no_filter/smallcity_800_nofilter.yaml
  # configs/local/skip_no_filter/smallcity_200_nofilter.yaml
  # configs/local/skip_no_filter/smallcity_200_nofilter_skip2.yaml
  # configs/local/skip_no_filter/smallcity_200_nofilter_skip3.yaml
  # configs/local/skip_no_filter/smallcity_200_nofilter_skip4.yaml
  # configs/local/skip_no_filter/smallcity_200_nofilter_skip5.yaml
  # configs/local/skip_no_filter/smallcity_200_nofilter_skip6.yaml
  # configs/local/skip_no_filter/smallcity_200_nofilter_skip7.yaml
  # configs/local/skip_no_filter/smallcity_200_nofilter_skip8.yaml
  # configs/local/skip_no_filter/smallcity_200_nofilter_skip9.yaml
  # configs/local/skip_no_filter/smallcity_200_nofilter_skip10.yaml
)

TOTAL=${#CONFIGS[@]}
T_START=$(date +%s)
FAILS=()
i=0

for cfg in "${CONFIGS[@]}"; do
  i=$((i + 1))
  echo
  echo "=========================================="
  echo " [$i/$TOTAL] $(basename "$cfg")"
  echo "=========================================="
  if python scripts/run_experiment.py "$cfg"; then
    echo "[OK] $cfg"
  else
    rc=$?
    echo "[WARN] $cfg failed (rc=$rc), continuing..."
    FAILS+=("$cfg")
  fi
done

T_END=$(date +%s)
ELAPSED=$((T_END - T_START))
echo
echo "=========================================="
echo " Done: $TOTAL configs in $((ELAPSED/60))m $((ELAPSED%60))s"
echo "=========================================="
if [ "${#FAILS[@]}" -gt 0 ]; then
  echo "Failed runs (${#FAILS[@]}):"
  for f in "${FAILS[@]}"; do echo "  $f"; done
  exit 1
fi
