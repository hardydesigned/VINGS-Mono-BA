#!/usr/bin/env bash
# Mapper-Skip-Suite: Tracking laeuft auf allen Frames, Mapper bekommt nur
# jeden N-ten KF. Configs unter configs/local/mapskip/.
# Ziel: PSNR/SSIM/LPIPS vs. mapper_kf_skip vergleichen.

set -u
cd "$(dirname "$0")/.."

source /home/philipp/anaconda3/etc/profile.d/conda.sh
conda activate vings

CONFIGS=(
  # configs/local/mapskip/smallcity_200_mapskip1.yaml
  # configs/local/mapskip/smallcity_200_mapskip2.yaml
  # configs/local/mapskip/smallcity_200_mapskip3.yaml
  # configs/local/mapskip/smallcity_200_mapskip4.yaml
  # configs/local/mapskip/smallcity_200_mapskip5.yaml
  # configs/local/mapskip/smallcity_200_mapskip6.yaml
  configs/local/mapskip/smallcity_200_mapskip7.yaml
  configs/local/mapskip/smallcity_200_mapskip8.yaml
  configs/local/mapskip/smallcity_200_mapskip9.yaml
  configs/local/mapskip/smallcity_200_mapskip10.yaml
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
