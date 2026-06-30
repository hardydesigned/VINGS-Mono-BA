#!/usr/bin/env bash
# =============================================================================
# Extra-variety slice sweep — runs the selector sweep over four NEW amtown03/AGZ
# windows picked for maximum motion variety (hover vs. pure cruise), on top of
# the existing s1000/s3100 (amtown) and s2675/s7950 (AGZ) slices.
#
# Slices (motion profile from the GT pose files, mean step per frame):
#   amtown03 s5000_200f  fast pure cruise   (~1122 mm/frame — fastest in flight)
#   amtown03 s5400_400f  decel / slow appr. (~95->68 mm/frame — hover-like)
#   agz      s5600_200f  pure deep hover    (~12 mm/frame — slowest AGZ window)
#   agz      s9000_400f  sustained cruise   (~48 mm/frame — fast AGZ window)
#
# Thin wrapper around run_sweep.sh --slice; reuses its VRAM/RAM-idle, timeout,
# idempotency and CSV-logging logic. Each slice writes its own CSV:
#   output/amtown03_s5000_200f_results.csv
#   output/amtown03_s5400_400f_results.csv
#   output/agz_s5600_200f_results.csv
#   output/agz_s9000_400f_results.csv
#
# Configs are produced by scripts/gen_slice_configs.py (run --gen to (re)build).
# Pass-through flags (forwarded to run_sweep.sh): --force, --only <substr>,
#   --start-at <variant>. Env: TIMEOUT_PER_RUN, SLEEP_BETWEEN, etc.
# Idempotent: status=OK rows are skipped on re-run.
# =============================================================================
set -u
cd "$(dirname "$0")/.."
SWEEP="scripts/run_sweep.sh"

# Slice list: "dataset:start:frames  human-label"
SLICES=(
  "amtown03:5000:200|amtown03 fast pure cruise (~1122 mm/frame)"
  "amtown03:5400:400|amtown03 decel/slow approach (~95->68 mm/frame)"
  "agz:5600:200|AGZ pure deep hover (~12 mm/frame)"
  "agz:9000:400|AGZ sustained fast cruise (~48 mm/frame)"
)

# Optional: (re)generate all slice configs first with `--gen`.
if [[ "${1:-}" == "--gen" ]]; then
  shift
  echo "=== (re)generating slice configs ==="
  python scripts/gen_slice_configs.py --dataset amtown03 --start 5000 --frames 200
  python scripts/gen_slice_configs.py --dataset amtown03 --start 5400 --frames 400
  python scripts/gen_slice_configs.py --dataset agz      --start 5600 --frames 200
  python scripts/gen_slice_configs.py --dataset agz      --start 9000 --frames 400
fi

echo "############################################################"
echo "# EXTRA-VARIETY SLICE SWEEP — ${#SLICES[@]} slices"
echo "# pass-through args: $*"
echo "############################################################"

i=0
for entry in "${SLICES[@]}"; do
  i=$((i + 1))
  IFS='|' read -r slice label <<< "$entry"
  echo
  echo "=== [$i/${#SLICES[@]}] $slice  —  $label ==="
  bash "$SWEEP" --slice "$slice" "$@"
done

echo
echo "############################################################"
echo "# EXTRA-VARIETY SLICE SWEEP DONE"
echo "#   CSVs: output/{amtown03_s5000_200f,amtown03_s5400_400f,"
echo "#               agz_s5600_200f,agz_s9000_400f}_results.csv"
echo "############################################################"
