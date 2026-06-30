#!/usr/bin/env bash
# =============================================================================
# AGZ selector sweep — runs the complete sweep first over ALL 200f configs
# (agz_0_10000 frames 7950..8149, pure motion), then over ALL 400f configs
# (frames 2675..3074, hover->motion). Thin wrapper around run_sweep.sh; reuses
# its battle-tested VRAM/RAM-idle, timeout, idempotency and CSV-logging logic.
#
# Produces, in the SAME style as the amtown03 sweeps:
#   output/agz_s7950_200f_results.csv   (200f, mirror of s3100_200f_results.csv)
#   output/agz_s2675_400f_results.csv   (400f, mirror of s1000_400f_results.csv)
#
# Companion notebooks (read those CSVs):
#   scripts/analyze_sweep_agz_s7950_200f_fair.ipynb
#   scripts/analyze_sweep_agz_s2675_400f_fair.ipynb
#
# Pass-through flags (forwarded to run_sweep.sh): --force, --only <substr>,
#   --start-at <variant>. Env: TIMEOUT_PER_RUN, SLEEP_BETWEEN, etc.
# Run order is 200f-then-400f as requested; either is idempotent (status=OK
# rows are skipped on re-run).
# =============================================================================
set -u
cd "$(dirname "$0")/.."
SWEEP="scripts/run_sweep.sh"

echo "############################################################"
echo "# AGZ SWEEP — 200f (s7950) then 400f (s2675)"
echo "# pass-through args: $*"
echo "############################################################"

echo
echo "=== [1/2] AGZ 200f sweep (frames 7950..8149, pure motion) ==="
bash "$SWEEP" --agz200 "$@"

echo
echo "=== [2/2] AGZ 400f sweep (frames 2675..3074, hover->motion) ==="
bash "$SWEEP" --agz400 "$@"

echo
echo "############################################################"
echo "# AGZ SWEEP DONE"
echo "#   200f CSV: output/agz_s7950_200f_results.csv"
echo "#   400f CSV: output/agz_s2675_400f_results.csv"
echo "# Notebooks:"
echo "#   scripts/analyze_sweep_agz_s7950_200f_fair.ipynb"
echo "#   scripts/analyze_sweep_agz_s2675_400f_fair.ipynb"
echo "############################################################"
