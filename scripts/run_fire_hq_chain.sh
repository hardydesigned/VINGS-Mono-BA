#!/usr/bin/env bash
# Serielle HQ-Kette fire1 -> fire3 (two_gate_v2 a3_loose, 288x512, iters 150, metric).
# Robust gegen transienten ENOMEM beim mmcv/hipconfig-Import (vm.overcommit=2): bis 4x retry.
set -uo pipefail
cd /home/philipp/Dokumente/Github/VINGS-Mono-BA
PY=/home/philipp/anaconda3/envs/vings/bin/python

run_one () {
  local cfg="$1" log="logs/${1}_twogate_hq.log"
  for attempt in 1 2 3 4; do
    echo "[chain] $cfg attempt $attempt at $(date +%H:%M:%S)"
    PYTHONPATH=scripts "$PY" scripts/run.py "configs/local/fire/${cfg}_two_gate_v2_a3_loose.yaml" > "$log" 2>&1
    rc=$?
    if grep -q "Cannot allocate memory" "$log" && ! grep -q "Profiling Summary" "$log"; then
      echo "[chain] $cfg ENOMEM at import (rc=$rc) -> retry in 20s"; sleep 20; continue
    fi
    echo "[chain] $cfg exited rc=$rc at $(date +%H:%M:%S)"; return $rc
  done
  echo "[chain] $cfg gave up after 4 ENOMEM attempts"; return 12
}

run_one fire1
run_one fire3
echo "[chain] HQ CHAIN DONE"
