#!/usr/bin/env bash
# Serielle Kette: fire2 (laeuft schon, PID via arg) -> fire1 -> fire3, two_gate_v2 a3_loose.
set -uo pipefail
cd /home/philipp/Dokumente/Github/VINGS-Mono-BA
PY=/home/philipp/anaconda3/envs/vings/bin/python
FIRE2_PID="${1:-}"

if [ -n "$FIRE2_PID" ]; then
  echo "[chain] waiting for fire2 two_gate pid $FIRE2_PID ..."
  until ! kill -0 "$FIRE2_PID" 2>/dev/null; do sleep 15; done
  echo "[chain] fire2 done."
fi

for cfg in fire1 fire3; do
  echo "[chain] starting $cfg two_gate at $(date +%H:%M:%S)"
  PYTHONPATH=scripts "$PY" scripts/run.py "configs/local/fire/${cfg}_two_gate_v2_a3_loose.yaml" \
      > "logs/${cfg}_twogate.log" 2>&1
  echo "[chain] $cfg exited rc=$? at $(date +%H:%M:%S)"
done
echo "[chain] ALL THREE TWO_GATE RUNS DONE"
