#!/usr/bin/env bash
# =============================================================================
# Master experiment sweep — runs the configured sweep variants sequentially.
# Order (per dataset): cheap-to-expensive baselines, then selectors.
#
# Modes:
#   default       full sweep (amtown03 6.2k + AGZ 10k)
#   --smoke       100-frame smoke variant for both datasets
#   --s3100       200-frame amtown03 slice starting at frame 3100 (single
#                 dataset, short selector folder names, skip values 1..20)
#
# For each run we:
#   • snapshot output dir contents BEFORE
#   • spawn `run_experiment.py <cfg>` with a wall-clock timeout
#   • collect metrics.json + profiling.json + ply scan (also on crash)
#   • append one row to a per-mode CSV (sweep_results.csv / smoke_results.csv /
#     s3100_200f_results.csv)
#   • flush page cache (best effort), wait until VRAM drops below threshold
#
# Idempotent: rows already present in the CSV with status=OK get SKIPPED on
# re-run. Pass --force to ignore the cache, or --only <substr> / --start-at
# <variant> to limit the loop.
#
# Pre-step: if AGZ frames 0..10000 are not extracted yet, prepare_agz.py is
# invoked once before the AGZ runs start. The s3100 mode skips this entirely
# since AGZ is not enqueued.
# =============================================================================

set -u
cd "$(dirname "$0")/.."
REPO="$PWD"

source /home/philipp/anaconda3/etc/profile.d/conda.sh
conda activate vings

# ── Configuration knobs ──────────────────────────────────────────────────────
CSV="${CSV:-$REPO/output/sweep_results.csv}"
TIMEOUT_PER_RUN="${TIMEOUT_PER_RUN:-21600}"        # 6h hard cap per run
VRAM_IDLE_THRESHOLD_MIB="${VRAM_IDLE_THRESHOLD_MIB:-800}"
VRAM_WAIT_MAX_SEC="${VRAM_WAIT_MAX_SEC:-90}"
RAM_FREE_MIN_GB="${RAM_FREE_MIN_GB:-2.0}"
SLEEP_BETWEEN="${SLEEP_BETWEEN:-15}"

mkdir -p "$(dirname "$CSV")"

# ── CLI flags ────────────────────────────────────────────────────────────────
FORCE=0
ONLY=""
START_AT=""
SMOKE=0
S3100=0
S1000=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --force)        FORCE=1; shift ;;
    --only)         ONLY="$2"; shift 2 ;;
    --start-at)     START_AT="$2"; shift 2 ;;
    --csv)          CSV="$2"; shift 2 ;;
    --smoke)        SMOKE=1; shift ;;
    --s3100)        S3100=1; shift ;;
    --s1000)        S1000=1; shift ;;
    -h|--help)
      sed -n '2,30p' "$0"; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

if (( SMOKE + S3100 + S1000 > 1 )); then
  echo "[sweep] --smoke / --s3100 / --s1000 are mutually exclusive" >&2
  exit 2
fi

# Mode selection ---------------------------------------------------------------
# - default :  full sweep (amtown03 6.2k + AGZ 10k)
# - --smoke :  100-frame smoke (both datasets)
# - --s3100 :  200-frame amtown03 slice starting at frame 3100. Single dataset,
#              short selector folder names (aim/coko/game/nurbs/orbslam),
#              skip values capped at 20.
if (( SMOKE )); then
  CSV="${CSV/sweep_results.csv/smoke_results.csv}"
  EXP_SUBDIR="exp_smoke"
  NAME_AMTOWN03="amtown03_smoke"
  NAME_AGZ="agz_smoke"
  SAVE_AMTOWN03="exp_amtown03_smoke"
  SAVE_AGZ="exp_agz_smoke"
  TIMEOUT_PER_RUN="${TIMEOUT_PER_RUN:-1800}"          # 30min per smoke run
  SLEEP_BETWEEN="${SLEEP_BETWEEN:-5}"
  SKIP_VALUES=(100 20 10 5 3 2 1)
  SELECTOR_DIRS=(adaptive_kf aim_slam coko_slam game_kfs mm3dgs nurbs_lvi orbslam3 vista)
  DATASETS=("amtown03|$SAVE_AMTOWN03|$NAME_AMTOWN03" "agz|$SAVE_AGZ|$NAME_AGZ")
elif (( S3100 )); then
  CSV="${CSV/sweep_results.csv/s3100_200f_results.csv}"
  EXP_SUBDIR="s3100_200f"
  NAME_AMTOWN03="amtown03_s3100_200f"
  SAVE_AMTOWN03="exp_amtown03_s3100_200f"
  TIMEOUT_PER_RUN="${TIMEOUT_PER_RUN:-3600}"          # 60min hard cap per 200f run
  SLEEP_BETWEEN="${SLEEP_BETWEEN:-10}"
  SKIP_VALUES=(20 10 5 3 2 1)
  # Short folder names per the s3100_200f layout
  SELECTOR_DIRS=(adaptive_kf aim coko game mm3dgs nurbs orbslam two_gate two_gate_v2 vista)
  DATASETS=("amtown03|$SAVE_AMTOWN03|$NAME_AMTOWN03")
elif (( S1000 )); then
  CSV="${CSV/sweep_results.csv/s1000_400f_results.csv}"
  EXP_SUBDIR="s1000_400f"
  NAME_AMTOWN03="amtown03_s1000_400f"
  SAVE_AMTOWN03="exp_amtown03_s1000_400f"
  TIMEOUT_PER_RUN="${TIMEOUT_PER_RUN:-3600}"          # 60min hard cap per 400f run
  SLEEP_BETWEEN="${SLEEP_BETWEEN:-10}"
  SKIP_VALUES=(20 10 5 3 2 1)
  SELECTOR_DIRS=(adaptive_kf aim coko game mm3dgs nurbs orbslam two_gate two_gate_v2 vista)
  DATASETS=("amtown03|$SAVE_AMTOWN03|$NAME_AMTOWN03")
else
  EXP_SUBDIR="exp"
  NAME_AMTOWN03="amtown03_full"
  NAME_AGZ="agz_10k"
  SAVE_AMTOWN03="exp_amtown03_full"
  SAVE_AGZ="exp_agz_0_10000"
  SKIP_VALUES=(100 20 10 5 3 2 1)
  SELECTOR_DIRS=(adaptive_kf aim_slam coko_slam game_kfs mm3dgs nurbs_lvi orbslam3 vista)
  DATASETS=("amtown03|$SAVE_AMTOWN03|$NAME_AMTOWN03" "agz|$SAVE_AGZ|$NAME_AGZ")
fi

# ── Run list (dataset|group|variant|config_path|save_dir) ───────────────────
# Order: VINGS baseline → cheap mapskip → expensive mapskip → cheap nofilter →
# expensive nofilter → selectors (per dataset). amtown03 first (smaller).
build_runs() {
  local DS BASE OUT
  for ds_pair in "${DATASETS[@]}"; do
    IFS='|' read -r DS OUT NAME <<< "$ds_pair"
    BASE="configs/local/$DS/$EXP_SUBDIR"
    SAVE="$REPO/output/$OUT"

    # 1) VINGS filter baseline
    echo "$DS|baseline|vings_filter|$BASE/baseline/${NAME}_vings_filter.yaml|$SAVE"

    # 2) mapskip (cheap to expensive)
    for n in "${SKIP_VALUES[@]}"; do
      echo "$DS|mapskip|mapskip_$n|$BASE/mapskip/${NAME}_mapskip_$n.yaml|$SAVE"
    done

    # 3) skip_no_filter (cheap to expensive)
    for n in "${SKIP_VALUES[@]}"; do
      echo "$DS|skip_no_filter|nofilter_skip_$n|$BASE/skip_no_filter/${NAME}_nofilter_skip_$n.yaml|$SAVE"
    done

    # 4) selectors. Pick up ALL yamls in the selector dir, default-variant
    # first, then param variants in sorted order. Variant name = filename
    # minus the leading "<NAME>_" prefix and ".yaml" suffix. Folder names
    # may be short forms (aim/coko/...) in s3100 mode -- they're treated
    # uniformly as both the directory name AND the default-variant name.
    for sel in "${SELECTOR_DIRS[@]}"; do
      local sel_dir="$BASE/$sel"
      [[ -d "$sel_dir" ]] || continue
      # default ('<NAME>_<sel>.yaml') first
      local default_cfg="$sel_dir/${NAME}_${sel}.yaml"
      if [[ -f "$default_cfg" ]]; then
        echo "$DS|$sel|$sel|$default_cfg|$SAVE"
      fi
      # then variants ('<NAME>_<sel>_<suffix>.yaml'), sorted
      while IFS= read -r cfg_path; do
        local fname=$(basename "$cfg_path" .yaml)
        local variant="${fname#${NAME}_}"
        [[ "$variant" == "$sel" ]] && continue   # already emitted default
        echo "$DS|$sel|$variant|$cfg_path|$SAVE"
      done < <(find "$sel_dir" -maxdepth 1 -name "${NAME}_${sel}_*.yaml" | sort)
    done
  done
}

# ── Helpers ──────────────────────────────────────────────────────────────────
free_vram_mib() {
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null \
    | head -n1 | tr -d '[:space:]'
}

wait_vram_idle() {
  local t0=$SECONDS
  while :; do
    local used=$(free_vram_mib)
    used=${used:-0}
    if (( used <= VRAM_IDLE_THRESHOLD_MIB )); then return 0; fi
    if (( SECONDS - t0 >= VRAM_WAIT_MAX_SEC )); then
      echo "[sweep] VRAM stuck at ${used}MiB after ${VRAM_WAIT_MAX_SEC}s — continuing anyway" >&2
      return 0
    fi
    sleep 3
  done
}

free_ram_gb() {
  # LC_ALL=C forces dot-decimals on German/locale-aware systems (otherwise
  # the comma breaks the downstream awk comparison).
  LC_ALL=C awk '/MemAvailable:/ { printf "%.2f", $2/1024/1024 }' /proc/meminfo
}

wait_ram_ok() {
  local t0=$SECONDS
  while :; do
    local avail=$(free_ram_gb)
    if LC_ALL=C awk -v a="$avail" -v m="$RAM_FREE_MIN_GB" \
         'BEGIN { exit !(a+0 >= m+0) }'; then return 0; fi
    if (( SECONDS - t0 >= 60 )); then
      echo "[sweep] RAM still low (${avail}GB) after 60s — continuing anyway" >&2
      return 0
    fi
    sleep 3
  done
}

cleanup_between_runs() {
  sync
  # Trigger Python-level GPU release in case any stale process holds it
  python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
  wait_ram_ok
  wait_vram_idle
}

already_done_ok() {
  local ds="$1" variant="$2"
  [[ -f "$CSV" ]] || return 1
  # exact-match dataset + variant + status=OK in a previous row
  awk -F',' -v ds="$ds" -v v="$variant" '
    NR==1 { for (i=1; i<=NF; i++) col[$i]=i; next }
    $col["dataset"]==ds && $col["variant"]==v && $col["status"]=="OK" { found=1; exit }
    END { exit !found }
  ' "$CSV"
}

snapshot_save_dir() {
  local save="$1" tmpfile="$2"
  : > "$tmpfile"
  [[ -d "$save" ]] || return 0
  find "$save" -maxdepth 1 -mindepth 1 -type d -printf '%f\n' > "$tmpfile"
}

map_status() {
  local rc="$1"
  case "$rc" in
    0)   echo "OK" ;;
    124) echo "TIMEOUT" ;;
    137) echo "OOM" ;;          # SIGKILL — most often the user's VRAM watchdog
    139) echo "FAIL" ;;          # SIGSEGV
    *)   echo "FAIL" ;;
  esac
}

# ── AGZ extraction (one-time) ────────────────────────────────────────────────
ensure_agz_frames() {
  local target="$HOME/Dokumente/datasets/agz/agz_0_10000/rectified"
  if [[ -d "$target/images" ]]; then
    local n=$(ls "$target/images" | wc -l)
    if (( n >= 10000 )); then
      echo "[sweep] AGZ 0..10000 already extracted ($n frames)"
      return 0
    fi
    echo "[sweep] AGZ extracted dir incomplete ($n/10000) — re-running prepare_agz"
  fi
  echo "[sweep] Extracting AGZ frames 1..10000 to $target …"
  python scripts/prepare_agz.py \
    --zip   "$HOME/Dokumente/datasets/AGZ.zip" \
    --out   "$HOME/Dokumente/datasets/agz/agz_0_10000" \
    --start 1 --count 10000 \
    --alpha 0.0
}

# ── Main loop ────────────────────────────────────────────────────────────────
RUNS=()
while IFS= read -r line; do
  [[ -n "$line" ]] && RUNS+=("$line")
done < <(build_runs)
TOTAL=${#RUNS[@]}

echo "[sweep] $TOTAL runs total. CSV → $CSV"
echo "[sweep] timeout=${TIMEOUT_PER_RUN}s, vram_idle≤${VRAM_IDLE_THRESHOLD_MIB}MiB, ram_free≥${RAM_FREE_MIN_GB}GB"

# AGZ extract only if at least one AGZ run is enqueued and not all skipped
NEED_AGZ=0
for r in "${RUNS[@]}"; do [[ "$r" == agz\|* ]] && NEED_AGZ=1 && break; done
if (( NEED_AGZ )); then ensure_agz_frames; fi

idx=0
t_sweep_start=$(date +%s)
skipped=0; ok=0; failed=0

for entry in "${RUNS[@]}"; do
  idx=$((idx + 1))
  IFS='|' read -r DS GROUP VARIANT CFG SAVE <<< "$entry"

  # filter options
  if [[ -n "$ONLY" && "$entry" != *"$ONLY"* ]]; then continue; fi
  if [[ -n "$START_AT" ]]; then
    if [[ "$VARIANT" == "$START_AT" || "$DS-$VARIANT" == "$START_AT" ]]; then
      START_AT=""
    else
      continue
    fi
  fi

  if (( FORCE == 0 )) && already_done_ok "$DS" "$VARIANT"; then
    echo "[sweep $idx/$TOTAL] SKIP $DS/$VARIANT (CSV says OK)"
    skipped=$((skipped + 1)); continue
  fi

  if [[ ! -f "$CFG" ]]; then
    echo "[sweep $idx/$TOTAL] MISSING CFG: $CFG" >&2
    failed=$((failed + 1)); continue
  fi

  echo
  echo "============================================================"
  echo " [$idx/$TOTAL] $DS / $VARIANT"
  echo " cfg: $CFG"
  echo "============================================================"

  cleanup_between_runs

  KB_FILE=$(mktemp /tmp/sweep_before_XXXXXX.txt)
  snapshot_save_dir "$SAVE" "$KB_FILE"

  t0=$(date +%s)
  set +e
  timeout --signal=TERM --kill-after=30 "${TIMEOUT_PER_RUN}s" \
    python scripts/run_experiment.py "$CFG"
  rc=$?
  set -e
  t1=$(date +%s)
  STATUS=$(map_status "$rc")

  echo "[sweep] $VARIANT done rc=$rc → $STATUS ($(( (t1 - t0) / 60 ))min)"

  python scripts/log_sweep_row.py \
    --csv          "$CSV" \
    --dataset      "$DS" \
    --group        "$GROUP" \
    --variant      "$VARIANT" \
    --config       "$CFG" \
    --save-dir     "$SAVE" \
    --known-before "$KB_FILE" \
    --start-ts     "$t0" \
    --end-ts       "$t1" \
    --exit-code    "$rc" \
    --status       "$STATUS" || \
    echo "[sweep] WARNING: log_sweep_row.py failed for $VARIANT" >&2
  rm -f "$KB_FILE"

  case "$STATUS" in
    OK) ok=$((ok + 1)) ;;
    *)  failed=$((failed + 1)) ;;
  esac

  sleep "$SLEEP_BETWEEN"
done

t_sweep_end=$(date +%s)
elapsed_min=$(( (t_sweep_end - t_sweep_start) / 60 ))
echo
echo "============================================================"
echo " sweep done — total ${elapsed_min}min"
echo "   ok      : $ok"
echo "   failed  : $failed"
echo "   skipped : $skipped"
echo "   csv     : $CSV"
echo "============================================================"
