#!/usr/bin/env bash
# =============================================================================
# VINGS-Mono - sequentielles Benchmark-Skript fuer alle lokalen Datasets
#
# Fuer jedes Dataset:
#   1. python scripts/run.py <config> ausfuehren
#   2. Laufzeit (wall clock) messen
#   3. Peak-RAM messen (via /usr/bin/time -v)
#   4. Peak-GPU-Speicher messen (nvidia-smi polling in Hintergrund)
#   5. PLY-Output pruefen
#   6. Zusammenfassung als Tabelle ausgeben + nach /results/vings/ schreiben
#
# Nutzung:
#   cd /root/VINGS-Mono-BA
#   conda activate vings
#   bash scripts/run_all_vings.sh
#
# Umgebungsvariablen (alle optional):
#   FORCE=1           Auch bereits abgeschlossene Runs wiederholen
#   ONLY="kitti*"     Glob-Filter: nur passende Dataset-Namen
#   SKIP="polytech"   Glob-Filter: diese Namen ueberspringen
#   OUT_BASE=/results/vings   Ausgabe-Basisverzeichnis (Default)
#   GPU_POLL_MS=2000  nvidia-smi Polling-Intervall in ms (Default: 2000)
# =============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# ---- Defaults ---------------------------------------------------------------
OUT_BASE="${OUT_BASE:-/root/results/vings}"
FORCE="${FORCE:-0}"
ONLY="${ONLY:-}"
SKIP="${SKIP:-}"
GPU_POLL_MS="${GPU_POLL_MS:-2000}"

# expandable_segments ist erst ab PyTorch 2.1 verfuegbar; torch 2.0.1 wird
# ignoriert oder wirft RuntimeError -> nicht setzen.
# export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "$OUT_BASE"

# ---- Dataset-Liste ----------------------------------------------------------
# Format: NAME|CONFIG_PATH
declare -a DATASET_LIST=(
    "kitti_odom_07|configs/local/kitti_odom_07.yaml"
    "hierarchical_smallcity|configs/local/hierarchical_smallcity.yaml"
    "hierarchical_campus|configs/local/hierarchical_campus.yaml"
    "waymo_405841|configs/local/waymo_405841.yaml"
    "bonn_crowd|configs/local/bonn_crowd.yaml"
    "urbanscene_polytech|configs/local/urbanscene_polytech.yaml"
)

# ---- Helper -----------------------------------------------------------------
declare -A STATUS
declare -A WALL
declare -A PEAK_RSS_MIB
declare -A PEAK_GPU_MIB
declare -A PLY_OK
declare -A N_FRAMES

match_glob() {
    local pat="$1" name="$2"
    [ -z "$pat" ] && return 0
    case "$name" in $pat) return 0 ;; *) return 1 ;; esac
}

human_time() {
    local s="$1"
    printf "%dm %02ds" $((s/60)) $((s%60))
}

already_done() {
    # A run is "done" if a .ply file exists in any output subdir containing the name
    local name="$1"
    ls "$OUT_BASE"/*${name}*/ply/*.ply 2>/dev/null | grep -q . && return 0
    return 1
}

start_gpu_poll() {
    local outfile="$1"
    # Poll GPU memory every GPU_POLL_MS ms and write to outfile
    (
        while true; do
            nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null \
                | awk '{print $1}' >> "$outfile"
            sleep "$(echo "scale=3; $GPU_POLL_MS/1000" | bc)"
        done
    ) >/dev/null 2>/dev/null &
    echo $!
}

stop_gpu_poll() {
    local pid="$1"
    kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
}

max_from_file() {
    local f="$1"
    [ -s "$f" ] || { echo "0"; return; }
    awk 'BEGIN{m=0} {v=$1+0; if(v>m)m=v} END{print m}' "$f"
}

# Check /usr/bin/time availability
if ! /usr/bin/time --version &>/dev/null 2>&1; then
    USE_TIME_V=0
    echo "[warn] /usr/bin/time -v nicht verfuegbar - RAM-Messung entfaellt"
else
    USE_TIME_V=1
fi

# ---- Header -----------------------------------------------------------------
echo "=================================================================="
echo " VINGS-Mono Benchmark"
echo " $(date)"
echo " Out: $OUT_BASE"
echo " FORCE=$FORCE  ONLY=${ONLY:-<alle>}  SKIP=${SKIP:-<keine>}"
echo "=================================================================="
echo ""

ORDER=()
TOTAL=0

# ---- Hauptschleife ----------------------------------------------------------
for entry in "${DATASET_LIST[@]}"; do
    name="${entry%%|*}"
    config="${entry##*|}"

    # Glob-Filter
    if ! match_glob "$ONLY" "$name"; then continue; fi
    if [ -n "$SKIP" ] && match_glob "$SKIP" "$name"; then
        STATUS[$name]="SKIP(filter)"
        ORDER+=("$name"); TOTAL=$((TOTAL+1))
        continue
    fi

    ORDER+=("$name"); TOTAL=$((TOTAL+1))

    echo "------------------------------------------------------------------"
    echo "[$name]"
    echo "  config : $config"

    # Config-Datei pruefen
    if [ ! -f "$config" ]; then
        echo "  SKIP: Config nicht gefunden: $config"
        STATUS[$name]="SKIP(no-config)"
        PLY_OK[$name]="n/a"; PEAK_RSS_MIB[$name]="n/a"; PEAK_GPU_MIB[$name]="n/a"; WALL[$name]=0
        continue
    fi

    # Frame-Anzahl bestimmen
    N_FRAMES[$name]="n/a"

    # Done-Check
    if [ "$FORCE" != "1" ] && already_done "$name"; then
        echo "  SKIP: PLY bereits vorhanden (FORCE=1 erzwingt Neuberechnung)."
        STATUS[$name]="SKIP(done)"; PLY_OK[$name]="OK"
        PEAK_RSS_MIB[$name]="n/a"; PEAK_GPU_MIB[$name]="n/a"; WALL[$name]=0
        continue
    fi

    outdir_run="$OUT_BASE"
    logfile="$OUT_BASE/${name}.log"
    gpu_pollfile="$OUT_BASE/${name}_gpu_poll.tmp"
    time_file="$OUT_BASE/${name}_time.tmp"

    mkdir -p "$outdir_run"
    : > "$logfile"
    : > "$gpu_pollfile"

    echo "  log    : $logfile"
    echo "  running..."

    # GPU-Polling starten
    GPU_POLL_PID=$(start_gpu_poll "$gpu_pollfile")

    start_ts=$(date +%s)

    # Ausfuehren mit optionalem RAM-Tracking
    run_ok=0
    if [ "$USE_TIME_V" = "1" ]; then
        /usr/bin/time -v \
            conda run -n vings \
                python scripts/run.py "$config" --prefix "$name" \
            >> "$logfile" 2> "$time_file"
        run_ok=$?
        # /usr/bin/time schreibt nach stderr; wir wollen beides
        cat "$time_file" >> "$logfile"
    else
        conda run -n vings \
            python scripts/run.py "$config" --prefix "$name" \
            >> "$logfile" 2>&1
        run_ok=$?
    fi

    end_ts=$(date +%s)
    WALL[$name]=$((end_ts - start_ts))

    # GPU-Polling stoppen
    stop_gpu_poll "$GPU_POLL_PID"

    # Peak-RAM aus /usr/bin/time -v parsen
    if [ "$USE_TIME_V" = "1" ] && [ -s "$time_file" ]; then
        rss_kb=$(grep "Maximum resident set size" "$time_file" | grep -oP '\d+' | tail -1)
        if [ -n "$rss_kb" ] && [ "$rss_kb" -gt 0 ]; then
            PEAK_RSS_MIB[$name]=$(( rss_kb / 1024 ))
        else
            PEAK_RSS_MIB[$name]="n/a"
        fi
    else
        PEAK_RSS_MIB[$name]="n/a"
    fi

    # Peak-GPU aus Polling
    peak_gpu=$(max_from_file "$gpu_pollfile")
    PEAK_GPU_MIB[$name]="$peak_gpu"

    # Aufraumen
    rm -f "$gpu_pollfile" "$time_file"

    # Status und PLY pruefen
    if [ "$run_ok" -eq 0 ]; then
        STATUS[$name]="OK"
    else
        # OOM-Check
        if grep -qE "CUDA out of memory|OutOfMemoryError" "$logfile" 2>/dev/null; then
            STATUS[$name]="FAIL(OOM)"
        else
            STATUS[$name]="FAIL(rc=$run_ok)"
        fi
    fi

    # PLY pruefen
    ply_files=$(ls "$OUT_BASE"/*${name}*/ply/*.ply 2>/dev/null | wc -l)
    if [ "$ply_files" -gt 0 ]; then
        PLY_OK[$name]="OK($ply_files)"
    else
        PLY_OK[$name]="none"
    fi

    echo "  -> ${STATUS[$name]}  wall=$(human_time "${WALL[$name]}")  "\
"rss=${PEAK_RSS_MIB[$name]}MiB  gpu=${PEAK_GPU_MIB[$name]}MiB  ply=${PLY_OK[$name]}"

done

# ---- Summary ----------------------------------------------------------------
SUMMARY_FILE="$OUT_BASE/summary_$(date +%Y%m%d_%H%M%S).log"

{
    echo ""
    echo "==================== VINGS-Mono Benchmark Summary ===================="
    echo "  Datum   : $(date)"
    echo "  Datasets: $TOTAL  (ONLY=${ONLY:-alle}  SKIP=${SKIP:-keine})"
    echo ""
    printf "%-28s  %-14s  %-10s  %-10s  %-10s  %-12s\n" \
        "dataset" "status" "wall" "rss_MiB" "gpu_MiB" "ply"
    printf "%-28s  %-14s  %-10s  %-10s  %-10s  %-12s\n" \
        "----------------------------" "--------------" "----------" \
        "----------" "----------" "------------"

    ok=0; fail=0; skip=0
    total_wall=0
    for name in "${ORDER[@]}"; do
        st="${STATUS[$name]:-?}"
        w="${WALL[$name]:-0}"
        wh="$(human_time "$w")"
        rss="${PEAK_RSS_MIB[$name]:-n/a}"
        gpu="${PEAK_GPU_MIB[$name]:-n/a}"
        ply="${PLY_OK[$name]:-n/a}"

        printf "%-28s  %-14s  %-10s  %-10s  %-10s  %-12s\n" \
            "$name" "$st" "$wh" "$rss" "$gpu" "$ply"

        case "$st" in
            OK*)    ok=$((ok+1));   total_wall=$((total_wall+w)) ;;
            FAIL*)  fail=$((fail+1)) ;;
            SKIP*)  skip=$((skip+1)) ;;
        esac
    done

    echo "----------------------------------------------------------------------"
    echo "  ok=$ok  fail=$fail  skip=$skip  total=$TOTAL"
    echo "  Gesamtlaufzeit erfolgreicher Runs: $(human_time $total_wall)"
    echo ""
    echo "  Einzel-Logs  : $OUT_BASE/<dataset>.log"
    echo "  PLY-Ausgaben : $OUT_BASE/<dataset>-*/ply/"
    echo "  Diese Summary: $SUMMARY_FILE"
    echo "======================================================================"
} | tee "$SUMMARY_FILE"

# fail/ok/skip were computed inside the subshell above (piped to tee) and cannot
# be exported back. Recount from STATUS to get the correct exit code.
_fail=0
for _n in "${ORDER[@]}"; do
    case "${STATUS[$_n]:-}" in FAIL*) _fail=$((_fail+1)) ;; esac
done
[ "$_fail" -eq 0 ] || exit 1
