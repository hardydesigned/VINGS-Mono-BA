#!/usr/bin/env bash
# =============================================================================
# VINGS-Mono - vollständiges Setup-Skript
# Getestet auf: Ubuntu 22.04/24.04, WSL2, RTX A6000/3080, CUDA 11.8, GCC 11
#
# Grundlage: backup/environment.yaml + set_env.sh + backup/setenv_vo.sh.
#   Python 3.9.19 + torch 2.0.1+cu118 + torch-scatter 2.0.2 +
#   submodules/{dbaf, gtsam(vio-branch), dbef/thirdparty/lietorch,
#   dbef/thirdparty/eigen, diff-surfel-rasterization, metric_modules}
#   + Checkpoints (droid.pth, metric_depth_vit_small_800k.pth,
#   lightglue/superpoint*.onnx, optional FastSAM-x.pt).
#
# Fixes gegenüber der README / set_env.sh:
#   - Submodule werden automatisch initialisiert (git submodule update --init --recursive)
#   - gtsam wird automatisch auf den vio-Branch umgeschaltet (README-Hinweis)
#   - torch-scatter wird aus der PyG-wheels-Quelle geholt (kein PyPI-Default)
#   - CUDA 11.8 Toolkit (nvcc + Header) muss ins Env, sonst scheitern Submodul-Builds
#   - Ubuntu 24.04 bringt nur gcc-13/14; CUDA 11.8 braucht <= gcc-11
#   - HuggingFace-URLs in der README nutzen teils /blob/ statt /resolve/
#     -> liefern HTML statt Binärdatei; hier korrigiert
#   - scripts/metric/metric_model.py hat einen hardcoded sys.path
#     (/data/wuke/workspace/VINGS-Mono/submodules/) -> wird auf das Repo gepatcht
#   - lietorch & eigen sitzen unter submodules/dbef/thirdparty/ (nicht dbaf!),
#     set_env.sh verschweigt das; wir bauen lietorch dort explizit
#
# ISOLATION - was dieses Skript am System ändert vs. was im Env bleibt:
#   System (sudo apt-get, NUR falls fehlend, nur additiv, kein Entfernen):
#     - libgl1, libglib2.0-0          (open3d/opencv Laufzeit-ABI)
#     - unzip, wget                   (Download-Helfer)
#     - gcc-11 / g++-11               (CUDA 11.8 Host-Compiler, nur falls kein GCC<=11 vorhanden)
#   Alles andere bleibt im conda-Env "$ENV_NAME":
#     - cmake, boost, CUDA-toolkit    via conda-forge / nvidia-channel -> Env
#     - alle pip-Packages             via "conda run -n $ENV_NAME pip" -> Env
#     - CUDA-Submodul-Builds          cc/cxx nur als subprocess-env-vars, NIE exportiert
#     - gtsam cmake CMAKE_INSTALL_PREFIX  -> Env-Prefix, nie System-Prefix
#     - BOOST_ROOT / EIGEN3_ROOT_DIR  -> Env-Prefix, verhindert Griff ins System-Boost
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

ENV_NAME="vings"
PY_VER="3.9.19"
TORCH_VER="2.0.1"
TORCHVISION_VER="0.15.2"
TORCHAUDIO_VER="2.0.2"
TORCH_SCATTER_VER="2.1.2"
CUDA_LABEL="11.8"

HF_BASE="https://huggingface.co/Promethe-us/VINGS-Mono-Checkpoints/resolve/main"
CKPT_DIR="$SCRIPT_DIR/ckpts"

log()  { echo -e "\n\033[1;34m[setup] $*\033[0m"; }
warn() { echo -e "\n\033[1;33m[warn]  $*\033[0m"; }
err()  { echo -e "\n\033[1;31m[ERROR] $*\033[0m" >&2; exit 1; }

# ---------------------------------------------------------------------------
# 0. Voraussetzungen prüfen + GCC <= 11 sicherstellen (CUDA 11.8)
# ---------------------------------------------------------------------------
log "Prüfe Voraussetzungen..."
command -v conda &>/dev/null || err "conda nicht gefunden. Bitte Miniconda/Anaconda installieren."
command -v git   &>/dev/null || err "git nicht gefunden."
command -v cmake &>/dev/null || warn "cmake fehlt - wird im conda-env nachinstalliert."

# System-Libs: NUR das absolute Minimum, NUR additiv (kein Entfernen).
# boost/tbb/cmake kommen NICHT von hier - die kommen aus dem conda-Env (s.u.).
# --no-install-recommends verhindert transitive System-Pakete.
if command -v apt-get &>/dev/null; then
    MISSING_APT=()
    ldconfig -p | grep -q "libGL\.so\.1"        || MISSING_APT+=("libgl1")
    ldconfig -p | grep -q "libglib-2\.0\.so\.0" || MISSING_APT+=("libglib2.0-0")
    command -v unzip &>/dev/null                 || MISSING_APT+=("unzip")
    command -v wget  &>/dev/null                 || MISSING_APT+=("wget")
    if [ ${#MISSING_APT[@]} -gt 0 ]; then
        echo "  Installiere fehlende System-Pakete (additiv): ${MISSING_APT[*]}"
        sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
            --no-install-recommends "${MISSING_APT[@]}" \
            || warn "Einige apt-Pakete konnten nicht installiert werden - ggf. manuell nachinstallieren."
    fi
fi

# GCC 11 finden (CUDA 11.8 ist offiziell inkompatibel mit GCC 12+).
find_gcc_for_cuda118() {
    for candidate in gcc-11 gcc-10 gcc; do
        if command -v "$candidate" &>/dev/null; then
            local ver
            ver=$("$candidate" -dumpversion | cut -d. -f1)
            if [ "$ver" -le 11 ]; then
                echo "$candidate"; return 0
            fi
        fi
    done
    return 1
}

GCC_BIN=$(find_gcc_for_cuda118 || true)
ALLOW_UNSUPPORTED=0
if [ -z "$GCC_BIN" ]; then
    echo "  GCC <= 11 nicht gefunden. Versuche gcc-11 zu installieren..."
    if command -v apt-get &>/dev/null; then
        if sudo DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends gcc-11 g++-11 2>/dev/null; then
            GCC_BIN="gcc-11"
        else
            warn "gcc-11 nicht installierbar - fallback auf system-gcc mit --allow-unsupported-compiler."
            GCC_BIN="$(command -v gcc)"
            ALLOW_UNSUPPORTED=1
        fi
    else
        warn "Kein apt-get - versuche trotzdem mit system-gcc (--allow-unsupported-compiler)."
        GCC_BIN="$(command -v gcc || true)"
        [ -n "$GCC_BIN" ] || err "Kein gcc gefunden. Bitte manuell installieren."
        ALLOW_UNSUPPORTED=1
    fi
fi

GXX_BIN="${GCC_BIN/gcc/g++}"
command -v "$GXX_BIN" &>/dev/null || GXX_BIN="g++"
command -v "$GXX_BIN" &>/dev/null || err "Kein passender g++ gefunden (erwartet: $GXX_BIN)."

GCC_VER=$("$GCC_BIN" -dumpversion | cut -d. -f1)
log "Nutze GCC $GCC_VER ($GCC_BIN / $GXX_BIN) für CUDA-Extension-Builds (allow_unsupported=$ALLOW_UNSUPPORTED)."

NVCC_EXTRA_FLAGS=""
[ "$ALLOW_UNSUPPORTED" = "1" ] && NVCC_EXTRA_FLAGS="-allow-unsupported-compiler"

# ---------------------------------------------------------------------------
# 1. Submodule initialisieren + gtsam auf vio-Branch
# ---------------------------------------------------------------------------
log "Initialisiere git-Submodule (rekursiv)..."
git submodule update --init --recursive

# README:
#   "For vio settings, you can use this branch of gtsam:
#    https://github.com/Promethe-us/gtsam/tree/vio."
# Ohne den Branch-Switch schlägt der vio-Pfad (ImuFactor, ConstantBias, ...) fehl.
log "Setze submodules/gtsam auf den 'vio'-Branch..."
(
    cd submodules/gtsam
    if git rev-parse --verify --quiet vio >/dev/null; then
        git checkout vio
    else
        git fetch origin vio:vio || warn "Konnte Branch 'vio' nicht fetchen - bitte manuell prüfen."
        git checkout vio || warn "Konnte nicht auf Branch 'vio' wechseln."
    fi
    git submodule update --init --recursive
)

# ---------------------------------------------------------------------------
# 2. Conda-Umgebung erstellen
# ---------------------------------------------------------------------------
log "Akzeptiere Anaconda Channel-ToS (main, r)..."
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r    || true

log "Erstelle conda-Umgebung '$ENV_NAME' (Python $PY_VER)..."
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "  Umgebung '$ENV_NAME' existiert bereits - überspringe conda create."
else
    conda create -n "$ENV_NAME" "python=$PY_VER" -y
fi

# ---------------------------------------------------------------------------
# 3. PyTorch 2.0.1 + cu118
# ---------------------------------------------------------------------------
log "Installiere PyTorch $TORCH_VER+cu118..."
if conda run -n "$ENV_NAME" python -c "import torch,sys; sys.exit(0 if torch.__version__.startswith('$TORCH_VER') else 1)" &>/dev/null; then
    echo "  torch $TORCH_VER bereits installiert - überspringe."
else
    conda run -n "$ENV_NAME" pip install \
        --retries 10 --timeout 120 \
        "torch==${TORCH_VER}" \
        "torchvision==${TORCHVISION_VER}" \
        "torchaudio==${TORCHAUDIO_VER}" \
        --index-url https://download.pytorch.org/whl/cu118
fi

# Sanity-Check
conda run -n "$ENV_NAME" python -c "
import torch
print('torch:', torch.__version__)
print('cuda available:', torch.cuda.is_available())
print('torch.version.cuda:', torch.version.cuda)
assert torch.version.cuda.startswith('11.8'), 'CUDA version mismatch!'
print('PyTorch-Prüfung OK')
"

# ---------------------------------------------------------------------------
# 4. CUDA Toolkit 11.8 (nvcc + Header) für Submodul-Builds
# ---------------------------------------------------------------------------
log "Installiere CUDA Toolkit $CUDA_LABEL (nvcc + Header für Submodul-Builds)..."
if conda run -n "$ENV_NAME" bash -c 'command -v nvcc' &>/dev/null; then
    NVCC_VER=$(conda run -n "$ENV_NAME" nvcc --version 2>/dev/null | grep -oP 'release \K[0-9]+\.[0-9]+' | head -1)
    if [ "$NVCC_VER" = "11.8" ]; then
        echo "  nvcc $NVCC_VER bereits im Env - überspringe cuda-toolkit install."
    else
        echo "  nvcc $NVCC_VER installiert, benötigt wird 11.8 - überschreibe..."
        conda install -n "$ENV_NAME" \
            -c "nvidia/label/cuda-${CUDA_LABEL}.0" \
            cuda-toolkit cuda-nvcc -y
    fi
else
    conda install -n "$ENV_NAME" \
        -c "nvidia/label/cuda-${CUDA_LABEL}.0" \
        cuda-toolkit cuda-nvcc -y
fi

# cmake für gtsam-Build (falls system-cmake zu alt ist)
log "Stelle cmake >= 3.22 im Env bereit (für gtsam-Build)..."
conda install -n "$ENV_NAME" -c conda-forge -y "cmake>=3.22" boost

# ---------------------------------------------------------------------------
# 5. torch-scatter (aus PyG-wheels für torch 2.0.x + cu118)
# ---------------------------------------------------------------------------
# set_env.sh pinnt 2.0.2, aber PyG hostet für torch 2.0.x NUR 2.1.1 / 2.1.2
# als fertige cp39-Wheels. 2.0.2 hat keinen pre-built Wheel für torch>=2.0 ->
# würde aus Source gebaut, schlägt wegen isolation fehl.
# 2.1.2 ist vollständig API-kompatibel (scatter/gather ops unverändert).
log "Installiere torch-scatter $TORCH_SCATTER_VER (torch-2.0.x/cu118 Wheel)..."
if conda run -n "$ENV_NAME" python -c "import torch_scatter" &>/dev/null; then
    echo "  torch_scatter bereits installiert - überspringe."
else
    conda run -n "$ENV_NAME" pip install --retries 10 --timeout 120 \
        "torch-scatter==${TORCH_SCATTER_VER}" \
        -f "https://data.pyg.org/whl/torch-2.0.1+cu118.html"
fi

# ---------------------------------------------------------------------------
# 6. Pip-Abhängigkeiten aus requirements.txt
# ---------------------------------------------------------------------------
log "Installiere pip-Deps aus requirements.txt..."
REQ_FILE="$(mktemp -t vings_req.XXXXXX.txt)"
trap 'rm -f "$REQ_FILE"' EXIT

# Herausgefilterte Pakete werden separat behandelt:
#   submodules/diff-surfel-rasterization  -> Block 8 (--no-build-isolation + GCC-Flags)
#   triton==2.0.0                         -> kommt bereits als Dep von torch 2.0.1 mit
#   mmcv==1.7.2                           -> braucht --no-build-isolation (setup.py nutzt
#                                            pkg_resources, das im pip-Isolation-Env fehlt)
#   mmengine==0.10.5                      -> wird nach mmcv installiert (Reihenfolge zählt)
grep -v -E "^(submodules/diff-surfel-rasterization|triton==|mmcv==|mmengine==)" \
    requirements.txt > "$REQ_FILE" || true

conda run -n "$ENV_NAME" pip install --retries 10 --timeout 120 -r "$REQ_FILE"

# mmcv 1.7.2 aus Source mit --no-build-isolation (macht pkg_resources aus dem Env sichtbar)
log "Installiere mmcv==1.7.2 (--no-build-isolation)..."
if conda run -n "$ENV_NAME" python -c "import mmcv; assert mmcv.__version__ == '1.7.2'" &>/dev/null; then
    echo "  mmcv 1.7.2 bereits installiert - überspringe."
else
    conda run -n "$ENV_NAME" pip install --retries 5 --timeout 300 \
        --no-build-isolation "mmcv==1.7.2"
fi

log "Installiere mmengine==0.10.5..."
if conda run -n "$ENV_NAME" python -c "import mmengine" &>/dev/null; then
    echo "  mmengine bereits installiert - überspringe."
else
    conda run -n "$ENV_NAME" pip install --retries 10 --timeout 120 "mmengine==0.10.5"
fi

# ---------------------------------------------------------------------------
# 7. GPU Compute Capability für saubere CUDA-Arch-Liste
# ---------------------------------------------------------------------------
GPU_ARCH=$(conda run -n "$ENV_NAME" python -c "
import torch
if torch.cuda.is_available():
    cc = torch.cuda.get_device_capability(0)
    print(f'{cc[0]}.{cc[1]}')
else:
    print('8.0')
" 2>/dev/null | tr -d '[:space:]')
[ -z "$GPU_ARCH" ] && GPU_ARCH="8.0"
log "GPU Compute Capability: $GPU_ARCH (TORCH_CUDA_ARCH_LIST wird darauf gepinnt)"

# Helper für editable pip-installs mit richtigem Host-Compiler
build_ext() { # build_ext <pfad> <import-check>
    local dir="$1" import_check="$2"
    if conda run -n "$ENV_NAME" python -c "import torch; $import_check" &>/dev/null; then
        echo "  $dir bereits gebaut - überspringe."
        return 0
    fi
    echo "  Baue $dir (TORCH_CUDA_ARCH_LIST=$GPU_ARCH, MAX_JOBS=2)..."
    find "$dir" -maxdepth 4 -type d \( -name "build" -o -name "*.egg-info" \) \
        -exec rm -rf {} + 2>/dev/null || true
    conda run -n "$ENV_NAME" \
        env CC="$GCC_BIN" CXX="$GXX_BIN" \
            NVCC_APPEND_FLAGS="$NVCC_EXTRA_FLAGS" \
            TORCH_CUDA_ARCH_LIST="$GPU_ARCH" \
            MAX_JOBS=2 \
        pip install -e "$dir" --no-build-isolation
}

# ---------------------------------------------------------------------------
# 8. diff-surfel-rasterization (2DGS CUDA-Kernel)
# ---------------------------------------------------------------------------
log "Baue submodules/diff-surfel-rasterization (GCC=$GCC_BIN)..."
build_ext "submodules/diff-surfel-rasterization" "from diff_surfel_rasterization import _C"

# ---------------------------------------------------------------------------
# 9. lietorch (unter submodules/dbef/thirdparty/lietorch)
# ---------------------------------------------------------------------------
# Das .gitmodules-File listet lietorch + eigen UNTER submodules/dbef/thirdparty/
# - nicht unter submodules/dbaf wie set_env.sh suggeriert. Eigen wird von lietorch
# beim Build als Include-Header gebraucht (muss vor setup.py install da sein).
LIETORCH_DIR="submodules/dbef/thirdparty/lietorch"
EIGEN_DIR="submodules/dbef/thirdparty/eigen"

if [ ! -f "$LIETORCH_DIR/setup.py" ]; then
    err "lietorch setup.py fehlt unter $LIETORCH_DIR - Submodule-Init prüfen."
fi
if [ ! -d "$EIGEN_DIR" ] || [ -z "$(ls -A "$EIGEN_DIR" 2>/dev/null)" ]; then
    warn "Eigen-Header fehlen unter $EIGEN_DIR - lietorch-Build kann scheitern."
fi

build_ext "$LIETORCH_DIR" "import lietorch; from lietorch import SE3"

# ---------------------------------------------------------------------------
# 10. dbaf / droid_backends (DROID-SLAM + DBA-Fusion CUDA-Kernel)
# ---------------------------------------------------------------------------
# dbaf/setup.py hat ZWEI setup()-Aufrufe: einen für droid_backends und einen für
# lietorch. pip kann damit nicht umgehen (metadata-generation-failed), da pip bei
# der egg_info-Phase nur einen setup()-Aufruf erwartet.
# Fix: zweiten setup()-Aufruf (lietorch) aus dbaf/setup.py entfernen.
# lietorch kommt ohnehin aus Step 9 (submodules/dbef/thirdparty/lietorch, v0.3).
log "Patche submodules/dbaf/setup.py (entferne doppelten setup()-Aufruf für lietorch)..."
conda run -n "$ENV_NAME" python3 - <<'PYEOF'
import pathlib
p = pathlib.Path("submodules/dbaf/setup.py")
src = p.read_text()
first_idx = src.index('setup(')
try:
    second_idx = src.index('setup(', first_idx + 1)
    line_start = src.rfind('\n', 0, second_idx) + 1
    patched = src[:line_start].rstrip() + '\n'
    p.write_text(patched)
    print("  Patch angewendet: zweiter setup()-Aufruf fuer lietorch entfernt.")
except ValueError:
    print("  Bereits gepatcht (kein zweiter setup()-Aufruf gefunden).")
PYEOF

log "Baue submodules/dbaf (droid_backends)..."
build_ext "submodules/dbaf" "import droid_backends"

# ---------------------------------------------------------------------------
# 11. metric_modules (Metric3D Wrapper) + Patch für hardcoded sys.path
# ---------------------------------------------------------------------------
log "Baue / installiere submodules/metric_modules..."
METRIC_DIR="submodules/metric_modules"
if [ -f "$METRIC_DIR/setup.py" ] || [ -f "$METRIC_DIR/pyproject.toml" ]; then
    build_ext "$METRIC_DIR" "import metric_modules"
else
    # Kein setup.py -> als reines Python-Package via sys.path einbinden.
    # Wir patchen unten den hardcoded sys.path in scripts/metric/metric_model.py,
    # sodass das submodules/-Verzeichnis relativ zum Repo resolved wird.
    echo "  $METRIC_DIR hat kein setup.py - wird per sys.path-Patch eingebunden."
fi

# Patch: scripts/metric/metric_model.py hat eine hardcoded Zeile
#   sys.path.append('/data/wuke/workspace/VINGS-Mono/submodules/')
# -> durch repo-relativen Pfad ersetzen (idempotent).
METRIC_PY="scripts/metric/metric_model.py"
if grep -q "/data/wuke/workspace/VINGS-Mono/submodules/" "$METRIC_PY"; then
    log "Patche hardcoded sys.path in $METRIC_PY..."
    conda run -n "$ENV_NAME" python3 - <<PYEOF
import pathlib, re
p = pathlib.Path("$METRIC_PY")
src = p.read_text()
# Ersetze die absolute sys.path.append-Zeile durch einen relativen Resolver
new = re.sub(
    r"sys\.path\.append\('/data/wuke/workspace/VINGS-Mono/submodules/'\)",
    "import os; sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'submodules'))",
    src,
)
if new != src:
    p.write_text(new)
    print("  Patch angewendet.")
else:
    print("  Pattern nicht gefunden - manuell prüfen!")
PYEOF
fi

# ---------------------------------------------------------------------------
# 12. GTSAM (vio-Branch) mit Python-Bindings bauen
# ---------------------------------------------------------------------------
# Baut nach build_gtsam/ und installiert die python-bindings ins aktive Env.
# Die python-bindings brauchen "pyparsing" (bereits über requirements drin) +
# pybind11 (in requirements.txt als Dep von gtsam transitive mitgezogen).
log "Baue gtsam mit Python-Bindings (vio-Branch, GCC=$GCC_BIN)..."

# Boost 1.85+ fügt std::optional-Serialisierung zu boost/serialization/optional.hpp
# hinzu. gtsam's std_optional_serialization.h definiert dieselben Templates nochmals
# -> Redefinition-Fehler. Fix: gtsam-Definition hinter BOOST_VERSION < 108500 gaten.
GTSAM_OPTIONAL_SER="submodules/gtsam/gtsam/base/std_optional_serialization.h"
if grep -q "BOOST_VERSION < 108500" "$GTSAM_OPTIONAL_SER"; then
    echo "  Boost-1.85-Patch bereits angewendet."
else
    log "Patche $GTSAM_OPTIONAL_SER (Boost 1.85 std::optional Konflikt)..."
    conda run -n "$ENV_NAME" python3 - <<'PYEOF'
import pathlib, re
p = pathlib.Path("submodules/gtsam/gtsam/base/std_optional_serialization.h")
src = p.read_text()
# Wrap the three function templates with BOOST_VERSION guard
old = (
    "template <class Archive, class T>\nvoid save(Archive& ar, const std::optional<T>& t"
)
if old in src:
    # Find the region from first 'template <class Archive, class T>' before save()
    # to the closing '}  // namespace serialization'
    ns_start = src.index("namespace boost {\nnamespace serialization {")
    ns_end   = src.index("}  // namespace serialization\n}  // namespace boost", ns_start)

    func_start = src.index("template <class Archive, class T>\nvoid save", ns_start)

    before = src[:func_start]
    funcs  = src[func_start:ns_end]
    after  = src[ns_end:]

    guarded = (
        "#if BOOST_VERSION < 108500\n// Boost 1.85+ ships its own std::optional serialization\n\n"
        + funcs.rstrip()
        + "\n\n#endif  // BOOST_VERSION < 108500\n\n"
    )
    p.write_text(before + guarded + after)
    print("  Patch angewendet.")
else:
    print("  Pattern nicht gefunden - manuell prüfen!")
PYEOF
fi

if conda run -n "$ENV_NAME" python -c "import torch; import gtsam; from gtsam import ImuFactor" &>/dev/null; then
    echo "  gtsam bereits mit vio-Bindings installiert - überspringe Build."
else
    ENV_PREFIX=$(conda run -n "$ENV_NAME" python -c "import sys; print(sys.prefix)")
    ENV_PY=$(conda run -n "$ENV_NAME" python -c "import sys; print(sys.executable)")

    conda run -n "$ENV_NAME" pip install --retries 10 --timeout 120 \
        "pybind11-stubgen" "pyparsing" "pybind11" || true

    rm -rf build_gtsam

    # BOOST_ROOT + CMAKE_PREFIX_PATH -> Env-Prefix:
    # Verhindert, dass cmake das System-Boost (/usr/lib) oder System-Eigen nimmt.
    # Alle gesetzten env-Vars sind nur für diesen subprocess, nie exportiert.
    BOOST_ROOT_ENV="${ENV_PREFIX}"
    CMAKE_PREFIX_ENV="${ENV_PREFIX}"

    conda run -n "$ENV_NAME" \
        env CC="$GCC_BIN" CXX="$GXX_BIN" \
            BOOST_ROOT="$BOOST_ROOT_ENV" \
            CMAKE_PREFIX_PATH="$CMAKE_PREFIX_ENV" \
        cmake -S submodules/gtsam -B build_gtsam \
            -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
            -DGTSAM_BUILD_PYTHON=1 \
            -DGTSAM_PYTHON_VERSION="${PY_VER%.*}" \
            -DPYTHON_EXECUTABLE="$ENV_PY" \
            -DCMAKE_INSTALL_PREFIX="$ENV_PREFIX" \
            -DCMAKE_PREFIX_PATH="$CMAKE_PREFIX_ENV" \
            -DCMAKE_BUILD_TYPE=RelWithDebInfo \
            -DGTSAM_BUILD_TESTS=OFF \
            -DGTSAM_BUILD_EXAMPLES_ALWAYS=OFF \
            -DGTSAM_BUILD_UNSTABLE=ON \
            -DGTSAM_USE_SYSTEM_EIGEN=OFF \
            -DGTSAM_WITH_TBB=OFF
    conda run -n "$ENV_NAME" \
        env CC="$GCC_BIN" CXX="$GXX_BIN" \
        cmake --build build_gtsam --config RelWithDebInfo -j"$(nproc)"
    # python-install nutzt PYTHON_EXECUTABLE aus dem configure-Schritt -> Env
    conda run -n "$ENV_NAME" \
        env CC="$GCC_BIN" CXX="$GXX_BIN" \
        cmake --build build_gtsam --target python-install
fi

# ---------------------------------------------------------------------------
# 13. Checkpoints herunterladen (HuggingFace)
# ---------------------------------------------------------------------------
log "Lade Checkpoints nach $CKPT_DIR ..."
mkdir -p "$CKPT_DIR" "$CKPT_DIR/lightglue"

dl() { # dl <url> <dest>
    local url="$1" dest="$2"
    if [ -s "$dest" ]; then
        echo "  $(basename "$dest") bereits vorhanden - überspringe."
        return 0
    fi
    echo "  lade $(basename "$dest") von $url ..."
    if command -v wget &>/dev/null; then
        wget -q --show-progress -O "$dest" "$url" || { rm -f "$dest"; return 1; }
    else
        curl -Lf --progress-bar -o "$dest" "$url" || { rm -f "$dest"; return 1; }
    fi
}

dl "$HF_BASE/droid.pth"                           "$CKPT_DIR/droid.pth" \
    || warn "Download droid.pth fehlgeschlagen - bitte manuell von $HF_BASE/droid.pth laden."
dl "$HF_BASE/metric_depth_vit_small_800k.pth"     "$CKPT_DIR/metric_depth_vit_small_800k.pth" \
    || warn "Download metric_depth_vit_small_800k.pth fehlgeschlagen."
dl "$HF_BASE/superpoint.onnx"                     "$CKPT_DIR/lightglue/superpoint.onnx" \
    || warn "Download superpoint.onnx fehlgeschlagen."
dl "$HF_BASE/superpoint_lightglue.onnx"           "$CKPT_DIR/lightglue/superpoint_lightglue.onnx" \
    || warn "Download superpoint_lightglue.onnx fehlgeschlagen."

# Optional: FastSAM (nur gebraucht wenn use_fastsam/dynamic-Pfade aktiviert sind)
if [ "${WITH_FASTSAM:-0}" = "1" ]; then
    dl "$HF_BASE/FastSAM-x.pt"                    "$CKPT_DIR/FastSAM-x.pt" \
        || warn "Download FastSAM-x.pt fehlgeschlagen."
else
    echo "  (optional) FastSAM-x.pt übersprungen - für Download 'WITH_FASTSAM=1 bash setup.sh' neu starten."
fi

# ---------------------------------------------------------------------------
# 14. Abschluss-Check aller kritischen Imports
# ---------------------------------------------------------------------------
log "Abschluss-Check aller kritischen Imports..."
conda run -n "$ENV_NAME" python - <<'PYEOF'
import torch
print(f"  torch          {torch.__version__}  (cuda={torch.version.cuda}, available={torch.cuda.is_available()})")
import torchvision, torchaudio
print(f"  torchvision    {torchvision.__version__}")
print(f"  torchaudio     {torchaudio.__version__}")
import torch_scatter
print(f"  torch_scatter  {torch_scatter.__version__}")

import numpy, scipy, PIL, cv2, tqdm, matplotlib, yaml
import open3d, plyfile, trimesh, imageio, lpips, kornia
print(f"  numpy          {numpy.__version__}")
print(f"  opencv         {cv2.__version__}")
print(f"  open3d         {open3d.__version__}")
print(f"  kornia         {kornia.__version__}")

from diff_surfel_rasterization import _C as _dsr_C
print("  diff_surfel_rasterization._C        OK")

import lietorch
from lietorch import SE3
print("  lietorch / SE3                      OK")

import droid_backends
print("  droid_backends                      OK")

import gtsam
from gtsam import ImuFactor, Pose3, Rot3, PriorFactorPose3
print(f"  gtsam          {gtsam.__version__ if hasattr(gtsam, '__version__') else '(no __version__)'}  (ImuFactor OK)")

import onnx, onnxruntime
print(f"  onnx           {onnx.__version__}")
print(f"  onnxruntime    {onnxruntime.__version__}")

print("\nAlle Imports erfolgreich.")
PYEOF

log "Setup abgeschlossen!"
echo ""
echo "  Umgebung aktivieren:  conda activate $ENV_NAME"
echo ""
echo "  Inferenz starten (Beispiele):"
echo "    python scripts/run.py configs/hierarchical/smallcity.yaml"
echo "    python scripts/run.py configs/rtg/hotel.yaml"
echo "    python scripts/run.py configs/kitti/sync/kitti_2011_09_30_drive_0028.yaml"
echo "    python scripts/run.py configs/kitti360/unsync/kitti360_2013_05_28_drive_0002.yaml"
echo ""
echo "  Vorher in den *.yaml configs setzen: dataset.root, output.save_dir,"
echo "  frontend.weight (-> $CKPT_DIR/droid.pth)."
