#!/bin/bash
# ============================================================================
# setup_linux.sh — Enhanced Linux installer for TRELLIS.2-stableprojectorz
#
# This replaces the Windows-focused install.py with a proper Linux setup.
# It builds CUDA packages from source instead of using Windows-only wheels.
#
# Prerequisites:
#   - NVIDIA GPU with CUDA support
#   - CUDA Toolkit 12.x (nvcc on PATH or CUDA_HOME set)
#   - Python 3.10+ (3.10 or 3.11 recommended)
#   - C++ compiler (g++ >= 9)
#   - libjpeg-dev, libopenexr-dev
#
# Quick start (full install):
#   bash setup_linux.sh --all
#
# Quick start (conda new env + full install):
#   bash setup_linux.sh --new-env --all
# ============================================================================

set -euo pipefail

# --- Color helpers ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

info()  { echo -e "${BLUE}[INFO]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; }
ok()    { echo -e "${GREEN}[OK]${NC} $*"; }

# --- Defaults ---
HELP=false
NEW_ENV=false
BASIC=false
FLASHATTN=false
CUMESH=false
OVOXEL=false
FLEXGEMM=false
NVDIFFRAST=false
NVDIFFREC=false
ALL=false
SKIP_SYSTEM=false
SKIP_MODELS=false
CUDA_VERSION="12.4"
TORCH_VERSION="2.6.0"
ERROR_FLAG=false
WORKDIR="$(cd "$(dirname "$0")" && pwd)"

# --- Parse Arguments ---
TEMP=$(getopt -o h --long help,new-env,basic,flash-attn,cumesh,o-voxel,flexgemm,nvdiffrast,nvdiffrec,all,skip-system-deps,skip-models,cuda-version:,torch-version: -n 'setup_linux.sh' -- "$@" 2>/dev/null || true)

if [ -n "$TEMP" ]; then
    eval set -- "$TEMP"
fi

if [ "$#" -eq 1 ]; then
    HELP=true
fi

while true; do
    case "$1" in
        -h|--help)       HELP=true; shift ;;
        --new-env)       NEW_ENV=true; shift ;;
        --basic)         BASIC=true; shift ;;
        --flash-attn)    FLASHATTN=true; shift ;;
        --cumesh)        CUMESH=true; shift ;;
        --o-voxel)       OVOXEL=true; shift ;;
        --flexgemm)      FLEXGEMM=true; shift ;;
        --nvdiffrast)    NVDIFFRAST=true; shift ;;
        --nvdiffrec)     NVDIFFREC=true; shift ;;
        --all)           ALL=true; shift ;;
        --skip-system-deps) SKIP_SYSTEM=true; shift ;;
        --skip-models)   SKIP_MODELS=true; shift ;;
        --cuda-version)  CUDA_VERSION="$2"; shift 2 ;;
        --torch-version) TORCH_VERSION="$2"; shift 2 ;;
        --)              shift; break ;;
        *)               ERROR_FLAG=true; break ;;
    esac
done

if [ "$ERROR_FLAG" = true ]; then
    error "Invalid argument"
    HELP=true
fi

if [ "$HELP" = true ]; then
    echo "Usage: setup_linux.sh [OPTIONS]"
    echo ""
    echo "Linux installer for TRELLIS.2-stableprojectorz"
    echo ""
    echo "Options:"
    echo "  -h, --help              Display this help message"
    echo "  --new-env               Create a new conda environment (trellis2)"
    echo "  --basic                 Install basic Python dependencies"
    echo "  --flash-attn            Install flash-attention (requires Ampere+ GPU)"
    echo "  --cumesh                Install CuMesh (builds from source)"
    echo "  --o-voxel               Install o-voxel (builds from source)"
    echo "  --flexgemm              Install FlexGEMM (builds from source)"
    echo "  --nvdiffrast            Install nvdiffrast (builds from source)"
    echo "  --nvdiffrec             Install nvdiffrec (builds from source)"
    echo "  --all                   Install everything (equivalent to --basic --flash-attn"
    echo "                          --cumesh --o-voxel --flexgemm --nvdiffrast --nvdiffrec)"
    echo "  --skip-system-deps      Skip installing system packages (apt)"
    echo "  --skip-models           Skip downloading model weights"
    echo "  --cuda-version VERSION  CUDA version for PyTorch (default: 12.4)"
    echo "  --torch-version VERSION PyTorch version (default: 2.6.0)"
    echo ""
    echo "Examples:"
    echo "  # Full install with new conda env:"
    echo "  bash setup_linux.sh --new-env --all"
    echo ""
    echo "  # Full install into current environment:"
    echo "  bash setup_linux.sh --all"
    echo ""
    echo "  # Custom CUDA/PyTorch versions:"
    echo "  bash setup_linux.sh --all --cuda-version 12.8 --torch-version 2.8.0"
    exit 0
fi

# --- If --all, enable all flags ---
if [ "$ALL" = true ]; then
    BASIC=true
    FLASHATTN=true
    CUMESH=true
    OVOXEL=true
    FLEXGEMM=true
    NVDIFFRAST=true
    NVDIFFREC=true
fi

# ============================================================================
# Pre-flight Checks
# ============================================================================
echo ""
echo "============================================================"
echo " TRELLIS.2-stableprojectorz — Linux Installer"
echo "============================================================"
echo ""

# Check OS
if [ "$(uname -s)" != "Linux" ]; then
    warn "This script is designed for Linux. You are on $(uname -s)."
    warn "For Windows, use install.py instead."
fi

# Detect GPU platform
if command -v nvidia-smi > /dev/null 2>&1; then
    PLATFORM="cuda"
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "NVIDIA GPU")
    ok "NVIDIA GPU detected: $GPU_NAME"
elif command -v rocminfo > /dev/null 2>&1; then
    PLATFORM="hip"
    warn "AMD GPU detected (ROCm). Some CUDA-only packages won't be available."
else
    warn "No supported GPU detected. CUDA packages will fail to build."
    warn "You can still install basic dependencies with --basic."
    PLATFORM="unknown"
fi

# Check Python
PYTHON_CMD=""
for cmd in python3 python; do
    if command -v "$cmd" > /dev/null 2>&1; then
        PY_VERSION=$($cmd -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
        PY_MAJOR=$($cmd -c "import sys; print(sys.version_info.major)")
        PY_MINOR=$($cmd -c "import sys; print(sys.version_info.minor)")
        if [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -ge 10 ] && [ "$PY_MINOR" -lt 13 ]; then
            PYTHON_CMD="$cmd"
            ok "Python $PY_VERSION found ($cmd)"
            break
        fi
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    error "Python 3.10, 3.11, or 3.12 is required but not found."
    error "Install it or use --new-env to create a conda environment."
    exit 1
fi

# Check CUDA Toolkit (needed for building extensions)
if [ "$PLATFORM" = "cuda" ]; then
    if command -v nvcc > /dev/null 2>&1; then
        NVCC_VERSION=$(nvcc --version | grep -oP 'release \K[\d.]+' || echo "unknown")
        ok "CUDA Toolkit found: version $NVCC_VERSION"
    else
        CUDA_HOME="${CUDA_HOME:-${CUDA_PATH:-/usr/local/cuda}}"
        if [ -x "$CUDA_HOME/bin/nvcc" ]; then
            export PATH="$CUDA_HOME/bin:$PATH"
            ok "CUDA Toolkit found at CUDA_HOME=$CUDA_HOME"
        else
            warn "CUDA Toolkit (nvcc) not found on PATH."
            warn "Set CUDA_HOME or add nvcc to PATH before building CUDA packages."
            warn "Download from: https://developer.nvidia.com/cuda-toolkit-archive"
        fi
    fi
fi

# Check C++ compiler
if command -v g++ > /dev/null 2>&1; then
    GCC_VERSION=$(g++ --version | head -1 | grep -oP '\K[\d.]+$' || echo "unknown")
    ok "g++ found: version $GCC_VERSION"
elif command -v clang++ > /dev/null 2>&1; then
    ok "clang++ found"
else
    error "No C++ compiler found. Install g++: sudo apt install -y g++"
    exit 1
fi

# ============================================================================
# System Dependencies
# ============================================================================
if [ "$SKIP_SYSTEM" = false ] && [ "$BASIC" = true ]; then
    info "Installing system dependencies..."
    sudo apt-get update -qq
    sudo apt-get install -y \
        libjpeg-dev \
        libopenexr-dev \
        libgl1-mesa-dev \
        libglib2.0-0 \
        g++ \
        git \
        2>/dev/null || warn "Some system packages may have failed to install."
    ok "System dependencies installed."
fi

# ============================================================================
# Step 1: Create Conda Environment (optional)
# ============================================================================
if [ "$NEW_ENV" = true ]; then
    info "Creating conda environment 'trellis2' with Python 3.11..."
    conda create -n trellis2 python=3.11 -y
    eval "$(conda shell.bash hook)"
    conda activate trellis2
    PYTHON_CMD=python
    ok "Conda environment 'trellis2' activated."
fi

# ============================================================================
# Step 2: Install PyTorch
# ============================================================================
if [ "$BASIC" = true ]; then
    # Auto-detect correct CUDA version for the requested PyTorch version
    # PyTorch 2.6.0–2.7.0 → cu124, 2.8.0 → cu126/cu128, 2.9.1+ → cu128
    case "$TORCH_VERSION" in
        2.4.*|2.5.*|2.6.*)
            AUTO_CUDA="12.4" ;;
        2.7.*)
            AUTO_CUDA="12.4" ;;
        2.8.*)
            AUTO_CUDA="12.8" ;;
        2.9.*)
            AUTO_CUDA="12.8" ;;
        2.10.*)
            AUTO_CUDA="12.8" ;;
        2.11.*)
            AUTO_CUDA="12.8" ;;
        *)
            AUTO_CUDA="$CUDA_VERSION" ;;  # fallback to user-specified
    esac

    # Override if user explicitly changed --cuda-version from default
    if [ "$CUDA_VERSION" != "12.4" ]; then
        # User explicitly set --cuda-version, respect it but warn if mismatch
        if [ "$CUDA_VERSION" != "$AUTO_CUDA" ]; then
            warn "PyTorch $TORCH_VERSION typically requires CUDA $AUTO_CUDA, but you specified CUDA $CUDA_VERSION."
            warn "If installation fails, try: --cuda-version $AUTO_CUDA"
        fi
    else
        # Default was used; auto-detect
        if [ "$CUDA_VERSION" != "$AUTO_CUDA" ]; then
            info "Auto-detected CUDA $AUTO_CUDA for PyTorch $TORCH_VERSION (overriding default $CUDA_VERSION)"
        fi
        CUDA_VERSION="$AUTO_CUDA"
    fi

    CU_TAG="cu${CUDA_VERSION//./}"
    TORCH_INDEX="https://download.pytorch.org/whl/${CU_TAG}"

    # Determine compatible torchvision/torchaudio versions
    if [ "$TORCH_VERSION" = "2.6.0" ]; then
        TORCHVISION_VERSION="0.21.0"
        TORCHAUDIO_VERSION="2.6.0"
    elif [ "$TORCH_VERSION" = "2.7.0" ]; then
        TORCHVISION_VERSION="0.22.0"
        TORCHAUDIO_VERSION="2.7.0"
    elif [ "$TORCH_VERSION" = "2.8.0" ]; then
        TORCHVISION_VERSION="0.23.0"
        TORCHAUDIO_VERSION="2.8.0"
    elif [ "$TORCH_VERSION" = "2.9.1" ]; then
        TORCHVISION_VERSION="0.24.1"
        TORCHAUDIO_VERSION="2.9.1"
    elif [ "$TORCH_VERSION" = "2.10.0" ]; then
        TORCHVISION_VERSION="0.25.0"
        TORCHAUDIO_VERSION="2.10.0"
    elif [ "$TORCH_VERSION" = "2.11.0" ]; then
        TORCHVISION_VERSION="0.26.0"
        TORCHAUDIO_VERSION="2.11.0"
    else
        TORCHVISION_VERSION=""
        TORCHAUDIO_VERSION=""
    fi

    TORCH_INSTALL="torch==${TORCH_VERSION}"
    [ -n "$TORCHVISION_VERSION" ] && TORCH_INSTALL="$TORCH_INSTALL torchvision==${TORCHVISION_VERSION}"
    [ -n "$TORCHAUDIO_VERSION" ] && TORCH_INSTALL="$TORCH_INSTALL torchaudio==${TORCHAUDIO_VERSION}"

    info "Installing PyTorch ${TORCH_VERSION} (CUDA ${CUDA_VERSION})..."
    $PYTHON_CMD -m pip install $TORCH_INSTALL --index-url $TORCH_INDEX

    # Verify torch
    TORCH_INSTALLED=$($PYTHON_CMD -c "import torch; print(torch.__version__)" 2>/dev/null || echo "not installed")
    CUDA_AVAILABLE=$($PYTHON_CMD -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "False")
    ok "PyTorch ${TORCH_INSTALLED} installed (CUDA available: ${CUDA_AVAILABLE})"

    # ============================================================================
    # Step 3: Basic Python Dependencies
    # ============================================================================
    # General Python packages come from PyPI (not the PyTorch wheel index)
    info "Installing basic Python dependencies from PyPI..."
    $PYTHON_CMD -m pip install \
        imageio \
        imageio-ffmpeg \
        tqdm \
        easydict \
        opencv-python-headless \
        ninja \
        trimesh \
        transformers \
        "gradio==6.0.1" \
        tensorboard \
        pandas \
        lpips \
        zstandard \
        kornia \
        timm \
        huggingface_hub \
        accelerate \
        psutil

    # Install xformers from PyTorch index (version must match torch)
    XFORMERS_VERSION=""
    case "$TORCH_VERSION" in
        2.6.0)  XFORMERS_VERSION="0.0.29.post3" ;;
        2.7.0)  XFORMERS_VERSION="0.0.30" ;;
        2.8.0)  XFORMERS_VERSION="0.0.32.post2" ;;
        2.9.1)  XFORMERS_VERSION="0.0.33" ;;
        2.10.0) XFORMERS_VERSION="0.0.35" ;;
        2.11.0) XFORMERS_VERSION="0.0.36" ;;
    esac

    if [ -n "$XFORMERS_VERSION" ]; then
        info "Installing xformers ${XFORMERS_VERSION} (matching PyTorch ${TORCH_VERSION})..."
        $PYTHON_CMD -m pip install "xformers==${XFORMERS_VERSION}" --index-url $TORCH_INDEX
    else
        warn "No known xformers version for PyTorch $TORCH_VERSION. Skipping xformers."
    fi

    info "Installing utils3d from GitHub..."
    $PYTHON_CMD -m pip install git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8

    # Pillow-SIMD (faster than standard Pillow, but only if compatible)
    # NOTE: Pillow-SIMD 9.5 is very old and may break gradio/other packages.
    # Only use it if a recent version is available; otherwise stick with standard Pillow.
    info "Configuring Pillow..."
    PIL_SIMD_VERSION=$($PYTHON_CMD -m pip index versions pillow-simd 2>/dev/null | head -1 | grep -oP '\d+\.\d+\.\d+' || echo "")
    if [ -n "$PIL_SIMD_VERSION" ]; then
        # Check if version is at least 10.0 (compatible with modern packages)
        PIL_MAJOR=$(echo "$PIL_SIMD_VERSION" | cut -d. -f1)
        if [ "$PIL_MAJOR" -ge 10 ]; then
            info "Installing Pillow-SIMD $PIL_SIMD_VERSION..."
            $PYTHON_CMD -m pip uninstall -y pillow 2>/dev/null || true
            $PYTHON_CMD -m pip install "pillow-simd>=$PIL_SIMD_VERSION" || {
                warn "Pillow-SIMD failed to install, falling back to standard Pillow..."
                $PYTHON_CMD -m pip install pillow
            }
        else
            warn "Pillow-SIMD $PIL_SIMD_VERSION is too old (need >=10.0). Using standard Pillow instead."
            $PYTHON_CMD -m pip install pillow
        fi
    else
        info "Pillow-SIMD not available. Installing standard Pillow..."
        $PYTHON_CMD -m pip install pillow
    fi
fi

# ============================================================================
# Step 4: Flash Attention
# ============================================================================
if [ "$FLASHATTN" = true ]; then
    if [ "$PLATFORM" = "cuda" ]; then
        # Check GPU compute capability
        GPU_MAJOR=$($PYTHON_CMD -c "
import torch
if torch.cuda.is_available():
    major, _ = torch.cuda.get_device_capability()
    print(major)
else:
    print(0)
" 2>/dev/null || echo "0")

        if [ "$GPU_MAJOR" -ge 8 ]; then
            info "Installing flash-attn 2.7.3..."
            $PYTHON_CMD -m pip install flash-attn==2.7.3 --no-build-isolation || {
                warn "flash-attn build failed. You can set ATTN_BACKEND=xformers as fallback."
            }
        else
            warn "GPU compute capability < 8.0 (pre-Ampere). Flash Attention is not supported."
            warn "Using xformers instead. Set ATTN_BACKEND=xformers when running."
        fi
    else
        warn "Flash Attention is only available on CUDA. Skipping."
    fi
fi

# ============================================================================
# Step 5: Install CUDA Extensions (prebuilt wheels preferred)
# ============================================================================
LINUX_WHL_DIR="$WORKDIR/whl/linux"
PY_CP_TAG="cp$($PYTHON_CMD -c 'import sys; print(f"{sys.version_info.major}{sys.version_info.minor}")')"
PREBUILT_FOUND=false

# Map torch version to wheel directory name
# e.g. 2.9.1 -> torch291, 2.7.0 -> torch270, 2.11.0 -> torch2110
TORCH_DIR_SUFFIX=$(echo "$TORCH_VERSION" | sed 's/\.//g')  # e.g. "291"
TORCH_WHL_DIR="torch${TORCH_DIR_SUFFIX}_${PY_CP_TAG}"      # e.g. "torch291_cp312"

# Try exact match first: torch version + python version
if [ -d "$LINUX_WHL_DIR/$TORCH_WHL_DIR" ]; then
    whl_count=$(find "$LINUX_WHL_DIR/$TORCH_WHL_DIR" -name "*.whl" 2>/dev/null | wc -l)
    if [ "$whl_count" -gt 0 ]; then
        info "Found $whl_count prebuilt Linux wheels matching PyTorch $TORCH_VERSION + Python $PY_CP_TAG"
        PREBUILT_DIR="$LINUX_WHL_DIR/$TORCH_WHL_DIR"
        PREBUILT_FOUND=true
    fi
fi

# Fallback: match only python version (may be wrong torch version)
if [ "$PREBUILT_FOUND" = false ] && [ -d "$LINUX_WHL_DIR" ]; then
    for subdir in "$LINUX_WHL_DIR"/*/; do
        if [ -d "$subdir" ] && echo "$subdir" | grep -q "$PY_CP_TAG"; then
            whl_count=$(find "$subdir" -name "*.whl" 2>/dev/null | wc -l)
            if [ "$whl_count" -gt 0 ]; then
                warn "No exact wheel match for PyTorch $TORCH_VERSION, using $(basename "$subdir") (may be incompatible)"
                info "Found $whl_count prebuilt Linux wheels in $(basename "$subdir")"
                PREBUILT_DIR="$subdir"
                PREBUILT_FOUND=true
                break
            fi
        fi
    done
fi

if [ "$PREBUILT_FOUND" = true ]; then
    info "Installing CUDA packages from prebuilt wheels (much faster than source builds)..."

    # Install in dependency order: cumesh and flex_gemm must come before o_voxel
    # because o_voxel depends on both. Use --no-deps to prevent pip from trying
    # to resolve git+ dependencies from old wheel metadata.
    INSTALL_ORDER=("cumesh" "flex_gemm" "nvdiffrast" "nvdiffrec_render" "custom_rasterizer" "o_voxel")

    for pkg_prefix in "${INSTALL_ORDER[@]}"; do
        for whl in "$PREBUILT_DIR"/${pkg_prefix}*.whl; do
            if [ -f "$whl" ]; then
                info "  Installing $(basename "$whl")"
                $PYTHON_CMD -m pip install --force-reinstall --no-deps "$whl" || {
                    warn "  Failed to install $(basename "$whl")"
                }
                break
            fi
        done
    done

    # Install any remaining wheels not in the priority list
    for whl in "$PREBUILT_DIR"/*.whl; do
        base=$(basename "$whl")
        already=false
        for pkg_prefix in "${INSTALL_ORDER[@]}"; do
            if [[ "$base" == ${pkg_prefix}* ]]; then
                already=true
                break
            fi
        done
        if [ "$already" = false ]; then
            info "  Installing $base"
            $PYTHON_CMD -m pip install --force-reinstall --no-deps "$whl" || warn "  Failed to install $base"
        fi
    done

    # Install o_voxel's runtime dependencies (not CUDA, just pip packages)
    if ls "$PREBUILT_DIR"/o_voxel*.whl 1>/dev/null 2>&1; then
        info "  Installing o_voxel runtime dependencies (plyfile, trimesh, etc.)..."
        $PYTHON_CMD -m pip install plyfile trimesh zstandard easydict 2>/dev/null || true
    fi

    ok "Prebuilt CUDA packages installed."
else
    warn "No prebuilt wheels found for Python $PY_CP_TAG."
    warn "Falling back to building from source (this takes 20-60 min)..."
    BUILD_DIR="/tmp/trellis2_extensions"
    mkdir -p "$BUILD_DIR"

    # --- nvdiffrast ---
    if [ "$NVDIFFRAST" = true ]; then
        if [ "$PLATFORM" = "cuda" ]; then
            info "Building nvdiffrast from source..."
            if [ ! -d "$BUILD_DIR/nvdiffrast" ]; then
                git clone -b v0.4.0 https://github.com/NVlabs/nvdiffrast.git "$BUILD_DIR/nvdiffrast"
            fi
            $PYTHON_CMD -m pip install "$BUILD_DIR/nvdiffrast" --no-build-isolation || {
                error "nvdiffrast build failed. Ensure CUDA toolkit is properly installed."
            }
        else
            warn "nvdiffrast requires CUDA. Skipping."
        fi
    fi

    # --- nvdiffrec ---
    if [ "$NVDIFFREC" = true ]; then
        if [ "$PLATFORM" = "cuda" ]; then
            info "Building nvdiffrec from source..."
            if [ ! -d "$BUILD_DIR/nvdiffrec" ]; then
                git clone -b renderutils https://github.com/IgorAherne/nvdiffrec.git "$BUILD_DIR/nvdiffrec"
            fi
            $PYTHON_CMD -m pip install "$BUILD_DIR/nvdiffrec" --no-build-isolation || {
                error "nvdiffrec build failed."
            }
        else
            warn "nvdiffrec requires CUDA. Skipping."
        fi
    fi

    # --- CuMesh ---
    if [ "$CUMESH" = true ]; then
        info "Building CuMesh from source..."
        if [ ! -d "$BUILD_DIR/CuMesh" ]; then
            git clone --recursive https://github.com/JeffreyXiang/CuMesh.git "$BUILD_DIR/CuMesh"
        fi
        $PYTHON_CMD -m pip install "$BUILD_DIR/CuMesh" --no-build-isolation || {
            error "CuMesh build failed."
        }
    fi

    # --- FlexGEMM ---
    if [ "$FLEXGEMM" = true ]; then
        info "Building FlexGEMM from source..."
        if [ ! -d "$BUILD_DIR/FlexGEMM" ]; then
            git clone --recursive https://github.com/JeffreyXiang/FlexGEMM.git "$BUILD_DIR/FlexGEMM"
        fi
        $PYTHON_CMD -m pip install "$BUILD_DIR/FlexGEMM" --no-build-isolation || {
            error "FlexGEMM build failed."
        }
    fi

    # --- o-voxel ---
    if [ "$OVOXEL" = true ]; then
        info "Building o-voxel from source..."
        OVOXEL_SRC="$WORKDIR/o-voxel"
        if [ -d "$OVOXEL_SRC" ]; then
            if [ ! -d "$OVOXEL_SRC/third_party/eigen/Eigen" ]; then
                warn "Eigen submodule not initialized. Running git submodule update..."
                cd "$WORKDIR"
                git submodule update --init --recursive
            fi
            rm -rf "$BUILD_DIR/o-voxel"
            cp -r "$OVOXEL_SRC" "$BUILD_DIR/o-voxel"
            $PYTHON_CMD -m pip install "$BUILD_DIR/o-voxel" --no-build-isolation || {
                error "o-voxel build failed. Ensure CUDA toolkit and eigen submodule are available."
            }
        else
            error "o-voxel source directory not found at $OVOXEL_SRC"
            error "Run: git submodule update --init --recursive"
        fi
    fi
fi

# ============================================================================
# Step 6: Initialize Submodules
# ============================================================================
cd "$WORKDIR"
info "Ensuring git submodules are initialized..."
git submodule update --init --recursive 2>/dev/null || warn "git submodule update failed (may not be critical)"

# ============================================================================
# Step 7: Download Models (optional)
# ============================================================================
if [ "$SKIP_MODELS" = false ] && [ "$BASIC" = true ]; then
    info "Downloading model weights..."
    $PYTHON_CMD -c "
import os, sys
sys.path.insert(0, '$WORKDIR')

# Download dinov3 and RMBG models
try:
    from install import download_models
    download_models()
except Exception as e:
    print(f'[WARN] Model download failed: {e}')
    print('You can download models manually later.')

# Download HuggingFace models (this can take a long time)
try:
    from install import download_hf_models
    download_hf_models()
except Exception as e:
    print(f'[WARN] HuggingFace model download failed: {e}')
    print('Models will be downloaded on first run.')
" || warn "Model download step encountered errors (non-fatal)."
fi

# ============================================================================
# Verification
# ============================================================================
echo ""
echo "============================================================"
echo " Verification"
echo "============================================================"

$PYTHON_CMD -c "
import sys
all_ok = True

# Check torch
try:
    import torch
    print(f'  [OK] PyTorch {torch.__version__}')
    if torch.cuda.is_available():
        print(f'  [OK] CUDA {torch.version.cuda} — {torch.cuda.get_device_name(0)}')
    else:
        print(f'  [WARN] CUDA not available (GPU may not be accessible in this session)')
except ImportError:
    print('  [ERROR] PyTorch not installed')
    all_ok = False

# Check key modules
modules = {
    'nvdiffrast': 'nvdiffrast',
    'o_voxel': 'o_voxel',
    'flash_attn': 'flash_attn',
    'flex_gemm': 'flex_gemm',
    'xformers': 'xformers',
    'trimesh': 'trimesh',
    'PIL': 'Pillow',
    'gradio': 'gradio',
    'cv2': 'opencv-python-headless',
    'transformers': 'transformers',
}

for mod, label in modules.items():
    try:
        __import__(mod)
        print(f'  [OK] {label}')
    except ImportError:
        print(f'  [WARN] {label} not found (may be optional)')

# Check trellis2
try:
    from trellis2.pipelines import Trellis2ImageTo3DPipeline
    print('  [OK] trellis2 pipeline')
except ImportError as e:
    print(f'  [WARN] trellis2 pipeline: {e}')

if all_ok:
    print()
    print('  Installation appears successful!')
else:
    print()
    print('  Some components are missing — check warnings above.')
"

echo ""
echo "============================================================"
echo " Installation Complete!"
echo "============================================================"
echo ""
echo "To run the Gradio web app:"
echo "  cd $WORKDIR"
echo "  python app.py"
echo ""
echo "To run a minimal example:"
echo "  cd $WORKDIR"
echo "  python example.py"
echo ""
echo "Environment variables you can set:"
echo "  ATTN_BACKEND=xformers     # Use xformers instead of flash-attn"
echo "  SPARSE_CONV_BACKEND=flex_gemm  # Sparse convolution backend"
echo "  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # Save VRAM"
echo ""
