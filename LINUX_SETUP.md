# TRELLIS.2-stableprojectorz — Linux Setup Guide

This guide explains how to get **TRELLIS.2-stableprojectorz** working on Linux using prebuilt wheels (where available) and source builds (for CUDA packages).

---

## ⚠️ Key Differences from Windows

| Aspect | Windows (original) | Linux (this guide) |
|--------|-------------------|-------------------|
| Python | 3.11 (cp311 wheels) | 3.10, 3.11, or 3.12 |
| CUDA wheels | `whl/*.win_amd64.whl` | **Must build from source** |
| Triton | `triton-windows` fork | Standard `triton` from PyPI |
| PyTorch | 2.8.0 + CUDA 12.8 | 2.9.1 + CUDA 12.8 (recommended for prebuilt wheels) |
| xformers | 0.0.32.post2 | Matches your PyTorch version |
| `SETUPTOOLS_USE_DISTUTILS` | Set to `stdlib` | Not needed on Linux (removed) |
| flex_gemm patches | Forces `EXPLICIT_GEMM` + TritonFallback | Uses native triton on Linux |

---

## Prerequisites

### Hardware
- **NVIDIA GPU** with at least 24GB VRAM (tested on A100/H100; RTX 3090/4090 work with lower resolutions)
- Ampere or newer (RTX 30xx+) recommended for flash-attn support

### Software
- **Ubuntu 20.04+** (or similar Linux distro)
- **NVIDIA Driver** 525+
- **CUDA Toolkit** 12.4+ ([download](https://developer.nvidia.com/cuda-toolkit-archive))
- **g++** 9+ (`sudo apt install g++`)
- **Conda** (recommended, [install](https://docs.anaconda.com/miniconda/install/))

### System Libraries
```bash
sudo apt install -y libjpeg-dev libopenexr-dev libgl1-mesa-dev libglib2.0-0 git g++
```

---

## Quick Install (Recommended)

### Option A: Using the Bash installer

```bash
cd /path/to/TRELLIS.2-stableprojectorz

# Full install with a new conda environment (RECOMMENDED):
# Uses Python 3.12 + PyTorch 2.9.1 to match prebuilt wheels
bash setup_linux.sh --new-env --all --torch-version 2.9.1

# Or install into your current Python 3.12 environment:
bash setup_linux.sh --all --torch-version 2.9.1

# With custom CUDA version:
bash setup_linux.sh --all --cuda-version 12.8 --torch-version 2.9.1
```

### Option B: Using the Python installer

```bash
cd /path/to/TRELLIS.2-stableprojectorz

# With a conda env (Python 3.12 recommended for prebuilt wheels):
conda create -n trellis2 python=3.12 -y
conda activate trellis2
python install_linux.py --torch-version 2.9.1

# Skip model downloads (do them later):
python install_linux.py --torch-version 2.9.1 --skip-models --skip-hf
```

### Option C: Manual step-by-step (fastest with prebuilt wheels)

```bash
# 1. Create conda environment with Python 3.12 (matches prebuilt wheels)
conda create -n trellis2 python=3.12 -y
conda activate trellis2

# 2. Install PyTorch 2.9.1 (matches prebuilt wheels — uses CUDA 12.8)
pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu128

# 3. Install basic dependencies
pip install imageio imageio-ffmpeg tqdm easydict opencv-python-headless \
    ninja trimesh transformers "gradio==6.0.1" tensorboard pandas lpips \
    zstandard kornia timm huggingface_hub accelerate psutil triton xformers \
    --index-url https://download.pytorch.org/whl/cu128

# 4. Install utils3d
pip install git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8

# 5. Install Pillow-SIMD
pip uninstall -y pillow && pip install pillow-simd

# 6. Install flash-attn (Ampere+ GPUs only)
pip install flash-attn --no-build-isolation

# 7. Install CUDA packages from prebuilt wheels (seconds, not hours!)
#    IMPORTANT: Install in dependency order — cumesh/flex_gemm BEFORE o_voxel
#    Use --no-deps to avoid pip trying to resolve git+ URLs in wheel metadata
pip install --no-deps whl/linux/torch291_cp312/cumesh*.whl
pip install --no-deps whl/linux/torch291_cp312/flex_gemm*.whl
pip install --no-deps whl/linux/torch291_cp312/nvdiffrast*.whl
pip install --no-deps whl/linux/torch291_cp312/nvdiffrec_render*.whl
pip install --no-deps whl/linux/torch291_cp312/custom_rasterizer*.whl
pip install --no-deps whl/linux/torch291_cp312/o_voxel*.whl
# Install o_voxel's non-CUDA dependencies:
pip install plyfile trimesh zstandard easydict

# 8. Download model weights (optional, ~20GB)
python -c "from huggingface_hub import snapshot_download; snapshot_download('microsoft/TRELLIS.2-4B')"
python -c "from huggingface_hub import snapshot_download; snapshot_download('microsoft/TRELLIS-image-large')"
```

---

## Prebuilt Linux Wheels

Prebuilt Linux wheels are available from the [visualbruno/ComfyUI-Trellis2](https://github.com/visualbruno/ComfyUI-Trellis2/tree/main/wheels/Linux) repository. These are already included in `whl/linux/` and the installers will use them automatically.

### Available Wheel Sets

| Directory | PyTorch Version | Python | Wheels Included |
|-----------|----------------|--------|-----------------|
| `whl/linux/torch270_cp312/` | 2.7.0 | 3.12 | cumesh, flex_gemm, nvdiffrast, o_voxel, custom_rasterizer |
| `whl/linux/torch291_cp312/` | 2.9.1 | 3.12 | cumesh, flex_gemm, nvdiffrast, **nvdiffrec_render**, o_voxel, custom_rasterizer |
| `whl/linux/torch2110_cp313/` | 2.11.0 | 3.13 | cumesh, flex_gemm, nvdiffrast, **nvdiffrec_render**, o_voxel |

> **Recommendation**: Use **Torch 2.9.1 + Python 3.12** — it has the most complete set including `nvdiffrec_render`.

### Package Availability on PyPI vs Source Build

| Package | Linux Wheel Available? | Prebuilt in whl/linux/ | Source Build Required? |
|---------|----------------------|----------------------|----------------------|
| PyTorch | ✅ Yes (PyPI/PyTorch index) | — | No |
| flash-attn | ✅ Yes (PyPI) | — | No |
| xformers | ✅ Yes (PyTorch index) | — | No |
| triton | ✅ Yes (PyPI) | — | No |
| gradio | ✅ Yes (PyPI) | — | No |
| nvdiffrast | ✅ Prebuilt | ✅ torch270/291/2110 | Fallback only |
| nvdiffrec_render | ✅ Prebuilt | ✅ torch291/2110 | Fallback only |
| CuMesh | ✅ Prebuilt | ✅ torch270/291/2110 | Fallback only |
| FlexGEMM | ✅ Prebuilt | ✅ torch270/291/2110 | Fallback only |
| o-voxel | ✅ Prebuilt | ✅ torch270/291/2110 | Fallback only |

> **Note**: The `whl/` directory (top-level) contains **Windows-only** wheels (`win_amd64`). Linux wheels are in `whl/linux/`.

### Manual Prebuilt Wheel Installation

If you prefer to install the prebuilt wheels manually, install them in dependency order with `--no-deps` to avoid pip trying to resolve git+ URLs from wheel metadata:

```bash
# For Python 3.12 + PyTorch 2.9.1 (recommended):
WHL_DIR=whl/linux/torch291_cp312
pip install --no-deps "$WHL_DIR"/cumesh*.whl "$WHL_DIR"/flex_gemm*.whl \
    "$WHL_DIR"/nvdiffrast*.whl "$WHL_DIR"/nvdiffrec_render*.whl \
    "$WHL_DIR"/custom_rasterizer*.whl "$WHL_DIR"/o_voxel*.whl
pip install plyfile trimesh zstandard easydict

# For Python 3.12 + PyTorch 2.7.0:
WHL_DIR=whl/linux/torch270_cp312
pip install --no-deps "$WHL_DIR"/cumesh*.whl "$WHL_DIR"/flex_gemm*.whl \
    "$WHL_DIR"/nvdiffrast*.whl "$WHL_DIR"/custom_rasterizer*.whl \
    "$WHL_DIR"/o_voxel*.whl
pip install plyfile trimesh zstandard easydict

# For Python 3.13 + PyTorch 2.11.0:
WHL_DIR=whl/linux/torch2110_cp313
pip install --no-deps "$WHL_DIR"/cumesh*.whl "$WHL_DIR"/flex_gemm*.whl \
    "$WHL_DIR"/nvdiffrast*.whl "$WHL_DIR"/nvdiffrec_render*.whl \
    "$WHL_DIR"/o_voxel*.whl
pip install plyfile trimesh zstandard easydict
```

> **Why `--no-deps`?** Some prebuilt wheels (especially `o_voxel`) may contain git+ URL dependencies in their metadata (e.g., `cumesh@ git+https://...`). Using `--no-deps` prevents pip from trying to fetch these, since the wheels are already provided locally. The install scripts handle this automatically.

---

## Running

### Web App (Gradio)
```bash
conda activate trellis2
python app.py
# Open http://127.0.0.1:8080 in your browser
```

### Minimal Example
```bash
conda activate trellis2
python example.py
```

### Environment Variables

```bash
# Use xformers instead of flash-attn (for pre-Ampere GPUs):
export ATTN_BACKEND=xformers

# Save GPU memory:
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Enable OpenEXR support in OpenCV:
export OPENCV_IO_ENABLE_OPENEXR=1
```

---

## Code Patches Applied

The following patches were applied to make the code Linux-compatible:

### 1. `pipeline_worker.py` — Platform-aware flex_gemm patches
- **Before**: Always forced `EXPLICIT_GEMM` algorithm and injected a slow `TritonFallback` for flex_gemm
- **After**: Detects platform; on Linux with triton installed, uses the native optimized triton kernels. Only applies the fallback when triton is genuinely unavailable.
- **Impact**: Significant performance improvement on Linux (native triton is much faster than the Python fallback loop)

### 2. `api_spz/core/state_manage.py` — Same platform-aware patches
- Same change as pipeline_worker.py for the API server path
- Also made `SETUPTOOLS_USE_DISTUTILS=stdlib` conditional (Windows-only workaround)

### 3. `app.py` — Conditional SETUPTOOLS_USE_DISTUTILS
- **Before**: Always set `SETUPTOOLS_USE_DISTUTILS=stdlib`
- **After**: Only set on Windows (`os.name == 'nt'`). This setting can cause issues with distutils on Linux.

### 4. `o-voxel/setup.py` — Comment clarifying TORCH_CUDA_ARCH_LIST
- The semicolon-separated format works on both Windows and Linux (PyTorch handles both), but added a clarifying comment.

---

## Troubleshooting

### "CUDA not available"
```bash
# Check your NVIDIA driver
nvidia-smi

# Check CUDA toolkit
nvcc --version

# Check PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
```

### "flash-attn build fails"
- Ensure CUDA toolkit is installed and `nvcc` is on PATH
- For pre-Ampere GPUs (GTX 10xx/20xx), flash-attn is not supported — use xformers instead:
  ```bash
  export ATTN_BACKEND=xformers
  ```

### "nvdiffrast/nvdiffrec build fails"
- Common causes: missing CUDA toolkit, wrong CUDA version, or missing C++ compiler
- Make sure `nvcc --version` shows the same CUDA version as your PyTorch
- Try setting `CUDA_HOME`:
  ```bash
  export CUDA_HOME=/usr/local/cuda-12.4
  ```

### "o-voxel build fails — eigen not found"
- Initialize the git submodule:
  ```bash
  git submodule update --init --recursive
  ```

### "Pillow-SIMD fails to install"
- Install the system dependency: `sudo apt install libjpeg-dev`
- If it still fails, standard Pillow works fine: `pip install pillow`

### "Out of VRAM (OOM)"
- Use 512³ resolution instead of 1024³
- Set: `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
- The StableProjectorz fork is optimized for 8GB GPUs at 1024³

### "triton not found" during flex_gemm usage
- Make sure you installed `triton` (NOT `triton-windows`):
  ```bash
  pip install triton
  ```

### "Cannot install cumesh and o-voxel — conflicting dependencies"
This happens when pip tries to resolve `o_voxel`'s git+ dependencies against the prebuilt `cumesh` wheel. The fix is to use `--no-deps`:
```bash
pip install --no-deps whl/linux/torch291_cp312/*.whl
pip install plyfile trimesh zstandard easydict
```
The install scripts (`install_linux.py` and `setup_linux.sh`) handle this automatically.

---

## GPU Compatibility Table

| GPU Series | Compute Capability | flash-attn | xformers | Min VRAM for 512³ | Min VRAM for 1024³ |
|------------|-------------------|------------|----------|-------------------|-------------------|
| GTX 10xx (Pascal) | 6.1 | ❌ | ✅ | 8 GB | 16 GB |
| RTX 20xx (Turing) | 7.5 | ❌ | ✅ | 8 GB | 16 GB |
| RTX 30xx (Ampere) | 8.6 | ✅ | ✅ | 8 GB | 16 GB |
| RTX 40xx (Ada) | 8.9 | ✅ | ✅ | 8 GB | 12 GB |
| H100 (Hopper) | 9.0 | ✅ | ✅ | — | — |
