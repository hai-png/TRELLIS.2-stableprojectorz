![](assets/teaser.webp)

# TRELLIS.2 — StableProjectorz Fork

<a href="https://arxiv.org/abs/2512.14692"><img src="https://img.shields.io/badge/Paper-Arxiv-b31b1b.svg" alt="Paper"></a>
<a href="https://huggingface.co/microsoft/TRELLIS.2-4B"><img src="https://img.shields.io/badge/Hugging%20Face-Model-yellow" alt="Hugging Face"></a>
<a href="https://huggingface.co/spaces/microsoft/TRELLIS.2"><img src="https://img.shields.io/badge/Hugging%20Face-Demo-blueviolet"></a>
<a href="https://microsoft.github.io/TRELLIS.2"><img src="https://img.shields.io/badge/Project-Website-blue" alt="Project Page"></a>
<a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green" alt="License"></a>

**TRELLIS.2** is a state-of-the-art large 3D generative model (4B parameters) for high-fidelity **image-to-3D** generation. This fork adds **Windows and Linux one-click installers**, **8 GB GPU optimization**, **bug fixes**, and **StableProjectorz API integration**.

https://github.com/user-attachments/assets/63b43a7e-acc7-4c81-a900-6da450527d8f

---

## What This Fork Adds

| Feature | Original Microsoft/TRELLIS.2 | This Fork |
|:---|:---:|:---:|
| Windows installer | — | One-click EXE |
| Linux installer | Manual `setup.sh` only | `setup_linux.sh` + `install_linux.py` with prebuilt wheels |
| Prebuilt CUDA wheels | None | Windows + Linux (3 PyTorch versions) |
| GPU VRAM requirement | 24 GB | **8 GB** (optimized for consumer GPUs) |
| Vertical lines bug | Present | **Fixed** (credit: [visualbruno](https://github.com/visualbruno)) |
| StableProjectorz API | — | FastAPI server (`api_spz/`) |
| Python versions | 3.8+ | **3.10 – 3.12** (Linux) / **3.11** (Windows) |
| PyTorch versions | 2.6.0 only | **2.6.0 – 2.11.0** (configurable) |
| GPU arch compatibility | A100/H100 only | **GTX 10xx – RTX 50xx** (with fallback) |
| Profiling tools | — | Python profiler, PyTorch profiler, CUDA Sync Hunter |

---

## Quick Start

### Windows

Download the one-click installer from [GitHub Releases](https://github.com/IgorAherne/TRELLIS.2-stableprojectorz/releases/tag/latest) (Python 3.11, CUDA 12.8, PyTorch 2.8.0).

### Linux

The fastest way to get started on Linux:

```bash
git clone https://github.com/IgorAherne/TRELLIS.2-stableprojectorz.git --recursive
cd TRELLIS.2-stableprojectorz

# Option A: Bash installer (recommended)
. ./setup_linux.sh --new-env --all --torch-version 2.9.1

# Option B: Python installer
python install_linux.py --torch-version 2.9.1
```

Then launch the web UI:
```bash
conda activate trellis2   # if you used --new-env
python app.py
```

Open **http://127.0.0.1:8080** in your browser.

---

## Installation

### Prerequisites

| Requirement | Minimum | Recommended |
|:---|:---|:---|
| **GPU** | 8 GB VRAM (Ampere+) | 24 GB VRAM (RTX 3090 / A100) |
| **NVIDIA Driver** | 525+ | 535+ |
| **CUDA Toolkit** | 12.4 | 12.8 |
| **OS** | Ubuntu 20.04 / Windows 10 | Ubuntu 22.04 / Windows 11 |
| **Conda** | Miniconda | Anaconda |
| **C++ Compiler** | g++ 9+ (Linux) | g++ 11+ (Linux) |

### Linux Installers

Three installation methods are available. All of them:
- Install PyTorch + xformers with the correct CUDA version
- Download model weights (DINOv3, RMBG-2.0, TRELLIS.2-4B) automatically
- Install prebuilt wheels when available, fall back to source build

#### Method 1: Bash Installer (`setup_linux.sh`)

```bash
. ./setup_linux.sh --new-env --all --torch-version 2.9.1
```

**Options:**

| Flag | Description |
|:---|:---|
| `--new-env` | Create a new conda environment named `trellis2` |
| `--all` | Install everything (PyTorch, flash-attn, CUDA extensions, models) |
| `--basic` | Install only Python dependencies |
| `--flash-attn` | Install flash-attention (requires Ampere+ GPU) |
| `--cumesh` | Install CuMesh |
| `--o-voxel` | Install O-Voxel |
| `--flexgemm` | Install FlexGEMM |
| `--nvdiffrast` | Install nvdiffrast |
| `--nvdiffrec` | Install nvdiffrec |
| `--skip-system-deps` | Skip `apt install` system packages |
| `--skip-models` | Skip downloading model weights |
| `--cuda-version VERSION` | CUDA version for PyTorch (default: 12.4) |
| `--torch-version VERSION` | PyTorch version (default: 2.6.0) |

#### Method 2: Python Installer (`install_linux.py`)

```bash
python install_linux.py --torch-version 2.9.1
```

**Options:**

| Flag | Description |
|:---|:---|
| `--cuda-version` | CUDA version: 12.4, 12.6, or 12.8 (default: 12.4) |
| `--torch-version` | PyTorch version (default: 2.6.0) |
| `--skip-models` | Skip downloading DINOv3 and RMBG models |
| `--skip-hf` | Skip downloading HuggingFace model weights |

#### Method 3: Manual Installation

See [LINUX_SETUP.md](LINUX_SETUP.md) for the complete step-by-step guide.

### Windows Installer

1. Download from [Releases](https://github.com/IgorAherne/TRELLIS.2-stableprojectorz/releases/tag/latest)
2. Run the installer (Python 3.11, CUDA 12.8, PyTorch 2.8.0)
3. Launch via the desktop shortcut or `python app.py`

### Supported PyTorch Versions

| PyTorch | CUDA | torchvision | xformers | Prebuilt Wheels? |
|:---|:---|:---|:---|:---:|
| 2.6.0 | 12.4 | 0.21.0 | 0.0.29 | — |
| 2.7.0 | 12.4 | 0.22.0 | 0.0.30 | cp312 |
| 2.8.0 | 12.8 | 0.23.0 | 0.0.32 | Windows only |
| **2.9.1** | **12.8** | **0.24.1** | **0.0.33** | **cp312 (recommended)** |
| 2.10.0 | 12.8 | 0.25.0 | 0.0.35 | — |
| 2.11.0 | 12.8 | 0.26.0 | 0.0.36 | cp313 |

> **Recommended:** PyTorch 2.9.1 + Python 3.12 — has the most complete set of prebuilt Linux wheels.

---

## Usage

### Web UI

```bash
python app.py [--host 127.0.0.1] [--port 8080]
```

Open the URL shown in your terminal. The UI provides:
- **Image upload** → automatic background removal (or use your own alpha mask)
- **Resolution**: 512, 1024, or 1536 voxel grid
- **6 render modes**: Normal, Clay, Base Color, HDRI Forest/Sunset/Courtyard
- **8-step view angle** slider
- **GLB export** with configurable decimation target and texture size
- **Advanced settings** for all 3 generation stages (guidance strength, rescale, sampling steps)

### Programmatic API

```python
import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import cv2
import imageio
from PIL import Image
import torch
from trellis2.pipelines import Trellis2ImageTo3DPipeline
from trellis2.utils import render_utils
from trellis2.renderers import EnvMap
import o_voxel

# 1. Setup Environment Map
envmap = EnvMap(torch.tensor(
    cv2.cvtColor(cv2.imread('assets/hdri/forest.exr', cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB),
    dtype=torch.float32, device='cuda'
))

# 2. Load Pipeline
pipeline = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
pipeline.cuda()

# 3. Load Image & Generate
image = Image.open("assets/example_image/T.png")
mesh = pipeline.run(image)[0]
mesh.simplify(16777216)  # nvdiffrast limit

# 4. Render Video
video = render_utils.make_pbr_vis_frames(render_utils.render_video(mesh, envmap=envmap))
imageio.mimsave("sample.mp4", video, fps=15)

# 5. Export to GLB
glb = o_voxel.postprocess.to_glb(
    vertices=mesh.vertices, faces=mesh.faces,
    attr_volume=mesh.attrs, coords=mesh.coords,
    attr_layout=mesh.layout, voxel_size=mesh.voxel_size,
    aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
    decimation_target=1000000, texture_size=4096,
    remesh=True, remesh_band=1, remesh_project=0,
    verbose=True
)
glb.export("sample.glb", extension_webp=True)
```

### StableProjectorz API Server

For integration with [StableProjectorz](https://stableprojectorz.com/) (AI-texturing tool):

```bash
python -m api_spz.main_api [--host 127.0.0.1] [--port 7960] [--device cuda]
```

This starts a FastAPI server with endpoints for 3D generation. See `api_spz/api-documentation.html` for the full API spec.

---

## GPU Compatibility

### Supported GPUs

| GPU Series | Compute Cap. | flash-attn | xformers | Min VRAM | Notes |
|:---|:---|:---:|:---:|:---|:---|
| GTX 10xx (Pascal) | sm_61 | — | — | 8 GB | Source build required |
| GTX 16xx / RTX 20xx (Turing) | sm_75 | — | Yes | 8 GB | No flash-attn; use xformers |
| RTX 30xx (Ampere) | sm_86 | Yes | Yes | 8 GB | Best consumer GPU support |
| RTX 40xx (Ada) | sm_89 | Yes | Yes | 8 GB | May need source rebuild |
| H100 (Hopper) | sm_90 | Yes | Yes | 24 GB | Full support |
| RTX 50xx (Blackwell) | sm_100/120 | Yes | Yes | 8 GB | PyTorch 2.11+ recommended |

### GPU Architecture Fallback

If the prebuilt `flex_gemm` wheel doesn't include CUDA kernels for your GPU (error: `no kernel image is available for execution`), the code automatically falls back to a pure-PyTorch neighbor map implementation. This is slower but works on every GPU.

For best performance, rebuild FlexGEMM from source targeting your GPU:
```bash
python -c "import torch; cc=torch.cuda.get_device_capability(); print(f'{cc[0]}.{cc[1]}')"
# Then:
TORCH_CUDA_ARCH_LIST='<your_compute_cap>' pip install --force-reinstall --no-deps /path/to/FlexGEMM
```

---

## Environment Variables

| Variable | Default | Description |
|:---|:---|:---|
| `ATTN_BACKEND` | `xformers` | Attention backend: `xformers` or `flash_attn` |
| `PYTORCH_ALLOC_CONF` | — | CUDA memory allocator config (e.g. `expandable_segments:True`) |
| `OPENCV_IO_ENABLE_OPENEXR` | — | Must be `1` for HDRI environment map loading |
| `TORCHDYNAMO_DISABLE` | — | Set to `1` by pipeline worker (avoids torch.compile overhead) |
| `SPARSE_DEBUG` | `0` | Set to `1` for VRAM debug prints during generation |
| `HF_TOKEN` | — | HuggingFace token for higher download rate limits |

---

## Project Structure

```
TRELLIS.2-stableprojectorz/
├── app.py                    # Gradio web UI
├── example.py                # Minimal programmatic example
├── pipeline_worker.py        # Subprocess GPU pipeline (avoids GIL contention)
├── install.py                # Windows one-click installer
├── install_linux.py          # Linux Python installer
├── setup_linux.sh            # Linux bash installer
├── trellis2/                 # Core ML pipeline package
│   ├── pipelines/            # Inference pipelines (image-to-3D, cascade, texture)
│   ├── models/               # Neural network architectures (VAE, DiT, flow matching)
│   ├── modules/              # Reusable modules (attention, sparse conv, image encoder)
│   ├── representations/      # Mesh & voxel data structures
│   ├── renderers/            # PBR rendering (nvdiffrast, env maps)
│   ├── datasets/             # Training datasets
│   ├── trainers/             # Training code
│   └── utils/                # Utilities (render, mesh, loss)
├── o-voxel/                  # O-Voxel submodule (sparse voxel ↔ mesh conversion)
├── api_spz/                  # StableProjectorz FastAPI server
│   ├── main_api.py           # API entry point
│   ├── core/                 # Pipeline state management
│   └── routes/               # Generation endpoints
├── assets/                   # HDRIs, example images, UI icons
├── whl/                      # Prebuilt CUDA wheels
│   ├── linux/                # Linux wheels (torch270_cp312, torch291_cp312, torch2110_cp313)
│   └── *.whl                 # Windows wheels (cp311)
├── MODELS/                   # Downloaded model weights (dinov3, RMBG-2.0)
└── tools/                    # Profiling utilities
```

---

## Troubleshooting

### `CUDA error: no kernel image is available for execution`

The prebuilt `flex_gemm` wheel doesn't include kernels for your GPU. The code will automatically fall back to a PyTorch implementation. For best performance, rebuild from source:
```bash
TORCH_CUDA_ARCH_LIST='8.9' pip install --force-reinstall --no-deps /tmp/FlexGEMM
```

### Out of Memory (OOM)

- Use **1024** resolution instead of 1536
- Close other GPU applications (Blender, Photoshop, Unity)
- Set `PYTORCH_ALLOC_CONF=expandable_segments:True`
- The low-VRAM mode is enabled by default and aggressively offloads models to CPU between stages

### `flash-attn` import error on older GPUs

Flash attention requires Ampere+ (compute capability ≥ 8.0). On older GPUs, use xformers:
```bash
export ATTN_BACKEND=xformers
```

### `PIL.Image` import error after installing Pillow-SIMD

Pillow-SIMD ≥ 10.0 has known compatibility issues with some packages. The installers automatically fall back to standard Pillow if needed. If you hit this manually:
```bash
pip uninstall pillow-simd && pip install pillow
```

### `transformers` attribute errors (`all_tied_weights_keys`)

This fork includes compatibility patches for transformers 4.49+ that automatically handle the `all_tied_weights_keys` property mismatch with custom model classes. No action needed.

### More Help

- [LINUX_SETUP.md](LINUX_SETUP.md) — Full Linux setup guide with 10 troubleshooting entries
- [Discord](https://discord.gg/aWbnX2qan2) — Community support
- [Issues](https://github.com/IgorAherne/TRELLIS.2-stableprojectorz/issues) — Bug reports

---

## Features (from the original TRELLIS.2)

### High Quality, Resolution & Efficiency

The 4B-parameter model generates high-resolution fully textured assets with exceptional fidelity. It uses a Sparse 3D VAE with 16x spatial downsampling to encode assets into a compact latent space.

| Resolution | Total Time* | Breakdown (Shape + Material) |
|:---|:---|:---|
| **512** | ~3 s | 2 s + 1 s |
| **1024** | ~17 s | 10 s + 7 s |
| **1536** | ~60 s | 35 s + 25 s |

*Tested on NVIDIA H100 GPU. Consumer GPUs will be slower but fully supported.

### Arbitrary Topology Handling

The **O-Voxel** representation breaks the limits of iso-surface fields:
- Open surfaces (e.g., clothing, leaves)
- Non-manifold geometry
- Internal enclosed structures

### Rich Texture Modeling

Full PBR material generation including **Base Color, Roughness, Metallic, and Opacity**, enabling photorealistic rendering and transparency support.

### Minimalist Processing

Data processing is streamlined for instant conversions that are rendering-free and optimization-free:
- **< 10 s** (single CPU): Textured Mesh → O-Voxel
- **< 100 ms** (CUDA): O-Voxel → Textured Mesh

---

## Pretrained Weights

| Model | Parameters | Resolution | Link |
|:---|:---|:---|:---|
| **TRELLIS.2-4B** | 4 Billion | 512 – 1536 | [Hugging Face](https://huggingface.co/microsoft/TRELLIS.2-4B) |

Model weights are downloaded automatically during installation. You can also manually download:
- **DINOv3** and **RMBG-2.0** from [GitHub Releases](https://github.com/IgorAherne/TRELLIS.2-stableprojectorz/releases/tag/extra-models) → extract to `MODELS/`
- **TRELLIS.2-4B** from [HuggingFace](https://huggingface.co/microsoft/TRELLIS.2-4B) → cached automatically on first run

---

## Related Packages

- **[O-Voxel](o-voxel/)** — Core library for converting between textured meshes and the O-Voxel representation
- **[FlexGEMM](https://github.com/JeffreyXiang/FlexGEMM)** — Efficient sparse convolution based on Triton
- **[CuMesh](https://github.com/JeffreyXiang/CuMesh)** — CUDA-accelerated mesh utilities (remeshing, decimation, UV-unwrapping)
- **[StableProjectorz](https://stableprojectorz.com/)** — Free AI-texturing tool with native TRELLIS.2 integration

---

## License

This project is released under the **[MIT License](LICENSE)**.

Certain dependencies operate under separate license terms:
- [**nvdiffrast**](https://github.com/NVlabs/nvdiffrast) — [NVlabs License](https://github.com/NVlabs/nvdiffrast/blob/main/LICENSE.txt)
- [**nvdiffrec**](https://github.com/NVlabs/nvdiffrec) — [NVlabs License](https://github.com/NVlabs/nvdiffrec/blob/main/LICENSE.txt)

---

## Citation

```bibtex
@article{
    xiang2025trellis2,
    title={Native and Compact Structured Latents for 3D Generation},
    author={Xiang, Jianfeng and Chen, Xiaoxue and Xu, Sicheng and Wang, Ruicheng and Lv, Zelong and Deng, Yu and Zhu, Hongyuan and Dong, Yue and Zhao, Hao and Yuan, Nicholas Jing and Yang, Jiaolong},
    journal={Tech report},
    year={2025}
}
```
