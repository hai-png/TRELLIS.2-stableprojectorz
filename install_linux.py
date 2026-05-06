#!/usr/bin/env python3
"""
Linux installer for TRELLIS.2-stableprojectorz

Adapted from the Windows install.py to work on Linux with:
  - PyTorch built for CUDA 12.4 (or 12.8)
  - Packages built from source instead of Windows-only wheels
  - Proper triton (not triton-windows)
  - Python 3.10+ support

Usage:
    python install_linux.py [--cuda-version 12.4] [--torch-version 2.6.0] [--skip-models] [--skip-hf]

Requirements:
    - Python 3.10, 3.11, or 3.12
    - NVIDIA GPU with CUDA support
    - CUDA Toolkit (nvcc) installed and on PATH
    - C++ compiler (g++ or clang++)
    - libjpeg-dev, libopenexr-dev
"""

import subprocess
import sys
import os
import time
import platform
from typing import Optional, Tuple
from pathlib import Path
import urllib.request
import urllib.error
import socket
import shutil
import argparse

MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds


class InstallationError(Exception):
    """Custom exception for installation failures"""
    pass


def get_current_script_dir() -> Path:
    """Helper to get the directory of the current script."""
    try:
        return Path(__file__).parent.resolve()
    except NameError:
        return Path(os.getcwd()).resolve()


def check_python_version():
    """Ensure we are running on a supported Python version."""
    current = sys.version_info[:2]
    if current < (3, 10) or current >= (3, 13):
        print(f"Error: This installer requires Python 3.10, 3.11, or 3.12")
        print(f"You are currently using Python {current[0]}.{current[1]}")
        sys.exit(1)
    print(f"[OK] Python {current[0]}.{current[1]} detected.")


def check_cuda_toolkit():
    """Check if CUDA toolkit is available."""
    try:
        result = subprocess.run(
            ["nvcc", "--version"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            version_line = [l for l in result.stdout.split('\n') if 'release' in l]
            if version_line:
                print(f"[OK] CUDA Toolkit found: {version_line[0].strip()}")
            return True
    except FileNotFoundError:
        pass

    # Check CUDA_HOME
    cuda_home = os.environ.get('CUDA_HOME', os.environ.get('CUDA_PATH', ''))
    if cuda_home and os.path.isfile(os.path.join(cuda_home, 'bin', 'nvcc')):
        print(f"[OK] CUDA Toolkit found at CUDA_HOME={cuda_home}")
        return True

    print("[WARNING] CUDA Toolkit (nvcc) not found on PATH.")
    print("  The CUDA packages (nvdiffrast, o-voxel, cumesh, flex_gemm) will fail to build.")
    print("  Install CUDA Toolkit and ensure 'nvcc' is on your PATH, or set CUDA_HOME.")
    return False


def check_system_deps():
    """Check for required system libraries."""
    missing = []
    for lib_name, test_file in [
        ("libjpeg-dev", "/usr/include/jpeglib.h"),
        ("libopenexr-dev", "/usr/include/OpenEXR/OpenEXRConfig.h"),
    ]:
        if not os.path.isfile(test_file):
            # Try alternative locations
            alt = subprocess.run(
                f"dpkg -l {lib_name} 2>/dev/null | grep -q '^ii'",
                shell=True, capture_output=True
            )
            if alt.returncode != 0:
                missing.append(lib_name)

    if missing:
        print(f"[WARNING] Missing system packages: {', '.join(missing)}")
        print(f"  Install with: sudo apt install -y {' '.join(missing)}")
        return False
    print("[OK] Required system libraries present.")
    return True


def check_connectivity(url: str = "https://pytorch.org", timeout: int = 5) -> Tuple[bool, Optional[str]]:
    """Check internet connectivity."""
    try:
        urllib.request.urlopen(url, timeout=timeout)
        return True, None
    except urllib.error.URLError as e:
        reason = getattr(e, 'reason', str(e))
        if isinstance(reason, socket.gaierror):
            return False, f"DNS resolution failed: {reason}"
        elif isinstance(reason, socket.timeout) or 'timed out' in str(e):
            return False, "Connection timed out"
        else:
            return False, f"Connection failed: {reason}"
    except Exception as e:
        return False, f"Unknown error: {str(e)}"


def run_command(cmd: str, desc: Optional[str] = None, max_retries: int = MAX_RETRIES, fatal: bool = True,
                cwd: Optional[str] = None) -> subprocess.CompletedProcess:
    """Run a command with retry logic."""
    last_error = None

    if "pip install" in cmd and "--progress-bar" not in cmd:
        cmd += " --progress-bar=on"

    for attempt in range(max_retries):
        try:
            if attempt > 0:
                print(f"\nRetry attempt {attempt + 1}/{max_retries} for: {desc or cmd}")
                connected, error_msg = check_connectivity()
                if not connected:
                    print(f"Waiting {RETRY_DELAY} seconds before retry...")
                    time.sleep(RETRY_DELAY)
                    continue

            result = subprocess.run(
                cmd, shell=True, text=True,
                stdout=sys.stdout, stderr=subprocess.PIPE,
                cwd=cwd
            )

            if result.returncode == 0:
                return result

            last_error = result
            print(f"\nCommand failed (attempt {attempt + 1}/{max_retries}):")
            if hasattr(result, 'stderr') and result.stderr:
                print(f"Error output:\n{result.stderr}")

        except Exception as e:
            last_error = e
            print(f"\nException during {desc or cmd} (attempt {attempt + 1}/{max_retries}):")
            print(str(e))

        if attempt < max_retries - 1:
            time.sleep(RETRY_DELAY)

    if fatal:
        raise InstallationError(f"Command failed after {max_retries} attempts: {last_error}")
    else:
        print(f"Warning: Command '{desc}' failed. Continuing...")
        return last_error


def pip_install(packages: str, desc: Optional[str] = None, **kwargs):
    """Shortcut for pip install commands."""
    cmd = f"{sys.executable} -m pip install --no-cache-dir {packages}"
    run_command(cmd, desc=desc or f"Installing {packages}", **kwargs)


def download_models():
    """Download and extract model zips from GitHub releases if not already present."""
    CODE_DIR = get_current_script_dir()
    models_dir = CODE_DIR / "MODELS"
    models_dir.mkdir(exist_ok=True)

    MODELS = [
        {
            "name": "dinov3",
            "urls": [
                "https://github.com/IgorAherne/TRELLIS.2-stableprojectorz/releases/download/extra-models/dinov3.zip",
                "https://sourceforge.net/projects/trellis-2-stableprojectorz/files/extra-models/dinov3.zip/download",
            ],
            "check_file": models_dir / "dinov3" / "model.safetensors",
        },
        {
            "name": "RMBG-2.0",
            "urls": [
                "https://github.com/IgorAherne/TRELLIS.2-stableprojectorz/releases/download/extra-models/RMBG-2.0.zip",
                "https://sourceforge.net/projects/trellis-2-stableprojectorz/files/extra-models/RMBG-2.0.zip/download",
            ],
            "check_file": models_dir / "RMBG-2.0" / "model.safetensors",
        },
    ]

    for model in MODELS:
        if model["check_file"].exists():
            print(f"[INFO] {model['name']} already present, skipping download.")
            continue

        zip_path = models_dir / f"{model['name']}.zip"
        downloaded = False

        for url_idx, url in enumerate(model["urls"]):
            if downloaded:
                break
            mirror_label = f"mirror {url_idx + 1}/{len(model['urls'])}"
            print(f"\nDownloading {model['name']} from {mirror_label}: {url}")

            for attempt in range(MAX_RETRIES):
                try:
                    if attempt > 0:
                        print(f"  Retry attempt {attempt + 1}/{MAX_RETRIES}...")
                        time.sleep(RETRY_DELAY)

                    def _reporthook(block_num, block_size, total_size):
                        dl = block_num * block_size
                        if total_size > 0:
                            pct = min(dl * 100 / total_size, 100)
                            mb_down = dl / (1024 * 1024)
                            mb_total = total_size / (1024 * 1024)
                            print(f"\r  {mb_down:.1f}/{mb_total:.1f} MB ({pct:.0f}%)", end="", flush=True)

                    urllib.request.urlretrieve(url, str(zip_path), reporthook=_reporthook)
                    print()
                    downloaded = True
                    break
                except Exception as e:
                    print(f"\n  Download failed: {e}")
                    if zip_path.exists():
                        zip_path.unlink()
                    if attempt == MAX_RETRIES - 1:
                        print(f"  All {MAX_RETRIES} attempts failed for {mirror_label}. Trying next mirror...")

        if not downloaded:
            raise InstallationError(
                f"Failed to download {model['name']} from all {len(model['urls'])} mirrors"
            )

        print(f"  Extracting {model['name']}...")
        import zipfile
        with zipfile.ZipFile(str(zip_path), 'r') as zf:
            zf.extractall(str(models_dir))
        zip_path.unlink()

        if not model["check_file"].exists():
            raise InstallationError(
                f"Extraction succeeded but {model['check_file'].name} not found."
            )
        print(f"  {model['name']} ready.")


def download_hf_models():
    """Pre-download HuggingFace model weights."""
    print("\n=============================================================================")
    print(" Downloading TRELLIS.2 model weights from HuggingFace.")
    print(" This may take 10-30 minutes on first install (~20GB total). Please wait")
    print(  "=============================================================================")

    from huggingface_hub import snapshot_download

    CODE_DIR = get_current_script_dir()
    cache_dir = str(CODE_DIR / "models" / "hub")

    repos = [
        "microsoft/TRELLIS.2-4B",
        "microsoft/TRELLIS-image-large",
    ]

    for repo_id in repos:
        print(f"  Downloading {repo_id}...")
        for attempt in range(MAX_RETRIES):
            try:
                snapshot_download(
                    repo_id,
                    cache_dir=cache_dir,
                    resume_download=True,
                )
                print(f"  {repo_id} ready.")
                break
            except Exception as e:
                print(f"  Download failed (attempt {attempt + 1}/{MAX_RETRIES}): {e}")
                if attempt < MAX_RETRIES - 1:
                    print(f"  Retrying in {RETRY_DELAY} seconds...")
                    time.sleep(RETRY_DELAY)
                else:
                    raise InstallationError(f"Failed to download {repo_id} after {MAX_RETRIES} attempts")


def _gpu_supports_flash_attn():
    """Check if GPU supports Flash Attention (requires Ampere / sm_80+)."""
    try:
        result = subprocess.run(
            [sys.executable, "-c",
             "import torch; major, _ = torch.cuda.get_device_capability(); print(major >= 8)"],
            capture_output=True, text=True, timeout=30
        )
        return result.stdout.strip() == 'True'
    except Exception:
        return True  # assume supported if detection fails


def install_dependencies(cuda_version: str = "12.4", torch_version: str = "2.6.0",
                         skip_models: bool = False, skip_hf: bool = False):
    """Install TRELLIS 2 dependencies for Linux."""
    CODE_DIR = get_current_script_dir()
    check_python_version()
    check_cuda_toolkit()

    try:
        connected, error_msg = check_connectivity()
        if not connected:
            print(f"Error: Internet connectivity check failed: {error_msg}")
            sys.exit(1)

        # 0. Download model weights (dinov3, RMBG-2.0) if not already present
        if not skip_models:
            download_models()

        # 1. PyTorch + CUDA
        # Determine PyTorch version strings
        if torch_version == "2.6.0":
            torchvision_version = "0.21.0"
            torchaudio_version = "2.6.0"
            xformers_version = "0.0.29.post3"
        elif torch_version == "2.8.0":
            torchvision_version = "0.23.0"
            torchaudio_version = "2.8.0"
            xformers_version = "0.0.32.post2"
        else:
            torchvision_version = ""
            torchaudio_version = ""
            xformers_version = ""

        cu_tag = f"cu{cuda_version.replace('.', '')}"
        torch_index = f"https://download.pytorch.org/whl/{cu_tag}"

        print(f"\n--- Installing PyTorch {torch_version} (CUDA {cuda_version}) ---")
        pip_install(
            f"torch=={torch_version} torchvision=={torchvision_version} torchaudio=={torchaudio_version} "
            f"--index-url {torch_index}",
            desc="Installing PyTorch"
        )

        # 2. General Dependencies
        print("\n--- Installing General Dependencies ---")
        general_deps = [
            "imageio", "imageio-ffmpeg", "tqdm", "easydict", "opencv-python-headless",
            "ninja", "trimesh", "transformers", "gradio==6.0.1", "tensorboard",
            "pandas", "lpips", "zstandard", "kornia", "timm",
            "huggingface_hub", "accelerate", "psutil",
            # Linux uses standard triton (NOT triton-windows)
            "triton",
        ]
        pip_install(" ".join(general_deps), desc="Installing pip packages")

        # 2.1. Install utils3d from Git
        print("\n--- Installing utils3d ---")
        pip_install(
            "git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8",
            desc="Installing utils3d (Git)"
        )

        # 2.2. Install xformers (fallback attention for pre-Ampere GPUs)
        if xformers_version:
            print(f"\n--- Installing xformers {xformers_version} ---")
            pip_install(
                f"xformers=={xformers_version} --index-url {torch_index}",
                desc="Installing xformers"
            )

        # 2.3. Install flash-attn (requires Ampere+ GPU)
        print("\n--- Installing flash-attn ---")
        if _gpu_supports_flash_attn():
            pip_install("flash-attn==2.7.3", desc="Installing flash-attn", fatal=False)
        else:
            print("  GPU does not support Flash Attention (pre-Ampere). Using xformers instead.")
            print("  You can set ATTN_BACKEND=xformers before running the app.")

        # 2.4. Pillow (use SIMD on Linux if available, else standard)
        print("\n--- Configuring Pillow ---")
        # Try Pillow-SIMD first (requires libjpeg-dev)
        try:
            subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "pillow"],
                           capture_output=True)
            pip_install("pillow-simd", desc="Installing Pillow-SIMD", fatal=False)
            # Verify
            check = subprocess.run(
                [sys.executable, "-c", "from PIL import Image; print('Pillow-SIMD OK')"],
                capture_output=True, text=True
            )
            if check.returncode != 0:
                print("  Pillow-SIMD failed, falling back to standard Pillow...")
                subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "Pillow-SIMD"],
                               capture_output=True)
                pip_install("pillow", desc="Installing standard Pillow")
            else:
                print("  Pillow-SIMD installed successfully.")
        except Exception:
            print("  Falling back to standard Pillow...")
            pip_install("pillow", desc="Installing standard Pillow")

        # 2.5. Pre-download HuggingFace model weights
        if not skip_hf:
            download_hf_models()

        # =====================================================================
        # 3. Install CUDA packages (prebuilt wheels preferred, source fallback)
        # =====================================================================
        print("\n" + "=" * 70)
        print(" Installing CUDA-accelerated packages")
        print(" (Prebuilt wheels from visualbruno/ComfyUI-Trellis2 preferred)")
        print(" (Falls back to building from source if no matching wheel found)")
        print("=" * 70)

        py_version = f"cp{sys.version_info.major}{sys.version_info.minor}"

        # Map torch versions to wheel subdirectory names from visualbruno repo
        # Torch270 = torch 2.7.0, Torch291 = torch 2.9.1, Torch2110 = torch 2.11.0
        torch_wheel_dir_map = {
            "2.7.0": "torch270_cp312",
            "2.9.1": "torch291_cp312",
            "2.11.0": "torch2110_cp313",
        }

        # Find the best matching local wheel directory
        whl_base = CODE_DIR / "whl" / "linux"
        wheel_dir = None
        wheel_source = "local"

        # Try exact match first
        if torch_version in torch_wheel_dir_map:
            candidate = whl_base / torch_wheel_dir_map[torch_version]
            if candidate.exists() and any(candidate.glob("*.whl")):
                wheel_dir = candidate

        # Try to find any directory matching our Python version
        if wheel_dir is None:
            for subdir in sorted(whl_base.iterdir()) if whl_base.exists() else []:
                if subdir.is_dir() and py_version in subdir.name:
                    wheel_dir = subdir
                    break

        # Try downloading from visualbruno's GitHub repo
        if wheel_dir is None:
            print("\n  No local prebuilt wheels found for this Python/PyTorch combo.")
            print("  Attempting to download from visualbruno/ComfyUI-Trellis2...")
            github_dirs = {
                "2.7.0": ("Torch270", "cp312"),
                "2.9.1": ("Torch291", "cp312"),
                "2.11.0": ("Torch2110", "cp313"),
            }
            if torch_version in github_dirs:
                gh_dir, gh_cp = github_dirs[torch_version]
                if gh_cp == py_version:
                    download_dir = whl_base / f"torch{torch_version.replace('.','_')}_{gh_cp}"
                    download_dir.mkdir(parents=True, exist_ok=True)
                    gh_base = f"https://raw.githubusercontent.com/visualbruno/ComfyUI-Trellis2/main/wheels/Linux/{gh_dir}"
                    # Try downloading each wheel
                    wheel_names = ["nvdiffrast", "nvdiffrec_render", "cumesh", "flex_gemm", "o_voxel"]
                    for wname in wheel_names:
                        # Find the exact filename from GitHub API
                        try:
                            api_result = subprocess.run(
                                f"curl -s https://api.github.com/repos/visualbruno/ComfyUI-Trellis2/contents/wheels/Linux/{gh_dir}",
                                shell=True, capture_output=True, text=True, timeout=15
                            )
                            if api_result.returncode == 0:
                                import json
                                files = json.loads(api_result.stdout)
                                for f in files:
                                    if f["name"].startswith(wname) and f["name"].endswith(".whl"):
                                        print(f"    Downloading {f['name']}...")
                                        urllib.request.urlretrieve(f["download_url"], str(download_dir / f["name"]))
                                        break
                        except Exception as e:
                            print(f"    Failed to download {wname}: {e}")
                    if any(download_dir.glob("*.whl")):
                        wheel_dir = download_dir

        # Install from prebuilt wheels
        # IMPORTANT: Install in dependency order — cumesh and flex_gemm must come
        # before o_voxel (o_voxel depends on both). We also install them with
        # --no-deps first, then re-install o_voxel with deps so pip can resolve
        # transitive dependencies like plyfile, trimesh, etc.
        cuda_packages_installed = False
        if wheel_dir is not None:
            print(f"\n--- Installing CUDA packages from prebuilt wheels ({wheel_dir.name}) ---")
            whl_files = sorted(wheel_dir.glob("*.whl"))

            # Define install order: dependencies first, then dependents
            # cumesh and flex_gemm must be installed before o_voxel
            priority_order = ['cumesh', 'flex_gemm', 'nvdiffrast', 'nvdiffrec_render',
                              'custom_rasterizer', 'o_voxel']
            def sort_key(p):
                name = p.stem.split('-')[0].lower()
                try:
                    return priority_order.index(name)
                except ValueError:
                    return len(priority_order)

            whl_files_sorted = sorted(whl_files, key=sort_key)

            for whl in whl_files_sorted:
                print(f"  Installing: {whl.name}")
                # Use --no-deps for CUDA packages to avoid pip trying to fetch
                # git dependencies from o_voxel's metadata (even though we fixed
                # the wheels, this is extra safety)
                run_command(
                    f'{sys.executable} -m pip install --no-cache-dir --no-deps "{whl}"',
                    desc=f"Installing {whl.name}",
                    fatal=False
                )

            # Now install o_voxel's non-CUDA dependencies (plyfile, trimesh, etc.)
            o_voxel_whl = [w for w in whl_files_sorted if w.stem.startswith('o_voxel')]
            if o_voxel_whl:
                print("  Installing o_voxel runtime dependencies (plyfile, trimesh, etc.)...")
                pip_install("plyfile trimesh zstandard easydict", desc="o_voxel dependencies", fatal=False)

            cuda_packages_installed = len(whl_files) > 0

        # Fallback: build from source
        if not cuda_packages_installed:
            print("\n  No prebuilt wheels matched. Falling back to building from source...")
            print("  (This may take 20-60 minutes depending on your hardware)")
            build_dir = Path("/tmp/trellis2_extensions")
            build_dir.mkdir(exist_ok=True)

            print("\n--- Building nvdiffrast from source ---")
            nvdiffrast_dir = build_dir / "nvdiffrast"
            if not nvdiffrast_dir.exists():
                run_command(
                    f"git clone -b v0.4.0 https://github.com/NVlabs/nvdiffrast.git {nvdiffrast_dir}",
                    desc="Cloning nvdiffrast"
                )
            pip_install(str(nvdiffrast_dir), desc="Installing nvdiffrast", fatal=False)

            print("\n--- Building nvdiffrec from source ---")
            nvdiffrec_dir = build_dir / "nvdiffrec"
            if not nvdiffrec_dir.exists():
                run_command(
                    f"git clone -b renderutils https://github.com/IgorAherne/nvdiffrec.git {nvdiffrec_dir}",
                    desc="Cloning nvdiffrec"
                )
            pip_install(str(nvdiffrec_dir), desc="Installing nvdiffrec", fatal=False)

            print("\n--- Building CuMesh from source ---")
            cumesh_dir = build_dir / "CuMesh"
            if not cumesh_dir.exists():
                run_command(
                    f"git clone --recursive https://github.com/JeffreyXiang/CuMesh.git {cumesh_dir}",
                    desc="Cloning CuMesh"
                )
            pip_install(str(cumesh_dir), desc="Installing CuMesh", fatal=False)

            print("\n--- Building FlexGEMM from source ---")
            flexgemm_dir = build_dir / "FlexGEMM"
            if not flexgemm_dir.exists():
                run_command(
                    f"git clone --recursive https://github.com/JeffreyXiang/FlexGEMM.git {flexgemm_dir}",
                    desc="Cloning FlexGEMM"
                )
            pip_install(str(flexgemm_dir), desc="Installing FlexGEMM", fatal=False)

            print("\n--- Building o-voxel from source ---")
            ovoxel_src = CODE_DIR / "o-voxel"
            if ovoxel_src.exists():
                ovoxel_build = build_dir / "o-voxel"
                if ovoxel_build.exists():
                    shutil.rmtree(ovoxel_build)
                shutil.copytree(str(ovoxel_src), str(ovoxel_build))
                pip_install(str(ovoxel_build), desc="Installing o-voxel", fatal=False)
            else:
                print("  [WARNING] o-voxel source not found. Run: git submodule update --init --recursive")

        print("\n" + "=" * 70)
        print(" Installation completed successfully!")
        print("=" * 70)

    except InstallationError as e:
        print(f"\nInstallation failed: {str(e)}")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def verify_installation():
    """Verify installation."""
    print("\n--- Verification ---")
    all_ok = True
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA Available: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU: {torch.cuda.get_device_name(0)}")

        modules_to_check = {
            "nvdiffrast": "nvdiffrast",
            "o_voxel": "o_voxel",
            "flash_attn": "flash_attn",
            "flex_gemm": "flex_gemm",
            "cumesh": "cumesh",
            "xformers": "xformers",
            "trimesh": "trimesh",
            "PIL": "Pillow",
        }

        for mod, label in modules_to_check.items():
            try:
                __import__(mod)
                print(f"  [OK] {label}")
            except ImportError:
                print(f"  [WARNING] {label} not found (may be optional)")

        # Explicit PIL Check
        try:
            from PIL import Image
            pil_version = getattr(Image, '__version__', 'unknown')
            print(f"  [OK] PIL version: {pil_version}")
        except ImportError:
            print("  [ERROR] PIL (Pillow) not found! This is required.")
            all_ok = False

        # Check trellis2 module
        try:
            from trellis2.pipelines import Trellis2ImageTo3DPipeline
            print("  [OK] trellis2 pipeline")
        except ImportError as e:
            print(f"  [WARNING] trellis2 pipeline import failed: {e}")

        return all_ok
    except ImportError as e:
        print(f"Verification failed: {str(e)}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Linux installer for TRELLIS.2-stableprojectorz")
    parser.add_argument("--cuda-version", default="12.4", choices=["12.4", "12.8"],
                        help="CUDA version for PyTorch (default: 12.4)")
    parser.add_argument("--torch-version", default="2.6.0",
                        help="PyTorch version (default: 2.6.0)")
    parser.add_argument("--skip-models", action="store_true",
                        help="Skip downloading dinov3 and RMBG models")
    parser.add_argument("--skip-hf", action="store_true",
                        help="Skip downloading HuggingFace model weights")
    args = parser.parse_args()

    # Ensure we're on Linux
    if platform.system() != "Linux":
        print(f"Warning: This installer is designed for Linux. You are on {platform.system()}.")
        print("For Windows, use install.py instead.")
        response = input("Continue anyway? [y/N] ")
        if response.lower() != 'y':
            sys.exit(0)

    install_dependencies(
        cuda_version=args.cuda_version,
        torch_version=args.torch_version,
        skip_models=args.skip_models,
        skip_hf=args.skip_hf,
    )

    if verify_installation():
        print("\nInstallation completed and verified!")
        print("You can now run: python app.py")
    else:
        print("\nInstallation completed but some verifications failed.")
        print("Check the warnings above. Some packages may need a GPU to import correctly.")
