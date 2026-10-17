# Installation without Docker

::: warning
This guide is derived from the official `docker/Dockerfile` and is intended for users who cannot use Docker (e.g., restricted shared servers, HPC environments without container runtime). The exact build steps are complex and version-sensitive — if you encounter compilation errors, please refer to the Dockerfile as the authoritative source and open an issue with your environment details.
:::

## Overview

Relax relies on several heavyweight CUDA extensions that are built from source with specific commits and patches:

- **Megatron-LM** (via Megatron-Bridge, with a custom patch)
- **SGLang** (with a custom patch for R3 rollout routing)
- **flash-attn** (v2 + v3 from a specific commit)
- **transformer-engine** (v2.14.1)
- **apex** (from a specific commit)
- **TransferQueue** (from a specific commit)

The official Docker image (`ghcr.io/redai-infra/relaxrl:latest`) is the recommended and most thoroughly tested installation path. Use this guide only when Docker is unavailable.

## Prerequisites

### Hardware

- NVIDIA GPU with compute capability >= 8.0 (Ampere or newer; Hopper recommended for FA3)
- At least 24 GB GPU memory for small-model training (Qwen3-0.6B / 4B)
- At least 50 GB free disk space for source builds, models, and checkpoints

### Software

| Component | Required Version | Notes |
|-----------|-----------------|-------|
| OS | Ubuntu 22.04 (LTS) | Other Linux distros may work but are untested |
| Python | 3.12 | Exact version used in the official image |
| CUDA Toolkit | 12.9 | Must match PyTorch build |
| NVIDIA Driver | >= 560 | Supports CUDA 12.9 |
| GCC / G++ | 11.x | Required for CUDA extension compilation |
| CMake | >= 3.18 | Required for apex and other builds |
| Git | >= 2.30 | For cloning dependencies |
| rsync | latest | For merging Megatron-Bridge sources |
| jq | latest | For patch verification |

### Environment Variables

Set these before starting:

```bash
export CUDA_HOME=/usr/local/cuda-12.9
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export MAX_JOBS=8  # Adjust based on your CPU cores; higher = faster build, more RAM
```

## Step 1: Install System Dependencies

```bash
sudo apt-get update && sudo apt-get install -y --no-install-recommends \
    build-essential patchelf tmux net-tools htop nano iputils-ping \
    autoconf automake libatlas-base-dev libgoogle-glog-dev libbz2-dev libc-ares2 \
    libleveldb-dev liblmdb-dev libprotobuf-dev libsnappy-dev libtool nasm protobuf-compiler pkg-config unzip sox \
    libsndfile1 libpng-dev libhdf5-dev gfortran rapidjson-dev ninja-build libedit-dev rsync jq \
    locales tzdata
```

## Step 2: Install PyTorch 2.9.0 with CUDA 12.9

The official image uses NVIDIA's PyTorch 2.9.0 build (from the 25.06 NGC container). Install the matching version:

```bash
pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cu129
```

Verify:

```bash
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}, GPU: {torch.cuda.get_device_name(0)}')"
```

## Step 3: Build Core CUDA Extensions

### 3.1 flash-attn v2

```bash
MAX_JOBS=64 pip -v install flash-attn==2.7.4.post1 --no-build-isolation --no-cache-dir
pip install --no-cache-dir flash-linear-attention==0.4.1
pip install --no-cache-dir tilelang -f https://tile-ai.github.io/whl/nightly/cu128/
```

### 3.2 flash-attn v3 (Hopper only)

::: warning
This step is only required for Hopper (H100/H800) GPUs. Skip on Ampere/Ada.
:::

```bash
cd /opt
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention/
git checkout 0f82fead0b2d59f6390c62a1d74e927159bc5a7b
git submodule update --init
cd hopper/
MAX_JOBS=96 python setup.py install

# Remove stray top-level module that conflicts with TE
export python_path=$(python -c "import site; print(site.getsitepackages()[0])")
mkdir -p $python_path/flash_attn_3
cp flash_attn_interface.py $python_path/flash_attn_3/flash_attn_interface.py
find $python_path -maxdepth 1 -name flash_attn_interface.py -delete
find $python_path -maxdepth 2 -path '*.egg/flash_attn_interface.py' -delete

# Verify
cd / && python -c "import importlib.util as u; assert u.find_spec('flash_attn_interface') is None; import flash_attn_config"
rm -rf /opt/flash-attention/
```

### 3.3 transformer-engine

```bash
pip -v install --no-cache-dir --no-build-isolation "transformer_engine[pytorch]==2.14.1"
```

### 3.4 torch_memory_saver

```bash
TMS_CUDA_MAJOR=$(python -c 'import torch; print(torch.version.cuda.split(".")[0])') \
pip install git+https://github.com/redai-infra/torch_memory_saver.git@afc13785c50119048e2dd8ac497cc9e29ec75bd4 --no-cache-dir --force-reinstall
```

### 3.5 apex

```bash
NVCC_APPEND_FLAGS="--threads 32" \
pip -v install --disable-pip-version-check --no-cache-dir \
    --no-build-isolation \
    --config-settings "--build-option=--cpp_ext --cuda_ext --parallel 8" \
    git+https://github.com/NVIDIA/apex.git@10417aceddd7d5d05d7cbf7b0fc2daad1105f8b4
```

### 3.6 Additional packages

```bash
pip install nvidia-modelopt[torch]>=0.37.0 --no-build-isolation --no-cache-dir
pip install "numpy<2" nvidia-cudnn-cu12==9.16.0.29 --no-cache-dir
```

## Step 4: Install Megatron-LM via Megatron-Bridge

Relax uses a merged Megatron-LM tree built from Megatron-Bridge sources and its submodule, with a custom patch.

```bash
export MEGATRON_BRIDGE_COMMIT=2faedbf6fe3c422835a44b2b360cadcb2a116a54
export MEGATRON_PATH=/root/Megatron-LM  # Adjust to your preferred path

rm -rf $MEGATRON_PATH
git clone https://github.com/NVIDIA-NeMo/Megatron-Bridge.git /root/Megatron-Bridge
cd /root/Megatron-Bridge/
git checkout ${MEGATRON_BRIDGE_COMMIT}
git submodule update --init --recursive
./scripts/switch_mcore.sh dev

mkdir -p $MEGATRON_PATH
cp -r src/megatron $MEGATRON_PATH/
rsync -avP 3rdparty/Megatron-LM/megatron/ $MEGATRON_PATH/megatron/
rm -rf /root/Megatron-Bridge
```

### Apply the megatron patch

```bash
# Copy the patch from the Relax repository
cp /path/to/Relax/docker/patch/latest/megatron.patch $MEGATRON_PATH/

cd $MEGATRON_PATH
patch -p1 < megatron.patch

# Verify patch applied cleanly
if grep -R -n '^<<<<<<< ' .; then
    echo "ERROR: Patch failed to apply cleanly. Please resolve conflicts."
    exit 1
fi
rm megatron.patch
```

### Set PYTHONPATH

```bash
export PYTHONPATH=$MEGATRON_PATH:$PYTHONPATH
```

Verify:

```bash
python -c "import megatron; print(f'Megatron imported from {megatron.__file__}')"
```

## Step 5: Install SGLang with Custom Patch

Relax uses SGLang v0.5.15.post1 with a custom patch for R3 (rollout routing replay).

::: tip
The base SGLang installation comes from the `lmsysorg/sglang:v0.5.15.post1-cu129` Docker image. Without Docker, you need to install SGLang from source at the matching version first.
:::

```bash
# Install SGLang from source at the matching version
pip install "sglang[all]==0.5.15.post1"

# Apply the custom sglang patch
export SGLANG_PATH=$(python -c "import sglang; print(sglang.__path__[0])")
cp /path/to/Relax/docker/patch/latest/sglang.patch $SGLANG_PATH/

cd $SGLANG_PATH
git add .
git update-index --refresh
git apply sglang.patch --3way

# Verify
if grep -R -n '^<<<<<<< ' .; then
    echo "ERROR: SGLang patch failed."
    exit 1
fi
rm sglang.patch
```

### Install patched sglang-router

The official `sglang-router` wheel drops the `routed_experts` field, breaking R3. Relax uses slime's patched fork:

```bash
pip install --no-cache-dir --force-reinstall --no-deps \
    https://github.com/zhuzilin/sgl-router/releases/download/v0.3.2-9daabcd/sglang_router-0.3.2-cp38-abi3-manylinux_2_28_x86_64.whl

# Verify the patched version
python -c "import sglang_router; assert 'slime' in sglang_router.__version__, sglang_router.__version__"
```

## Step 6: Install TransferQueue

```bash
pip install "transferqueue @ git+https://github.com/redai-infra/TransferQueue.git@58054a33834aadbcf76aacd6b1e32e25c030f2c9" --no-deps
```

## Step 7: Install Python Dependencies and Relax

```bash
cd /path/to/Relax

# Install pure-Python dependencies
pip install --ignore-installed PyJWT
pip install -r requirements.txt --no-cache-dir

# Install additional packages
pip install --no-cache-dir "compressed_tensors>=0.13.0" tensordict==0.10.0 pyvers==0.1.0 'nvidia-modelopt[hf]==0.44.0' --no-deps

# Install Relax in development mode
pip install -e .
```

## Step 8: Build INT4 QAT CUDA Extension

```bash
cd /path/to/Relax/relax/backends/megatron/kernels/int4_qat
pip install . --no-build-isolation --no-cache-dir
```

## Environment Variables Summary

Add these to your `~/.bashrc` or a dedicated environment script:

```bash
# CUDA
export CUDA_HOME=/usr/local/cuda-12.9
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Megatron
export MEGATRON_PATH=/path/to/Megatron-LM
export PYTHONPATH=$MEGATRON_PATH:$PYTHONPATH

# Relax
export RELAX=/path/to/Relax
export PYTHONPATH=$RELAX:$PYTHONPATH
```

## Verifying the Installation

Run these checks in order:

```bash
# 1. Basic imports
python -c "
import torch
import ray
import sglang
import megatron
import relax
print('All basic imports successful')
print(f'  torch: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
print(f'  GPU: {torch.cuda.get_device_name(0)}')
"

# 2. CUDA extensions
python -c "
import flash_attn
import transformer_engine
print('CUDA extensions imported successfully')
print(f'  flash_attn: {flash_attn.__version__}')
"

# 3. Relax components
python -c "
from relax.controller import Controller
from relax.registry import ALGOS
print(f'Registered algorithms: {list(ALGOS.keys())}')
print('Relax components loaded successfully')
"
```

::: warning
Passing these import checks does not guarantee that full training will work. End-to-end training involves Ray cluster initialization, SGLang engine startup, NCCL communication, and Megatron training loops — each of which may have environment-specific issues. Start with a small model (Qwen3-0.6B) and a few training steps to validate.
:::

## Troubleshooting

### Compilation errors during flash-attn / apex / TE builds

- Ensure `CUDA_HOME` is set correctly and `nvcc --version` shows 12.9
- Ensure GCC version is 11.x (`gcc --version`)
- Reduce `MAX_JOBS` if you run out of RAM during compilation
- Check that your NVIDIA driver supports CUDA 12.9 (`nvidia-smi`)

### `ModuleNotFoundError: No module named 'megatron'`

- Ensure `PYTHONPATH` includes your `$MEGATRON_PATH`
- Verify the merged directory exists: `ls $MEGATRON_PATH/megatron/`

### `ModuleNotFoundError: No module named 'sglang'`

- SGLang must be installed before applying the patch
- Verify: `python -c "import sglang; print(sglang.__version__)"` should show 0.5.15.post1

### NCCL errors or Ray cluster failures

- Ensure `--network=host` equivalent: all nodes must be reachable
- Set `NCCL_SOCKET_IFNAME` to the correct network interface
- For single-node training, ensure `ray start --head` can bind to localhost

### Patch conflicts

If `git apply` or `patch` reports conflicts:

1. Check that you have the exact base version (SGLang 0.5.15.post1, Megatron-Bridge commit 2faedbf6)
2. Try `git apply --3way` for three-way merge
3. If conflicts persist, refer to the Dockerfile for the exact patch context

## Getting Help

If you encounter issues not covered here:

1. Check the [official Dockerfile](https://github.com/redai-infra/Relax/blob/main/docker/Dockerfile) — it is the authoritative build reference
2. Search existing [GitHub Issues](https://github.com/redai-infra/Relax/issues)
3. Open a new issue with:
   - Your OS, GPU model, NVIDIA driver version, CUDA version
   - The exact error message and stack trace
   - The step at which the failure occurred
   - Whether you are using the exact versions specified in this guide

::: tip
For most users, the Docker image remains the recommended installation path. This guide is provided as a reference for constrained environments and may not be updated as frequently as the Dockerfile.
:::
