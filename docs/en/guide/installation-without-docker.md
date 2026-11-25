# Installation without Docker

::: warning
This guide mirrors `docker/Dockerfile` at commit `7354e4fa48ca154b35bf3a51d2a4c66d42997a30` (2026-11-25). Relax pins its CUDA stack tightly and the pins move often — before running anything below, diff the versions and commit hashes against the `docker/Dockerfile` in your own checkout. The Dockerfile is authoritative; if a step here disagrees with it, follow the Dockerfile and open an issue.
:::

Relax is normally built as a multi-stage Docker image. This guide reproduces the same stages on a host or an HPC node that cannot run containers. It is a build map, not a one-shot script: several stages exist only to produce wheels that the final stage installs, and the image expects specific commits and patch files from this repository.

## What the image does

| Stage | Purpose |
|-------|---------|
| `base` | NGC 25.06 base: PyTorch 2.9.0 (cu129), Python 3.12, system packages |
| `sglang` | SGLang 0.5.21 checkout plus a pinned cuDNN wheel |
| `train` | Training CUDA extensions: flash-attn, FLA, tilelang, DeepSeek-V4 DSA deps, FA3, Transformer Engine, apex, INT4 QAT |
| `hybridep_builder` | HybridEP (DeepEP fork) wheel, built against the same Torch ABI |
| `router_builder` | `sglang-router` wheel carrying the R3 rollout-routing and PD/DP-aware patches |
| `tree_core_builder` | SGLang HiCache tree-core binding |
| `relax` | Megatron-LM via Megatron-Bridge, Python dependencies, TransferQueue, FLA 0.5.2, repository patches, editable SGLang |
| final | cu129 runtime requirements, cuDNN Frontend DSA patch, torch_memory_saver symlink, tree-core install |

If you can extract the `ghcr.io/redai-studio/relaxrl:sglang-v0.5.21-cu129` layers without Docker (skopeo plus umoci on a machine that has them), do that first — it gives you the exact base stage, including NVIDIA's PyTorch build. The manual path below starts from a plain Ubuntu 22.04 + CUDA 12.9 host.

## Prerequisites

- NVIDIA GPU with compute capability >= 8.0 (Ampere or newer); the FA3 and DSA fast paths assume Hopper.
- >= 24 GB GPU memory for the small-model path (Qwen3-0.6B / 4B).
- >= 80 GB free disk for sources, wheels, models and checkpoints.
- Ubuntu 22.04, Python 3.12, CUDA Toolkit 12.9, NVIDIA driver >= 560, GCC 11.x, CMake >= 3.18, Git >= 2.30, rsync. Rust 1.97.1 only for the router stage.

Environment variables used throughout:

```bash
export CUDA_HOME=/usr/local/cuda-12.9
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export MAX_JOBS=8        # raise on big machines, but watch RAM during extension builds
export RELAX_SRC=/path/to/Relax
```

## Step 1 — System packages

```bash
sudo apt-get update && sudo apt-get install -y --no-install-recommends \
    build-essential patchelf tmux net-tools htop nano iputils-ping \
    autoconf automake libatlas-base-dev libgoogle-glog-dev libbz2-dev libc-ares2 \
    libleveldb-dev liblmdb-dev libprotobuf-dev libsnappy-dev libtool nasm protobuf-compiler pkg-config unzip sox \
    libsndfile1 libpng-dev libhdf5-dev gfortran rapidjson-dev ninja-build libedit-dev rsync

sudo apt-get install -y locales tzdata
sudo locale-gen en_US.UTF-8
export LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8
```

## Step 2 — PyTorch

The image ships NVIDIA's PyTorch 2.9.0 build from the 25.06 NGC container. Outside the image, the closest equivalent is the official cu129 wheel:

```bash
pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cu129
python -c "import torch; print(torch.__version__, torch.version.cuda)"
```

Install `torchvision`/`torchaudio` only if a downstream package needs them, and keep them on the same Torch ABI.

## Step 3 — cuDNN and SGLang

```bash
pip install nvidia-cudnn-cu12==9.20.0.48

git clone https://github.com/sgl-project/sglang.git /sgl-workspace/sglang
cd /sgl-workspace/sglang
git checkout e00930c5489053f26d86b179cee0d087f846acbb   # the image asserts this exact commit
```

SGLang is installed editable in step 11, after its patch is applied. Do not `pip install sglang` from PyPI — the image uses this checkout with `SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SGLANG=0.5.21`.

## Step 4 — Training CUDA extensions

```bash
FLASH_ATTENTION_FORCE_BUILD=TRUE MAX_JOBS=64 \
    pip install flash-attn==2.7.4.post1 --no-build-isolation --no-cache-dir
pip install --no-cache-dir flash-linear-attention==0.4.2
pip install --no-cache-dir tilelang -f https://tile-ai.github.io/whl/nightly/cu128/
```

The Dockerfile can also consume a prebuilt wheel cache (`PREBUILD_CACHE_TAR_URL`) which pins `tilelang==0.1.12` alongside `flash-linear-attention==0.4.2`; without such a cache you get the nightly tilelang build.

## Step 5 — DeepSeek-V4 DSA dependencies

Neither of these is usable from PyPI, so both are built from pinned sources:

```bash
git clone https://github.com/Dao-AILab/fast-hadamard-transform.git /opt/fht
cd /opt/fht && git checkout f134af63deb2df17e1171a9ec1ea4a7d8604d5ca
MAX_JOBS=64 pip install --no-cache-dir --no-build-isolation --no-deps .

git clone --recurse-submodules https://github.com/deepseek-ai/FlashMLA.git /opt/FlashMLA
cd /opt/FlashMLA && git checkout b7643bd54521f563b839b98289b5cd048c062ba2
git submodule update --init --recursive
MAX_JOBS=64 pip install --no-cache-dir --no-build-isolation --no-deps .
cd / && python -c "import flash_mla; assert hasattr(flash_mla, 'flash_mla_sparse_fwd')"
```

Without them, Megatron Core's DSA asserts fail at import or on the first sparse-attention forward.

## Step 6 — FlashAttention 3 (Hopper)

FA3 is built from a specific flash-attention commit and then rearranged so Transformer Engine can import it:

```bash
cd /opt
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
git checkout 0f82fead0b2d59f6390c62a1d74e927159bc5a7b
git submodule update --init
cd hopper
MAX_JOBS=96 python setup.py install

export python_path=$(python -c "import site; print(site.getsitepackages()[0])")
mkdir -p $python_path/flash_attn_3
cp flash_attn_interface.py $python_path/flash_attn_3/flash_attn_interface.py
find $python_path -maxdepth 1 -name flash_attn_interface.py -delete
find $python_path -maxdepth 2 -path '*.egg/flash_attn_interface.py' -delete
cd / && python -c "import importlib.util as u; assert u.find_spec('flash_attn_interface') is None; import flash_attn_config"
```

The deletion is not cosmetic. That commit's `setup.py` also installs a top-level `flash_attn_interface.py`; importing it in the same process re-registers the `flash_attn_3::_flash_attn_forward` custom op, which tears down the library handle Transformer Engine bound to and makes later forwards abort with an internal schema assertion.

## Step 7 — Transformer Engine, memory saver, apex, quantisation extras

```bash
NVTE_PYTORCH_FORCE_BUILD=TRUE \
    pip install -v --no-cache-dir --no-build-isolation "transformer_engine[pytorch]==2.14.1"

TMS_CUDA_MAJOR=$(python -c 'import torch; print(torch.version.cuda.split(".")[0])') \
    pip install git+https://github.com/redai-studio/torch_memory_saver.git@afc13785c50119048e2dd8ac497cc9e29ec75bd4 \
    --no-cache-dir --force-reinstall

pip install "nvidia-modelopt[torch]>=0.37.0" --no-build-isolation --no-cache-dir
pip install "numpy<2" "scipy<1.18" "ml-dtypes<0.6" nvidia-cudnn-cu12==9.20.0.48 --no-cache-dir

NVCC_APPEND_FLAGS="--threads 32" \
    pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation \
    --config-settings "--build-option=--cpp_ext --cuda_ext --parallel 8" \
    git+https://github.com/NVIDIA/apex.git@10417aceddd7d5d05d7cbf7b0fc2daad1105f8b4
```

## Step 8 — INT4 QAT kernel and TE wheel metadata

```bash
cd $RELAX_SRC/relax/backends/megatron/kernels/int4_qat
pip install . --no-build-isolation --no-cache-dir

pip install --no-cache-dir --no-deps pandas==3.0.6 pyarrow==24.0.0
python $RELAX_SRC/docker/scripts/repair_te_wheel_metadata.py \
    $RELAX_SRC/docker/patch/transformer_engine/2.14.1-cu12-wheel-tag.patch
pip check && python -c "import pandas, pyarrow, datasets"
```

The metadata repair re-tags the Transformer Engine wheel so `pip check` stays quiet against the cu129 Torch ABI; skip it only if your TE build already carries the right tag.

## Step 9 — Megatron-LM via Megatron-Bridge

```bash
export MEGATRON_BRIDGE_COMMIT=6862ee170a2569607e8ebc5dde91f119779093ca
export MEGATRON_PATH=/root/Megatron-LM

rm -rf $MEGATRON_PATH /root/Megatron-Bridge
git clone https://github.com/NVIDIA-NeMo/Megatron-Bridge.git /root/Megatron-Bridge
cd /root/Megatron-Bridge
git checkout $MEGATRON_BRIDGE_COMMIT
git submodule update --init --recursive
./scripts/switch_mcore.sh dev

mkdir -p $MEGATRON_PATH
cp -r src/megatron $MEGATRON_PATH/
rsync -avP 3rdparty/Megatron-LM/megatron/ $MEGATRON_PATH/megatron/
rm -rf /root/Megatron-Bridge
export PYTHONPATH=$MEGATRON_PATH:$PYTHONPATH
```

Then apply the repository patch. The image applies only the block between the `# BEGIN common` and `# END common` markers:

```bash
cp $RELAX_SRC/docker/patch/latest/megatron.patch $MEGATRON_PATH/
cd $MEGATRON_PATH
sed -n '/^# BEGIN common$/,/^# END common$/p' megatron.patch | git apply --check -
sed -n '/^# BEGIN common$/,/^# END common$/p' megatron.patch | git apply -
sha256sum megatron.patch > /etc/relax-megatron-patch.sha256
python -m compileall -q megatron
```

`megatron.patch` embeds the upstream commit it was generated against. If `git apply --check` fails, your Megatron-Bridge commit is wrong — fix the commit rather than editing the patch.

## Step 10 — sglang-router and HiCache tree-core

Both are compiled artifacts, not installs from PyPI.

```bash
# router: Rust 1.97.1 + maturin, sources under docker/patch/sglang-router
export ROUTER_RUST_VERSION=1.97.1
export ROUTER_BUILD_JOBS=16
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain $ROUTER_RUST_VERSION
export PATH=/root/.cargo/bin:$PATH
python -m pip install --no-cache-dir "maturin==1.9.6"
bash $RELAX_SRC/docker/scripts/build_sglang_router.sh \
    $RELAX_SRC/docker/patch/sglang-router /opt/router-wheels

# tree-core: inside the pinned SGLang checkout, with its patch applied
cd /sgl-workspace/sglang
test "$(git rev-parse HEAD)" = "e00930c5489053f26d86b179cee0d087f846acbb"
git apply $RELAX_SRC/docker/patch/latest/sglang.patch
python $RELAX_SRC/docker/scripts/build_sglang_tree_core.py build /sgl-workspace/sglang /opt/relax-tree-core
```

The router wheel must report version `0.3.2+slime.dpaware` with source commit `9daabcd801c861088d93b8ce8a2a831490ce4bee`; the image asserts both. A stock `sglang-router` wheel silently drops the `routed_experts` field and breaks R3 rollout routing.

## Step 11 — Python dependencies and Relax itself

```bash
pip install --no-cache-dir "ray[default,serve]==2.56.1"
pip install --ignore-installed PyJWT
pip install -r $RELAX_SRC/requirements.txt --no-cache-dir
pip install --no-cache-dir "compressed_tensors>=0.13.0" tensordict==0.10.0 pyvers==0.1.0 \
    'nvidia-modelopt[hf]==0.44.0' --no-deps

# TransferQueue plus its assignment-transform patch
pip install "transferqueue @ git+https://github.com/redai-studio/TransferQueue.git@9784ad0d8a5f0db93ff8326379ab2a6d06e640a0" --no-deps
TQ_SITE="$(python -c 'import importlib.util; from pathlib import Path; print(Path(importlib.util.find_spec("transfer_queue").origin).parent.parent)')"
patch --batch --forward --fuzz=0 -d "$TQ_SITE" -p1 \
    < $RELAX_SRC/docker/patch/transferqueue/9784ad0-assignment-transform.patch

# async checkpoint API expected by the pinned Megatron Core revision
pip install --no-cache-dir \
    "nvidia-resiliency-ext @ git+https://github.com/NVIDIA/nvidia-resiliency-ext.git@6c5f2a13c7688d92a7ac7ee6e464721eb8b7345d"

# FLA 0.5.2 is the runtime version; the 0.4.2 tree stays available for compat
pip install --no-cache-dir --force-reinstall --no-deps "fla-core==0.5.2" "flash-linear-attention==0.5.2"
cd "$(python -c 'import importlib.util; from pathlib import Path; print(Path(importlib.util.find_spec("fla").origin).parent.parent)')"
patch --batch --forward --fuzz=0 -p1 < $RELAX_SRC/docker/patch/fla/0.5.2.patch
python -m compileall -q fla/modules/conv/triton
export RELAX_FLA_CONV_NB_MODE=auto

python $RELAX_SRC/docker/scripts/install_legacy_fla.py --target /opt/relax/compat/fla-0.4.2 \
    --patch $RELAX_SRC/docker/patch/fla/0.4.2-legacy-runtime.patch \
    --manifest /etc/relax-fla042-runtime.json

# router wheel built in step 10
pip install --no-cache-dir --force-reinstall --no-deps /opt/router-wheels/*.whl

# SGLang, editable, at the pinned commit
pip install --no-cache-dir setuptools-scm==9.2.2
SGLANG_BUILD_RUST_EXTS=none SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SGLANG=0.5.21 \
    pip install --no-cache-dir --no-deps --no-build-isolation -e /sgl-workspace/sglang/python
pip install --no-cache-dir transformers==5.16.0 tokenizers==0.23.2

pip install -e $RELAX_SRC
```

## Step 12 — HybridEP wheel

Only needed if you run hybrid expert parallelism; the image always builds it.

```bash
export HYBRID_EP_COMMIT=eb9cee7de5a24193bf09500668d3a619d3d3f3fb
pip install --no-deps "nvidia-nvshmem-cu12==3.4.5" "pynvml==13.0.1"
git clone --no-checkout https://github.com/deepseek-ai/DeepEP.git /tmp/HybridEP
cd /tmp/HybridEP
git checkout --detach "$HYBRID_EP_COMMIT"
patch --batch --forward --fuzz=0 -p1 < $RELAX_SRC/docker/patch/deepep/eb9cee7-hybridep-namespace.patch
CUDA_VISIBLE_DEVICES="" TORCH_CUDA_ARCH_LIST="9.0" HYBRID_EP_MULTINODE=0 \
    CPATH="/usr/local/cuda/include/cccl${CPATH:+:$CPATH}" \
    LIBRARY_PATH="/usr/local/cuda/lib64/stubs${LIBRARY_PATH:+:$LIBRARY_PATH}" \
    MAX_JOBS=16 pip wheel --no-build-isolation --no-deps --wheel-dir /opt/ep-wheels .
pip install --force-reinstall --no-deps /opt/ep-wheels/*.whl
```

## Step 13 — cu129 runtime layer

```bash
python -c 'import importlib.metadata as m; print("\n".join("{}=={}".format(d.metadata["Name"], d.version) for d in m.distributions() if d.metadata["Name"].lower().replace("_", "-") in ("torch", "torchvision", "torchaudio", "triton", "transformer-engine", "transformer-engine-cu12", "transformer-engine-torch") or d.metadata["Name"].lower().startswith("nvidia-")))' > /tmp/cu12-abi-constraints.txt
pip install --no-cache-dir -c /tmp/cu12-abi-constraints.txt -r $RELAX_SRC/docker/cu12-runtime-requirements.txt
python $RELAX_SRC/docker/scripts/repair_te_wheel_metadata.py \
    $RELAX_SRC/docker/patch/transformer_engine/2.14.1-cu12-wheel-tag.patch

CUDNN_SITE="$(python -c 'import importlib.metadata as m; print(m.distribution("nvidia-cudnn-frontend").locate_file(""))')"
patch --batch --forward --fuzz=0 -d "$CUDNN_SITE" -p1 \
    < $RELAX_SRC/docker/patch/cudnn/1.30.0-dsa-sm90-signature-cache.patch

python $RELAX_SRC/docker/scripts/validate_hybridep_install.py
python $RELAX_SRC/docker/scripts/validate_final_runtime.py --profile cu12

cd "$(python -c 'from pathlib import Path; import torch_memory_saver; print(Path(torch_memory_saver.__file__).parent.parent)')"
ln -sf torch_memory_saver_hook_mode_preload_cu12.abi3.so torch_memory_saver_hook_mode_preload.abi3.so

python $RELAX_SRC/docker/scripts/build_sglang_tree_core.py install /opt/relax-tree-core
```

The cuDNN Frontend patch requires `nvidia-cudnn-frontend==1.31.0+cu12` and `nvidia-cutlass-dsl==4.6.2`; the patching script asserts both before touching files.

## Sanity checks

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python -c "from importlib.metadata import version; [print(n, version(n)) for n in ('sglang','transformers','tokenizers','transformer-engine','flash-linear-attention','fla-core')]"
python -c "from sglang_router import sglang_router_rs; import sglang_router, json, importlib.resources as r; print(json.loads(r.files('sglang_router').joinpath('_relax_build.json').read_text()))"
python -c "import megatron; print(megatron.__file__)"
python $RELAX_SRC/docker/scripts/check_gpu_imports.py
```

::: warning
A clean import sweep is not proof that training works. Ray startup, SGLang engine bring-up, NCCL and the Megatron loop each fail in their own way. Validate with Qwen3-0.6B and a handful of steps before queueing a real run.
:::

## Troubleshooting

- `schema_.has_value() INTERNAL ASSERT FAILED` on actor forward: the stray top-level `flash_attn_interface.py` is still importable. Redo the `find ... -delete` lines from step 6 and re-check `u.find_spec('flash_attn_interface') is None`.
- `flash_mla_sparse_fwd` missing: FlashMLA was installed without its submodules; re-clone with `--recurse-submodules`.
- `pip check` complains about Transformer Engine: run the wheel-tag repair from steps 8 and 13 again.
- `git apply` rejects `sglang.patch` or `megatron.patch`: your SGLang or Megatron-Bridge commit does not match the pins; both patches assume the commits in steps 3 and 9.
- Router import fails, or `_relax_build.json` reports a plain `0.3.2`: a stock PyPI wheel is on the path. Reinstall the wheel from step 10 with `--force-reinstall --no-deps`.
- `nvcc` killed during extension builds: lower `MAX_JOBS`, and build FA3 and apex one at a time.

## Getting help

1. Diff your versions against `docker/Dockerfile` — most failures here are pin drift.
2. Search existing [GitHub Issues](https://github.com/redai-studio/Relax/issues).
3. Open an issue with OS, GPU model, driver, CUDA version, the failing step and the exact error text.

::: tip
Docker remains the recommended path. This guide exists for restricted shared servers and HPC systems without a container runtime, and it will lag the Dockerfile between updates.
:::
