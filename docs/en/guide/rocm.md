# AMD ROCm

Relax runs on AMD Instinct GPUs through a ROCm Docker image built from `docker/Dockerfile.rocm`. The training scripts, launchers, and configuration are the same as on NVIDIA GPUs; the image supplies a ROCm build of the software stack underneath them:

- **Base**: SGLang's official ROCm 10 image (Ubuntu 24.04, ROCm 10 SDK, PyTorch 2.11 for ROCm, aiter kernels), pinned by digest.
- **SGLang 0.5.21** with the same Relax patch, Rust radix-tree cache, and DP-aware router as the CUDA 13 image, plus a ROCm fix from `docker/patch/rocm/`.
- **Megatron** built from the same Megatron-Bridge commit and patch sequence as the CUDA 13 image.
- **Transformer Engine, apex, and flash-attn** as prebuilt ROCm wheels, checked against SHA-256 digests during the build.

## Supported GPUs

The `GPU_ARCH` build argument selects the GPU generation. Each value uses a matching SGLang base image and set of prebuilt wheels.

| `GPU_ARCH` | GPUs | SGLang base image |
| --- | --- | --- |
| `gfx950` (default) | MI350X, MI355X | `lmsysorg/sglang:v0.5.21-rocm10-mi35x` |
| `gfx942` | MI300X, MI325X | `lmsysorg/sglang:v0.5.21-rocm10-mi30x` |

Both targets come from the same Dockerfile. The `gfx942` variant has not yet been validated on MI300-series hardware, so check a short run before relying on it for long jobs.

## Build the Image

Run the build from the repository root. The build does not need a GPU, but it downloads sources and packages from GitHub, PyPI, and the Rust toolchain servers, and the resulting image is about 50 GB.

```bash
cd Relax

# MI350X / MI355X
DOCKER_BUILDKIT=1 docker build -f docker/Dockerfile.rocm -t relax:dev-rocm-gfx950 .

# MI300X / MI325X
DOCKER_BUILDKIT=1 docker build -f docker/Dockerfile.rocm --build-arg GPU_ARCH=gfx942 -t relax:dev-rocm-gfx942 .
```

Docker's standard proxy arguments (`--build-arg HTTP_PROXY=...`, `HTTPS_PROXY`, `NO_PROXY`) apply when the build host needs a proxy.

The build stops at the `sha256sum -c SHA256SUMS` step if a prebuilt wheel no longer matches its recorded digest. That failure means the upstream release asset changed; review the new file before updating the digest in `docker/Dockerfile.rocm`.

## Start a Container

Mount the repository and your model and data directories, and pass the ROCm devices into the container:

```bash
docker run -it --init \
  --device /dev/kfd --device /dev/dri \
  --group-add video --group-add render \
  --ipc host --network host \
  --security-opt seccomp=unconfined \
  -v $(pwd)/Relax:/root/Relax \
  -v /path/to/models_and_data:/root/exp \
  relax:dev-rocm-gfx950 /bin/bash
```

`/dev/kfd` and `/dev/dri`, together with the `video` and `render` groups, give the container access to the AMD GPUs; the ROCm equivalent of `--gpus all`. `--ipc host` provides the shared memory that PyTorch and colocated training use between processes, and `--network host` lets Ray and the SGLang engines use the host's network directly. `--init` runs a small init process that reaps exited Ray workers, which otherwise accumulate as defunct processes during long jobs.

Examples that run agent tools inside Apptainer sandboxes, such as [Mini-SWE-Agent](https://github.com/redai-studio/Relax/tree/main/examples/mini_swe_agent), also need `--privileged` so that Apptainer can mount its images.

Inside the container, confirm that PyTorch sees the GPUs:

```bash
rocm-smi
python -c "import torch; print(torch.cuda.device_count(), torch.version.hip)"
```

PyTorch for ROCm keeps the `torch.cuda` API, so the first number is the GPU count and the second is the HIP version.

## Run Training

Launch training with the same scripts and environment variables as in the [Quick Start](./quick-start.md). For example, with the Qwen3-4B model and DAPO-Math data under `/root/exp`:

```bash
cd /root/Relax
export EXP_DIR=/root/exp
bash scripts/training/text/run-qwen3-4B-8xgpu.sh
```

The image already contains the patched Megatron tree at `/root/Megatron-LM`, which is the default `MEGATRON` path in `scripts/entrypoint/local.sh`, so no extra path setup is needed. The local launcher assumes 8 GPUs unless `NUM_GPUS` is set.

## ROCm Runtime Settings

The image sets these environment variables. Keep them unless you are debugging the behavior they control:

| Variable | Value | Purpose |
| --- | --- | --- |
| `HSA_ENABLE_IPC_MODE_LEGACY` | `1` | ROCm 7.2 and later share GPU memory between processes through dma-buf, which leaks exported memory in colocated training. Legacy IPC avoids the leak. |
| `RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO` | `0` | Keeps GPUs visible to Ray actors that request no GPUs, so device checks in those actors do not fail. |
| `SGLANG_RUST_BUILD_MODE` | `never` | Uses the Rust radix-tree cache built into the image instead of compiling one in every new container. |

The SGLang base image also enables aiter kernels with `SGLANG_USE_AITER=1`.
