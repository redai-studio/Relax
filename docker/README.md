# Docker Release Guidelines

## Overview

We publish two types of Docker images for Relax:

### 1. Stable Version

- Based on specific SGLang releases
- Patches are stored and maintained for these versions
- Recommended for production use

### 2. Latest Version

- Aligns with `lmsysorg/sglang:latest`
- Contains the most recent features and improvements
- Recommended for development and testing

## Build GPU Images

Run these commands from the repository root. Set `REGISTRY` to your registry and optional namespace; `IMAGE_REPOSITORY` defaults to `relax`.

```bash
export REGISTRY=registry.example.com/team

# Build and push all three train images.
make docker-train

# Build and push all three dev images, building missing train images first.
make docker-dev

# Build only the Blackwell dev image locally.
DO_PUSH=0 make docker-dev-cu13-b300
```

`make train` is an alias for `make docker-train`. Each variant checks its own image tag, so an existing cu12 image does not skip missing cu13 images.

`train` images contain the Dockerfile's `train` stage. `dev` images build the final `relax` stage with the remaining dependencies and patches. `make docker-dev` builds CUDA 12, CUDA 13 Blackwell, and CUDA 13 Hopper variants; use the individual targets below to select one.

| Variant             | Dockerfile               | Train target               | Dev target               |
| ------------------- | ------------------------ | -------------------------- | ------------------------ |
| CUDA 12.9           | `docker/Dockerfile`      | `docker-train-cu12`        | `docker-dev-cu12`        |
| CUDA 13.0 Blackwell | `docker/Dockerfile.cu13` | `docker-train-cu13-b300`   | `docker-dev-cu13-b300`   |
| CUDA 13.0 Hopper    | `docker/Dockerfile.cu13` | `docker-train-cu13-hopper` | `docker-dev-cu13-hopper` |

The Blackwell targets pass `GPU_ARCH=b300`; the Hopper targets pass `GPU_ARCH=hopper`. `docker/Dockerfile.cu13` replaces the former `docker/Dockerfile_Blackwell`.

> CUDA 13 is a candidate build configuration. Full image builds and GPU validation, including B300 forward/backward validation, are required before promotion. These build targets do not establish Blackwell support in the published `latest` image.

### Image Tags and Reuse

Images use `<REGISTRY>/<IMAGE_REPOSITORY>:<tag>`, with these default tags:

| Variant           | Train tag                               | Dev tag                               |
| ----------------- | --------------------------------------- | ------------------------------------- |
| CUDA 12           | `train-YYYYMMDD-<hash8>-cu12`           | `dev-YYYYMMDD-<hash8>-cu12`           |
| CUDA 13 Blackwell | `train-YYYYMMDD-<hash8>-cu13-blackwell` | `dev-YYYYMMDD-<hash8>-cu13-blackwell` |
| CUDA 13 Hopper    | `train-YYYYMMDD-<hash8>-cu13-hopper`    | `dev-YYYYMMDD-<hash8>-cu13-hopper`    |

The date is the build date and `hash8` is the first eight characters of the current Git commit. `DO_PUSH=1` is the default: existing remote images are skipped, and new images are pushed. With `DO_PUSH=0`, existing local images are skipped and nothing is pushed. `REGISTRY` is required in both modes because it forms part of the image name.

### Base and Train Image Overrides

The Makefile reads these variables from the environment or from command-line assignments such as `make docker-dev CU13_BASE_IMAGE=...`:

| Variant           | Base image override | Train image override         |
| ----------------- | ------------------- | ---------------------------- |
| CUDA 12           | `BASE_IMAGE`        | `TRAIN_IMAGE`                |
| CUDA 13 Blackwell | `CU13_BASE_IMAGE`   | `CU13_BLACKWELL_TRAIN_IMAGE` |
| CUDA 13 Hopper    | `CU13_BASE_IMAGE`   | `CU13_HOPPER_TRAIN_IMAGE`    |

When a base override is unset, the corresponding Dockerfile selects its pinned base. `CU13_BASE_IMAGE` applies to both CUDA 13 variants independently of `BASE_IMAGE`. Each variant has a separate train image because its CUDA kernels target different architectures; there is no shared `CU13_TRAIN_IMAGE` variable.

For dev builds, a nonempty train override is used directly without building the train stage. Without an override, the matching default train tag is checked and built if missing. For train targets, the same variable selects the output image tag. `DEV_IMAGE` can override the CUDA 12 dev output tag through a Make command-line assignment.

```bash
# Reuse an existing Blackwell train image; replace the date/hash placeholders.
CU13_BLACKWELL_TRAIN_IMAGE="$REGISTRY/relax:train-YYYYMMDD-HASH8-cu13-blackwell" \
  DO_PUSH=0 make docker-dev-cu13-b300
```

See the [English installation guide](../docs/en/guide/installation.md) or [中文安装指南](../docs/zh/guide/installation.md) for container setup and direct Docker build examples.

## Pre-Release Testing

Before each update, we perform comprehensive testing on the following models using H100 GPUs:

| Model              | Sync | Async |
| ------------------ | ---- | ----- |
| Qwen3-4B           | ✓    | ✓     |
| Qwen3-30B-A3B      | ✓    | ✓     |
| Qwen3-omni-30B-A3B | ✓    | ✓     |
| Qwen3.5-35B-A3B    | ✓    | ✓     |

## Testing Modes

- **Sync**: Synchronous training mode
- **Async**: Asynchronous training mode

All models are tested in both modes to ensure stability and compatibility across different training scenarios.
