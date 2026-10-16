# Installation

## Prerequisites

Before installing Relax, ensure you have the following:

- Python 3.12
- CUDA 12.9+ (for GPU support)
- Ray 2.0+
- PyTorch 2.10+

## Installation Methods

### Method 1: Using Docker Image (Recommended)

Since Relax may include temporary patches for sglang/megatron, we strongly recommend using our latest Docker image to avoid potential environment configuration issues. The image comes with all dependencies pre-installed.

The commands below use the published `latest` image. Self-built CUDA 12 and CUDA 13 variants are described separately below.

Run the following commands to clone the repository, pull the latest image, and start an interactive container:

```bash
# Clone the repository
git clone https://github.com/redai-studio/Relax.git

# Pull the Docker image
docker pull ghcr.io/redai-studio/relaxrl:latest

# Run the container, mounting the local repository to /root/Relax inside the container
docker run -it --gpus all -v $(pwd)/Relax:/root/Relax ghcr.io/redai-studio/relaxrl:latest /bin/bash
```

#### Build Your Own Images

Run the Make targets from the repository root. Replace `registry.example.com/team` with your registry and optional namespace; `IMAGE_REPOSITORY` defaults to `relax`.

```bash
cd Relax
export REGISTRY=registry.example.com/team

# Build and push CUDA 12, CUDA 13 Blackwell, and CUDA 13 Hopper train images.
make docker-train

# Build and push the three dev images, building missing train images first.
make docker-dev

# Build only the Blackwell dev image locally.
DO_PUSH=0 make docker-dev-cu13-b300
```

`make train` is an alias for `make docker-train`. Each variant checks its own image tag, so an existing cu12 image does not skip missing cu13 images.

`train` images contain the `train` stage; `dev` images build the final `relax` stage with the remaining dependencies and patches. Select a single variant with these targets:

| Variant | Dockerfile | Train target | Dev target |
| --- | --- | --- | --- |
| CUDA 12.9 | `docker/Dockerfile` | `docker-train-cu12` | `docker-dev-cu12` |
| CUDA 13.0 Blackwell | `docker/Dockerfile.cu13` | `docker-train-cu13-b300` | `docker-dev-cu13-b300` |
| CUDA 13.0 Hopper | `docker/Dockerfile.cu13` | `docker-train-cu13-hopper` | `docker-dev-cu13-hopper` |

Blackwell uses `GPU_ARCH=b300`; Hopper uses `GPU_ARCH=hopper`. The suffixes in image tags are `cu12`, `cu13-blackwell`, and `cu13-hopper`. Default image names are `<REGISTRY>/<IMAGE_REPOSITORY>:train-YYYYMMDD-<hash8>-<suffix>` and `<REGISTRY>/<IMAGE_REPOSITORY>:dev-YYYYMMDD-<hash8>-<suffix>`, using the build date and the current Git commit's first eight characters.

`DO_PUSH=1` is the default: existing remote images are skipped, and new images are pushed. With `DO_PUSH=0`, existing local images are skipped and nothing is pushed. `REGISTRY` is required in both modes.

The Makefile reads these overrides from the environment or Make command-line assignments:

| Variant | Base image override | Train image override |
| --- | --- | --- |
| CUDA 12 | `BASE_IMAGE` | `TRAIN_IMAGE` |
| CUDA 13 Blackwell | `CU13_BASE_IMAGE` | `CU13_BLACKWELL_TRAIN_IMAGE` |
| CUDA 13 Hopper | `CU13_BASE_IMAGE` | `CU13_HOPPER_TRAIN_IMAGE` |

Unset base overrides use the corresponding Dockerfile's pinned base. `CU13_BASE_IMAGE` applies to both CUDA 13 variants independently of `BASE_IMAGE`. Train images are architecture-specific, so there is no shared `CU13_TRAIN_IMAGE` variable. During dev builds, a nonempty train override is reused directly; otherwise, the matching default train image is checked and built if missing. During train builds, the same variable selects the output image tag.

```bash
# Reuse an existing Blackwell train image; replace the date/hash placeholders.
CU13_BLACKWELL_TRAIN_IMAGE="$REGISTRY/relax:train-YYYYMMDD-HASH8-cu13-blackwell" \
  DO_PUSH=0 make docker-dev-cu13-b300
```

::: warning CUDA 13 validation status
`docker/Dockerfile.cu13` is a candidate build configuration. Full image builds and GPU validation, including B300 forward/backward validation, are required before promotion. These build targets do not establish Blackwell support in the published `latest` image.
:::

#### Build Directly with Docker

The existing Docker commands remain available. Run these examples from the repository root:

```bash
# Build sglang runtime docker image, for deployment only
DOCKER_BUILDKIT=1 docker build \
  -f docker/Dockerfile \
  --target sglang \
  -t {your image name}:{tag} \
  --build-arg HTTP_PROXY={proxy_address_optional} \
  --build-arg HTTPS_PROXY={proxy_address_optional} \
  --build-arg NO_PROXY={no_proxy_addresses_optional} \
  .

# build relax runtime docker image, for training and deployment
DOCKER_BUILDKIT=1 docker build \
  -f docker/Dockerfile \
  --target relax \
  -t {your image name}:{tag} \
  --build-arg HTTP_PROXY={proxy_address_optional} \
  --build-arg HTTPS_PROXY={proxy_address_optional} \
  --build-arg NO_PROXY={no_proxy_addresses_optional} \
  .
```

For more details on Docker releases, see [Docker README](https://github.com/redai-studio/Relax/blob/main/docker/README.md).

### Method 2: Install from Source

```bash
# Clone the repository
git clone https://github.com/redai-studio/Relax.git
cd Relax

# Install dependencies
pip install -r requirements.txt

# Install Relax in development mode
pip install -e .

# Set environment variable for example scripts
export RELAX="your relax path"
# Equivalent to
export PYTHONPATH=your_relax_path:$PYTHONPATH
```

Note that Relax depends on sglang and megatron. You need to install them from their official websites:

```bash
# Set environment variable for example scripts
export MEGATRON="your megatron path"
# Equivalent to
export PYTHONPATH=your_megatron_path:$PYTHONPATH
```

Additionally, Relax depends on [Megatron Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge) for weight conversion. Follow the install steps in [`docker/Dockerfile`](https://github.com/redai-studio/Relax/blob/main/docker/Dockerfile): merge the Bridge sources with the Megatron-LM submodule into a single directory and add it to `PYTHONPATH`:

```bash
export MEGATRON_BRIDGE_COMMIT=2faedbf6fe3c422835a44b2b360cadcb2a116a54
git clone https://github.com/NVIDIA-NeMo/Megatron-Bridge.git
cd Megatron-Bridge && git checkout ${MEGATRON_BRIDGE_COMMIT} && \
    git submodule update --init --recursive && ./scripts/switch_mcore.sh dev
mkdir -p /your/path/Megatron-LM
cp -r src/megatron /your/path/Megatron-LM/
rsync -avP 3rdparty/Megatron-LM/megatron/ /your/path/Megatron-LM/megatron/
export PYTHONPATH=/your/path/Megatron-LM:$PYTHONPATH
```

## Next Steps

- [Quick Start Guide](./quick-start.md) - Run your first experiment
- [Configuration Guide](./configuration.md) - Learn about configuration options
- [Examples](../examples/deepeyes.md) - Explore example projects
