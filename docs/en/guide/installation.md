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

The current image supports H-series GPUs.

Run the following commands to clone the repository, pull the latest image, and start an interactive container:

```bash
# Clone the repository
git clone https://github.com/redai-studio/Relax.git

# Pull the Docker image
docker pull ghcr.io/redai-studio/relaxrl:latest

# Run the container, mounting the local repository to /root/Relax inside the container
docker run -it --gpus all -v $(pwd)/Relax:/root/Relax ghcr.io/redai-studio/relaxrl:latest /bin/bash
```

Alternatively, build the image from the Dockerfile:

```bash
# Navigate to the Relax root directory
cd Relax

# Build sglang runtime docker image, for deployment only
DOCKER_BUILDKIT=1 docker build \
  -f docker/Dockerfile \
  --target sglang \
  -t {your image name}:{tag} \
  --build-arg HTTP_PROXY={代理地址（可选配置）} \
  --build-arg HTTPS_PROXY={代理地址（可选配置）} \
  --build-arg NO_PROXY={bypass代理地址（可选配置）} \
  .

# build relax runtime docker image, for training and deployment
DOCKER_BUILDKIT=1 docker build \
  -f docker/Dockerfile \
  --target relax \
  -t {your image name}:{tag} \
  --build-arg HTTP_PROXY={代理地址（可选配置）} \
  --build-arg HTTPS_PROXY={代理地址（可选配置）} \
  --build-arg NO_PROXY={bypass代理地址（可选配置）} \
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
