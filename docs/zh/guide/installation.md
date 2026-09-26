# 安装

## 前置要求

在安装 Relax 之前，请确保您具备以下条件：

- Python 3.12
- CUDA 12.9+（用于 GPU 支持）
- Ray 2.0+
- PyTorch 2.10+

## 安装方法

### 方法 1：使用 Docker 镜像（推荐）

由于 Relax 可能会包含针对 sglang/megatron 的临时补丁（patch）。为避免潜在的环境配置问题，强烈建议用户使用我们提供的最新 Docker 镜像，它已预置好所有依赖。

当前镜像支持 H 系列 GPU 运行。

请执行以下命令，克隆代码仓库、拉取最新镜像并启动一个交互式容器：

```bash
# 克隆代码仓库
git clone https://github.com/redai-studio/Relax.git

# 拉取 Docker 镜像
docker pull ghcr.io/redai-studio/relaxrl:latest

# 运行容器，将本地代码仓库挂载到容器内的 /root/Relax
docker run -it --gpus all -v $(pwd)/Relax:/root/Relax ghcr.io/redai-studio/relaxrl:latest /bin/bash
```

或者基于 Dockerfile 构建镜像：

```bash
# 进入 Relax 根目录
cd Relax

# 构建 sglang 运行时 docker 镜像，用于部署
DOCKER_BUILDKIT=1 docker build \
  -f docker/Dockerfile \
  --target sglang \
  -t {your image name}:{tag} \
  --build-arg HTTP_PROXY={代理地址（可选配置）} \
  --build-arg HTTPS_PROXY={代理地址（可选配置）} \
  --build-arg NO_PROXY={bypass代理地址（可选配置）} \
  .

# 构建 relax 运行时 Docker 镜像，用于训练或部署
DOCKER_BUILDKIT=1 docker build \
  -f docker/Dockerfile \
  --target relax \
  -t {your image name}:{tag} \
  --build-arg HTTP_PROXY={代理地址（可选配置）} \
  --build-arg HTTPS_PROXY={代理地址（可选配置）} \
  --build-arg NO_PROXY={bypass代理地址（可选配置）} \
  .
```

更多 Docker 发布信息请参见 [Docker README](https://github.com/redai-studio/Relax/blob/main/docker/README.md)。

### 方法 2：从源码安装

```bash
# 克隆仓库
git clone https://github.com/redai-studio/Relax.git
cd Relax

# 安装依赖
pip install -r requirements.txt

# 以开发模式安装 Relax
pip install -e .

# scripts 的示例脚本中需要执行
export RELAX="your relax path"
# 等价于
export PYTHONPATH=your_relax_path:$PYTHONPATH
```

请注意 Relax 依赖 sglang 和 megatron，需要您前往官网自行安装：

```bash
# scripts 的示例脚本中需要执行
export MEGATRON="your megatron path"
# 等价于
export PYTHONPATH=your_megatron_path:$PYTHONPATH
```

此外 Relax 依赖 [Megatron Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge) 进行权重转换。安装方式参考 [`docker/Dockerfile`](https://github.com/redai-studio/Relax/blob/main/docker/Dockerfile)，将 Bridge 源码与 Megatron-LM submodule 合并到同一目录后加入 `PYTHONPATH`：

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

## 下一步

- [快速开始指南](./quick-start.md) - 运行您的第一个实验
- [配置说明](./configuration.md) - 了解配置选项
- [示例](../examples/deepeyes.md) - 探索示例项目
