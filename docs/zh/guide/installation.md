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

下方命令使用已发布的 `latest` 镜像。自行构建的 CUDA 12 和 CUDA 13 变体将在后文单独说明。

请执行以下命令，克隆代码仓库、拉取最新镜像并启动一个交互式容器：

```bash
# 克隆代码仓库
git clone https://github.com/redai-studio/Relax.git

# 拉取 Docker 镜像
docker pull ghcr.io/redai-studio/relaxrl:latest

# 运行容器，将本地代码仓库挂载到容器内的 /root/Relax
docker run -it --gpus all -v $(pwd)/Relax:/root/Relax ghcr.io/redai-studio/relaxrl:latest /bin/bash
```

#### 自行构建镜像

在仓库根目录运行 Make 目标。请将 `registry.example.com/team` 替换为你的镜像仓库地址及可选的命名空间；`IMAGE_REPOSITORY` 默认为 `relax`。

```bash
cd Relax
export REGISTRY=registry.example.com/team

# 构建并推送 CUDA 12、CUDA 13 Blackwell 和 CUDA 13 Hopper 的 train 镜像。
make docker-train

# 构建并推送三个 dev 镜像，缺少对应的 train 镜像时先自动构建。
make docker-dev

# 仅在本地构建 Blackwell dev 镜像。
DO_PUSH=0 make docker-dev-cu13-b300
```

`make train` 是 `make docker-train` 的别名。三个变体分别检查各自的镜像标签，cu12 已存在时仍会继续检查并构建缺少的 cu13 镜像。

`train` 镜像包含 `train` 阶段；`dev` 镜像构建最终的 `relax` 阶段，包含其余依赖和补丁。可以使用以下目标单独构建一个变体：

| 变体 | Dockerfile | Train 目标 | Dev 目标 |
| --- | --- | --- | --- |
| CUDA 12.9 | `docker/Dockerfile` | `docker-train-cu12` | `docker-dev-cu12` |
| CUDA 13.0 Blackwell | `docker/Dockerfile.cu13` | `docker-train-cu13-b300` | `docker-dev-cu13-b300` |
| CUDA 13.0 Hopper | `docker/Dockerfile.cu13` | `docker-train-cu13-hopper` | `docker-dev-cu13-hopper` |

Blackwell 使用 `GPU_ARCH=b300`，Hopper 使用 `GPU_ARCH=hopper`。镜像标签的后缀分别为 `cu12`、`cu13-blackwell` 和 `cu13-hopper`。默认镜像名为 `<REGISTRY>/<IMAGE_REPOSITORY>:train-YYYYMMDD-<hash8>-<suffix>` 和 `<REGISTRY>/<IMAGE_REPOSITORY>:dev-YYYYMMDD-<hash8>-<suffix>`，其中日期为构建日期，`hash8` 为当前 Git commit 的前八位。

默认 `DO_PUSH=1`：远端已存在的镜像会跳过，新构建的镜像会推送。设置 `DO_PUSH=0` 时检查并跳过本地已存在的镜像，不执行推送。两种模式均需设置 `REGISTRY`。

Makefile 从环境变量或 Make 命令行赋值中读取以下覆盖项：

| 变体 | 基础镜像覆盖项 | Train 镜像覆盖项 |
| --- | --- | --- |
| CUDA 12 | `BASE_IMAGE` | `TRAIN_IMAGE` |
| CUDA 13 Blackwell | `CU13_BASE_IMAGE` | `CU13_BLACKWELL_TRAIN_IMAGE` |
| CUDA 13 Hopper | `CU13_BASE_IMAGE` | `CU13_HOPPER_TRAIN_IMAGE` |

未设置基础镜像覆盖项时，使用对应 Dockerfile 固定的基础镜像。`CU13_BASE_IMAGE` 同时用于两个 CUDA 13 变体，独立于 `BASE_IMAGE`。Train 镜像区分架构，因此没有共用的 `CU13_TRAIN_IMAGE` 变量。构建 dev 镜像时，非空的 train 覆盖项会被直接复用；未设置时检查对应的默认 train 镜像，缺少则自动构建。构建 train 镜像时，同一变量指定输出镜像标签。

```bash
# 复用已有的 Blackwell train 镜像；请替换日期和哈希占位符。
CU13_BLACKWELL_TRAIN_IMAGE="$REGISTRY/relax:train-YYYYMMDD-HASH8-cu13-blackwell" \
  DO_PUSH=0 make docker-dev-cu13-b300
```

::: warning CUDA 13 验证状态
`docker/Dockerfile.cu13` 是候选构建配置。正式发布前仍需完成完整镜像构建和 GPU 验证，包括 B300 前向与反向验证。提供这些构建目标不代表已发布的 `latest` 镜像已经支持 Blackwell。
:::

#### 直接使用 Docker 构建

原有 Docker 命令仍可使用。请在仓库根目录运行以下示例：

```bash
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
