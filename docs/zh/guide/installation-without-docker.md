# 无 Docker 环境安装指南

::: warning
本指南基于官方 `docker/Dockerfile` 整理，面向无法使用 Docker 的用户（例如受限共享服务器、无容器运行时的 HPC 环境）。具体的编译步骤复杂且版本敏感——如果遇到编译错误，请以 Dockerfile 为准，并携带你的环境信息提交 issue。
:::

## 概述

Relax 依赖多个需要从源码编译的重量级 CUDA 扩展，且锁定了特定 commit 和 patch：

- **Megatron-LM**（通过 Megatron-Bridge 构建，带自定义 patch）
- **SGLang**（带 R3 rollout 路由回放的自定义 patch）
- **flash-attn**（v2 + 从特定 commit 编译的 v3）
- **transformer-engine**（v2.14.1）
- **apex**（从特定 commit 编译）
- **TransferQueue**（从特定 commit 编译）

官方 Docker 镜像（`ghcr.io/redai-infra/relaxrl:latest`）是推荐且经过最充分测试的安装方式。仅在无法使用 Docker 时才参考本指南。

## 前置条件

### 硬件

- NVIDIA GPU，计算能力 >= 8.0（Ampere 及以上；Hopper 推荐用于 FA3）
- 小模型训练（Qwen3-0.6B / 4B）至少需要 24 GB 显存
- 至少 50 GB 可用磁盘空间，用于源码编译、模型和 checkpoint

### 软件

| 组件 | 要求版本 | 说明 |
|------|---------|------|
| 操作系统 | Ubuntu 22.04 (LTS) | 其他 Linux 发行版理论上可用但未测试 |
| Python | 3.12 | 官方镜像使用的精确版本 |
| CUDA Toolkit | 12.9 | 必须与 PyTorch 构建版本匹配 |
| NVIDIA 驱动 | >= 560 | 支持 CUDA 12.9 |
| GCC / G++ | 11.x | CUDA 扩展编译所需 |
| CMake | >= 3.18 | apex 等编译所需 |
| Git | >= 2.30 | 克隆依赖仓库 |
| rsync | 最新版 | 合并 Megatron-Bridge 源码 |
| jq | 最新版 | patch 验证 |

### 环境变量

开始前设置以下变量：

```bash
export CUDA_HOME=/usr/local/cuda-12.9
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export MAX_JOBS=8  # 根据 CPU 核心数调整；越高编译越快，占用内存越多
```

## 步骤 1：安装系统依赖

```bash
sudo apt-get update && sudo apt-get install -y --no-install-recommends \
    build-essential patchelf tmux net-tools htop nano iputils-ping \
    autoconf automake libatlas-base-dev libgoogle-glog-dev libbz2-dev libc-ares2 \
    libleveldb-dev liblmdb-dev libprotobuf-dev libsnappy-dev libtool nasm protobuf-compiler pkg-config unzip sox \
    libsndfile1 libpng-dev libhdf5-dev gfortran rapidjson-dev ninja-build libedit-dev rsync jq \
    locales tzdata
```

## 步骤 2：安装 PyTorch 2.9.0（CUDA 12.9）

官方镜像使用 NVIDIA 的 PyTorch 2.9.0 构建（来自 25.06 NGC 容器）。安装匹配版本：

```bash
pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cu129
```

验证：

```bash
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}, GPU: {torch.cuda.get_device_name(0)}')"
```

## 步骤 3：编译核心 CUDA 扩展

### 3.1 flash-attn v2

```bash
MAX_JOBS=64 pip -v install flash-attn==2.7.4.post1 --no-build-isolation --no-cache-dir
pip install --no-cache-dir flash-linear-attention==0.4.1
pip install --no-cache-dir tilelang -f https://tile-ai.github.io/whl/nightly/cu128/
```

### 3.2 flash-attn v3（仅 Hopper 架构）

::: warning
此步骤仅适用于 Hopper（H100/H800）GPU。Ampere/Ada 架构请跳过。
:::

```bash
cd /opt
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention/
git checkout 0f82fead0b2d59f6390c62a1d74e927159bc5a7b
git submodule update --init
cd hopper/
MAX_JOBS=96 python setup.py install

# 移除与 TE 冲突的顶层模块
export python_path=$(python -c "import site; print(site.getsitepackages()[0])")
mkdir -p $python_path/flash_attn_3
cp flash_attn_interface.py $python_path/flash_attn_3/flash_attn_interface.py
find $python_path -maxdepth 1 -name flash_attn_interface.py -delete
find $python_path -maxdepth 2 -path '*.egg/flash_attn_interface.py' -delete

# 验证
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

### 3.6 附加包

```bash
pip install nvidia-modelopt[torch]>=0.37.0 --no-build-isolation --no-cache-dir
pip install "numpy<2" nvidia-cudnn-cu12==9.16.0.29 --no-cache-dir
```

## 步骤 4：通过 Megatron-Bridge 安装 Megatron-LM

Relax 使用从 Megatron-Bridge 源码及其子模块合并的 Megatron-LM 目录树，并应用自定义 patch。

```bash
export MEGATRON_BRIDGE_COMMIT=2faedbf6fe3c422835a44b2b360cadcb2a116a54
export MEGATRON_PATH=/root/Megatron-LM  # 调整为你偏好的路径

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

### 应用 megatron patch

```bash
# 从 Relax 仓库复制 patch
cp /path/to/Relax/docker/patch/latest/megatron.patch $MEGATRON_PATH/

cd $MEGATRON_PATH
patch -p1 < megatron.patch

# 验证 patch 干净应用
if grep -R -n '^<<<<<<< ' .; then
    echo "错误：patch 应用失败，请解决冲突。"
    exit 1
fi
rm megatron.patch
```

### 设置 PYTHONPATH

```bash
export PYTHONPATH=$MEGATRON_PATH:$PYTHONPATH
```

验证：

```bash
python -c "import megatron; print(f'Megatron 导入自 {megatron.__file__}')"
```

## 步骤 5：安装带自定义 patch 的 SGLang

Relax 使用 SGLang v0.5.15.post1，带 R3（rollout 路由回放）的自定义 patch。

::: tip
基础 SGLang 安装来自 `lmsysorg/sglang:v0.5.15.post1-cu129` Docker 镜像。无 Docker 环境下，需要先从源码安装匹配版本的 SGLang。
:::

```bash
# 从源码安装匹配版本的 SGLang
pip install "sglang[all]==0.5.15.post1"

# 应用自定义 sglang patch
export SGLANG_PATH=$(python -c "import sglang; print(sglang.__path__[0])")
cp /path/to/Relax/docker/patch/latest/sglang.patch $SGLANG_PATH/

cd $SGLANG_PATH
git add .
git update-index --refresh
git apply sglang.patch --3way

# 验证
if grep -R -n '^<<<<<<< ' .; then
    echo "错误：SGLang patch 应用失败。"
    exit 1
fi
rm sglang.patch
```

### 安装打过 patch 的 sglang-router

官方 `sglang-router` wheel 会丢弃 `routed_experts` 字段，导致 R3 失效。Relax 使用 slime 的打过 patch 的 fork：

```bash
pip install --no-cache-dir --force-reinstall --no-deps \
    https://github.com/zhuzilin/sgl-router/releases/download/v0.3.2-9daabcd/sglang_router-0.3.2-cp38-abi3-manylinux_2_28_x86_64.whl

# 验证打过 patch 的版本
python -c "import sglang_router; assert 'slime' in sglang_router.__version__, sglang_router.__version__"
```

## 步骤 6：安装 TransferQueue

```bash
pip install "transferqueue @ git+https://github.com/redai-infra/TransferQueue.git@58054a33834aadbcf76aacd6b1e32e25c030f2c9" --no-deps
```

## 步骤 7：安装 Python 依赖和 Relax

```bash
cd /path/to/Relax

# 安装纯 Python 依赖
pip install --ignore-installed PyJWT
pip install -r requirements.txt --no-cache-dir

# 安装附加包
pip install --no-cache-dir "compressed_tensors>=0.13.0" tensordict==0.10.0 pyvers==0.1.0 'nvidia-modelopt[hf]==0.44.0' --no-deps

# 以开发模式安装 Relax
pip install -e .
```

## 步骤 8：编译 INT4 QAT CUDA 扩展

```bash
cd /path/to/Relax/relax/backends/megatron/kernels/int4_qat
pip install . --no-build-isolation --no-cache-dir
```

## 环境变量汇总

将以下内容添加到 `~/.bashrc` 或专用的环境脚本中：

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

## 验证安装

按顺序运行以下检查：

```bash
# 1. 基础导入
python -c "
import torch
import ray
import sglang
import megatron
import relax
print('所有基础导入成功')
print(f'  torch: {torch.__version__}')
print(f'  CUDA 可用: {torch.cuda.is_available()}')
print(f'  GPU: {torch.cuda.get_device_name(0)}')
"

# 2. CUDA 扩展
python -c "
import flash_attn
import transformer_engine
print('CUDA 扩展导入成功')
print(f'  flash_attn: {flash_attn.__version__}')
"

# 3. Relax 组件
python -c "
from relax.controller import Controller
from relax.registry import ALGOS
print(f'已注册算法: {list(ALGOS.keys())}')
print('Relax 组件加载成功')
"
```

::: warning
通过这些导入检查并不保证完整训练可以正常运行。端到端训练涉及 Ray 集群初始化、SGLang 引擎启动、NCCL 通信和 Megatron 训练循环——每一步都可能遇到环境特定的问题。建议先用小模型（Qwen3-0.6B）跑几个 training step 来验证。
:::

## 常见问题排查

### flash-attn / apex / TE 编译错误

- 确保 `CUDA_HOME` 设置正确，`nvcc --version` 显示 12.9
- 确保 GCC 版本为 11.x（`gcc --version`）
- 如果编译时内存不足，降低 `MAX_JOBS`
- 检查 NVIDIA 驱动是否支持 CUDA 12.9（`nvidia-smi`）

### `ModuleNotFoundError: No module named 'megatron'`

- 确保 `PYTHONPATH` 包含 `$MEGATRON_PATH`
- 验证合并目录存在：`ls $MEGATRON_PATH/megatron/`

### `ModuleNotFoundError: No module named 'sglang'`

- 必须先安装 SGLang，再应用 patch
- 验证：`python -c "import sglang; print(sglang.__version__)"` 应显示 0.5.15.post1

### NCCL 错误或 Ray 集群失败

- 确保所有节点网络互通（相当于 Docker 的 `--network=host`）
- 设置 `NCCL_SOCKET_IFNAME` 为正确的网卡接口
- 单节点训练时，确保 `ray start --head` 能绑定到 localhost

### Patch 冲突

如果 `git apply` 或 `patch` 报告冲突：

1. 检查基础版本是否精确匹配（SGLang 0.5.15.post1、Megatron-Bridge commit 2faedbf6）
2. 尝试 `git apply --3way` 进行三方合并
3. 如果冲突持续，请参考 Dockerfile 中的精确 patch 上下文

## 获取帮助

如果遇到本指南未涵盖的问题：

1. 查看[官方 Dockerfile](https://github.com/redai-infra/Relax/blob/main/docker/Dockerfile)——它是权威的构建参考
2. 搜索已有的 [GitHub Issues](https://github.com/redai-infra/Relax/issues)
3. 提交新 issue，包含：
   - 操作系统、GPU 型号、NVIDIA 驱动版本、CUDA 版本
   - 精确的错误信息和堆栈跟踪
   - 失败发生的步骤
   - 是否使用了本指南中指定的精确版本

::: tip
对大多数用户而言，Docker 镜像仍然是推荐的安装方式。本指南作为受限环境的参考提供，更新频率可能不如 Dockerfile。
:::
