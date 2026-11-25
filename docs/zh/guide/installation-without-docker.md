# 无 Docker 环境安装指南

::: warning
本指南对应 `docker/Dockerfile` 在 commit `7354e4fa48ca154b35bf3a51d2a4c66d42997a30`（2026-11-25）时的状态。Relax 的 CUDA 套件钉得很死，版本和 commit 又变动频繁——执行下文任何命令前，请先与你本地 checkout 里的 `docker/Dockerfile` 逐项核对版本号和 commit。以 Dockerfile 为准：如果本文某一步与它不一致，按 Dockerfile 执行，并带上环境信息提 issue。
:::

Relax 平时以多阶段 Docker 镜像的方式构建。本文把这几个阶段在不能跑容器的主机或 HPC 节点上重现。它是一张构建地图，不是一次跑完的脚本：其中有几个阶段只是为了产出最终阶段要安装的 wheel，并且镜像依赖本仓库内特定的 commit 和补丁文件。

## 镜像里各阶段在做什么

| 阶段 | 作用 |
|------|------|
| `base` | NGC 25.06 基础镜像：PyTorch 2.9.0（cu129）、Python 3.12、系统包 |
| `sglang` | SGLang 0.5.21 代码 checkout，外加钉死版本的 cuDNN wheel |
| `train` | 训练用 CUDA 扩展：flash-attn、FLA、tilelang、DeepSeek-V4 DSA 依赖、FA3、Transformer Engine、apex、INT4 QAT |
| `hybridep_builder` | HybridEP（DeepEP fork）wheel，按同一 Torch ABI 编译 |
| `router_builder` | `sglang-router` wheel，带 R3 rollout 路由与 PD/DP aware 补丁 |
| `tree_core_builder` | SGLang HiCache tree-core 绑定 |
| `relax` | Megatron-LM（经 Megatron-Bridge）、Python 依赖、TransferQueue、FLA 0.5.2、仓库补丁、可编辑安装的 SGLang |
| 最终阶段 | cu129 运行期依赖、cuDNN Frontend DSA 补丁、torch_memory_saver 软链、tree-core 安装 |

如果你能在另一台有 skopeo、umoci 的机器上把 `ghcr.io/redai-studio/relaxrl:sglang-v0.5.21-cu129` 的层解出来，优先这么做——你会直接得到完全一致的 base 阶段，包括 NVIDIA 的 PyTorch 构建。下文的手工路线从一台干净的 Ubuntu 22.04 + CUDA 12.9 主机开始。

## 前置条件

- NVIDIA GPU，计算能力 >= 8.0（Ampere 及以上）；FA3 和 DSA 的快速路径默认 Hopper。
- 小模型路线（Qwen3-0.6B / 4B）至少 24 GB 显存。
- 至少 80 GB 可用磁盘，用于源码、wheel、模型和 checkpoint。
- Ubuntu 22.04、Python 3.12、CUDA Toolkit 12.9、NVIDIA 驱动 >= 560、GCC 11.x、CMake >= 3.18、Git >= 2.30、rsync。Rust 1.97.1 只在 router 阶段需要。

全程使用的环境变量：

```bash
export CUDA_HOME=/usr/local/cuda-12.9
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export MAX_JOBS=8        # 大机器可以调高，但编译扩展时注意内存
export RELAX_SRC=/path/to/Relax
```

## 步骤 1：系统包

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

## 步骤 2：PyTorch

镜像自带的是 25.06 NGC 容器里的 NVIDIA PyTorch 2.9.0 构建。在镜像之外，最接近的是官方 cu129 wheel：

```bash
pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cu129
python -c "import torch; print(torch.__version__, torch.version.cuda)"
```

只有当某个下游包确实需要时再装 `torchvision`/`torchaudio`，并保持它们与 Torch 同一 ABI。

## 步骤 3：cuDNN 与 SGLang

```bash
pip install nvidia-cudnn-cu12==9.20.0.48

git clone https://github.com/sgl-project/sglang.git /sgl-workspace/sglang
cd /sgl-workspace/sglang
git checkout e00930c5489053f26d86b179cee0d087f846acbb   # 镜像会校验这个 commit
```

SGLang 在步骤 11 才可编辑安装，且必须先打上补丁。不要从 PyPI `pip install sglang`——镜像用的就是这份 checkout，配合 `SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SGLANG=0.5.21`。

## 步骤 4：训练用 CUDA 扩展

```bash
FLASH_ATTENTION_FORCE_BUILD=TRUE MAX_JOBS=64 \
    pip install flash-attn==2.7.4.post1 --no-build-isolation --no-cache-dir
pip install --no-cache-dir flash-linear-attention==0.4.2
pip install --no-cache-dir tilelang -f https://tile-ai.github.io/whl/nightly/cu128/
```

Dockerfile 还支持消费一份预编译 wheel 缓存（`PREBUILD_CACHE_TAR_URL`），那条路径会把 `tilelang==0.1.12` 和 `flash-linear-attention==0.4.2` 一起钉死；没有这份缓存就是上面这种 nightly tilelang。

## 步骤 5：DeepSeek-V4 DSA 依赖

这两个包都没法直接用 PyPI 版本，必须按钉死的源码编译：

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

缺了它们，Megatron Core 的 DSA 断言会在 import 时或第一次稀疏注意力前向时失败。

## 步骤 6：FlashAttention 3（Hopper）

FA3 从 flash-attention 的特定 commit 编译，再调整目录结构让 Transformer Engine 能正确 import：

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

那几行删除不是洁癖。该 commit 的 `setup.py` 还会额外装一个顶层 `flash_attn_interface.py`；同一进程里再 import 它会重复注册 `flash_attn_3::_flash_attn_forward` 自定义算子，把 Transformer Engine 已绑定的 library 句柄销毁，之后的前向会以内部 schema 断言失败告终。

## 步骤 7：Transformer Engine、torch_memory_saver、apex、量化相关包

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

## 步骤 8：INT4 QAT 内核与 TE wheel 元数据

```bash
cd $RELAX_SRC/relax/backends/megatron/kernels/int4_qat
pip install . --no-build-isolation --no-cache-dir

pip install --no-cache-dir --no-deps pandas==3.0.6 pyarrow==24.0.0
python $RELAX_SRC/docker/scripts/repair_te_wheel_metadata.py \
    $RELAX_SRC/docker/patch/transformer_engine/2.14.1-cu12-wheel-tag.patch
pip check && python -c "import pandas, pyarrow, datasets"
```

元数据修复会把 Transformer Engine 的 wheel tag 改成与 cu129 Torch ABI 匹配，这样 `pip check` 不会报错；如果你的 TE 构建本来就带正确 tag，可以跳过。

## 步骤 9：通过 Megatron-Bridge 安装 Megatron-LM

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

然后打仓库补丁。镜像只应用 `# BEGIN common` 与 `# END common` 标记之间的那一段：

```bash
cp $RELAX_SRC/docker/patch/latest/megatron.patch $MEGATRON_PATH/
cd $MEGATRON_PATH
sed -n '/^# BEGIN common$/,/^# END common$/p' megatron.patch | git apply --check -
sed -n '/^# BEGIN common$/,/^# END common$/p' megatron.patch | git apply -
sha256sum megatron.patch > /etc/relax-megatron-patch.sha256
python -m compileall -q megatron
```

`megatron.patch` 内嵌了它生成时对应的上游 commit。如果 `git apply --check` 失败，说明你的 Megatron-Bridge commit 不对——改 commit，不要改补丁。

## 步骤 10：sglang-router 与 HiCache tree-core

两者都是编译产物，不是 PyPI 上能直接装的东西。

```bash
# router：Rust 1.97.1 + maturin，源码补丁在 docker/patch/sglang-router
export ROUTER_RUST_VERSION=1.97.1
export ROUTER_BUILD_JOBS=16
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain $ROUTER_RUST_VERSION
export PATH=/root/.cargo/bin:$PATH
python -m pip install --no-cache-dir "maturin==1.9.6"
bash $RELAX_SRC/docker/scripts/build_sglang_router.sh \
    $RELAX_SRC/docker/patch/sglang-router /opt/router-wheels

# tree-core：在钉死的 SGLang checkout 内，且已打好补丁
cd /sgl-workspace/sglang
test "$(git rev-parse HEAD)" = "e00930c5489053f26d86b179cee0d087f846acbb"
git apply $RELAX_SRC/docker/patch/latest/sglang.patch
python $RELAX_SRC/docker/scripts/build_sglang_tree_core.py build /sgl-workspace/sglang /opt/relax-tree-core
```

router wheel 的版本必须是 `0.3.2+slime.dpaware`，source commit 为 `9daabcd801c861088d93b8ce8a2a831490ce4bee`，镜像会断言这两项。官方 `sglang-router` wheel 会静默丢掉 `routed_experts` 字段，直接破坏 R3 rollout 路由。

## 步骤 11：Python 依赖与 Relax 本体

```bash
pip install --no-cache-dir "ray[default,serve]==2.56.1"
pip install --ignore-installed PyJWT
pip install -r $RELAX_SRC/requirements.txt --no-cache-dir
pip install --no-cache-dir "compressed_tensors>=0.13.0" tensordict==0.10.0 pyvers==0.1.0 \
    'nvidia-modelopt[hf]==0.44.0' --no-deps

# TransferQueue 及其 assignment-transform 补丁
pip install "transferqueue @ git+https://github.com/redai-studio/TransferQueue.git@9784ad0d8a5f0db93ff8326379ab2a6d06e640a0" --no-deps
TQ_SITE="$(python -c 'import importlib.util; from pathlib import Path; print(Path(importlib.util.find_spec("transfer_queue").origin).parent.parent)')"
patch --batch --forward --fuzz=0 -d "$TQ_SITE" -p1 \
    < $RELAX_SRC/docker/patch/transferqueue/9784ad0-assignment-transform.patch

# 钉死的 Megatron Core revision 需要的异步 checkpoint API
pip install --no-cache-dir \
    "nvidia-resiliency-ext @ git+https://github.com/NVIDIA/nvidia-resiliency-ext.git@6c5f2a13c7688d92a7ac7ee6e464721eb8b7345d"

# 运行期用 FLA 0.5.2；0.4.2 保留为兼容树
pip install --no-cache-dir --force-reinstall --no-deps "fla-core==0.5.2" "flash-linear-attention==0.5.2"
cd "$(python -c 'import importlib.util; from pathlib import Path; print(Path(importlib.util.find_spec("fla").origin).parent.parent)')"
patch --batch --forward --fuzz=0 -p1 < $RELAX_SRC/docker/patch/fla/0.5.2.patch
python -m compileall -q fla/modules/conv/triton
export RELAX_FLA_CONV_NB_MODE=auto

python $RELAX_SRC/docker/scripts/install_legacy_fla.py --target /opt/relax/compat/fla-0.4.2 \
    --patch $RELAX_SRC/docker/patch/fla/0.4.2-legacy-runtime.patch \
    --manifest /etc/relax-fla042-runtime.json

# 步骤 10 构建的 router wheel
pip install --no-cache-dir --force-reinstall --no-deps /opt/router-wheels/*.whl

# SGLang，可编辑安装，使用钉死的 commit
pip install --no-cache-dir setuptools-scm==9.2.2
SGLANG_BUILD_RUST_EXTS=none SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SGLANG=0.5.21 \
    pip install --no-cache-dir --no-deps --no-build-isolation -e /sgl-workspace/sglang/python
pip install --no-cache-dir transformers==5.16.0 tokenizers==0.23.2

pip install -e $RELAX_SRC
```

## 步骤 12：HybridEP wheel

只有用混合专家并行（hybrid EP）时才需要；镜像里是无条件构建的。

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

## 步骤 13：cu129 运行期层

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

cuDNN Frontend 的补丁要求 `nvidia-cudnn-frontend==1.31.0+cu12` 与 `nvidia-cutlass-dsl==4.6.2`；打补丁的脚本会先断言这两项，再动文件。

## 基本检查

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python -c "from importlib.metadata import version; [print(n, version(n)) for n in ('sglang','transformers','tokenizers','transformer-engine','flash-linear-attention','fla-core')]"
python -c "from sglang_router import sglang_router_rs; import sglang_router, json, importlib.resources as r; print(json.loads(r.files('sglang_router').joinpath('_relax_build.json').read_text()))"
python -c "import megatron; print(megatron.__file__)"
python $RELAX_SRC/docker/scripts/check_gpu_imports.py
```

::: warning
import 全通过不代表能训练。Ray 启动、SGLang 引擎拉起、NCCL、Megatron 训练循环各有各的失败方式。先用 Qwen3-0.6B 跑几个 step 验证，再提交正式任务。
:::

## 常见问题

- actor 前向报 `schema_.has_value() INTERNAL ASSERT FAILED`：顶层那个 `flash_attn_interface.py` 仍可被 import。重做步骤 6 里的 `find ... -delete`，并复查 `u.find_spec('flash_attn_interface') is None`。
- 找不到 `flash_mla_sparse_fwd`：FlashMLA 没带子模块安装，用 `--recurse-submodules` 重新 clone。
- `pip check` 对 Transformer Engine 报错：把步骤 8 和步骤 13 里的 wheel tag 修复再跑一遍。
- `git apply` 拒绝 `sglang.patch` 或 `megatron.patch`：你的 SGLang 或 Megatron-Bridge commit 与钉死值不符；两个补丁都假定步骤 3、9 里的 commit。
- router import 失败，或 `_relax_build.json` 里版本只是 `0.3.2`：装的是 PyPI 官方 wheel。用步骤 10 产出的 wheel 加 `--force-reinstall --no-deps` 重装。
- 编译扩展时 `nvcc` 被杀：调低 `MAX_JOBS`，FA3 和 apex 分开编。

## 获取帮助

1. 把版本与 `docker/Dockerfile` 逐项对一遍——这里绝大多数失败都是 pin 漂移。
2. 搜索已有的 [GitHub Issues](https://github.com/redai-studio/Relax/issues)。
3. 提 issue 时附上系统、GPU 型号、驱动版本、CUDA 版本、失败发生在哪一步，以及完整报错。

::: tip
Docker 仍是推荐路径。本指南面向受限共享服务器和无容器运行时的 HPC 系统，两次更新之间会落后于 Dockerfile。
:::
