# 本轮 Megatron + MoK 增量镜像

把本轮实验依赖固化到镜像中：Megatron 的最终8文件补丁、MoK 原生640维尾块支持、PR10 调度器，以及 ThunderKittens MN-major 修复。运行时无需挂载实验目录或复用旧 `.so`。

## 构建

使用**当前完整的 CUDA 13 训练镜像**作为基底，最好填写不可变的 `repository@sha256:…`。容器内部没有可核实的镜像引用，因此 `BASE_IMAGE` 必须由构建者提供。不要替换成原始 SGLang 镜像；完整训练镜像还包含 Bridge、TE、HybridEP 等依赖。

从 Relax 仓库根目录执行：

```bash
export BASE_IMAGE='填写当前完整训练镜像的引用'
docker build --build-arg BASE_IMAGE="$BASE_IMAGE" \
  -t relax-mok:pr10-sm103 docker/mok
```

构建上下文只有 `docker/mok`，不包含数据集、模型、ClearML 配置或历史编译产物。需要联网获取两个固定 commit；编译不需要 GPU，不更新 Torch、CUDA、驱动或其他训练依赖。

基底必须提供 Python 3.12、Torch `2.13.0+cu130`、CUDA 13.0 NVCC、Git、GNU Make 和 patch。默认使用 `/opt/sglang/bin/python`，解释器路径不同可传 `--build-arg PYTHON_BIN=...`；默认 Megatron 位于 `/root/Megatron-LM`，路径不同可传 `--build-arg MEGATRON_ROOT=...`。目标为当前 Blackwell 的 `SM103`。

构建会检查基底 Megatron 的全部1509个 Python 文件和文件集合，再无 fuzz 应用补丁；不匹配会明确报错。**当前仓库最新 `Dockerfile.cu13` 的补丁链与实验镜像不同，不能直接假定是相同基底。** 此处增量 Dockerfile 保留实际实验的原有依赖。

## 固定来源与修改范围

| 组件           | 固定来源                                                                            | 本地增量                                                                                                              |
| -------------- | ----------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| Megatron       | 当前镜像源码快照；8个修改文件的前后 SHA256 及全1509文件集合摘要见 `manifest.json`   | 最终8文件合并补丁，含 TP2/SP/ETP1、固定输入补齐/越界校验、native shared、参数属性保存、可重载 EP group 与存储缓存失效 |
| MoK            | `QiZhangNV/mixture-of-kittens@ee7bdaba515fd3d557105d71809ec74cd523d5a1`             | 640维尾块与 Python 校验、调度器、已有验证用例                                                                         |
| 调度器         | `cursor/mixture-of-kittens` PR10，commit `79b320ab5236dd0d031ea580063571271924bdae` | 保留本地 `schedule_entrypoint` 绑定名称；最终源码 SHA 固定                                                            |
| ThunderKittens | `1c3920d993404dd49a6d4c7267ea11d583bd5c68`                                          | `tcgen05.cuh` MN-major 修复                                                                                           |

Megatron 补丁来自当前最终工作副本相对原始镜像的完整差异，包含后续 fixedN 和 reload 修复；只重放早期 `megatron-local-mok.patch` 会漏改动。MoK 的 Python 包和 CUDA 源码与本轮已运行版本逐文件核对。旧实验二进制不打包，新 `.so` 在新镜像内用其自身 ABI 构建。历史二进制 SHA 仅记录来源，构建路径和工具链可能使新二进制 SHA 不同。

`build.py` 使用 `pip wheel --no-deps --no-build-isolation`，再安装生成的 wheel。CUDA driver stub 只用于编译链接，不写入最终运行时的 `LD_LIBRARY_PATH`。没有 NVIDIA driver 挂载的普通 `docker build` 只检查源码、安装路径和 SM103 cubin，完整扩展导入留到容器启动后验证。

增量镜像也定点应用 TE 2.18 的 bounded Adam 修复，避免8卡完整模型超过临时 NVTETensor 句柄上限。按源码 SHA 判断原版／已修复版，其他版本拒绝；不重装 TE。32卡 PP4 的已完成实验未触发该限制。

## 启动后验证

先在新镜像内检查导入、来源和 Megatron 配置：

```bash
docker run --rm --gpus all --entrypoint python relax-mok:pr10-sm103 \
  /opt/relax/mok-build/verify.py --runtime
```

在空闲的单机8卡 NVLink 域上，使用现有容器的 GPU/IPC/共享内存配置启动，再运行已有 kernel 验证：

```bash
python -m torch.distributed.run --standalone --nproc-per-node=8 \
  -m pytest --import-mode=importlib -q \
  /opt/relax/mixture-of-kittens/tests/test_relax_640.py
python -m torch.distributed.run --standalone --nproc-per-node=8 \
  /opt/relax/mok-build/validation/mcore_tp2_smoke.py
```

第一条核对 BF16 输出、输入/路由/专家梯度，以及非零累积梯度；覆盖原生 `H2560/I640/E512/K10`、空 rank、丢弃 token 和跨 macro。第二条与原生 Megatron 层比较，覆盖 TP2/SP/EP8/ETP1、native shared、padding 和 recompute。Pytest 使用 importlib 模式，防止源码目录遮住已安装 wheel。

也可在挂载当前 Relax 仓库的运行容器中执行已有 CPU 合约测试：

```bash
CUDA_VISIBLE_DEVICES='' python -m pytest -q \
  /opt/relax/mok-build/validation/test_mok_fixed_tokens.py \
  /opt/relax/mok-build/validation/test_mok_live_ep_group.py
```

此构建包已在独立目录重放补丁并核对源码指纹；本次环境没有 Docker CLI，**尚未实际构建或验证新镜像**。上述 GPU 命令供新镜像构建后执行，未在占卡期间额外启动 GPU 测试。

## 使用镜像中的安装

所有 Ray worker 和 driver 使用同一个新镜像。训练时设置 `MEGATRON=/root/Megatron-LM`（或构建时指定的路径），并让 Ray 的 `PYTHONPATH` 使用当前 Relax 仓库和该路径。移除旧 `mok-integration-*`、`patched/mok` 等实验覆盖路径，否则会遮住镜像中的 wheel；`verify.py --runtime` 会检测这种情况。

`macrobatch_size` 是训练配置，不应固化为所有任务通用的镜像默认值。256K 长文本实验可使用随包的 `configs/mok-256k.json`：

```bash
export MEGATRON=/root/Megatron-LM
export MOE_BACKEND=mok
export MOK_BACKEND_CONFIG="$(cat /opt/relax/mok-build/configs/mok-256k.json)"
```

该配置的 fixedN 为131072、macro 为1572864。第08组 fixedN=8192、macro=327680，见 `configs/mok-group08.json`。两组不能直接互换；fixedN 必须覆盖真实物理输入，短输入无需按长文本上限占用 workspace。

本 Dockerfile 只封装 Megatron/MoK；Relax 框架改动和训练脚本保留在仓库，随 PR 正常安装或挂载。本轮 SFT 截断 `<think>` 兼容也需要当前 Relax 版本。

## 实验边界

已完成的对照使用32×Blackwell SM103 GPU，**EP8 位于各节点的8卡 NVLink 域内**。本包没有新增跨 NVLink 域的 EFA token 传输；配置枚举允许 EP16 不代表当前机器能够直接互访16卡显存。

本轮为短跑性能验证，未建立收敛等价性。第04组 step7 的 preclip global gradient norm 相对差异约14.18%，原因未定位；第08组最大约6.12%。这些差异需随 PR 如实记录，不能只凭 loss 接近宣称完整训练数值等价。
