# Qwen3.8 Flash/NV QSA

可选的 cuDNN QSA 后端替换 Qwen3.8 mixed QSA 的长文档 Grouped attention 分支，短文档仍使用因果 Flash Attention。原有 selector 及其四 token block ID 保持不变。

适配器支持 CUDA BF16、D256、CP1 和冻结的 Stage-1 indexer。本地 attention 头数支持 TP1 的 24Q/2KV 和 TP2 的 12Q/1KV。Packed 输入必须携带 producer 提供的 CPU 文档边界；每篇文档独立处理 token 位置和不完整的因果尾块。

## 运行环境

使用兼容的 CUDA 13 Blackwell 训练环境，包含 Megatron/Bridge、HybridEP 和 Flash Attention。可选依赖为 [NVIDIA cuDNN Frontend PR1765](https://github.com/NVIDIA/cudnn-frontend/pull/1765)，固定到 commit **627c23f62e860fe2f76d6b38d2efcc2ee84c70cd**，CuTeDSL 使用 **4.8.1**。

仓库的 CUDA 13 Blackwell 镜像通过 PIP_CONSTRAINT 将 CuTeDSL 固定为 4.6.2，直接安装 4.8.1 会发生约束冲突。仅创建 venv 也不会清除继承的环境变量。应构建 Qwen3.8 NV 专用派生镜像，避免改动正在使用的基础环境。将下面内容保存为 Dockerfile.qsa：

~~~dockerfile
# syntax=docker/dockerfile:1.7
ARG RELAX_BASE_IMAGE
FROM ${RELAX_BASE_IMAGE}

RUN python - <<'PY'
import os
import re
from pathlib import Path

source = Path(os.environ["PIP_CONSTRAINT"])
replaced = re.compile(
    r"^\s*nvidia[-_.](?:cutlass[-_.]dsl|cudnn[-_.]frontend)\s*(?:[<=>!~@]|$)",
    re.IGNORECASE,
)
lines = [line for line in source.read_text().splitlines() if not replaced.match(line)]
lines.append("nvidia-cutlass-dsl==4.8.1")
Path("/etc/relax-qsa-runtime-constraints.txt").write_text("\n".join(lines) + "\n")
PY

ENV PIP_CONSTRAINT=/etc/relax-qsa-runtime-constraints.txt
RUN python -m pip install --no-cache-dir --upgrade \
      'nvidia-cutlass-dsl[cu13]==4.8.1' \
      'nvidia-cudnn-frontend @ git+https://github.com/NVIDIA/cudnn-frontend.git@627c23f62e860fe2f76d6b38d2efcc2ee84c70cd' && \
    python -c 'import importlib.metadata as m; assert m.version("nvidia-cutlass-dsl") == "4.8.1"; from cudnn.QSA.api import qsa_forward, qsa_backward'
~~~

这会复制基础约束，只替换 CuTeDSL 和 frontend 两个包的约束：CuTeDSL 固定版本，frontend 在安装命令中固定源码 commit。Torch、NumPy、Transformer Engine 等其余约束继续生效，不要全局取消 PIP_CONSTRAINT，也不要修改原始约束文件。

~~~bash
docker build --build-arg RELAX_BASE_IMAGE='<your-cu13-blackwell-image>' \
  -f Dockerfile.qsa -t relax-qsa:pr1765 .
~~~

所有 Ray 节点（包括 head）都要使用派生镜像。从源码构建固定版本还需要上游构建依赖和兼容的 cuDNN 安装，具体准备方式见该版本的[上游安装说明](https://github.com/NVIDIA/cudnn-frontend/tree/627c23f62e860fe2f76d6b38d2efcc2ee84c70cd)。仅有名为 cudnn 的其他 Python 包并不足够。Relax 不会自动安装或替换依赖；框架默认仍使用 Grouped，也不会导入这个可选后端。

构建和导入检查只覆盖依赖解析、CuTeDSL 版本及 QSA 入口，不代表已经验证 GPU 数值、TP/PP 重算，或其他 kernel 与新版 CuTeDSL 的兼容性。训练前须在派生镜像中验证相关路径；这份 Qwen3.8 专用环境不能直接视作所有 Blackwell 配方已验证的替代镜像。

## 选择训练入口

两套脚本默认使用 4 节点 × 8 GPU、48 层、global 24Q/2KV、TP2/PP4/CP1/EP8/ETP1 和 sequence parallelism，采用 BF16、GBS512/MB1、每两层一次的 full uniform recompute，以及一层 MTP。

在任一 NV 脚本的命令末尾追加 `--tensor-model-parallel-size 1` 即可选择 TP1。Megatron 会在 TP1 下关闭 sequence parallelism。

| 脚本 | 各 PP 阶段层数 | LR decay iterations | 输入 |
| --- | --- | --- | --- |
| 纯文本 | 14 / 12 / 12 / 10 | 996 | 文本消息和 tool calls |
| 多模态 | 12 / 14 / 14 / 8 | 2023 | 带图像引用的消息 |

纯文本分配让每个 PP 阶段各有三个主干 QSA 层。最后一段还包含 MTP QSA 和输出头，计入后 QSA 数量为 3/3/3/4。多模态脚本保留自身的 PP 分配；调整该布局属于单独的调优工作。

在仓库根目录运行。MODEL_DIR 是完整 HF checkpoint 目录，DATA_DIR 是 JSONL 文件或目录，SAVE_DIR 是所有节点都可访问的持久 checkpoint 根目录：

~~~bash
export MODEL_DIR=/models/qwen38
export DATA_DIR=/datasets/text
export SAVE_DIR=/checkpoints/qwen38
export MEGATRON=/workspace/Megatron-LM
export RELAX_HYBRID_EP_COMMIT='<installed-hybridep-commit>'
export RAY_JOBS_ADDRESS='http://<ray-head>:8265'

EXP_NAME=qsa-text-nv NUM_STEPS=50 \
  bash examples/models/qwen3.8-flash-next/scripts/run-qwen38-qsa-nv-text-sft-32xgpu-blackwell.sh
~~~

进行图文 SFT 时，改为提供多模态 HF checkpoint 和数据，并使用：

~~~bash
EXP_NAME=qsa-mm-nv NUM_STEPS=50 \
  bash examples/models/qwen3.8-flash-next/scripts/run-qwen38-qsa-nv-sft-32xgpu-blackwell.sh
~~~

两套脚本都读取 messages 和 tools 字段。多模态脚本额外把 image 映射到 images、将每张图像限制为 128 token，并在训练 rank 上预处理图像。纯文本脚本省去这三个多模态选项，不会把图像数据转成文本；使用该入口时应提供纯文本数据集。

脚本通过 scripts/entrypoint/ray-job.sh 准备集群，再提交 Ray Job。需要独占 Ray Serve 工作环境；直接调用时发现已有运行作业会拒绝启动，不提供并发作业隔离。默认阻塞等待提交的作业结束；只有需要单独监控返回的 job 时，才设置 RAY_NO_WAIT=1。

可以先查看有效配置，不触发 Ray 初始化、网络访问、GPU 导入或输出目录创建：

~~~bash
DRY_RUN=1 \
  bash examples/models/qwen3.8-flash-next/scripts/run-qwen38-qsa-nv-text-sft-32xgpu-blackwell.sh
~~~

## 分流规则与对照

两套脚本会把相同设置传给 driver 和训练 Actor：

| 设置 | 脚本值 | 含义 |
| --- | --- | --- |
| RELAX_QWEN38_QSA_COMMUNITY_BACKEND | mixed | 沿用现有 selector 和 packed 输入路径 |
| RELAX_QWEN38_QSA_MIXED_LONG_BACKEND | cudnn_pr1765 | 长文档使用 NV |
| RELAX_QWEN38_QSA_MIXED_FLASH_MAX_LEN | 2051 | 不超过 2051 token 的文档使用 Flash |
| RELAX_QWEN38_QSA_MIXED_FLEX_MAX_LEN | 0 | 关闭中间的 Flex 分支 |
| RELAX_QWEN38_QSA_NV_MAX_PACK_TOKENS | 0 | 每篇长文档分别调用 NV |
| NVTE_USE_CUTLASS_GROUPED_GEMM | 0 | 默认关闭独立的 MoE GEMM 选项 |
| CUDA_DEVICE_MAX_CONNECTIONS | 1 | 对齐 TP/sequence-parallel 运行时 |

2051 由 512 个完整四 token block 加最多三个尾部 token 构成；长度 2052 的文档进入稀疏长分支。分流依据 producer 提供的真实单篇文档长度，而不是整个拼接 pack 的长度。

要运行相同纯文本配置的 Grouped 对照，只覆盖长分支和实验身份：

~~~bash
RELAX_QWEN38_QSA_MIXED_LONG_BACKEND=grouped EXP_NAME=qsa-text-grouped \
  bash examples/models/qwen3.8-flash-next/scripts/run-qwen38-qsa-nv-text-sft-32xgpu-blackwell.sh
~~~

不同 arm 要使用不同 EXP_NAME 或 SAVE_DIR，避免从对方的 checkpoint 恢复。可以单独设置 NVTE_USE_CUTLASS_GROUPED_GEMM=1 做 MoE 对照，但它既不选择 QSA 实现，也不能证明某个 MoE kernel 实际执行。NV mini-pack 属于实验选项，两套脚本均保持关闭。

## Checkpoint 与吞吐口径

脚本把 load/save 配置为稳定的 SAVE_DIR/EXP_NAME 路径，每 SAVE_INTERVAL 步保存一次（默认 50），保留两个 Actor checkpoint。恢复同一实验时保持路径不变，不要追加新时间戳；每次启动的 ClearML task 和日志名称会追加时间戳。脚本通过现有 Megatron 后端配置 checkpoint，dry-run 只检查配置，不代表 optimizer/RNG 状态已成功保存和恢复。

两套脚本关闭自定义 record-function 标签和 autotune 日志，默认不启用 PyTorch profiler；producer 队列超时为 120 分钟，自动全局重启次数为零。需要显式采集诊断 trace 时，参考[性能调优](./performance-tuning.md)。

对比 Actor 吞吐时，在相同 step 窗口使用 **有效训练 token 总数 / Actor 训练总秒数 / GPU 数量**。端到端吞吐则使用总 step 秒数，不能混算。保留慢步，将冷编译和 checkpoint 保存单列，并对齐数据、拓扑、packing 和重算设置。QSA 单算子的提速不会等比例转成整个训练流水线的提速。

## Qwen3.8 QSA 算子与 Mixed 配置

QSA 首先由 selector 选择参与计算的历史块，再由 attention 算子完成计算。本文整理当前支持的算子、并行配置条件及开启方式。相关开关仅改变 attention 的执行路径，不改变 selector 的选择结果。

### 算子、并行条件与开启方式

表中的并行条件以当前模型的全局 **24Q/2KV/D256** 为例。TP 决定每张卡的本地头数；CP 决定同一条序列是否沿 token 维度跨卡切分。**CP1 表示不做上下文切分，不表示只使用一张 GPU。**

表中符号说明：

- **Q/KV/D**：每个 TP rank 上的 attention Q 头数、KV 头数和每头维度，不是 selector 的头数。
- **R4**：每个历史块包含 4 个 token。常用的 2048-token 选择预算对应 512 个块 ID，另补当前块内的因果尾部 token。
- **GQA**：Q 头数能被 KV 头数整除。
- 表中 `FUSED_ATTENTION`、`SPARSE_BLOCKMASK`、`MIXED_LONG_BACKEND` 等短名均需加上 **`RELAX_QWEN38_QSA_`** 前缀；`MCORE_QSA_SPARSE_BACKEND` 使用原名。

非 Mixed 路线先清空 community 并关闭 fused attention，避免继承模型脚本的默认开关；然后按表选择。环境变量要在构建模型前设置，并一致传给 driver 和训练 Actor。

```bash
export RELAX_QWEN38_QSA_COMMUNITY_BACKEND=''
export RELAX_QWEN38_QSA_FUSED_ATTENTION=0
```

| 算子 | 具体做法 | 并行配置条件 | 主要限制 | 开启方式 |
| --- | --- | --- | --- | --- |
| cuDNN PR1765 | 将选中的块转换为 byte mask，调用 NVIDIA QSA 算子 | **要求 CP1**，接受 TP1 的 24Q/2KV/D256 和 TP2 的 12Q/1KV/D256 | CUDA BF16；冻结 indexer、indexer loss 为 0；R4、512 列 int32 块 ID；需要额外依赖 | 通过 Mixed 设置 `MIXED_LONG_BACKEND=cudnn_pr1765`，完整配置见下文；`MIXED_FLASH_MAX_LEN=0` 可使所有文档使用 cuDNN |
| Grouped | 对共享同一 KV 头的 Q 头进行分组计算，将选中的块展开为 token 索引，仅访问这些 token | **要求 CP1**；本地 Q 头数能被 KV 头数整除，D256。本模型 TP1/2/4/8 的本地布局均满足头数条件，不固定要求 TP2 | 通过 Mixed 接入；CUDA BF16、冻结 indexer；反向用 FP32 atomic 累加 K/V 梯度 | Mixed 公共配置 + `MIXED_LONG_BACKEND=grouped`，将 Flash 和 Flex 上限都设为 `0` |
| Flash | 对短文档直接执行完整的因果 attention | **要求 CP1**。Grouped 路线接受本模型的 TP1/2/4/8，cuDNN 路线接受 TP1/2。所有文档都走 Flash 时也会检查所选路线的头数 | 仅作为 Mixed 内部分支；不能用它替代长文档的稀疏选择；需要 Flash Attention | 通过 Mixed 的 `MIXED_FLASH_MAX_LEN` 控制，默认自动取选择预算 + 3 |
| Triton v1 | 按 bitset 标记的稀疏块计算 attention，复用 selector 结果 | 本地 Q 头数能被 KV 头数整除，本模型 TP1/2/4/8 满足头数条件；CP1 直接计算，CP>1 由原生框架收集 Q/K/V 后计算，再切分本地输出 | CUDA、R4、D 为 64/128/256；selection 为单行；非 THD 输入要求 batch=1；不满足条件时回退到 Flex | `MCORE_QSA_SPARSE_BACKEND=flex`、`FUSED_ATTENTION=triton` |
| Triton v2 | 沿用 v1 的选择结果，调整循环、tile 和反向计算 | TP 和 CP 条件与 v1 相同：不固定为 TP2；CP>1 通过框架处理通信 | 同 v1 | `MCORE_QSA_SPARSE_BACKEND=flex`、`FUSED_ATTENTION=triton_v2` |
| Triton v3 | 沿用 v2 前向，反向可共享 FP32 dS，减少重复计算 | TP 和 CP 条件与 v1 相同；使用 CP local-query 时不启用共享 dS 优化，仍保留常规反向路径 | 同 v1；共享 dS 另限 SM103/BF16/D256、Q/K 长度不超过 128K，并受地址范围和临时显存限制；不满足时使用常规反向路径 | `MCORE_QSA_SPARSE_BACKEND=flex`、`FUSED_ATTENTION=triton_v3` |
| flex | 根据选择结果生成 mask，调用 PyTorch FlexAttention | 本地 Q 头数能被 KV 头数整除，本模型 TP1/2/4/8 满足头数条件；CP1 直接计算，CP>1 由原生框架收集 Q/K/V、计算并切分输出 | 需要 FlexAttention 和编译支持 | `MCORE_QSA_SPARSE_BACKEND=flex`、`FUSED_ATTENTION=0`、`SPARSE_BLOCKMASK=0` |
| BlockMask Flex | 使用 FlexAttention，优化 BlockMask 的构建 | TP 和 CP 条件与 Flex 相同；BlockMask 优化仅改变 mask 构建，不改变并行布局 | 优化要求 CUDA、R4、非空 bitset，否则回退到原生 Flex | Flex 配置 + `SPARSE_BLOCKMASK=1` |
| id_sparse | 直接读取选中的块 ID，通过原生 Triton kernel 完成计算 | 本地 Q 头数能被 KV 头数整除，本模型 TP1/2/4/8 满足头数条件；CP1 直接计算，CP>1 由原生框架收集 Q/K/V、计算并切分输出 | 连续 CUDA Q/K/V；D 为不超过 256 的 2 的幂；int32 IDs/位置；正 top-k；压缩比不超过 32 且能整除 32 | 清空 community，`MCORE_QSA_SPARSE_BACKEND=id_sparse`、`FUSED_ATTENTION=0` |


并行配置说明：

- **原生与 Mixed 的区别**：原生 `id_sparse` 可接原生 CP 流程；设置 `COMMUNITY_BACKEND=mixed` 后，同一入口的计算由 Mixed 替换，必须遵守 CP1 限制。
- **CP**：原生路径在 CP>1 时先汇集完整 Q/K/V，计算完成后切分为本地输出。Flex/Triton 可配合 `LOCAL_QUERY=1` 只保留本地 Q、汇集 K/V，但要求 THD、CUDA、R4 bitset 和冻结的 Stage-1 indexer 等条件；不满足时回退到原生路径。Mixed 当前只允许 CP1。
- **SP**：selector 会按 TP group 汇集序列。SP 不改变表中的本地 attention 头数要求；可选 TP selector 只用于 Mixed，并要求汇集后的序列长度能被 TP 整除。
- **PP / EP**：分别负责层分配和 MoE 通信。Qwen3.8 Mixed 使用 HybridEP 时，需要 CP1、ETP1、EP4 或 EP8，以及可整除的总 rank 数。EP×ETP 组的 rank 必须在同一台主机上且互相可访问，加载的模块和安装包必须匹配 `RELAX_HYBRID_EP_COMMIT`。

### Mixed 几何与文档长度分流

**Mixed 是按单篇文档的 token 长度选择 attention 算子的分流机制，本身不是独立的计算算子。** 当前有两条路线：**Flash/Flex/Grouped** 和 **Flash/cuDNN**。

第一条路线中，`mixed0` 和 `mixed16k` 是两种配置名称：分别表示关闭 Flex 分支、将 Flex 分支上限设为 16K。Grouped 是该路线中实际执行长文档计算的算子。两种配置使用相同的并行条件，仅分流边界不同。

第二条路线将长文档后端替换为 `cudnn_pr1765`：将 selector 选出的块转换为 byte mask，交给 NVIDIA cuDNN Frontend PR1765 的 QSA 实现计算；短文档仍使用 Flash，没有 Flex 中间分支。它与 Grouped 使用相同的选择结果，区别在于稀疏数据的表示与计算实现。

`mixed0` / `mixed16k` 不是 backend 可选值，两者均设置 `COMMUNITY_BACKEND=mixed`、`MIXED_LONG_BACKEND=grouped`。两条路线都要求 CUDA BF16、CP1、冻结的 Stage-1 indexer，且 indexer loss 为 0。Packed 输入需要携带 producer 提供的准确文档边界，每篇文档各自计算位置和因果尾块。

#### 本地头数与 TP

| Mixed 长文档后端 | 本地 attention 条件 | 对 global 24Q/2KV/D256 模型的含义 |
| --- | --- | --- |
| `grouped` | Q/KV 头数为正，Q 能被 KV 整除，D256 | TP1/2/4/8 分别得到 24Q/2KV、12Q/1KV、6Q/1KV、3Q/1KV，均能通过几何检查 |
| `cudnn_pr1765` | **24Q/2KV/D256** 和 **12Q/1KV/D256** | 分别对应 TP1 和 TP2。仅含短文档的 batch 也适用此检查 |

TP4/8 时，模型投影层会在 TP rank 之间共享 KV 组。对 TP4 的 6Q/1KV 或 TP8 的 3Q/1KV，设置 `RELAX_QWEN38_QSA_MIXED_LONG_BACKEND=grouped`。

#### 按单篇文档长度分流

以常用的 **2048-token 选择预算（512 个 R4 块）**为例：

| 路线 | ≤2051 token | 2052～16384 token | >16384 token |
| --- | --- | --- | --- |
| mixed16k：Flash/Flex/Grouped，Flex 上限 16K | Flash | Flex | Grouped |
| mixed0：Flash/Grouped，Flex 上限 0 | Flash | Grouped | Grouped |
| Flash/cuDNN | Flash | cuDNN PR1765 | cuDNN PR1765 |

2051 = 2048 个选中 token + 最多 3 个因果尾部 token。在这个范围内，不需要丢弃可见历史，因而可用完整的因果 Flash attention。长度按**单篇文档**计算：例如 32K 的 pack 里装了多篇 1K 文档，这些文档仍各自走 Flash。

Grouped 路线的自动 Flash 边界随选择预算变化，为 `4 × 块 ID 列数 + 3`；cuDNN 路线固定要求 512 列块 ID，自动边界为 2051。

#### 开启方式

先设置两条 Mixed 路线共用的配置。模型自身仍需设置为冻结 indexer、indexer loss 为 0；这些环境变量只负责选路。

```bash
export RELAX_QWEN38_QSA_COMMUNITY_BACKEND=mixed
export MCORE_QSA_SPARSE_BACKEND=id_sparse
export RELAX_QWEN38_QSA_FUSED_ATTENTION=0
export RELAX_QWEN38_QSA_FUSED_SELECTOR=0
```

**路线一：Flash/Flex/Grouped。** 以下为 mixed16k 配置，在 16K 以内保留 Flex 中间分支；将 `MIXED_FLEX_MAX_LEN` 改为 `0` 即为 mixed0，采用 Flash/Grouped 两段分流。

```bash
export RELAX_QWEN38_QSA_MIXED_LONG_BACKEND=grouped
export RELAX_QWEN38_QSA_MIXED_FLASH_MAX_LEN=-1
export RELAX_QWEN38_QSA_MIXED_FLEX_MAX_LEN=16384
```

**路线二：Flash/cuDNN。** 没有 Flex 中间分支；超过 Flash 边界后使用 cuDNN。

```bash
export RELAX_QWEN38_QSA_MIXED_LONG_BACKEND=cudnn_pr1765
export RELAX_QWEN38_QSA_MIXED_FLASH_MAX_LEN=-1
export RELAX_QWEN38_QSA_MIXED_FLEX_MAX_LEN=0
export RELAX_QWEN38_QSA_NV_MAX_PACK_TOKENS=0
```

cuDNN 路线需要兼容的 CUDA 13 Blackwell 环境，以及额外的 NVIDIA cuDNN Frontend PR1765 包，当前接入固定到 commit `627c23f62e860fe2f76d6b38d2efcc2ee84c70cd`，CuTeDSL 为 `4.8.1`。仓库基础镜像将 CuTeDSL 固定为 `4.6.2`，需使用准备好这些依赖的专用环境。`NV_MAX_PACK_TOKENS=0` 表示逐篇调用 cuDNN。

默认值与配置说明：

- 框架默认长后端是 `grouped`，Flash 上限为 `-1`（自动），Flex 上限为 `16384`；公共 QSA 训练脚本会把 Flex 上限设为 `0`，实际以启动脚本为准。
- Grouped 路线的 Flex 上限范围是 `0..16384`。若显式设置 Flash 上限，必须同时将 Flex 上限设为 `0`；保留 Flex 中间分支时，Flash 上限用 `-1`。
- 公共模型脚本默认开启 fused attention，Mixed 不能与它同时使用，所以公共配置中的 `FUSED_ATTENTION=0` 必须显式设置。

实现入口：[Mixed 分发](https://github.com/fxsc03/Relax/blob/fit/fengxiaoshi/QSA/relax/models/qwen3_8/ops/qsa_mixed_ops/mixed.py)、[几何检查](https://github.com/fxsc03/Relax/blob/fit/fengxiaoshi/QSA/relax/models/qwen3_8/ops/qsa_mixed_ops/geometry.py)、[cuDNN 适配](https://github.com/fxsc03/Relax/blob/fit/fengxiaoshi/QSA/relax/models/qwen3_8/ops/qsa_mixed_ops/cudnn.py)、[Triton 选择](https://github.com/fxsc03/Relax/blob/fit/fengxiaoshi/QSA/relax/models/qwen3_8/qsa_fused.py)。
