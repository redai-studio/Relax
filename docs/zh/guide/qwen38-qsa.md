# Qwen3.8 Flash/NV QSA

可选的 cuDNN QSA 后端替换 Qwen3.8 mixed QSA 的长文档 Grouped attention 分支，短文档仍使用因果 Flash Attention。原有 selector 及其四 token block ID 保持不变。

适配器支持 CUDA BF16、TP2 下 local 12Q/1KV、D256、CP1，以及冻结的 Stage-1 indexer，不支持 TP8 下的 local 3Q/1KV attention。Packed 输入必须携带 producer 提供的 CPU 文档边界；每篇文档独立处理 token 位置和不完整的因果尾块。

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

两套脚本均使用 4 节点 × 8 GPU、48 层、global 24Q/2KV、TP2/PP4/CP1/EP8/ETP1 和 sequence parallelism，采用 BF16、GBS512/MB1、每两层一次的 full uniform recompute，以及一层 MTP。

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
