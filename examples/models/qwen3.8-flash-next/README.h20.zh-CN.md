# Qwen3.8 Flash Next：40 卡 H20 SFT

[English](./README.h20.md)

本 recipe 用于 5 个节点、每节点 8 张 H20 的多模态 SFT。模型保留完整 48 层和
1 层 MTP，采用 TP4/PP5/DP2/EP4，流水线布局为
`Et*8|t*11|t*11|t*12|t*6mL`，即 8/11/11/12/6 层加 MTP。
默认 packing token 预算为 65,536，global batch size 为 1,024。
视觉模型、视觉投影、PLE 和 attention indexer 冻结，PLE 放在 CPU；
冻结的视觉组件仍参与多模态前向计算。训练参数保留在 GPU，使用完整 uniform 激活重计算。

## 环境要求

- 5 个专用节点，每节点 8 张 H20。所有节点使用相同的 Relax 源码，以及兼容的
  Megatron-LM、Megatron Bridge、DeepEP、Transformer Engine 和支持 Qwen3.8 的
  Transformers 依赖。标准镜像中的 Megatron-LM 路径为 `/root/Megatron-LM`；
  自定义安装位置可通过 `MEGATRON` 覆盖。recipe 不安装依赖，也不修改训练镜像。
- 完整的 Hugging Face 模型目录，包含权重、配置、tokenizer 和多模态 processor 文件。
- 使用 `messages`、可选 `tools` 字段及受支持多模态消息格式的 JSONL SFT 数据。
  所有 worker 都必须能访问数据引用的媒体路径。Chat template 使用
  `preserve_thinking=true` 和 `add_generation_prompt=false`。
- 运行环境已配置 ClearML 凭据。

模型、数据、输出目录和源码必须在所有节点以相同路径可访问。
启动前，在 worker 环境中检查模型配置、tokenizer/processor 和训练入口能否正常导入。

## 配置

[`launch.sh`](launch.sh) 集中管理路径、实验名称和常用训练参数。
可以编辑其中的默认值，也可以通过环境变量覆盖。以下三个路径必须由用户提供：

```bash
export MODEL_DIR="/path/to/hf-model"
export DATA_DIR="/path/to/train.jsonl"
export SAVE_DIR="/path/to/run-output"
```

`DATA_DIR` 可以是具体的 `.jsonl` 文件，也可以是包含 `train.jsonl` 的目录。
`SAVE_DIR` 用于存放提交日志，独立实验应使用不同的输出目录。无需额外的 `launch.env` 文件。

| 变量                 | 默认值                          | 用途                                                      |
| -------------------- | ------------------------------- | --------------------------------------------------------- |
| `RELAX_DIR`          | 根据 `launch.sh` 的位置自动定位 | 选择完整的 Relax checkout，包含本 recipe。                |
| `EXP_NAME`           | `qwen38-qsa-40xh20`             | 稳定的实验名称；ClearML task 名称会追加启动时间。         |
| `PROJECT_NAME`       | `Relax/sft/qwen38-qsa`          | ClearML 项目。                                            |
| `NUM_STEPS`          | `10`                            | 本次请求执行的训练更新步数。                              |
| `GBS`                | `1024`                          | Global batch size，需要能被固定的 dense DP 大小 2 整除。  |
| `SEQ_LENGTH`         | `65536`                         | 序列长度和最大输入上下文长度。                            |
| `MAX_TOKENS_PER_GPU` | `65536`                         | 动态 batching/packing 的 token 预算，与序列长度分开配置。 |
| `LR` / `MIN_LR`      | `2e-5` / `2e-6`                 | 峰值和最小学习率。                                        |
| `LR_DECAY_ITERS`     | `2023`                          | Cosine 学习率调度长度，不会随 `NUM_STEPS` 自动调整。      |
| `LR_WARMUP_FRACTION` | `0.05`                          | 学习率调度中用于 warmup 的比例。                          |
| `SEED`               | `1234`                          | 同时设置训练和数据/rollout 随机种子。                     |

例如，设置三个必需路径后，可以覆盖 batch size 和实验名：

```bash
GBS=512 NUM_STEPS=10 EXP_NAME=qwen38-h20-gbs512 \
bash examples/models/qwen3.8-flash-next/launch.sh
```

通过 `launch.sh` 启动，或直接调用训练脚本，都会从 `launch.sh` 读取相同的默认配置。
`RELAX_DIR` 可以指定另一个完整的 checkout，该路径必须在所有节点一致，且目标 checkout
必须包含本 recipe。目标 checkout 中的训练脚本、模型配置和 worker 导入路径会配套使用。
在已有 Ray 集群上启动时，标准 Ray Jobs 入口也来自目标 checkout。
`MEGATRON` 继续在训练脚本中配置。

调整 batch size、序列长度或 packing 预算后，需要重新检查显存占用和训练行为。
固定的 40 卡并行布局、冻结和 offload 设置仍位于训练脚本中。

## 启动

对于多节点 TorchJob，在**每个节点**从仓库根目录执行以下命令。
标准入口会启动 Ray，并由 head 节点提交训练，其他节点作为 worker 加入集群。

```bash
bash scripts/entrypoint/spmd-multinode.sh examples/models/qwen3.8-flash-next/launch.sh
```

任务环境需要提供 `MASTER_ADDR`（head Pod 的主机名，与其 `POD_NAME` 一致）、
`POD_NAME`、`HOST_IP` 和 `WORLD_SIZE=5`。每个节点默认使用 8 张 GPU。
这个入口会清理残留进程并创建新的 Ray 集群，因此应在专用任务节点上使用。

这些节点变量必须在启动 SPMD 入口前通过任务环境设置。
SPMD 入口在执行 `launch.sh` 前读取 `WORLD_SIZE`，放在 `launch.sh` 内设置会太晚。

如果 TorchJob 使用另一个 checkout，应从目标 checkout 同时调用 `spmd-multinode.sh`
和 `launch.sh`。`launch.sh` 中的 `RELAX_DIR` 无法切换已经启动的 SPMD 入口。

对于**已经运行的 Ray 集群**，仅在 Ray head 执行：

```bash
bash examples/models/qwen3.8-flash-next/launch.sh
```

该方式会调用 `scripts/entrypoint/ray-job.sh`，默认保留已有任务，
并在提交前检查是否存在其他活动的 Relax 训练任务；存在时拒绝重复提交。
Ray Jobs 地址默认为 head 本机的 `http://127.0.0.1:8265`，
仅在使用其他服务地址时覆盖 `RAY_JOBS_ADDRESS`。

两种启动方式使用同一个 `scripts/` 下的训练脚本，默认阻塞等待训练结束并持续输出日志。
提交日志写入 `${SAVE_DIR}/logs/<submission-id>.submit.log`。
如需异步提交，设置 `RAY_NO_WAIT=1`；阻塞运行时保持该变量未设置或为空。
`RAY_NO_WAIT=0` 也是非空值，同样会启用异步提交。

## 训练步数、指标与 checkpoint

默认执行 **10 步短测**，通过 `NUM_STEPS` 调整训练长度。
`EXP_NAME` 和 `PROJECT_NAME` 可以覆盖脚本中的 ClearML 实验名和项目默认值。
Task 名称包含实验名和启动时间，每次提交会创建独立的 task。

每次启动都从 HF 模型初始化。当前 recipe **不保存或恢复训练 checkpoint**。
额外的命令行参数会透传给训练入口。

## 预览与编译缓存

设置三个路径变量后，可以预览训练参数和运行时配置。
预览不会提交任务、导入 GPU 依赖或创建输出文件：

```bash
DRY_RUN=1 bash examples/models/qwen3.8-flash-next/launch.sh
```

编译缓存默认使用节点本地的 `${TMPDIR:-/tmp}/relax-qwen38` 子目录，
标准入口的 kernel-cache 配置仍然有效。仅在需要更换本地缓存目录时，
覆盖 `TORCHINDUCTOR_CACHE_DIR`、`TRITON_CACHE_DIR` 或 `CUDA_CACHE_PATH`。
