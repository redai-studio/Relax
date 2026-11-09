# Qwen3.5 CPT

Relax 复用 SFT pipeline 执行纯文本 CPT。数据为 JSONL，每行可使用 `text` 字段，或 ms-swift 的单 assistant `messages` 格式提供原始文本；训练不套用 chat template。Qwen3.5 CPT 模板会规范化文本，并屏蔽空 `<think>` 前缀。

## 训练入口

| 脚本                                | 资源       | 核心配置                                           |
| ----------------------------------- | ---------- | -------------------------------------------------- |
| `run-qwen3.5-4B-mixture-8xgpu.sh`   | 单机 8 卡  | 4B dense，GBS32，DP8，分源指标和 ClearML           |
| `run-qwen3.5-35B-A3B-cpt-8xgpu.sh`  | 单机 8 卡  | GBS128，TP4 / PP1 / CP1 / EP8                      |
| `run-qwen3.5-35B-A3B-cpt-16xgpu.sh` | 双机 16 卡 | GBS2048，TP2 / PP2（22/18 层）/ CP1 / EP4          |
| `run-qwen3.5-35B-A3B-cpt.sh`        | 8～128 卡  | 按节点数扩展 GBS，TP2 / PP2（22/18 层）/ CP1 / EP4 |

35B-A3B 配置使用 dynamic batching、`seq-length=4096`、`max-tokens-per-gpu=32768`、BF16、sequence parallel、Flash Attention、full recompute 和 shared expert overlap。

35B-A3B 脚本此前验证的镜像：

```text
relax:qs-20260916-89830e62
```

镜像中的 `/root/Megatron-LM` 已包含 expert grad norm 修复。

## 单机 8 卡

4B 多数据集 CPT 固定使用 `run-qwen3.5-4B-mixture-8xgpu.sh`。在包含本功能的 Relax checkout 根目录、已有 Ray 集群的 head 上执行。模型、数据、输出目录及 ClearML 配置需要在 head 和 worker 上以同一路径可读；输出目录还需可写。

### 准备数据

每行是一篇文档，例如 `{"text": "领域文档正文"}`。将多个数据目录写入 `mixture.yaml`：

```yaml
datasets:
  general:
    path: /data/general/
    weight: 3
  code:
    path: /data/code/
    weight: 2
  science:
    path: /data/science/
    weight: 1
```

YAML key 是曲线中的数据集名称；这里用 `science` 会生成 `loss_domain_science`。权重按逻辑 epoch 的样本配额生效，单个 step 不保证固定比例。目录下使用 JSONL / Parquet，三种来源采用相同的字段结构。相对路径以 YAML 所在目录为基准。长文档应提前切分；当前脚本使用右侧截断，超过容量的尾部会被丢弃。

### 启动一次新实验

```bash
export RAY_ADDRESS="<head-ip>:6379"
export HF_CHECKPOINT=/path/to/Qwen3.5-4B
export PROMPT_DATA=/path/to/mixture.yaml
export CPT_ROOT=/path/to/output
export CLEARML_CONFIG_FILE=/shared/path/clearml.conf
export PROJECT_NAME=Relax/CPT-MultiDataset
export RUN_NAME="qwen35-4b-cpt-$(date +%Y%m%d-%H%M%S)"

RAY_NO_WAIT=1 RELAX_SKIP_CLUSTER_CLEANUP=1 TRAIN_STEPS=20 \
bash scripts/entrypoint/ray-job.sh scripts/training/cpt/run-qwen3.5-4B-mixture-8xgpu.sh
```

每次新实验使用新的 `RUN_NAME`，它同时是 Ray submission ID、ClearML 实验名和 checkpoint 子目录名。该脚本从 HF 权重新开训练；复用实验名或输出目录不会自动续训。续训需要显式配置 Megatron `--load` 并保持 YAML、数据快照和关键训练配置一致，不能仅修改 `TRAIN_STEPS` 当作恢复训练。

`WANDB_ARGS` 默认启用 `--use-clearml --use-metrics-service`。通过 `CLEARML_CONFIG_FILE` 传递共享配置文件路径，不把凭据写进脚本、runtime-env 或日志。使用已有集群时先确认有可用的单机 8 卡，且没有其他训练占用；`RELAX_SKIP_CLUSTER_CLEANUP=1` 不会替你清理旧任务。

### 查看结果

脚本异步提交，训练日志由 Ray 保存。提交成功不等于训练完成，在 head 上使用：

```bash
ray job status --address=http://127.0.0.1:8265 "$RUN_NAME"
ray job logs --address=http://127.0.0.1:8265 "$RUN_NAME"
```

ClearML 打开 `PROJECT_NAME` 下对应 `RUN_NAME` 的 Scalars，保持 **Group by = Metric**：

| ClearML 图表 title                                | 曲线 series               | 内容                              |
| ------------------------------------------------- | ------------------------- | --------------------------------- |
| `train/loss`                                      | `loss`                    | 本 step 的整体训练 loss，独立纵轴 |
| `train/loss_domain_general`                       | `loss`                    | general 有效监督 token 的平均 NLL |
| `train/loss_domain_code`                          | `loss`                    | code 有效监督 token 的平均 NLL    |
| `train/loss_domain_science`                       | `loss`                    | science 有效监督 token 的平均 NLL |
| `train/dataset/tokens`、`samples`                 | 各数据集名称              | 各源实际 token 数、样本数         |
| `train/dataset/token_fraction`、`sample_fraction` | 各数据集名称              | 各源实际占比                      |
| `train`                                           | `grad_norm`、`lr-pg_*` 等 | 其他训练指标                      |

原始指标 key 仍是 `train/loss` 和 `train/dataset/<name>/<stat>`；只有 ClearML 显示映射按上述规则分图。无需手动双击图例隔离总 loss。跨图量纲不同，各自使用独立纵轴。某源本步没有有效监督 token 时不填零 loss，只上报计数。总 loss 及各源统计均复用训练 forward，不额外遍历数据集。

在新 ClearML 项目中可设置分组、横轴、平滑和图表可见性后点击 **Set as default** 保存项目默认视图；Plotly 图例的临时单曲线隔离不保证随该设置保存。旧验证实验可能仍有 `train/dataset/loss` 等历史图；新实验使用唯一 `RUN_NAME`，不会继承这些历史上报数据。

### 默认参数和 checkpoint

| 环境变量                               | 默认值       | 用途                                          |
| -------------------------------------- | ------------ | --------------------------------------------- |
| `TRAIN_STEPS`                          | `20`         | 首次短跑步数，正式训练按计划覆盖              |
| `GLOBAL_BATCH_SIZE`                    | `32`         | 全局每步样本数                                |
| `SEQ_LENGTH`                           | `1024`       | 模型序列长度配置                              |
| `MAX_TOKENS_PER_GPU`                   | `2048`       | 动态打包 token 容量及数据截断容量（当前 CP1） |
| `PREFETCH_WORKERS` / `PREFETCH_BUFFER` | `4` / `1024` | 预处理线程数 / 已处理样本缓存容量             |
| `IN_FLIGHT_STEPS`                      | `2`          | CPU prepack 和训练重叠的窗口                  |
| `SAVE_CHECKPOINT` / `SAVE_INTERVAL`    | `1` / `1000` | 启用保存 / 保存间隔，训练正常结束时也保存     |

Checkpoint 保存到 `${CPT_ROOT}/checkpoints/${RUN_NAME}`，包含 Megatron 模型、优化器及 SFT 数据游标；以 `latest_checkpointed_iteration.txt` 指示最新迭代。临时性能短跑可设置 `SAVE_CHECKPOINT=0`；正式任务保留保存，并预留足够存储空间。本次 4B 原生完整 checkpoint 实测约 59.6 GB，未进行 HF 导出。

该配置保留 checkpoint 的共享 embedding/output 权重，冻结视觉模块，关闭 MTP，使用 chunked logits 和 loss recompute。实际验证环境为单机 8 张 L20Y（每卡约 80 GB）、PyTorch 2.11.0+cu129、Megatron Core 0.19.0、Transformer Engine 2.14.1、Transformers 5.12.1；其他硬件、模型大小和并行拓扑需另行验证。

### 换业务数据后的验收

先短跑，保持模型、硬件、配比、数据快照和 token 预算一致。检查：

- 每步整体和分源 loss 有限，分源 counts 可加总，零监督源不伪造 loss；当前按 token 归一化配置下，分源 loss 按实际 token 占比加权应与整体 loss 对齐。
- 实际样本 / token 占比合理；单步波动正常，不能用单步是否严格满足 YAML 比例判断采样错误。
- 确认 Ray 状态成功、ClearML 曲线步数完整、checkpoint 与游标实际落盘。完整训练数值等价仍需独立验证，不能仅凭输入哈希一致认定梯度轨迹等价。

35B-A3B 示例：

```bash
CPT_ROOT=/path/to/Relax-CPT-data \
HF_CHECKPOINT=/path/to/Qwen3.5-35B-A3B \
PROMPT_DATA=/path/to/train.jsonl \
TRAIN_STEPS=10 \
bash scripts/training/cpt/run-qwen3.5-35B-A3B-cpt-8xgpu.sh
```

## 双机 16 卡

在 Ray Head 节点执行：

```bash
cd /path/to/Relax
ray serve shutdown -y || true

HF_CHECKPOINT=/path/to/Qwen3.5-35B-A3B \
PROMPT_DATA=/path/to/train.jsonl \
OUTPUT_ROOT=/path/to/outputs \
RUN_NAME=qwen35-cpt-16xgpu \
TRAIN_STEPS=10 \
RAY_NO_WAIT=1 \
bash scripts/entrypoint/ray-job.sh \
  scripts/training/cpt/run-qwen3.5-35B-A3B-cpt-16xgpu.sh
```

若仓库、模型和数据按相邻目录放置，16 卡脚本可直接使用默认值：

```text
<parent>/Relax
<parent>/Qwen3.5-35B-A3B
<parent>/Relax-CPT-data/cpt-real-200-20260909/train-repeat2-2048-v2.jsonl
```

日志写入 `${OUTPUT_ROOT}/${RUN_NAME}/train.log`。可覆盖 `TRAIN_STEPS`、`PROMPT_DATA`、`HF_CHECKPOINT`、`RUN_NAME`、`OUTPUT_ROOT`、`SEQ_LENGTH`、`MAX_TOKENS_PER_GPU`、`LR_DECAY_STEPS` 和 `WARMUP_STEPS`。

## 可扩展 H800 配置

通用脚本默认每个节点使用 8 张 GPU，支持 1、2、4、8、16 个节点：

```bash
NUM_NODES=4 \
HF_CHECKPOINT=/path/to/Qwen3.5-35B-A3B \
PROMPT_DATA=/path/to/train.jsonl \
RAY_NO_WAIT=1 \
bash scripts/entrypoint/ray-job.sh \
  scripts/training/cpt/run-qwen3.5-35B-A3B-cpt.sh
```

也可直接设置 `NUM_GPUS=8|16|32|64|128`。默认使用 TP2、PP2、CP1、EP4，GBS 按
`NUM_GPUS * 128` 扩展；16 卡配置已经实测，其他规模是扩展默认值，需要在对应集群上验证性能。
`TP_SIZE`、`PP_SIZE`、`CP_SIZE`、`EP_SIZE`、`ETP_SIZE`、`GLOBAL_BATCH_SIZE` 和
`MAX_TOKENS_PER_GPU` 均可通过环境变量覆盖。

该默认值保持每个 dense DP rank 每步 512 条样本，属于弱扩展配置；固定训练步数时，总样本量会随卡数增长。数据集至少需要覆盖一个 GBS，否则会在单步内跨 epoch 重复采样。
`NUM_NODES` 和 `GPUS_PER_NODE` 用于计算 Ray Actor 请求的总 GPU 数，不强制指定物理节点布局。

## 数据语义

- 支持 `{"text": "..."}` 与 `{"messages": [{"role": "assistant", "content": "..."}]}` 逐行混合；两者都只提取原始文本并使用相同的 CPT loss mask。
- `--prompt-data` 可直接指向 YAML 配比文件，在线循环采样多个 JSONL、Parquet、目录或文件列表。
- CPT 不套用 chat template。
- `--sft-cpt-template qwen3_5` 会去除首尾空白、规范化 `<think>` 换行，并屏蔽空思考前缀及其后的空白。
- 非空推理正文继续参与监督；`<thinking>` 作为普通文本。
- 数据层生成与 token 等长的 mask；Megatron 在样本内左移一次并屏蔽末位，避免跨文档预测。

多数据集配置示例：

```yaml
datasets:
  wiki:
    path: /data/wiki.jsonl
    weight: 0.6
  code:
    path:
      - /data/code-1.jsonl
      - /data/code-2.jsonl
    weight: 0.3
  books:
    path: /data/books/
    weight: 0.1
```

权重会自动归一化；必须全部配置或全部省略。默认逻辑 epoch 会覆盖每个数据源至少一次，较小的数据源按权重循环采样。权重在完整逻辑 epoch 内精确兑现，shuffle 后单个 step 的比例会有统计波动。也可设置顶层 `epoch_size`，但不能小于覆盖全部数据源一次所需的最小值。YAML 中的相对路径以配置文件所在目录为基准。
