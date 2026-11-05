# Qwen3.5-35B-A3B CPT

Relax 复用 SFT pipeline 执行纯文本 CPT。数据为 JSONL，每行可使用 `text` 字段，或 ms-swift 的单 assistant `messages` 格式提供原始文本；训练不套用 chat template。Qwen3.5 CPT 模板会规范化文本，并屏蔽空 `<think>` 前缀。

## 训练入口

| 脚本                                | 资源       | 核心配置                                           |
| ----------------------------------- | ---------- | -------------------------------------------------- |
| `run-qwen3.5-35B-A3B-cpt-8xgpu.sh`  | 单机 8 卡  | GBS128，TP4 / PP1 / CP1 / EP8                      |
| `run-qwen3.5-35B-A3B-cpt-16xgpu.sh` | 双机 16 卡 | GBS2048，TP2 / PP2（22/18 层）/ CP1 / EP4          |
| `run-qwen3.5-35B-A3B-cpt.sh`        | 8～128 卡  | 按节点数扩展 GBS，TP2 / PP2（22/18 层）/ CP1 / EP4 |

这些配置均使用 dynamic batching、`seq-length=4096`、`max-tokens-per-gpu=32768`、BF16、sequence parallel、Flash Attention、full recompute 和 shared expert overlap。

已验证镜像：

```text
relax:qs-20260916-89830e62
```

镜像中的 `/root/Megatron-LM` 已包含 expert grad norm 修复。

## 单机 8 卡

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
