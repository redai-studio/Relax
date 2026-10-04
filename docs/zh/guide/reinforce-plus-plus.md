# REINFORCE++ 训练

REINFORCE++ 是一种不需要 Critic 的策略梯度算法。Relax 根据奖励计算优势值（advantage），再使用 PPO 风格的裁剪目标更新策略。已有 GRPO 训练配置时，可以沿用 Actor 和 Rollout 的部署方式，替换算法参数。

## 概述

Relax 提供两个变体，主要区别是如何计算优势值，以及如何约束策略偏离参考模型：

- **REINFORCE++**（`reinforce_plus_plus`）：将最终奖励与逐 token 的 k1 KL 惩罚合并，再累积得到各 token 的回报值（return）。这样，参考策略惩罚会参与优势值计算。
- **REINFORCE++-baseline**（`reinforce_plus_plus_baseline`）：为同一提示词生成多个回答，用每个回答的奖励减去组内平均奖励，得到原始优势值。参考策略惩罚通过独立的 k2 KL 损失加入，不影响优势值。

baseline 的组均值包含当前回答；它既不像 RLOO 那样排除当前回答，也不像默认的 GRPO 奖励处理那样除以组内标准差。不同算法的对比见[算法参考](../examples/algorithms.md)。

两种变体都会在整个训练批次的有效回答 token 上归一化优势值，包括跨数据并行 rank 的统计。提示词、padding 和被 mask 的 token 不参与统计，较长回答因 token 更多而占更大权重。损失则先在每个回答内求均值，再跨回答求均值。

## 快速开始

[单 GPU 示例](../../../examples/algorithms/run-qwen3-0.6B-1xgpu-reinforce-plus-plus.sh)使用 Qwen3-0.6B 和 `math` 奖励函数。先完成[安装](./installation.md)，准备带 CUDA GPU 的训练环境，以及以下输入：

- Hugging Face 格式的 Qwen3-0.6B checkpoint 和 tokenizer；示例也用它作为参考模型。
- 包含 `question`、`answer` 列的训练 Parquet 文件。`answer` 只保存最终答案，例如 `42`，不要直接使用 GSM8K 的完整解题过程。
- 可写的输出目录。脚本从其 `actor` 子目录加载和保存 checkpoint，因此新任务应使用新目录；在容器内运行时，将输出挂载到持久存储。

::: warning 使用独占训练环境
`ray-job.sh` 会清理旧的 Relax/SGLang worker 和训练作业，关闭 Ray Serve 应用，并移除已有 placement group。不要在有其他任务的共享主机或 Ray 集群上运行。也不要直接启动示例脚本：它默认走本地启动流程，会停止 Ray 并清理 Python 进程。
:::

下面假设训练容器只暴露一张 GPU，且尚未启动 Ray。从仓库根目录运行，将路径替换为本地路径，并确保 Ray worker 能访问这些文件：

```bash
export MODEL_PATH=/path/to/Qwen3-0.6B
export PROMPT_DATA=/path/to/gsm8k/main/train_clean.parquet
export OUTPUT_DIR=/path/to/runs/reinforce-plus-plus

ray start --head --num-gpus=1 --dashboard-host=127.0.0.1 --dashboard-port=8265

ADVANTAGE_ESTIMATOR=reinforce_plus_plus \
bash scripts/entrypoint/ray-job.sh \
  examples/algorithms/run-qwen3-0.6B-1xgpu-reinforce-plus-plus.sh
```

已有专用 Ray 环境时，跳过 `ray start`；Jobs API 不在 `http://127.0.0.1:8265` 时，设置 `RAY_ADDRESS`。`--num-gpus=1` 只声明 Ray 的资源容量，不控制 GPU 可见性。

要运行 baseline，换用另一个输出目录，并修改 `ADVANTAGE_ESTIMATOR`：

```bash
OUTPUT_DIR=/path/to/runs/reinforce-plus-plus-baseline \
ADVANTAGE_ESTIMATOR=reinforce_plus_plus_baseline \
bash scripts/entrypoint/ray-job.sh \
  examples/algorithms/run-qwen3-0.6B-1xgpu-reinforce-plus-plus.sh
```

示例使用同步 colocate 模式，Actor 与 Rollout 分时共用一张 GPU，上下文并行度为 1。

## 配置

### 调整示例脚本

通过环境变量调整示例，例如在启动命令前设置 `NUM_ROLLOUT=100 LR=5e-7`。下表列出常用设置，默认值来自示例脚本，而不是 Relax 参数解析器。

| 环境变量 | 示例默认值 | 说明 |
|---|---|---|
| `NUM_ROLLOUT` | `50` | rollout 迭代次数 |
| `ROLLOUT_BATCH_SIZE` | `4` | 每次 rollout 的提示词数 |
| `N_SAMPLES_PER_PROMPT` | `8` | 每个提示词的回答数；baseline 必须大于 1 |
| `GLOBAL_BATCH_SIZE` | `32` | 每个训练批次的回答数 |
| `ROLLOUT_MAX_RESPONSE_LEN` | `1024` | 训练和评测的回答 token 上限 |
| `LR` | `1e-6` | 学习率 |
| `KL_COEF` | `0.01` | REINFORCE++ 的奖励 KL 系数；baseline 不使用此变量 |
| `KL_LOSS_COEF` | `0.01` | baseline 的独立 KL 损失系数；REINFORCE++ 不使用此变量 |
| `MAX_TOKENS_PER_GPU` / `LOG_PROBS_MAX_TOKENS_PER_GPU` | `4096` / `4096` | 每张 GPU 的动态训练 / log probability 前向 token 预算 |
| `SGLANG_MEM_FRACTION_STATIC` | `0.45` | SGLang 的静态显存比例 |

示例默认每次采样 `4 × 8 = 32` 个回答，用一个训练批次完成更新。调整批量时，保持 `GLOBAL_BATCH_SIZE = ROLLOUT_BATCH_SIZE × N_SAMPLES_PER_PROMPT` 即可沿用这一设置。

设置 `EVAL_DATA` 后才启用评测，数据格式与训练数据相同。评测默认每 10 次 rollout 进行一次，每个提示词生成 4 个回答，不在训练前评测；可通过 `EVAL_INTERVAL` 和 `N_SAMPLES_PER_EVAL_PROMPT` 调整。checkpoint 默认每 50 次 rollout 保存一次，可通过 `SAVE_INTERVAL` 调整。

### 在自己的脚本中启用

保留模型、数据和资源配置，将算法参数替换为下面两组之一。当前仅支持 Megatron 同步 colocate 训练，需要 `--colocate --context-parallel-size 1` 和 `--ref-load <参考模型路径>`，不支持 `--fully-async`、`--hybrid` 或 `--calculate-per-token-loss`。

**REINFORCE++：**

```bash
ALGORITHM_ARGS=(
  --advantage-estimator reinforce_plus_plus
  --normalize-advantages
  --gamma 1.0
  --kl-coef 0.01
  --kl-loss-type k1
  --kl-loss-coef 0
)
```

这里使用 k1 KL 惩罚，即 `-kl_coef × (log_prob_old - log_prob_ref)`。最终奖励加到最后一个有效回答 token，再从后向前累积；`gamma=1.0` 表示不对后续奖励做折扣。`--kl-coef` 必须为正，且不能同时启用 `--use-kl-loss`。

**REINFORCE++-baseline：**

```bash
ALGORITHM_ARGS=(
  --advantage-estimator reinforce_plus_plus_baseline
  --normalize-advantages
  --n-samples-per-prompt 8
  --kl-coef 0
  --use-kl-loss
  --kl-loss-type k2
  --kl-loss-coef 0.01
)
```

这组参数将组内奖励差复制到回答 token，并用独立的 k2 惩罚 `0.5 × (log_prob_current - log_prob_ref)²` 约束策略。`--kl-loss-coef` 必须为正。每个提示词须保留完整的回答组；缺少回答或组标识不一致时，会报奖励组不完整。

baseline 依赖内置的组均值处理，因此不能使用 `--disable-rewards-normalization`、`--custom-reward-post-process-path` 或 `--agentic-custom-advantage-path`，也不支持 `--use-unbiased-kl`。

Relax 默认选择 GRPO，关闭优势值归一化，两个 KL 系数均为零。只修改 `--advantage-estimator` 不会自动补齐上述设置，迁移已有脚本时应一起替换算法参数。完整选项见[配置说明](./configuration.md)。

## 监控与常见问题

示例默认将 TensorBoard event 写入 `OUTPUT_DIR/actor/tensorboard_log`，提交日志保存在 `OUTPUT_DIR/logs`。如需更改 event 目录，在 `ray start` 前执行 `export TENSORBOARD_DIR=/path/to/tensorboard`。`ray-job.sh` 不转发该变量，因此 Ray 已启动后，只在提交终端设置它不会将覆盖值传入作业。

评估训练效果时，结合评测奖励、`rollout/response_len/mean` 和 `rollout/truncated_ratio`，而不是只看损失。回答经常达到长度上限时，可在显存允许的情况下增大 `ROLLOUT_MAX_RESPONSE_LEN`。

### 奖励或优势值一直为零

先检查标签和生成回答。`math` 奖励函数从回答中的 `\boxed{...}` 提取最终答案，无法提取时返回零；标签应是最终答案，而非完整解题过程。

baseline 中，同组奖励都相同时，减去组均值后原始优势值为零，奖励无法区分该提示词下的回答。归一化后的值还受整批 token 统计影响。可先查看 `rollout/reinforce_pp_advantage_raw_std` 和 `rollout/reinforce_pp_zero_variance`，而不是修改归一化参数；原始优势值全部相同时，归一化结果为零，整个批次没有有效 token 时则会报错。

### 一直等待调度或显存不足

用 `ray status` 检查是否有可用 GPU，以及足够运行服务和奖励 worker 的 CPU 资源。示例默认使用 4 个奖励 worker、最多 16 个并发请求，可用 `REWARD_NUM_WORKERS` 和 `REWARD_MAX_CONCURRENCY` 调整。

训练或 log probability 前向阶段 OOM 时，降低对应的 token 预算；推理侧显存不足时，调整 `SGLANG_MEM_FRACTION_STATIC`。具体排查方法见 [OOM 排查](./oom-troubleshooting.md)。

### 如何查看 KL 惩罚

`train/ppo_kl` 比较旧策略与当前策略，不代表策略与参考模型的差异。baseline 的参考模型惩罚记录在 `train/kl_loss`，乘以 `--kl-loss-coef` 后加入总损失。REINFORCE++ 的惩罚已包含在回报值中，没有独立的 `train/kl_loss`；`rollout/returns` 与 `rollout/raw_reward` 可辅助排查，但二者差值不是参考策略 KL 的直接估计。

## 相关文档

- [数据集设计](./dataset-design.md)：准备提示词和标签。
- [自定义训练](./customize-training.md)：调整训练脚本。
- [Metrics 服务](./metrics-service-detailed.md)：配置指标输出。
