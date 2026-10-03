# DPO 训练

Direct Preference Optimization（DPO）让模型学会优先选择一对回答中更好的那个。本指南使用 Qwen3-0.6B 和 UltraFeedback，在单张 GPU 上训练。

先完成[安装](./installation.md)。以下命令都在 Relax 仓库根目录运行。

## 准备数据

生成示例数据集：

```bash
python scripts/data/prepare_ultrafeedback_preferences.py \
  --output-dir /data/ultrafeedback
```

脚本从固定版本的 UltraFeedback 中选取 4,096 对训练样本和 512 对评测样本，分别生成 JSONL 和 Parquet 文件。

使用自己的数据时，每行放一对回答。`chosen` 是更好的回答，`rejected` 是另一条回答。两个消息列表的对话历史必须相同，最后一条助手回答不同。每行的 `prompt_id` 必须唯一。

```json
{
  "prompt_id": "capital-1",
  "chosen": [{"role": "user", "content": "What is the capital of France?"}, {"role": "assistant", "content": "Paris."}],
  "rejected": [{"role": "user", "content": "What is the capital of France?"}, {"role": "assistant", "content": "London."}]
}
```

## 下载模型

示例使用固定的模型版本。将模型下载到训练时使用的目录：

```bash
export MODEL_DIR=/models
export MODEL_REVISION=c1899de289a04d12100db370d81485cdf75e47ca
export HF_CHECKPOINT="${MODEL_DIR}/Qwen3-0.6B-${MODEL_REVISION}"

hf download Qwen/Qwen3-0.6B \
  --revision "${MODEL_REVISION}" \
  --local-dir "${HF_CHECKPOINT}"
```

保留完整的下载目录，包括 `.cache/huggingface`。DPO 会通过其中的下载记录检查参考模型的版本。

## 启动 DPO 训练

设置数据和检查点路径：

```bash
export PROMPT_DATA=/data/ultrafeedback/ultrafeedback_train.parquet
export EVAL_PROMPT_DATA=/data/ultrafeedback/ultrafeedback_eval.parquet
export SAVE_DIR=/checkpoints/dpo
export EXP_NAME=qwen3-0.6b-ultrafeedback-dpo-gpu1
```

运行 [DPO 训练脚本](../../../scripts/training/dpo/run-qwen3-0.6B-ultrafeedback-1xgpu.sh)：

```bash
NUM_GPUS=1 bash scripts/training/dpo/run-qwen3-0.6B-ultrafeedback-1xgpu.sh
```

脚本使用下载模型的冻结副本作为参考模型。检查点保存在 `${SAVE_DIR}/${EXP_NAME}`，日志保存在 `log/`。

启动前可以设置以下环境变量来调整默认配置：

| 变量 | 默认值 | 含义 |
| --- | --- | --- |
| `NUM_ROLLOUT` | `200` | 总优化步数，续训时包含已完成的步数。 |
| `GLOBAL_BATCH_SIZE` | `32` | 所有 GPU 每个优化步共处理的样本对数。 |
| `MAX_TOKENS_PER_GPU` | `8192` | 每个微批次的 token 上限，包含两份提示词及各自的回答。 |
| `LR` | `5e-7` | 学习率。 |
| `SAVE_INTERVAL` | `50` | 每隔多少步保存检查点。 |
| `EVAL_INTERVAL` | `200` | 每隔多少步执行评测。 |

脚本设置了 `--dpo-beta 0.1`。提示词和单条回答加起来最多保留 1,024 个 token，其中回答最多保留 512 个 token。超出部分会被截断。要调整这些设置，请修改训练脚本。

一对回答算一个训练样本。Relax 在 GPU 之间分配数据、组合微批次时，会把同一对回答放在一起。

## 恢复 DPO 训练

使用相同的路径和训练配置重新运行脚本。脚本会从 `${SAVE_DIR}/${EXP_NAME}` 加载检查点，并从已保存的步数继续训练。要另开一次训练，请更换 `EXP_NAME`。

保留完整的检查点目录，包括每个已保存迭代目录中的 `relax_dpo_reference.json`。DPO 通过这个文件检查参考模型是否发生变化。

当前恢复检查还会逐字节比较参考模型的输出。请使用原来的 GPU 型号和软件环境。即使权重没有变化，更换环境也可能导致这项检查失败。

## 查看训练指标

DPO 在 `train/dpo/` 下记录以下指标：

| 指标 | 含义 |
| --- | --- |
| `loss` | DPO 训练损失。 |
| `logps_chosen`、`logps_rejected` | 模型对两条回答给出的对数概率。 |
| `ref_logps_chosen`、`ref_logps_rejected` | 参考模型对两条回答给出的对数概率。 |
| `reward_chosen`、`reward_rejected`、`reward_margin` | DPO 奖励及两者的差值。 |
| `strict_accuracy`、`tie_rate`、`tie_aware_accuracy` | 偏好准确率和平局比例。 |

## 训练奖励模型

奖励模型为每条回答输出一个分数。训练会提高更好回答相对于另一条回答的分数。

沿用前面设置的模型和数据路径。为奖励模型设置单独的检查点目录和实验名称：

```bash
export SAVE_DIR=/checkpoints/reward-modeling
export EXP_NAME=qwen3-0.6b-ultrafeedback-rm-gpu1

NUM_GPUS=1 bash scripts/training/reward_modeling/run-qwen3-0.6B-ultrafeedback-1xgpu.sh
```

[奖励模型脚本](../../../scripts/training/reward_modeling/run-qwen3-0.6B-ultrafeedback-1xgpu.sh)默认训练 200 步，每步处理 32 对样本，学习率为 `1e-5`。启动前可以设置 `NUM_ROLLOUT`、`GLOBAL_BATCH_SIZE`、`MAX_TOKENS_PER_GPU`、`LR` 或 `EVAL_INTERVAL` 来调整配置。脚本每 50 步保存一次；要调整保存间隔，请修改脚本中的 `--save-interval`。

恢复训练时，使用相同路径和训练配置重新运行脚本。保留奖励模型训练保存的完整 Megatron 检查点。脚本会同时恢复模型、优化器、学习率调度器和随机数生成器的状态。PPO critic 的检查点与这个奖励模型不兼容。

## 在训练期间评测

两个脚本都从 `EVAL_PROMPT_DATA` 读取评测数据。启动前设置 `EVAL_INTERVAL`，指定评测间隔。例如，`EVAL_INTERVAL=50` 会在完成第 50、100 步时评测，后续以此类推。

`eval/dpo_*` 或 `eval/rm_*` 下的指标包括损失、两条回答的分数及差值、准确率、平局比例和样本对数。

Relax 会保留最后一个不足整批的评测批次。使用数据并行时，每批样本必须能平均分给各张 GPU。例如，10 对样本按每批 8 对划分为 8 对和 2 对，两批都能用于 DP=2。

## 支持的配置

- 使用同步训练和纯文本数据。
- 将 TP、CP、PP 设为 1。支持数据并行。
- 使用偏好训练目标时，保留 `--task-type causal_lm`。
- 可以使用普通的 CPU 数据预取。目前不支持异步预打包、MTP、chunked logits 和 LoRA。

使用无参考模型的 DPO 时，在训练命令中添加 `--dpo-reference-free`，并移除 `--dpo-reference-repository` 和 `--dpo-reference-revision`。这种模式不记录参考模型的对数概率。

## 下一步

- [SFT 训练](./sft-training.md)
- [训练配置](./customize-training.md)
