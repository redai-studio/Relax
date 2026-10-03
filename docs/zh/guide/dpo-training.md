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

## 支持的配置

- 使用同步训练和纯文本数据。
- 将 TP、CP、PP 设为 1。支持数据并行。
- 使用偏好训练目标时，保留 `--task-type causal_lm`。
- 可以使用普通的 CPU 数据预取。目前不支持异步预打包、MTP、chunked logits 和 LoRA。

使用无参考模型的 DPO 时，在训练命令中添加 `--dpo-reference-free`，并移除 `--dpo-reference-repository` 和 `--dpo-reference-revision`。这种模式不记录参考模型的对数概率。

## 下一步

- [SFT 训练](./sft-training.md)
- [训练配置](./customize-training.md)
