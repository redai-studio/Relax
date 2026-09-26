# DPO 训练

Relax 通过离线 SFT 数据链路支持 Direct Preference Optimization（DPO）。Task 31 的公开 recipe 是 [`run-qwen3-0.6B-ultrafeedback-1xgpu.sh`](../../../scripts/training/dpo/run-qwen3-0.6B-ultrafeedback-1xgpu.sh)。

V1 支持 TP=CP=PP=1 的同步纯文本 SFT 路径。偏好训练要求 `--task-type causal_lm`，不支持 `--sft-async-prepack`、MTP（含 MTP-only）、chunked logits 和 LoRA；仍可使用 SFT 的 CPU 数据预取。global batch size 和 optimizer scheduler 的增量均按 preference pair 计数，也适用于显式指定实际大小的较小批次。

## 准备偏好数据子集

从固定的数据集 revision 生成确定性的 UltraFeedback 子集：

```bash
python scripts/data/prepare_ultrafeedback_preferences.py \
  --output-dir /data/task31-ultrafeedback
```

命令会生成 train/eval JSONL、Parquet 以及 `manifest.json`。对于已发布的 Task 31 子集，训练前应将生成结果与[可复现性证据包](https://github.com/user-attachments/files/31305744/task31-pr1-dpo-evidence-public-v2.tar.gz)中的 manifest 对比。manifest 固定 source revision、选中的 prompt ID、拒绝原因计数和输出文件 SHA-256；派生数据文件本身不提交到 Git。

每行输入承载一个完整 preference pair：

```json
{
  "prompt_id": "stable-id",
  "chosen": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}],
  "rejected": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
}
```

chosen/rejected 必须共享完全相同的 prompt，并包含不同且非空的 assistant completion。

## 启动标准 DPO

下载固定版本的 Qwen checkpoint，然后设置标准入口所需的模型、数据和输出路径：

```bash
export MODEL_DIR=/models
export MODEL_REVISION=c1899de289a04d12100db370d81485cdf75e47ca # 完整的 40 位 commit SHA
export HF_CHECKPOINT="${MODEL_DIR}/Qwen3-0.6B-${MODEL_REVISION}"
export PROMPT_DATA=/data/task31-ultrafeedback/ultrafeedback_train.parquet
export SAVE_DIR=/checkpoints/task31-dpo

hf download Qwen/Qwen3-0.6B --revision "${MODEL_REVISION}" --local-dir "${HF_CHECKPOINT}"
bash scripts/training/dpo/run-qwen3-0.6B-ultrafeedback-1xgpu.sh
```

recipe 默认运行 200 个 optimizer step，每个 global batch 为 32 个 preference pair，`beta=0.1`，单分支最大 1,024 token。可以显式覆盖 `GLOBAL_BATCH_SIZE`、`NUM_ROLLOUT`、`MAX_TOKENS_PER_GPU` 和 `SAVE_INTERVAL`。

标准 DPO 会先在本地 `HF_CHECKPOINT` 目录中校验固定的 repository revision，再从该目录重建冻结 reference。checkpoint 带有 reference identity sidecar，其中保存 canonical parameter digest 和固定 probe digest；sidecar 缺失或不一致时，会在下一次 forward 前失败。

probe digest 是对冻结 reference log-probability 的逐字节 SHA-256，因此 resume 假定 GPU 型号、驱动、镜像与内核栈与原运行完全一致。在不同硬件或软件环境上 resume 会按设计触发 probe 校验失败——这表示环境不匹配，而非数据损坏。

只有明确需要 reference-free DPO 时才使用 `--dpo-reference-free`，不要同时传入标准 reference identity 参数。

## Pair-aware batching

一个 preference pair 对应一个 TransferQueue row。chosen/rejected 分支长度相加后写入 `custom_meta.total_lengths`；固定版本的 `SeqlenBalancedSampler` 分配完整 row，并保证各 data-parallel rank 的 pair 数相同。只有 rank 收到 pair row 后才展开两个分支，因此动态 micro-batch 重排不会破坏 pair identity。

## 指标

DPO 在 `train/dpo/` 命名空间下记录以下训练指标：

- `loss`、`logps_chosen` 和 `logps_rejected`；
- 标准模式下的 `ref_logps_chosen` 和 `ref_logps_rejected`；
- `reward_chosen`、`reward_rejected` 和 `reward_margin`；
- `strict_accuracy`、`tie_rate` 和 `tie_aware_accuracy`。

如需声明分布式一致性，应在相同镜像、模型/数据 revision、超参数和 batch 语义下分别运行 DP=1、DP=2，并保留原始日志与 reference digest。
