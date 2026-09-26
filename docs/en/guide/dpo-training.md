# DPO Training

Relax supports Direct Preference Optimization (DPO) through the offline SFT data path. The public Task 31 recipe is [`run-qwen3-0.6B-ultrafeedback-1xgpu.sh`](../../../scripts/training/dpo/run-qwen3-0.6B-ultrafeedback-1xgpu.sh).

V1 supports synchronous text-only SFT with TP=CP=PP=1. Preference objectives require `--task-type causal_lm` and reject `--sft-async-prepack`, MTP (including MTP-only training), chunked logits, and LoRA. Regular SFT CPU dataset prefetch remains available. Global batch size and optimizer scheduler increments count preference pairs, including a smaller explicitly sized batch.

## Prepare the preference subset

Generate the deterministic UltraFeedback subset from its pinned dataset revision:

```bash
python scripts/data/prepare_ultrafeedback_preferences.py \
  --output-dir /data/task31-ultrafeedback
```

The command creates train/eval JSONL and Parquet files plus `manifest.json`. For the published Task 31 subset, compare the generated manifest with the [reproducibility evidence bundle](https://github.com/user-attachments/files/31305744/task31-pr1-dpo-evidence-public-v2.tar.gz) before training. The manifest fixes the source revision, selected prompt IDs, rejection counts, and output SHA-256 values. Derived dataset files are intentionally not stored in Git.

Each input row contains one complete preference pair:

```json
{
  "prompt_id": "stable-id",
  "chosen": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}],
  "rejected": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
}
```

Chosen and rejected branches must have an identical prompt and different, non-empty assistant completions.

## Launch standard DPO

Download the pinned Qwen checkpoint, then set the model, data, and output locations expected by the standard entrypoint:

```bash
export MODEL_DIR=/models
export MODEL_REVISION=c1899de289a04d12100db370d81485cdf75e47ca # full 40-character commit SHA
export HF_CHECKPOINT="${MODEL_DIR}/Qwen3-0.6B-${MODEL_REVISION}"
export PROMPT_DATA=/data/task31-ultrafeedback/ultrafeedback_train.parquet
export SAVE_DIR=/checkpoints/task31-dpo

hf download Qwen/Qwen3-0.6B --revision "${MODEL_REVISION}" --local-dir "${HF_CHECKPOINT}"
bash scripts/training/dpo/run-qwen3-0.6B-ultrafeedback-1xgpu.sh
```

The recipe defaults to 200 optimizer steps, 32 preference pairs per global batch, `beta=0.1`, and a 1,024-token branch limit. `GLOBAL_BATCH_SIZE`, `NUM_ROLLOUT`, `MAX_TOKENS_PER_GPU`, and `SAVE_INTERVAL` can be overridden explicitly.

Standard DPO verifies the pinned repository revision against the local `HF_CHECKPOINT` directory, then reconstructs the frozen reference from that directory. Checkpoints include a reference-identity sidecar containing canonical parameter and fixed-probe digests. A missing or mismatched sidecar fails before the next forward pass.

The probe digest is a byte-exact SHA-256 over frozen-reference log-probabilities, so resume assumes the same GPU model, driver, image, and kernel stack as the original run. Resuming on different hardware or software fails the probe check by design — treat it as an environment mismatch, not data corruption.

Use `--dpo-reference-free` only when reference-free DPO is intended; do not combine it with the standard reference identity arguments.

## Pair-aware batching

One preference pair is one TransferQueue row. Its chosen and rejected branch lengths are combined into `custom_meta.total_lengths`; the pinned `SeqlenBalancedSampler` assigns complete rows and keeps equal pair counts across data-parallel ranks. Branches are expanded only after a rank receives its rows, so dynamic micro-batch reordering cannot split pair identity.

## Metrics

DPO emits the following training metrics under the `train/dpo/` namespace:

- `loss`, `logps_chosen`, and `logps_rejected`;
- `ref_logps_chosen` and `ref_logps_rejected` in standard mode;
- `reward_chosen`, `reward_rejected`, and `reward_margin`;
- `strict_accuracy`, `tie_rate`, and `tie_aware_accuracy`.

For distributed parity claims, run DP=1 and DP=2 with the same image, model/data revisions, hyperparameters, and batch semantics, and retain the raw logs and reference digests.
