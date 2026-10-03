# DPO Training

Direct Preference Optimization (DPO) trains a model to prefer one answer over another. This guide uses Qwen3-0.6B and UltraFeedback on one GPU.

Complete [Installation](./installation.md) first. Run the commands below from the Relax repository root.

## Prepare the data

Create the example dataset:

```bash
python scripts/data/prepare_ultrafeedback_preferences.py \
  --output-dir /data/ultrafeedback
```

The script selects 4,096 training pairs and 512 evaluation pairs from a fixed UltraFeedback version. It writes JSONL and Parquet files for each split.

For your own data, put one pair on each row. `chosen` is the preferred answer. `rejected` is the other answer. Use the same conversation history in both message lists. End each list with a different assistant answer. Give each row a unique `prompt_id`.

```json
{
  "prompt_id": "capital-1",
  "chosen": [{"role": "user", "content": "What is the capital of France?"}, {"role": "assistant", "content": "Paris."}],
  "rejected": [{"role": "user", "content": "What is the capital of France?"}, {"role": "assistant", "content": "London."}]
}
```

## Download the model

The example uses a fixed model version. Download it to the directory that training will use:

```bash
export MODEL_DIR=/models
export MODEL_REVISION=c1899de289a04d12100db370d81485cdf75e47ca
export HF_CHECKPOINT="${MODEL_DIR}/Qwen3-0.6B-${MODEL_REVISION}"

hf download Qwen/Qwen3-0.6B \
  --revision "${MODEL_REVISION}" \
  --local-dir "${HF_CHECKPOINT}"
```

Keep the complete download directory, including `.cache/huggingface`. DPO uses its download metadata to check the reference model version.

## Start DPO training

Set the data and checkpoint paths:

```bash
export PROMPT_DATA=/data/ultrafeedback/ultrafeedback_train.parquet
export SAVE_DIR=/checkpoints/dpo
export EXP_NAME=qwen3-0.6b-ultrafeedback-dpo-gpu1
```

Run the [DPO training script](../../../scripts/training/dpo/run-qwen3-0.6B-ultrafeedback-1xgpu.sh):

```bash
NUM_GPUS=1 bash scripts/training/dpo/run-qwen3-0.6B-ultrafeedback-1xgpu.sh
```

The script uses a frozen copy of the downloaded model as the reference model. It saves checkpoints to `${SAVE_DIR}/${EXP_NAME}` and writes logs to `log/`.

Set these environment variables before you run the script to change its defaults:

| Variable | Default | Meaning |
| --- | --- | --- |
| `NUM_ROLLOUT` | `200` | Total optimizer steps, including completed steps when you resume. |
| `GLOBAL_BATCH_SIZE` | `32` | Answer pairs per optimizer step, across all GPUs. |
| `MAX_TOKENS_PER_GPU` | `8192` | Tokens per micro-batch, including both copies of the prompt and both answers. |
| `LR` | `5e-7` | Learning rate. |
| `SAVE_INTERVAL` | `50` | Steps between checkpoint saves. |

The script sets `--dpo-beta 0.1`. It limits each prompt and answer to 1,024 tokens in total, with at most 512 tokens in the answer. It truncates longer inputs. To change these settings, edit the script.

One pair counts as one training sample. Relax keeps its two answers together when it divides work across GPUs and micro-batches.

## Resume DPO training

Run the same script with the same paths and training settings. It loads the checkpoint from `${SAVE_DIR}/${EXP_NAME}` and continues from the saved step. Use a different `EXP_NAME` to start a separate run.

Keep the full checkpoint directory, including `relax_dpo_reference.json` inside each saved iteration. DPO uses this file to check that the reference model has not changed.

The current resume check also compares reference model outputs byte for byte. Use the original GPU model and software environment. An environment change can cause this check to fail even when the weights are unchanged.

## Read the training metrics

DPO records these metrics under `train/dpo/`:

| Metrics | Meaning |
| --- | --- |
| `loss` | DPO training loss. |
| `logps_chosen`, `logps_rejected` | Model log-probabilities for the two answers. |
| `ref_logps_chosen`, `ref_logps_rejected` | Reference model log-probabilities. |
| `reward_chosen`, `reward_rejected`, `reward_margin` | DPO rewards and their difference. |
| `strict_accuracy`, `tie_rate`, `tie_aware_accuracy` | Preference accuracy and ties. |

## Supported configurations

- Use synchronous training with text data.
- Set TP, CP, and PP to 1. Data parallelism is supported.
- Keep `--task-type causal_lm` with the preference objective.
- Use ordinary CPU data prefetch if needed. Asynchronous prepacking, MTP, chunked logits, and LoRA are not supported.

For reference-free DPO, add `--dpo-reference-free` to the training command. Remove `--dpo-reference-repository` and `--dpo-reference-revision` from that command. Reference-free training does not record reference log-probabilities.

## Next steps

- [SFT training](./sft-training.md)
- [Training configuration](./customize-training.md)
