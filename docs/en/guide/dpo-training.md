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

For your own data, put one pair on each row. `chosen` is the preferred answer. `rejected` is the other answer. Use the same conversation history in both message lists. End each list with a different assistant answer.

```json
{
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

You can also set `HF_CHECKPOINT` to a local Hugging Face-format SFT model directory containing the model weights, configuration, and tokenizer.

## Start DPO training

Set the data and checkpoint paths:

```bash
export PROMPT_DATA=/data/ultrafeedback/ultrafeedback_train.parquet
export EVAL_PROMPT_DATA=/data/ultrafeedback/ultrafeedback_eval.parquet
export SAVE_DIR=/checkpoints/dpo
export EXP_NAME=qwen3-0.6b-ultrafeedback-dpo-gpu1
```

Run the [DPO training script](../../../scripts/training/dpo/run-qwen3-0.6B-ultrafeedback-1xgpu.sh):

```bash
NUM_GPUS=1 bash scripts/training/dpo/run-qwen3-0.6B-ultrafeedback-1xgpu.sh
```

The script selects DPO with `--loss-type dpo` and sets `--ref-load "${HF_CHECKPOINT}"` to load the frozen reference. In a custom training command, omitting `--ref-load` uses `--hf-checkpoint` instead. The script saves checkpoints to `${SAVE_DIR}/${EXP_NAME}` and writes logs to `log/`.

To start DPO from a native Megatron SFT checkpoint, set `--ref-load /checkpoints/sft` in your training command and omit `--load`; Relax initializes the policy from the model weights and starts a new run. Keep `--hf-checkpoint` pointing to matching Hugging Face configuration and tokenizer files. Use `--ref-ckpt-step` to select the reference iteration, or let the checkpoint tracker select it. If you explicitly set `--load` to the SFT checkpoint, also add `--finetune`.

Set these environment variables before you run the script to change its defaults:

| Variable | Default | Meaning |
| --- | --- | --- |
| `NUM_ROLLOUT` | `200` | Total optimizer steps, including completed steps when you resume. |
| `GLOBAL_BATCH_SIZE` | `32` | Answer pairs per optimizer step, across all GPUs. |
| `MAX_TOKENS_PER_GPU` | `8192` | Token budget per micro-batch, including both copies of the prompt and both answers. |
| `LR` | `5e-7` | Learning rate. |
| `SAVE_INTERVAL` | `50` | Steps between checkpoint saves. |
| `EVAL_INTERVAL` | `200` | Steps between evaluations. |

The script sets `--dpo-beta 0.1`, `--preference-max-length 1024`, and `--preference-max-completion-length 512`. These explicit length limits truncate each prompt and answer to at most 1,024 tokens in total, with at most 512 tokens in the answer. To change these settings, edit the script.

One pair counts as one training sample. Relax keeps its two answers together when it divides work across GPUs and micro-batches. After applying the explicit length limits, the default `--sft-oversize-strategy keep` preserves a pair that exceeds the token budget and places it alone in a micro-batch. To truncate or skip such pairs, explicitly choose another [oversize strategy](./configuration.md#oversize-sample-handling).

## Resume DPO training

Run the same script with the same paths and training settings. It loads the checkpoint from `${SAVE_DIR}/${EXP_NAME}` and continues from the saved step. Use a different `EXP_NAME` to start a separate run.

For a custom training command, set `--load` to the DPO checkpoint directory and resume without `--finetune`. Keep `--ref-load` pointing to the original reference checkpoint, or keep the same `--hf-checkpoint` if `--ref-load` was omitted.

Keep the full checkpoint directory, including `relax_dpo_reference.json` inside each saved iteration. On resume, DPO compares a hash of the loaded reference weights with the saved hash and raises an error if they differ.

## Read the training metrics

DPO records these metrics under `train/dpo/`:

| Metrics | Meaning |
| --- | --- |
| `loss` | DPO training loss. |
| `logps_chosen`, `logps_rejected` | Model log-probabilities for the two answers. |
| `ref_logps_chosen`, `ref_logps_rejected` | Reference model log-probabilities. |
| `reward_chosen`, `reward_rejected`, `reward_margin` | DPO rewards and their difference. |
| `strict_accuracy`, `tie_rate`, `tie_aware_accuracy` | Preference accuracy and ties. |

## Train a reward model

A reward model gives each answer a scalar score. Training increases the score of the preferred answer relative to the other answer.

Use the model and data paths from the steps above. Set a separate checkpoint directory and experiment name:

```bash
export SAVE_DIR=/checkpoints/reward-modeling
export EXP_NAME=qwen3-0.6b-ultrafeedback-rm-gpu1

NUM_GPUS=1 bash scripts/training/reward_modeling/run-qwen3-0.6B-ultrafeedback-1xgpu.sh
```

The script selects reward model training with `--loss-type rm`. The [reward model script](../../../scripts/training/reward_modeling/run-qwen3-0.6B-ultrafeedback-1xgpu.sh) defaults to 200 optimizer steps, 32 pairs per step, and a learning rate of `1e-5`. Set `NUM_ROLLOUT`, `GLOBAL_BATCH_SIZE`, `MAX_TOKENS_PER_GPU`, `LR`, or `EVAL_INTERVAL` before launch to change these settings. The script saves every 50 steps. To change the save interval, edit `--save-interval` in the script.

To resume, run the same script with the same paths and training settings. Keep the full Megatron checkpoint from reward model training. The script restores the model, optimizer, learning rate scheduler, and random number generator state together. A PPO critic checkpoint is not compatible with this reward model.

## Evaluate during training

Both scripts read evaluation data from `EVAL_PROMPT_DATA`. Set `EVAL_INTERVAL` before launch to choose how often evaluation runs. For example, `EVAL_INTERVAL=50` evaluates after steps 50, 100, and so on.

Metrics under `eval/dpo_*` or `eval/rm_*` include loss, chosen/rejected scores, their difference, accuracy, tie rate, and the number of pairs.

Relax includes the final partial evaluation batch. With data parallelism, every batch must divide evenly across the GPUs. For example, 10 pairs with a global batch size of 8 form batches of 8 and 2 pairs; both work with DP=2.

## Supported configurations

- Use synchronous training with text data.
- Set TP, CP, and PP to 1. Data parallelism is supported.
- Keep `--task-type causal_lm` with the preference objective.
- Use ordinary CPU data prefetch if needed. Asynchronous prepacking, MTP, chunked logits, and LoRA are not supported.

For reference-free DPO, add `--dpo-reference-free` to the training command. This mode does not load a frozen reference or record reference log-probabilities.

## Next steps

- [Training configuration](./customize-training.md)
