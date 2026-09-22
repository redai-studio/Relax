# MTP Training

This guide covers Multi-Token Prediction (MTP) training in Relax: joint and separated training during SFT, plus joint training during RL post-training.

Complete [Installation](./installation.md) first. See [SFT Training](./sft-training.md) for the common SFT data format and flags. The RL section also assumes that a baseline GRPO job works (see [Quick Start](./quick-start.md)).

## MTP SFT: Joint and Two-Stage Training

The example below uses Qwen3.5-35B-A3B and `pokemon-gpt4o-captions`. Training reads the English and Chinese parquet files, each with 833 rows, for 1,666 rows in total. Samples store messages in `conversations` and images in `images`.

| Mode | Script | What it trains |
| --- | --- | --- |
| Joint training | [`run-qwen3.5-35B-A3B-pokemon-mtp-8xgpu.sh`](../../../scripts/training/sft/run-qwen3.5-35B-A3B-pokemon-mtp-8xgpu.sh) | Initializes from the original HF checkpoint and updates the main model and one MTP layer together. |
| Stage 1 | [`run-qwen3.5-35B-A3B-pokemon-8xgpu.sh`](../../../scripts/training/sft/run-qwen3.5-35B-A3B-pokemon-8xgpu.sh) | Runs standard SFT only and exports an HF checkpoint for Stage 2. |
| Stage 2 | [`run-qwen3.5-35B-A3B-pokemon-mtp-only-8xgpu.sh`](../../../scripts/training/sft/run-qwen3.5-35B-A3B-pokemon-mtp-only-8xgpu.sh) | Loads Stage 1, freezes non-MTP parameters, and updates only MTP. |

### Joint Training

Joint training uses these key flags:

```text
--enable-mtp-training
--mtp-num-layers 1
--mtp-loss-scaling-factor 0.2
--mtp-detach-paths none
```

This mode needs only one training run, but the main-model target and MTP draft change at the same time. The current
example script uses `--mtp-detach-paths none`, so gradients from the MTP auxiliary loss update both the main model
and MTP parameters. The flag independently controls the `embedding`, `backbone`, and `lm-head` gradient paths. Its
default detaches all three paths, leaving the main model to receive gradients only from the standard SFT loss.

### Separated Two-Stage Training

Stage 1 passes no MTP training flags and optimizes only the standard SFT loss. Its `--save-hf` export writes the trained main model to `${SAVE_DIR}/sft/qwen3.5-35B-A3B-sft-pokemon-gpu8-hf` while retaining the untrained MTP weights from the original checkpoint.

Stage 2 initializes from that HF directory by default and uses these flags to freeze the main model and train only MTP:

```text
--mtp-only-training
--mtp-num-layers 1
--mtp-loss-scaling-factor 0.2
```

When both stages share the same externally supplied `SAVE_DIR`, run them in sequence:

```bash
SAVE_DIR=/path/to/checkpoints \
  bash scripts/entrypoint/ray-job.sh \
  scripts/training/sft/run-qwen3.5-35B-A3B-pokemon-8xgpu.sh

SAVE_DIR=/path/to/checkpoints \
  bash scripts/entrypoint/ray-job.sh \
  scripts/training/sft/run-qwen3.5-35B-A3B-pokemon-mtp-only-8xgpu.sh
```

Set `INIT_HF_DIR` explicitly for Stage 2 when the stages use different checkpoint roots.

### Pokémon Measurements

The benchmark uses 100 fixed samples from the Chinese parquet with `seed=20260811` and `temperature=0`. Metrics are computed from per-request speculative counters returned by the native `/generate` endpoint:

- Micro accept rate = `sum(accepted) / sum(drafted)`
- Micro accept length = `sum(completion_tokens) / sum(spec_verify_ct)`

| Checkpoint | MTP gradient mode | Micro accept rate | Micro accept length |
| --- | --- | ---: | ---: |
| Original base (joint-training control) | — | 65.7761% | 2.3184 |
| Joint training (detach all main-model paths) | `--mtp-detach-paths embedding backbone lm-head` | 64.8330% | 2.2999 |
| Joint training (fully joint gradients) | `--mtp-detach-paths none` | 70.8709% | 2.4250 |
| Two-stage Stage 1 | MTP not trained | 43.5425% | 1.8726 |
| Two-stage Stage 2 | `--mtp-only-training` | 68.5460% | 2.3725 |

Under the same 128-token generation setting, detach joint training is 0.9431 percentage points below the base, with accept length lower by 0.0185. No-detach joint training is 5.0948 percentage points above the base, with accept length higher by 0.1066; compared with detach joint training, it gains 6.0379 percentage points and 0.1251 accept length. Stage 1 is an unaligned intermediate state: the target has received standard SFT while MTP still contains the original base weights. After Stage 2 trains only MTP, accept rate improves by 25.0035 percentage points and accept length by 0.4999. Stage 1 and Stage 2 have identical actual output lengths: 11,846 tokens in total, mean 118.46, median 117, range 42–152, with all 100 responses ending in `stop`.

The base acceptance initially looked substantially higher mainly because output length biased the micro aggregate. At a 512-token generation limit, 96/100 base responses reached the limit and contained long repetitive tails. Micro accept rate aggregates all draft tokens, so long sequences receive more weight and those tails inflate the base result. The base, detach joint-training, and no-detach joint-training rows in the table all use a 128-token generation limit.

::: warning Comparison scope
The base, detach joint-training, and no-detach joint-training rows are directly comparable. The two-stage measurements used a different generation limit and should not be compared with the first three rows; Stage 1 and Stage 2 used the same samples and generation settings and are directly comparable with each other.
:::

## Joint MTP Training During RL

### When to Enable MTP RL Training

Enable MTP joint training when you want the MTP head's weights to stay calibrated with the evolving policy during RL, so that the same checkpoint can later serve speculative decoding (EAGLE/NEXTN in SGLang or vLLM) without a separate distillation pass.

If you only need the base policy and have no downstream speculative-decoding plan, leave MTP off — the auxiliary loss adds a small amount of compute and memory per step.

### Prerequisites

- The HF checkpoint must already contain MTP weights, i.e. `config.json` has `num_nextn_predict_layers >= 1`. Today this covers **Qwen3.5**, **Qwen3-next**, **MiMo-7B-RL**, **DeepSeek-V3 / V3.1**, and **GLM-4.7-MoE**.
- The Megatron backend (this is the only training backend in Relax). MTP requires the Megatron MTP patch shipped at [`docker/patch/megatron/20260506-85bced0ae.patch`](../../../docker/patch/megatron/20260506-85bced0ae.patch); the official Relax image applies it automatically.
- Combined 1F1B pipeline schedule must be off — it is incompatible with MTP and is asserted out at [`relax/backends/megatron/model.py:493`](../../../relax/backends/megatron/model.py).

### Flags

| Flag | Default | Meaning |
| --- | --- | --- |
| `--enable-mtp-training` | off | Switches on MTP forward injection and MTP auxiliary loss. |
| `--mtp-num-layers` | required when enabled | Number of MTP layers in the model. Must match the HF checkpoint. |
| `--mtp-loss-scaling-factor` | `0.1` (recommended for RL) | Scalar multiplied onto the MTP loss before it is added to the main loss. |
| `--mtp-detach-paths` | `embedding backbone lm-head` | MTP loss paths to detach. Select any combination of `embedding`, `backbone`, and `lm-head`, or use `none` by itself for fully joint gradients. |

Relax registers these options through its Megatron argument provider. When MTP training is enabled,
`--mtp-num-layers` must also be set.

### Scripts

| Script | Model | Resources | Notes |
| --- | --- | --- | --- |
| [`run-qwen35-9B-mtp-8xgpu.sh`](../../../scripts/training/text/run-qwen35-9B-mtp-8xgpu.sh) | `Qwen3.5-9B` | 8 GPU colocate | Dense model, cheap smoke test before scaling up. |
| [`run-qwen35-35B-A3B-mtp-16xgpu.sh`](../../../scripts/training/text/run-qwen35-35B-A3B-mtp-16xgpu.sh) | `Qwen3.5-35B-A3B` | 16 GPU (2-node) colocate | Production target. Mirrors the baseline GRPO script plus `MTP_ARGS`. |

Both scripts read the same env-var overrides:

```bash
MTP_NUM_LAYERS=1 MTP_LOSS_SCALING_FACTOR=0.1 \
  bash scripts/training/text/run-qwen35-9B-mtp-8xgpu.sh
```

### Tuning the Scaling Factor

The default `0.1` is conservative for RL because the main GRPO loss carries gradient noise from advantage estimation. If you observe:

- `train/mtp_loss` plateaus very early while `train/loss` looks normal → try `0.2`–`0.3`.
- `train/loss` becomes unstable after enabling MTP → drop to `0.05`.
- MTP grads dominate (look at `train/grad_norm` relative to the no-MTP baseline) → drop scaling.

For comparison, the SFT MTP script uses `0.2` because SFT gradients are cleaner. The slime defaults are also `0.2`; lowering for RL is a Relax-specific recommendation.

### What to Watch in Logs

A healthy run emits, per train step:

```text
train/loss              # main GRPO loss, comparable to baseline
train/grad_norm         # comparable to baseline (≤2×)
train/mtp_loss          # bounded, gradually decreasing
```

If `train/mtp_loss` never appears, the MTP block was not built — usually a checkpoint mismatch (the HF ckpt has no MTP weights). Confirm with:

```bash
python -c "from transformers import AutoConfig; \
  print(AutoConfig.from_pretrained('/path/to/ckpt').num_nextn_predict_layers)"
```


## See also

- [SFT Training](./sft-training.md) — common SFT data, checkpoint, and configuration details.
- [Update Weights Pipeline](./update-weights-pipeline.md) — how Megatron→SGLang weight sync handles MTP layers.
