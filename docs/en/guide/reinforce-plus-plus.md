# REINFORCE++ Training

REINFORCE++ is a policy gradient algorithm that does not require a Critic. In Relax, it computes advantages from rewards and updates the policy with a PPO-style clipped objective. If you already have a GRPO training setup, you can keep the same Actor and Rollout deployments and replace the algorithm arguments.

## Overview

Relax provides two variants. They differ in how they compute advantages and how they penalize deviation from the reference policy:

- **REINFORCE++** (`reinforce_plus_plus`) combines the final reward with per-token k1 KL penalties, then accumulates them into token returns. The reference-policy penalty therefore contributes to the advantages.
- **REINFORCE++-baseline** (`reinforce_plus_plus_baseline`) generates multiple responses to each prompt and subtracts the group's mean reward from each response reward to obtain raw advantages. It applies reference-policy regularization through a separate k2 KL loss, leaving advantages independent of that penalty.

The baseline mean includes the current response, unlike RLOO's leave-one-out mean. It also does not divide by the group standard deviation as GRPO does by default. See the [Algorithm Reference](../examples/algorithms.md) for other algorithms.

Both variants normalize advantages over valid response tokens across the entire training batch, including all data-parallel ranks. Prompt tokens, padding, and masked tokens are excluded. Longer responses contribute more tokens to these statistics, while the loss is averaged within each response and then across responses.

## Quick Start

The [one-GPU recipe](../../../examples/algorithms/run-qwen3-0.6B-1xgpu-reinforce-plus-plus.sh) uses Qwen3-0.6B with the `math` reward. Complete [Installation](./installation.md) in a CUDA GPU training environment and prepare:

- A Hugging Face Qwen3-0.6B checkpoint with its tokenizer. The recipe also uses it as the reference policy.
- A training Parquet file with `question` and `answer` columns. Store only the final answer in `answer`, such as `42`, rather than the full GSM8K solution rationale.
- A writable output directory. Checkpoints are loaded from and saved to its `actor` subdirectory, so use a new directory for a new run. Mount the output on persistent storage when using a container.

::: warning Use a dedicated training environment
`ray-job.sh` cleans up previous Relax/SGLang workers and training jobs, shuts down Ray Serve applications, and removes existing placement groups. Do not run it on a shared host or Ray cluster with other workloads. Avoid launching the recipe directly: its default local startup also stops Ray and cleans up Python processes.
:::

The following assumes that your training container exposes one GPU and Ray is not yet running. Run from the repository root, replacing the paths with local paths accessible to the Ray workers:

```bash
export MODEL_PATH=/path/to/Qwen3-0.6B
export PROMPT_DATA=/path/to/gsm8k/main/train_clean.parquet
export OUTPUT_DIR=/path/to/runs/reinforce-plus-plus

ray start --head --num-gpus=1 --dashboard-host=127.0.0.1 --dashboard-port=8265

ADVANTAGE_ESTIMATOR=reinforce_plus_plus \
bash scripts/entrypoint/ray-job.sh \
  examples/algorithms/run-qwen3-0.6B-1xgpu-reinforce-plus-plus.sh
```

Skip `ray start` if a dedicated Ray runtime is already running. Set `RAY_ADDRESS` if its Jobs API is not at `http://127.0.0.1:8265`. `--num-gpus=1` declares Ray resource capacity; it does not control GPU visibility.

To run the baseline, change `ADVANTAGE_ESTIMATOR` and use a separate output directory:

```bash
OUTPUT_DIR=/path/to/runs/reinforce-plus-plus-baseline \
ADVANTAGE_ESTIMATOR=reinforce_plus_plus_baseline \
bash scripts/entrypoint/ray-job.sh \
  examples/algorithms/run-qwen3-0.6B-1xgpu-reinforce-plus-plus.sh
```

The recipe uses synchronous colocate training: Actor and Rollout take turns on one GPU, with context parallelism set to 1.

## Configuration

### Recipe Settings

Override settings with environment variables, for example by adding `NUM_ROLLOUT=100 LR=5e-7` before the launch command. The table lists common settings with defaults from the recipe, not the Relax argument parser.

| Environment variable | Recipe default | Description |
|---|---|---|
| `NUM_ROLLOUT` | `50` | Number of rollout iterations |
| `ROLLOUT_BATCH_SIZE` | `4` | Prompts per rollout |
| `N_SAMPLES_PER_PROMPT` | `8` | Responses per prompt; must exceed 1 for the baseline |
| `GLOBAL_BATCH_SIZE` | `32` | Responses per training batch |
| `ROLLOUT_MAX_RESPONSE_LEN` | `1024` | Response-token limit for training and evaluation |
| `LR` | `1e-6` | Learning rate |
| `KL_COEF` | `0.01` | REINFORCE++ reward-side KL coefficient; unused by the baseline |
| `KL_LOSS_COEF` | `0.01` | Baseline's separate KL-loss coefficient; unused by REINFORCE++ |
| `MAX_TOKENS_PER_GPU` / `LOG_PROBS_MAX_TOKENS_PER_GPU` | `4096` / `4096` | Dynamic training / log-probability forward-pass token budget per GPU |
| `SGLANG_MEM_FRACTION_STATIC` | `0.45` | SGLang's static GPU-memory fraction |

By default, each rollout produces `4 × 8 = 32` responses for one training batch. Keep `GLOBAL_BATCH_SIZE = ROLLOUT_BATCH_SIZE × N_SAMPLES_PER_PROMPT` when changing batch sizes to retain this setup.

Evaluation is enabled only when you set `EVAL_DATA`, using the same data format as training. It runs every 10 rollout iterations with 4 responses per prompt and skips evaluation before training. Override these settings with `EVAL_INTERVAL` and `N_SAMPLES_PER_EVAL_PROMPT`. Checkpoints are saved every 50 rollout iterations, configurable through `SAVE_INTERVAL`.

### Using Your Own Script

Keep your model, data, and resource configuration, and replace the algorithm arguments with one of the sets below. Both variants currently require Megatron synchronous colocate training, `--colocate --context-parallel-size 1`, and `--ref-load <reference-checkpoint>`. They do not support `--fully-async`, `--hybrid`, or `--calculate-per-token-loss`.

**REINFORCE++:**

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

This uses the k1 penalty `-kl_coef × (log_prob_old - log_prob_ref)`. The final reward is added to the last valid response token, and rewards are accumulated backwards; `gamma=1.0` means no discounting. `--kl-coef` must be positive, and `--use-kl-loss` must remain off.

**REINFORCE++-baseline:**

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

This broadcasts each group-centered reward to the response tokens and applies the separate k2 penalty `0.5 × (log_prob_current - log_prob_ref)²`. `--kl-loss-coef` must be positive. Keep complete response groups for each prompt; missing responses or inconsistent group identifiers produce an incomplete-group error.

The baseline relies on built-in group-mean reward processing, so it cannot use `--disable-rewards-normalization`, `--custom-reward-post-process-path`, or `--agentic-custom-advantage-path`. It also does not support `--use-unbiased-kl`.

Relax defaults to GRPO, with advantage normalization off and both KL coefficients set to zero. Changing only `--advantage-estimator` does not supply the required settings, so replace the algorithm arguments together when adapting an existing script. See [Configuration](./configuration.md) for the full option list.

## Monitoring and Common Problems

By default, TensorBoard events use `OUTPUT_DIR/actor/tensorboard_log`, and submission logs use `OUTPUT_DIR/logs`. To change the event directory, run `export TENSORBOARD_DIR=/path/to/tensorboard` before `ray start`. The `ray-job.sh` helper does not forward this variable, so setting it only in the submission shell will not pass the override to a job on an already-running Ray runtime.

Assess training with evaluation rewards, `rollout/response_len/mean`, and `rollout/truncated_ratio`, rather than loss alone. If responses often reach the length limit, increase `ROLLOUT_MAX_RESPONSE_LEN` if memory permits.

### Rewards or Advantages Stay at Zero

Check labels and generated responses first. The `math` reward extracts the final answer from `\boxed{...}` and returns zero if extraction fails. Labels should contain final answers, not full solution rationales.

For the baseline, equal rewards within a prompt group produce zero raw advantages after subtracting the group mean: those rewards do not distinguish between that prompt's responses. Normalized values also depend on the full batch's token statistics. Check `rollout/reinforce_pp_advantage_raw_std` and `rollout/reinforce_pp_zero_variance` before changing normalization. Constant raw advantages normalize to zero, while a batch with no valid tokens raises an error.

### Pending Actors or GPU Out-of-Memory Errors

Use `ray status` to check GPU availability and CPU capacity for services and reward workers. The recipe defaults to 4 reward workers and up to 16 concurrent requests; adjust them with `REWARD_NUM_WORKERS` and `REWARD_MAX_CONCURRENCY`.

For OOM during training or log-probability computation, reduce the corresponding token budget. For inference allocation issues, adjust `SGLANG_MEM_FRACTION_STATIC`. See [OOM Troubleshooting](./oom-troubleshooting.md) for details.

### Which KL Metric Should I Read?

`train/ppo_kl` compares the old and current policies, not the policy and reference model. For the baseline, `train/kl_loss` records the reference penalty before it is multiplied by `--kl-loss-coef` and added to the total loss. REINFORCE++ includes its penalty in returns, so it has no separate `train/kl_loss`. `rollout/returns` and `rollout/raw_reward` can help with diagnosis, but their difference is not a direct estimate of reference-policy KL.

## Related Documentation

- [Dataset Design](./dataset-design.md): prepare prompts and labels.
- [Customize Training](./customize-training.md): adapt the training script.
- [Metrics Service](./metrics-service-detailed.md): configure metric outputs.
