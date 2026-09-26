# PPO Training

Relax provides PPO (Proximal Policy Optimization) as a first-class synchronous actor-critic training path for the Megatron backend.

## Overview

Select PPO with `--advantage-estimator ppo`. Unlike GRPO, GSPO, CISPO, and SAPO, PPO trains a separate Critic model. The Critic predicts token-level values, the Advantages service computes Generalized Advantage Estimation (GAE), and the Actor optimizes the clipped policy objective.

The currently supported PPO topology is synchronous colocate mode. The included starter recipe runs on 8 GPUs: Actor, Critic, and SGLang time-share the same placement group, while the CPU-only Advantages service exchanges `values`, `advantages`, and `returns` through TransferQueue.

::: warning Current scope
Fully-async PPO is not currently supported. Use the synchronous colocate recipe documented on this page.
:::

## Architecture

```text
┌─────────────┐   rollout data   ┌───────────────┐
│   Rollout   │ ───────────────> │ TransferQueue │
└──────▲──────┘                  └───────┬───────┘
       │                                 │
       │                         ┌───────▼───────┐
       │                         │    Critic     │
       │                         │ values + loss │
       │                         └───────┬───────┘
       │                                 │ values
       │                         ┌───────▼───────┐
       │                         │  Advantages   │
       │                         │  GAE outputs  │
       │                         └───────┬───────┘
       │                    advantages + returns
       │                                 │
       │                         ┌───────▼───────┐
       └──── weight update ───── │     Actor     │
                                 │ PPO-Clip loss │
                                 └───────────────┘
```

| Component | Responsibility | TransferQueue fields |
|---|---|---|
| **Rollout** | Generates responses, rewards, and rollout log-probabilities | `tokens`, `rewards`, `rollout_log_probs`, masks and lengths |
| **Critic** | Predicts values and trains with clipped value loss | Produces `values`; consumes locally computed `returns` for value training |
| **Advantages** | Computes GAE after Critic values are ready | Consumes `values`; produces `advantages` and `returns` |
| **Actor** | Trains the policy with PPO-Clip and publishes updated weights | Consumes `advantages`, `returns`, and old log-probabilities |

## Quick Start

The starter script expects these directories:

- `${MODEL_DIR}/Qwen3.5-9B`
- `${DATA_DIR}/dapo-math-17k/dapo-math-17k.jsonl`
- `${DATA_DIR}/aime-2024/aime-2024.jsonl`

Launch the 8-GPU colocate recipe:

```bash
MODEL_DIR=/path/to/models \
DATA_DIR=/path/to/data \
EXP_DIR=/path/to/experiments \
bash scripts/training/text/run-qwen35-9B-8xgpu-ppo.sh
```

The recipe enables PPO with:

```bash
PPO_ARGS=(
   --advantage-estimator ppo
   --gamma 1.0
   --lambd 0.95
   --eps-clip 0.2
   --eps-clip-high 0.2
   --entropy-coef 0.0
   --use-rollout-logprobs
   --kl-coef 0.0
   --value-clip 0.5
   --critic-lr 1e-5
   --num-critic-only-steps 5
   --critic-lr-warmup-iters 5
)
```

Its required resource topology is:

```bash
--resource '{"actor": [1, 8], "critic": [1, 8], "rollout": [1, 8], "advantages": [1, 0]}' \
--colocate
```

When the `actor`, `critic`, and `rollout` resource shapes match in colocate mode, they share one placement group and take turns occupying GPU memory. The example lowers `--sglang-mem-fraction-static` and enables optimizer CPU offload to leave enough memory for this schedule.

## Configuration

### PPO Parameters

| Parameter | Default | Description |
|---|---|---|
| `--advantage-estimator ppo` | `grpo` | Select the PPO service graph and enable the Critic path |
| `--gamma` | `1.0` | GAE discount factor |
| `--lambd` | `1.0` | GAE lambda |
| `--eps-clip` | `0.2` | Lower PPO-Clip margin |
| `--eps-clip-high` | `None` | Upper PPO-Clip margin; when unset, it follows `--eps-clip` |
| `--value-clip` | `0.2` | Critic value clipping range |
| `--entropy-coef` | `0.0` | Entropy bonus coefficient |
| `--num-critic-only-steps` | `0` | Number of initial rollout steps that train only the Critic |
| `--critic-lr` | `None` | Critic learning rate; when unset, it follows `--lr` |
| `--critic-lr-warmup-iters` | `0` | Linear warmup iterations for the Critic |
| `--critic-load` | `None` | Critic checkpoint to load; when unset, it follows `--load` |
| `--critic-save` | `None` | Critic checkpoint output directory |
| `--use-rollout-logprobs` | off | Use SGLang rollout log-probabilities as the old policy values |

### Resource Requirements

Every supported PPO configuration must include `critic` and `advantages` entries in `--resource`.

- Use `--use-rollout-logprobs`; synchronous PPO does not deploy a separate `actor_fwd` service.
- Keep `--use-kl-loss` disabled and `--kl-coef 0.0`. The synchronous PPO service graph has no Reference producer for `ref_log_probs`.

::: warning Colocate KL configuration
If synchronous PPO receives `--use-kl-loss` or `--kl-coef != 0`, argument processing logs a warning, disables `--use-kl-loss`, and resets `--kl-coef` to `0.0`. Remove these options from copied GRPO scripts rather than relying on automatic normalization.
:::

### Checkpoint Resume

Actor and Critic must resume from the same iteration. Relax reads `latest_checkpointed_iteration.txt` from `--load` and `--critic-load` and fails before service launch when the iterations differ.

Use one of these consistent states:

1. Both Actor and Critic cold-start from `--hf-checkpoint`.
2. Both load Megatron checkpoints from the same iteration.

Set `--critic-save` when Critic checkpoints must be persisted.

## Best Practices

1. Use the provided synchronous colocate topology; fully-async PPO is not currently supported.
2. Warm up the Critic with `--num-critic-only-steps` when its value head starts from the policy checkpoint.
3. Keep separate Actor and Critic learning rates; the example uses `1e-6` and `1e-5`, respectively.
4. Monitor `value_loss`, `value_clipfrac`, `pg_loss`, `pg_clipfrac`, and `ppo_kl` together.
5. Budget memory for Actor, Critic, optimizer states, and SGLang. Matching resource shapes save GPUs through time-sharing but do not reduce host-memory requirements.

## Troubleshooting

### Missing `critic` or `advantages` resource

PPO validates both roles before deployment. Add both entries to `--resource`; Advantages can use `[1, 0]` because it is CPU-only.

### Missing old-policy log-probabilities

Use `--use-rollout-logprobs` with the supported synchronous colocate topology.

### Actor and Critic resume mismatch

Point `--load` and `--critic-load` to checkpoints with the same tracker iteration, or remove both Megatron checkpoints and cold-start both models from the HF checkpoint.

### GPU out of memory during service switching

Check that Actor, Critic, and Rollout have matching colocate resource shapes, lower `--sglang-mem-fraction-static`, and retain optimizer CPU offload. The PPO path forces train-model offload so Actor and Critic can release GPU memory between phases.

## Next Steps

- [Algorithm Reference](../examples/algorithms.md) — Compare PPO with GRPO, GSPO, CISPO, and SAPO
- [Configuration](./configuration.md) — Review all training arguments
- [Architecture](./architecture.md) — Understand the Ray Serve component model
