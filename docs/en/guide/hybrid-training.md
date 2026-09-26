# Hybrid Training Mode

## Overview

**Hybrid mode** is a third execution mode in Relax that sits between [Colocate (Sync)](./architecture.md) and [Fully Async](./fully-async-training.md). It combines:

- the **streaming data pipeline** of Fully Async (TransferQueue + `max-staleness` for off-policy tolerance), with
- the **in-process weight sharing** of Colocate (TensorBackuper + `_switch_model`, so ref / actor_fwd / advantages all run on the actor's own GPUs).

Concretely, Actor and Rollout still run on **separate GPU placement groups** (like Fully Async), but the actor no longer ships weights to standalone ActorFwd / Reference / Advantages services. Instead it cycles a single set of weights between `actor`, `ref`, `old_actor`, and `teacher` tags via a CPU/GPU `TensorBackuper`, computing every forward pass locally and pushing weights to rollout through the sync `UpdateWeightFromTensor` path.

### Mode Comparison

| Dimension           | Colocate (Sync)                          | Fully Async                                                  | Hybrid                                                                              |
| ------------------- | ---------------------------------------- | ------------------------------------------------------------ | ----------------------------------------------------------------------------------- |
| **GPU layout**      | Actor and Rollout time-share same GPUs   | Actor / Rollout / ActorFwd / Reference each have own GPUs    | Actor and Rollout on separate GPUs; ref / actor_fwd / adv share actor's GPUs        |
| **Data pipeline**   | TransferQueue, batch-synchronous         | TransferQueue + StreamingDataLoader, fully streaming         | TransferQueue + sub-batch streaming (`num-iters-per-train-update`)                  |
| **Weight sync**     | In-process tensor copy                   | NCCL broadcast via DCS (Checkpoint Engine)                   | Sync `UpdateWeightFromTensor` to rollout; TensorBackuper for ref/actor_fwd          |
| **Staleness**       | `max_staleness = 0` (strict on-policy)   | Configurable `max_staleness`                                 | Configurable `max_staleness`                                                        |
| **Roles deployed**  | `actor`, `critic`, `rollout`             | `actor`, `critic`, `rollout`, `advantages`, `reference`, `actor_fwd` | `actor`, `critic`, `rollout` (same as Colocate; ref/actor_fwd live inside actor)    |
| **DP token balance**| `--balance-data` on static/seqlen path   | Dynamic batch auto-balances DP tokens; `--balance-data` is accepted with no extra effect | Dynamic batch auto-balances DP tokens; static/seqlen path can use `--balance-data`   |

### When to Use Hybrid

Pick **Hybrid** when:

- You want the throughput benefits of dedicated rollout GPUs and pipelined data flow, but
- Your model is large enough that running independent ref / actor_fwd services would waste GPUs, or
- You want actor-local ref / actor_fwd computation while keeping the streaming dynamic-batch path, which automatically balances DP tokens.

Pick **Fully Async** when you have spare GPUs for separate ref / actor_fwd / advantages services and want true cross-step pipelining.

Pick **Colocate** when GPU count is tight and you can tolerate serial rollout → train cycles.

______________________________________________________________________

## Architecture

### Role Layout

Hybrid uses the same role set as Colocate — only `actor`, `critic` (optional), and `rollout` are deployed as Ray Serve services. The decision lives in `relax/core/registry.py`:

```python
def process_role(config):
    if config.hybrid:
        # hybrid mode: actor handles ref/actor_fwd internally
        # via _switch_model, only need actor + rollout services
        return ROLES_COLOCATE
    if config.fully_async:
        ...
```

But unlike Colocate, the actor and rollout placement groups are **disjoint**, matching Fully Async semantics. From `relax/core/controller.py`:

```python
if colocate and not self.config.hybrid:
    # Sync colocate: actor and rollout share GPUs via time-sharing (offload/onload)
    actor_rollout_pgs = create_placement_group(num_gpus=num_gpus)
else:
    # fully_async (pure or hybrid): actor and rollout use separate GPUs
    actor_rollout_pgs = None
```

### Flag Resolution

`--hybrid` is the only public switch. `relax/utils/arguments.py` resolves it into the two underlying flags downstream machinery already understands:

```python
if args.hybrid:
    args.fully_async = True
    args.colocate = True
```

Passing `--fully-async --colocate` directly is rejected; use `--hybrid` instead. This single-switch design keeps `args.hybrid` as the canonical hybrid-only branch in the registry, controller dispatch, and `train_hybrid` call site.

### Diagram

```
┌────────────────────────────────────────────────────────────────────────────┐
│                        Controller (Orchestrator)                           │
│                     relax/core/controller.py                               │
│                                                                            │
│       ┌───────────────────────────────────┐    ┌────────────────────┐      │
│       │            Actor Service          │    │  Rollout Service   │      │
│       │  (own placement group, N GPUs)    │    │ (own PG, M GPUs)   │      │
│       │                                   │    │   SGLang engines   │      │
│       │  ┌────────────────────────────┐   │    └─────────┬──────────┘      │
│       │  │ TensorBackuper             │   │              ▲                 │
│       │  │  tags: actor / ref /       │   │              │                 │
│       │  │        old_actor / teacher │   │              │                 │
│       │  │  _switch_model(tag) swaps  │   │              │                 │
│       │  │  weights on the same GPUs  │   │              │                 │
│       │  └────────────────────────────┘   │              │                 │
│       │  train_hybrid():                  │              │                 │
│       │    ├─ ref forward   (switch:ref)  │              │                 │
│       │    ├─ actor forward (switch:actor)│                                │
│       │    ├─ advantages    (in-process)  │              │                 │
│       │    └─ train         (switch:actor)│                                │
│       └──────────────┬────────────────────┘              │                 │
│                      │ UpdateWeightFromTensor (sync) ───┘                  │
└──────────────────────┼─────────────────────────────────────────────────────┘
                       ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                      TransferQueue (Data Plane)                           │
│  Rollout writes train_N partition incrementally ──► Actor consumes        │
│  in sub-batches via get_meta(batch_size, batch_index) with max-staleness  │
└───────────────────────────────────────────────────────────────────────────┘
```

______________________________________________________________________

## The `train_hybrid` Loop

`relax/backends/megatron/actor.py:708` implements the hybrid training step in three phases:

1. **Collect sub-batches and compute forward log-probs (small memory footprint)**

   The global batch is split into `num_iters_per_train_update` sub-batches. For each sub-batch the actor:

   - pulls data from TransferQueue (`_get_data_from_transfer_queue("train", rollout_id, fields, batch_size, batch_index)`)
   - runs `_switch_model("ref")` (if ref weights are backed up) and computes ref log-probs
   - runs `_switch_model("teacher")` (if OPD teacher weights are backed up) and computes teacher log-probs
   - runs `_switch_model("old_actor" or "actor")` and computes current actor log-probs
   - appends the enriched sub-batch to an in-memory list

2. **Merge sub-batches and compute advantages globally**

   All sub-batch dicts are concatenated into one `rollout_data`, then `compute_advantages_and_returns(self.args, rollout_data)` runs once over the merged batch. This is the **key correctness reason** for the two-phase design — advantage normalization must see the full DP-group batch, not per-sub-batch slices.

3. **Train on the merged batch and push weights**

   A single `train(...)` call runs the optimizer step on the merged batch. Afterwards the actor backs up the new weights to the `actor` tag and (on the ref-update interval) refreshes the `ref` tag, then calls `self.update_weights()` to push the updated weights to rollout via `UpdateWeightFromTensor`.

The sub-batched forward keeps peak activation memory bounded — matching Fully Async behavior — while the merged training step preserves Colocate-style global statistics.

______________________________________________________________________

## Configuration

### Required Flags

| Flag                            | Purpose                                                                              |
| ------------------------------- | ------------------------------------------------------------------------------------ |
| `--hybrid`                      | Enable hybrid mode (resolves to `fully_async=True, colocate=True` internally)        |
| `--resource '{...}'`            | Declare `actor` and `rollout` placement groups separately, e.g. `{"actor":[1,4],"rollout":[1,4]}` |
| `--num-iters-per-train-update`  | Number of sub-batches per global batch (larger → smaller peak memory, more TQ polls) |
| `--max-staleness`               | Off-policy budget (0 = strict on-policy, >0 allows staleness)                        |

### Optional but Common

| Flag                          | Notes                                                                                                    |
| ----------------------------- | -------------------------------------------------------------------------------------------------------- |
| `--balance-data`              | Accepted in hybrid. With `--use-dynamic-batch-size`, DP token balancing is automatic through the streaming sampler; on static/seqlen paths, it enables DP token balancing. |
| `--num-data-storage-units`    | Number of TransferQueue storage actors.                                                                  |
| `--use-streaming-dataset`     | Stream prompts from disk instead of loading into memory.                                                 |
| `--ref-update-interval`       | Periodically refresh the cached ref weights from the latest actor weights.                               |

### Default Overrides

When `--hybrid` is set, `relax/utils/arguments.py` defaults the following (unless the user passes them explicitly):

- `offload_train = False` and `offload_rollout = False` — actor and rollout are on separate GPUs, so no offload needed
- `compute_advantages_and_returns = True` — actor must compute advantages internally
- `fully_async = True`, `colocate = True` — derived from `--hybrid`

::: tip
Hybrid and pure fully async both use `StreamingTokenBudgetSampler` when `--use-dynamic-batch-size` is enabled, so DP token balancing is automatic on that path. Keep `--balance-data` for static/seqlen-balanced batches; it has no additional effect on dynamic batching.
:::

______________________________________________________________________

## Quick Start

A reference launch script for an 8-GPU multimodal hybrid run lives at
`scripts/training/multimodal/run-qwen35-9B-8xgpu-openr1mm-hybrid-async.sh`.

The hybrid invocation it builds:

```bash
ray job submit --address="http://127.0.0.1:8265" \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- python3 -m relax.entrypoints.train \
    --resource '{"actor": [1, 4], "rollout": [1, 4]}' \
    --max-staleness 2 \
    --num-data-storage-units 1 \
    --num-iters-per-train-update 8 \
    --balance-data \
    --hybrid \
    "${MODEL_ARGS[@]}" \
    "${CKPT_ARGS[@]}" \
    "${ROLLOUT_ARGS[@]}" \
    "${OPTIMIZER_ARGS[@]}" \
    "${GRPO_ARGS[@]}" \
    "${PERF_ARGS[@]}" \
    "${SGLANG_ARGS[@]}" \
    "${MISC_ARGS[@]}"
```

Key points in this configuration:

- 8 total GPUs split 4 + 4 between actor and rollout
- `max-staleness 2` — actor may consume rollout output up to 2 steps behind the freshest weights
- `num-iters-per-train-update 8` — each global batch is split into 8 sub-batches for forward passes
- `balance-data` — accepted by this script; DP token balancing is handled by the dynamic batch path
- GRPO algorithm with `--use-kl-loss` and `--use-tis` (these are algorithm flags, orthogonal to hybrid)

______________________________________________________________________

## Troubleshooting

### `train_hybrid(rollout_id=N) batch_index=K stalled for ... seconds`

This warning fires in `relax/backends/megatron/actor.py` when the actor's TransferQueue poll for the next sub-batch keeps returning empty while the partition is not marked `all_consumed`. Typical causes:

- Rollout under-filled this partition (dropped samples without refilling).
- Rollout is paused on a health-check failure or restart.
- Staleness budget exhausted: rollout cannot produce new data because it is waiting for fresh weights.

Check rollout-side logs and partition status before assuming a code bug.

### Dynamic Batch DP Token Balancing

With `--fully-async` or `--hybrid` plus `--use-dynamic-batch-size`, DP token balancing is automatic through `StreamingTokenBudgetSampler`. If token load still looks uneven, check the token budget (`--max-tokens-per-gpu`), DP size, and streaming sampler stall logs before changing execution mode.

### Rollout sees stale weights for a long time

Hybrid uses the sync `UpdateWeightFromTensor` path at the end of each `train_hybrid` call. If you see large weight-update gaps, check:

- `update_weights()` timing in actor logs
- Whether rollout health-checks are paging the actor (`_check_services_health()` is called before weight sync)

______________________________________________________________________

## Next Steps

Planned follow-ups for hybrid mode:

- **Integrate DCS for weight sync** — replace the current synchronous `UpdateWeightFromTensor` path with the Distributed Checkpoint Service so weight broadcast to rollout can overlap with the next training iteration, closing the remaining sync gap at the end of every `train_hybrid` call.
- **Split `train_actor` into `num_iters_per_train_update` iterations** — today `num_iters_per_train_update` only chunks the forward phase; the merged training step still runs once on the full global batch. Extend the actor train step to also iterate `num_iters_per_train_update` times so optimizer updates can be pipelined with TransferQueue consumption and peak training-side memory drops further.

Related docs:

- [Fully Async Training Pipeline](./fully-async-training.md) — the streaming-data engine hybrid borrows
- [Architecture](./architecture.md) — overview of Relax's service layering
- [Update Weights Pipeline](./update-weights-pipeline.md) — how `UpdateWeightFromTensor` and DCS differ
