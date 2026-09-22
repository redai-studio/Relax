# Fully Async Training Pipeline

## Overview

The **Fully Async training pipeline** is a high-throughput RLHF/RL training mode designed to maximize GPU utilization. Unlike the Colocate (synchronous) mode, Fully Async deploys **training (Actor)**, **inference (Rollout)**, **forward computation (ActorFwd / Reference)**, and **advantage calculation (Advantages)** on separate GPU clusters. Services exchange data through TransferQueue and synchronize weights asynchronously via the Distributed Checkpoint Service (DCS).

### Design Comparison

| Dimension           | Colocate (Synchronous)                                                                    | Fully Async                                                                                       |
| ------------------- | ----------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| **GPU Sharing**     | Actor and Rollout share the same GPUs                                                     | Actor, Rollout, ActorFwd/Reference each have dedicated GPUs                                       |
| **Execution Model** | Serial: Rollout completes → switch to Train → weight update                               | Fully parallel: Rollout, Train, forward computation run simultaneously                            |
| **Weight Sync**     | In-process tensor copy (colocated)                                                        | Cross-node NCCL broadcast via DCS (Checkpoint Engine)                                             |
| **Data Flow**       | Also via TransferQueue, but synchronous: Rollout writes the full batch before Actor reads | Via TransferQueue + StreamingDataLoader with async streaming (production and consumption overlap) |
| **Staleness**       | `max_staleness=0` (strict On-Policy)                                                      | Configurable `max_staleness` (allows some Off-Policy)                                             |
| **Roles**           | `actor`, `critic`, `rollout`                                                              | `actor`, `critic`, `rollout`, `advantages`, `reference`, `actor_fwd`                              |

::: tip
Both modes use TransferQueue as the data transport layer. In Colocate mode, Rollout and Actor time-share the same GPUs — Rollout writes a full batch to TransferQueue, then yields GPUs for Actor to train. In Fully Async mode, services run on independent GPUs in parallel, enabling concurrent data production and consumption.
:::

### Key Advantages

1. **Eliminate GPU idle time** — Rollout and Training run simultaneously; Rollout engines continue generating data during training
2. **Flexible resource allocation** — Training and inference can use different numbers of GPUs, adapting to heterogeneous hardware
3. **Controllable On/Off-Policy degree** — The `max_staleness` parameter precisely controls data freshness
4. **Pipelined weight updates** — DCS enables weight distribution to overlap with training computation

______________________________________________________________________

## Architecture

### System Diagram

```
┌───────────────────────────────────────────────────────────────────────────┐
│                        Controller (Orchestrator)                          │
│                     relax/core/controller.py                              │
│                                                                           │
│    ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────────┐  ┌─────────┐  │
│    │ Rollout  │  │  Actor   │  │ ActorFwd │  │ Reference  │  │  Adv    │  │
│    │ Service  │  │ Service  │  │ Service  │  │  Service   │  │ Service │  │
│    └──┬───────┘  └──┬───────┘  └──┬───────┘  └──┬─────────┘  └──┬──────┘  │
└───────┼─────────────┼─────────────┼─────────────┼────────────────┼────────┘
        │             │             │             │                │
        ▼             ▼             ▼             ▼                ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                      TransferQueue (Data Plane)                           │
│                                                                           │
│  ┌────────────────┐       ┌──────────────────────────────────┐            │
│  │ TQ Controller  │◄──────┤  SimpleStorageUnit × N           │            │
│  │ (Metadata Mgr) │       │  (Partitioned Data Storage)      │            │
│  └────────────────┘       └──────────────────────────────────┘            │
│                                    ▲                                      │
│                                    │                                      │
│                ┌───────────────────┼────────────────────┐                 │
│                │ StreamingDataset / StreamingDataLoader │                 │
│                │ (relax/utils/data/stream_dataloader.py)│                 │
│                └────────────────────────────────────────┘                 │
└───────────────────────────────────────────────────────────────────────────┘
        │             │             │             │
        ▼             ▼             ▼             ▼
┌───────────────────────────────────────────────────────────────────────────┐
│              Distributed Checkpoint Service (DCS)                         │
│                                                                           │
│  ┌──────────────┐     ┌──────────────────────────────────┐                │
│  │  Coordinator │◄───┤  CheckpointEngineClient × N      │                 │
│  │  (HTTP REST) │    │  (Per-rank weight send/recv)     │                 │
│  └──────────────┘     └──────────────────────────────────┘                │
│                                                                           │
│  ┌───────────────────────────────────────────────┐                        │
│  │  DeviceDirectBackend (NCCL/GLOO)              │                        │
│  │  - Actor → Rollout: weight broadcast to SGLang│                        │
│  │  - Actor → ActorFwd/Ref: PP-aware broadcast   │                        │
│  └───────────────────────────────────────────────┘                        │
└───────────────────────────────────────────────────────────────────────────┘
```

### Service Roles

In Fully Async mode, the system deploys 6 roles (defined by the `ROLES` StrEnum in `relax/core/registry.py`):

```python
class ROLES(StrEnum):
    actor: str = "actor"           # Policy model training
    critic: str = "critic"         # Value model training (optional)
    rollout: str = "rollout"       # SGLang inference engine, generates samples
    advantages: str = "advantages" # Advantage and return computation
    reference: str = "reference"   # Reference model forward (KL divergence)
    actor_fwd: str = "actor_fwd"   # Current policy forward (log prob)
```

Role selection logic (`relax/core/registry.py`):

```python
def process_role(config):
    if config.fully_async:
        return ROLES           # All 6 roles
    else:
        return ROLES_COLOCATE  # Only actor, critic, rollout
```

______________________________________________________________________

## Data Flow: StreamingDataLoader on TransferQueue

### TransferQueue in Both Modes

Both Colocate and Fully Async modes use TransferQueue for data transfer. The key difference is the **timing relationship** between production and consumption:

```
Colocate mode (serial):
  Rollout fully writes partition train_N ── all ready ──► Actor reads train_N at once
  (Same GPUs time-shared; Rollout offloads then Actor wakes up to train)
  (ref log prob, advantages computed inside Actor's train_actor() serially)

Fully Async mode (streaming parallel):
  Rollout writes partition train_N incrementally ──► Actor consumes via StreamingDataLoader
  Rollout can start train_N+1 simultaneously    ──► ActorFwd/Reference/Advantages consume train_N in parallel
  (Different GPU clusters run fully in parallel; ref log prob, adv computed independently and written back to TQ)
```

**Partition mechanism**:

- **Partition ID format**: `train_{rollout_id}`, e.g. `train_0`, `train_1`, `train_2`
- **Producer (Rollout)**: writes data to `train_{rollout_id}` after completing a rollout
- **Consumers (Actor/ActorFwd/Reference/Advantages)**: read from the corresponding partition, tracked by `task_name`
- **Partition cleanup**: Actor calls `async_clear_partition()` after training completes

**Storage capacity and max_staleness**:

```python
# relax/core/controller.py
total_storage_size = (
    self.config.rollout_batch_size
    * (self.config.max_staleness + 1)
    * self.config.n_samples_per_prompt
)
```

TransferQueue must be able to buffer `max_staleness + 1` rollout batches simultaneously. For example, with `max_staleness=2`, `rollout_batch_size=8`, `n_samples_per_prompt=8`, this requires `8 × 3 × 8 = 192` sample slots.

**Task names** track consumption progress for different consumers:

| Consumer   | task_name                                           | Data Fields Consumed                                                    |
| ---------- | --------------------------------------------------- | ----------------------------------------------------------------------- |
| Actor      | `actor_train` (StreamDataLoader) / `train` (legacy) | tokens, loss_masks, log_probs, ref_log_probs, advantages, returns, etc. |
| ActorFwd   | `actor_log_probs`                                   | tokens, total_lengths, response_lengths, loss_masks, rollout_log_probs  |
| Reference  | `ref_log_probs`                                     | tokens, total_lengths, response_lengths, loss_masks, rollout_log_probs  |
| Advantages | `compute_advantages_and_returns`                    | rollout_log_probs, log_probs, ref_log_probs, rewards, etc.              |

### StreamingDataLoader and StreamingDataset

In Fully Async mode, Actor uses `StreamingDataLoader` for **streaming data consumption**. Unlike Colocate mode where Actor waits for Rollout to fully generate a batch before reading, StreamingDataLoader can consume data as it is being incrementally written to TransferQueue. This is the core mechanism enabling "training and inference in parallel".

#### StreamingDataset

```python
# TransferQueue (installed from https://github.com/redai-studio/TransferQueue)
class StreamingDataset(IterableDataset):
    """Streaming dataset that dynamically fetches data from TransferQueue"""

    def __init__(self, config, batch_size, micro_batch_size, data_fields,
                 partition_id, task_name, dp_rank, fetch_batch_fn, process_batch_fn):
        self.buffer = []       # Cache for fetched batches
        self.batch_index = 0   # Current consumption position

    def __iter__(self):
        while not consumed:
            if self.batch_index <= len(self.buffer) - 1:
                # Read from cache (supports multi-pass training)
                yield from self.process_batch_fn(...)
            else:
                # Fetch new data from TransferQueue
                batch_data, batch_meta = self.fetch_batch_fn(...)
                if batch_data is not None:
                    self.buffer.append((batch_data, batch_meta))
```

**Key features**:

- **On-demand fetching**: fetches one `global_batch_size / num_iters_per_train_update` batch at a time
- **Buffer reuse**: `buffer` supports iterating over the same batch multiple times (e.g. multi-epoch training)
- **Partition switching**: `step(partition_id)` clears the buffer and switches to a new rollout data partition

#### Fetch Function (fetch_batch_fn)

Fully Async mode uses a customized `get_data_from_transfer_queue()` function (`relax/utils/data/stream_dataloader.py`):

```python
# broadcast_pp is the inverse of fully_async
fetch_batch_fn = partial(get_data_from_transfer_queue,
                         broadcast_pp=not getattr(args, "fully_async", False))
```

**Broadcast strategy differences**:

| Mode        | `broadcast_pp` | Data Fetch Node                            | Broadcast Scope                                     |
| ----------- | -------------- | ------------------------------------------ | --------------------------------------------------- |
| Colocate    | `True`         | `tp_rank==0 && pp_rank==0`                 | TP group + PP group                                 |
| Fully Async | `False`        | `tp_rank==0` (each PP stage independently) | TP group only (each PP stage fetches independently) |

- **Colocate mode**: Rollout has already written the full batch to TransferQueue. Actor starts on the same GPUs, PP rank 0 fetches data from TQ and broadcasts to other PP stages. All data is available at once for training.
- **Fully Async mode**: Each PP stage is on a separate rank and fetches data from TransferQueue independently, avoiding cross-PP-stage communication overhead. Since data may still be written incrementally, StreamingDataLoader automatically retries when data is not yet ready.

#### create_stream_dataloader

```python
# relax/utils/data/stream_dataloader.py
def create_stream_dataloader(args, rollout_id, task_name, data_fields, dp_rank):
    dataset = StreamingDataset(
        config=args.tq_config,
        batch_size=args.micro_batch_size * args.n_samples_per_prompt,
        micro_batch_size=args.micro_batch_size,
        data_fields=data_fields,
        partition_id=f"train_{rollout_id}",
        task_name=task_name,
        dp_rank=dp_rank,
        fetch_batch_fn=fetch_batch_fn,
        process_batch_fn=split_dict,
    )
    dataloader = StreamingDataLoader(dataset)

    # Compute training steps per rollout
    num_steps_per_rollout = (args.rollout_batch_size * args.n_samples_per_prompt
                            // args.global_batch_size)
    num_microbatches = [
        args.global_batch_size // dp_world_size // args.micro_batch_size
        for _ in range(num_steps_per_rollout)
    ]
    return [dataloader for _ in range(vpp_size)], num_microbatches
```

______________________________________________________________________

## Async Weight Sync: Distributed Checkpoint Service (DCS)

### DCS Role in Fully Async

After Actor completes training, weights must be distributed to:

1. **Rollout (SGLang engines)** — update inference engine weights
2. **ActorFwd** — update the forward model for current policy log prob computation
3. **Reference** — update the reference model (per `ref_update_interval`)

### DCS Architecture

```
                          ┌──────────────────────┐
                          │   DCS Coordinator    │
                          │   (Ray Serve HTTP)   │
                          │                      │
                          │ - Node Registration  │
                          │ - Topology Discovery │
                          │ - Weight Meta Buffer │
                          │ - Group Rank Assign  │
                          └──────────┬───────────┘
                                     │
              ┌──────────────────────┼──────────────────────┐
              │                      │                      │
    ┌─────────▼──────────┐ ┌─────────▼──────────┐ ┌─────────▼──────────┐
    │ CheckpointEngine   │ │ CheckpointEngine   │ │ CheckpointEngine   │
    │ Client (Actor)     │ │ Client (ActorFwd)  │ │ Client (Reference) │
    │                    │ │                    │ │                    │
    │ DeviceDirectBackend│ │ DeviceDirectBackend│ │ DeviceDirectBackend│
    │ (NCCL broadcast)   │ │ (NCCL recv)        │ │ (NCCL recv)        │
    └────────────────────┘ └────────────────────┘ └────────────────────┘
```

### Weight Update Flow

#### Actor → Rollout

```python
# relax/backends/megatron/actor.py
def update_weights_fully_async(self, rollout_id, rollout_only=False, actor_fwd_only=False):
    dist.barrier(group=get_gloo_group())
    if not rollout_only:
        run(self.checkpoint_engine_client.init_process_groups_for_actor_fwd_ref(rollout_id))
    run(self.checkpoint_engine_client.update_weights_for_rollout(rollout_only, actor_fwd_only))
```

Internal flow of `update_weights_for_rollout` (`DeviceDirectBackend`):

1. **Pause Rollout inference**: HTTP request to SGLang engine `/pause_generation`
2. **Flush KV Cache**: HTTP request `/flush_cache`
3. **Distribute weights**:
   - **Non-expert parameters**: `all_gather` TP shards → full parameters, then PP source rank broadcasts to Rollout (HF format) and ActorFwd/Reference (raw format)
   - **Expert parameters**: additional EP `all_gather`, then same as above
4. **Resume Rollout inference**: HTTP request `/continue_generation`

#### Actor → ActorFwd/Reference

ActorFwd and Reference receive weights via DCS PP-aware communication groups:

- Each Actor PP stage creates an independent NCCL process group (`update_actor_pp_{pp_rank}`)
- ActorFwd/Reference ranks join these groups to receive weights for the corresponding PP stage
- The receiver polls the Coordinator for weight metadata, allocates empty tensors, then receives via `dist.broadcast`

______________________________________________________________________

## max_staleness: On-Policy vs Off-Policy Control

### Concept

**Staleness** measures the version gap between the rollout data used for training and the current model weights.

- **Staleness = 0**: training data must come from the current model version
- **Staleness = N**: training data can come from current or previous N model versions

```bash
--max-staleness 2    # Allow Rollout to be up to 2 steps ahead of Actor
```

### Impact on Training

```
max_staleness = 0 (On-Policy):
  Rollout step 0 → Actor trains step 0 → Rollout step 1 → Actor trains step 1 → ...
  (Rollout must wait for Actor to consume current data before continuing)

max_staleness = 2 (Partial Off-Policy):
  Rollout: step 0 → step 1 → step 2 → [wait] → step 3 → step 4 → step 5 → [wait] → ...
  Actor:   ........................step 0 → step 1 → step 2 → ...............step 3 → ...
  (Rollout can be up to 2 steps ahead; pauses when exceeding the limit)
```

### Implementation

```python
# relax/components/rollout.py
def satisfy_staleness(partition_list, current_rollout_id, max_staleness):
    """Check if the current rollout is within the allowed staleness bound."""
    if not partition_list:
        return True
    oldest_step = min(int(p.split("_")[-1]) for p in partition_list)
    return current_rollout_id + 1 - oldest_step <= max_staleness
```

If there are `max_staleness` or more unconsumed partitions in TransferQueue, Rollout pauses and waits for Actor to catch up.

### Effect of Different max_staleness Values

| `max_staleness` | Training Semantics     | Throughput | Stability        | Typical Scenario             |
| --------------- | ---------------------- | ---------- | ---------------- | ---------------------------- |
| **0**           | Strict On-Policy       | Low        | Highest          | Debugging, small models      |
| **1**           | Near On-Policy         | Medium     | High             | Production, medium models    |
| **2-4**         | Mild Off-Policy        | High       | Medium           | Large models, slow inference |
| **>4**          | Significant Off-Policy | Highest    | Needs validation | Extreme throughput priority  |

::: tip
For production, `max_staleness=1~2` is recommended to balance throughput and training stability. Combine with `--eps-clip` and `--eps-clip-high` clipping parameters to mitigate Off-Policy instability.
:::

______________________________________________________________________

## Dynamic Batch Size (Streaming Token Budget)

### Concept

When sample lengths vary widely, a fixed micro-batch count wastes GPU. With `--use-dynamic-batch-size`, the Actor no longer splits a fixed number of micro-batches; instead it packs each micro-batch up to a **token budget** (`--max-tokens-per-gpu`), accumulating samples until close to the limit.

Under fully-async, this is **streaming**: the Actor pulls the next micro-batch from TransferQueue on demand while training, without waiting for the whole rollout batch to be ready.

```bash
--use-dynamic-batch-size       # enable dynamic batch size
--max-tokens-per-gpu 20480     # per-GPU token budget per micro-batch
```

### How It Works

```
Rollout produces samples → TransferQueue
                      │
   ┌──────────────────┴──────────────────────────┐
   │ StreamingTokenBudgetSampler (TQ controller) │
   │ · groups samples by full GRPO group         │
   │ · balances token totals across DP ranks     │
   │ · emits one mb once a token budget is filled│
   └──────────────────┬──────────────────────────┘
                      │ cached per batch_index (all DPs prepared at once)
   ┌──────────────────┴───────────────────────┐
   │ StreamingTQIterator (Actor side)         │
   │ · pulls mbs until StopIteration          │
   │ · after mb N, prefetches mb N+1's cache  │
   └──────────────────┬───────────────────────┘
                      │
     streaming schedule (streaming_schedules.py, PP=1 / PP>1)
```

- **Sampler (`StreamingTokenBudgetSampler`)**: runs on the TransferQueue controller. It accumulates ready samples by full GRPO group (`n_samples_per_prompt` each), balances token totals across DP ranks, and emits a micro-batch once a token budget is filled.
- **Per-`batch_index` cache**: the first request for a `batch_index` prepares and caches **every DP rank's slice at once**. Any later request — from any DP rank or PP stage, in any order — gets identical data, so under PP>1 all stages stay in lockstep and end at the same micro-batch count.
- **Iterator (`StreamingTQIterator`)**: runs on the Actor side, pulling micro-batches one by one until the partition is fully consumed (`StopIteration`). After pulling mb N it **prefetches** mb N+1's cache in the background so the next pull is a cache hit with less wait.

### With the Streaming Schedule

The micro-batch count is not known upfront — the Iterator drives it until `StopIteration`. So fully-async + dynamic batch uses dedicated streaming forward/backward schedules (`relax/backends/megatron/streaming_schedules.py`) for PP=1 and PP>1; all micro-batches accumulate gradients with a single gradient sync at the end.

::: tip
Dynamic batch mainly improves MFU by packing variable-length samples to fill the GPU. Note that with `max_staleness=0`, each training step still waits for Rollout to produce the first complete GRPO groups; use `max_staleness>=1` to let Rollout run ahead and remove that wait.
:::

______________________________________________________________________

## Training Loop

### Actor Training Loop

```python
# relax/components/actor.py
def _background_run(self):
    while True:
        if self._stop_event.is_set():
            break
        with self._lock:
            local_step = self.step
        if local_step >= self.config.num_rollout:
            break
        self._execute_training()
        run(self.data_system_client.async_clear_partition(f"train_{local_step}"))
        with self._lock:
            self.step += 1

def _execute_training(self):
    if self.step < self.config.num_critic_only_steps:
        return  # Skip critic-only warmup phase
    if self.config.fully_async:
        ray.get(self.actor_model.train_fully_async(self.step))
        self._maybe_save_model()
    else:
        ray.get(self.actor_model.async_train(self.step))
```

### ActorFwd and Reference Workflow

1. Fetch data in batches from TransferQueue (`_get_data_from_transfer_queue`)
2. Execute forward computation (`forward_only`) to get log probs
3. Write results back to TransferQueue (`_put_data_to_transfer_queue`)
4. After all data is consumed, call `recv_weight_fully_async()` to receive new weights

### Advantages Service

The Advantages service waits for both `ref_log_probs` and `log_probs` to be ready in TransferQueue, then computes advantages and returns and writes them back. The dependency is handled automatically by TransferQueue's `get_meta` — it blocks until the required fields are available.

______________________________________________________________________

## Data Flow Timeline

```
Time ──────────────────────────────────────────────────────────────────────►

Rollout:  ┌──generate(step=N)──┐     ┌──generate(step=N+1)────┐    ...
          │ SGLang inference   │     │  (if staleness allows) │
          │ + reward scoring   │     │                        │
          └─────────┬──────────┘     └────────────────────────┘
                    │
                    ▼ Write to TransferQueue (partition=train_N)
                    │ Fields: tokens, loss_masks, rollout_log_probs,
                    │         rewards, total_lengths, response_lengths, ...
                    │
    ┌───────────────┼──────────────────────┐
    │               │                      │
    ▼               ▼                      ▼
  ActorFwd:      Reference:            Advantages:
read train_N    read train_N        wait for log_probs
  compute        compute            and ref_log_probs
 log_probs     ref_log_probs               │
 write to TQ    write to TQ                │
    │               │                      │
    └───────────────┼──────────────────────┘
                    │ All forward results ready
                    ▼
              Advantages Service:
                read rollout_log_probs + log_probs + ref_log_probs + rewards
                compute advantages + returns
                write to TransferQueue
                    │
                    ▼
              Actor (Training):
                StreamingDataLoader streams data
                 → Megatron forward + backward + optimizer step
                 → DCS distributes new weights to Rollout, ActorFwd, Reference
                 → Clear partition train_N

    ┌───────────────┼──────────────────────┐
    │               │                      │
    ▼               ▼                      ▼
 Rollout:         ActorFwd:             Reference:
 update weights   recv_weight            recv_weight (if needed)
 resume inference (NCCL broadcast)      (NCCL broadcast)
```

______________________________________________________________________

## Configuration

### CLI Parameters

| Parameter                      | Default | Description                                                                       |
| ------------------------------ | ------- | --------------------------------------------------------------------------------- |
| `--fully-async`                | `false` | Enable the Fully Async training pipeline                                          |
| `--max-staleness`              | `0`     | Maximum allowed staleness (0=On-Policy, >0=partial Off-Policy)                    |
| `--num-data-storage-units`     | `1`     | Number of TransferQueue SimpleStorageUnit actors                                  |
| `--num-iters-per-train-update` | `1`     | Number of training iterations per global batch                                    |
| `--checkpoint-engine-backend`  | `nccl`  | DCS communication backend (`nccl` or `gloo`)                                      |
| `--polling-mode`               | `true`  | TransferQueue Controller uses polling for metadata                                |
| `--ref-update-interval`        | `None`  | Reference model update period (None=no update)                                    |
| `--use-dynamic-batch-size`     | `false` | Stream-pack micro-batches by token budget (see "Dynamic Batch Size")             |
| `--max-tokens-per-gpu`         | `None`  | Per-GPU token budget per micro-batch (required when dynamic batch is enabled)     |
| `--resource`                   | -       | JSON role resource allocation, e.g. `'{"actor": [1, 2], "rollout": [1, 4], ...}'` |

### Example Configuration

```bash
# 8 GPU Fully Async (from scripts/training/text/run-qwen3-4B-8xgpu-async.sh)
ray job submit -- python3 relax/entrypoints/train.py \
    --resource '{"actor": [1, 2], "rollout": [1, 4], "reference": [1, 1], "actor_fwd": [1, 1], "advantages": [1, 0]}' \
    --max-staleness 2 \
    --num-data-storage-units 1 \
    --num-iters-per-train-update 8 \
    --fully-async \
    --use-health-check \
    ...
```

**Resource allocation breakdown**:

- **Actor**: 1 replica × 2 GPU (TP=2 training)
- **Rollout**: 1 replica × 4 GPU (4 SGLang engines)
- **Reference**: 1 replica × 1 GPU (single-GPU forward)
- **ActorFwd**: 1 replica × 1 GPU (single-GPU forward)
- **Advantages**: 1 replica × 0 GPU (CPU-only computation)

______________________________________________________________________

## Fault Tolerance

### Restart Strategy

| Failed Role       | Strategy         | Reason                                                      |
| ----------------- | ---------------- | ----------------------------------------------------------- |
| Actor             | Global Restart   | Actor is the core training service; all others depend on it |
| Rollout           | Global Restart   | Complex engine state, difficult to recover in-place         |
| ActorFwd          | Global Restart   | Weight communication group state is hard to recover         |
| Reference         | In-place Restart | Similar to Advantages, safe to redeploy                     |
| Advantages        | In-place Restart | Stateless service, safe to redeploy                         |
| Any role ≥3 times | Global Restart   | System unstable, full re-initialization needed              |

### Fault Tolerance During Weight Update

```python
# relax/backends/megatron/actor.py — MegatronTrainRayActor.train_async()
rollout_only, actor_fwd_only = self._check_services_health()
# rollout_only=True: skip ActorFwd weight update (ActorFwd unavailable)
# actor_fwd_only=True: skip Rollout weight update (Rollout unavailable)
self.update_weights_fully_async(rollout_id, rollout_only, actor_fwd_only)
```

______________________________________________________________________

## Performance Tuning

### Key Tuning Parameters

| Parameter                      | Recommended | Impact                                                               |
| ------------------------------ | ----------- | -------------------------------------------------------------------- |
| `--max-staleness`              | 1-2         | Balance throughput vs training stability                             |
| `--num-iters-per-train-update` | 4-8         | Larger values improve data utilization but increase per-step latency |
| `--num-data-storage-units`     | 1-2         | More storage units improve parallel data access                      |

### GPU Resource Allocation Strategy

```
Total GPUs: N
├── Actor (training): ~25-30% (needs TP/PP/CP support)
├── Rollout (inference): ~50-60% (inference throughput is the bottleneck)
├── ActorFwd: ~5-10% (single GPU usually sufficient)
├── Reference: ~5-10% (single GPU usually sufficient)
└── Advantages: 0 GPU (CPU-only computation)
```

______________________________________________________________________

## Colocate vs Fully Async Comparison

```
                Colocate Mode                           Fully Async Mode
          (Same GPUs, time-shared)                 (Dedicated GPU clusters)
            ┌──────────────────┐                     ┌──────────────────────┐
  Time ──►  │   Rollout        │                     │  Rollout ──────────► │
            │ (SGLang infer)   │                     │  (continuous infer)  │
            │ write TQ train_N │                     │                      │
            ├──────────────────┤                     │  Actor  ──────────►  │
            │ offload rollout  │                     │  (StreamDataLoader   │
            │ wake up actor    │                     │   streaming + train) │
            ├──────────────────┤                     │                      │
            │   Actor Train    │                     │  ActorFwd ────────►  │
            │ (read TQ train_N)│                     │  (compute log prob)  │
            │ (incl ref/adv)   │                     │                      │
            ├──────────────────┤                     │  Reference ────────► │
            │   Weight Update  │                     │  (compute ref logp)  │
            │ (tensor copy)    │                     │                      │
            ├──────────────────┤                     │  Advantages ──────►  │
            │ offload actor    │                     │  (compute adv/ret)   │
            │ wake up rollout  │                     │                      │
            ├──────────────────┤                     │  DCS weight sync     │
            │   Rollout        │                     │  (overlaps training) │
            │   (continue)     │                     └──────────────────────┘
            └──────────────────┘
         GPU utilization: ~30-50%                    GPU utilization: ~70-90%
         All operations strictly serial              All operations parallel
         Data via TransferQueue, no overlap           Data via TransferQueue, streaming overlap
```

______________________________________________________________________

## Next Steps

- [Architecture](./architecture.md) — Overall Relax architecture design
- [Distributed Checkpoint](./distributed-checkpoint.md) — DCS detailed documentation
- [Health Check Manager](./health-check-manager.md) — Health monitoring and fault recovery
