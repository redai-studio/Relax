# Frozen Conditioning Cache

Image SFT can reuse deterministic features when the same effective input reaches a frozen encoder. SenseNova reuses the understanding prefix K/V and image indexes; Qwen-Image reuses text embeddings and deterministic VAE posterior parameters. Both use the same bounded, node-local cache. Posterior sampling, diffusion noise, timesteps and trainable branches remain online. RL integration is outside this implementation.

## Launch

Prepare the model and data described in [SenseNova I2I SFT](./i2i-sft.md) or [Qwen-Image T2I SFT](./t2i-sft.md). Set `MODEL_PATH`, `DATA_ROOT`, `METADATA_PATH`, `DATASET_SIZE`, `SAVE_DIR` and `RAY_DASHBOARD_ADDRESS` to worker-visible paths and the existing Ray Jobs endpoint. `DATASET_SIZE` must equal the metadata row count.

```bash
RELAX_SKIP_CLUSTER_CLEANUP=1 \
RELAX_CONDITION_CACHE=1 RELAX_CONDITION_CACHE_MAX_BYTES=4294967296 \
bash scripts/entrypoint/ray-job.sh \
    scripts/training/diffusion/run-sensenova-i2i-sft-8xgpu.sh
```

For Qwen, use `scripts/training/diffusion/run-qwen-image-sft.sh` with the same cache settings. Both use the normal training entrypoint; no validation tools or metrics directory are needed. Set `RELAX_CONDITION_CACHE=0` to return to online encoding. Cache reuse is most useful when frozen conditions repeat; low-reuse workloads still pay lookup and cache-build costs and can run more slowly.

## Controls

The two training scripts forward these settings to Ray workers. Custom launchers must include them in `RUNTIME_ENV_JSON.env_vars`.

| Variable | Default | Meaning |
| --- | --- | --- |
| `RELAX_CONDITION_CACHE` | `0` | Enable reuse; `0` keeps online encoding. |
| `RELAX_CONDITION_CACHE_MAX_BYTES` | 8 GiB | Positive per-node payload budget. |
| `RELAX_CONDITION_CACHE_DIR` | `/dev/shm/relax-conditioning` | Writable node-local root; local disk is also supported. |
| `RELAX_CONDITION_PREFETCH_DEPTH` | `0` | SenseNova lookahead entries; requires image prefetch. Qwen requires `0`. |
| `RELAX_CONDITION_PREFETCH_MAX_BYTES` | 512 MiB | Per-worker bound on prefetch reservations and borrowed entries. |

## Reuse and invalidation

Keys cover checkpoint/tokenizer file contents, the adapter schema, effective input contents, preprocessing, dtype and output-affecting settings. A filename alone never authorizes reuse. Random crops, augmentation and caption dropout keep their execution and RNG order; the key uses the resulting input. Qwen keys the actual image pixels and VAE settings. It caches posterior parameters with their dtype and sampling-relevant memory layout; posterior sampling still runs every time in the original RNG order. The frozen VAE is replicated in both online and cache runs to avoid repeated parameter gathering. Posterior reuse supports contiguous and channels-last-3d layouts; other layouts require disabling the cache. Unfreezing a replicated encoder requires rebuilding its FSDP ownership.

The adapter declares the frozen parameter boundary. Parameter replacement, normal in-place mutation, unfreezing, buffer changes and semantic configuration changes disable reuse and fall back online. SenseNova checks RoPE and other conditioning state while allowing its deterministic lazy initialization. Changes to generation-only LoRA do not invalidate independent frozen conditions. State checks run outside FSDP forward to avoid confusing temporary parameter views with replacement. The guard cannot detect writes that bypass version counters through `.data`; checkpoint files must also remain unchanged during a run.

All ranks in the parameter-sharding group agree before skipping an encoder. Any miss or restore failure makes the whole group encode that microstep online, preserving matching collectives. Missing files, failed checksums and schema/shape errors invalidate the affected entry and fall back. Trainable conditions retain their autograd graph and are never cached.

## Capacity, ownership and release

Workers in one attempt share immutable safetensors payloads on each node. Every new lookup verifies the file SHA and preserves tensor dtype. The caller borrows a lease; eviction cannot remove a borrowed entry. GPU copies belong to the current forward. The runtime retains source leases until completion events are ready, without waiting on each hit, and drains them at step end or shutdown. Pending copies per rank are limited to its effective-batch sample count and the payload byte budget; insufficient capacity falls back online. There is no persistent GPU feature cache.

The payload cap includes committed, leased and pending files. LRU evicts only unborrowed entries; oversized entries or insufficient capacity bypass caching. Writer staging has a separate bound equal to the payload budget, checked before CPU copies. Optional SenseNova prefetch has its own per-worker byte/depth bounds and drains before storage closes. Hints only schedule reads; actual forward rebuilds dependencies and checks frozen state.

These limits are not total RSS limits: inputs, temporary GPU copies and metadata also consume memory. A tmpfs cache consumes host RAM; disk caches can consume OS page cache. Concurrent training attempts have independent budgets, which must be added when sizing the node.

Normal completion, cancellation and caught exceptions close leases and clients; the last client removes the namespace. After abrupt worker death, OS locks are released and the next cache initialization reclaims abandoned namespaces once their client/payload locks are free. There is no cleanup daemon. Checkpoint resume starts with a cold cache; cache files are not part of the training checkpoint.

## Model adapters

[`ConditioningSpec` and `ConditioningResolver`](../../../relax/models/generative_sft.py) keep the cache model-independent. The adapter declares schema/version, effective dependencies and `cacheable`, then calls `resolve(spec, compute, restore)` inside normal model forward. `compute` returns a flat tensor dictionary; `restore` validates fields/shape/dtype and constructs the model's forward objects. Borrowed tensors must not be modified or retained after their lease. `conditioning_state()` optionally declares parameter/buffer/semantic state for mutation checks. A future RL caller would need its own ownership and collective boundary; this change does not transfer rollout features.

Versioned validation tools and recorded experiment evidence are available in the [standalone validation repository](https://github.com/Religious-J/relax-condition-cache-validation/tree/117d4fb037d93e2123734ccacf3e411e5f5b7b1e).
