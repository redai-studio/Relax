# Frozen Conditioning Cache

Image SFT can reuse deterministic features when the same effective input reaches a frozen encoder. SenseNova reuses the understanding prefix K/V and image indexes; Qwen-Image reuses text embeddings. Both use the same bounded, node-local cache. VAE encoding and sampling, diffusion noise, timesteps and trainable branches remain online. RL integration is outside this implementation.

## Launch

Prepare the model and data described in [SenseNova I2I SFT](./i2i-sft.md) or [Qwen-Image T2I SFT](./t2i-sft.md). Set `MODEL_PATH`, `DATA_ROOT`, `METADATA_PATH`, `DATASET_SIZE`, `SAVE_DIR` and `RAY_DASHBOARD_ADDRESS` to worker-visible paths and the existing Ray Jobs endpoint. `DATASET_SIZE` must equal the metadata row count.

```bash
RELAX_SKIP_CLUSTER_CLEANUP=1 USE_CLEARML=0 \
RELAX_CONDITION_CACHE=1 RELAX_CONDITION_CACHE_MAX_BYTES=4294967296 \
RELAX_CONDITION_CACHE_DIR="${CACHE_DIR:?Set a writable node-local cache root}" \
RELAX_CONDITION_CACHE_RUN_DIR="${RUN_DIR:?Set a fresh shared metrics directory}" \
bash scripts/entrypoint/ray-job.sh \
    scripts/training/diffusion/run-sensenova-condition-cache.sh
```

For Qwen, replace the training script with `run-qwen-image-sft-condition-cache.sh`; it adds cache controls to the basic online recipe. Set `RELAX_CONDITION_CACHE=0` and use a fresh output directory for the online baseline. Keep data order, effective batch, resolution, optimizer and training settings identical. Run a training script directly with `DRY_RUN=1` to inspect its command without submitting a job or creating directories.

## Controls

The launchers forward these settings to Ray workers; custom launchers must do the same.

| Variable | Default | Meaning |
| --- | --- | --- |
| `RELAX_CONDITION_CACHE` | `0` | Enable reuse; `0` keeps online encoding. |
| `RELAX_CONDITION_PROFILE` | Runtime `0`, launchers `1` | Measure conditioning even with reuse disabled. |
| `RELAX_CONDITION_CACHE_MAX_BYTES` | Runtime/SenseNova 8 GiB, Qwen 1 GiB | Positive per-node payload budget. |
| `RELAX_CONDITION_CACHE_DIR` | `/dev/shm/relax-conditioning` | Writable node-local root; local disk is also supported. |
| `RELAX_CONDITION_CACHE_RUN_DIR` | Launcher `${SAVE_DIR}/${EXP_NAME}/metrics` | Fresh worker-visible directory for `events-rank*.jsonl`; also enables measurement. |
| `RELAX_CONDITION_VERIFY` | `0` | Verify the first effective batch; requires cache and a run directory. |
| `RELAX_CONDITION_VERIFY_RTOL` | `0.01` | Loss/gradient/FP32-update relative tolerance, from `0` to `0.01`. |
| `RELAX_CONDITION_PREFETCH_DEPTH` | `0` | SenseNova lookahead entries; requires image prefetch. Qwen requires `0`. |
| `RELAX_CONDITION_PREFETCH_MAX_BYTES` | 512 MiB | Per-worker bound on prefetch reservations and borrowed entries. |

## Reuse and invalidation

Keys cover checkpoint/tokenizer file contents, the adapter schema, effective input contents, preprocessing, dtype and output-affecting settings. A filename alone never authorizes reuse. Random crops, augmentation and caption dropout keep their execution and RNG order; the key uses the resulting input. Qwen VAE encoding and posterior sampling run every time.

The adapter declares the frozen parameter boundary. Parameter replacement, normal in-place mutation, unfreezing, buffer changes and semantic configuration changes disable reuse and fall back online. SenseNova checks RoPE and other conditioning state while allowing its deterministic lazy initialization. Changes to generation-only LoRA do not invalidate independent frozen conditions. State checks run outside FSDP forward to avoid confusing temporary parameter views with replacement. The guard cannot detect writes that bypass version counters through `.data`; checkpoint files must also remain unchanged during a run.

All ranks in the parameter-sharding group agree before skipping an encoder. Any miss or restore failure makes the whole group encode that microstep online, preserving matching collectives. Missing files, failed checksums and schema/shape errors invalidate the affected entry and fall back. Trainable conditions retain their autograd graph and are never cached.

## Capacity, ownership and release

Workers in one attempt share immutable safetensors payloads on each node. Every new lookup verifies the file SHA and preserves tensor dtype. The caller borrows a lease; eviction cannot remove a borrowed entry. GPU copies belong to the current forward, and the runtime waits for copy completion before returning the lease. There is no persistent GPU feature cache.

The payload cap includes committed, leased and pending files. LRU evicts only unborrowed entries; oversized entries or insufficient capacity bypass caching. Writer staging has a separate bound equal to the payload budget, checked before CPU copies. Optional SenseNova prefetch has its own per-worker byte/depth bounds and drains before storage closes. Hints only schedule reads; actual forward rebuilds dependencies and checks frozen state.

These limits are not total RSS limits: inputs, temporary GPU copies and metadata also consume memory. A tmpfs cache consumes host RAM; disk caches can consume OS page cache. Concurrent training attempts have independent budgets, which must be added when sizing the node.

Normal completion, cancellation and caught exceptions close leases and clients; the last client removes the namespace. After abrupt worker death, OS locks are released and the next cache initialization reclaims abandoned namespaces once their client/payload locks are free. There is no cleanup daemon or persistent cache recovery requirement. Cache statistics report accounted payload/reservation bytes, not physical peaks; unknown measurements remain `null`, and shared node values must not be summed across ranks. Verify release and physical CPU/GPU/shared-memory/disk peaks with external monitoring.

## Adapter and validation

[`ConditioningSpec` and `ConditioningResolver`](../../../relax/models/generative_sft.py) keep the cache model-independent. The adapter declares schema/version, effective dependencies and `cacheable`, then calls `resolve(spec, compute, restore)` inside normal model forward. `compute` returns a flat tensor dictionary; `restore` validates fields/shape/dtype and constructs the model's forward objects. Borrowed tensors must not be modified or retained after their lease. `conditioning_state()` optionally declares parameter/buffer/semantic state for mutation checks. A future RL caller would need its own ownership and collective boundary; this change does not transfer rollout features.

Use separate verification and performance runs. Verification restores the initial weights, optimizer and RNG, then compares loss, complete effective-batch accumulated/reduced gradients and actual FP32 `theta_after - theta_before`, per parameter. All values must be finite; zero references require zero results. Deterministic runs should use tolerance `0`; non-deterministic results need repeated online references and a justified near-zero absolute tolerance as well as the 1% relative ceiling.

Performance measurements must include data preparation, condition loading, forward, backward and optimizer updates. Repeat matched online/cache runs; report cold build, steady throughput, full job time, payback and low-reuse overhead separately. Existing FSDP grouping, encoder placement and checkpointing optimizations belong in both arms. Features may be larger than raw inputs, so reduced encoding does not imply fewer transferred bytes. See the [validation and summary guide](../../../scripts/benchmarks/conditioning_cache/README.md) for commands, provenance and resource reporting.
