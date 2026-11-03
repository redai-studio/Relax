# Frozen Conditioning Cache

Image-generation SFT can reuse deterministic outputs from frozen conditioning encoders when the same effective inputs appear again. Relax stores these tensors in a bounded, node-local file cache so workers on the same node can share them without retaining a separate CPU copy in each process.

Current integrations are SenseNova image-edit LoRA SFT and Qwen-Image text-to-image LoRA SFT. SenseNova reuses reference-image and text prefix KV tensors together with image-position indexes; Qwen-Image reuses frozen text embeddings through the same public cache. Full-parameter SenseNova training keeps these computations online because the conditioning parameters are trainable. Other model adapters and RL paths need their own integration; this feature does not transfer conditions from SGLang or an RL rollout engine.

The launch recipe and branch-grouping controls below target SenseNova. See [Text-to-Image SFT](./t2i-sft.md) for Qwen-Image data, its independent encoder boundary and its launch script.

## Enable the SFT recipe

Prepare an existing Ray cluster and the model, paired-image metadata, and worker-visible storage described in [Image-to-Image SFT](./i2i-sft.md). Set `MODEL_PATH`, `DATA_ROOT`, `METADATA_PATH`, `DATASET_SIZE`, and `SAVE_DIR` for your workload. `DATASET_SIZE` must equal the metadata row count. Also set the Ray Jobs endpoint, a writable node-local `CACHE_DIR`, and a new shared `RUN_DIR` for measurement files.

Run the comparison recipe from the repository root:

```bash
: "${RAY_DASHBOARD_ADDRESS:?Set the Ray Jobs HTTP endpoint}"
: "${MODEL_PATH:?Set the worker-visible model directory}"
: "${DATA_ROOT:?Set the worker-visible image root}"
: "${METADATA_PATH:?Set the paired-image metadata path}"
: "${DATASET_SIZE:?Set the exact metadata row count}"
: "${SAVE_DIR:?Set the checkpoint output root}"
export RAY_DASHBOARD_ADDRESS MODEL_PATH DATA_ROOT METADATA_PATH DATASET_SIZE SAVE_DIR

RELAX_CONDITION_CACHE=1 \
RELAX_CONDITION_CACHE_MAX_BYTES=4294967296 \
RELAX_CONDITION_CACHE_DIR="${CACHE_DIR:?Set a writable node-local cache root}" \
RELAX_CONDITION_CACHE_RUN_DIR="${RUN_DIR:?Set a new shared metrics directory}" \
USE_CLEARML=0 \
bash scripts/entrypoint/ray-job.sh \
    scripts/training/diffusion/run-sensenova-condition-cache.sh
```

This example selects a 4 GiB payload budget per node. Choose it from the available host or tmpfs memory after allowing room for the model, data preparation, and staging buffers. The recipe defaults to 8 GPUs, an effective global batch of 8, 64 optimizer steps, and one conditioning image. `GPUS` may be 1, 2, 4, or 8; it must divide the fixed effective batch. For multiple nodes, set `GPUS_PER_NODE` so it divides `GPUS`. The cache is shared within each node, not across nodes.

The recipe fixes LoRA and the effective batch. To establish an online baseline, run the same command with `RELAX_CONDITION_CACHE=0`, a fresh `RUN_DIR`, and a separate experiment output. Keep the data order, resolution, GPU layout, optimizer, prefetch, and training settings unchanged. `DRY_RUN=1` prints the submission command without launching a job.

## Configuration

These settings are read by the image SFT runtime. The recipe exports them to the Ray workers; custom launch scripts must also include them in their worker runtime environment.

| Environment variable | Runtime default | Effect |
|---|---|---|
| `RELAX_CONDITION_CACHE` | `0` | Enable reusable frozen conditions. Use `0` for online encoding. |
| `RELAX_CONDITION_PROFILE` | `0` | Attach conditioning measurements even when caching is off. The comparison recipe defaults this to `1`. |
| `RELAX_CONDITION_CACHE_MAX_BYTES` | `8589934592` | Per-node payload limit in bytes. An enabled cache requires a positive value. |
| `RELAX_CONDITION_CACHE_DIR` | `/dev/shm/relax-conditioning` | Writable node-local root shared by training workers on that node. Use an absolute path; local disk is also supported. |
| `RELAX_CONDITION_CACHE_RUN_DIR` | Unset | New worker-visible directory for `events-rank*.jsonl`. Setting it also enables measurement. The recipe defaults to `${SAVE_DIR}/${EXP_NAME}/metrics`. |
| `RELAX_CONDITION_VERIFY` | `0` | Compare the first effective training batch through online and cached paths. Requires caching and a run directory. Use a separate run from throughput measurement. |
| `RELAX_CONDITION_VERIFY_RTOL` | `0.01` | Relative tolerance for loss, accumulated gradients, and FP32 parameter updates in the comparison. The recipe accepts values from `0` to `0.01`. |
| `RELAX_CONDITION_VERIFY_SNAPSHOT` | `0` | Save per-rank reference gradients and FP32 updates after successful verification, for comparisons across configurations. Verification only. |
| `RELAX_CONDITION_PREFETCH_DEPTH` | `0` | Maximum future conditioning entries to prefetch; zero disables it. The recipe also requires image prefetch. |
| `RELAX_CONDITION_PREFETCH_MAX_BYTES` | `536870912` | Per-worker lease budget, including in-flight lookup reservations and consumer borrows. |

With all measurement settings unset and caching off, SFT uses its original online path. Enabling the cache also attaches its measurements; `RELAX_CONDITION_PROFILE=0` does not remove cache coordination or accounting.

When using an HTTP proxy, exclude localhost, the Ray head, and internal node addresses through both `NO_PROXY` and `no_proxy` so Ray Jobs and node discovery reach the cluster directly.

The comparison recipe uses BF16 computation, FP32 optimizer master weights, and LoRA rank/alpha 32 by default. Its other defaults and launch arguments are in [`run-sensenova-condition-cache.sh`](../../../scripts/training/diffusion/run-sensenova-condition-cache.sh). These are recipe choices, not global parser defaults.

Two independent training controls are available in the recipe. `PREFIX_GRADIENT_CHECKPOINTING=0` disables activation recomputation; despite its name, this also controls the generation decoder. Measure its memory cost on the actual workload. `FSDP_BRANCH_GROUPS=1` sets `fsdp_branch_groups: true` in the SFT configuration and separates each decoder's understanding and generation modules into independent FSDP communication groups. An unused branch then avoids the whole-layer parameter all-gather. This option defaults off and requires all conditioning parameters to be frozen; full training or LoRA targets that train understanding fail explicitly before sharding. Do not unfreeze conditioning within an existing grouped FSDP instance: disable this option and create a new training task. The common FSDP runtime consumes adapter-owned declarations without SenseNova layer names.

Apply these controls equally to online and cached runs. Report training-configuration improvements separately from cache gains instead of comparing an optimized cached run against an old online configuration.

Optional reference snapshots are stored in the run's `reference-snapshots/` directory. They contain full effective-batch reduced gradients, actual FP32 `theta_after - theta_before`, and fingerprints of initial parameters, optimizer, inputs, and RNG. Rank files are atomically written and checksummed before publishing a complete manifest; failures restore the pre-verification training state. Snapshots are retained verification artifacts, outside the cache payload budget and cache cleanup. Leave this option off for throughput measurements.

## Reuse boundaries and invalidation

A cache key combines the checkpoint/tokenizer file content fingerprint with the adapter's schema and effective input dependencies. SenseNova includes the prompt, ordered reference-image contents and sizes, target dimensions, actual token IDs, special-token IDs, position indexes, image grids, prompt template, preprocessing parameters, attention settings, and dtype. Renaming a sample does not make it reusable, and changing its contents under the same filename does not preserve a hit.

Each lookup prepares the actual encoder inputs on CPU, and online computation consumes that same preparation instead of tokenizing again after key construction. Vocabulary or tokenizer-setting changes invalidate reuse whenever they affect the effective tokens or indexes. If every effective encoder input remains identical, a frozen deterministic output remains reusable. This avoids serializing the full vocabulary on every lookup. Numerical verification independently exercises the original tokenization and device-index path as its online reference.

The runtime fingerprints checkpoint and tokenizer files at startup. Before each sample, SenseNova also checks the conditioning parameters outside FSDP forward: their identity, version counters (including local DTensor storage), trainability, dtype, shape, and stride. It also checks registered conditioning buffers and actual RoPE, normalization, attention, vision, and activation settings. SenseNova records metadata when it deterministically materializes lazy positional tensors; their normal first initialization and supported device movement preserve reuse, while an external replacement or in-place edit disables it. Observed standard in-place updates such as `copy_`, parameter replacement, or a transition to trainable conditioning permanently disable reuse for that model instance. All ranks continue through the coordinated online path. Restoring the original parameters or attaching a new resolver does not reenable reuse; start a new training task to use it again. The runtime does not automatically rebuild a cache for changed encoder weights.

This guard does not detect nonstandard writes that bypass version counters, such as mutation through `.data`. Keep checkpoint/tokenizer files immutable during a run, since their startup file fingerprint is not refreshed. Effective tokenizer and preprocessing settings remain part of the per-lookup dependency key. Changing the adapter's encoding algorithm or feature schema also requires a new schema version in its declaration.

SenseNova checks the actual conditioning parameters' `requires_grad` state. Trainable conditioning computations bypass cache reads and writes and retain their autograd graph. Frozen generation-independent conditions may be reused while generation LoRA weights continue to update.

For LoRA training, the adapter uses independent FSDP boundaries for the vision encoder and text embeddings. An effective hit skips their calls and the associated weight all-gathers. The unused language-model head also has its own boundary, so these training steps do not gather its weights. Online and cached runs use the same boundaries; keep this policy in the baseline when measuring reuse.

Image decoding and sample preparation still run before lookup. The cache skips repeated conditioning preprocessing, input transfer, and encoding on an effective hit. It does not cache the supervised target's training computation, sampled noise, timesteps, generated trajectories, or rewards. For adapters with random crops, augmentations, or caption dropout, preserve their normal order and key the resulting effective inputs rather than silently reusing a previous random choice. This integration does not introduce those augmentations into SenseNova. Qwen optionally reuses deterministic VAE posterior parameters while sampling online every time; see [Text-to-Image SFT](./t2i-sft.md).

## Storage, ownership, and release

The cache writes immutable safetensors payloads with atomic publication and verifies the full file's SHA256 on every lookup. CPU tensors retain their dtype. A lookup returns a lease that prevents eviction while tensors are borrowed. Supported aligned layouts use one payload-byte transfer and create views with the original dtypes over one GPU storage; other layouts use individual tensor copies. The SFT runtime waits for the copy stream's completion event before releasing the lease, including partial-copy failures, and creates a fresh model-specific wrapper for the current forward. No persistent GPU feature cache is created.

The payload limit includes committed files, leased entries, and pending file writes. LRU eviction removes only unborrowed entries. An entry that cannot fit is bypassed, and online encoding continues. Serialization has a separate staging limit, checked before CPU copies begin. In the SFT runtime this limit equals `RELAX_CONDITION_CACHE_MAX_BYTES`, so an 8 GiB payload setting also allows up to an 8 GiB writer staging reservation. The common cache API exposes `max_staging_bytes` for custom integrations. Entry admission uses a conservative buffer estimate, so a tensor may fit as a file but still exceed the staging limit.

Payload and staging limits are not a total process RSS limit. Training inputs, GPU copies, Python metadata, and small per-reader checksum buffers also use memory. JSON indexes and client markers are bounded by the stored entries and live workers; detailed resource snapshots report metadata separately. When using tmpfs, payload files consume host memory as shared-memory storage. With a local-disk directory, the same payload budget limits file contents, and operating-system page caching can also consume host memory.

Workers share a namespace for one training attempt and node. Each worker closes its leases and client on completion, cancellation, or an exception that reaches cleanup. The last live client removes the namespace. After `SIGKILL`, `os._exit`, or worker loss, operating-system locks are released; the next cache initialization reclaims abandoned namespaces only after all client and payload locks are free. There is no cleanup daemon. A subsequent launch builds its own cache rather than treating these files as resumable training state.

Payload and staging budgets apply to one training namespace on each node. Concurrent jobs require the sum of their budgets; there is no global memory quota across jobs. Avoid unrelated node contention during performance comparisons.

## Adapter contract and distributed execution

Concurrent writers share the staging budget. Each reserves capacity in a short critical section, performs CPU copies, serialization and file writes outside the namespace lock, then publishes atomically under the lock. Independent owner locks protect in-flight writes. Recovery removes abandoned temporary files and reservations; client close waits for active operations.

Optional conditioning prefetch overlaps future file lookup and SHA verification with the current backward/update. It uses already decoded samples from the image queue; dependency preparation stays on the training thread. Actual forward rebuilds the content key and checks frozen state. One background reader charges complete file bytes until the consumer finishes its device copy. This extends shared mmap leases, without persistent GPU features or a pinned pool; device copies remain on the training thread. Longer leases can affect eviction under capacity pressure.

Retained cursor positions in the sliding window reuse their scheduling hints. Leaving the window, an exception, or close discards those hints; their count is bounded by prefetch depth. Hints only schedule I/O and never authorize feature reuse by sample ID. Actual forward independently validates the input and frozen state.

Consumption checks file identity, size and timestamps and falls back to synchronous lookup on changes; every new lookup still verifies the complete SHA. Close drains the reader before closing storage, including that work in lifecycle timing. `prefetch_hit` counts consumed results; `prefetch_stats` separately reports background reads, waits, discards and budget peaks. Speculative reads are not effective sample hits.

The public declarations are in [`relax/models/generative_sft.py`](../../../relax/models/generative_sft.py). An adapter supplies a `ConditioningSpec(name, dependencies, cacheable)` and calls `ConditioningResolver.resolve(spec, compute, restore)` inside the model's normal FSDP forward boundary.

- `name` versions the feature schema. `dependencies` contains JSON-compatible content, preprocessing, and dtype metadata; the runtime adds its weight fingerprint.
- `cacheable` must reflect the actual frozen parameter boundary. When it is false, the online result must keep gradients and must not be stored or detached.
- `compute` returns a flat tensor dictionary. `restore` validates names, shapes, dtypes, and devices and rebuilds the current forward's model-specific objects. It must not mutate borrowed tensors or retain CPU mappings after the lease ends.

SenseNova additionally supplies `conditioning_state()` for the mutation guard. Its optional `semantic` field contains CPU-only JSON configuration; the common runtime stores an immutable canonical digest and fails closed on changes or invalid declarations. Adapters must also place output-affecting configuration in their per-input dependencies. SenseNova uses schema `sensenova.prefix.v3` for this contract. Its process-local parameter identities detect replacement but never enter a shared cache key. Custom adapters that supply this hook must inspect sharded master metadata outside the FSDP forward boundary, rather than treating FSDP's temporary full-parameter views as encoder replacements.

Each rank can require different inputs. Before skipping an encoder, the runtime requires every rank in the FSDP parameter-sharding group to have valid conditions for its own inputs. If any rank misses or fails to restore a payload, all ranks run the online encoder for that microstep. This keeps FSDP collectives aligned. A local hit can therefore become a forced recomputation; use the effective hit rate when interpreting results.

Complete frozen replicas can instead use rank-local decisions. The adapter declares
the conditioning modules; the backend validates native FSDP exclusion, no nested
FSDP boundaries and no distributed parameters. The adapter must additionally ensure
that compute/restore contain no explicit collectives. The runtime agrees on the
policy once and fixes it for its lifetime. Unanimous decisions remain the default;
content keys, mutation guards, capacity and release rules are unchanged.

Missing files, checksum failures, or adapter schema/shape errors fall back to online computation. A bad key is invalidated without clearing unrelated entries; deletion waits for outstanding borrows. The storage layer has no knowledge of SenseNova layer names. Another SFT adapter can use the same resolver and cache contract. Future RL integration would need to define its own dependencies, ownership, and distributed skip boundary; no RL feature handoff is provided here.

## Measure a workload

Reuse helps when deterministic conditions repeat often enough and their original encoding cost exceeds hashing, lookup, coordination, and transfer costs. Low-reuse data can become slower. Feature tensors may also be larger than raw conditioning inputs, so a hit does not imply fewer transferred bytes.

Use fresh run directories for repeated online/cached comparisons and measure complete training loops, including data preparation, forward, backward, and optimizer updates. Report startup and cache construction separately from steady throughput. Choose the cold window from the dataset and expected reuse pattern rather than relying on the summarizer's default of 16 steps.

```bash
python scripts/tools/benchmark_condition_cache.py \
    --online "${ONLINE_RUN_DIR:?Set the online run directory}" \
    --cached "${CACHED_RUN_DIR:?Set the cached run directory}" \
    --cold-steps "${COLD_STEPS:?Set the cold-window step count}" \
    --output "${REPORT_JSON:?Set the summary JSON path}"
```

The tool accepts multiple run directories per group and separate `--low-reuse-online` and `--low-reuse-cached` groups. It compares matching complete workloads, uses the slowest rank's step time, and counts the global batch once. Local encoding time is diagnostic, not end-to-end throughput. Full Ray job duration additionally requires externally collected `job.json`; trainer events alone exclude model and Ray startup.

Inspect `encoding_calls`, `local_hit`, `effective_hit`, `forced_recompute`, input/feature transfer bytes, and cache lookup/store times together. Every step records CPU RSS, GPU allocated/reserved memory, and node-wide shared-memory use. Most steps use `stats_light()` with `cache_storage_scope=index_accounting`: entry counts, resident bytes, and reservations come from the locked index without scanning payloads. `leased_bytes`, `metadata_bytes`, and `total_bytes` are `null` because that snapshot does not measure them; these values are unknown, not zero or a retained detailed reading.

The first executed step, steps whose index is a multiple of 16, and the final step request full `stats()`. A successful filesystem snapshot has `cache_storage_scope=filesystem` and `cache_stats_detailed=true`. Close also requests detailed statistics after releasing the client, but a closed cache reports `last_detailed_snapshot`; this is not a live measurement, and the runtime leaves current cache/disk usage unknown. Verify release using the namespace's filesystem state and external before/after resource snapshots. Node-shared cache bytes must not be summed once per rank. For low effective hit rates, check input/key changes, staging bypasses, capacity pressure, and misses on other ranks before increasing the cache size.

See [Image-to-Image SFT](./i2i-sft.md) for data and resume behavior, and [Performance Tuning](./performance-tuning.md) for workload measurement.

The reusable plan runner, provenance checks, repeated-plan summary, and FP32 snapshot comparison are documented in [the benchmark tools](../../../scripts/benchmarks/conditioning_cache/README.md).
