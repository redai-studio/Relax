# Hierarchical Cache

Hierarchical caching extends reusable prompt state beyond GPU memory. SGLang HiCache backs up prefixes to CPU memory and can use a storage backend for additional capacity. Later requests can restore evicted prefixes instead of computing them again. This can help workloads with repeated long prompts, such as multi-turn conversations and agents; the benefit depends on reuse and the cost of moving data.

| Tier | Location | How a request uses it |
| --- | --- | --- |
| L1 | GPU memory | Reuse resident prefix state directly |
| L2 | CPU memory | Restore a backed-up prefix to GPU after GPU eviction |
| L3, optional | Storage backend, such as local SSD or a distributed store | Read a stored prefix into CPU memory, then restore it to GPU |

Hierarchical caching controls where prefix state is kept. [PD disaggregation](./pd-disaggregation.md) controls which engines process prompts and generate tokens. The two can be configured together; neither requires a particular agent framework or training algorithm.

## Enable CPU caching

Use the environment and matching SGLang patch from [Installation](./installation.md). The model's cache implementation must support the selected HiCache layout and parallelism settings. Relax forwards `--sglang-*` arguments to SGLang.

To enable CPU caching on engines that share the same settings, add these arguments to an existing training configuration:

```bash
--sglang-enable-hierarchical-cache \
--sglang-hicache-ratio 2.0 \
--sglang-hicache-write-policy write_through
```

With `hicache_storage_backend` left unset, this enables GPU+CPU caching. `write_through` backs up eligible prefixes to CPU memory, and `hicache_ratio` sizes the host pool relative to its device pool. Capacity and model-specific state are discussed below.

For PD, configure the prefill and decode groups separately through `--sglang-config`. The following example adds CPU caching to the prefill side of the allocation used in the [PD guide](./pd-disaggregation.md#configure-engine-groups-with-yaml):

```yaml
sglang:
  - name: default
    num_gpus_per_engine: 2
    engine_groups:
      - worker_type: prefill
        num_gpus: 4
        overrides:
          enable_hierarchical_cache: true
          hicache_ratio: 2.0
          hicache_write_policy: write_through
          hicache_storage_backend: null
      - worker_type: decode
        num_gpus: 4
        overrides:
          enable_hierarchical_cache: false
          hicache_storage_backend: null
          disaggregation_decode_enable_radix_cache: false
```

Save the YAML and pass its path with `--sglang-config /path/to/pd-cache.yaml`. With PP1 and DP1, it creates two prefill and two decode engines, each TP2. Eight GPUs is an example allocation; adapt the groups to the model and available resources. Group-level `overrides` use SGLang field names and take precedence over the corresponding CLI arguments. Add any cache settings required by the model separately.

## Add a storage tier

`hicache_storage_backend` selects L3 independently of CPU caching. SGLang provides `file`, `mooncake`, `hf3fs`, `nixl`, and other backends, as well as dynamic backend loading. Their dependencies, supported cache components, and sharing semantics differ; use the [HiCache documentation](https://docs.sglang.io/docs/advanced_features/hicache) for the SGLang version in your environment.

For a local file backend, replace the prefill group's storage settings with:

```yaml
hicache_storage_backend: file
hicache_storage_prefetch_policy: wait_complete
hicache_storage_backend_extra_config: '{"enable_metadata_cache": false, "max_size": 34359738368, "min_free_space": 68719476736, "eviction_ratio": 0.9}'
```

These are example settings: `max_size` allows 32 GiB per storage-owning rank, `min_free_space` requires 64 GiB free on the backing filesystem, and `eviction_ratio` targets 90% of the quota after eviction. They are not whole-job budgets. `wait_complete` waits for the storage prefetch to finish; it can improve reuse but adds latency when IO is slow.

For Relax-managed native prefill engines, create a dedicated absolute directory on local SSD and pass it to **Ray workers** as `RELAX_HICACHE_FILE_ROOT`. Merge the following into the job's runtime environment, replacing the path with your own:

```yaml
env_vars:
  RELAX_HICACHE_FILE_ROOT: /local/ssd/relax-cache/run-001
```

The root must already exist. Relax creates `prefill-engine-0000`, `prefill-engine-0001`, and so on below it, using each engine's logical rank. Setting the variable only on the submitter does not configure remote workers. This directory helper applies only to `file` on native prefill engines and currently requires a single-node engine with `dp_size=1`; it does not manage regular, external, or custom engines.

The file backend keeps its LRU and quota accounting inside each process. Separate engine directories keep eviction and quota ownership clear. TP ranks within an engine use native filename suffixes to distinguish shards; MLA uses a single storage owner for replicated data. With two TP2 prefill engines storing separate shards and the quota above, the aggregate file budget is 128 GiB. The 64 GiB free-space threshold is shared by the filesystem, not reserved again for each rank.

Sharing L3 across engines is possible with a suitable backend and compatible cache keys. Pointing independent file-backend instances at one directory does not provide coordinated eviction or quotas, and the current per-engine weight namespace does not enable cross-engine reuse. Stop the job and its workers before removing its cache directory.

## Request flow and cache compatibility

1. The selected engine matches the request against its local prefix cache.
2. A GPU hit is reused directly; CPU state is restored when the matching prefix is no longer on GPU.
3. If a storage backend is configured, a storage hit can restore missing host state before loading it to GPU.
4. Missing or incompatible state is recomputed through prefill.

For hybrid models, the reusable state can include both attention KV pages and recurrent state such as Mamba checkpoints. Recovery requires compatible state for the same prefix and weights on all participating ranks. KV files alone may be insufficient, and a chunk's length does not guarantee a checkpoint at its end.

Each DP replica has its own prefix tree and GPU/CPU pools. TP ranks within a replica hold shards of that replica's state and coordinate cache operations. Attention DP similarly separates local state by attention-DP group. A request routed to another replica does not automatically inherit an L1/L2 hit; reuse across replicas depends on routing or a shared storage tier.

DP1 is a limit of the directory helper above and of the managed static PD startup described in the [PD guide](./pd-disaggregation.md#routing-and-limits). It is not a general restriction of hierarchical caching. Removing that check alone would not add per-DP storage accounting or DP-aware PD routing.

## Capacity and weight updates

| Setting | Effect |
| --- | --- |
| `hicache_ratio` | Host-pool capacity relative to the corresponding device pool |
| `hicache_size` | Fixed host-pool size in GB; a positive value overrides the ratio |
| `hicache_write_policy` | When eligible prefixes are backed up, for example `write_through` |
| `hicache_io_backend` / `hicache_mem_layout` | GPU↔CPU transfer mechanism and host layout; select a combination supported by the model |
| `hicache_storage_prefetch_policy` | When to wait for or stop storage prefetch |
| `hicache_storage_backend_extra_config` | Backend-specific configuration, including file quotas |

`max_total_tokens` is optional. When omitted, SGLang derives GPU token-pool capacity from its memory budget. Set it when an explicit cap is needed, such as a controlled capacity comparison. It is neither a required HiCache setting nor a per-request output limit. Since a ratio-sized host pool follows device capacity, changing the GPU pool can also change CPU memory use.

Budget host memory across all engines and ranks, including every cache component. Hybrid models may allocate substantial recurrent-state pools in addition to KV. Add training offload, weights, Ray, and OS usage, and check both startup allocation logs and container limits. `hicache_ratio=2` is a capacity ratio, not a two-gigabyte or whole-machine limit. The `direct` IO option controls GPU/CPU transfers; it does not mean file reads bypass the OS page cache.

RL changes model weights between rollout iterations, so cached states must be invalidated when their weights change. For file L3, the repository's SGLang v0.5.17 patch creates a new namespace at engine startup and advances it on weight updates, after draining active cache IO. Old files become unreachable by new requests but remain on disk until eviction. Restarted engines start with a cold namespace; ordinary offload/onload without a weight change preserves that namespace. Other storage backends do not automatically receive this file-specific namespace management.

Cancellation also needs to preserve buffer ownership: an aborted request can still have a storage read in flight. Its buffers remain owned until IO finishes. This lets cache completion be drained before cache reset or a weight update.

## Check cache use and diagnose problems

Inspect each engine's effective settings, host allocations, and storage directory. Distinguish GPU hits, CPU restoration, and L3 restoration; the existence of cache files alone does not establish that a request used them. Where available, native generation responses expose `cached_tokens_details` with `device`, `host`, and `storage` counts. Combine these with IO metrics and complete rollout time.

| Symptom | Check |
| --- | --- |
| No CPU or storage hits | The prefix may still fit on GPU, be too short for a checkpoint, have been invalidated by weights, or reach another engine |
| Files exist but storage hits stay zero | Check namespace, complete cache components across ranks, and whether L1/L2 already serves the prefix |
| Host allocation failure | Sum all component pools across engines/ranks, then include training offload and container limits |
| File writes fail or stop | Check permissions, quota, filesystem free space, and storage-worker errors |
| Restoration is slower than recomputation | Compare prefix length, IO and transfer costs, and prefetch waiting time |

For performance comparisons, keep the model, requests, concurrency, and GPU/CPU budgets consistent. Larger cache capacity can reduce repeated prefill, but storage latency or time spent in tools may dominate end-to-end rollout.

For a complete agent training recipe, see the [Mini SWE example](../../../examples/mini_swe_agent/README.md#train-with-cpu-and-ssd-caching), including its launch commands and model-specific YAML. General task concurrency and export behavior are covered in [Agentic Rollout](./agentic-rollout.md).
