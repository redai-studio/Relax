# CPU Cache

CPU caching extends reusable prompt state from GPU memory into host memory. SGLang HiCache restores prefixes evicted from the GPU, reducing repeated prefill work. Multi-turn conversations and agents that repeatedly submit long prefixes can benefit; actual gains depend on prefix reuse and transfer costs.

| Tier | Location | Request behavior |
| --- | --- | --- |
| L1 | GPU memory | Reuse a resident prefix directly |
| L2 | CPU memory | Restore a backed-up prefix to the GPU after device eviction |

**CPU cache works independently in regular deployment and can also be combined with [PD disaggregation](./pd-disaggregation.md).** A regular engine handles both prefill and decode; PD assigns these stages to separate engines. Cache placement and engine topology are independent settings. This guide and its recipes support GPU+CPU caching only; L3 storage is outside this change.

## Regular deployment

Use the environment and SGLang patch from the [installation guide](./installation.md). Confirm that your model's cache implementation supports the chosen HiCache layout and parallel configuration. Relax forwards SGLang settings through `--sglang-*` options.

When all regular engines use the same cache settings, add these options to an existing training recipe:

```bash
--sglang-enable-hierarchical-cache \
--sglang-hicache-ratio 2.0 \
--sglang-hicache-write-policy write_through
```

Leave `hicache_storage_backend` unset or `null`. These options require no prefill/decode groups, PD router, or storage directory. `write_through` backs up eligible prefixes to CPU memory; `hicache_ratio` sizes host pools relative to their corresponding device pools.

Alternatively, use `--sglang-config` to configure CPU cache on regular engines:

```yaml
sglang:
  - name: default
    num_gpus_per_engine: 2
    engine_groups:
      - worker_type: regular
        num_gpus: 8
        overrides:
          enable_hierarchical_cache: true
          hicache_ratio: 2.0
          hicache_write_policy: write_through
          hicache_storage_backend: null
```

With PP1 and DP1, this creates four TP2 regular engines. Group `overrides` use SGLang field names and take precedence over corresponding command-line options. GPU count and TP2 are recipe choices; add any model-specific cache settings separately.

## PD deployment

PD deployments can configure the two sides separately. This example enables CPU cache only on prefill engines:

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

Select the saved YAML with `--sglang-config /path/to/cpu-cache.yaml`. This creates two prefill and two decode engines, each TP2. Disabling cache on decode engines is a choice in this PD recipe; it does not restrict regular engines. See the [PD guide](./pd-disaggregation.md#routing-and-limits) for managed PD resource, routing, and parallelism constraints.

## Compatibility and capacity

Each request matches the selected engine's local prefix tree. A GPU hit is reused directly; a CPU hit is restored to the GPU; missing or incompatible state is recomputed. Hybrid models may require both attention KV pages and recurrent state such as Mamba checkpoints. Participating ranks must hold compatible state for the same prefix and weights. A prefill chunk boundary does not guarantee a reusable checkpoint at that boundary.

Each DP replica owns its prefix tree and GPU/CPU pools. TP ranks in a replica hold their shards and coordinate cache operations; Attention DP similarly partitions local state by attention-DP group. Routing a request to another replica does not transfer its cache hits. The Mini SWE recipes include regular TP2 engines, 2P2D TP2, and 1P1D with attention DP2 × TP2 and EP4; these are specific layouts, not general CPU cache parallelism limits.

The DP2 PD recipe requires the paired router fix described in the [Mini SWE example](../../../examples/mini_swe_agent/README.md#train-with-cpu-caching). It uses a stable session key for DP-aware consistent hashing on both sides. P/D owners can differ: the router must pass each engine's selected rank and tell decode which prefill rank supplies KV. Verify both actual ranks and host restores on each prefill DP rank. A stable decode rank alone does not prove prefill cache affinity. CPU cache itself does not require PD or this routing policy.

| Setting | Purpose |
| --- | --- |
| `hicache_ratio` | Host capacity relative to the corresponding device pool |
| `hicache_size` | Fixed host capacity in GB; a positive value overrides the ratio |
| `hicache_write_policy` | When eligible prefixes are backed up, such as `write_through` |
| `hicache_io_backend` / `hicache_mem_layout` | GPU↔CPU transfer method and host layout; choose a model-supported combination |

`max_total_tokens` is an optional capacity cap. Without it, SGLang derives GPU token-pool capacity from its memory budget. Set it explicitly for controlled capacity comparisons; it is neither a CPU cache requirement nor a per-request output limit.

Budget host memory across every cache-enabled engine and rank, including both KV and recurrent state pools, training offload, weights, Ray, and the operating system. `hicache_ratio=2` means a capacity ratio, not 2 GB or a machine-wide limit. At equal per-rank capacities, four TP2 regular engines require twice the host-pool budget of the 2P2D recipe above. Check actual allocation logs and container limits.

## RL weight updates and pausing

CPU cache contents depend on model weights. Relax's weight synchronization pauses engines and flushes GPU/CPU prefix caches before updating weights, preventing new weights from reusing old prefixes. Engine restarts also start with empty caches. Use the framework lifecycle for training offload/onload.

Pausing generation does not complete pending GPU↔CPU transfers. The accompanying patch continues processing cache completion events while regular and prefill engines are paused, so in-flight buffers can be released safely before flush. For Full KV+Mamba models with chunked prefill, it also preserves CPU backup of a protected checkpoint before a final short tail, without fabricating missing Mamba state.

## Verify hits and troubleshoot

Inspect actual engine roles, `enable_hierarchical_cache`, host allocations, and `hicache_storage_backend=null`. Native generation responses can attribute hits through `cached_tokens_details.device` and `.host`; `.storage` should be zero in this CPU configuration. Details may be `null` for a zero-hit response. A host-pool allocation alone does not prove CPU restoration: also check H2D counters and output correctness.

| Symptom | Check |
| --- | --- |
| No CPU hits | Prefix still on GPU, insufficient checkpoint length, weight invalidation, or routing to another engine |
| Host allocation failure | All component pools across engines/ranks, training offload, and container limits |
| Flush waits | Remaining requests, GPU/CPU transfers, or unprocessed completion events |
| Restore slower than recompute | Prefix length, transfer costs, and full rollout time |

The [Mini SWE example](../../../examples/mini_swe_agent/README.md#train-with-cpu-caching) includes regular and PD launch commands and Qwen3.5-specific YAML. See [Agentic Rollout](./agentic-rollout.md) for concurrency and export contracts. Functional validation does not establish a speedup; performance comparisons require equal models, requests, concurrency, and cache capacities, plus independent repeated runs.
