# CPU and Mooncake Cache

CPU caching extends reusable prompt state from GPU memory into host memory. SGLang HiCache restores prefixes evicted from the GPU, reducing repeated prefill work. Multi-turn conversations and agents that repeatedly submit long prefixes can benefit; actual gains depend on prefix reuse and transfer costs.

| Tier | Location | Request behavior |
| --- | --- | --- |
| L1 | GPU memory | Reuse a resident prefix directly |
| L2 | CPU memory | Restore a backed-up prefix to the GPU after device eviction |

**CPU cache works independently in regular deployment and can also be combined with [PD disaggregation](./pd-disaggregation.md).** A regular engine handles both prefill and decode; PD assigns these stages to separate engines. Cache placement and engine topology are independent settings. The same engines can optionally use Mooncake Store as L3, independently of whether PD is enabled.

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

## Mooncake Store L3

Mooncake Store adds a third tier: **GPU → CPU → Store**, with restoration in the reverse direction. L3 describes its position in this cache hierarchy; a DRAM-backed Store is not an SSD cache. The Mooncake transport used for P/D transfer and the Mooncake Store service are separate features. Enabling P/D transfer does not enable L3.

Use the CUDA 12 image built from `docker/Dockerfile`, which applies the patched SGLang 0.5.17 and pins `mooncake-transfer-engine==0.3.12.post1`. The CUDA 13 images use SGLang 0.5.21 and do not include these L3 lifecycle changes. Start a dedicated Mooncake Store master and storage client before starting the engines. For a single-node TCP validation setup, choose unused ports and set `MOONCAKE_RPC_PORT`, `MOONCAKE_METADATA_PORT`, `MOONCAKE_METRICS_PORT`, and `MOONCAKE_CLIENT_PORT`:

```bash
mooncake_master --rpc_address=127.0.0.1 --port="$MOONCAKE_RPC_PORT" \
  --enable_http_metadata_server=true --http_metadata_server_host=127.0.0.1 \
  --http_metadata_server_port="$MOONCAKE_METADATA_PORT" --metrics_port="$MOONCAKE_METRICS_PORT"

# Run separately and keep it alive for the duration of the job.
mooncake_client --host=127.0.0.1 --port="$MOONCAKE_CLIENT_PORT" \
  --master_server_address="127.0.0.1:$MOONCAKE_RPC_PORT" \
  --metadata_server="http://127.0.0.1:$MOONCAKE_METADATA_PORT/metadata" \
  --protocol=tcp --global_segment_size="16 GB"
```

Create a JSON connection file using those ports:

```json
{
  "local_hostname": "127.0.0.1",
  "master_server_address": "127.0.0.1:<RPC port>",
  "metadata_server": "http://127.0.0.1:<metadata port>/metadata",
  "master_metrics_port": 0,
  "check_server": false,
  "protocol": "tcp",
  "device_name": "",
  "global_segment_size": 0,
  "standalone_storage": false
}
```

Replace the port placeholders. `global_segment_size: 0` makes engine clients use the separate storage client's capacity. A positive value contributes capacity from every cache-enabled rank, divided by attention TP size: a DP2 × TP2 prefill engine contributes twice the configured value in total. This is additional to the CPU HiCache pools. Store capacity is shared across namespaces; old epochs can occupy space until Store eviction. The example does not enable SSD offload, multi-node storage, or standalone dummy-client allocation.

After preparing Mini SWE inputs and output directories as in the [example](../../../examples/mini_swe_agent/README.md#train-with-cpu-caching), launch:

```bash
HICACHE_TIER=mooncake HICACHE_DEPLOYMENT=regular \
HICACHE_MOONCAKE_CONFIG=/shared/path/mooncake.json \
  bash scripts/entrypoint/ray-job.sh examples/mini_swe_agent/run_mini_swe_agent_hicache.sh
```

Choose `HICACHE_DEPLOYMENT=pd` for 2P2D TP2 or `pd-dp2` for 1P1D attention DP2 × TP2 / EP4 with DP-aware consistent hashing. The wrapper embeds the connection settings in `${EXP_ROOT}/hicache-mooncake.yaml`; Ray workers need the shared output directory, not the original connection file or submitter's environment. Regular/P engines use `hicache_storage_backend: mooncake` and `wait_complete`; D remains without HiCache. `HICACHE_TIER=cpu` remains the default.

The runtime support here targets Qwen3.5 Unified Full KV + Mamba. Both components must match the same prefix, owner and weights, on every attention TP rank. Each owner receives a new incarnation on startup. Every weight mutation, including unversioned buckets and post-processing, drains storage workers, advances the epoch, and flushes local state before modifying weights. Failure is agreed across global TP; a partial mutation fails the engine. Namespace checks compare ranks within each owner, not different DP owners. Other cache implementations do not receive this Unified namespace and are rejected before online weight mutation with Mooncake.

Mooncake L3 is rejected when `enable_lora` is set. LoRA adapter updates do not run the weight-mutation drain, so a new adapter could otherwise restore KV/Mamba state computed with the old weights. Use CPU-only HiCache or disable LoRA until both adapter-update paths perform the storage drain and namespace invalidation.

A normal GPU/CPU flush or offload/onload without a weight change preserves the L3 namespace. A restart creates a new incarnation. Global Store clearing is disabled because the service may contain other engines' data. Missing or short KV/Mamba objects cause recomputation; cancellation retains host-buffer ownership until the reader exits. Storage shutdown has a bounded join; an unresponsive reader causes failure rather than freeing a buffer still in use. This does not promise that a blocked C++ transport call can be interrupted.

To prove L3 use, first finish a request and its asynchronous backup, flush GPU/CPU cache without clearing Store, and repeat the prompt on the same owner. Require `cached_tokens_details.storage > 0`, H2D activity and agreement with an uncached reference. Check both prefill DP owners in the DP2 recipe. A file/object count or a GPU/CPU hit does not establish L3 consumption. Compare latency with cold recomputation at fixed input and capacity; cache functionality alone does not establish an end-to-end speedup.
