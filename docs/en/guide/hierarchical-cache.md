# CPU and SSD Hierarchical Cache

Multi-turn coding agents repeatedly send a growing conversation after executing tools. SGLang HiCache keeps reusable prompt state in CPU memory and can back it up to local SSD, so a later request can restore a prefix that no longer fits in GPU memory. This extends cache capacity; whether it improves a task depends on prefix reuse, memory pressure, transfer costs, and tool execution time.

The [Mini SWE cache entry point](../../../examples/mini_swe_agent/run_mini_swe_agent_hicache.sh) runs the same R2E-Gym task and GRPO training recipe as the [PD example](./pd-disaggregation.md). Both configurations use **eight GPUs, two prefill engines and two decode engines, each TP2**. The trainer shares those GPUs with rollout through `--colocate`. HiCache is enabled only on prefill engines; decode engines keep CPU/SSD caching and decode radix caching disabled.

This recipe targets Qwen3.5-35B-A3B's Unified Full KV and Mamba cache on a single node, with native Relax-managed SGLang engines and DP1. It uses the repository's patched SGLang v0.5.17. Other model/cache combinations, multi-node file storage, dynamic storage attach/detach, and Session-specific KV controls are outside this recipe.

## Cache data flow

| Tier | Location | How the next request uses it |
| --- | --- | --- |
| L1 | Prefill GPU memory | Reuse a resident prefix directly |
| L2 | Prefill CPU memory | Restore a backed-up prefix to GPU after GPU eviction |
| L3, optional | Local SSD files | Read a stored prefix into CPU memory, then restore it to GPU |

The example uses `write_through`: eligible prefix state is copied from GPU to CPU, then the file backend backs it up asynchronously when enabled. Native eviction frees GPU or host capacity as needed. Storage reads are triggered by incoming requests; no Session hints, proactive Session prefetch, or explicit demotion API is required.

Qwen3.5 needs both Full KV pages and a Mamba checkpoint for the **same prefix and weights**. A collection of Full KV files alone is insufficient. A request can restore only a compatible checkpoint available on all TP ranks. Missing or incomplete state falls back to computing the prompt. Chunked prefill also preserves the last valid checkpoint when its final short chunk creates no new Mamba checkpoint. When requests share a prefix, Mamba can track the branching point instead of the chunk end; an 8192-token chunk therefore does not guarantee an 8192-token reusable checkpoint.

## Prepare and launch

First follow the [PD environment setup](./pd-disaggregation.md#prepare-the-environment) and [Mini SWE data setup](../../../examples/mini_swe_agent/README.md). Use a dedicated single-node Ray cluster and submit from that node. The canonical Ray entry point performs cluster cleanup, so ensure the cluster is available for this run.

Set the model, local Mini SWE venv, prepared datasets/SIF images, and new output paths:

```bash
cd /path/to/Relax

export MINI_SWE_AGENT_VENV=/local/path/mini-swe-agent-venv
export MODEL_DIR=/path/to/models
export R2E_DATA_PATH=/path/to/r2e_data
export R2E_SIF_DIR=/path/to/r2e_sifs
export RUN_ID=mini-swe-cache-$(date +%Y%m%d-%H%M%S)
export SAVE_DIR=/path/to/checkpoints/${RUN_ID}
export EXP_ROOT=/path/to/experiments/${RUN_ID}
export RAY_ADDRESS=<ray-head-ip>:6379
export RAY_JOB_ADDRESS=http://<ray-head-ip>:8265
export NCCL_SOCKET_IFNAME=<network-interface>
```

For GPU+CPU caching:

```bash
HICACHE_TIER=cpu NUM_ROLLOUT=3 ROLLOUT_BATCH_SIZE=1 RAY_NO_WAIT=1 \
  bash scripts/entrypoint/ray-job.sh \
  examples/mini_swe_agent/run_mini_swe_agent_hicache.sh
```

For GPU+CPU+SSD caching, choose a **new, empty absolute directory on local SSD** that the Ray workers can access:

```bash
export RELAX_HICACHE_FILE_ROOT=/local/ssd/relax-cache/${RUN_ID}

HICACHE_TIER=ssd NUM_ROLLOUT=3 ROLLOUT_BATCH_SIZE=1 RAY_NO_WAIT=1 \
  bash scripts/entrypoint/ray-job.sh \
  examples/mini_swe_agent/run_mini_swe_agent_hicache.sh
```

Use separate `RUN_ID`, `SAVE_DIR`, and `EXP_ROOT` values for separate runs. `ssd` is the default tier. The wrapper validates the SSD root and forwards it through Ray's runtime environment; an export on the submitter alone does not configure worker processes. It then uses the existing PD driver, which owns the agent server until training ends. `NUM_ROLLOUT=3` and `ROLLOUT_BATCH_SIZE=1` mean three training iterations with one task Group and eight samples per iteration. The task retains its normal tool loop and scoring.

The original `run_mini_swe_agent.sh` and the pure PD `run_mini_swe_agent_pd.sh` remain separate entry points. The cache wrapper selects [the CPU configuration](../../../examples/mini_swe_agent/pd-hicache-cpu-2p2d-tp2.yaml) or [the SSD configuration](../../../examples/mini_swe_agent/pd-hicache-ssd-2p2d-tp2.yaml) through `--sglang-config`.

## Capacity and storage layout

These are **recipe settings**, not universal SGLang defaults:

| Setting | Value | Scope and effect |
| --- | --- | --- |
| `max_total_tokens` | `196608` | GPU token-pool capacity per engine; each TP rank stores its shard |
| `max_mamba_cache_size` | `666` | Prefill GPU Mamba cache slots |
| `page_size` | `64` | Full KV page length in tokens |
| `mamba_track_interval` | `256` | Mamba checkpoint interval in tokens |
| `chunked_prefill_size` | `8192` | Maximum prefill chunk size |
| `hicache_ratio` / `hicache_size` | `2.0` / `0` | Size host pools from the corresponding device pools; no fixed host-size override |
| `hicache_write_policy` | `write_through` | Back up eligible resident prefixes |
| `hicache_io_backend` / `hicache_mem_layout` | `direct` / `page_first_direct` | GPU↔host transfer and host layout |
| SSD prefetch policy | `wait_complete` | Wait for the native storage operation before using its result |
| File `max_size` | `34359738368` bytes (32 GiB) | File quota **per TP rank**, not per engine or run |
| File `min_free_space` | `68719476736` bytes (64 GiB) | Minimum free space on the backing filesystem |
| File `eviction_ratio` | `0.9` | Target fraction of the quota after LRU eviction |
| File `enable_metadata_cache` | `false` | Check file presence directly |

With the supplied BF16 Qwen3.5 configuration, these two host pools need about 47 GB per prefill TP rank, or about 175 GiB across four ranks. Use the actual startup allocations when budgeting a different model or dtype.

Host memory must cover **both** Full KV and Mamba pools, multiplied by four prefill TP ranks, plus weights, training offload, Ray, sandboxes, and OS memory. `hicache_ratio=2` is a ratio of pool capacity, not a two-gigabyte allocation or an aggregate machine limit. Mamba states can dominate host usage. Inspect startup allocation logs and the container's memory limit before increasing either pool; free host memory alone does not show the container limit.

Relax creates `prefill-engine-0000`, `prefill-engine-0001`, and so on beneath the run root, using each engine's logical rank. Each engine's two TP ranks share its subdirectory, while filenames and file-quota accounting distinguish their rank. The two prefill engines therefore allow up to **128 GiB of cached files in total** at the configured per-rank quota, before temporary-write and filesystem overhead. The 64 GiB free-space threshold applies to the filesystem; it is not an additional reserved allocation for each rank.

Use node-local storage with enough space and IO capacity. Putting the root on a shared filesystem changes latency and contention. The `direct` setting controls GPU/CPU transfer; it does **not** bypass the OS page cache for SSD file reads. A fast repeated file read alone does not establish physical SSD throughput.

## Weight updates, cancellation, and recovery

Each prefill engine creates a new file-cache incarnation at startup. A weight update drains active cache operations, stops storage workers, advances a shared TP weight epoch, and clears in-memory cache state before mutating weights. All TP ranks agree on the namespace and update result. A sharded weight synchronization can advance the file epoch multiple times; this backend counter is separate from the trainer’s weight-version number. Failed or partially applied updates fail the engine so it can be rebuilt; they cannot return successful service with mixed weights.

New requests cannot match files from an older weight epoch or an earlier engine incarnation. Old files can remain on disk and still count toward the native file quota until eviction. Namespace isolation is not immediate file deletion. After an engine recovers, its directory is stable but its new incarnation starts cold. Ordinary memory offload/onload without a weight change preserves the namespace; it does not guarantee that a particular L1/L2 entry survives.

Storage readers retain ownership of their buffers until IO has finished, including cancellation and read errors. Paused schedulers continue advancing cache completion events. Failed file writes release their backup ownership and leave available CPU cache usable; failed or incomplete reads recompute the prefix. A storage miss therefore changes cost, not the expected model state.

After a job and its workers have stopped, archive or remove that run's cache directory according to your storage policy. Do not delete live cache files to free space or reuse a nonempty root for a new run. The launch wrapper rejects a nonempty SSD run directory. Normal job progress, logs, agent-server cleanup, and stop commands follow the [PD guide](./pd-disaggregation.md#follow-progress-and-stop).

## Check cache use and diagnose problems

Check each prefill engine's effective arguments and startup logs: HiCache enabled, the intended host allocations, and `file` with the expected engine directory for SSD runs. Decode engines should remain without HiCache. Ray job acceptance, server readiness, and the existence of files each establish only part of the path.

For a repeated-prefix check, distinguish a GPU hit from CPU restoration and storage restoration. A native generation response can report `cached_tokens_details` for `device`, `host`, and `storage` when cache reporting is available. Combine response data with storage/transfer metrics: the useful result is a restored prefix consumed by a completed request. For hybrid state, check Full KV and Mamba together. To compare tiers, keep the GPU/host budgets, model, requests, concurrency, and training settings fixed; use the original PD configuration only after aligning those budgets.

| Symptom | Check |
| --- | --- |
| No L2/L3 hits | The reusable prefix may still be on GPU, shorter than a checkpoint, newly invalidated by weights, or routed to the other prefill engine |
| Files exist but storage hits stay zero | Check namespace, complete Full KV/Mamba state on every TP rank, and whether L1/L2 already serves the prefix |
| Host allocation failure | Sum Full KV and Mamba allocations across all prefill ranks; inspect container limits and training offload |
| File writes fail or stop | Check root permissions, per-rank quota, filesystem free space, and storage-worker errors |
| Storage restoration is slower than recomputation | Check prefix length, file IO, OS cache state, transfer time, and `wait_complete` latency |
| Startup rejects arguments | Use this repository's matching SGLang patch and the supplied YAML; do not copy research-only Session options |

For task concurrency and exports, see [Agentic Rollout](./agentic-rollout.md). For router layout and recovery, see [PD Disaggregation](./pd-disaggregation.md).
