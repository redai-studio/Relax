# Compiler Cache Reuse

Relax can reuse node-local TorchInductor and Triton compilation artifacts across training launches. A shared filesystem stores immutable cache deltas, while each GPU node restores them into a local writable directory before training actors start.

## Overview

Compiler cache reuse is opt-in. Set `RELAX_KERNEL_CACHE_DIR` to an absolute path on a filesystem visible to every node, then launch training through one of the standard entrypoints. Relax restores compatible cache files before the job starts and publishes new files periodically and when the training driver exits.

Only the Megatron training actors receive `TORCHINDUCTOR_CACHE_DIR` and `TRITON_CACHE_DIR`. Ray Serve, Transfer Queue, and SGLang processes do not write into this cache.

The cache contains generated TorchInductor and Triton files such as Python wrappers, shared objects, PTX, cubins, intermediate representations, autotuning results, and metadata. It does not contain model weights, optimizer state, data, NCCL communicators, DeepEP buffers, or in-memory CUDA state.

## Architecture

```text
┌────────────────────┐     restore      ┌────────────────────┐
│ Shared Delta Store │ ───────────────> │ Node-local Cache   │
└─────────▲──────────┘                  └─────────┬──────────┘
          │ periodic/final publish               │
          │                                      ▼
┌─────────┴──────────┐                  ┌────────────────────┐
│ Detached Cache     │ <── heartbeat ── │ Training Driver    │
│ Agent per GPU Node │                  └────────────────────┘
└────────────────────┘
```

For `ray-job.sh`, the head process verifies that all alive GPU nodes have the same build fingerprint, then starts one detached cache agent per GPU node. For local and SPMD launches, each node restores its cache before joining Ray, and the head later attaches the agents.

## Quick Start

Use a shared directory and invoke the normal training entrypoint:

```bash
export RELAX_KERNEL_CACHE_DIR="$(pwd)/.cache/relax-kernels"

bash scripts/entrypoint/ray-job.sh \
    scripts/training/sft/run-qwen3.5-397B-A17B-vl-lora-mtp-sft-128k-128xgpu.sh
```

The path must be absolute after shell expansion and must be visible from every GPU node. The feature is disabled when `RELAX_KERNEL_CACHE_DIR` is unset.

The same opt-in variable works with the other standard entrypoints:

```bash
# Local launch; the training script sources local.sh as usual.
export RELAX_KERNEL_CACHE_DIR="$(pwd)/.cache/relax-kernels"
bash scripts/training/sft/run-qwen3.5-397B-A17B-vl-lora-mtp-sft-128k-128xgpu.sh

# SPMD launch; export the variable in every pod before invoking the entrypoint.
export RELAX_KERNEL_CACHE_DIR="/shared/relax-kernels"
bash scripts/entrypoint/spmd-multinode.sh \
    scripts/training/sft/run-qwen3.5-397B-A17B-vl-lora-mtp-sft-128k-128xgpu.sh
```

## Configuration

| Environment variable | Default | Description |
|---|---:|---|
| `RELAX_KERNEL_CACHE_DIR` | unset | Absolute shared directory. Setting it enables the feature. |
| `RELAX_KERNEL_CACHE_KEY` | automatic | Graph and topology profile. The automatic value hashes the training script, extra arguments, and selected model/parallelism overrides. |
| `RELAX_KERNEL_CACHE_LOCAL_DIR` | automatic | Writable node-local root. Relax derives a stable path under `/tmp/relax-kernel-cache/`. |
| `RELAX_KERNEL_CACHE_BUILD_KEY` | empty | Optional operator-controlled build discriminator for changes not represented by the automatic fingerprint. |
| `RELAX_KERNEL_CACHE_COMPRESSION` | `none` | Delta archive compression: `none` or `gzip`. |
| `RELAX_KERNEL_CACHE_SYNC_INTERVAL_SEC` | `900` | Periodic snapshot interval. |
| `RELAX_KERNEL_CACHE_LEASE_TIMEOUT_SEC` | `600` | Time an attached agent may miss driver heartbeats before finalizing. |
| `RELAX_KERNEL_CACHE_STARTUP_TIMEOUT_SEC` | `7200` | Maximum time an unclaimed startup agent remains alive. |
| `RELAX_KERNEL_CACHE_HEARTBEAT_INTERVAL_SEC` | `30` | Driver heartbeat interval. |
| `RELAX_KERNEL_CACHE_EXIT_TIMEOUT_SEC` | `600` | Driver wait limit for final publication. |

Normally, set only `RELAX_KERNEL_CACHE_DIR`. Use a stable explicit key only when two launches intentionally share the same graph and topology:

```bash
export RELAX_KERNEL_CACHE_DIR="/shared/relax-kernels"
export RELAX_KERNEL_CACHE_KEY=example

bash scripts/entrypoint/ray-job.sh \
    scripts/training/sft/run-qwen3.5-397B-A17B-vl-lora-mtp-sft-128k-128xgpu.sh
```

::: warning
Do not reuse an explicit key after changing the model graph, LoRA targets, TP/PP/CP/EP/VPP layout, attention backend, recomputation policy, MTP settings, or token/image shape policy. Remove the override or assign a new key.
:::

## Compatibility and Lifecycle

Relax isolates cache data with two identifiers:

- The profile key covers the training script, command-line overrides, and selected model, parallelism, LoRA, attention, recomputation, and shape settings.
- The build fingerprint covers Python and compiler package versions, GPU model and compute capability, image/source revisions, and selected Megatron source files.

In `ray-job` mode, all alive GPU nodes must report the same build fingerprint. A CPU-only Ray head is not used as the compiler compatibility reference. A mismatch fails before cache restoration or training submission.

Each node restores compatible ready deltas into a local writable cache. During training, a detached agent periodically publishes completed files. On normal completion, failure propagated to the driver, or `ray job stop`, the driver requests a final snapshot and waits up to `RELAX_KERNEL_CACHE_EXIT_TIMEOUT_SEC`.

::: warning
`SIGKILL`, node loss, pod eviction, or Ray cluster termination cannot run a final hook. In those cases, only previously published periodic deltas are recoverable. Choose a shorter synchronization interval when preemption risk is high, balanced against shared-storage I/O.
:::

## Best Practices

1. Use an immutable container and consistent Torch, Triton, Transformer Engine, FLA, DeepEP, Megatron, and Relax revisions across jobs.
2. Keep the shared store on shared storage, but keep the active compiler directories node-local and writable. Do not point `TORCHINDUCTOR_CACHE_DIR` or `TRITON_CACHE_DIR` directly at shared storage.
3. Let the automatic key handle normal launches. Use `RELAX_KERNEL_CACHE_BUILD_KEY` to isolate an environment or patch difference not captured automatically.
4. Warm representative pipeline stages and dynamic-shape buckets before treating a cache as complete. A cache hit for one shape does not eliminate compilation for unseen token or image shapes.
5. Monitor shared-storage capacity and snapshot I/O before enabling short synchronization intervals on large clusters.

## Troubleshooting

### Cache restores but training still recompiles

Some recompilation is expected for unseen shapes. Also verify that training actor logs contain the resolved node-local directory. Compiler-specific variables are intentionally injected only into training actors.

### No cache is restored

Check that:

1. `RELAX_KERNEL_CACHE_DIR` is absolute and visible from every node.
2. The profile key and build fingerprint match a previously published ready delta.
3. The previous job ran long enough to publish a periodic or final snapshot.
4. Shared archives and manifests were not left with a partial suffix.

### GPU nodes report different fingerprints

Verify the container image, GPU architecture, installed compiler packages, Relax checkout, and node-local Megatron files. Do not force a shared fingerprint across incompatible nodes; correct the environment or use a separate cache namespace.

## Next Steps

- [Performance Tuning](./performance-tuning.md) — measure cold-start and steady-state throughput.
- [Debugging Guide](./debugging.md) — collect evidence when a distributed launch fails.
- [Configuration](./configuration.md) — configure Relax training jobs.
