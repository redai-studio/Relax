# Bounded synchronous checkpoint staging

Adapted from `fix/wuhuan/k3_ckpt_oom` (commit `399fcb8e`), rebased onto the
current Relax Megatron patch. Apply this patch **after** the versioned Megatron
patch; both CUDA Dockerfiles do this automatically.

Synchronous `torch_dist` saves previously used `AsyncRequest.execute_sync` to
copy all local checkpoint tensors to CPU before writing. The new synchronous
writer leaves source tensors in place, stages a bounded window, writes buckets
sequentially, and releases each staged tensor after serialization. Training
remains blocked until writing and distributed finalization finish. DCP planning,
metadata, load format, and asynchronous saving behavior are unchanged.

`MEGATRON_SYNC_SAVE_BOUNDED_STAGING` selects the synchronous save implementation:

- Unset or `0` (default): retain the original full CPU preload and multithreaded
  writer. The staging budget is ignored.
- `1`: explicitly opt into bounded staging.
- Other values are rejected. Asynchronous saves do not consult this switch.

Set the same value on every training rank before launch. The K3 GRPO recipe
explicitly sets `MEGATRON_SYNC_SAVE_BOUNDED_STAGING=1` in its Ray runtime
environment and forwards both variables to training actors. To roll back that
recipe, change its explicit value to `0`; the Megatron patch itself remains
disabled by default.
`MEGATRON_SYNC_SAVE_STAGE_BYTES=0` alone does **not** select the old implementation.

`MEGATRON_SYNC_SAVE_STAGE_BYTES` sets the live staging budget **per rank**:

- Default: `1073741824` (1 GiB), with a producer thread overlapping copies and I/O.
- `0`: stage and write one tensor at a time, without prefetch or pinned staging.
- Negative values are rejected.
- A tensor larger than the budget is admitted only when the window is empty.
  Live staging is therefore bounded by `max(budget, largest_tensor)`, not strictly
  by the configured budget. Eight ranks per node multiply this allowance by eight.

The bound covers live staging tensors, including the tensor currently being
written. It does **not** bound total RSS, pinned allocator caches, existing
optimizer/offload state, metadata, serialization scratch, or filesystem page
cache. Quantized CUDA tensors are dequantized before accounting. CPU views are
compacted to avoid serializing their entire backing allocation.

Compared with the reference branch, producer startup and cleanup exceptions now
reach the same results queue as write failures, with the first failure preserved.
Non-`Exception` failures in the producer thread are converted to `RuntimeError`.
This keeps surviving ranks on the normal collective finalization path; it cannot
recover from process death or a broken process group.

## Validation and deployment

Run `pytest -q tests/backends/megatron/test_sync_checkpoint_staging.py` with the
supported Megatron/PyTorch installation. Tests apply the patch to a temporary
source copy, never to the installed training runtime. CUDA cases require an idle
GPU; hide GPUs with `CUDA_VISIBLE_DEVICES=''` for CPU-only checks.

CPU tests exercise real DCP save/load, compact views, optimizer restoration and
next update, cached plans, actual live tensor lifetimes, oversized items, and
copy/write/start/cleanup failures. CUDA tests additionally cover D2H and caller
stream ordering. These do not substitute for a full-size, multi-node save and
reload with per-node memory measurements.

Existing running jobs retain their imported code. Deploy through a rebuilt image
or apply the patch consistently on all nodes before starting a new job; changing
files underneath a running training job does not reliably change its writer.
