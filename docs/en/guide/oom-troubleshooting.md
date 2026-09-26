# OOM Troubleshooting

A practical guide to diagnosing and resolving CUDA Out-Of-Memory (OOM) errors during Relax RL training. All parameters mentioned here are documented in the [Configuration Reference](./configuration.md).

---

## Diagnosing OOM

### Step 1: Determine Where OOM Occurs

OOM errors in Relax typically happen in one of these phases:

| Phase | Symptom | Typical Cause |
|---|---|---|
| Model initialization | OOM during startup | Model too large for available GPUs |
| Forward pass (training) | OOM during `train_actor` | `--max-tokens-per-gpu` too high or recomputation not enabled |
| Log probs computation | OOM during `train_log_probs` | Long sequences consuming excessive activation memory |
| Weight synchronization | OOM during `update_weights` | Weight buffer too large for remaining GPU memory |
| NCCL communication | OOM inside all-reduce/all-gather | Insufficient memory for communication buffers |

### Step 2: Capture Memory Snapshots

Relax provides built-in memory profiling tools to capture detailed allocation information.

#### PyTorch Memory Snapshot

Record memory history and dump snapshots automatically:

```bash
python3 relax/entrypoints/train.py \
    --memory-snapshot-dir /path/to/snapshots \
    --memory-snapshot-num-steps 3 \
    --memory-recorder torch \
    # ... other args
```

This records memory allocation history from the start and dumps a snapshot after the specified number of steps. If an OOM occurs, a snapshot is automatically dumped at the point of failure.

Visualize the snapshot using [PyTorch Memory Visualizer](https://pytorch.org/memory_viz):

```bash
python -m torch.utils.viz._memory_viz trace_plot /path/to/snapshots/*.pickle -o memory_trace.html
```

#### Memray Profiler

For CPU+GPU memory profiling, use the memray recorder:

```bash
python3 relax/entrypoints/train.py \
    --memory-snapshot-dir /path/to/snapshots \
    --memory-snapshot-num-steps 3 \
    --memory-recorder memray \
    # ... other args
```

::: warning
`--memory-snapshot-num-steps` is required when using memray.
:::

### Step 3: Enable NCCL Communication Memory Check

When OOM occurs inside NCCL collective operations, standard stack traces may not show how much memory was available. The `--enable-cuda-memory-check` flag adds memory monitoring around every low-level NCCL call:

```bash
python3 relax/entrypoints/train.py \
    --enable-cuda-memory-check \
    # ... other args
```

When enabled:

- Before each NCCL call (all-reduce, all-gather, broadcast, etc.), available GPU memory is checked.
- If free memory is below 5 GB, `torch.cuda.empty_cache()` is called automatically to reclaim fragmented memory.
- If the NCCL call fails, memory information is attached to the exception for diagnosis.

::: tip
`--enable-cuda-memory-check` introduces approximately **20% training performance degradation**. Use it during debugging only, not for production training.
:::

### Step 4: Use Profiler Memory Tracking

For finer-grained analysis, enable memory tracking in the PyTorch Profiler:

```bash
python3 relax/entrypoints/train.py \
    --profile-target train_overall \
    --profile-step-start 2 \
    --profile-step-end 4 \
    --profile-with-memory \
    --use-tensorboard \
    --tb-project-name /path/to/tb_logs \
    # ... other args
```

The `--profile-with-memory` flag records CUDA memory allocations/deallocations in the profiler trace, visible in the TensorBoard Memory view.

---

## Common Solutions

### 1. Enable Activation Recomputation

The most effective way to reduce training memory. Trades compute time for memory:

```bash
--recompute-granularity full \
--recompute-method uniform \
--recompute-num-layers 1
```

This is used in all standard Relax training scripts and is highly recommended.

### 2. Reduce Max Tokens Per GPU

Lower `--max-tokens-per-gpu` to reduce the token count packed into each micro-batch:

```bash
# Before (OOM)
--max-tokens-per-gpu 12288

# After (reduced)
--max-tokens-per-gpu 8192
```

For log probs computation, set a separate budget if needed:

```bash
--log-probs-max-tokens-per-gpu 8192
```

### 3. Enable Dynamic Batching to Prevent OOM

With a fixed `--micro-batch-size`, a batch of unusually long sequences can exceed GPU memory. Dynamic batching caps total tokens per micro-batch, keeping memory usage predictable and preventing OOM from variable-length inputs:

```bash
--use-dynamic-batch-size \
--max-tokens-per-gpu 8192
```

Start with a conservative `--max-tokens-per-gpu` and increase gradually. This replaces `--micro-batch-size` — when dynamic batching is enabled, micro-batch size is determined automatically based on the token budget.

### 4. Use Fixed Micro-Batch Size

If you are already using dynamic batching and still experiencing OOM, the `--max-tokens-per-gpu` value may be too high. Alternatively, you can switch to a fixed micro-batch size with shorter sequences:

```bash
# Remove --use-dynamic-batch-size and set explicitly
--micro-batch-size 1
```

### 5. Enable Optimizer CPU Offload

Move optimizer states (Adam moments) to CPU memory. Critical for large models (30B+):

```bash
--optimizer-cpu-offload
```

For better performance with CPU offload, overlap data transfers:

```bash
--optimizer-cpu-offload \
--overlap-cpu-optimizer-d2h-h2d
```

### 6. Recompute Loss Function

Save memory by recomputing the loss function instead of caching intermediate results:

```bash
--recompute-loss-function
```

### 7. Chunk Log Probs Computation

Split log probs computation into smaller chunks to reduce peak memory:

```bash
--log-probs-chunk-size 4
```

A value of `-1` (default) computes all at once. Smaller values use less memory but take longer.

### 8. Reduce Weight Update Buffer

For MoE models with many parameters, the weight update buffer can consume significant memory. Reduce the buffer size:

```bash
# Default is 512 MB
--update-weight-buffer-size 268435456  # 256 MB
```

### 9. Disable Weights Backuper

The weights backuper keeps a copy of model weights in host memory for recovery. Disabling it saves host memory:

```bash
--disable-weights-backuper
```

::: warning
Disabling the weights backuper means automatic weight recovery is unavailable if training fails.
:::

### 10. Adjust Training Memory Margin

Relax reserves memory to prevent fragmentation. Adjust the margin:

```bash
# Default is 1 GB (1073741824 bytes)
--train-memory-margin-bytes 536870912  # 512 MB
```

### 11. Tune SGLang Memory Fraction

In colocate mode, SGLang and training share GPU memory. Reduce SGLang's allocation to leave more room for training:

```bash
# Default varies; typical values
--sglang-mem-fraction-static 0.7  # down from 0.8
```

### 12. Rebalance Pipeline Parallel Layers Across Stages

When PP is enabled and you see clearly uneven per-stage memory usage, the last stage is usually the heaviest — it carries extra cost from loss computation and output embedding. Shift layers onto the first stage to relieve the last stage:

```bash
# Example: Qwen3.5 35B has 40 layers, PP=2
# Even split assigns 20 layers per stage; the last stage tends to OOM
--decoder-first-pipeline-num-layers 30
```

`--decoder-last-pipeline-num-layers` is also available if you'd rather reduce the last stage explicitly. Adjust in small steps — move 1–2 layers at a time and watch the per-stage memory water mark.

### 13. Enable `expandable_segments` (Requires `--selective-offload`)

`expandable_segments:True` reduces fragmentation, but it is **incompatible with `torch_memory_saver`** — the default mechanism behind `--offload-train`. TMS cannot track VMM-backed expandable segments, and because its hook is armed from `TMS_INIT_ENABLE` inside the preloaded `.so` before any Python code runs, its own guard never executes. You get a raw driver error instead of a readable message:

```
[torch_memory_saver.cpp] CUresult error: 1 (invalid argument) file=csrc/utils.h func=cu_mem_create line=169
```

Two conditions make it usable:

1. **Use `--selective-offload`** — an application-level offload that never loads the `torch_memory_saver` hook, so the conflict disappears.
2. **Scope the variable to the training actors with `--train-env-vars`**, not `RUNTIME_ENV_JSON`:

```bash
--selective-offload \
--train-env-vars '{"PYTORCH_CUDA_ALLOC_CONF":"expandable_segments:True"}' \
```

`--train-env-vars` is merged only into the training actors' environment and never into the rollout environment, so SGLang keeps using `torch_memory_saver` untouched.

::: danger
Do **not** put `expandable_segments:True` in `RUNTIME_ENV_JSON` — that env is global and reaches the rollout workers too. SGLang's `release_memory_occupation` / `resume_memory_occupation` (enabled by `--offload-rollout`) rely on `torch_memory_saver` and have **no fallback**, so inference offload breaks.
:::

::: tip
On torch 2.9, `PYTORCH_CUDA_ALLOC_CONF` is deprecated in favour of `PYTORCH_ALLOC_CONF`. Note that `torch_memory_saver`'s own sanity check only inspects the legacy name, so the new name gets no guard at all.
:::

Colocate weight update ships tensors to SGLang over CUDA IPC, which for expandable-segment tensors goes through `pidfd_open`/`pidfd_getfd`. This works on a modern kernel with normal ptrace permissions; if a hardened container blocks it you would see `Tensors allocated with expandable_segments:True cannot be shared between processes`.

---

## OOM in Specific Phases

### Training Forward Pass OOM

**Symptoms**: OOM during `train_actor` step.

**Checklist**:
1. Enable recomputation: `--recompute-granularity full --recompute-method uniform --recompute-num-layers 1`
2. Enable `--use-dynamic-batch-size` with a conservative `--max-tokens-per-gpu`
3. Lower `--max-tokens-per-gpu`
4. Enable `--recompute-loss-function`
5. Try `--optimizer-cpu-offload`
6. If PP is enabled and the last stage is clearly heavier, use `--decoder-first-pipeline-num-layers` to shift layers onto the first stage

### Log Probs Computation OOM

**Symptoms**: OOM during `train_log_probs` step.

**Checklist**:
1. Set a lower `--log-probs-max-tokens-per-gpu` (separate from training)
2. Use `--log-probs-chunk-size 4` to chunk the computation
3. Lower `--max-tokens-per-gpu`

### Weight Synchronization OOM

**Symptoms**: OOM during `update_weights` (weight transfer from training to inference).

**Checklist**:
1. Reduce `--update-weight-buffer-size`
2. In colocate mode, reduce `--sglang-mem-fraction-static`
3. Try `--disable-weights-backuper`

### NCCL Communication OOM

**Symptoms**: OOM inside NCCL calls with opaque stack traces.

**Checklist**:
1. Enable `--enable-cuda-memory-check` to get detailed memory info at failure point
2. Capture memory snapshot with `--memory-snapshot-dir` and `--memory-recorder torch`
3. Reduce `--train-memory-margin-bytes` if memory is over-reserved
4. Increase `--train-memory-margin-bytes` if fragmentation is the issue

---

## Quick Reference

| Goal | Parameter |
|---|---|
| Reduce activation memory | `--recompute-granularity full --recompute-method uniform --recompute-num-layers 1` |
| Cap per-batch tokens (OOM prevention) | `--use-dynamic-batch-size --max-tokens-per-gpu <value>` |
| Reduce per-batch memory | `--max-tokens-per-gpu <lower value>` |
| Move optimizer to CPU | `--optimizer-cpu-offload` |
| Recompute loss | `--recompute-loss-function` |
| Chunk log probs | `--log-probs-chunk-size 4` |
| Reduce weight sync memory | `--update-weight-buffer-size <lower value>` |
| Save host memory | `--disable-weights-backuper` |
| Debug NCCL OOM | `--enable-cuda-memory-check` |
| Capture memory snapshot | `--memory-snapshot-dir <path> --memory-recorder torch` |
| Profile memory usage | `--profile-with-memory` |
| Adjust memory margin | `--train-memory-margin-bytes <bytes>` |
| SGLang memory (colocate) | `--sglang-mem-fraction-static <fraction>` |
| Rebalance PP stage layers | `--decoder-first-pipeline-num-layers <count>` |
| Reduce fragmentation | `--selective-offload` + `--train-env-vars '{"PYTORCH_CUDA_ALLOC_CONF":"expandable_segments:True"}'` |

---

## Next Steps

- [Performance Tuning](./performance-tuning.md) — maximize throughput after resolving OOM
- [Configuration Reference](./configuration.md) — full parameter list
- [Debugging Guide](./debugging.md) — isolating training and inference issues
