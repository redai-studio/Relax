# Performance Tuning

A practical guide to maximizing training throughput in Relax. All parameters mentioned here are documented in the [Configuration Reference](./configuration.md).

---

## Profiling

Before tuning, identify the bottleneck. Relax provides three complementary profiling tools that cover **inference engine**, **training backend**, and **GPU memory**. All trace files are saved under `traces/<tb_experiment_name>/` by default, separated by subdirectory:

| Tool | Target | Default Output Directory | Viewer |
|---|---|---|---|
| SGLang Profiling | CUDA kernel / operator analysis for rollout inference | `traces/<tb_experiment_name>/sglang_trace/` | TensorBoard or `https://ui.perfetto.dev/` |
| Training Profiling | Operator analysis for Actor training / log-probs computation | `traces/<tb_experiment_name>/train_trace/` | TensorBoard or `https://ui.perfetto.dev/` |
| Memory Profiling | GPU memory allocation history, OOM diagnosis | `traces/<tb_experiment_name>/memory_snapshot/` | [PyTorch Memory Viz](https://pytorch.org/memory_viz) |

### Trace File Naming

- **Training traces** include `rank{global}_dp{dp}_tp{tp}_pp{pp}` in filenames, e.g. `train_overall_rank0_dp0_tp0_pp0.1713780123.pt.trace.json.gz`
- **Memory snapshots** also include rank tags, e.g. `memory_snapshot_time1713780123_rank0_dp0_tp0_pp0_snapshot.pickle`
- **SGLang traces** use `engine{i}` prefix to distinguish engine instances, e.g. `engine0-1713780123-TP-0.trace.json.gz`

### 1. SGLang Inference Profiling

Runs `torch.profiler` on all SGLang engines during rollout via the `/start_profile` and `/stop_profile` HTTP APIs. Does not interfere with training-side profiling.

**Example usage** — profile every rollout step:

```bash
python3 relax/entrypoints/train.py \
    --sglang-profile \
    --tb-experiment-name my-experiment \
    # ... other args
```

**Selective step range** — only profile steps 2, 3, 4 (start/end are both inclusive; recommended to avoid excessive trace files):

```bash
python3 relax/entrypoints/train.py \
    --sglang-profile \
    --sglang-profile-step-start 2 \
    --sglang-profile-step-end 4 \
    --tb-experiment-name my-experiment \
    # ... other args
```

You can also use `--sglang-profile-steps` to specify a non-contiguous list (takes precedence over start/end):

```bash
--sglang-profile-steps 2 5 10
```

All step parameters use **absolute rollout IDs** (0-indexed), i.e., step 0, step 1, ... regardless of `--start-rollout-id`.

**Advanced parameters**:

| Parameter | Default | Description |
|---|---|---|
| `--sglang-profile-step-start` | None | Start of the profiling rollout step range (**inclusive**, 0-indexed) |
| `--sglang-profile-step-end` | None | End of the profiling rollout step range (**inclusive**, 0-indexed). E.g. start=2, end=4 profiles steps 2, 3, 4 |
| `--sglang-profile-steps` | None | Non-contiguous step list; takes precedence over start/end |
| `--sglang-profile-num-steps` | 3 | Number of SGLang forward steps to profile per rollout. -1 profiles the entire rollout step |
| `--sglang-profile-activities` | CPU GPU | Activities to profile |
| `--sglang-profile-by-stage` | False | Profile prefill / decode stages separately |
| `--sglang-profile-with-stack` | False | Record Python call stacks |
| `--sglang-profile-record-shapes` | False | Record tensor shape information |
| `--sglang-profile-output-dir` | None | Custom output directory. Defaults to `traces/<tb_experiment_name>/sglang_trace` |

### 2. Training Profiling (PyTorch Profiler)

Profiles Actor training steps using `torch.profiler`, producing TensorBoard-compatible trace files.

**Example usage** — profile steps 2, 3, 4 (start/end are both inclusive):

```bash
python3 relax/entrypoints/train.py \
    --use-pytorch-profiler \
    --profile-step-start 2 \
    --profile-step-end 4 \
    --tb-experiment-name my-experiment \
    # ... other args
```

::: tip
`--profile-step-start` and `--profile-step-end` are both **inclusive** and represent **step offsets** from the current training launch, not absolute rollout IDs. The counter resets on checkpoint resumption. E.g. start=2, end=4 profiles steps 2, 3, 4 (3 steps).

Same inclusive semantics as `--sglang-profile-step-start/end`.
:::

**Detail flags**:

| Flag | Effect |
|---|---|
| `--profile-with-stack` | Record Python call stack in each trace event. Useful for identifying which code path triggers an expensive operation |
| `--profile-with-memory` | Track CUDA memory allocations/deallocations in the trace. Helps find memory spikes |
| `--profile-with-flops` | Estimate FLOPs for each operator. Useful for calculating hardware utilization (MFU) |

**Full example**:

```bash
python3 relax/entrypoints/train.py \
    --use-pytorch-profiler \
    --profile-target train_overall \
    --profile-step-start 2 \
    --profile-step-end 4 \
    --profile-with-stack \
    --profile-with-memory \
    --profile-with-flops \
    --tb-experiment-name my-experiment \
    # ... other args
```

::: warning
Enabling `--profile-with-stack` and `--profile-with-memory` adds overhead. Use them for diagnostic runs, not for production training.
:::

### 3. GPU Memory Profiling

Records CUDA memory allocation/deallocation history for diagnosing memory leaks and OOM issues. Automatically dumps a memory snapshot on OOM.

**Minimal usage** — enable recording and proactively dump after step 2:

```bash
python3 relax/entrypoints/train.py \
    --record-memory-history \
    --memory-snapshot-num-steps 2 \
    --tb-experiment-name my-experiment \
    # ... other args
```

**Advanced parameters**:

| Parameter | Default | Description |
|---|---|---|
| `--memory-snapshot-path` | snapshot.pickle | Snapshot filename suffix |
| `--memory-snapshot-dir` | None | Custom output directory. Defaults to `traces/<tb_experiment_name>/memory_snapshot` |
| `--memory-snapshot-num-steps` | None | Proactively dump a snapshot after the specified number of steps (0-indexed; setting 3 dumps after step 2) |
| `--memory-recorder` | torch | Backend: `torch` (PyTorch built-in) or `memray` (requires `pip install memray`) |

View snapshots: visit [PyTorch Memory Viz](https://pytorch.org/memory_viz) and drag in the generated `.pickle` file.

### Combined Usage

In practice, all three profiling tools can be enabled simultaneously for a comprehensive view. Here is a complete combined example:

```bash
python3 relax/entrypoints/train.py \
    # --- SGLang Inference Profiling ---
    --sglang-profile \
    --sglang-profile-step-start 2 \
    --sglang-profile-step-end 4 \
    # --- Training Profiling ---
    --use-pytorch-profiler \
    --profile-step-start 2 \
    --profile-step-end 4 \
    # --- Memory Profiling ---
    --record-memory-history \
    --memory-snapshot-num-steps 2 \
    # --- Experiment name (determines trace output directory) ---
    --tb-experiment-name my-profiling-run \
    # ... other training args
```

The above configuration produces the following directory structure:

```
traces/my-profiling-run/
├── sglang_trace/                          # SGLang engine traces (subdirectory per rollout step)
│   ├── rollout_2/
│   │   ├── engine0-...-TP-0.trace.json.gz
│   │   ├── engine0-...-TP-1.trace.json.gz
│   │   ├── engine1-...-TP-0.trace.json.gz
│   │   └── ...
│   ├── rollout_3/
│   │   └── ...
│   └── rollout_4/
│       ├── engine0-...-TP-0.trace.json.gz
│       └── ...
├── train_trace/                           # Training traces
│   ├── train_overall_rank0_dp0_tp0_pp0.....pt.trace.json.gz
│   ├── train_overall_rank1_dp0_tp1_pp0.....pt.trace.json.gz
│   └── ...
└── memory_snapshot/                       # Memory snapshots
    ├── memory_snapshot_time..._rank0_dp0_tp0_pp0_snapshot.pickle
    ├── memory_snapshot_time..._rank1_dp0_tp1_pp0_snapshot.pickle
    └── ...
```

---

## Dynamic Batching

Dynamic batching packs variable-length samples so each micro-batch approaches a target token count, improving GPU utilization compared to fixed-size micro-batches. It also serves as an effective OOM prevention mechanism — with a fixed `--micro-batch-size`, a batch of unusually long sequences can exceed GPU memory, while dynamic batching caps the total tokens per micro-batch to `--max-tokens-per-gpu`, keeping memory usage predictable.

```bash
--use-dynamic-batch-size \
--max-tokens-per-gpu 9216
```

When using Context Parallelism (CP), set `--max-tokens-per-gpu` to approximately `max_response_len / cp_size`.

If computing log probs is a separate bottleneck, you can set a different token budget for that phase:

```bash
--log-probs-max-tokens-per-gpu 12288
```

::: tip
If you experience OOM during training, switching from fixed `--micro-batch-size` to `--use-dynamic-batch-size` with a conservative `--max-tokens-per-gpu` is often the first step. See [OOM Troubleshooting](./oom-troubleshooting.md) for more details.
:::

---

## Parallelism Configuration

### Tensor and Sequence Parallelism

For models that fit on a single node, Tensor Parallelism (TP) + Sequence Parallelism (SP) is the most common setup:

```bash
--tensor-model-parallel-size 2 \
--sequence-parallel
```

Larger models (30B+) typically use TP=2 or TP=4 with SP enabled.

### MoE Expert Parallelism

For Mixture-of-Experts models (e.g., Qwen3-30B-A3B), distribute experts across GPUs:

```bash
--expert-model-parallel-size 2 \
--expert-tensor-parallel-size 1
```

### Context Parallelism

For long-context training, split the sequence across GPUs:

```bash
--context-parallel-size 2
```

---

## Activation Recomputation

Recomputation trades compute for memory. For most RL training workloads, enabling recomputation is recommended:

```bash
--recompute-granularity full \
--recompute-method uniform \
--recompute-num-layers 1
```

This configuration recomputes activations uniformly across layers. Adjust `--recompute-num-layers` based on your memory/compute tradeoff. See [Megatron-LM documentation](https://github.com/NVIDIA/Megatron-LM) for details on `selective` granularity and `block` method.

---

## Multimodal Processing Parallelism

When training multimodal models (e.g., Qwen3-VL), the HuggingFace processor for image/video data can become a CPU bottleneck due to Python's GIL. The `--mm-processor-pool-size` parameter creates a `ProcessPoolExecutor` to bypass GIL contention:

```bash
--mm-processor-pool-size 4
```

| Value | Behavior |
|---|---|
| `0` (default) | Uses `ThreadPoolExecutor` — subject to GIL contention |
| `> 0` | Creates a `ProcessPoolExecutor` with the specified number of workers for true CPU parallelism |

::: tip
Start with a pool size equal to the number of CPU cores available per GPU. For example, on a node with 64 CPUs and 8 GPUs, try `--mm-processor-pool-size 8`.
:::

---

## SGLang Inference Engine Tuning

### Memory Allocation

Control how much GPU memory SGLang reserves for KV cache:

```bash
--sglang-mem-fraction-static 0.8
```

Typical values range from 0.6 to 0.85. In colocate mode, a higher value (0.8) leaves less room for training but improves inference throughput. In fully async mode with separate GPUs, you can push this higher.

### Inference TP Size

Set the number of GPUs per inference engine instance:

```bash
--rollout-num-gpus-per-engine 1
```

For large models, increase this to match the model's minimum TP requirement. Using TP=1 for inference when possible gives the best per-query throughput.

---

## Partial Rollout

In long response scenarios (e.g., code generation, chain-of-thought reasoning), waiting for all samples to fully complete generation can leave training GPUs idle for extended periods. Partial Rollout avoids this by allowing incomplete samples to be interrupted and recycled back to the data buffer, so training can proceed with the samples that have finished:

```bash
--partial-rollout
```

### Controlling Abort Frequency

By default, a sample can be aborted an unlimited number of times. Set `--partial-rollout-max-aborted-count` to guarantee that a sample eventually completes generation after being aborted a certain number of times:

```bash
--partial-rollout \
--partial-rollout-max-aborted-count 3
```

### On-Policy Masking

When a sample is recycled and continues generation in a later rollout, its earlier tokens were generated under a previous policy version. Use `--mask-offpolicy-in-partial-rollout` to mask those off-policy tokens so only on-policy generated tokens participate in training:

```bash
--partial-rollout \
--mask-offpolicy-in-partial-rollout
```

::: tip
Partial Rollout is most effective for workloads where `max_response_len` is large (e.g., 8K+) and response lengths vary significantly. For short, uniform-length responses, the overhead of recycling may not be worthwhile.
:::

---

## Data Loading Optimization

### Streaming Dataset

For very large datasets that don't fit in memory, use streaming mode:

```bash
--use-streaming-dataset \
--streaming-buffer-size 10000
```

### Data Balancing

Distribute token counts evenly across data parallel ranks to reduce idle time:

```bash
--balance-data
```

On the static/seqlen-balanced path, `--balance-data` enables `SeqlenBalancedSampler` and uses `karmarkar_karp` to balance tokens across data parallel ranks.

In fully async with `--use-dynamic-batch-size`, the dynamic batch path automatically performs DP token balancing through `StreamingTokenBudgetSampler`. Passing `--balance-data` is accepted there, but it does not add another balancing pass.

::: warning
With `--balance-data`, different responses for the same prompt may be assigned to different training steps.
:::

---

## Weight Update Pipeline

For MoE models with large parameter counts, chunk the weight update to avoid memory spikes:

```bash
--update-weight-buffer-size 536870912  # 512 MB
```

Reduce this value if you observe memory pressure during weight synchronization.

---

## Fully Async Training

For maximum throughput, use the fully async training pipeline with dedicated GPU clusters for training and rollout:

```bash
--fully-async \
--max-staleness 1 \
--num-data-storage-units 1
```

### Staleness Tuning

The `--max-staleness` parameter controls how many versions behind the rollout data can be relative to the current training model. It directly affects the tradeoff between throughput and data freshness:

| Value | Behavior |
|---|---|
| `1` (default) | Training consumes only data from the current or previous version. Lower throughput but fresher data |
| `2-3` | Allows moderately stale data. Good balance for most workloads |
| Higher | Higher throughput by reducing training idle time, but data may be generated under an older policy |

::: tip
Start with `--max-staleness 1` and increase if you observe the training process frequently waiting for fresh rollout data. Monitor training loss stability — if loss becomes unstable with higher staleness, reduce the value.
:::

See [Fully Async Training](./fully-async-training.md) for the complete setup guide.

---

## Recommended Configurations

### Qwen3-4B on 8 GPUs (Colocate)

```bash
--tensor-model-parallel-size 2 \
--sequence-parallel \
--recompute-granularity full \
--recompute-method uniform \
--recompute-num-layers 1 \
--use-dynamic-batch-size \
--max-tokens-per-gpu 9216 \
--sglang-mem-fraction-static 0.8 \
--colocate
```

### Qwen3-30B-A3B on 16 GPUs (Fully Async)

```bash
--tensor-model-parallel-size 2 \
--sequence-parallel \
--expert-model-parallel-size 2 \
--recompute-granularity full \
--recompute-method uniform \
--recompute-num-layers 1 \
--optimizer-cpu-offload \
--sglang-mem-fraction-static 0.6 \
--fully-async
```

---

## GPU Utilization After Tuning

The table below summarizes measured GPU utilization for each launch script after performance tuning. Use it as a health baseline for the corresponding configuration.

| Task | GPU UTIL (%) (after tuning) | GPU SM (%) (after tuning) |
|---|---|---|
| `run-qwen3-4B-8xgpu` | 87 | 61 |
| `run-qwen3-30B-A3B-int4-8xgpu` | 71 | 47 |
| `run-qwen3-vl-4B-8xgpu` | 70 | 32 |
| `run_deepeyes` | 66 | 32 |
| `run-qwen35-9B-8xgpu-openr1mm-hybrid-async` | 69 | 42 |
| `run-qwen35-35B-A3B-16xgpu` | 64 | 35 |
| `run-qwen35-9B-8xgpu` | 78 | 64 |
| `run-qwen36-35B-A3B-8xgpu-image` | 63 | 35 |

---

## Next Steps

- [OOM Troubleshooting](./oom-troubleshooting.md) — when tuning causes memory issues
- [Configuration Reference](./configuration.md) — full parameter list
- [Debugging Guide](./debugging.md) — isolating training and inference issues
