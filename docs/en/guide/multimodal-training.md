# Multimodal Training Optimizations

Multimodal training spends time reading and decoding media, running processors, transferring data, and computing model outputs. SFT and RL produce data through different paths. Identify the expensive stage before choosing an optimization.

| Scope | Optimization | Purpose |
|---|---|---|
| Shared: SFT / RL | Bound media size and locate CPU / storage / GPU bottlenecks | Reduce unnecessary processing and compute before increasing concurrency |
| SFT: Megatron | Automatic prefetch and rank-side image reconstruction | Overlap next-step preparation with training and reduce pixel transfer through TransferQueue |
| RL: built-in SGLang rollout | Multiprocess processor | Parallelize rollout-side multimodal processing to relieve CPU bottlenecks |

## Shared: Identify Bottlenecks and Data Budgets

These principles apply to both SFT and RL. Supported media types, resolutions, and preprocessing still depend on the model implementation.

- **Bound media size per sample.** Image count, resolution, and video frame count affect CPU processing, transfer, and GPU compute. Before changing image token limits or video sampling, verify model support and evaluate task quality; these changes are not necessarily lossless optimizations.
- **Separate reading, preprocessing, and compute.** More processor workers may not help slow storage; a deeper prefetch window cannot raise production throughput when CPUs are saturated. Compare end-to-end time, GPU waiting, CPU utilization, and storage throughput first.
- **Budget for the actual process count.** Multiprocess processing adds processor copies and memory use. SFT producers, training ranks, and RL rollout workers have separate pools; their settings are not interchangeable.

See [Performance Tuning](./performance-tuning.md) for general compute and memory optimization. The following sections distinguish the SFT and RL data paths.

## SFT: Automatic Prefetch and Rank-Side Image Reconstruction

This section applies to **Megatron SFT**. Image-reference transport requires cooperation between the SFT producer and training ranks; it does not automatically apply to RL rollout.

### Automatic Selection and Usage

Megatron SFT automatically lets each rank fetch from TransferQueue. Users set the in-flight step budget, and the framework selects preparation according to the data and backend capabilities:

| Condition | Automatically selected path |
|---|---|
| Only one step allowed in flight | Synchronous fetch with the original multimodal payload |
| At least two steps in flight and `--multimodal-keys` contains `image` | Next-step prefetch; eligible image samples are reconstructed on training ranks |
| At least two steps in flight, no `image`, and async prepack constraints are met | Background fetch, microbatch partitioning and packing, plus early device preparation for the first microbatch |
| At least two steps in flight but neither case above applies | Raw data prefetch |

Async prepack currently requires NCCL, THD, PP=1, CP=1, no VPP, no dynamic CP, and no routing/indexer replay. Image data takes the image-prefetch path. Users do not need to combine per-rank fetch, train-data-prefetch, or async prepack switches.

Add the image-column mapping and budget to an existing SFT launch configuration:

```bash
--multimodal-keys '{"image":"images"}' \
--sft-max-in-flight-steps 2
```

This is an argument fragment, not a complete launch command; see [SFT Training](./sft-training.md) for the full configuration. The budget includes the current training step. When unset, it follows `--max-staleness`, whose default of 0 corresponds to one step. A larger budget consumes more buffering resources and does not guarantee higher throughput.

### How Image-Reference Transport Works

The producer still runs the full processor to produce tokens, loss masks, and image grids, and performs length checks. For reconstructable images, transport replaces `pixel_values` with file references and shape metadata. Training ranks read the same files and rebuild pixels in the background.

```text
SFT producer                         Vision training rank
  full processor                       background prefetch
  tokens / masks / grid ── TQ ──────>   read shared image files
  image refs / pixel shape ─────────>   rebuild pixels on CPU
                                        validate grid and shape
                                        attach pixels before training
```

This reduces pixel transfer and overlaps rank-side reconstruction with training; it does not remove producer-side image processing. It also adds CPU work and image-storage reads on training nodes.

The consumer does not regenerate tokens or loss masks. Reconstruction currently covers the Kimi K3 SFT image path and the generic HF image processor path; pixels are replaced only when the output has a recognized `pixel_values` / `image_grid_thw` structure. Only vision ranks whose model chunks have `pre_process` capability rebuild images; other PP stages retain metadata.

### Input Formats and Automatic Fallback

File references must resolve to the same readable images on every training node. Prefer absolute paths on shared storage when preparing data, for example:

```json
{
  "messages": [
    {"role": "user", "content": "<image>\nDescribe this image."},
    {"role": "assistant", "content": "A cat sitting on a chair."}
  ],
  "images": ["/shared/data/images/cat.jpg"]
}
```

| Input form | SFT image transport behavior |
|---|---|
| File-path string or a path-only `{"path": "..."}` object | Sends references when reconstruction conditions are met |
| `data:image/...;base64,...`, `{"base64": "..."}`, raw bytes, or an object containing bytes | Automatically retains the sample's pixel payload |
| Bare base64 string without a data URI prefix | Interpreted as a file path, not detected as base64; use an inline format above |
| Processor output without a reconstructable image structure | Retains the original payload |
| Video or audio fields | Do not use this image-reference optimization; media support still depends on the model and data-processing path |

Fallback applies **per sample**: if any image cannot be represented as a file reference, that sample keeps its image pixels. Other samples in the same batch can still use references. No unsupported-input policy or manual pixel-fallback switch is required.

File errors depend on the stage. Producer read or processing failures follow `--sft-invalid-multimodal-strategy` (`error` by default, with `skip` available). After the producer succeeds, a training-rank read failure or a rebuilt grid / pixel-shape mismatch aborts training rather than silently falling back.

### Performance Observation and Troubleshooting

| Metric | Meaning |
|---|---|
| `sft_rank_image_read` | Image reconstruction read and decode time, summed across samples |
| `sft_rank_image_process` | Image reconstruction processor time, summed across samples |
| `sft_train_prefetch_wait` | Training-thread wait for the prefetch future, potentially including image reconstruction |
| `per_rank_fetch_time` | TransferQueue fetch time, excluding subsequent image reconstruction |

Reconstruction tasks run concurrently, so the first two metrics can sum to more than elapsed wall-clock time. Evaluate full training-step time rather than TQ traffic reduction or one rank's fetch time alone.

- **Rank-side read failure:** Check mount paths, permissions, and file contents on every training node. A successful producer read does not verify other nodes' mounts.
- **Grid / shape mismatch:** Check that producer and training nodes use matching model, processor, and image-processing configuration, especially image token limits.
- **Slow producer:** Check invalid samples, disk reads, and CPU utilization before adjusting producer `--sft-prefetch-num-workers`, `--sft-prefetch-chunk-size`, or the in-flight budget. Small chunks limit the number of samples processed concurrently.
- **Slow rank-side reconstruction:** Check CPU and shared-storage load on training nodes. Each rank performing reconstruction currently uses a fixed pool of 8 processor processes; producer `--sft-prefetch-num-workers` does not resize this pool.

## RL: Rollout-Side Multimodal Preprocessing

The built-in SGLang rollout processes multimodal inputs before generation requests and prepares multimodal tensors for training. Its processor parallelism is independent of SFT image-reference transport. Whether a custom rollout uses this path depends on its implementation.

When the rollout-side HF processor is the bottleneck and CPU and memory have headroom, configure a process pool, for example:

```bash
--mm-processor-pool-size 4
```

The default, `0`, uses threaded execution. A positive value creates that many processor subprocesses per rollout worker process. No pool is created without a processor object; pool-creation failures log a warning and fall back to threads. Multiply by the actual rollout worker count when budgeting resources; this setting is not a global process limit.

This parameter does not control the SFT producer or SFT rank-side image reconstruction, nor does it directly change SGLang GPU parallelism. Measure CPU preprocessing and inference-engine time separately; see [Performance Tuning](./performance-tuning.md#sglang-inference-engine-tuning) for the latter.

## Next Steps

- [SFT Training](./sft-training.md)
- [Dataset Design](./dataset-design.md)
- [Performance Tuning](./performance-tuning.md)
