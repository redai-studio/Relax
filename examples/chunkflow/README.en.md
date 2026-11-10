# ChunkFlow

[中文](README.md) · [Full guide](../../docs/en/guide/chunkflow.md)

Split long sequences into chunks, reuse native Attention / GDN computation, and connect cross-chunk states and gradients. **Start with long-context, memory-constrained training; speedups depend on the model and configuration.**

## Measured benefits

These completed short runs are configuration-specific. Each pair shares data and training settings, differing only in the ChunkFlow switch and tracking name; Dense K1 additionally activates ChunkFlow recomputation. Times average steps 1–29, excluding step 0. Memory is the maximum across all steps and ranks. Speedup = Baseline / ChunkFlow; below 1 means slower. “—” means unverified, not an estimate.

### Dense · Qwen3-4B

**8 GPUs · TP2 / PP4 · GBS1 · LongAlign · 64K limit · chunk4096 · K1 · 30 steps**. No native recomputation. This historical K1 comparison does not establish K0 performance.

| Method        | Mean train time | Mean step time | MFU | Training speedup | End-to-end speedup | Peak allocated / reserved | Allocated saving |
| ------------- | --------------: | -------------: | --: | ---------------: | -----------------: | ------------------------: | ---------------: |
| Baseline      |          3.89 s |              — |   — |            1.00× |                  — |         52.28 / 79.03 GiB |                — |
| **ChunkFlow** |      **2.20 s** |              — |   — |        **1.76×** |                  — |     **21.01 / 22.58 GiB** |       **59.81%** |

Mean relative loss difference: **0.51%**. This configuration improves time and memory; do not extrapolate to larger batches or K0.

### MoE · Qwen3.6-35B-A3B

**8 GPUs · TP2 / PP4 / EP2 · GBS8 · LongAlign · 64K limit · chunk4096 · K0 · 30 steps**. Both arms use native selective `gdn moe` recomputation; ChunkFlow recomputation is off.

| Method        | Mean train time | Mean step time | MFU | Training speedup | End-to-end speedup | Peak allocated / reserved | Allocated saving |
| ------------- | --------------: | -------------: | --: | ---------------: | -----------------: | ------------------------: | ---------------: |
| Baseline      |         15.99 s |              — |   — |            1.00× |                  — |         56.38 / 70.26 GiB |                — |
| **ChunkFlow** |     **18.90 s** |              — |   — |        **0.85×** |                  — |     **37.79 / 40.58 GiB** |       **32.97%** |

Mean relative loss difference: **0.39%**. This configuration saves **18.59 GiB**, at approximately **18.2%** more training time: a memory tradeoff, not a speedup.

Use **allocated** for the primary memory comparison; reserved includes allocator caching. Close short-run losses do not establish gradient parity or long-term convergence.

## When to enable

| Scenario                                              | Recommendation                                                                           |
| ----------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| Long-context Dense, small batches, pipeline idle time | Benchmark it; the historical K1 configuration benefited                                  |
| Long-context MoE with memory pressure                 | Try it if additional compute time is acceptable                                          |
| Short sequences, ample memory, throughput first       | Leave off unless a matched comparison shows a benefit                                    |
| QSA / PLE models                                      | Experimental: prefix replay repeats computation; savings and speedups are not guaranteed |

The current path requires **packed text, CP1, zero dropout, disabled RoPE fusion, and per-token loss**. PP4 requires experimental opt-in, already set by the recipes. CP>1, VPP, multimodal inputs, full-layer and selective `core_attn` recomputation are not directly supported. QSA only allows native selective `moe` recomputation.

Qwen3.8-Flash-Next completed a 10-step, 8K-limit K0 comparison on a length-filtered subset: allocated memory decreased only 2.42%, with no observed speedup. The 32K comparison on original LongAlign remains incomplete; it does not establish long-context QSA performance.

## Quick start

Use the Relax revision in [relax.version](relax.version) and its training image. Recipes automatically apply the integration patch; they install no dependencies and save no checkpoints.

```bash
export MODEL_DIR=/path/to/models
export PROMPT_DATA=/path/to/LongAlign-10k/long.jsonl

# Dense: ChunkFlow enabled; defaults to 64K / chunk4096 / K1
bash examples/chunkflow/scripts/run-qwen3-4B-sft-8xgpu.sh

# MoE: defaults to 64K / chunk4096 / K1; K0 disables ChunkFlow recomputation
KEEP_LAST_K=0 bash examples/chunkflow/scripts/run-qwen36-35B-A3B-sft-8xgpu.sh
```

## Code layout

```text
examples/chunkflow/
├── adapters/                 # Native computation adapters
│   ├── __init__.py           # Registry and Dense Attention
│   ├── gdn.py                # GDN convolution and recurrent state
│   ├── qsa.py                # Native QSA prefix replay
│   ├── ple.py                # Native PLE prefix adaptation
│   └── prefix.py             # Shared prefix layout, RoPE and SP
├── integration.py            # Install adapters during model creation
├── scheduler.py / units.py   # Chunk and execution plans
├── schedules.py              # Single-stage forward/backward
├── pipeline_schedules.py     # Multi-stage forward/backward
├── runtime.py / state_store.py # State, gradients and lifetime
├── unit_batch.py / metadata.py # Batch splitting and boundaries
├── rope.py                   # Chunk position offsets
├── arguments.py              # Flags and validation
├── setup.py / relax.patch / relax.version # Pinned integration
├── configs/                  # Native GDN and QSA settings
├── scripts/                  # Dense, MoE, QSA and shared submission
└── test/{cpu,gpu}/           # Core regressions and CUDA validation
```

Execution: **recipe → integration patch → adapter installation → chunk plan → native forward → cross-chunk backward → state cleanup**. Implementation details and test commands are in the [full guide](../../docs/en/guide/chunkflow.md).
