<h1 align="center">ChunkFlow</h1>

<p align="center">Long-sequence chunking · Native computation · Lower memory pressure</p>

<p align="center">
  <a href="#measured-benefits">Benchmarks</a> ·
  <a href="#when-to-enable">Recommendations</a> ·
  <a href="#quick-start">Quick start</a> ·
  <a href="README.md">中文</a>
</p>

______________________________________________________________________

## Measured benefits

### Dense · Qwen3-4B

> **Configuration**　8 GPUs · TP2 / PP4 · GBS8 · chunk4096

Data: 5,000 synthetic Chinese SFT samples. Length distribution: 70% at 0–2K, 20% at 2–8K, 8% at 8–16K, and 2% at 16–32K.

| Method        | Mean train time | Mean step time | Training MFU | Training speedup | End-to-end speedup | Memory ref.¹ (GiB) |     Saved¹ |
| ------------- | --------------: | -------------: | -----------: | ---------------: | -----------------: | -----------------: | ---------: |
| Baseline      |          3.08 s |         4.38 s |        23.1% |            1.00× |              1.00× |              52.28 |          — |
| **ChunkFlow** |      **1.86 s** |     **3.10 s** |    **38.2%** |        **1.65×** |          **1.40×** |          **21.01** | **59.81%** |

### MoE · Qwen3.6-35B-A3B

> **Configuration**　8 GPUs · TP2 / PP4 / EP2 · GBS8 · chunk8192 · LongAlign

| Method        | Mean train time | Mean step time | Training MFU | Training speedup | End-to-end speedup | Memory ref.¹ (GiB) |     Saved¹ |
| ------------- | --------------: | -------------: | -----------: | ---------------: | -----------------: | -----------------: | ---------: |
| Baseline      |          16.2 s |         17.5 s |        13.8% |            1.00× |              1.00× |              56.38 |          — |
| **ChunkFlow** |      **17.0 s** |     **18.4 s** |    **13.1%** |       **0.950×** |         **0.950×** |          **37.79** | **32.97%** |

Speedups follow the experiment summary: above 1 means faster, below 1 means slower.

¹ Memory reference: LongAlign, 64K, chunk4096; Dense uses GBS1 / K1, MoE uses GBS8 / K0 with native selective `gdn moe`.

______________________________________________________________________

## When to enable

| Scenario                                              | Recommendation                                                                           |
| ----------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| Long-context Dense, small batches, pipeline idle time | Benchmark it; training speedups have been observed                                       |
| Long-context MoE with memory pressure                 | Try it if additional compute time is acceptable                                          |
| Short sequences, ample memory, throughput first       | Leave off unless a matched comparison shows a benefit                                    |
| QSA / PLE models                                      | Experimental: prefix replay repeats computation; savings and speedups are not guaranteed |

______________________________________________________________________

## Quick start

Use the Relax revision in [relax.version](relax.version) and its training image. Recipes automatically apply the integration patch; they install no dependencies and save no checkpoints.

```bash
export MODEL_DIR=/path/to/model
export PROMPT_DATA=/path/to/data
```

**Dense · Qwen3-4B**　Defaults: 64K / chunk4096 / K1

```bash
bash examples/chunkflow/scripts/run-qwen3-4B-sft-8xgpu.sh
```

**MoE · Qwen3.6-35B-A3B**　64K / chunk4096; K0 below disables ChunkFlow recomputation

```bash
KEEP_LAST_K=0 bash examples/chunkflow/scripts/run-qwen36-35B-A3B-sft-8xgpu.sh
```

### Integrate an existing recipe

Integrate with a compatible Megatron recipe, keeping its model, data, and launch method.

**① Install the integration patch** (once, from the repository root)

```bash
python3 examples/chunkflow/setup.py
```

**② Add the ChunkFlow arguments** (append to the training command; do not run separately)

```bash
--use-chunkflow \
--chunkflow-chunk-size 4096 \
--chunkflow-keep-last-k 0 \
--no-rope-fusion
```

`K=0` retains all activations without recomputation; `K=1` keeps only the last chunk per group and recomputes earlier chunks before backward.

Do not duplicate settings already present in the recipe:

| Setting                                        | Reason                                                                                       |
| ---------------------------------------------- | -------------------------------------------------------------------------------------------- |
| `--context-parallel-size 1`                    | Cross-chunk state management does not currently support CP splitting                         |
| `--calculate-per-token-loss`                   | Count tokens without changing sample weights through chunking                                |
| `--no-rope-fusion`                             | Use the RoPE path that supports chunk position offsets                                       |
| `--attention-dropout 0.0 --hidden-dropout 0.0` | Current adapters require zero dropout to avoid random-mask issues during chunking and replay |

**PP follows the existing configuration automatically**: ChunkFlow selects its executor from Megatron’s actual PP size. No separate pipeline argument is needed. For PP>1, training workers still require `RELAX_CHUNKFLOW_PIPELINE_UNTESTED=1` to opt into the experimental executor; pass it through Ray’s `runtime_env.env_vars`. Example recipes already set it.

Use adapted packed-text models only; retain the corresponding recipe's model settings for GDN / QSA.

### Key parameters

| Argument                  | Starting point  | Adjustment                                                                                                                                |
| ------------------------- | --------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| `--chunkflow-chunk-size`  | `4096`          | Must be below actual sample length to split it; smaller chunks increase chunk count and call overhead                                     |
| `--chunkflow-keep-last-k` | `0`             | `0` disables recomputation; `1` retains only the last chunk's activations and recomputes earlier chunks before backward; keep `0` for QSA |
| `--seq-length`            | Match your data | Sample length limit, not chunk size; samples above the limit are skipped                                                                  |

For a baseline, remove only `--use-chunkflow` and keep other arguments and data identical. Do not enable native full-layer or selective `core_attn` recomputation alongside ChunkFlow.
