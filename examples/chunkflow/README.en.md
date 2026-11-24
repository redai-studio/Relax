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

| Method        | Mean train time | Mean step time | Training MFU | Training speedup | End-to-end speedup |
| ------------- | --------------: | -------------: | -----------: | ---------------: | -----------------: |
| Baseline      |          3.08 s |         4.38 s |        23.1% |            1.00× |              1.00× |
| **ChunkFlow** |      **1.86 s** |     **3.10 s** |    **38.2%** |        **1.65×** |          **1.40×** |

**Memory reference**: LongAlign / 64K / chunk4096 / GBS1 / K1; peak allocated memory drops from **52.28 GiB → 21.01 GiB**, saving **59.81%**.

### MoE · Qwen3.6-35B-A3B

> **Configuration**　8 GPUs · TP2 / PP4 / EP2 · GBS8 · chunk8192 · LongAlign

| Method        | Mean train time | Mean step time | Training MFU | Training speedup | End-to-end speedup |
| ------------- | --------------: | -------------: | -----------: | ---------------: | -----------------: |
| Baseline      |          16.2 s |         17.5 s |        13.8% |            1.00× |              1.00× |
| **ChunkFlow** |      **17.0 s** |     **18.4 s** |    **13.1%** |       **0.950×** |         **0.950×** |

**Memory reference**: LongAlign / 64K / chunk4096 / GBS8 / K0; peak allocated memory drops from **56.38 GiB → 37.79 GiB**, saving **32.97%**.

______________________________________________________________________

## When to enable

| Scenario                                              | Recommendation                                        |
| ----------------------------------------------------- | ----------------------------------------------------- |
| Long-context Dense, small batches, pipeline idle time | Benchmark it; training speedups have been observed    |
| Long-context MoE with memory pressure                 | Try it if additional compute time is acceptable       |
| Short sequences, ample memory, throughput first       | Leave off unless a matched comparison shows a benefit |

______________________________________________________________________

## Quick start

**Version dependency**: based on Relax [d548769b](https://github.com/redai-studio/Relax/commit/d548769b77f12e445e464f99d4332290a3e5bd3d) and its matching training image; see [relax.version](relax.version) for the pinned revision.

```bash
export MODEL_DIR=/path/to/model
export PROMPT_DATA=/path/to/data
```

**Dense · Qwen3-4B**　64K / chunk4096 / K0

```bash
bash examples/chunkflow/scripts/run-qwen3-4B-sft-8xgpu.sh
```

**MoE · Qwen3.6-35B-A3B**　64K / chunk4096 / K0

```bash
KEEP_LAST_K=0 bash examples/chunkflow/scripts/run-qwen36-35B-A3B-sft-8xgpu.sh
```

### Integrate an existing recipe

**① Install the integration patch**

```bash
python3 examples/chunkflow/setup.py
```

**② Add the ChunkFlow arguments**

```bash
--use-chunkflow \
--chunkflow-chunk-size 4096 \
--chunkflow-keep-last-k 0 \
--no-rope-fusion
```

Do not duplicate settings already present in the recipe:

| Setting                                        | Reason                                                                                                                   |
| ---------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| `--context-parallel-size 1`                    | Cross-chunk state management does not currently support CP splitting                                                     |
| `--calculate-per-token-loss`                   | Count tokens without changing sample weights through chunking                                                            |
| `--no-rope-fusion`                             | Use the RoPE path that supports chunk position offsets                                                                   |
| `--attention-dropout 0.0 --hidden-dropout 0.0` | Current adapters require zero dropout to avoid random-mask issues during chunking and replay                             |
| Megatron recomputation compatibility           | Disable full-layer / selective `core_attn`                                                                               |
| Expert parallelism (EP)                        | All ranks in each EP group must share the same chunk execution plan; mismatches fail collectively before model execution |
| RL policy loss                                 | GSPO / OPSM / M2PO objectives requiring full-sequence or batch statistics are not supported; this does not restrict SFT  |
| MTP training                                   | Cross-chunk multi-token prediction is not supported; disable `--enable-mtp-training`                                     |
| Deferred head weight gradients                 | Disable `defer_embedding_wgrad_compute`; native FIFO buffers do not support chunk backward ordering                      |
| MoE load-balancing auxiliary loss              | Use `none` or zero coefficients; cross-chunk routing statistics are not supported                                        |
| Distributed optimizer                          | Keep `--num-distributed-optimizer-instances 1` (default)                                                                 |

### Key parameters

| Argument                  | Starting point  | Adjustment                                                                                                              |
| ------------------------- | --------------- | ----------------------------------------------------------------------------------------------------------------------- |
| `--chunkflow-chunk-size`  | `4096`          | Must be below actual sample length to split it; smaller chunks increase chunk count and call overhead                   |
| `--chunkflow-keep-last-k` | `0`             | `0` disables recomputation; `1` retains only the last chunk's activations and recomputes earlier chunks before backward |
| `--seq-length`            | Match your data | Sample length limit, not chunk size; samples above the limit are skipped                                                |
