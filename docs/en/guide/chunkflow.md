# ChunkFlow

ChunkFlow splits long sequences into chunks during Megatron training, schedules forward and backward passes according to their dependencies, and preserves K/V state and gradients across chunks. It changes the size of work units in the pipeline and how activations are retained. It is disabled by default; enable it with `--use-chunkflow`.

MoE models continue to use Megatron's native routing, dispatch/combine, and expert computation. Chunking also adds call overhead, so compare speed and memory usage with unchunked training under the same configuration.

## Qwen3-4B Example

Run the following command from the repository root in an eight-GPU environment with Megatron, Megatron Bridge, Flash Attention, and a Ray cluster configured. Place the model at `${MODEL_DIR}/Qwen3-4B` and replace the data path with an existing SFT parquet file. The example enables ClearML and requires its access credentials to be configured.

```bash
export MODEL_DIR=/path/to/models
export PROMPT_DATA=/path/to/train.parquet
export CHUNK_SIZE=4096 KEEP_LAST_K=0
bash scripts/entrypoint/ray-job.sh scripts/training/sft/run-qwen3-4B-8xgpu-chunkflow.sh chunkflow
```

The default data columns are `question` and `answer`; override them with `INPUT_KEY` and `LABEL_KEY`. See [SFT Training](./sft-training.md) for data formats. Replace the final `chunkflow` argument with `baseline` to run the unchunked comparison.

The example defaults to TP2/PP4, CP1, GBS8, a 32K sequence limit, 100 steps, and FA3. These are script defaults. Set `RELAX_CHUNKFLOW_FLASH_BACKEND=fa2` if FA3 is unavailable. The script prints `LOG_FILE` and streams logs to both the terminal and that file.

## Configuration

Use these arguments when enabling ChunkFlow in another training script:

| Argument | Default | Description |
| --- | --- | --- |
| `--use-chunkflow` | Disabled | Install chunked attention and select the ChunkFlow schedule. |
| `--chunkflow-chunk-size` | `0` | Tokens per chunk; must be positive when enabled. |
| `--chunkflow-pack-policy` | `staggered` | Used by the example. RoPE offsets are unsupported for `lockstep` units containing multiple dependent segments. |
| `--chunkflow-keep-last-k` | `0` | `0` retains activations for every chunk. A positive value retains only the last K chunks in each group and recomputes earlier chunks before backward. Exposed as `KEEP_LAST_K` in the example. |
| `--chunkflow-pipeline` | `0` | For PP greater than 1, set this to the same value as `--pipeline-model-parallel-size`. |

PP greater than 1 uses an experimental non-interleaved pipeline executor and also requires `RELAX_CHUNKFLOW_PIPELINE_UNTESTED=1`. The 4B example sets this environment variable by default and passes the corresponding PP argument.

## Compatibility Limits

- The current path requires CP=1, packed THD, causal attention, per-token loss, and a single model chunk. VPP is unsupported.
- The example uses zero dropout and disables RoPE fusion. The explicit MRoPE position path currently supports text-only batches.
- GDN requires kernels that propagate gradients through initial and final states and Megatron's `_forward_compute` interface. Disable preprocessing fusion; deterministic mode and full-layer recomputation are unsupported.
- MoE auxiliary losses, capacity limits, and token-dropping configurations need separate validation. Do not assume they remain equivalent after chunking.
- PP=1 does not currently update upstream PackingMetrics. Use actual step time for performance comparisons.

Keep the data, random seed, parallel configuration, precision, and recomputation settings consistent when comparing loss, time, and memory. Use the framework's existing profiler for diagnosis; see [Performance Tuning](./performance-tuning.md).
