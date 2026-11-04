# ChunkFlow

ChunkFlow splits long sequences into chunks during Megatron training, schedules forward and backward passes according to their dependencies, and preserves K/V state and gradients across chunks. It changes the size of work units in the pipeline and how activations are retained. It is disabled by default; enable it with `--use-chunkflow`.

MoE models continue to use Megatron's native routing, dispatch/combine, and expert computation. Chunking also adds call overhead, so compare speed and memory usage with unchunked training under the same configuration.

## Enabling ChunkFlow

Add the following argument fragment to an existing Megatron training script, keeping its model, data, and resource configuration. See [SFT Training](./sft-training.md) for SFT data preparation and launch instructions.

```text
--use-chunkflow
--chunkflow-chunk-size 4096
--context-parallel-size 1
--calculate-per-token-loss
--no-rope-fusion
--attention-dropout 0.0
--hidden-dropout 0.0
```

`4096` is an example chunk size; adjust it for the task. For an unchunked comparison, remove `--use-chunkflow` and keep the other training settings.

ChunkFlow defaults to FA2. To use FA3, install its interface in the training environment and set `RELAX_CHUNKFLOW_FLASH_BACKEND=fa3`.

## Configuration

The available arguments are:

| Argument | Default | Description |
| --- | --- | --- |
| `--use-chunkflow` | Disabled | Install chunked attention and select the ChunkFlow schedule. |
| `--chunkflow-chunk-size` | `0` | Tokens per chunk; must be positive when enabled. |
| `--chunkflow-pack-policy` | `staggered` | RoPE offsets are unsupported for `lockstep` units containing multiple dependent segments. |
| `--chunkflow-keep-last-k` | `0` | `0` retains activations for every chunk. A positive value retains only the last K chunks in each group and recomputes earlier chunks before backward. |
| `--chunkflow-pipeline` | `0` | For PP greater than 1, set this to the same value as `--pipeline-model-parallel-size`. |

PP greater than 1 uses an experimental non-interleaved pipeline executor and also requires `RELAX_CHUNKFLOW_PIPELINE_UNTESTED=1`. When submitting through Ray, include the required ChunkFlow environment variables in `runtime_env.env_vars` so training workers can read them.

## Compatibility Limits

- The current path requires CP=1, packed THD, causal attention, per-token loss, and a single model chunk. VPP is unsupported.
- Use zero dropout and disable RoPE fusion. The explicit MRoPE position path currently supports text-only batches.
- GDN requires kernels that propagate gradients through initial and final states and Megatron's `_forward_compute` interface. Disable preprocessing fusion; deterministic mode and full-layer recomputation are unsupported.
- MoE auxiliary losses, capacity limits, and token-dropping configurations need separate validation. Do not assume they remain equivalent after chunking.
- PP=1 does not currently update upstream PackingMetrics. Use actual step time for performance comparisons.

Keep the data, random seed, parallel configuration, precision, and recomputation settings consistent when comparing loss, time, and memory. Use the framework's existing profiler for diagnosis; see [Performance Tuning](./performance-tuning.md).
