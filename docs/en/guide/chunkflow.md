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

ChunkFlow calls the model's original Megatron/Transformer Engine attention implementation. Kernel selection belongs to that implementation. Remove `RELAX_CHUNKFLOW_FLASH_BACKEND` from older recipes; setting it now raises an error.

The connection layer supplies historical K/V and bottom-right causal masking, so current queries can attend to the correct prefix. It creates fresh physical packed-sequence boundaries for attention and keeps RoPE position offsets separate. Module identities, parameters, and checkpoint keys remain unchanged. Native autograd may retain concatenated historical K/V for each chunk, so measure memory again when migrating from the former custom attention backward.

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

- The built-in dense adapter accepts the exact Megatron `SelfAttention` + `TEDotProductAttention` pair. Other classes and subclasses require explicit adapters. The non-TE `DotProductAttention` implementation does not support the required packed THD call and is not substituted with a custom kernel.
- The current path requires CP=1, packed THD, causal attention, per-token loss, and a single model chunk. VPP is unsupported.
- Use zero dropout and disable RoPE fusion. The explicit MRoPE position path currently supports text-only batches.
- Use `--chunkflow-keep-last-k` for attention recomputation. Megatron full-layer recomputation and selective `core_attn` recomputation are rejected because they replay cache writes without the ChunkFlow replay marker.
- Sliding windows, non-vanilla softmax, QK clipping/logit statistics, external attention masks/biases, and the Qwen3.5 optimized recipe need separate adapters. QSA and models with PLE are rejected: preserving their original modules alone does not supply sparse-indexer, n-gram, or convolution history.
- GDN requires kernels that propagate gradients through initial and final states and Megatron's `_forward_compute` interface. Disable preprocessing fusion; deterministic mode and full-layer recomputation are unsupported.
- MoE auxiliary losses, capacity limits, and token-dropping configurations need separate validation. Do not assume they remain equivalent after chunking.
- PP=1 does not currently update upstream PackingMetrics. Use actual step time for performance comparisons.

Keep the data, random seed, parallel configuration, precision, and recomputation settings consistent when comparing loss, time, and memory. Use the framework's existing profiler for diagnosis; see [Performance Tuning](./performance-tuning.md).

## Adding an Attention Adapter

Register an `AttentionAdapter` with `register_attention_adapter(attention_type, core_type, adapter)` from `relax.backends.megatron.chunkflow.adapters` before ChunkFlow installation in each training worker, for example in a custom model provider. Registration matches both classes exactly and rejects duplicate entries. Validation runs before attention installation; installers return an undo callback and must undo their own mutations if they fail.

For an operator whose history is ordinary K/V, `dense_attention_adapter(name=..., execute=..., validate=...)` reuses ChunkFlow's state and gradient connections. Its executor receives the saved native `forward` and a `DenseAttentionCall`: THD queries, K/V fragments, physical Q/KV boundaries, CPU lengths, and maximum lengths. The executor calls the native operator with bottom-right causal alignment and returns the output with native autograd intact. It must not reuse RoPE offset metadata as physical padding. Full-layer adapters can manage other history types; registering a sparse operator as dense does not implement its indexer state.

GDN continues to call Megatron's convolution and recurrence kernels through the existing state adapter. ChunkFlow supplies initial/final states and connects state gradients; it does not implement the GDN calculation. Validate each new adapter against unchunked outputs, input and parameter gradients, packed-sequence boundaries, and ChunkFlow recomputation before using it for training.
