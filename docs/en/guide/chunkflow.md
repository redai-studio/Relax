# ChunkFlow

ChunkFlow is a standalone example under `examples/chunkflow/`. It splits long sequences into chunks during Megatron training, schedules their forward and backward passes, and preserves cross-chunk state and gradients. The example owns its implementation, arguments, and tests; an included integration patch connects it to Relax.

MoE models continue to use Megatron's native routing, dispatch/combine, and expert computation. Chunking also adds call overhead, so compare speed and memory usage with unchunked training under the same configuration.

## Pinned version and installation

`examples/chunkflow/relax.version` pins the Relax source baseline to `d548769b77f12e445e464f99d4332290a3e5bd3d`, the revision before ChunkFlow was introduced. Use that revision's Megatron/Bridge, Transformer Engine, and FLA training environment rather than installing a separate attention backend.

Copy the complete `examples/chunkflow/` directory into a Relax checkout at that revision, then run from the repository root:

```bash
python examples/chunkflow/setup.py
python examples/chunkflow/setup.py --check
```

The installer checks Relax sources against the pinned commit before applying `examples/chunkflow/relax.patch`. The patch connects argument parsing, model installation, training schedule selection, and chunk positions. Repeated installation is idempotent; incompatible or partial edits are rejected without overwriting them. A descendant checkout containing this example also works when its Relax core still matches the pinned baseline and this patch.

Use `python examples/chunkflow/setup.py --reverse` to remove the integration patch. The installer manages source files only; it does not install dependencies or start a cluster.

## Dense and MoE examples

Both examples run text-only SFT on eight GPUs with TP=2, PP=1, and CP=1. Qwen3-4B uses EP=1; Qwen3.6-35B-A3B uses EP=8 and ETP=1, including GDN state adaptation.

Set `MODEL_DIR`, `DATA_DIR`, and `SAVE_DIR` first. Model directories are `${MODEL_DIR}/Qwen3-4B` and `${MODEL_DIR}/Qwen3.6-35B-A3B`. Data is `${DATA_DIR}/train.jsonl`, with a `messages` conversation list in each row; see [SFT Training](./sft-training.md). `SAVE_DIR` must be persistent and accessible to the training nodes.

```bash
# Preview arguments without connecting to Ray or requiring GPUs.
DRY_RUN=1 bash examples/chunkflow/scripts/run-qwen3-4B-sft-8xgpu.sh
DRY_RUN=1 bash examples/chunkflow/scripts/run-qwen36-35B-A3B-sft-8xgpu.sh

# Set RAY_ADDRESS and RAY_JOBS_ADDRESS for an existing cluster, then choose one model.
bash scripts/entrypoint/ray-job.sh examples/chunkflow/scripts/run-qwen3-4B-sft-8xgpu.sh
bash scripts/entrypoint/ray-job.sh examples/chunkflow/scripts/run-qwen36-35B-A3B-sft-8xgpu.sh
```

`RAY_JOBS_ADDRESS` is the cluster's Jobs HTTP URL. Submission blocks by default; set `RAY_NO_WAIT=1` for nonblocking submission. Scripts upload the repository working directory by default so both driver and workers can import the example. A custom `WORKING_DIR` must also contain patched Relax and the complete example.

Script defaults are `SEQ_LENGTH=8192`, `CHUNK_SIZE=2048`, `KEEP_LAST_K=1`, `NUM_STEPS=100`, and `SAVE_INTERVAL=10`; override them with the corresponding environment variables. Save and resume use the stable `${SAVE_DIR}/${EXP_NAME}` directory and retain two training checkpoints. Keep both values unchanged to resume. Timestamps appear only in ClearML task names and logs under `examples/chunkflow/scripts/log/`. `PROJECT_NAME` defaults to `Relax/sft/chunkflow`; each model has a distinct, overridable `EXP_NAME`.

The MoE example uses native all-to-all dispatch, disables router auxiliary loss, and disables GDN preprocessing fusion through `examples/chunkflow/configs/gdn.yaml`. CPU optimizer offload requires sufficient host memory. Neither example enables full-layer recomputation, VPP, MTP, or multimodal input.

## Qwen3.8-Flash-Next (QSA + GDN + PLE)

The experimental 32-GPU recipe uses `${MODEL_DIR}/Qwen3.8-Flash-Next`, the same `train.jsonl` format, TP2/PP1/CP1, EP8/ETP1, and sequence parallelism. It keeps the checkpoint's PLE and hyper-connection architecture. PLE parameters and the hard-selection indexer are frozen; PLE tables use native CPU offload, and the optimizer also offloads to CPU. Plan host memory for both. The GPU count is a launch configuration, not a guarantee that the model fits any 32-GPU machine.

```bash
DRY_RUN=1 bash examples/chunkflow/scripts/run-qwen38-flash-next-sft-32xgpu.sh
# Submit after setting the directories and existing cluster addresses.
bash scripts/entrypoint/ray-job.sh examples/chunkflow/scripts/run-qwen38-flash-next-sft-32xgpu.sh
```

`configs/qwen38.yaml` disables GDN preprocessing fusion and QSA Stage-2 indexer loss. The script selects native `flex` attention and disables the Qwen3.8 fused/community selector overrides, local-query/replay patches, and streamed PLE. `_common.sh` forwards these model environment settings to Ray workers. Use the same settings when running an unchunked reference.

`adapters/qsa.py` replays each layer's complete hidden-state prefix through native `QwenSparseSelfAttention`. Native projection, pooling, TopK, masking, and attention remain responsible for computation; the adapter returns only the current chunk's outputs. Selection temporarily restores the original parent pack's row count, document count, and candidate width before calling native TopK. Preserving only visible candidates can change tie selection. `adapters/ple.py` similarly supplies the original prefix token IDs and hidden states to native PLE hashing, gating, and causal convolution. `adapters/prefix.py` shares physical-boundary, padding, RoPE, and SP layout handling. Their history caches use separate namespaces so PLE and QSA at the same layer do not overwrite one another.

Native autograd computes each replay's parameter gradients. Historical hidden-state gradients accumulate in `ChunkState` and reconnect when earlier chunks run backward; `Fr` reuses those cached leaves. Prefix replay repeats computation and can retain substantial activations, while the selector still uses the full parent layout. Treat this as a compatibility implementation and measure its cost before adopting it for long-context training.

This adapter requires the pinned native QSA classes, a bitset backend (`flex`, or `dense_masked` for small references), PP1/CP1, packed text input, zero dropout, and ChunkFlow-owned recomputation. It rejects `id_sparse`, Stage-2 auxiliary loss, inference caches, external masks/biases, and the optimized method overrides above. The CPU numerical tests load the shipped native selector/attention and PLE methods with small construction stubs; they do not validate CUDA kernels, distributed collectives, or a full Flash-Next training run.

## Using other training scripts

After applying the integration patch, add the following arguments to another Megatron training script while retaining its model, data, and resource configuration:

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

The scheduler performs chunking; attention does not split the input a second time. Calls without a `chunkflow_unit` marker go directly to the original `forward`, preserving its arguments and behavior.

## Configuration

Arguments live in `examples/chunkflow/arguments.py`. The table lists parser defaults; the scripts use the chunk size and activation retention defaults described above.

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
- `--calculate-per-token-loss` is enforced at startup: averaging each chunk as a separate sample changes the training objective. ChunkFlow also rejects `--fully-async` together with `--use-dynamic-batch-size`; its fixed plan cannot consume a streaming iterator with an unknown micro-batch count. Disable one of those two options.
- Use zero dropout and disable RoPE fusion. The explicit MRoPE position path currently supports text-only batches.
- Use `--chunkflow-keep-last-k` for attention recomputation. Megatron full-layer recomputation and selective `core_attn` recomputation are rejected because they replay cache writes without the ChunkFlow replay marker.
- Sliding windows, non-vanilla softmax, QK clipping/logit statistics, external attention masks/biases, and the Qwen3.5 optimized recipe need separate adapters. QSA and PLE use the restricted native-prefix adapters described above; they require PP1 even when the experimental pipeline is enabled.
- The full-layer GDN adapter matches Megatron's exact `GatedDeltaNet` class; subclasses require explicit registration. It requires kernels that propagate gradients through initial and final states and Megatron's `_forward_compute` interface. Disable preprocessing fusion; deterministic mode and full-layer recomputation are unsupported.
- MoE auxiliary losses, capacity limits, and token-dropping configurations need separate validation. Do not assume they remain equivalent after chunking.
- PP=1 does not currently update upstream PackingMetrics. Use actual step time for performance comparisons.

Keep the data, random seed, parallel configuration, precision, and recomputation settings consistent when comparing loss, time, and memory. Use the framework's existing profiler for diagnosis; see [Performance Tuning](./performance-tuning.md).

## Adding an Attention Adapter

Register an `AttentionAdapter` with `register_attention_adapter(attention_type, core_type, adapter)` from `examples.chunkflow.adapters` before ChunkFlow installation in each training worker, for example in a custom model provider. Dense K/V adapters match both classes exactly. Use `core_type=None` for a full-layer adapter: it matches the outer class exactly and takes precedence, supporting modules such as GDN or attention with an indexer. A full-layer adapter must validate its internal implementation. Duplicate registrations are errors.

All adapters share one installation flow: validate every layer, then install. Each installer returns an undo callback. A failure in a later layer restores already installed layers in reverse order; installers must also undo their own mutations if they fail. GDN's shared convolution wrapper restores the original function when its last user is uninstalled.

For an operator whose history is ordinary K/V, `dense_attention_adapter(name=..., execute=..., validate=...)` reuses ChunkFlow's state and gradient connections. Its executor receives the saved native `forward` and a `DenseAttentionCall`: THD queries, K/V fragments, physical Q/KV boundaries, CPU lengths, and maximum lengths. The executor calls the native operator with bottom-right causal alignment and returns the output with native autograd intact. It must not reuse RoPE offset metadata as physical padding. Full-layer adapters can manage other history types; registering a sparse operator as dense does not implement its indexer state.

State management also shares one interface: `runtime.history(layer_number, group_id, chunk_id, recompute=...)` returns the `ChunkState` objects up to and including the current chunk, reusing cached leaves during recomputation. Dense attention reads all earlier K/V; GDN reads the preceding convolution and recurrence states. `ChunkState.link_tensor` reconnects historical gradients to state already used in the current output; `link_output` connects a separate final state returned by a native kernel. Only adapters that explicitly handle native selective checkpointing should enable `allow_replay=True`.

GDN still calls Megatron's convolution and recurrence kernels; the shared layer manages state and gradient connections. If a native module already exposes a differentiable chunk interface, a full-layer adapter can call it without changing the scheduler. The QSA/PLE adapters supply their missing history through prefix replay; this trades repeated native computation for a smaller adaptation surface. Validate each new adapter against unchunked outputs, input and parameter gradients, packed-sequence boundaries, and ChunkFlow recomputation before using it for training.

## Running tests

Tests live alongside the implementation in `examples/chunkflow/test/`, separated into `cpu/` and `gpu/`. Test filenames no longer have a `chunkflow` prefix.

```bash
CUDA_VISIBLE_DEVICES="" python -m pytest examples/chunkflow/test/cpu -q
python -m pytest examples/chunkflow/test/gpu -q
```

Some CPU tests still require Megatron Python dependencies. GPU tests require real CUDA and Megatron/TE or FLA. Dependency skips do not establish numerical or distributed-training correctness.
