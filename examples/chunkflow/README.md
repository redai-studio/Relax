# ChunkFlow example

[中文](README.zh-CN.md) · [Full guide](../../docs/en/guide/chunkflow.md)

ChunkFlow schedules sequence chunks through Megatron models and connects their history and gradients. Attention and GDN computation use the model's native implementations. This directory owns the implementation, argument definitions, integration patch, launch scripts, and tests.

## Relax version

The supported source baseline is **`d548769b77f12e445e464f99d4332290a3e5bd3d`**, recorded in `relax.version`. Use that revision's training environment, including Megatron/Bridge, TE, and FLA. Python 3.10 or newer is required by Relax.

For a separate checkout at the pinned revision, copy this complete directory to its `examples/chunkflow/`. From the repository root:

```bash
python examples/chunkflow/setup.py
python examples/chunkflow/setup.py --check
```

The installer checks the pinned Relax sources and applies `relax.patch` to three core files. It accepts this example's descendant checkout when the core still matches the baseline, rejects incompatible edits, and can be rerun. Remove the patch with `python examples/chunkflow/setup.py --reverse`. No dependencies are installed automatically.

## Train Dense or MoE

Provide these directories on the training nodes:

| Variable    | Contents                                                  |
| ----------- | --------------------------------------------------------- |
| `MODEL_DIR` | `Qwen3-4B/` or `Qwen3.6-35B-A3B/` Hugging Face checkpoint |
| `DATA_DIR`  | `train.jsonl`, each row containing a `messages` list      |

For example, one data row is:

```json
{"messages": [{"role": "user", "content": "What is 1 + 1?"}, {"role": "assistant", "content": "2"}]}
```

Use longer conversations to exercise multiple chunks. Both recipes are text-only SFT with eight GPUs, TP2/PP1/CP1, zero dropout, and ChunkFlow activation recomputation. The MoE recipe uses EP8/ETP1, native all-to-all dispatch, no router auxiliary loss, and unfused GDN preprocessing. It also uses CPU optimizer offload.

ChunkFlow requires `--calculate-per-token-loss` to keep normalization independent of chunk boundaries. It rejects `--fully-async` combined with `--use-dynamic-batch-size`, because its fixed execution plan cannot consume an unknown number of streaming micro-batches. The supplied recipes use per-token loss and synchronous training.

After exporting `MODEL_DIR` and `DATA_DIR`, preview without Ray or GPUs:

```bash
DRY_RUN=1 bash examples/chunkflow/scripts/run-qwen3-4B-sft-8xgpu.sh
DRY_RUN=1 bash examples/chunkflow/scripts/run-qwen36-35B-A3B-sft-8xgpu.sh
```

On a dedicated eight-GPU node, choose one recipe and run it directly:

```bash
bash examples/chunkflow/scripts/run-qwen3-4B-sft-8xgpu.sh
bash examples/chunkflow/scripts/run-qwen36-35B-A3B-sft-8xgpu.sh
```

The scripts check the integration patch, then reuse `scripts/entrypoint/local.sh` to start Ray and submit to the local Jobs service. No manual `ray start` is needed. This standard entrypoint clears local Python, SGLang, and Ray processes, so use a dedicated training node.

To use an existing cluster, set `RAY_ADDRESS` to its GCS address and `RAY_JOBS_ADDRESS` to its Jobs HTTP URL before running the same command. The script uses the standard `ray-job.sh` environment and cleanup; an unreachable cluster causes an error rather than a local restart. An external entrypoint that already set `RELAX_ENTRYPOINT_MODE` is reused. Submission blocks unless `RAY_NO_WAIT=1` is set. The repository is uploaded as the working directory by default, including this example and the applied patch.

Override `CHUNK_SIZE` (4096), `KEEP_LAST_K` (1), `SEQ_LENGTH` (8192), or `NUM_STEPS` (100) through environment variables. `KEEP_LAST_K=0` retains every chunk's activation graph. Model and training options are visible in each recipe; `_common.sh` handles standard entrypoint selection and submission.

All three recipes accept `PROMPT_DATA` to override `${DATA_DIR}/train.jsonl`. For LongAlign with a `messages` column, set `PROMPT_DATA="${DATA_DIR}/LongAlign-10k/long.jsonl"` and choose `SEQ_LENGTH` for the tokenized conversations, for example `65536`. Samples exceeding that limit are skipped; the dataset's stored length may differ from the model tokenizer's count.

All three recipes load initial Hugging Face weights and do not save or resume training checkpoints. They do not use `SAVE_DIR` or `SAVE_INTERVAL`, including previously exported values. `PROJECT_NAME` defaults to `Relax/sft/chunkflow`; `EXP_NAME` identifies the experiment. ClearML task names and logs under `scripts/log/` include a timestamp for each run.

## Qwen3.8-Flash-Next (QSA + GDN + PLE)

The experimental 32-GPU recipe uses `${MODEL_DIR}/Qwen3.8-Flash-Next`, the same `train.jsonl` format, TP2/PP1/CP1, EP8/ETP1, and sequence parallelism. It keeps the checkpoint's PLE and hyper-connection architecture. PLE parameters and the hard-selection indexer are frozen; PLE tables use native CPU offload, and the optimizer also offloads to CPU. Plan host memory for both. The GPU count is a launch configuration, not a guarantee that the model fits any 32-GPU machine.

```bash
DRY_RUN=1 bash examples/chunkflow/scripts/run-qwen38-flash-next-sft-32xgpu.sh
# Run on every node after setting the directories and node variables below.
bash examples/chunkflow/scripts/run-qwen38-flash-next-sft-32xgpu.sh
```

For a new cluster, the recipe defaults to `WORLD_SIZE=4` nodes and `NUM_GPUS=8` per node; their product must be 32. Set `MASTER_ADDR` on every node to the head node's resolvable **`POD_NAME`**, set `POD_NAME` to the current node's hostname, and set `HOST_IP` to its reachable IP. The standard `spmd-multinode.sh` entrypoint identifies the head by the exact equality `MASTER_ADDR == POD_NAME`, starts Ray on each node, and submits once from the head after all nodes join. Workers remain running; starting only the head leaves it waiting for workers. Use dedicated nodes: this entrypoint clears local processes and requires permission to set reserved network ports with `sysctl`.

For an existing 32-GPU cluster, set `RAY_ADDRESS` and `RAY_JOBS_ADDRESS` and run the same recipe only on the submitter. No node variables or cluster startup are needed in this mode. Preview mode initializes neither kind of cluster.

`configs/qwen38.yaml` disables GDN preprocessing fusion and QSA Stage-2 indexer loss. The script selects native `flex` attention and disables the Qwen3.8 fused/community selector overrides, local-query/replay patches, and streamed PLE. `_common.sh` forwards these model environment settings to Ray workers. Use the same settings when running an unchunked reference.

`adapters/qsa.py` replays each layer's complete hidden-state prefix through native `QwenSparseSelfAttention`. Native projection, pooling, TopK, masking, and attention remain responsible for computation; the adapter returns only the current chunk's outputs. Selection temporarily restores the original parent pack's row count, document count, and candidate width before calling native TopK. Preserving only visible candidates can change tie selection. `adapters/ple.py` similarly supplies the original prefix token IDs and hidden states to native PLE hashing, gating, and causal convolution. `adapters/prefix.py` shares physical-boundary, padding, RoPE, and SP layout handling. Their history caches use separate namespaces so PLE and QSA at the same layer do not overwrite one another.

Native autograd computes each replay's parameter gradients. Historical hidden-state gradients accumulate in `ChunkState` and reconnect when earlier chunks run backward; `Fr` reuses those cached leaves. Prefix replay repeats computation and can retain substantial activations, while the selector still uses the full parent layout. Treat this as a compatibility implementation and measure its cost before adopting it for long-context training.

This adapter requires the pinned native QSA classes, a bitset backend (`flex`, or `dense_masked` for small references), PP1/CP1, packed text input, zero dropout, and ChunkFlow-owned recomputation. It rejects `id_sparse`, Stage-2 auxiliary loss, inference caches, external masks/biases, and the optimized method overrides above. The CPU numerical tests load the shipped native selector/attention and PLE methods with small construction stubs; they do not validate CUDA kernels, distributed collectives, or a full Flash-Next training run.

## Code and tests

| Files                                                       | Responsibility                                     |
| ----------------------------------------------------------- | -------------------------------------------------- |
| `scheduler.py`, `units.py`                                  | Chunk and execution plans                          |
| `unit_batch.py`, `rope.py`, `metadata.py`                   | Inputs, original positions, packed boundaries      |
| `integration.py`, `adapters/__init__.py`, `adapters/gdn.py` | Model installation and native computation adapters |
| `adapters/qsa.py`, `adapters/ple.py`, `adapters/prefix.py`  | Native QSA/PLE prefix replay                       |
| `runtime.py`, `state_store.py`                              | Cross-chunk state, gradients, cleanup              |
| `schedules.py`, `pipeline_schedules.py`                     | Forward/backward execution                         |
| `arguments.py`, `setup.py`, `relax.patch`                   | Configuration and pinned integration               |

```bash
CUDA_VISIBLE_DEVICES="" python -m pytest examples/chunkflow/test/cpu -q
python -m pytest examples/chunkflow/test/gpu -q
```

CPU tests may require Megatron imports; GPU tests require CUDA and the native backends. Skips are not a numerical-validation result. PP>1 is experimental and requires explicit opt-in. CP>1, VPP, multimodal inputs, and full-layer recomputation are outside the supplied recipes; consult the full guide for adapter-specific limits.
