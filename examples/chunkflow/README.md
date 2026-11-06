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
| `SAVE_DIR`  | Persistent training checkpoints                           |

For example, one data row is:

```json
{"messages": [{"role": "user", "content": "What is 1 + 1?"}, {"role": "assistant", "content": "2"}]}
```

Use longer conversations to exercise multiple chunks. Both recipes are text-only SFT with eight GPUs, TP2/PP1/CP1, zero dropout, and ChunkFlow activation recomputation. The MoE recipe uses EP8/ETP1, native all-to-all dispatch, no router auxiliary loss, and unfused GDN preprocessing. It also uses CPU optimizer offload.

After exporting the three directory variables, preview without Ray or GPUs:

```bash
DRY_RUN=1 bash examples/chunkflow/scripts/run-qwen3-4B-sft-8xgpu.sh
DRY_RUN=1 bash examples/chunkflow/scripts/run-qwen36-35B-A3B-sft-8xgpu.sh
```

Set `RAY_ADDRESS` and `RAY_JOBS_ADDRESS` to your existing cluster's GCS and Jobs HTTP addresses. Choose one recipe:

```bash
bash scripts/entrypoint/ray-job.sh examples/chunkflow/scripts/run-qwen3-4B-sft-8xgpu.sh
bash scripts/entrypoint/ray-job.sh examples/chunkflow/scripts/run-qwen36-35B-A3B-sft-8xgpu.sh
```

The existing Ray entrypoint performs cluster cleanup; use it with the intended dedicated training cluster. Submission blocks unless `RAY_NO_WAIT=1` is set. The repository is uploaded as the working directory by default, including this example and the applied patch.

Override `CHUNK_SIZE` (2048), `KEEP_LAST_K` (1), `SEQ_LENGTH` (8192), `NUM_STEPS` (100), or `SAVE_INTERVAL` (10) through environment variables. `KEEP_LAST_K=0` retains every chunk's activation graph. Model and training options are visible in each recipe; `_common.sh` only handles submission.

Save and resume use `${SAVE_DIR}/${EXP_NAME}` and retain two checkpoints. Keep those values stable to resume; change `EXP_NAME` for a separate experiment. `PROJECT_NAME` defaults to `Relax/sft/chunkflow`. Timestamps affect ClearML task names and logs under `scripts/log/`, not checkpoint paths.

## Code and tests

| Files                                     | Responsibility                                     |
| ----------------------------------------- | -------------------------------------------------- |
| `scheduler.py`, `units.py`                | Chunk and execution plans                          |
| `unit_batch.py`, `rope.py`, `metadata.py` | Inputs, original positions, packed boundaries      |
| `integration.py`, `adapters.py`, `gdn.py` | Model installation and native computation adapters |
| `runtime.py`, `state_store.py`            | Cross-chunk state, gradients, cleanup              |
| `schedules.py`, `pipeline_schedules.py`   | Forward/backward execution                         |
| `arguments.py`, `setup.py`, `relax.patch` | Configuration and pinned integration               |

```bash
CUDA_VISIBLE_DEVICES="" python -m pytest examples/chunkflow/test/cpu -q
python -m pytest examples/chunkflow/test/gpu -q
```

CPU tests may require Megatron imports; GPU tests require CUDA and the native backends. Skips are not a numerical-validation result. PP>1 is experimental and requires explicit opt-in. QSA, PLE, CP>1, VPP, multimodal inputs, and full-layer recomputation are outside the supplied recipes; consult the full guide for adapter-specific limits.
