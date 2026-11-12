# Qwen3.8 long-context SFT on Blackwell

The 64-GPU recipes use TP8/PP2/EP8, DP4, GBS64, a 24+24 decoder layout,
one MTP layer, full/uniform2 recomputation and CPU-offloaded optimizer states.
They share the existing 256k recipe; all model, data and output paths are supplied
by the operator. Use the matching Relax checkout and patched Megatron image on
every worker.

## Environment

Long tensor offsets require the HC/grouped-QSA int64 fixes in this checkout and
`docker/patch/fla/0.5.2-conv-long-offsets.patch`. Both `docker/Dockerfile` and
`docker/Dockerfile.cu13` apply the FLA patch after the existing 0.5.2 convolution
patches. An older image needs this patch applied during image construction;
copying only the launch scripts is insufficient. No runtime monkeypatch is used.

Set `MODEL_DIR` to the HF checkpoint containing all 48 decoder layers and MTP
weights, `DATA_DIR` to the prepared SFT data, `SAVE_DIR` to the output root, and
`MEGATRON` to the installed Megatron repository. These paths must be available on
all nodes. The image supplies `RELAX_HYBRID_EP_COMMIT` as in the neighboring
Blackwell recipes. Run on an exclusive existing Ray cluster; the entrypoint
defaults to preserving cluster state.

## Variable-length data

```bash
bash examples/models/qwen3.8-flash-next/scripts/run-qwen38-qsa-sft-512k-64xgpu-blackwell.sh
```

The nominal sequence length is 524288, the soft packing budget is 262144, and
`data_pad_size_multiplier=4096` gives a 32768-token alignment under TP8. Oversized
singletons are retained; this command does **not** filter or truncate data at
512k. Audit the input maximum and its memory requirements before running.
Invalid multimodal records fail instead of being silently skipped.

GC and CUDA cache release run before gradient finalization on each update.
Evaluation splitting and checkpoint writes are disabled by default, matching
the validation runs. `SAVE_INTERVAL=100` enables checkpoint writes through the
shared recipe, but save/resume has not been validated for this delivery.

The historical full-corpus validation used a 655360 nominal limit and no eval
split, including five samples above 512k (maximum 617476 tokens):

```bash
SEQ_LENGTH=655360 NUM_EPOCH=2 \
  bash examples/models/qwen3.8-flash-next/scripts/run-qwen38-qsa-sft-512k-64xgpu-blackwell.sh \
  --num-rollout 98
```

The 98-step limit is specific to that 6236-row corpus at GBS64, not a universal
dataset traversal length. No private dataset or checkpoint is bundled.

## Fixed 512k validation

```bash
bash examples/models/qwen3.8-flash-next/scripts/run-qwen38-qsa-sft-512k-fixed-64xgpu-blackwell.sh
```

Supply equal-length prepared 512k examples. This entry defaults to ten updates,
packing budget 131072 and padding multiplier 16384. Long individual samples
remain intact. It sets `RELAX_MEMORY_RELEASE_FIRST_ATTEMPT_ONLY=1`, so both
finalization and optimizer cache releases happen only on rollout 0, step 0.
The first attempt is not necessarily the first successful optimizer update.

This mode is restricted to fresh fixed-shape runs: it rejects an existing
`SAVE_DIR/EXP_NAME` checkpoint path and explicit `--load` overrides before Ray
submission. The backend also rejects a nonzero initial rollout and requires
`empty_unused_memory_level=1`. Automatic global restart is disabled. Do not use
first-attempt-only release for variable-length data or checkpoint recovery.

Both entries support `DRY_RUN=1` to render effective arguments and worker
environment without Ray, GPU initialization or output-directory creation.

## Validation scope

The original `8311d07` worktrees with these kernel fixes and memory-release
policies passed the following on 64 Blackwell GPUs (TPS includes step wall time):

| Workload               | Completed validation              | TPS/card after warmup | MFU    | Minimum sampled free memory |
| ---------------------- | --------------------------------- | --------------------- | ------ | --------------------------- |
| Fixed 512k             | 10 steps; all rank records finite | 1716.53               | 3.675% | 20.22 GiB                   |
| Variable-length corpus | 98 steps; 6236 unique rows        | 900.85                | 1.906% | 44.66 GiB                   |

These are historical measurements, not GPU revalidation of the rebase onto
`57d8371`. The fixed run excludes step 0; the corpus run measures steps 1–97.
The corpus rate is not an isolated measurement of its \<=512k subset. The MFU
target of 7.5% was not reached. Sampled free memory is not an exact transient
minimum; these tests do not establish save/resume or long-context model quality.

This delivery excludes ChunkFlow, Ulysses and tensor-census instrumentation.
