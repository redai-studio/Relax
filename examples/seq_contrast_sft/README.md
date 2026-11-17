# Sequence contrastive SFT

This example follows `examples/seq_cls_sft`: select a task with
`--loss-type sft --task-type embedding --embedding-loss-type infonce` and reuse the existing SFT producer,
TransferQueue, Actor, Megatron forward/backward, optimizer, scheduler and native
checkpoint path. It adds a pair dataset and InfoNCE loss, without a separate
trainer or Actor implementation.

The embedding task pools one hidden state per sequence. An embedding pair
dataset supplies one `query` and one `positive` branch per row, each with
`input_ids` and optional `embedding_position` (a zero-based index; the
default is the last real token). The LamRA example datasets supply the
position before the first `<emb>`. InfoNCE consumes the pooled vectors.
For an older LamRA launch command, use the updated task flags and example
dataset path. The exporter accepts checkpoints saved under
either task name.

Qwen2-VL uses a dedicated Bridge in `relax/models/qwen2_vl`. Megatron Core
runs the language model with Qwen2-VL's own vision tower and multimodal
position encoding.

The workflow supports Qwen2-VL-2B-Instruct and dense Qwen3.5-0.8B, 2B, 4B, and 9B.
The launchers keep their original `run-qwen2-vl-*` filenames and select the
architecture using the `config.json` in the `MODEL_DIR` you pass.

## Inputs

- `MODEL_DIR`: the HF checkpoint directory for the current stage. Ret PT uses
  the prepared checkpoint containing `<emb>` in the tokenizer and its corresponding
  embedding row. Ret SFT uses the merged Ret PT export.
- `${DATA_DIR}/nli/pairs.jsonl`: one complete query/positive pair per row. Use the
  native four-message form below, or `query`/`positive` dictionaries each with
  pretokenized `input_ids`.

```json
{"messages":[{"role":"user","content":"A query"},{"role":"assistant","content":"<emb>"},{"role":"user","content":"Its positive"},{"role":"assistant","content":"<emb>"}]}
```

Raw conversations use the existing SFT processor pool. This NLI adapter is
text-only; it does not load raw images or hard negatives. Each branch must fit
`seq_length`, and overlength or malformed rows raise an error. Rows are never
silently filtered or truncated. The final batch repeats the current epoch's
head; native checkpoint resume reconstructs that padded-epoch cursor.

## Launch

On an already provisioned, exclusively allocated Ray cluster, provide
`MODEL_DIR`, `DATA_DIR`, a **persistent** `SAVE_DIR` and the Ray Jobs HTTP
`RAY_ADDRESS`, then run from the repository root. The SFT and Rank stages also
need their prepared data and image paths accessible from the workers.

```bash
bash examples/seq_contrast_sft/scripts/run-qwen2-vl-2B-nli-pt-8xgpu.sh
```

The script requests eight GPUs, TP=PP=CP=1 and a 512-**pair** global batch. The
default 1,078 updates correspond to two padded epochs only for 275,601 rows.
For other inputs, explicitly override `--num-rollout`; the custom dataset uses
this update count rather than `--num-epoch`.

The example selects Bridge's native **Kaiming-uniform LoRA A initialization**
(`RELAX_LORA_A_INIT_METHOD=kaiming`); LoRA B retains Bridge's zero initialization.
The initializer is included in the Ray runtime environment. Set
`RELAX_LORA_A_INIT_METHOD=xavier` explicitly to use Bridge's Xavier option, with a
new `EXP_NAME`. Other Relax recipes keep their existing initializer defaults.

The native checkpoint path is `${SAVE_DIR}/${EXP_NAME}` for both save and load,
with an interval of 200 updates and retention of two checkpoints. Reuse the same
path to resume the same experiment; use a new `EXP_NAME` for a different run.
The default Ret PT experiment name is `<model>-nli-bridge-lora-kaiming-dp8`. Start this
recipe in a new directory: historical precision-profile checkpoints used a
different norm layout and optimizer/RNG settings and are not supported resume
inputs for this recipe.
Each launch writes a local log while the resume path stays fixed. Submission blocks unless
`RAY_NO_WAIT` is explicitly nonempty.

## Training contract and validation limits

- Pair rows stay atomic through DP sampling, then expand to adjacent
  `[query, positive]` sequences. Pool the hidden state preceding the first
  `<emb>`, using a frozen vocabulary head/embedding and vision tower.
- One optimizer update contains the complete contrastive negative set in a
  **single packed microbatch**. Exceeding `max_tokens_per_gpu` fails rather than
  splitting InfoNCE into independent objectives. The configured token budget is
  a packing limit, **not a proven memory fit** on a particular GPU.
- The loss is one-way mean cosine InfoNCE at temperature 0.05. DP gathers remote
  detached features, retains local gradients, and preserves the native averaged
  `1/DP` parameter-gradient scale through the shared Megatron loss wrapper.
- Initial support is synchronous, DP-only, THD training. Model parallelism,
  dynamic CP, independent microbatch accumulation, periodic SFT eval/predict,
  CPU prepack, and in-training `--save-hf` are rejected. Resume from native
  checkpoints and use the offline exporter below to hand off weights.
- The Ret stages use Bridge's **CanonicalLoRA**. Qwen2-VL targets independent
  Q/K/V, output, gate/up and down projections. Qwen3.5 targets language attention
  output and MLP down projections. Its model dimensions, mRoPE and embedding
  tying come from the HF checkpoint.
- Data ordering uses the existing `NLIPairDataset`/`IndexManager` path. The script
  retains `--data-parallel-random-init` for per-rank RNG state. There are no
  reference-order replay, padded/eager projection overrides, HF norm/attention
  replacements, or custom optimizer/gradient-communication hooks.

CPU tests exercise the production iterator/loss dispatch, pair/resume contracts,
real two-process Gloo loss/gradients, HF vision embeddings/positions and mapping
key coverage. A submission test checks that the example passes the initializer
through Ray's runtime environment while preserving the existing runtime options.
The language boundary in the CPU vision tests is a recording module, not MCore
attention.

The three-stage workflow has small-data single-node coverage. Full benchmark
scores, multi-node training and exact GPU checkpoint-resume behavior require
separate validation for the selected configuration.

## Three-stage workflow

### 1. Prepare Ret PT inputs

Set `BASE` to the original HF checkpoint for either supported model family and
`NLI_CSV` to a CSV with `sent0` and `sent1` columns. Use an `EXP_PREFIX` unique
to the model and run, plus a separate `DATA_DIR` for each run's prepared files.
Keep the original data and images at their source paths. Keep the derived
directory variables in the same shell for the later stages.

```bash
export DATA_DIR SAVE_DIR RAY_ADDRESS

EMB_MODEL="${SAVE_DIR}/${EXP_PREFIX}-emb"
PT_MODEL="${SAVE_DIR}/${EXP_PREFIX}-ret-pt/merged"
RET_MODEL="${SAVE_DIR}/${EXP_PREFIX}-ret-sft/merged"
TRAIN_RETRIEVAL="${SAVE_DIR}/${EXP_PREFIX}-ret-sft/retrieval/train"
TEST_RETRIEVAL="${SAVE_DIR}/${EXP_PREFIX}-ret-sft/retrieval/test"

python examples/seq_contrast_sft/tools/prepare_nli.py \
  --base "${BASE}" --input "${NLI_CSV}" \
  --output-model "${EMB_MODEL}" \
  --pairs "${DATA_DIR}/nli/pairs.jsonl"
MODEL_DIR="${EMB_MODEL}" EXP_NAME="${EXP_PREFIX}-ret-pt" \
  bash examples/seq_contrast_sft/scripts/run-qwen2-vl-2B-nli-pt-8xgpu.sh
```

### 2. Train Ret SFT and mine candidates

Use the [offline exporter](#offline-export) to merge Ret PT against `EMB_MODEL`
and write `PT_MODEL`. Set `TRAIN_QUERIES` and `TRAIN_CANDIDATES` to the M-BEIR
training query and candidate JSONLs, `INSTRUCTIONS` to the query instruction TSV,
and `IMAGE_ROOT` to the directory from which M-BEIR image paths resolve. The
preparer writes both `pairs.jsonl` and `pairs.meta.json` for the launcher.

```bash
python examples/seq_contrast_sft/tools/prepare_mbeir.py \
  --queries "${TRAIN_QUERIES}" --candidates "${TRAIN_CANDIDATES}" \
  --instructions "${INSTRUCTIONS}" --image-root "${IMAGE_ROOT}" \
  --tokenizer "${PT_MODEL}" --output "${DATA_DIR}/mbeir/pairs.jsonl"
MODEL_DIR="${PT_MODEL}" EXP_NAME="${EXP_PREFIX}-ret-sft" \
  bash examples/seq_contrast_sft/scripts/run-qwen2-vl-2B-mbeir-sft-8xgpu.sh
```

The data loader shuffles queries and samples a positive and instruction at each
visit. The launcher computes updates from the prepared query count and batch
size. The default batch size is 512; 1,332,790 queries take 2,604 updates.
Images are checked when read by the processor.

Export Ret SFT against `PT_MODEL` to `RET_MODEL`. Use `RET_MODEL` to mine the
training top100 and produce a separate fixed top50 for held-out evaluation. Set
`TEST_QUERIES`, `TEST_CANDIDATES`, and `TEST_QRELS` to the files for the same test
split. The training and test candidate pools need at least 100 and 50 entries,
respectively.

```bash
python examples/seq_contrast_sft/tools/retrieve.py \
  --model "${RET_MODEL}" --queries "${TRAIN_QUERIES}" \
  --candidates "${TRAIN_CANDIDATES}" --instructions "${INSTRUCTIONS}" \
  --image-root "${IMAGE_ROOT}" --top-k 100 --output "${TRAIN_RETRIEVAL}"

python examples/seq_contrast_sft/tools/retrieve.py \
  --model "${RET_MODEL}" --queries "${TEST_QUERIES}" \
  --candidates "${TEST_CANDIDATES}" --instructions "${INSTRUCTIONS}" \
  --qrels "${TEST_QRELS}" --image-root "${IMAGE_ROOT}" \
  --top-k 50 --output "${TEST_RETRIEVAL}"
```

Each retrieval command writes to a new output directory. The held-out Ret Recall
is in `${TEST_RETRIEVAL}/metrics.json`.

### 3. Train and evaluate Rank SFT

Use `${TRAIN_RETRIEVAL}/rankings.json` as `--rerank-data` in the
[Rank SFT recipe](../lamra_rerank/README.md#prepare-training-data). This prepares
pointwise and listwise training JSONL before submission. Set `RANK_BASE` to
`RET_MODEL` for the three-stage workflow and use a fresh Rank experiment name.
The [Rank evaluation](../lamra_rerank/README.md#evaluate-fixed-retrieval-candidates)
uses the files in `TEST_RETRIEVAL` and the exported Rank model.

### Offline export

Use the same command for all three stages. `BASE_FOR_STAGE` is the exact HF
checkpoint the stage loaded. Set `FINAL_CHECKPOINT` to the final `iter_*`
directory under `${SAVE_DIR}/${STAGE_EXP_NAME}` and `UPDATES` to the number of
optimizer updates. Use fresh `ADAPTER_OUT` and `MERGED_OUT` paths.

| Stage    | `STAGE_EXP_NAME`         | `BASE_FOR_STAGE` | `MERGED_OUT`    |
| -------- | ------------------------ | ---------------- | --------------- |
| Ret PT   | `${EXP_PREFIX}-ret-pt`   | `${EMB_MODEL}`   | `${PT_MODEL}`   |
| Ret SFT  | `${EXP_PREFIX}-ret-sft`  | `${PT_MODEL}`    | `${RET_MODEL}`  |
| Rank SFT | `${EXP_PREFIX}-rank-sft` | `${RANK_BASE}`   | `${RANK_MODEL}` |

```bash
ADAPTER_OUT="${SAVE_DIR}/${STAGE_EXP_NAME}/final_adapter"
CUDA_VISIBLE_DEVICES="" python examples/lamra_rerank/tools/export_checkpoint.py \
  --base "${BASE_FOR_STAGE}" --checkpoint "${FINAL_CHECKPOINT}" \
  --expected-updates "${UPDATES}" --adapter-output "${ADAPTER_OUT}" --output "${MERGED_OUT}"
```

The exporter reads the saved LoRA tensors and verifies the FP32 merge and HF
reload. Resume training from the native checkpoint. Use the merged HF directory
for the next stage or evaluation.
