# LamRA Rank SFT and evaluation

Train a generative ranker through Relax's existing multimodal SFT and Megatron
Bridge path. For the [three-stage workflow](../seq_contrast_sft/README.md),
initialize from the merged Ret SFT checkpoint with a fresh Rank LoRA adapter,
optimizer and scheduler. The launcher also accepts an original supported HF
base for a standalone Rank run. A query produces a positive pair labeled `Yes`,
a negative pair labeled `No`, and a list of 2–5 candidates with an `Ans: (n)`
answer. The vision encoder and projector stay frozen.

Supported models are Qwen2-VL-2B-Instruct and dense Qwen3.5-0.8B, 2B, 4B and 9B.
Qwen2-VL targets independent Q/K/V/O/gate/up/down LoRA projections. Qwen3.5
targets language attention output and MLP down projections, with thinking mode
disabled in training and scoring. Both use `r=128`, `alpha=256`, dropout `0.05`,
BF16, seed 42, LR `4e-5`, cosine decay, 3% warmup and one padded epoch. The
launcher defaults to eight DP GPUs with TP=PP=CP=1.

## Prepare training data

Use the variables from the Ret README. `RANK_BASE` points directly to the HF
directory with `config.json` and the tokenizer. Use this directory for data
preparation, training and export. All workers must be able to resolve the M-BEIR
images beneath `IMAGE_ROOT`.

```bash
RANK_BASE="${RET_MODEL}"
EXP_NAME="${EXP_PREFIX}-rank-sft"
RANK_MODEL="${SAVE_DIR}/${EXP_NAME}/merged"

python examples/lamra_rerank/tools/prepare_data.py \
  --queries "${TRAIN_QUERIES}" --candidates "${TRAIN_CANDIDATES}" \
  --instructions "${INSTRUCTIONS}" --image-root "${IMAGE_ROOT}" \
  --rerank-data "${TRAIN_RETRIEVAL}/rankings.json" \
  --tokenizer "${RANK_BASE}" \
  --output "${DATA_DIR}/rerank/train.jsonl"
```

Preparation preserves the reference prompt and normalizes each text field before
tokenize → truncate to 256 tokens → decode. It samples one deterministic triplet
per training query. Both negative branches exclude **all known positives**.
Queries with no true negative are skipped and listed in the manifest; a shortage
of listwise negatives reduces the number of choices. This deliberately avoids
the reference fallback that can label a positive candidate `No`.

Generate the training top100 and fixed test rankings in the
[Ret workflow](../seq_contrast_sft/README.md).

Outputs include the JSONL, row counts, source/output hashes, skipped query IDs and
unique image paths. Preparation does not decode or validate every image. A missing
or invalid image fails training rather than silently shortening a batch. Use
`--max-queries` only for an explicit small preparation audit.

`LamraRerankDataset` reuses the shared SFT pipeline while preserving the rank
collator's supervision boundary: the assistant header, answer, end token and
following newline contribute to loss. With the Qwen2-VL reference tokenizer, the two
pointwise answers have 6 supervised tokens each and the listwise answer has 10.
The recipe converts images to RGB and applies the factor-28 reference resize
before passing them to the model's processor. The script pins the PIL processor
through `RELAX_HF_PROCESSOR_USE_FAST=0`,
including spawned processor workers. Full sequences are kept; the reference's `model_max_length=2048` did not
actually truncate its processor output.

Image paths cross the processor-pool boundary; decoding, RGB conversion and
reference resizing run inside its workers. The recipe uses 32 prefetch threads
and chunks of 64 samples. The launcher defaults `TOKENIZERS_PARALLELISM=false`;
set the same value in the Ray worker runtime configuration because an existing
worker setting takes precedence over environment passthrough. This avoids a
large tokenizer thread pool in every process and preserves model inputs.
Locating byte-identical input images on a node-local disk can reduce
small-file latency; keep their durable sources and validate every copied image.

## Train and resume

```bash
MODEL_DIR="${RANK_BASE}" EXP_NAME="${EXP_NAME}" \
  bash examples/lamra_rerank/scripts/run-qwen2-vl-2B-rerank-8xgpu.sh
```

The script derives the number of updates from `train.manifest.json` and uses
`--calculate-per-token-loss`. The default batch is **192 supervised sequences**,
corresponding in count to 64 query triplets. Relax shuffles individual sequences;
its update membership and sample order differ from the reference HF query
sampler. This recipe does not promise numerical replay of that training run.

The launcher default for `EXP_NAME` is `<model>-lamra-rank-r128-dp8`. Metrics are
enabled and TensorBoard is disabled. `SAVE_INTERVAL` defaults to 1000 updates and the last two native
checkpoints are retained. The final update is also saved. Native `--load` and
`--save` both use `${SAVE_DIR}/${EXP_NAME}`: keep these values stable to resume,
or explicitly change `EXP_NAME` for a new run.

Submission follows the existing entrypoint and blocks by default. Set
`RAY_NO_WAIT=1` when nonblocking submission is needed, then inspect the actual Ray
job logs and training progress. The recipe uses pure SFT without colocate or a
rollout service and overlaps input preparation with training.

## Export for evaluation

Export the final native checkpoint on CPU. Set `FINAL_CHECKPOINT` to the final
`iter_*` directory under `${SAVE_DIR}/${EXP_NAME}` and `EXPECTED_UPDATES` to the
completed update count. Pass `RANK_BASE`, the same HF directory loaded during
training, as `--base`.

```bash
CUDA_VISIBLE_DEVICES="" python examples/lamra_rerank/tools/export_checkpoint.py \
  --checkpoint "${FINAL_CHECKPOINT}" \
  --expected-updates "${EXPECTED_UPDATES}" \
  --base "${RANK_BASE}" \
  --adapter-output "${SAVE_DIR}/${EXP_NAME}/final_adapter" \
  --output "${RANK_MODEL}"
```

The exporter checks scheduler/update counts, extracts every canonical adapter,
verifies the FP32 merge equation and unchanged frozen parameters, and reloads the
serialized HF model exactly. Partial or completed export directories require
inspection before reuse.

## Evaluate fixed retrieval candidates

Use the four files produced by the Ret top50 command in `TEST_RETRIEVAL`, along
with the same test queries, candidate pool and qrels. Set `EVAL_GPUS` to the number
of local GPUs available for scoring. Each worker loads its own Rank model replica.
Use a fresh `RANK_PAIRS` path and `RANK_EVAL` directory for each evaluation.

```bash
EVAL_GPUS=8
RANK_EVAL="${SAVE_DIR}/${EXP_NAME}/evaluation"
RANK_PAIRS="${DATA_DIR}/rerank/test_pairs.jsonl"

python examples/lamra_rerank/tools/prepare_eval.py \
  --queries "${TEST_QUERIES}" --candidates "${TEST_CANDIDATES}" \
  --instructions "${INSTRUCTIONS}" --qrels "${TEST_QRELS}" \
  --query-names "${TEST_RETRIEVAL}/query_names.json" \
  --candidate-names "${TEST_RETRIEVAL}/candidate_names.json" \
  --scores "${TEST_RETRIEVAL}/scores.json" \
  --retrieval-identity "${TEST_RETRIEVAL}/model_identity.json" \
  --image-root "${IMAGE_ROOT}" --output "${RANK_PAIRS}"

torchrun --standalone --nproc_per_node="${EVAL_GPUS}" examples/lamra_rerank/tools/evaluate.py \
  --model "${RANK_MODEL}" --pairs "${RANK_PAIRS}" --output "${RANK_EVAL}" \
  --batch-size 4

python examples/lamra_rerank/tools/summarize.py \
  --pairs "${RANK_PAIRS}" --scores "${RANK_EVAL}" \
  --world-size "${EVAL_GPUS}" --output "${RANK_EVAL}/metrics.json"
```

Scoring uses the first generated-token logits and
`softmax([logit_Yes, logit_No])[Yes]`, retaining generation-config logit processing
while generating only one token. Evaluation uses untruncated query/candidate
text, strict BF16 model loading and the same image preprocessing as training.
The report includes retrieval-only, rank-only and fixed equal-score fusion
(`ret_score + p_yes`) Recall@1/5/10/20/50. It validates every pair identity and
the complete test coverage. No coefficient is selected on test results. The
MSCOCO-specific upstream script weights differ from equal fusion; they are not
the Fashion200K protocol used here.
