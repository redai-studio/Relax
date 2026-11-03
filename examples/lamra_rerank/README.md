# Qwen2-VL LamRA reranking

Train a generative ranker through Relax's existing multimodal SFT and Megatron
Bridge path. For retrieval followed by reranking, initialize rank training from
the completed retrieval model's **fully merged Qwen2-VL-2B checkpoint**, with a
fresh rank LoRA adapter, optimizer and scheduler. The original
Qwen2-VL-2B-Instruct checkpoint is also supported as a base. A query produces three supervised sequences: a positive
candidate with `Yes`, a negative candidate with `No`, and a list of 2–5 candidates
with an `Ans: (n)` answer. The vision encoder and projector stay frozen.

The recipe uses language-only independent Q/K/V/O/gate/up/down LoRA adapters
(`r=128`, `alpha=256`, dropout `0.05`), BF16, seed 42, LR `4e-5`, cosine decay,
3% warmup, and one padded epoch. Eight GPUs use DP with TP=PP=CP=1. No additional
framework arguments or dependencies are required.

## Prepare training data

Set `MODEL_DIR` to a directory containing `Qwen2-VL-2B-Instruct`, `DATA_DIR` to
your dataset root, and `SAVE_DIR` to persistent checkpoint storage. M-BEIR image
references must resolve beneath the supplied image root on all workers.

`Qwen2-VL-2B-Instruct` is the checkpoint directory name expected by the launcher;
it can contain the fully merged retrieval checkpoint and its tokenizer. Set
`QWEN2_VL_VOCAB_SIZE` to that base's vocabulary size, including any added tokens.
Use a new `EXP_NAME` for a fresh rank stage, and use the same base and tokenizer
for data preparation, training and export.

```bash
python examples/lamra_rerank/tools/prepare_data.py \
  --queries "${DATA_DIR}/M-BEIR/query/union_train/mbeir_union_up_train.jsonl" \
  --candidates "${DATA_DIR}/M-BEIR/cand_pool/global/mbeir_union_train_cand_pool.jsonl" \
  --instructions "${DATA_DIR}/M-BEIR/instructions/query_instructions.tsv" \
  --rerank-data "${DATA_DIR}/rerank_data_for_training/rerank_data_all.json" \
  --image-root "${DATA_DIR}/M-BEIR" \
  --tokenizer "${MODEL_DIR}/Qwen2-VL-2B-Instruct" \
  --output "${DATA_DIR}/rerank/train.jsonl"
```

Preparation preserves the reference prompt and normalizes each text field before
tokenize → truncate to 256 tokens → decode. It samples one deterministic triplet
per training query. Both negative branches exclude **all known positives**.
Queries with no true negative are skipped and listed in the manifest; a shortage
of listwise negatives reduces the number of choices. This deliberately avoids
the reference fallback that can label a positive candidate `No`.

Outputs include the JSONL, row counts, source/output hashes, skipped query IDs and
unique image paths. Preparation does not decode or validate every image. A missing
or invalid image fails training rather than silently shortening a batch. Use
`--max-queries` only for an explicit small preparation audit.

`LamraRerankDataset` reuses the shared SFT pipeline while preserving the rank
collator's supervision boundary: the assistant header, answer, end token and
following newline contribute to loss. For the original tokenizer, the two
pointwise answers have 6 supervised tokens each and the listwise answer has 10.
Images follow the reference RGB conversion and factor-28 resize with 4–300 visual
tokens. The script pins the PIL processor through `RELAX_HF_PROCESSOR_USE_FAST=0`,
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
RAY_ADDRESS="http://your-ray-head:8265" \
  bash scripts/entrypoint/ray-job.sh \
  examples/lamra_rerank/scripts/run-qwen2-vl-2B-rerank-8xgpu.sh
```

The script derives the number of updates from `train.manifest.json` and uses
`--calculate-per-token-loss`. The default batch is **192 supervised sequences**,
corresponding in count to 64 query triplets. Relax shuffles individual sequences;
its update membership and sample order differ from the reference HF query
sampler. This recipe does not promise numerical replay of that training run.

`PROJECT_NAME` defaults to `Relax/sft/lamra-rerank`; `EXP_NAME` defaults to
`qwen2-vl-2b-lamra-rank-r128-dp8`. ClearML and metrics are enabled, TensorBoard is
disabled. `SAVE_INTERVAL` defaults to 1000 updates and the last two native
checkpoints are retained. The final update is also saved. Native `--load` and
`--save` both use `${SAVE_DIR}/${EXP_NAME}`: keep these values stable to resume,
or explicitly change `EXP_NAME` for a new run.

Submission follows the existing entrypoint and blocks by default. Set
`RAY_NO_WAIT=1` when nonblocking submission is needed, then inspect the actual Ray
job logs and training progress. The recipe uses pure SFT without colocate or a
rollout service and overlaps input preparation with training.

## Export for evaluation

Export the final native checkpoint on CPU. Set `EXPECTED_UPDATES` to the completed
update count and `ITERATION` to its checkpoint iteration (`EXPECTED_UPDATES - 1`).
Pass the exact FP32 base checkpoint used to initialize rank training as `--base`.

```bash
CUDA_VISIBLE_DEVICES="" python examples/lamra_rerank/tools/export_checkpoint.py \
  --checkpoint "${SAVE_DIR}/${EXP_NAME}/iter_$(printf '%07d' "${ITERATION}")" \
  --expected-updates "${EXPECTED_UPDATES}" \
  --base "${MODEL_DIR}/Qwen2-VL-2B-Instruct" \
  --adapter-output "${SAVE_DIR}/${EXP_NAME}/final_adapter" \
  --output "${SAVE_DIR}/${EXP_NAME}/merged"
```

The exporter checks scheduler/update counts, extracts every canonical adapter,
verifies the FP32 merge equation and unchanged frozen parameters, and reloads the
serialized HF model exactly. Partial or completed export directories require
inspection before reuse.

## Evaluate fixed retrieval candidates

Supply the retrieval model's saved query IDs, top candidates and raw scores,
along with its `model_identity.json`. The following example evaluates all
Fashion200K task0 queries, each with 50 fixed candidates. Retrieval candidates
must come from the same query/candidate IDs and data split as the qrels.

```bash
python examples/lamra_rerank/tools/prepare_eval.py \
  --queries "${DATA_DIR}/M-BEIR/query/test/mbeir_fashion200k_task0_test.jsonl" \
  --candidates "${DATA_DIR}/M-BEIR/cand_pool/global/mbeir_union_test_cand_pool.jsonl" \
  --instructions "${DATA_DIR}/M-BEIR/instructions/query_instructions.tsv" \
  --qrels "${DATA_DIR}/M-BEIR/qrels/test/mbeir_fashion200k_task0_test_qrels.txt" \
  --query-names "${DATA_DIR}/retrieval/mbeir_fashion200k_task0_test_model_query_names.json" \
  --candidate-names "${DATA_DIR}/retrieval/mbeir_fashion200k_task0_test_model_cand_names.json" \
  --scores "${DATA_DIR}/retrieval/mbeir_fashion200k_task0_test_model_scores.json" \
  --retrieval-identity "${DATA_DIR}/retrieval/model_identity.json" \
  --image-root "${DATA_DIR}/M-BEIR" \
  --output "${DATA_DIR}/rerank/fashion200k_task0_pairs.jsonl"

torchrun --standalone --nproc_per_node=8 examples/lamra_rerank/tools/evaluate.py \
  --model "${SAVE_DIR}/${EXP_NAME}/merged" \
  --pairs "${DATA_DIR}/rerank/fashion200k_task0_pairs.jsonl" \
  --output "${SAVE_DIR}/${EXP_NAME}/evaluation/fashion200k_task0" \
  --batch-size 4

python examples/lamra_rerank/tools/summarize.py \
  --pairs "${DATA_DIR}/rerank/fashion200k_task0_pairs.jsonl" \
  --scores "${SAVE_DIR}/${EXP_NAME}/evaluation/fashion200k_task0" \
  --world-size 8 \
  --output "${SAVE_DIR}/${EXP_NAME}/evaluation/fashion200k_task0_metrics.json"
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
