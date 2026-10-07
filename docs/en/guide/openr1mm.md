# OpenR1MM Reward and Evaluation

## Overview

`openr1mm_accuracy_format` adds independent answer accuracy and format scores, each in `[0, 1]`, for a total reward in `[0, 2]`. Its implementation is `relax/engine/rewards/openr1mm_accuracy_format.py`. It preserves the experimental XML/prefilled-think scoring behavior; the existing `openr1mm` reward and its native-channel support remain unchanged.

Accuracy compares the final response and reference answer, not intermediate reasoning. Explicit multiple-choice labels are checked before mathematical equivalence with `math_verify`. Format requires one nonempty `<think>...</think>` block followed by one nonempty `<answer>...</answer>` block. A prompt-prefilled `<think>` opener and an optional trailing `<|im_end|>` are supported. Accuracy and format are independent: a wrong answer with valid format earns 1, as can a correct answer without that format. This is not a general semantic judge; mathematical extraction and text normalization have limits.

Evaluation uses separate local 0/1 accuracy on MathVista-testmini and MMMU validation. It calls no external model, adds no format point, and does not run the training dynamic sampling filter. The [upstream Open-R1-Multimodal evaluation](https://github.com/EvolvingLMMs-Lab/open-r1-multimodal/blob/main/local_scripts/lmms_eval_qwen2vl.sh) uses a different lmms-eval/GPT-4o protocol, including MMMU-Pro. These local scores are not directly comparable to that protocol.

## Prepare Evaluation Data

Run from the repository root in the existing Relax environment (`huggingface_hub`, `pyarrow`, and Pillow are used). Keep images and converted data on storage shared by the training workers. Set `DATA_DIR` to your dataset root.

First convert the OpenR1MM training parquet if necessary; the overlap audit expects its `image` column to contain a list of encoded image bytes:

```bash
python scripts/tools/process_openr1.py \
  --input-dir "$DATA_DIR/multimodal-open-r1-8k-verified/data/train-00000-of-00001.parquet" \
  --output-dir "$DATA_DIR/multimodal-open-r1-8k-verified/data/train-00000-of-00001_converted_noextract.parquet"
```

Download only the required evaluation splits, then convert and audit against the actual training parquet:

```bash
python examples/openr1mm/download_eval.py --output "$DATA_DIR/openr1mm-eval"
python examples/openr1mm/prepare_eval.py \
  --source "$DATA_DIR/openr1mm-eval/source" \
  --train "$DATA_DIR/multimodal-open-r1-8k-verified/data/train-00000-of-00001_converted_noextract.parquet" \
  --output "$DATA_DIR/openr1mm-eval"
```

The download defaults pin MathVista to `2b6ad69445fbb5695c9b165475e8decdbeb97747` and MMMU to `876ce5cb130f7f7e290ce4d9984357737d4db5cf`. Override with `--mathvista-revision` / `--mmmu-revision` if needed; `sources.json` records resolved revisions. The downloader rejects nonempty source directories with unknown or different revisions; use a fresh `--output` directory when changing revisions. Interrupted downloads at the same revision can resume. The converter emits:

- `mathvista_testmini.parquet` and `mmmu_validation.parquet`: full converted splits.
- `mathvista_testmini_disjoint.parquet` and `mmmu_validation_disjoint.parquet`: rows with no exact training-image overlap.
- `overlap-report.json`: full/disjoint counts and excluded source IDs.

Each row contains `prompt`, `image`, JSON-encoded `label`, and source `id`. MMMU image references in questions and options are expanded in occurrence order. Overlap means identical decoded RGB pixels **and dimensions**, even across different encodings. A row is excluded if any referenced image overlaps. This does not remove resized, near-duplicate, or semantically overlapping problems.

One preparation against the verified 8k training set retained 997/1000 MathVista and 900/900 MMMU rows. Counts depend on the training file. Processor/tokenizer prompt-length filtering can reduce them further (that run evaluated 992 MathVista rows). Always check actual evaluated counts, not just parquet counts.

## Training and Evaluation Configuration

In your model-specific training recipe, use `--rm-type openr1mm_accuracy_format`. To add evaluation, append these flags to the training command (retain your existing topology, optimizer, and dataset settings):

```bash
--rm-type openr1mm_accuracy_format \
--custom-rm-path examples.openr1mm.eval_reward.reward \
--eval-config examples/openr1mm/eval.yaml \
--eval-interval 20 \
--eval-temperature 0 \
--n-samples-per-eval-prompt 1 \
--eval-max-prompt-len 8192 \
--eval-max-response-len 8192 \
--eval-max-context-len 16384 \
--sglang-context-length 16384
```

Leave `--skip-eval-before-train` unset to evaluate before the first update. The YAML selects the two disjoint parquets with greedy decoding and one response per question. The engine context limit must accommodate evaluation; raising it does not raise the training rollout limits.

`eval.yaml` resolves `${oc.env:DATA_DIR}`. Export `DATA_DIR` and ensure it is in Ray runtime `env_vars` on the driver and workers. Use an absolute `--eval-config` path if the job's working directory differs from the repository root. The repository and converted files must be visible to workers. For training, preserve your model's chat template and multimodal mapping (`--input-key prompt --label-key label --multimodal-keys '{"image":"image"}' --apply-chat-template`).

The custom reward routes samples tagged `eval_benchmark` by this YAML to local accuracy; ordinary training samples still call `get_openr1mm_accuracy_format_reward`. Do not attach this reserved metadata to training samples. DAPO filtering may still be enabled for training with `--dynamic-sampling-filter-path relax.engine.filters.dynamic_sampling_filters.check_reward_nonzero_std`; it is separate from evaluation.

## Scoring and Troubleshooting

The evaluation scorer isolates the final answer, accepts a choice letter or exact option text, and also accepts `B. option text` **only when both the letter and option text match**. Conflicting letters/content and lists of alternatives must not receive credit. This avoids the observed false drop when a model changes from `B` to `B. $7`. Free-form scoring uses normalized exact text, candidate aliases, numeric comparison with MathVista precision, and list comparison; units, prose, and symbolic equivalence are not generally normalized.

When a score drops, compare identical prompts/labels, effective sample counts, truncation, and final-answer styles before diagnosing model regression. Evaluate historical outputs with the same scorer version; changing only the newest point produces an incomparable curve. Training reward can exceed 1 because it includes format, while these eval metrics cannot.

Run CPU regression checks with:

```bash
python -m pytest tests/engine/rewards/test_openr1mm_accuracy_format.py \
  tests/examples/test_openr1mm_eval_reward.py tests/examples/test_openr1mm_prepare_eval.py
```

## Next Steps

- [Customize Training](./customize-training.md)
- [Dataset Design](./dataset-design.md)
