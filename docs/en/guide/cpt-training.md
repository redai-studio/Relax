# Continued Pretraining (CPT)

Relax provides a text-only CPT mode through the existing SFT pipeline, primarily for Qwen3.5 Dense and MoE models on the Megatron backend. It reuses the existing model configuration and checkpoint conversion flow without introducing a separate training service.

```bash
--loss-type sft \
--sft-training-mode cpt \
--sft-cpt-template qwen3_5 \
--prompt-data /path/to/train.jsonl \
--input-key text \
--use-dynamic-batch-size \
--max-tokens-per-gpu 8192 \
--sft-oversize-strategy truncate_right
```

`--sft-training-mode` defaults to `sft`, so existing SFT and RL scripts do not need to change. CPT still uses the `causal_lm` objective; do not set `--task-type` or `--loss-type` to `cpt`.

For Qwen3.5-35B-A3B, you can use the provided launcher directly:

```bash
CPT_ROOT=/path/to/Relax-CPT \
HF_CHECKPOINT=/path/to/Qwen3.5-35B-A3B \
PROMPT_DATA=/path/to/cpt.jsonl \
bash scripts/training/cpt/run-qwen3.5-35B-A3B-cpt-8xgpu.sh
```

For a two-node, 16-GPU run, use `run-qwen3.5-35B-A3B-cpt-16xgpu.sh` from the same directory and provide the existing Ray cluster's `RAY_ADDRESS` and the rank-0 GPU node's `MASTER_ADDR`. The delivery image loads Megatron from `/root/Megatron-LM` by default and checks the expert grad norm fix before launch; no additional runtime patch mount is required.

## Data and Supervision Semantics

Each row represents one document. Existing JSONL and Parquet readers are supported. For example:

```json
{"text": "This is a complete domain document."}
```

The ms-swift pretraining data format is also supported by setting `--input-key messages`:

```json
{"messages": [{"role": "assistant", "content": "This is a complete domain document."}]}
```

If `--input-key` is omitted and the row does not contain the default `input` field, Relax tries `text` and then `messages`. An explicitly specified custom column name does not fall back. Each `messages` value must contain exactly one plain-text assistant message.

- Text is passed directly to the tokenizer without applying a chat template or adding system, user, assistant, or thinking prefixes.
- `--sft-cpt-template raw` (the default) preserves source whitespace and reasoning content, and supervises every token.
- `--sft-cpt-template qwen3_5` matches ms-swift's generative preprocessing for Qwen3.5: it trims surrounding whitespace, normalizes newlines around `<think>`, and uses `all+ignore_empty_think` to mask an empty thinking prefix and the whitespace immediately following it. Non-empty reasoning remains supervised; `<thinking>` is treated as ordinary text and is not equivalent to `<think>`.
- `raw` preserves the tokenizer's default special-token behavior and appends EOS when missing. `qwen3_5` splits text by loss weight, merges adjacent spans with the same weight, encodes each span separately, and appends EOS at the end. It does not append EOS again when the input already ends with EOS or `<|endoftext|>`, and it does not add BOS for Qwen3.5 text encoding.
- The data layer creates a mask with the same length as the tokens. Megatron shifts it left once within each sample and masks the final position, preventing the first token of the next document from becoming a prediction target. The first position of the Qwen3.5 template mask is 0, matching ms-swift's `labels[0] = -100` behavior.
- Training, an explicit validation set, and an automatically split validation set all use CPT encoding. PPL evaluation reuses SFT's token-weighted metric.

Text-only CPT does not accept images, video, audio, tool calls, multi-turn conversations, classification labels, custom dataset classes, or selective-loss settings. Empty text, a tokenizer without EOS, an invalid column name, and similar invalid input cause an error. `--sft-predict-interval` is intended for conversational generation evaluation and is not supported in this mode; use loss and PPL evaluation instead.

## Relationship to the ms-swift Baseline

The implementation follows `PretrainArguments` and the generation template in the adjacent ms-swift repository: CPT disables the chat template. Qwen3.5 argument initialization also changes `all` to `all+ignore_empty_think` automatically. Select the `qwen3_5` template to match this default behavior. The Relax training runtime does not require ms-swift to be installed.

Current implementation boundaries:

| Capability | Relax CPT |
| --- | --- |
| Full-parameter training and existing Qwen3.5 model support | Reuses SFT/Megatron |
| Streaming reads, prefetching, and deterministic shuffling | Reuses the SFT dataset |
| Packing, CP, and other parallel configurations | Reuses the Megatron SFT data path and its limitations |
| Oversized documents | Supports the existing `keep`, `skip`, `truncate_left`, and `truncate_right` strategies |
| ms-swift `truncation_strategy=split` | Not implemented; split long documents offline, with one segment per row |
| ms-swift `cached_dataset` export format | Not read directly; use JSONL or Parquet supported by Relax |
| Multimodal CPT | Not implemented |

`capacity = max_tokens_per_gpu × context_parallel_size`. The example uses right truncation, so an oversized document loses its tail and may also lose its final EOS. To preserve the complete corpus, split documents offline and reserve space for EOS. `keep` can exceed dynamic-batching capacity; `skip` changes the number of rows actually consumed and may cause some data to be reread or skipped after checkpoint recovery. For stable continued training, use pre-split data with one training sample per row.

An explicit single-message `loss_scale=1` overrides empty-think masking according to ms-swift semantics; other non-unit weights remain unsupported. Qwen3.5 truncation preserves native image and video pad tokens and resets the first label after truncation. This is encoding-boundary compatibility only and does not mean multimodal CPT is supported.

Use the Qwen3.5 template only with the corresponding models. Do not apply Gemma's thinking scaffold or SFT's assistant-only mask directly to CPT. The current compatibility implementation also preserves ms-swift's boundary behavior: text before `<think>` may be dropped during normalization, and consecutive empty think blocks may also mask the following answer. These behaviors result from splitting on the first closing tag and matching the regular-expression segments a second time; check for them during data preparation.
