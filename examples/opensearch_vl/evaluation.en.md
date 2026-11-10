# Post-training model evaluation

[中文](evaluation.md)

Configure all tool services before running a full benchmark. QS and MaaS credentials alone do not provide search, image search, or OCR. The runner rejects missing configuration. A bounded run that explicitly allows missing tools is a smoke test.

## Evaluation protocol

Compare the initial Qwen3.5-4B model with the Relax SFT model:

| Dataset            | Scope                            | Metric |
| ------------------ | -------------------------------- | ------ |
| SimpleVQA          | All 1,013 English test questions | Pass@1 |
| VDR-Bench-testmini | All 500 questions                | Pass@1 |

The [original benchmark](https://arxiv.org/html/2605.05185v1#S5) lets the model call tools and produce a final answer, which GPT-4o grades using the VDR rubric. This recipe reuses its dispatcher, system prompt and judge prompt, with an OpenAI-compatible API for the evaluated model. Run each question once; accuracy is correct answers divided by all questions.

Use temperature 0, a 32K total context budget and at most 50 turns for both models. Keep the complete history, a shared tool cache and the same judge model. Context overflow, missing answers and failed cases remain in the denominator. Report a second score after removing overlap with SFT train. Only the judge receives reference answers. A MaaS judge produces an internal judge score, reported separately from the paper's GPT-4o score.

## Tool environment

```bash
git clone https://github.com/shawn0728/OpenSearch-VL.git /path/to/OpenSearch-VL
git -C /path/to/OpenSearch-VL checkout 1ac26e2c35a3e813e022813c88888db457d88f7d
cp examples/opensearch_vl/configs/benchmark.env.example /path/to/benchmark.env
# Fill in the service configuration, then:
source /path/to/benchmark.env
```

| Configuration                                                     | Purpose                                                            |
| ----------------------------------------------------------------- | ------------------------------------------------------------------ |
| `OPENSEARCH_ROOT`                                                 | Pinned tool checkout                                               |
| `AGENT_BASE_URL / AGENT_MODEL`                                    | Evaluated model's OpenAI-compatible API                            |
| `AGENT_PROVENANCE_FILE`                                           | Verified model identity and training provenance                    |
| `SERPER_API_KEY / JINA_API_KEY / QWEN_API_BASE / QWEN_MODEL_NAME` | Text search, page reading and fixed summarization service          |
| `SERPAPI_URL / SERPAPI_API_KEY`                                   | Google Lens image-search API                                       |
| `IMAGE_UPLOAD_URL`                                                | Multipart `file` upload returning a public `{"url":"https://..."}` |
| `LAYOUT_PARSING_API_URL / SR_MODEL_PATH`                          | PP-StructureV3 endpoint and super-resolution weights               |
| `JUDGE_BASE_URL / JUDGE_MODEL / JUDGE_API_KEY`                    | Explicitly selected MaaS judge                                     |

Set the judge base URL to the endpoint without `/chat/completions`; use `JUDGE_API_KEY_HEADER=api-key`. Keep keys outside Git. The judge transport limits requests to 1 QPS. Record its actual model name and retain the original grading prompt.

Install Pillow, requests, the HF processor and OpenCV with `dnn_superres`. Preserve thinking and raw `<tool_call>` output in the model server; do not enable a parser that consumes those tags.

## Datasets and overlap

Download the fixed dataset revisions:

```python
from huggingface_hub import hf_hub_download

benchmark_data_root = "/path/to/benchmark-datasets"
hf_hub_download("m-a-p/SimpleVQA", repo_type="dataset",
    revision="037cf89fb6f1212691756b66d1ecde6c2ce89e54", filename="test.parquet",
    local_dir=f"{benchmark_data_root}/simplevqa")
hf_hub_download("Osilly/Vision-DeepResearch-Eval", repo_type="dataset",
    revision="deeaf45779a3bbd407d8f0ccb9b4831fc78e81c9", filename="data/testmini500_v1.parquet",
    local_dir=f"{benchmark_data_root}/vdr")
```

Set `BENCHMARK_DATA_ROOT` to that directory. The loader selects SimpleVQA rows with `language == "EN"`. Set `BENCHMARK_OVERLAP_INDEX` to a JSON object built from the actual `train.jsonl`, with `questions` and `images` hash lists. Question hashes remove `<image>` from the first user message, casefold, collapse whitespace and SHA256 the UTF-8 text. Image hashes are the training rows' `metadata.image_hashes`: SHA256 of the original RGB size string followed by RGB bytes. A question or any image match marks overlap; the deduplicated score uses only nonmatching cases.

`AGENT_PROVENANCE_FILE` records QS model/version/subversion, `served_model_name`, full branch/commit, training Job/Trial, `kind`, `epochs` and `hashes_verified=true`. Verify downloaded files against their upload receipt before recording this flag. The `relax` label requires `kind=trained-final` and at least two epochs; upload-validation versions can only serve as baselines.

## Run

The original Qwen template drops historical thinking. Generate the training-compatible template and use it for both token counting and serving:

```bash
export HF_CHECKPOINT="${MODEL_DIR}/Qwen3.5-4B"
export BENCHMARK_CHAT_TEMPLATE=/path/to/preserve-thinking.jinja
PYTHONPATH=examples/opensearch_vl/tools:. python3 - <<'PY'
import os
from pathlib import Path
from transformers import AutoProcessor
from benchmark import preserve_history_template
processor = AutoProcessor.from_pretrained(os.environ["HF_CHECKPOINT"], local_files_only=True)
Path(os.environ["BENCHMARK_CHAT_TEMPLATE"]).write_text(preserve_history_template(processor))
PY

export EXP_NAME=qwen3.5-4b-opensearch-vl-32k-8xgpu
python3 -m sglang.launch_server \
  --model-path "${MODEL_DIR}/Qwen3.5-4B" \
  --host 127.0.0.1 --port 30000 --tp 2 --context-length 32768 \
  --chat-template "${BENCHMARK_CHAT_TEMPLATE}" --served-model-name Qwen3.5-4B
```

In another terminal with the environment loaded:

```bash
export AGENT_BASE_URL=http://127.0.0.1:30000/v1
export AGENT_MODEL=Qwen3.5-4B
SCRIPT=examples/opensearch_vl/scripts/run-qwen3.5-4B-8xgpu.sh
bash "$SCRIPT" benchmark simplevqa base
bash "$SCRIPT" benchmark vdr base
```

Stop the baseline server and restart it with the selected Relax HF directory. The training script initially exports to `${SAVE_DIR}/${EXP_NAME}/final-hf-<step>`; automatic publication renames it to include branch, commit and training Job/Trial. If scratch has been cleaned, download the published trained-final version from QS. Do not use an upload-validation version as the trained model. Update `AGENT_PROVENANCE_FILE` to the selected model; keep the original processor, tool configuration, cache and judge unchanged.

```bash
export AGENT_MODEL=your-served-relax-model-name
bash "$SCRIPT" benchmark simplevqa relax
bash "$SCRIPT" benchmark vdr relax
bash "$SCRIPT" compare
```

## Outputs

- `${SAVE_DIR}/${EXP_NAME}/benchmark-comparison.json`: base and Relax Pass@1, differences and deduplicated scores.
- `benchmark/<base|relax>/<simplevqa|vdr>/trajectories.jsonl`: full tool trajectories, final answers and judge verdicts.
- `summary.json` in each run directory: case counts, failures, tool errors and truncations. Diagnose abnormal exits; do not count only successful cases.

Keep measured results in experiment records with the actual judge, model version and training provenance. Do not report smoke tests as full benchmark results.
