# Evaluation approaches and tool setup

[中文](evaluation-setup.md) · [Running evaluation](evaluation.en.md)

SimpleVQA and VDR data files do not prescribe a service stack. Setup depends on whether the experiment uses direct answers, a search agent, or a paper's complete tool environment. Both current `benchmark ... base/relax` labels run the OpenSearch-VL agent. `base` selects initial weights; it does not disable tools.

## What public implementations do

| Implementation                                                                                                                                                                           | Workflow                                                                                                   | Preparation                                                                                                                                  |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| [Official SimpleVQA inference](https://github.com/SimpleVQA/SimpleVQA/blob/b94d1f3d05d7f488afcdcf95bfcd69bcb279111e/data_engineering/vqa_request/simplevqa_request_lvlm_api_parallel.py) | Send image and question to the model, then score answers separately                                        | Evaluated model and judge; no search, OCR or super-resolution service                                                                        |
| [lmms-eval SimpleVQA](https://github.com/EvolvingLMMs-Lab/lmms-eval/tree/main/lmms_eval/tasks/simplevqa)                                                                                 | Generate a short answer and compute normalized exact match                                                 | Model and data; its default 32 generated tokens differ from this agent protocol                                                              |
| [Official Vision-DeepResearch evaluation](https://github.com/Osilly/Vision-DeepResearch/blob/1f13ea7261de0ec8e9961d3ef2c9c33fcd6d27b1/rllm/eval/README.md)                               | Multi-turn search, page visits, Python and cropped-image search; extraction and judging can share a server | Search/reader access, image hosting, summary and judge services; this entry point does not require separate OCR or super-resolution services |
| [WebWatcher](https://github.com/Alibaba-NLP/DeepResearch/blob/main/WebAgent/WebWatcher/README.md)                                                                                        | Run tool inference, then a judge script                                                                    | Search, image search, Jina, storage, summarizer and judge; the evaluation command assumes these are ready                                    |
| [OpenSearch-VL](https://github.com/shawn0728/OpenSearch-VL/tree/1ac26e2c35a3e813e022813c88888db457d88f7d/opensearch_vl)                                                                  | The multi-turn environment reused here, including OCR, search, cropping and enhancement                    | The current blocker list describes this complete environment                                                                                 |

The official SimpleVQA [judge](https://github.com/SimpleVQA/SimpleVQA/blob/b94d1f3d05d7f488afcdcf95bfcd69bcb279111e/llm_as_a_vqa_judger.py) and [aggregation](https://github.com/SimpleVQA/SimpleVQA/blob/b94d1f3d05d7f488afcdcf95bfcd69bcb279111e/simpleVQA_eval.py) distinguish correct, incorrect and unattempted answers and report attempted-answer accuracy and F1. Exact match and this recipe's MaaS semantic grading are different metrics.

VDR's [results](https://github.com/Osilly/Vision-DeepResearch/tree/1f13ea7261de0ec8e9961d3ef2c9c33fcd6d27b1#performance) include Direct Answer and Agent Workflow. Its evaluation example uses 300 SimpleVQA questions and 500 VDR questions; this recipe selects all 1,013 English SimpleVQA questions and 500 VDR questions. Record subset, tools, judge, temperature, context and failure handling before comparing scores.

## Choosing an experiment

A direct-answer comparison can check whether SFT changes basic VQA: image and question → fixed answer prompt → shared judge. It needs no search, upload, OCR or super-resolution services. The current script does not implement this mode. Missing-tool smoke runs are not direct-answer evaluations.

Tool-use evaluation requires working tools. A smaller search-agent environment is possible, as is the complete OpenSearch-VL setup. Removing tools or changing providers or the summarizer defines a separate configuration that both initial and SFT models must share.

The selected main protocol remains the complete environment in the [evaluation guide](evaluation.en.md). The preparation below does not change it.

## Preparing the current environment

| Capability           | Preparation                                                                                                                                                                 | Check before use                                                                                                  |
| -------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| Text search          | Obtain or reuse [Serper](https://serper.dev/) access and set `SERPER_API_KEY`, or use the upstream `API_HOST / API_USER / API_KEY` gateway                                  | Nonempty `organic` results with titles, links and snippets                                                        |
| Page reading         | Configure [Jina Reader](https://docs.jina.ai/) and `JINA_API_KEY`, or a compatible gateway                                                                                  | Read actual page content, not a challenge page, empty text or search snippet                                      |
| Reverse image search | Configure [SerpApi Google Lens](https://serpapi.com/google-lens-api), `SERPAPI_URL=https://serpapi.com/search.json` and `SERPAPI_API_KEY`                                   | Current adapter uses GET with `engine=google_lens / url / api_key`; check application errors and actual matches   |
| Summarization        | Set fixed `QWEN_API_BASE / QWEN_MODEL_NAME`                                                                                                                                 | Upstream sends no MaaS authentication headers; setting its URL alone is insufficient                              |
| Image hosting        | Reuse existing COS access through a local `IMAGE_UPLOAD_URL` bridge                                                                                                         | Accept multipart `file`, return `{"url":"https://..."}`, and allow the external search service to fetch the image |
| OCR/layout           | Reuse or deploy [PP-StructureV3](https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_usage/tutorials/ocr_pipelines/PP-StructureV3.html); set `LAYOUT_PARSING_API_URL` | Check schema and readable text from an actual text image                                                          |
| Super-resolution     | OpenCV contrib `dnn_superres`, matching EDSR weights and `SR_MODEL_PATH`                                                                                                    | Load weights and verify output dimensions; the upstream success string is insufficient                            |

### Reader authentication and rate limits

Jina Reader supports anonymous use. Its [rate-limit table](https://jina.ai/en-US/api-dashboard/rate-limit/) currently lists 20 RPM anonymously and 500 RPM with a free key; confirm actual account limits. Anonymous use can support small checks. Requiring `JINA_API_KEY` is this runner's preflight policy. Supporting anonymous runs requires changing both that check and shared rate limiting. Failed page reads may silently fall back to search snippets.

### Reusing MaaS for summaries

Both current OpenSearch summary requests contain text, so the existing MaaS endpoint is a candidate without another model download. An adapter must add authentication, validate upstream nonstandard payload fields and retain the summary prompts. Fix the summarizer across initial/SFT runs and record any model change.

If summaries and judging share the supplied 1-QPS MaaS domain, both must pass through one limiter. The current runner limits judge calls only. Upstream may return a text excerpt when summarization fails, so a readable tool result does not establish that model summarization succeeded.

### Reusing COS

Upstream `cos_upload.py` loads an external `upload.py` plugin; the public checkout has no working uploader for this environment. The current runner instead supports an HTTP uploader. Its private bridge should receive the image, upload using existing coscmd configuration and return its URL. Use content hashes and run identifiers under this experiment's own `/wuhuan2/lib/relax/` directory. Retain objects until both evaluations and shared-cache reuse finish.

Search providers must fetch this URL. Passing base64 to the evaluated model does not supply image-search access. Keep configuration, internal addresses, credentials and bridge deployment parameters outside Git.

### Matching the OCR protocol

Install compatible dependencies and the serving plugin in an isolated environment using the [PaddleX serving guide](https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_deploy/serving.html). An example launch is:

```bash
paddlex --install serving
paddlex --serve --pipeline PP-StructureV3 --device cpu --host 127.0.0.1 --port 18080
export LAYOUT_PARSING_API_URL=http://127.0.0.1:18080/layout-parsing
```

This is a proposed launch, not a deployed service or a CPU throughput guarantee. Measure correctness and latency before selecting CPU or a GPU service; preserve the training environment.

The adapter posts JSON containing base64 `file` and `fileType=1`. Optional authentication uses `Authorization: token ...`. It reads `result.layoutParsingResults[0].prunedResult.parsing_res_list`, selecting `block_content` for labels `paragraph_title / text / vision_footnote`. Other OCR or vision-language APIs require schema adaptation.

### Running actual super-resolution

The [OpenCV tutorial](https://docs.opencv.org/4.x/d5/d29/tutorial_dnn_superres_upscale_image_single.html) explains loading a scale-specific model with `dnn_superres`. The [EDSR repository](https://github.com/Saafke/EDSR_Tensorflow/tree/06c7bd65b0305c2955328f8f2721ea86c341f660/models) supplies `.pb` weights. Inference needs neither EDSR training nor TensorFlow. Use `EDSR_x4.pb` for 4×; other scales require matching weights.

Record revision and SHA256 after downloading, then check the actual operation:

```python
import os
import cv2
import numpy as np

sr = cv2.dnn_superres.DnnSuperResImpl_create()
sr.readModel(os.environ["SR_MODEL_PATH"])
sr.setModel("edsr", 4)
image = np.random.default_rng(0).integers(0, 256, (16, 16, 3), dtype=np.uint8)
result = sr.upsample(image)
assert result.shape == (64, 64, 3)
```

## Reducing provider count

[OpenVLHarness](https://github.com/uclanlp/OpenVLHarness/tree/11149b6862f268def37e9e6bb37bd462ac8ed5f1) uses Serper for text search, Lens and scraping, with a shared auxiliary model. This is an alternative design, not a supported setting in the current adapter: its Lens uses SerpApi GET and its reader uses Jina. Replacing URLs alone does not adapt protocols.

Adopting one provider requires validating request, response and error mapping and recording a new tool configuration. It reduces account management; search, hosting, extraction and judging still exist.

Prepare in this order: choose protocol → obtain search access → connect COS and image search → connect fixed summaries and shared limiting → validate OCR and actual upscaling → run complete small-sample agents → measure calls, cost and latency → freeze configuration for both full runs. Filled environment variables alone do not establish readiness.
