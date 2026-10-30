# Qwen3.8 Flash/NV QSA

The optional cuDNN QSA backend replaces the long-document Grouped attention branch of Qwen3.8 mixed QSA. Short documents still use causal Flash Attention. The selector and its native four-token block IDs remain unchanged.

The adapter supports CUDA BF16, TP2 local 12Q/1KV heads with D256, CP1, and a frozen Stage-1 indexer. It does not enable TP8 local 3Q/1KV attention. Packed inputs require the producer's CPU document boundaries; token positions and the incomplete causal tail remain local to each document.

## Runtime requirements

Use a compatible CUDA 13 Blackwell training environment, including Megatron/Bridge, HybridEP and Flash Attention. The optional dependency is [NVIDIA cuDNN Frontend PR1765](https://github.com/NVIDIA/cudnn-frontend/pull/1765), pinned to commit **627c23f62e860fe2f76d6b38d2efcc2ee84c70cd**, with CuTeDSL **4.8.1**.

The repository's CUDA 13 Blackwell image pins CuTeDSL to 4.6.2 through PIP_CONSTRAINT. A direct installation of 4.8.1 conflicts with that constraint, and creating a venv alone does not remove the inherited environment variable. Build a dedicated Qwen3.8 NV image instead of changing the running base environment. Save the following as Dockerfile.qsa:

~~~dockerfile
# syntax=docker/dockerfile:1.7
ARG RELAX_BASE_IMAGE
FROM ${RELAX_BASE_IMAGE}

RUN python - <<'PY'
import os
import re
from pathlib import Path

source = Path(os.environ["PIP_CONSTRAINT"])
replaced = re.compile(
    r"^\s*nvidia[-_.](?:cutlass[-_.]dsl|cudnn[-_.]frontend)\s*(?:[<=>!~@]|$)",
    re.IGNORECASE,
)
lines = [line for line in source.read_text().splitlines() if not replaced.match(line)]
lines.append("nvidia-cutlass-dsl==4.8.1")
Path("/etc/relax-qsa-runtime-constraints.txt").write_text("\n".join(lines) + "\n")
PY

ENV PIP_CONSTRAINT=/etc/relax-qsa-runtime-constraints.txt
RUN python -m pip install --no-cache-dir --upgrade \
      'nvidia-cutlass-dsl[cu13]==4.8.1' \
      'nvidia-cudnn-frontend @ git+https://github.com/NVIDIA/cudnn-frontend.git@627c23f62e860fe2f76d6b38d2efcc2ee84c70cd' && \
    python -c 'import importlib.metadata as m; assert m.version("nvidia-cutlass-dsl") == "4.8.1"; from cudnn.QSA.api import qsa_forward, qsa_backward'
~~~

This copies the base constraints and replaces only the CuTeDSL and frontend distribution pins: CuTeDSL has an explicit version, and the frontend has an explicit source commit in the installation command. All other constraints, including Torch, NumPy and Transformer Engine, remain in effect. Do not unset PIP_CONSTRAINT globally or edit the original constraints file.

~~~bash
docker build --build-arg RELAX_BASE_IMAGE='<your-cu13-blackwell-image>' \
  -f Dockerfile.qsa -t relax-qsa:pr1765 .
~~~

Use the derived image on every Ray node, including the head. Building the pinned frontend also requires its upstream build dependencies and a compatible cuDNN installation; follow the [upstream installation instructions](https://github.com/NVIDIA/cudnn-frontend/tree/627c23f62e860fe2f76d6b38d2efcc2ee84c70cd). A different package that only exports a module named cudnn is insufficient. Relax does not install or replace the dependency automatically. Grouped remains the framework default and does not import this optional backend.

The build and import checks cover dependency resolution, the selected CuTeDSL version and the QSA entry points. They do not establish GPU numerical correctness, TP/PP recomputation or compatibility of other kernels with the newer CuTeDSL. Validate those paths in the derived image before training; do not treat this Qwen3.8-specific variant as a validated replacement for every Blackwell recipe.

## Select the recipe

Both recipes use 4 nodes × 8 GPUs, 48 layers, global 24Q/2KV heads, TP2/PP4/CP1/EP8/ETP1 with sequence parallelism, BF16, GBS512/MB1, full uniform recomputation over two layers and one MTP layer.

| Recipe | Pipeline layer counts | LR decay iterations | Input |
| --- | --- | --- | --- |
| Pure text | 14 / 12 / 12 / 10 | 996 | Text messages and tool calls |
| Multimodal | 12 / 14 / 14 / 8 | 2023 | Messages with image references |

The text partition places three backbone QSA layers on each pipeline stage. The last stage also owns the MTP QSA layer and output head, so the total QSA counts are 3/3/3/4. The multimodal recipe keeps its separate partition; changing its PP layout is a separate tuning decision.

Run from the repository root. MODEL_DIR is the complete HF checkpoint directory, DATA_DIR is a JSONL file or directory, and SAVE_DIR is a persistent checkpoint root visible on all nodes:

~~~bash
export MODEL_DIR=/models/qwen38
export DATA_DIR=/datasets/text
export SAVE_DIR=/checkpoints/qwen38
export MEGATRON=/workspace/Megatron-LM
export RELAX_HYBRID_EP_COMMIT='<installed-hybridep-commit>'
export RAY_JOBS_ADDRESS='http://<ray-head>:8265'

EXP_NAME=qsa-text-nv NUM_STEPS=50 \
  bash examples/models/qwen3.8-flash-next/scripts/run-qwen38-qsa-nv-text-sft-32xgpu-blackwell.sh
~~~

For image-plus-text SFT, supply a multimodal HF checkpoint and data, then use:

~~~bash
EXP_NAME=qsa-mm-nv NUM_STEPS=50 \
  bash examples/models/qwen3.8-flash-next/scripts/run-qwen38-qsa-nv-sft-32xgpu-blackwell.sh
~~~

Both use messages and tools fields. The multimodal recipe additionally maps image to images, limits each image to 128 tokens and preprocesses images on training ranks. The text recipe omits those three multimodal options; it does not turn image data into text. Choose a text-only dataset for that recipe.

The scripts delegate cluster setup to scripts/entrypoint/ray-job.sh and then submit a Ray Job. They require an exclusive Ray Serve workspace and refuse an already-running job when invoked directly. They do not provide concurrent-job isolation. Submission blocks by default; set RAY_NO_WAIT=1 only when you want to monitor the returned job separately.

To inspect the effective command without Ray setup, network access, GPU imports or output-directory creation:

~~~bash
DRY_RUN=1 \
  bash examples/models/qwen3.8-flash-next/scripts/run-qwen38-qsa-nv-text-sft-32xgpu-blackwell.sh
~~~

## Routing and controlled comparisons

Each recipe sets the same routing on the driver and training actors:

| Setting | Recipe value | Effect |
| --- | --- | --- |
| RELAX_QWEN38_QSA_COMMUNITY_BACKEND | mixed | Keep the existing selector and packed-input path |
| RELAX_QWEN38_QSA_MIXED_LONG_BACKEND | cudnn_pr1765 | Use NV for long documents |
| RELAX_QWEN38_QSA_MIXED_FLASH_MAX_LEN | 2051 | Documents of at most 2051 tokens use Flash |
| RELAX_QWEN38_QSA_MIXED_FLEX_MAX_LEN | 0 | Disable the intermediate Flex branch |
| RELAX_QWEN38_QSA_NV_MAX_PACK_TOKENS | 0 | Call NV separately for each long document |
| NVTE_USE_CUTLASS_GROUPED_GEMM | 0 | Keep the independent MoE GEMM option off |
| CUDA_DEVICE_MAX_CONNECTIONS | 1 | Match the TP/sequence-parallel runtime |

The 2051 boundary includes 512 complete four-token blocks and at most three partial-tail tokens. A document of length 2052 enters the sparse long branch. Use the producer's real document lengths, not the total concatenated pack length.

For a Grouped control with the same text recipe, set only the long backend and a distinct experiment identity:

~~~bash
RELAX_QWEN38_QSA_MIXED_LONG_BACKEND=grouped EXP_NAME=qsa-text-grouped \
  bash examples/models/qwen3.8-flash-next/scripts/run-qwen38-qsa-nv-text-sft-32xgpu-blackwell.sh
~~~

Use distinct EXP_NAME or SAVE_DIR values for separate arms so one does not resume the other's checkpoint. NVTE_USE_CUTLASS_GROUPED_GEMM=1 can be supplied independently for a MoE comparison; it does not select the QSA implementation or prove that a particular MoE kernel ran. The NV mini-pack setting is experimental and remains disabled in these recipes.

## Checkpoints and throughput

The recipes configure load and save at the stable SAVE_DIR/EXP_NAME path, save every SAVE_INTERVAL steps (default 50), and retain two Actor checkpoints. Keep that path stable to resume; do not append a fresh timestamp. ClearML task names and logs get a timestamp for each launch. The scripts configure checkpointing through the existing Megatron backend; a dry run only checks configuration, not successful optimizer/RNG restoration.

Both recipes disable custom record-function tags and autotune logging and do not enable the PyTorch profiler. They use a 120-minute producer queue timeout and zero automatic global restarts. See [Performance Tuning](./performance-tuning.md) for explicit diagnostic profiling.

Compare Actor throughput using **sum of logical training tokens / sum of Actor training seconds / GPU count** over the same step window. End-to-end throughput uses total step seconds instead. Preserve slow steps, list cold compilation and checkpoint saves separately, and compare equal data/topology/packing/recomputation settings. A faster QSA kernel does not imply the same speedup for the full training pipeline.
