#!/bin/bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# 16 x 8 H800, 128K multimodal SFT with a frozen vision tower.
# bash scripts/entrypoint/ray-job.sh examples/models/deepseek-v4.1/scripts/run-deepseek-v4.1-flash-1017-multiturn-sft-128xh800.sh
set -eo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
: "${MODEL_DIR:?Set MODEL_DIR to the shared model root}"
: "${DATA_DIR:?Set DATA_DIR to the training data root}"
: "${SAVE_DIR:?Set SAVE_DIR to the shared checkpoint root}"
export PROMPT_DATA="${PROMPT_DATA:-${DATA_DIR}/train.jsonl}"
export VAL_DATA="${VAL_DATA-}"
export PROJECT_NAME="${PROJECT_NAME:-Relax/sft/multiturn}"
export EXP_NAME="${EXP_NAME:-deepseek-v41-1017-multiturn-sft-128xh800}"
export NUM_NODES="${V41_SFT_NODES:-16}" TP_SIZE="${TP_SIZE:-2}" PP_SIZE="${PP_SIZE:-2}" CP_SIZE="${CP_SIZE:-32}"
export EP_SIZE="${EP_SIZE:-$((8 * NUM_NODES / PP_SIZE))}" MOE_DISPATCHER=alltoall
export SEQ_LENGTH=131072
export MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-$((SEQ_LENGTH / CP_SIZE))}"
export GLOBAL_BATCH_SIZE=64 NUM_EPOCHS=2 SEED=1234
export LR="${LR:-1e-5}" LR_DECAY_STYLE=cosine ADAM_BETA2=0.95
export SAVE_INTERVAL="${SAVE_INTERVAL:-100}" MAX_ACTOR_CKPT_TO_KEEP="${MAX_ACTOR_CKPT_TO_KEEP:-10}"

# ray-job.sh builds the worker environment before invoking this recipe. Use the
# training interpreter's libraries on image shells and bare SSH shells alike.
if [[ -n "${RUNTIME_ENV_JSON:-}" ]]; then
    set +x
    RUNTIME_ENV_JSON=$(python3 - <<'PY'
import json
import os
import shutil
import sysconfig
from pathlib import Path

runtime = json.loads(os.environ["RUNTIME_ENV_JSON"])
env = runtime.setdefault("env_vars", {})
site = Path(sysconfig.get_path("purelib"))
nvcc = shutil.which("nvcc")
cuda = os.environ.get("CUDA_HOME") or (str(Path(nvcc).parent.parent) if nvcc else None)
libraries = [str(site / "torch/lib"), str(site / "nvidia/nvshmem/lib")]
if cuda:
    libraries = [str(Path(cuda) / "compat"), *libraries, str(Path(cuda) / "lib64")]
libraries.extend(env.get("LD_LIBRARY_PATH", "").split(os.pathsep))
env["LD_LIBRARY_PATH"] = os.pathsep.join(dict.fromkeys(path for path in libraries if path))
env.update(OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="2")
experiment = os.environ["EXP_NAME"]
if os.environ.get("V41_MXFP4_QAT") == "1":
    experiment += "-mxfp4-qat"
env["TENSORBOARD_DIR"] = str(Path(os.environ["SAVE_DIR"]) / experiment / "tensorboard_log")
print(json.dumps(runtime))
PY
    )
    export RUNTIME_ENV_JSON
fi

args=(
    --sft-loss-last-turn-only
    --rollout-shuffle
    --image-max-token-num 1024
    --sft-invalid-multimodal-strategy skip
    --lr-warmup-fraction 0.05
    --min-lr "${MIN_LR:-1e-6}"
    --moe-permute-fusion
    --sft-chunked-logits
    --sft-logits-chunk-size 1024
    --sft-prefetch-num-workers "${SFT_PREFETCH_NUM_WORKERS:-16}"
    --sft-prefetch-buffer-size "${SFT_PREFETCH_BUFFER_SIZE:-64}"
    --sft-tq-timeout-minutes 90
    --use-tensorboard
)
exec bash "${script_dir}/run-deepseek-v4.1-flash-1017-vision-sft.sh" "${args[@]}" "$@"
