#!/bin/bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.

set -ex
set -o pipefail
now=$(date "+%Y-%m-%d-%H:%M:%S")
echo "当前时间: $now"

###############################################################################
#                                 klx args                                    #
###############################################################################
set +x
export WANDB_API_KEY="${WANDB_API_KEY:=YOUR-KEY}"
set -x
export WORKDIR="${WORKDIR:-/workspace}"
export MODEL_DIR="${MODEL_DIR:-/workspace/deepeyes}"
export DATA_DIR="${DATA_DIR:-/workspace/deepeyes}"
export SAVE_DIR="${SAVE_DIR:-/workspace/deepeyes}"
export PROJECT_NAME=deepeyes_v2_agentic

export MEGATRON=${WORKDIR}/Megatron-LM

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export RL_MODEL_TYPE="Qwen36_35B_multimodal"
export XMLIR_USE_HYDRA_LINEAR=${XMLIR_USE_HYDRA_LINEAR:-1}
export XMLIR_ENABLE_FAST_FC=${XMLIR_ENABLE_FAST_FC:-1}
export XMLIR_MATMUL_FAST_MODE=${XMLIR_MATMUL_FAST_MODE:-1}
export XMLIR_MEMCPY_RETRY_SYNC=${XMLIR_MEMCPY_RETRY_SYNC:-true}

export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-eth0}
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-eth0}
export TP_SOCKET_IFNAME=${TP_SOCKET_IFNAME:-eth0}
export BKCL_RDMA_NICS=${BKCL_RDMA_NICS:-bond0,bond1,bond2,bond3,bond4,bond5,bond6,bond7}

unset http_proxy
unset https_proxy

###############################################################################
#                                 ENVIRONMENT                                 #
###############################################################################

TIMESTAMP=$(date "+%Y-%m-%d-%H:%M:%S")

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
# Auto-source env.sh if present (gitignored, machine-specific overrides).
# shellcheck source=/dev/null
[ -f "${SCRIPT_DIR}/env.sh" ] && source "${SCRIPT_DIR}/env.sh"

if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    source "${SCRIPT_DIR}/../../scripts/entrypoint/local.sh"
fi
source "${MODEL_CONFIG_DIR}/qwen36-35B-A3B.sh"

###############################################################################
#                                    DIRS                                     #
###############################################################################

PROJECT_NAME="${PROJECT_NAME:=Relax/dev/deepeyes-v2}"
EXP_NAME="qwen36-35B-A3B-deepeyes-v2-agentic-${TIMESTAMP}"

if [ -z "${MODEL_DIR:-}" ] || [ -z "${DATA_DIR:-}" ] || [ -z "${SAVE_DIR:-}" ]; then
    echo "ERROR: MODEL_DIR, DATA_DIR, and SAVE_DIR must be set."
    echo "Example: MODEL_DIR=/path/to/models DATA_DIR=/path/to/data SAVE_DIR=/path/to/save bash $0"
    exit 1
fi
mkdir -p ${SAVE_DIR}

###############################################################################
#                             STARTUP CLEANUP                                 #
###############################################################################
# Sessions that die by SIGKILL / OOM / uncaught crash never run
# ApptainerJupyterSession.close(), so their /tmp/relax-apptainer-* dirs
# (kernel conn files + bind-mounted imgs/inputs holding base64-decoded
# dataset images) leak. Over 2000 rollouts × 32 batch × 8 samples ≈ 512K
# sessions, even a 1% leak fills the container disk and triggers eviction.
# mtime > 240 min is well above any single session's max wall-clock (bounded
# by max_turns × code_timeout_s ≈ tens of minutes), so live sessions are
# never touched.
find /tmp -maxdepth 1 -name 'relax-apptainer-*' -mmin +240 -exec rm -rf {} + 2>/dev/null || true
# Per-run agent log dirs accumulate one file per session; keep 3 days.
find "${SCRIPT_DIR}/log/agent" -maxdepth 1 -type d -mtime +3 -exec rm -rf {} + 2>/dev/null || true

# SIF path under DATA_DIR layout; user may override APPTAINER_IMAGE_PATH to
# point at a shared NFS copy elsewhere.
export APPTAINER_IMAGE_PATH="${APPTAINER_IMAGE_PATH:-${DATA_DIR}/sif/deepeyes_v2_kernel.sif}"
if [ ! -f "${APPTAINER_IMAGE_PATH}" ]; then
    echo "ERROR: SIF not found at ${APPTAINER_IMAGE_PATH}."
    echo "Run: DATA_DIR=${DATA_DIR} bash ${SCRIPT_DIR}/scripts/prepare.sh"
    exit 1
fi
DEEPEYES_V2_APP_PYTHON="${DEEPEYES_V2_APP_ENV_ROOT:-/tmp/deepeyes-v2-app-env}/.venv/bin/python"
if [ ! -x "${DEEPEYES_V2_APP_PYTHON}" ]; then
    echo "ERROR: DeepEyes V2 app environment not found at ${DEEPEYES_V2_APP_PYTHON}."
    echo "Run: bash ${SCRIPT_DIR}/scripts/prepare_app_env.sh"
    exit 1
fi

###############################################################################
#                              JUDGE MODEL API                                #
###############################################################################

source "${SCRIPT_DIR}/sglang_judge_service_klx.sh"

###############################################################################
#                                  MODEL CONFIG                               #
###############################################################################

CKPT_ARGS=(
    --hf-checkpoint ${MODEL_DIR}/Qwen3.6-35B-A3B
    --ref-load ${MODEL_DIR}/Qwen3.6-35B-A3B
    --warm-hf-checkpoint-page-cache
    --save ${SAVE_DIR}/Qwen3.6-35B-A3B-Checkpoint_v2
    --megatron-to-hf-mode bridge
    --save-interval 200
    --max-actor-ckpt-to-keep 1
)

###############################################################################
#                                  DATASETS                                   #
###############################################################################

# Layout produced by scripts/prepare.sh:
#   ${DATA_DIR}/data/{perception_all_*,reason,search,vstar_test}.parquet
#   ${DATA_DIR}/sif/deepeyes_v2_kernel.sif
TRAIN_FILES=(
    "'${DATA_DIR}/data/perception_all_1.parquet@[0:5000]'"
    "'${DATA_DIR}/data/reason.parquet@[0:5000]'"
)
TEST_FILES=("${DATA_DIR}/data/vstar_test.parquet@[0:256]")
PROMPT_SET="[$(IFS=,; echo "${TRAIN_FILES[*]}")]"

###############################################################################
#                               ROLLOUT CONFIG                                #
###############################################################################

NUM_ROLLOUT="${NUM_ROLLOUT:=2000}"

# Sandbox env vars propagated into every Ray worker so the per-session
# agent process can find apptainer / search cache.
# SANDBOX_CONFIG_PATH is required — the agent reads it in _build_executor
# to find the apptainer backend YAML config (image path, bind paths, etc).
EXTRA_ENV_VARS_JSON="\"SANDBOX_BACKEND\": \"apptainer_jupyter\",
    \"SANDBOX_CONFIG_PATH\": \"${SCRIPT_DIR}/apptainer_env/apptainer_config.yaml\",
    \"APPTAINER_IMAGE_PATH\": \"${APPTAINER_IMAGE_PATH}\",
    \"DEEPEYES_V2_APP_PYTHON\": \"${DEEPEYES_V2_APP_PYTHON}\",
    \"DEEPEYES_V2_SEARCH_CACHE_PATHS\": \"${DEEPEYES_V2_SEARCH_CACHE_PATHS:-}\",
    \"DEEPEYES_JUDGE_BASE_URL\": \"${DEEPEYES_JUDGE_BASE_URL:-}\",
    \"DEEPEYES_JUDGE_MODELS\": \"${DEEPEYES_JUDGE_MODELS:-}\",
    \"DEEPEYES_JUDGE_API_KEY\": \"${DEEPEYES_JUDGE_API_KEY:-}\",
    \"XMLIR_ENABLE_H2D_SSE_COPY\": \"${XMLIR_ENABLE_H2D_SSE_COPY:-1}\",
    \"USE_CAST_FC_FUSION\": \"${USE_CAST_FC_FUSION:-1}\""
source "${SCRIPT_DIR}/../../scripts/entrypoint/runtime-env-klx.sh"

ROLLOUT_ARGS=(
    --prompt-data "${PROMPT_SET}"
    --input-key prompt
    --label-key reward_model
    --multimodal-keys '{"image":"images"}'
    --reward-key score
    --metadata-key extra_info
    --custom-rm-path examples.deepeyes_v2_agentic.reward_deepeyes_v2.reward_func
    --use-agentic-rollout
    --agent-command ". ${SCRIPT_DIR}/run_agent_app.sh"
    --agent-cwd "${SCRIPT_DIR}"
    # Per-run agent log dir: every session's stdout/stderr is tee'd to
    # ${dir}/${session_id}.log (run_agent_app.sh). Successful AND failed
    # sessions are both kept, which is what makes hang/timeout diagnosis
    # possible — Relax's own tmpdir-based capture drops both.
    --agent-env "AGENT_DEBUG_LOG_DIR=${SCRIPT_DIR}/log/agent/${TIMESTAMP}"
    # 30-min default is too generous: normal sessions take 1-3 min, a
    # zombie session still burning chat completions after 10 min is
    # ~always doomed. Faster session-level SIGKILL clears prepare-gate
    # IR backlog quicker so new groups actually get served.
    --num-rollout ${NUM_ROLLOUT}
    --rollout-batch-size ${ROLLOUT_BATCH_SIZE:-32}
    --micro-batch-size 1
    --n-samples-per-prompt ${N_SAMPLES_PER_PROMPT:-8}
    --rollout-max-context-len 16384
    --rollout-max-response-len 4096
    --rollout-max-prompt-len 4096
    --rollout-temperature 1
    --global-batch-size 256
    --rollout-shuffle
    --use-streaming-dataset
    --agentic-prepare-pool-size 0
)

###############################################################################
#                                EVAL CONFIG                                  #
###############################################################################

EVAL_ARGS=(
    --skip-eval-before-train
    --eval-interval 500
    --eval-prompt-data vstar ${TEST_FILES}
    --n-samples-per-eval-prompt 8
    --eval-max-response-len 4096
    --eval-top-p 0.7
    --agentic-eval-concurrency ${AGENTIC_EVAL_CONCURRENCY:-128}
)

###############################################################################
#                              ALGORITHM CONFIG                               #
###############################################################################

GRPO_ARGS=(
    --advantage-estimator grpo
    --kl-loss-coef 0.00
    --kl-loss-type low_var_kl
    --entropy-coef 0.00
    --eps-clip 0.2
    --eps-clip-high 0.28
    --eps-clip-c 3
    --use-tis
)

###############################################################################
#                              OPTIMIZER CONFIG                               #
###############################################################################

OPTIMIZER_ARGS=(
    --optimizer adam
    --lr 1e-6
    --lr-decay-style constant
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.98
    --optimizer-cpu-offload
    --overlap-cpu-optimizer-d2h-h2d
    --use-precision-aware-optimizer
    --no-rope-fusion
)

###############################################################################
#                               SGLANG CONFIG                                 #
###############################################################################

SGLANG_ARGS=(
    --rollout-num-gpus-per-engine 2
    --sglang-mem-fraction-static 0.7
    --sglang-disable-custom-all-reduce
    --sglang-page-size 64
    --sglang-attention-backend kunlun
    --sglang-disable-radix-cache
    --sglang-max-running-requests 256
    # --sglang-disable-cuda-graph
    --sglang-cuda-graph-bs 1 2 4 8 $(seq 16 8 256)
    --sglang-router-policy round_robin

    --sglang-mm-attention-backend fa3
    --sglang-mm-enable-dp-encoder
)

###############################################################################
#                               LOGGING CONFIG                                #
###############################################################################

LOG_ARGS=(
    --tb-experiment-name deepeyes_v2_agentic-klx-${now}
    --use-wandb
    --wandb-project ${PROJECT_NAME}
    --wandb-group deepeyes_v2_agentic-klx-${now}
    --disable-wandb-random-suffix
    --no-use-metrics-service
)

###############################################################################
#                              MEGATRON CONFIG                                #
###############################################################################

MEGATRON_ARGS=(
    --tensor-model-parallel-size 4
    --sequence-parallel
    --pipeline-model-parallel-size 1
    --context-parallel-size 1
    --expert-model-parallel-size 8
    --expert-tensor-parallel-size 1
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1
    --max-tokens-per-gpu 16384
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --accumulate-allreduce-grads-in-fp32
    --attention-softmax-in-fp32
    --attention-backend flash
    --use-dynamic-batch-size

    --moe-flex-dispatcher-backend deepep
    --moe-token-dispatcher-type flex
    --moe-grouped-gemm true
)

###############################################################################
#                              RESOURCE CONFIG                                #
###############################################################################

RAY_RESOURCE_ARGS=(
    --resource '{"actor": [1, 8], "rollout": [1, 8]}'
    --max-staleness 0
    --num-data-storage-units 1
    # --use-health-check
    --colocate
)

###############################################################################
#                                 LAUNCH JOB                                  #
###############################################################################

mkdir -p logs

# xtrace off for the submit command: --runtime-env-json embeds EXTRA_ENV_VARS_JSON
# (may carry DEEPEYES_JUDGE_API_KEY etc.) and, after the merge below, WANDB_API_KEY;
# the whole line would otherwise be echoed by `set -x` into logs/${EXP_NAME}.log.
set +x
# Merge WANDB_API_KEY into the job runtime_env. `ray job submit` runs the entrypoint
# on the (possibly pre-existing) cluster, whose workers do NOT inherit this submit
# shell's exports — so wandb online init can't authenticate unless the key travels
# via runtime_env. Read from os.environ (no shell interpolation) and keep xtrace off
# so the value never hits the log. NOTE: the key still lands in Ray's job runtime_env
# metadata (dashboard / `ray job list`); use a cluster-preset env or `wandb login`
# (~/.netrc) instead if that surface is unacceptable.
if [ -n "${WANDB_API_KEY}" ] && [ "${WANDB_API_KEY}" != "YOUR-KEY" ]; then
    RUNTIME_ENV_JSON=$(python3 - <<'PY'
import json
import os

payload = json.loads(os.environ["RUNTIME_ENV_JSON"])
payload.setdefault("env_vars", {})["WANDB_API_KEY"] = os.environ["WANDB_API_KEY"]
print(json.dumps(payload))
PY
)
fi
ray job submit ${RAY_NO_WAIT:+--no-wait} --address="http://127.0.0.1:8265" \
    --runtime-env-json "${RUNTIME_ENV_JSON}" \
    -- python3 relax/entrypoints/train.py \
    --selective-offload \
    "${RAY_RESOURCE_ARGS[@]}" \
    "${MODEL_ARGS[@]}" \
    "${CKPT_ARGS[@]}" \
    "${ROLLOUT_ARGS[@]}" \
    "${GRPO_ARGS[@]}" \
    "${OPTIMIZER_ARGS[@]}" \
    "${SGLANG_ARGS[@]}" \
    "${LOG_ARGS[@]}" \
    "${MEGATRON_ARGS[@]}" \
    "${EVAL_ARGS[@]}" \
    2>&1 | tee logs/${EXP_NAME}.log
set -x
