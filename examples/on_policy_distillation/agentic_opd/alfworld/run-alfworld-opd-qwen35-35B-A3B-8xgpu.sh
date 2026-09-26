#!/bin/bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# ALFWorld agentic OPD (On-Policy Distillation), Qwen3.5-35B-A3B (MoE), 8xGPU colocate.
# Rollout: 1P1D TP2/EP1 on 4 GPUs; teacher: TP4 on the other 4 GPUs.

set -ex
set -o pipefail

export NCCL_NVLS_ENABLE=0

now=$(date "+%Y-%m-%d-%H:%M:%S")

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    source "${SCRIPT_DIR}/../../../../scripts/entrypoint/local.sh"
fi

source "${MODEL_CONFIG_DIR}/qwen35-35B-A3B.sh"

PROJECT_NAME="${PROJECT_NAME:=Relax/dev/agentic_opd_alfworld}"
EXP_DIR="${EXP_DIR:-/root/exps}"
MODEL_DIR="${MODEL_DIR:-${EXP_DIR}}"
DATA_DIR="${DATA_DIR:-/root/alfworld-relax}"

CONDA_HOME="${CONDA_HOME:-/root/miniconda3}"
ALFWORLD_CONDA_ENV="${ALFWORLD_CONDA_ENV:-relax-opd-alfworld}"
ALFWORLD_VENV="${ALFWORLD_VENV:-}"
ALFWORLD_DATA="${ALFWORLD_DATA:-/root/alfworld}"
ALFWORLD_MAX_TURNS="${ALFWORLD_MAX_TURNS:-40}"
ALFWORLD_HISTORY_LENGTH="${ALFWORLD_HISTORY_LENGTH:-2}"

NUM_ROLLOUT="${NUM_ROLLOUT:=150}"

ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-256}"
ROLLOUT_N_GROUPS="${ROLLOUT_N_GROUPS:-1}"
# Match GRPO: this is a per-turn cap, independent of the trajectory budget.
ROLLOUT_RESP_LENGTH="${ROLLOUT_RESP_LENGTH:-2048}"
ROLLOUT_PROMPT_LENGTH="${ROLLOUT_PROMPT_LENGTH:-2048}"
# Bound the whole multi-turn training trajectory independently of the turn cap.
ROLLOUT_MAX_CONTEXT_LENGTH="${ROLLOUT_MAX_CONTEXT_LENGTH:-32768}"
AGENTIC_CONCURRENCY="${AGENTIC_CONCURRENCY:-${ROLLOUT_BATCH_SIZE}}"
EVAL_ROLLOUT_RESP_LENGTH="${EVAL_ROLLOUT_RESP_LENGTH:-${ROLLOUT_RESP_LENGTH}}"

STUDENT_MODEL_NAME="${STUDENT_MODEL_NAME:-Qwen3.5-35B-A3B}"
TEACHER_MODEL_PATH="${TEACHER_MODEL_PATH:-${EXP_DIR}/Qwen3.5-35B-A3B-GRPO-alfworld-50step}"

OPD_KL_COEF="${OPD_KL_COEF:-1.0}"
OPD_LOSS_COEF="${OPD_LOSS_COEF:-0.0}"
OPD_KL_TYPE="${OPD_KL_TYPE:-reverse_kl}"
OPD_TOKEN_SELECTION="${OPD_TOKEN_SELECTION:-student_sampled}"

ROLLOUT_GPUS="${ROLLOUT_GPUS:-4}"
TEACHER_GPUS="${TEACHER_GPUS:-4}"
ACTOR_GPUS="${ACTOR_GPUS:-8}"

# Student and teacher are online together during rollout; actor reuses all 8 GPUs.
ROLLOUT_NUM_GPUS_PER_ENGINE="${ROLLOUT_NUM_GPUS_PER_ENGINE:-2}"
SGLANG_CONFIG="${SGLANG_CONFIG:-${SCRIPT_DIR}/sglang-1p1d-tp2-qwen35-35B-A3B.yaml}"
if [ "${ROLLOUT_GPUS}" -ne 4 ] || [ "${TEACHER_GPUS}" -ne 4 ] || \
   [ "${ACTOR_GPUS}" -ne 8 ] || [ "${ROLLOUT_NUM_GPUS_PER_ENGINE}" -ne 2 ]; then
    echo "This PD recipe requires rollout=4, teacher=4, actor=8 GPUs and TP2 rollout engines." >&2
    exit 1
fi
if [ ! -f "${SGLANG_CONFIG}" ]; then
    echo "SGLang PD config not found: ${SGLANG_CONFIG}" >&2
    exit 1
fi
# One D engine serves all resident sessions; allow 25% scheduling margin.
SGLANG_MAX_RUNNING_REQUESTS="${SGLANG_MAX_RUNNING_REQUESTS:-$(((AGENTIC_CONCURRENCY * ROLLOUT_N_GROUPS * 5 + 3) / 4))}"

# Forward these to Ray actors and SGLang HTTP processes. The pool value
# selects hybrid-model metadata handling; Mooncake transport remains RDMA.
export SGLANG_TIMEOUT_KEEP_ALIVE="${SGLANG_TIMEOUT_KEEP_ALIVE:-600}"
export SGLANG_MOONCAKE_CUSTOM_MEM_POOL="${SGLANG_MOONCAKE_CUSTOM_MEM_POOL:-INTRA_NODE_NVLINK}"
export RELAX_PROPAGATE_ENV_VARS="${RELAX_PROPAGATE_ENV_VARS:+${RELAX_PROPAGATE_ENV_VARS},}SGLANG_TIMEOUT_KEEP_ALIVE,SGLANG_MOONCAKE_CUSTOM_MEM_POOL"
RUNTIME_ENV_JSON=$(python3 - <<'PYENV'
import json
import os

runtime = json.loads(os.environ["RUNTIME_ENV_JSON"])
env_vars = runtime.setdefault("env_vars", {})
# Keep both caller-provided lists when the driver rebuilds actor environments.
propagate = env_vars.get("RELAX_PROPAGATE_ENV_VARS", "") + "," + os.environ["RELAX_PROPAGATE_ENV_VARS"]
env_vars["RELAX_PROPAGATE_ENV_VARS"] = ",".join(dict.fromkeys(x.strip() for x in propagate.split(",") if x.strip()))
for name in ("SGLANG_TIMEOUT_KEEP_ALIVE", "SGLANG_MOONCAKE_CUSTOM_MEM_POOL"):
    env_vars[name] = os.environ[name]
print(json.dumps(runtime))
PYENV
)
export RUNTIME_ENV_JSON

EXP_NAME=agentic-opd-alfworld-sampledkl-1p1d-tp${ROLLOUT_NUM_GPUS_PER_ENGINE}-${STUDENT_MODEL_NAME}-${now}
SAVE_DIR=${EXP_DIR}/save/${EXP_NAME}
mkdir -p "${SAVE_DIR}"

CKPT_ARGS=(
   --hf-checkpoint ${MODEL_DIR}/${STUDENT_MODEL_NAME}/
   --ref-load ${MODEL_DIR}/${STUDENT_MODEL_NAME}/
   --megatron-to-hf-mode bridge
   --save ${SAVE_DIR}
   --save-interval 2000
)

ROLLOUT_ARGS=(
   --prompt-data ${DATA_DIR}/train.parquet
   --input-key prompt
   --label-key label
   --metadata-key extra_info
   --reward-key score
   --rollout-shuffle
   --apply-chat-template-kwargs '{"enable_thinking": true}'

   --custom-rm-path examples.on_policy_distillation.agentic_opd.alfworld.reward_alfworld.reward_func
   --use-agentic-rollout
   --agent-command ". ${SCRIPT_DIR}/run_agent_app.sh"
   --agent-cwd "${SCRIPT_DIR}"
   --agent-env
      "CONDA_HOME=${CONDA_HOME}"
      "ALFWORLD_CONDA_ENV=${ALFWORLD_CONDA_ENV}"
      "ALFWORLD_VENV=${ALFWORLD_VENV}"
      "ALFWORLD_DATA=${ALFWORLD_DATA}"
      "ALFWORLD_MAX_TURNS=${ALFWORLD_MAX_TURNS}"
      "ALFWORLD_HISTORY_LENGTH=${ALFWORLD_HISTORY_LENGTH}"
      # Match the turn cap used below for finish_length/context_exhausted handling.
      "ALFWORLD_MAX_RESPONSE_LEN=${ROLLOUT_RESP_LENGTH}"
      "OMP_NUM_THREADS=1"
      "MKL_NUM_THREADS=1"
      "OPENBLAS_NUM_THREADS=1"

   --num-rollout              ${NUM_ROLLOUT}
   --rollout-batch-size       ${ROLLOUT_BATCH_SIZE}
   --n-samples-per-prompt     ${ROLLOUT_N_GROUPS}
   --rollout-max-prompt-len   ${ROLLOUT_PROMPT_LENGTH}
   --rollout-max-response-len ${ROLLOUT_RESP_LENGTH}
   --rollout-max-context-len  ${ROLLOUT_MAX_CONTEXT_LENGTH}
   --agentic-concurrency     ${AGENTIC_CONCURRENCY}
   --rollout-temperature      1.0
   --rollout-top-p            1.0
   --global-batch-size $((ROLLOUT_BATCH_SIZE * ROLLOUT_N_GROUPS))
   --use-fault-tolerance
   --use-streaming-dataset
)

EVAL_ARGS=(
   --eval-interval 5
   --eval-prompt-data alfworld_unseen ${DATA_DIR}/test.parquet
   --n-samples-per-eval-prompt 1
   --eval-max-response-len ${EVAL_ROLLOUT_RESP_LENGTH}
   --eval-temperature 0.4
   --eval-top-p 1.0
)

OPD_ARGS=(
   --use-opd
   --opd-type sglang

   --teacher-hf-checkpoint ${TEACHER_MODEL_PATH}
   --warm-hf-checkpoint-page-cache

   --teacher-sglang-mem-fraction-static 0.7
   # Teacher scores the complete trajectory as input and needs the same headroom.
   --teacher-sglang-context-length $((ROLLOUT_MAX_CONTEXT_LENGTH + 64))
   --teacher-sglang-chunked-prefill-size ${TEACHER_PREFILL_CHUNK:-8192}
   --teacher-sglang-max-prefill-tokens ${TEACHER_MAX_PREFILL_TOKENS:-16384}
   --teacher-num-gpus-per-engine 4
   # Pin teacher admission independently of the student PD batch capacity.
   --teacher-sglang-max-running-requests ${TEACHER_MAX_RUNNING_REQUESTS:-64}
   --teacher-sglang-disable-cuda-graph

   --opd-kl-coef ${OPD_KL_COEF}
   --opd-loss-coef ${OPD_LOSS_COEF}
   --opd-kl-type ${OPD_KL_TYPE}
   --opd-token-selection ${OPD_TOKEN_SELECTION}

   --opd-teacher-timeout-s ${OPD_TEACHER_TIMEOUT_S:-6000}

   --opd-disable-rl-reward
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --eps-clip 0.2
   --eps-clip-high 0.3
   --kl-loss-coef 0.01
   --kl-loss-type low_var_kl
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.01
   --adam-beta1 0.9
   --adam-beta2 0.999
   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
)

PERF_ARGS=(
   --tensor-model-parallel-size ${TP_SIZE:-4}
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size ${EP_SIZE:-8}
   --expert-tensor-parallel-size ${ETP_SIZE:-1}
   --calculate-per-token-loss
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --use-dynamic-batch-size
   # A complete multi-turn trajectory must fit the per-GPU token budget at CP1.
   --max-tokens-per-gpu ${ACTOR_MAX_TOKENS_PER_GPU:-${ROLLOUT_MAX_CONTEXT_LENGTH}}
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine ${ROLLOUT_NUM_GPUS_PER_ENGINE}
   --sglang-config "${SGLANG_CONFIG}"
   # SGLang reserves input/output slots internally. Keep the RL/training budget
   # at 32768 and add service headroom for legal requests near that limit.
   --sglang-context-length $((ROLLOUT_MAX_CONTEXT_LENGTH + 64))
   --sglang-router-policy consistent_hashing
   --sglang-router-prefill-policy consistent_hashing
   --sglang-router-decode-policy consistent_hashing
   --sglang-load-format dummy
   --sglang-enable-weights-cpu-backup
   --sglang-max-running-requests ${SGLANG_MAX_RUNNING_REQUESTS}
   --sglang-cuda-graph-max-bs-decode ${SGLANG_MAX_RUNNING_REQUESTS}
)

WANDB_ARGS=(
   --use-clearml
   --use-metrics-service
   --tb-project-name    ${PROJECT_NAME}
   --tb-experiment-name ${EXP_NAME}
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   --no-rope-fusion
)

mkdir -p log

if [ -z "${RAY_DASHBOARD:-}" ]; then
    if [ -n "${RAY_ADDRESS:-}" ]; then
        RAY_DASHBOARD="http://${RAY_ADDRESS%%:*}:8265"
    else
        RAY_DASHBOARD="http://${HOST_IP:-127.0.0.1}:8265"
    fi
fi

ray job submit ${RAY_NO_WAIT:+--no-wait} --address="${RAY_DASHBOARD}" \
   ${WORKING_DIR:+--working-dir "${WORKING_DIR}"} \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 -m relax.entrypoints.train \
   --resource "{\"actor\": [1, ${ACTOR_GPUS}], \"rollout\": [1, ${ROLLOUT_GPUS}], \"teacher\": [1, ${TEACHER_GPUS}]}" \
   --max-staleness 0 \
   --num-data-storage-units 1 \
   --colocate \
   --use-health-check \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${OPD_ARGS[@]}" \
   "${GRPO_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${MISC_ARGS[@]}" \
   "${EVAL_ARGS[@]}" \
   "${WANDB_ARGS[@]}" \
   2>&1 | tee log/${EXP_NAME}.log
