#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Qwen3-4B SFT: launch through ray-job.sh with baseline or chunkflow.
set -eo pipefail

now=$(date "+%Y-%m-%d-%H:%M:%S")
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
MODEL_DIR="${MODEL_DIR:?Set MODEL_DIR to the directory containing Qwen3-4B}"
PROMPT_DATA="${PROMPT_DATA:-${DATA_DIR:+${DATA_DIR}/chunkflow-mixed/train.parquet}}"
PROMPT_DATA="${PROMPT_DATA:?Set PROMPT_DATA or DATA_DIR containing chunkflow-mixed/train.parquet}"
NUM_ROLLOUT="${NUM_ROLLOUT:-100}"
SEQ_LENGTH="${SEQ_LENGTH:-32768}"
GBS="${GBS:-8}"
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-${SEQ_LENGTH}}"

TP_SIZE="${TP_SIZE:-2}"
PP_SIZE="${PP_SIZE:-4}"
for key in TP_SIZE PP_SIZE GBS NUM_ROLLOUT SEQ_LENGTH MAX_TOKENS_PER_GPU; do
   if ! [[ "${!key}" =~ ^[1-9][0-9]*$ ]]; then
      echo "ERROR: ${key} must be a positive integer" >&2
      exit 2
   fi
done
if [ "$((TP_SIZE * PP_SIZE))" -ne 8 ] || [ "$((36 % PP_SIZE))" -ne 0 ] ||
   [ "$((8 % TP_SIZE))" -ne 0 ] || [ "${NUM_GPUS:-8}" != 8 ]; then
   echo "ERROR: require TP_SIZE * PP_SIZE = 8, PP_SIZE dividing 36 layers, TP_SIZE dividing 8 KV heads" >&2
   exit 2
fi
export NUM_GPUS=8
if [ "$#" -gt 1 ]; then
   echo "Usage: bash $0 {baseline|chunkflow}" >&2
   exit 2
fi
MODE="${1:-chunkflow}"
case "${MODE}" in
   baseline)
      CHUNK_SIZE=0
      ;;
   chunkflow)
      CHUNK_SIZE="${CHUNK_SIZE:-4096}"
      ;;
   *)
      echo "Usage: bash $0 {baseline|chunkflow}" >&2
      exit 2
      ;;
esac
PACK_POLICY="${PACK_POLICY:=staggered}"
KEEP_LAST_K="${KEEP_LAST_K:=0}"
if ! [[ "${CHUNK_SIZE}" =~ ^(0|[1-9][0-9]*)$ && "${KEEP_LAST_K}" =~ ^(0|[1-9][0-9]*)$ ]] ||
   { [ "${MODE}" != baseline ] && [ "${CHUNK_SIZE}" -eq 0 ]; }; then
   echo "ERROR: CHUNK_SIZE must be positive for ChunkFlow; KEEP_LAST_K must be nonnegative" >&2
   exit 2
fi
if [ "${PACK_POLICY}" != staggered ]; then
   echo "ERROR: this training example requires PACK_POLICY=staggered" >&2
   exit 2
fi
if [ "${CHUNK_SIZE}" -gt 0 ]; then
   CHUNKFLOW_ARGS=(--use-chunkflow --chunkflow-chunk-size ${CHUNK_SIZE} --chunkflow-pack-policy "${PACK_POLICY}")
   ARM="chunk${CHUNK_SIZE}-${PACK_POLICY}"
   CHUNKFLOW_ARGS+=(--chunkflow-pipeline ${PP_SIZE})
   ARM="${ARM}-pp${PP_SIZE}"
   if [ "${KEEP_LAST_K}" -gt 0 ]; then
      CHUNKFLOW_ARGS+=(--chunkflow-keep-last-k ${KEEP_LAST_K})
      ARM="${ARM}-k${KEEP_LAST_K}"
   fi
else
   CHUNKFLOW_ARGS=()
   ARM="baseline"
fi

PROJECT_NAME="${PROJECT_NAME:-Relax/sft/chunkflow}"
EXP_NAME="${EXP_NAME:-qwen3-4b-sft-${ARM}}"
RUN_NAME="${RUN_NAME:-${EXP_NAME}-${now}}"
TRAIN_SEED="${TRAIN_SEED:-1235}"
if ! [[ "${TRAIN_SEED}" =~ ^(0|[1-9][0-9]*)$ ]]; then
   echo "ERROR: TRAIN_SEED must be a nonnegative integer" >&2
   exit 2
fi

# Profiler timings include instrumentation overhead.
PROFILE_ARGS=()
if ! [[ "${CHUNKFLOW_PROFILE:-0}" =~ ^[01]$ ]]; then
   echo "ERROR: CHUNKFLOW_PROFILE must be 0 or 1" >&2
   exit 2
fi
if [ "${CHUNKFLOW_PROFILE:-0}" = 1 ]; then
   PROFILE_START="${PROFILE_START:-10}"
   PROFILE_END="${PROFILE_END:-12}"
   if ! [[ "${PROFILE_START}" =~ ^(0|[1-9][0-9]*)$ && "${PROFILE_END}" =~ ^(0|[1-9][0-9]*)$ ]] ||
      [ "${PROFILE_END}" -lt "${PROFILE_START}" ] || [ "${NUM_ROLLOUT}" -le "$((PROFILE_END + 1))" ]; then
      echo "ERROR: profiling needs 0 <= PROFILE_START <= PROFILE_END and NUM_ROLLOUT > PROFILE_END + 1" >&2
      exit 2
   fi
   PROFILE_ARGS=(--use-pytorch-profiler --profile-target train_overall
      --profile-step-start "${PROFILE_START}" --profile-step-end "${PROFILE_END}")
   echo "Profiler output (relative to worker cwd): traces/${RUN_NAME}/train_trace"
fi

if [ ! -d "${MODEL_DIR}/Qwen3-4B" ]; then
   echo "ERROR: ${MODEL_DIR}/Qwen3-4B does not exist. Set MODEL_DIR / DATA_DIR, e.g." >&2
   echo "  export MODEL_DIR=/path/to/models" >&2
   echo "  export DATA_DIR=/path/to/data" >&2
   exit 1
fi
if [ ! -f "${PROMPT_DATA}" ]; then
   echo "ERROR: ${PROMPT_DATA} does not exist." >&2
   echo "Set PROMPT_DATA to an existing SFT parquet with question and answer columns." >&2
   exit 1
fi

if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
   source "${SCRIPT_DIR}/../../entrypoint/local.sh"
fi
source "${MODEL_CONFIG_DIR:-${SCRIPT_DIR}/../../models}/qwen3-4B.sh"

CKPT_ARGS=(
   --hf-checkpoint "${MODEL_DIR}/Qwen3-4B/"
   --megatron-to-hf-mode bridge
   --num-rollout ${NUM_ROLLOUT}
)

SFT_ARGS=(
   --loss-type sft
   --prompt-data "${PROMPT_DATA}"
   --input-key "${INPUT_KEY:-question}"
   --label-key "${LABEL_KEY:-answer}"
   --global-batch-size ${GBS}
   --use-dynamic-batch-size
   --seq-length ${SEQ_LENGTH}
   --max-tokens-per-gpu ${MAX_TOKENS_PER_GPU}
   --sft-oversize-strategy skip
)

export RELAX_CHUNKFLOW_PIPELINE_UNTESTED="${RELAX_CHUNKFLOW_PIPELINE_UNTESTED:=1}"
export RELAX_CHUNKFLOW_FLASH_BACKEND="${RELAX_CHUNKFLOW_FLASH_BACKEND:-fa3}"
case "${RELAX_CHUNKFLOW_FLASH_BACKEND}" in
   fa2|fa3) ;;
   *) echo "ERROR: RELAX_CHUNKFLOW_FLASH_BACKEND must be fa2 or fa3" >&2; exit 2 ;;
esac

RUNTIME_ENV_JSON=$(python3 -c '
import json, os
env = json.loads(os.environ["RUNTIME_ENV_JSON"])
target = env.setdefault("env_vars", {})
for key in ("RELAX_CHUNKFLOW_PIPELINE_UNTESTED", "RELAX_CHUNKFLOW_FLASH_BACKEND"):
    target[key] = os.environ[key]
print(json.dumps(env))
')
export RUNTIME_ENV_JSON

TP_ARGS=(--tensor-model-parallel-size ${TP_SIZE})
if [ "${TP_SIZE}" -gt 1 ]; then
   TP_ARGS+=(--sequence-parallel)
fi

PERF_ARGS=(
   "${TP_ARGS[@]}"
   --pipeline-model-parallel-size ${PP_SIZE}
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --calculate-per-token-loss
   --no-rope-fusion
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-5
   --lr-decay-style cosine
   --min-lr 1e-6
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
   --clip-grad 1.0
)

MISC_ARGS=(
   --seed "${TRAIN_SEED}"
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

CLEARML_ARGS=(
   --use-clearml
   --tb-project-name "${PROJECT_NAME}"
   --tb-experiment-name "${RUN_NAME}"
)

LOG_FILE="${LOG_FILE:-log/${EXP_NAME}-gpu${NUM_GPUS}-${now}.log}"
mkdir -p "$(dirname -- "${LOG_FILE}")"
RAY_DASHBOARD_ADDRESS="${RAY_DASHBOARD_ADDRESS:-http://127.0.0.1:8265}"
SUBMIT_ARGS=(--address="${RAY_DASHBOARD_ADDRESS}")
case "${RAY_NO_WAIT:-0}" in
   0|"") ;;
   1) SUBMIT_ARGS+=(--no-wait) ;;
   *) echo "ERROR: RAY_NO_WAIT must be 0 or 1" >&2; exit 2 ;;
esac
if [ -n "${WORKING_DIR:-}" ]; then
   SUBMIT_ARGS+=(--working-dir "${WORKING_DIR}")
fi
echo "LOG_FILE=${LOG_FILE}"
echo "MODE=${MODE} TP=${TP_SIZE} PP=${PP_SIZE} GBS=${GBS} CHUNK_SIZE=${CHUNK_SIZE} STEPS=${NUM_ROLLOUT} SEED=${TRAIN_SEED}"
ray job submit "${SUBMIT_ARGS[@]}" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 -m relax.entrypoints.train \
   --resource "{\"sft\": [1, 0], \"actor\": [1, ${NUM_GPUS}]}" \
    "${MODEL_ARGS[@]}" \
    "${CKPT_ARGS[@]}" \
    "${SFT_ARGS[@]}" \
    "${OPTIMIZER_ARGS[@]}" \
    "${PERF_ARGS[@]}" \
    "${CHUNKFLOW_ARGS[@]}" \
    "${PROFILE_ARGS[@]}" \
    "${CLEARML_ARGS[@]}" \
    "${MISC_ARGS[@]}"  2>&1 | tee "${LOG_FILE}"

if [ "${RAY_NO_WAIT:-0}" != 1 ]; then
   if ! grep -q "Actor training completed step $((NUM_ROLLOUT - 1))/${NUM_ROLLOUT}" "${LOG_FILE}"; then
      echo "ERROR: expected training steps not completed; inspect ${LOG_FILE}" >&2
      exit 1
   fi
   if [ "${CHUNK_SIZE}" -gt 0 ] &&
      { ! grep -Eq 'chunkflow: chunked attention installed on [1-9][0-9]* layer' "${LOG_FILE}" ||
        ! grep -q "chunkflow: flash attention backend=${RELAX_CHUNKFLOW_FLASH_BACKEND}" "${LOG_FILE}"; }; then
      echo "ERROR: requested ChunkFlow attention installation/backend not confirmed in ${LOG_FILE}" >&2
      exit 1
   fi
   if [ "${MODE}" = baseline ] && grep -q "chunkflow: chunked attention installed" "${LOG_FILE}"; then
      echo "ERROR: baseline unexpectedly installed ChunkFlow" >&2
      exit 1
   fi
fi
