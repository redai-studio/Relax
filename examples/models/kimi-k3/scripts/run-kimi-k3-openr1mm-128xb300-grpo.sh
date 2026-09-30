#!/bin/bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Full-parameter Kimi K3 GRPO, 16 x 8 B300, actor/rollout colocated.
# Training: TP4/PP4 (21+24+24+24)/CP2/EP32/ETP1, including vision.
# Rollout: 8 engines with attention TP8/DP2 and MoE EP16, two nodes per engine.
# Requires the current SGLang MXFP4 reload patch on EVERY node.
# CPU optimizer offload matches the SFT recipe; measure load/repack/train peaks.
# Source env.sh, then submit through scripts/entrypoint/ray-job.sh.
# DRY_RUN=1 bash <this-script> prints arguments without touching Ray.
# Defaults: 200 training rollouts, saving every 200 rollouts.
# Saves weights only, matching the SFT recipe.
# Initialize from HF_CHECKPOINT by default; LOAD_DIR enables checkpoint recovery.

set -eo pipefail
export RAY_NO_WAIT="${RAY_NO_WAIT-0}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
RELAX_ROOT="$(cd -- "${SCRIPT_DIR}/../../../.." &>/dev/null && pwd)"
MODEL_CONFIG_DIR="${MODEL_CONFIG_DIR:-${RELAX_ROOT}/scripts/models}"
source "${MODEL_CONFIG_DIR}/kimi-k3.sh"
HF_CHECKPOINT="${HF_CHECKPOINT:-${MODEL_DIR:?Set MODEL_DIR to the HF model directory}/Kimi-K3}"
PROMPT_SET="${PROMPT_SET:-${DATA_DIR:?Set DATA_DIR to the prepared dataset directory}/multimodal-open-r1-8k-verified/data/train-00000-of-00001_converted_noextract.parquet}"
now=$(date '+%Y-%m-%d-%H-%M-%S')
EXP_NAME="${EXP_NAME:-kimi-k3-openr1mm-grpo-128xb300}"
PROJECT_NAME="${PROJECT_NAME:-Relax/rl/kimi-k3-openr1mm}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${SAVE_DIR:?Set SAVE_DIR to a persistent checkpoint root}/${EXP_NAME}}"
SYSTEM_PROMPT="A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>"

CKPT_ARGS=(
    --hf-checkpoint "${HF_CHECKPOINT}"
    --trust-remote-code
    --megatron-to-hf-mode bridge
    --save "${CHECKPOINT_DIR}"
    --save-interval "${SAVE_INTERVAL:-200}"
    --no-save-optim
    --max-actor-ckpt-to-keep "${MAX_ACTOR_CKPT_TO_KEEP:-1}"
)
if [ -n "${LOAD_DIR:-}" ]; then
    CKPT_ARGS+=(--load "${LOAD_DIR}")
fi
# Routed weight updates are enabled by default; set to 0 to use broadcast.
if [ "${COLOCATE_EXPERT_WEIGHT_ROUTING:-1}" = "1" ]; then
    CKPT_ARGS+=(--colocate-expert-weight-routing)
fi
if [ -n "${DUMP_DETAILS:-}" ]; then
    CKPT_ARGS+=(--dump-details "${DUMP_DETAILS}")
fi
ROLLOUT_ARGS=(
    --prompt-data "${PROMPT_SET}"
    --input-key prompt
    --label-key label
    --multimodal-keys '{"image":"image"}'
    --system-prompt "${SYSTEM_PROMPT}"
    --apply-chat-template
    --rollout-shuffle
    --balance-data
    --rm-type openr1mm
    # --dynamic-sampling-filter-path relax.engine.filters.dynamic_sampling_filters.check_reward_nonzero_std
    # --over-sampling-batch-size "${OVER_SAMPLING_BATCH_SIZE:-128}"
    --num-rollout "${NUM_ROLLOUT:-200}"
    --rollout-batch-size "${ROLLOUT_BATCH_SIZE:-64}"
    --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT:-8}"
    --global-batch-size "${GLOBAL_BATCH_SIZE:-512}"
    --rollout-max-response-len "${MAX_RESPONSE_LEN:-8192}"
    --rollout-max-prompt-len "${MAX_PROMPT_LEN:-2048}"
    --rollout-temperature 1.0
    --image-max-token-num "${IMAGE_MAX_TOKEN_NUM:-1024}"
    --rollout-health-check-timeout 120
)
PERF_ARGS=(
    --tensor-model-parallel-size 4
    --sequence-parallel
    --pipeline-model-parallel-size 4
    --decoder-first-pipeline-num-layers 21
    --decoder-last-pipeline-num-layers 24
    --context-parallel-size 2
    --expert-model-parallel-size 32
    --expert-tensor-parallel-size 1
    --calculate-per-token-loss
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1
    --use-dynamic-batch-size
    --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU:-16384}"
    --log-probs-max-tokens-per-gpu "${LOG_PROBS_MAX_TOKENS_PER_GPU:-16384}"
    # TP4 * 1024 fits the per-GPU token budget.
    --data-pad-size-multiplier 1024
    # EFA: retain the K3 alltoall path, without DeepEP/NVSHMEM.
    --moe-token-dispatcher-type alltoall
    --optimizer-cpu-offload
    --optimizer-offload-fraction "${OPTIMIZER_OFFLOAD_FRACTION:-0.9}"
    --selective-offload
    --overlap-cpu-optimizer-d2h-h2d
    --use-precision-aware-optimizer
    --main-grads-dtype bf16
    --no-pin-cpu-grads
    --no-pin-cpu-params
    --empty-unused-memory-level "${EMPTY_UNUSED_MEMORY_LEVEL:-1}"
    --disable-weights-backuper
    # Empty allocator settings override inherited expandable segments under TMS.
    --train-env-vars '{"FLA_TILELANG":"0","OPEN_TRAINING_MXFP4_FAKE_QAT_FLAG":"1","TORCHINDUCTOR_COMPILE_THREADS":"1","PYTORCH_ALLOC_CONF":"","PYTORCH_CUDA_ALLOC_CONF":""}'
)
GRPO_ARGS=(
    --advantage-estimator grpo
    # Do not enable use-kl-loss: even a zero coefficient would load a reference.
    --kl-coef 0.0
    --kl-loss-coef 0.0
    --entropy-coef 0.0
    --eps-clip 0.2
    --eps-clip-high 0.28
    --use-tis
)
OPTIMIZER_ARGS=(
    --optimizer adam
    --lr "${LR:-1e-6}"
    --lr-decay-style constant
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.98
    --clip-grad 1.0
)
SGLANG_ARGS=(
    --rollout-num-gpus-per-engine 16
    --sglang-enable-dp-attention
    --sglang-dp-size 2
    --sglang-ep-size 16
    --sglang-moe-runner-backend flashinfer_mxfp4
    --sglang-moe-a2a-backend none
    --use-slime-router
    # The initial full actor push supplies real weights before the first rollout.
    --sglang-load-format "${SGLANG_LOAD_FORMAT:-dummy}"
    --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC:-0.7}"
    # Validation observed 74 concurrent requests on one engine, falling back to eager at 64.
    --sglang-cuda-graph-max-bs "${SGLANG_CUDA_GRAPH_MAX_BS:-128}"
    --sglang-server-concurrency "${SGLANG_SERVER_CONCURRENCY:-256}"
    --sglang-watchdog-timeout 3600
)
METRICS_ARGS=(
    --use-tensorboard
    --use-metrics-service
    --tb-project-name "${PROJECT_NAME}"
    --tb-experiment-name "${EXP_NAME}-${now}"
)
if [ "${USE_CLEARML:-1}" = 1 ]; then
    METRICS_ARGS+=(--use-clearml)
fi
TRAIN_ARGS=(
    --resource '{"actor": [1, 128], "rollout": [1, 128]}'
    --num-gpus-per-node 8
    --colocate
    --max-staleness 0
    --num-data-storage-units 16
    --use-health-check
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --attention-backend flash
    --update-weight-buffer-size "${UPDATE_WEIGHT_BUFFER_SIZE:-8589934592}"
    "${MODEL_ARGS[@]}"
    "${CKPT_ARGS[@]}"
    "${ROLLOUT_ARGS[@]}"
    "${PERF_ARGS[@]}"
    "${GRPO_ARGS[@]}"
    "${OPTIMIZER_ARGS[@]}"
    "${SGLANG_ARGS[@]}"
    "${METRICS_ARGS[@]}"
    "$@"
)
if [ "${DRY_RUN:-0}" = 1 ]; then
    printf '%q ' python3 -m relax.entrypoints.train "${TRAIN_ARGS[@]}"
    printf '\n'
    exit 0
fi
if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    source "${RELAX_ROOT}/scripts/entrypoint/ray-job.sh"
fi
# Also clear inherited allocator settings in the driver and rollout workers.
unset PYTORCH_ALLOC_CONF PYTORCH_CUDA_ALLOC_CONF
RUNTIME_ENV_JSON=$(python3 - <<'PY'
import json
import os
runtime = json.loads(os.environ['RUNTIME_ENV_JSON'])
runtime.setdefault('env_vars', {}).update({
    'PYTORCH_ALLOC_CONF': '',
    'PYTORCH_CUDA_ALLOC_CONF': '',
    'TORCHINDUCTOR_COMPILE_THREADS': '1',
    # Explicitly enable bounded model-save staging for this RL recipe.
    'MEGATRON_SYNC_SAVE_BOUNDED_STAGING': os.environ.get('MEGATRON_SYNC_SAVE_BOUNDED_STAGING', '1'),
    'MEGATRON_SYNC_SAVE_STAGE_BYTES': os.environ.get('MEGATRON_SYNC_SAVE_STAGE_BYTES', '1073741824'),
    # Initial full sync exceeds Serve's user-event-loop probe timeout.
    # Keep probes responsive while synchronous service methods wait on Ray.
    'RAY_SERVE_RUN_SYNC_IN_THREADPOOL': '1',
})
env = runtime['env_vars']
propagate = env.get('RELAX_PROPAGATE_ENV_VARS', '').split(',')
env['RELAX_PROPAGATE_ENV_VARS'] = ','.join(dict.fromkeys(
    name for name in [*propagate, 'MEGATRON_SYNC_SAVE_BOUNDED_STAGING', 'MEGATRON_SYNC_SAVE_STAGE_BYTES'] if name
))
print(json.dumps(runtime))
PY
)
export RUNTIME_ENV_JSON
mkdir -p "${RELAX_ROOT}/log"
RAY_SUBMIT_ARGS=()
if [ "${RAY_NO_WAIT}" = "1" ]; then
    RAY_SUBMIT_ARGS+=(--no-wait)
fi
ray job submit "${RAY_SUBMIT_ARGS[@]}" \
    --address="${RAY_DASHBOARD_ADDRESS:-http://127.0.0.1:8265}" \
    ${WORKING_DIR:+--working-dir "${WORKING_DIR}"} \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- python3 -m relax.entrypoints.train "${TRAIN_ARGS[@]}" \
    2>&1 | tee "${RELAX_ROOT}/log/${EXP_NAME}-${now}.log"
