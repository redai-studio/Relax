#!/usr/bin/env bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Full 48-layer Qwen3.8 GRPO: TP2/PP1/CP1/EP8/ETP1 + SP; colocated SGLang TP8.
# Eight Blackwell GPUs are reused between rollout and training via offloading.
# MODEL_DIR is the exact HF checkpoint; DATA_DIR is the converted RL parquet file.
# Prepare DATA_DIR with scripts/tools/process_openr1.py --mode train (prompt/image/label).
# SAVE_DIR is the persistent checkpoint root; keep EXP_NAME stable for resume.
# MOE_BACKEND=hybridep (default) or mok. Requires the compatible patched MCore/MoK.
# Two-step GPU smoke tests cover rollout, training and weight sync with both backends.
# Set SAVE_CHECKPOINTS=0 NUM_ROLLOUT=2 for smoke tests; checkpoint save/resume is unverified.
# MODEL_DIR=... DATA_DIR=... SAVE_DIR=... MEGATRON=... RAY_JOBS_ADDRESS=... bash <script>
# DRY_RUN=1 prints argv/runtime JSON; default Ray submission blocks and follows logs.

set -eo pipefail
export RAY_NO_WAIT="${RAY_NO_WAIT-}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
RELAX_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
# Use this checkout and the explicitly selected compatible Megatron source.
export RELAX="${RELAX_ROOT}"
: "${MEGATRON:?Set MEGATRON to the compatible patched Megatron-LM checkout}"
export MEGATRON
export PYTHONPATH="${RELAX}:${MEGATRON}${PYTHONPATH:+:${PYTHONPATH}}"
: "${MODEL_DIR:?Set MODEL_DIR to the exact HF checkpoint directory}"
: "${DATA_DIR:?Set DATA_DIR to the converted OpenR1MM RL parquet file}"
: "${SAVE_DIR:?Set SAVE_DIR to a persistent checkpoint root}"
export MOE_BACKEND="${MOE_BACKEND:-hybridep}"
case "${MOE_BACKEND}" in
    hybridep) : "${RELAX_HYBRID_EP_COMMIT:?HybridEP must be installed in the Blackwell runtime}" ;;
    mok) ;;
    *) echo "MOE_BACKEND must be hybridep or mok" >&2; exit 2 ;;
esac
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-8192}"
MAX_CONTEXT_LEN="${MAX_CONTEXT_LEN:-4096}"
if [[ ! "${MAX_TOKENS_PER_GPU}" =~ ^[1-9][0-9]*$ ]] \
    || (( MAX_TOKENS_PER_GPU < 1024 || MAX_TOKENS_PER_GPU % 512 != 0 )); then
    echo "MAX_TOKENS_PER_GPU must be at least 1024 and divisible by 512" >&2
    exit 2
fi
# Full packed-budget padding keeps SP-local token shapes aligned across EP ranks,
# including logprob forwards, short responses and dynamic-batch dummy microbatches.
DATA_PAD_SIZE_MULTIPLIER="${DATA_PAD_SIZE_MULTIPLIER:-$((MAX_TOKENS_PER_GPU / 2))}"
if [[ "${DATA_PAD_SIZE_MULTIPLIER}" != "$((MAX_TOKENS_PER_GPU / 2))" ]]; then
    echo "DATA_PAD_SIZE_MULTIPLIER must equal MAX_TOKENS_PER_GPU / 2" >&2
    exit 2
fi
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-16}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-8}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-128}"
for value in "${ROLLOUT_BATCH_SIZE}" "${N_SAMPLES_PER_PROMPT}" "${GLOBAL_BATCH_SIZE}"; do
    if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
        echo "Rollout and global batch sizes must be positive integers" >&2
        exit 2
    fi
done
if (( N_SAMPLES_PER_PROMPT < 2 || GLOBAL_BATCH_SIZE % 4 != 0 \
    || ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT != GLOBAL_BATCH_SIZE )); then
    echo "Require n_samples >= 2, global batch divisible by DP4, and rollout_batch * n_samples = global_batch" >&2
    exit 2
fi
export RELAX_SKIP_CLUSTER_CLEANUP="${RELAX_SKIP_CLUSTER_CLEANUP:-1}"
if [[ "${DRY_RUN:-0}" != 1 && -z "${RELAX_ENTRYPOINT_MODE:-}" ]]; then
    : "${RAY_JOBS_ADDRESS:?Set RAY_JOBS_ADDRESS to the Ray Jobs HTTP endpoint}"
    exec bash "${RELAX_ROOT}/scripts/entrypoint/ray-job.sh" "${BASH_SOURCE[0]}" "$@"
fi
source "${RELAX_ROOT}/scripts/models/qwen38-flash-next.sh"
# CP1 cannot use streamed PLE. Keep the frozen table on its owning GPU ranks;
# CPU-offloaded PLE synchronization is not validated by this colocated recipe.
export RELAX_QWEN38_PLE_STREAMED=0
export QWEN48_PLE_CPU_OFFLOAD=0
export SGLANG_VLM_CACHE_SIZE_MB=0
# Match Megatron's FP32 router. Other AutoModel parity knobs remain opt-in:
# their BF16 rounding contracts are not established for Megatron fused kernels.
export RELAX_QWEN38_FP32_ROUTER=1

PROJECT_NAME="${PROJECT_NAME:-Relax/rl/openr1mm}"
EXP_NAME="${EXP_NAME:-qwen38-flash-next-openr1mm-8xb300-${MOE_BACKEND}}" # gitleaks:allow -- preserve existing experiment/checkpoint identity
now=$(date "+%Y%m%d-%H%M%S")
export SUBMISSION_ID="${SUBMISSION_ID:-${EXP_NAME}-${now}-$$}"
# Keep frozen PLE on GPU for colocated weight synchronization (unlike offline SFT).
export QWEN38_PLE_CPU_OFFLOAD=0
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_NVLS_ENABLE=0
export NVLINK_DOMAIN_SIZE=8
export USE_MNNVL=0
export MCORE_QSA_SPARSE_BACKEND=id_sparse
export RELAX_QWEN38_QSA_COMMUNITY_BACKEND=mixed
export RELAX_QWEN38_QSA_FUSED_ATTENTION=0
export RELAX_QWEN38_QSA_FUSED_SELECTOR=0
export RELAX_QWEN38_QSA_ROUTED_QKV=1
export RELAX_QWEN38_ROUTER_SORTED_REPLAY=1
export RELAX_QWEN38_QSA_RECOMPUTE_REPLAY=1
export RELAX_QWEN38_QSA_MIXED_FLASH_MAX_LEN=-1
export RELAX_QWEN38_QSA_MIXED_FLEX_MAX_LEN=0
unset NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN
CKPT_ARGS=(
    --hf-checkpoint "${MODEL_DIR}"
    --megatron-to-hf-mode bridge
    --load "${SAVE_DIR}/${EXP_NAME}"
)
case "${SAVE_CHECKPOINTS:-1}" in
    1)
        CKPT_ARGS+=(
            --save "${SAVE_DIR}/${EXP_NAME}"
            --save-interval "${SAVE_INTERVAL:-50}"
            --max-actor-ckpt-to-keep 2
        )
        ;;
    0) ;;
    *) echo "SAVE_CHECKPOINTS must be 0 or 1" >&2; exit 2 ;;
esac
ROLLOUT_ARGS=(
    --prompt-data "${DATA_DIR}"
    --input-key prompt
    --label-key label
    --multimodal-keys '{"image":"image"}'
    --apply-chat-template
    --apply-chat-template-kwargs '{"enable_thinking":false}'
    --system-prompt 'Solve the problem and put your final answer inside <answer> </answer> tags.'
    --rollout-shuffle
    --rm-type openr1mm
    --num-rollout "${NUM_ROLLOUT:-200}"
    --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
    --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
    --global-batch-size "${GLOBAL_BATCH_SIZE}"
    --rollout-max-prompt-len "${MAX_PROMPT_LEN:-2048}"
    --rollout-max-response-len "${MAX_RESPONSE_LEN:-2048}"
    --rollout-max-context-len "${MAX_CONTEXT_LEN}"
    --rollout-temperature 1.0
    --image-max-token-num "${IMAGE_MAX_TOKEN_NUM:-128}"
    --balance-data
)
PERF_ARGS=(
    --tensor-model-parallel-size 2
    --pipeline-model-parallel-size 1
    --context-parallel-size 1
    --expert-model-parallel-size 8
    --expert-tensor-parallel-size 1
    --sequence-parallel
    --moe-grouped-gemm
    --micro-batch-size 1
    --use-dynamic-batch-size
    --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}"
    --log-probs-max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}"
    --data-pad-size-multiplier "${DATA_PAD_SIZE_MULTIPLIER}"
    --seq-length "${MAX_CONTEXT_LEN}"
    --cuda-graph-impl none
    # Avoid the runtime's torch_memory_saver VMM allocation failure on this Blackwell setup.
    --selective-offload
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 2
    --freeze-vision-model
    --freeze-vision-projection
    --freeze-params-name-list 'self_attention\.indexer\.' '(^|\.)per_layer_embedding\.'
    --no-rope-fusion
    --attention-backend flash
    --bf16
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --accumulate-allreduce-grads-in-fp32
    --attention-softmax-in-fp32
)
MOE_ARGS=()
if [[ "${MOE_BACKEND}" == hybridep ]]; then
    MOE_ARGS+=(
        --moe-token-dispatcher-type flex
        --moe-flex-dispatcher-backend hybridep
        --moe-flex-dispatcher-num-sms 80
        --moe-shared-expert-overlap
    )
    # Packing already pads each rank to the same aligned size unless an
    # individual sequence can exceed the packing budget.
    if (( MAX_CONTEXT_LEN > MAX_TOKENS_PER_GPU )); then
        MOE_ARGS+=(--moe-hybridep-pad-variable-tokens)
    fi
else
    MOK_CONFIG="${MOK_BACKEND_CONFIG:-}"
    if [[ -z "${MOK_CONFIG}" ]]; then MOK_CONFIG='{"shared_expert_mode":"native"}'; fi
    MOE_ARGS+=(
        --moe-token-dispatcher-type alltoall
        --moe-megakernel-backend mok
        --moe-megakernel-backend-config "${MOK_CONFIG}"
    )
fi
GRPO_ARGS=(
    --advantage-estimator grpo
    --kl-loss-coef 0.0
    --calculate-per-token-loss
    --eps-clip 0.2
    --eps-clip-high 0.28
    --entropy-coef 0.0
    # TIS and mismatch metrics require a separately validated custom TIS function.
)
OPTIMIZER_ARGS=(
    --optimizer adam
    --lr 1e-6
    --lr-decay-style constant
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.98
    --clip-grad 1.0
    --optimizer-cpu-offload
    --overlap-cpu-optimizer-d2h-h2d
    --use-precision-aware-optimizer
    --moe-router-load-balancing-type none
    --moe-aux-loss-coeff 0.0
)
SGLANG_ARGS=(
    --rollout-num-gpus-per-engine 8
    --sglang-mem-fraction-static 0.8
    --sglang-disable-cuda-graph
    --sglang-enable-deterministic-inference
    --sglang-bf16-gemm-backend torch
    --sglang-linear-attn-prefill-backend triton
    --sglang-max-running-requests 16
    --sglang-max-mamba-cache-size 64
    # Preserve visual processing: do not copy the text-only AutoModel override.
)
TRACKING_ARGS=(--use-metrics-service)
if [ "${USE_CLEARML:-1}" = 1 ]; then
    TRACKING_ARGS+=(
        --use-clearml
        --tb-project-name "${PROJECT_NAME}"
        --tb-experiment-name "${EXP_NAME}-${now}"
    )
fi

# Actor and rollout workers must receive the same model sources and precision controls.
export RUNTIME_ENV_JSON
RUNTIME_ENV_JSON=$(python3 - <<'PY'
import json
import os
runtime = json.loads(os.environ.get("RUNTIME_ENV_JSON") or "{}")
env = runtime.setdefault("env_vars", {})
for name, value in os.environ.items():
    if name.startswith(("CLEARML_", "RELAX_QWEN38_", "RELAX_GDN_", "RELAX_HC_", "RELAX_MOE_", "RELAX_QSA_")) or name in (
        "PYTHONPATH", "MEGATRON", "RELAX", "PYTHONNOUSERSITE", "QWEN48_PLE_CPU_OFFLOAD",
        "SGLANG_VLM_CACHE_SIZE_MB", "CUDA_HOME", "TRITON_F32_DEFAULT",
        "QWEN38_PLE_CPU_OFFLOAD", "CUDA_DEVICE_MAX_CONNECTIONS", "NCCL_NVLS_ENABLE",
        "NVLINK_DOMAIN_SIZE", "USE_MNNVL", "MCORE_QSA_SPARSE_BACKEND",
        "RELAX_HYBRID_EP_COMMIT", "LD_LIBRARY_PATH",
    ):
        env[name] = value
propagated = [*env.get("RELAX_PROPAGATE_ENV_VARS", "").split(","), *env]
env["RELAX_PROPAGATE_ENV_VARS"] = ",".join(dict.fromkeys(name for name in propagated if name))
print(json.dumps(runtime))
PY
)
TRAIN_ARGS=(
    --resource '{"actor":[1,8],"rollout":[1,8]}'
    --colocate
    --max-staleness 0
    --max-global-restart "${MAX_GLOBAL_RESTART:-0}"
    --seed "${SEED:-1234}"
    --actor-num-nodes 1
    --actor-num-gpus-per-node 8
    --num-data-storage-units 1
    "${MODEL_ARGS[@]}"
    "${CKPT_ARGS[@]}"
    "${ROLLOUT_ARGS[@]}"
    "${PERF_ARGS[@]}"
    "${MOE_ARGS[@]}"
    "${GRPO_ARGS[@]}"
    "${OPTIMIZER_ARGS[@]}"
    "${SGLANG_ARGS[@]}"
    "${TRACKING_ARGS[@]}"
    "$@"
)
if [[ "${DRY_RUN:-0}" == 1 ]]; then
    python3 - "${RUNTIME_ENV_JSON}" python3 -m relax.entrypoints.train "${TRAIN_ARGS[@]}" <<'PYDRY'
import json
import sys
print(json.dumps({"runtime_env": json.loads(sys.argv[1]), "argv": sys.argv[2:]}, indent=2))
PYDRY
    exit 0
fi
mkdir -p "${SCRIPT_DIR}/log"
ray job submit ${RAY_NO_WAIT:+--no-wait} \
    --address="${RAY_JOBS_ADDRESS:-${RAY_DASHBOARD_ADDRESS:-http://127.0.0.1:8265}}" \
    --submission-id="${SUBMISSION_ID}" \
    ${WORKING_DIR:+--working-dir "${WORKING_DIR}"} \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- python3 -m relax.entrypoints.train "${TRAIN_ARGS[@]}" \
    2>&1 | tee "${SCRIPT_DIR}/log/${EXP_NAME}-${now}.log"
