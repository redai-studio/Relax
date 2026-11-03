#!/usr/bin/env bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Qwen3.6-35B-A3B multimodal SFT on 16 Blackwell GPUs; image-built runtime required.
# Set MODEL_DIR, DATA_DIR and SAVE_DIR; RAY_ADDRESS is the Ray Jobs HTTP URL.
# DRY_RUN=1 renders arguments and actor environment without submitting a job.
# Checkpoints save/resume at SAVE_DIR/EXP_NAME every 50 steps, retaining five.
# Unlike the archived acceptance wrapper, saving/resuming is enabled by default.
# Change EXP_NAME for a new experiment. SAVE_CHECKPOINT=0 disables saving.
# Supply overrides directly in the environment; no external config indirection.
# Launch this script directly with bash. If invoking ray-job.sh externally, set RELAX_SKIP_CLUSTER_CLEANUP=1 there too.
# Example: export MODEL_DIR=/models DATA_DIR=/data SAVE_DIR=/checkpoints; then run with DRY_RUN=1.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export RELAX="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
: "${MODEL_DIR:?Set MODEL_DIR}"
: "${DATA_DIR:?Set DATA_DIR}"
: "${SAVE_DIR:?Set SAVE_DIR to a persistent checkpoint root}"
export RAY_ADDRESS="${RAY_ADDRESS:-http://127.0.0.1:8265}"
export RAY_NO_WAIT="${RAY_NO_WAIT-}"
export RELAX_SKIP_CLUSTER_CLEANUP=1
if [[ "${DRY_RUN:-0}" != 1 && -z "${RELAX_ENTRYPOINT_MODE:-}" ]]; then
  exec bash "$RELAX/scripts/entrypoint/ray-job.sh" "${BASH_SOURCE[0]}" "$@"
fi
source "$RELAX/scripts/models/qwen36-35B-A3B.sh"
NUM_GPUS=16
TRAIN_MODE=sft
TP_SIZE=1
PP_SIZE=2
CP_SIZE=4
EP_SIZE=4
ETP_SIZE=1
if [[ "${MOE_FLEX_DISPATCHER_BACKEND:-hybridep}" != hybridep ]]; then
  echo 'This 16-GPU multimodal recipe uses hybridep; select a DeepEP recipe for other backends' >&2
  exit 2
fi
MOE_FLEX_DISPATCHER_BACKEND=hybridep
HF_CHECKPOINT="${HF_CHECKPOINT:-$MODEL_DIR/Qwen3.6-35B-A3B}"
EXP_NAME="${EXP_NAME:-qwen36-35b-sft-multimodal-16xB300}"  # gitleaks:allow -- stable experiment/checkpoint identity
RUN_TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
TRAIN_STEPS="${TRAIN_STEPS:-100}"
[[ "$TRAIN_STEPS" =~ ^[1-9][0-9]*$ ]] || { echo 'TRAIN_STEPS must be positive' >&2; exit 2; }
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-1600}"
export RELAX_LOGPROB_CE_IMPL=native RELAX_MTP_CE_IMPL=native RELAX_LOGPROB_FAST_PATH=0
export TRAIN_CUDA_CONNECTIONS=1
RECOMPUTE_METHOD=block
RECOMPUTE_NUM_LAYERS=20
export RELAX_DISABLE_VISION_BRANCH_SYNC=0
PROMPT_DATA="${PROMPT_DATA:-$DATA_DIR/multimodal-mixture.yaml}"
# Fixed EP4 intranode HybridEP and image-built FLA closure for this recipe.
export RELAX_LEGACY_FLA_ROOT="${RELAX_LEGACY_FLA_ROOT:-/opt/relax/compat/fla-0.4.2}"
if [[ "${DRY_RUN:-0}" != 1 ]]; then
  [[ -f "$RELAX_LEGACY_FLA_ROOT/fla/ops/common/chunk_o.py" ]] || {
    echo "Missing image-built legacy FLA runtime: $RELAX_LEGACY_FLA_ROOT" >&2; exit 2;
  }
fi
export PYTHONPATH="$RELAX_LEGACY_FLA_ROOT${PYTHONPATH:+:$PYTHONPATH}"
SFT_MAX_IN_FLIGHT_STEPS="${SFT_MAX_IN_FLIGHT_STEPS:-2}"
SFT_PREFETCH_BUFFER_SIZE="${SFT_PREFETCH_BUFFER_SIZE:-128}"
DATA_PAD_SIZE_MULTIPLIER="${DATA_PAD_SIZE_MULTIPLIER:-8192}"
INVALID_MULTIMODAL_STRATEGY="${INVALID_MULTIMODAL_STRATEGY:-skip}"
CONVERSATION_KEY_MAP="${CONVERSATION_KEY_MAP:-}"
if [[ -z "$CONVERSATION_KEY_MAP" ]]; then
  CONVERSATION_KEY_MAP='{"tool_call":"assistant","step_loss_mask":"legacy_step_loss_mask"}'
fi

SFT_ARGS=(
  --resource "{\"sft\":[1,0],\"actor\":[1,$NUM_GPUS]}" --max-staleness 0 --num-data-storage-units 8
  --hf-checkpoint "$HF_CHECKPOINT" --ref-load "$HF_CHECKPOINT" --load "${LOAD_DIR:-$SAVE_DIR/$EXP_NAME}"
  --megatron-to-hf-mode bridge --warm-hf-checkpoint-page-cache --max-actor-ckpt-to-keep 5
  --num-rollout "$TRAIN_STEPS" --loss-type sft --prompt-data "$PROMPT_DATA"
  --seq-length "${SEQ_LENGTH:-262144}" --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU:-65536}" --global-batch-size "$GLOBAL_BATCH_SIZE"
  --use-dynamic-batch-size --balance-data
  --sft-oversize-strategy "${SFT_OVERSIZE_STRATEGY:-skip}" --data-pad-size-multiplier "${DATA_PAD_SIZE_MULTIPLIER:-1024}"
  --sft-max-in-flight-steps "${SFT_MAX_IN_FLIGHT_STEPS:-3}"
  --sft-prefetch-buffer-size "${SFT_PREFETCH_BUFFER_SIZE:-2048}" --sft-prefetch-chunk-size 128
  --sft-prefetch-num-workers "${SFT_PREFETCH_NUM_WORKERS:-16}" --sft-tq-timeout-minutes 60
  --sft-chunked-logits --sft-logits-chunk-size "${SFT_LOGITS_CHUNK_SIZE:-1024}"
  --eval-size "${EVAL_SIZE:-0.01}" --eval-interval "${EVAL_INTERVAL:-0}"
  --tensor-model-parallel-size "$TP_SIZE" --pipeline-model-parallel-size "$PP_SIZE"
  --context-parallel-size "$CP_SIZE" --expert-model-parallel-size "$EP_SIZE" --expert-tensor-parallel-size "$ETP_SIZE"
  --linear-cp-mode headwise --log-probs-chunk-size 2048 --recompute-loss-function
  --recompute-granularity full --recompute-method "$RECOMPUTE_METHOD" --recompute-num-layers "$RECOMPUTE_NUM_LAYERS"
  --calculate-per-token-loss --moe-token-dispatcher-type flex --moe-flex-dispatcher-backend "$MOE_FLEX_DISPATCHER_BACKEND"
  --cross-entropy-loss-fusion --moe-shared-expert-overlap --overlap-grad-reduce --overlap-param-gather
  --per-rank-fetch --sft-train-data-prefetch
)
OPTIMIZER_ARGS=(
  --optimizer adam --lr "${LR:-1e-5}" --min-lr "${MIN_LR:-1e-6}" --lr-decay-style cosine
  --lr-decay-iters "$TRAIN_STEPS" --lr-warmup-iters "${WARMUP_STEPS:-$((TRAIN_STEPS / 20))}"
  --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.98 --adam-eps 1e-8 --clip-grad 1.0
  --optimizer-cpu-offload --optimizer-offload-fraction "${OPTIMIZER_OFFLOAD_FRACTION:-0.8}"
  --overlap-cpu-optimizer-d2h-h2d --use-precision-aware-optimizer --no-rope-fusion
  --moe-router-load-balancing-type aux_loss --moe-aux-loss-coeff 1e-4 --moe-router-dtype fp32
)
MONITOR_ARGS=(
  --seed 42 --attention-dropout 0.0 --hidden-dropout 0.0 --accumulate-allreduce-grads-in-fp32
  --attention-softmax-in-fp32 --attention-backend auto --use-health-check
  --train-env-vars "{\"CUDA_DEVICE_MAX_CONNECTIONS\":\"$TRAIN_CUDA_CONNECTIONS\"}"
  --use-clearml --use-metrics-service --tb-project-name "${PROJECT_NAME:-Relax/$TRAIN_MODE/qwen_moe}"
  --tb-experiment-name "$EXP_NAME-${RUN_TIMESTAMP}"
)
ARGS=("${MODEL_ARGS[@]}" "${SFT_ARGS[@]}" "${OPTIMIZER_ARGS[@]}" "${MONITOR_ARGS[@]}")
if [[ "${SAVE_CHECKPOINT:-1}" == 1 ]]; then
  ARGS+=(--save "$SAVE_DIR/$EXP_NAME" --save-interval "${SAVE_INTERVAL:-50}")
fi
ARGS+=(--moe-hybridep-pad-variable-tokens)

ARGS+=(--decoder-first-pipeline-num-layers 20 --decoder-last-pipeline-num-layers 20)
ARGS+=(--input-key messages --mtp-num-layers 1 --enable-mtp-training --mtp-loss-scaling-factor 0.1
  --mtp-detach-paths none --sft-ignore-empty-think)
DEFAULT_MULTIMODAL_KEYS='{"image":"images"}'
ARGS+=(--multimodal-keys "${MULTIMODAL_KEYS:-$DEFAULT_MULTIMODAL_KEYS}"
  --sft-invalid-multimodal-strategy "${INVALID_MULTIMODAL_STRATEGY:-error}")
if [[ -n "${CONVERSATION_KEY_MAP:-}" ]]; then ARGS+=(--conversation-key-map "$CONVERSATION_KEY_MAP"); fi

# Explicit trailing arguments permit tuning, but must not bypass modality/step contracts.
for arg in "$@"; do
  case "$arg" in
    --num-epoch*|--num-rollout*|--train-env-vars*|--sft-training-mode*|--sft-cpt-template*|--moe-flex-dispatcher-backend*|\
    --tensor-model-parallel-size*|--pipeline-model-parallel-size*|--context-parallel-size*|\
    --expert-model-parallel-size*|--expert-tensor-parallel-size*|--resource*)
      echo "Use the entry configuration and TRAIN_STEPS instead of $arg" >&2; exit 2 ;;
  esac
done
ARGS+=(--sft-packing-strategy binpack --max-global-restart "${MAX_GLOBAL_RESTART:-0}"
  --metadata-key meta --tool-key tools --image-max-token-num "${IMAGE_MAX_TOKEN_NUM:-512}"
  --apply-chat-template-kwargs '{"enable_thinking":false,"preserve_thinking":true,"add_non_thinking_prefix":true,"legacy_tool_call_content":true}')
case "${FREEZE_VISION:-1}" in
  1) ARGS+=(--freeze-vision-model --freeze-vision-projection) ;;
  0) ;;
  *) echo 'FREEZE_VISION must be 0 or 1' >&2; exit 2 ;;
esac
ARGS+=("$@")

export TRITON_PTXAS_PATH="${TRITON_PTXAS_PATH:-/usr/local/cuda/bin/ptxas}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
if [[ "${DRY_RUN:-0}" != 1 ]]; then : "${NCCL_SOCKET_IFNAME:?Set NCCL_SOCKET_IFNAME}"; fi
export NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME="${NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME:-${NCCL_SOCKET_IFNAME:-}}"
RUNTIME_ENV_JSON=$(python3 - <<'PY'
import json, os
runtime=json.loads(os.environ.get('RUNTIME_ENV_JSON') or '{}')
env=runtime.setdefault('env_vars',{})
legacy_fla=os.environ.get('RELAX_LEGACY_FLA_ROOT')
if legacy_fla:
 env['PYTHONPATH']=legacy_fla+':'+env.get('PYTHONPATH','')
 env['RELAX_LEGACY_FLA_ROOT']=legacy_fla
env.update({'CUDA_DEVICE_MAX_CONNECTIONS':'1','NCCL_NVLS_ENABLE':'0',
 'TORCH_NCCL_AVOID_RECORD_STREAMS':'1','NVTE_BWD_LAYERNORM_SM_MARGIN':'20','NVTE_FWD_LAYERNORM_SM_MARGIN':'20',
 'NCCL_IGNORE_CPU_AFFINITY':'1','NCCL_GRAPH_REGISTER':'0','NVTE_USE_CUTLASS_GROUPED_GEMM':'1',
 'TORCH_DIST_INIT_BARRIER':'1','TORCH_NCCL_BLOCKING_WAIT':'0','TORCH_NCCL_ASYNC_ERROR_HANDLING':'1',
 'TORCH_DISTRIBUTED_DEFAULT_TIMEOUT':'3600','SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK':'256'})
for key in ('RELAX_LOGPROB_CE_IMPL','RELAX_MTP_CE_IMPL','RELAX_LOGPROB_FAST_PATH','RELAX_DISABLE_VISION_BRANCH_SYNC',
 'TRITON_PTXAS_PATH','PYTORCH_ALLOC_CONF','NCCL_SOCKET_IFNAME','NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME',
 'RELAX_SFT_DATA_NODE_IP','CUDA_HOME','CPATH','TORCHINDUCTOR_COMPILE_THREADS','TRITON_CACHE_DIR','HYBRID_EP_CACHE_DIR'):
 if key in os.environ: env[key]=os.environ[key]
print(json.dumps(runtime))
PY
)
RAY_ARGS=(job submit --address="${RAY_ADDRESS:?Set RAY_ADDRESS to the Jobs HTTP URL}" --runtime-env-json="$RUNTIME_ENV_JSON")
if [[ -n "${RELAX_SFT_DATA_NODE_IP:-}" ]]; then
  DATA_NODE_RESOURCES=$(python3 - <<'PYDATA'
import json, os
print(json.dumps({"node:" + os.environ["RELAX_SFT_DATA_NODE_IP"]: 0.001}))
PYDATA
)
  RAY_ARGS+=(--entrypoint-resources "$DATA_NODE_RESOURCES")
fi
[[ -z "$RAY_NO_WAIT" ]] || RAY_ARGS+=(--no-wait)
[[ -z "${WORKING_DIR:-}" ]] || RAY_ARGS+=(--working-dir "$WORKING_DIR")
if [[ "${DRY_RUN:-0}" == 1 ]]; then
  python3 - "$RUNTIME_ENV_JSON" "${ARGS[@]}" <<'PYRENDER'
import json, sys
print(json.dumps({'args': sys.argv[2:], 'runtime_env': json.loads(sys.argv[1])}, indent=2))
PYRENDER
  exit 0
fi
mkdir -p "$SCRIPT_DIR/log"
ray "${RAY_ARGS[@]}" -- python3 -m relax.entrypoints.train "${ARGS[@]}" 2>&1 | tee "$SCRIPT_DIR/log/$EXP_NAME-$RUN_TIMESTAMP.log"
