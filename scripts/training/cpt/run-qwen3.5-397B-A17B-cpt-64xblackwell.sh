#!/usr/bin/env bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Qwen3.5-397B-A17B cpt on 64 Blackwell GPUs; use the merged CUDA 13 image.
# MODEL_DIR=/models DATA_DIR=/data SAVE_DIR=/checkpoints bash \
#   scripts/training/cpt/run-qwen3.5-397B-A17B-cpt-64xblackwell.sh [--multimodal]
# DRY_RUN=1 prints argv/runtime JSON without contacting Ray. TRAIN_STEPS defaults to 100.
# EXP_NAME is stable per mode/modality; resume from SAVE_DIR/EXP_NAME (save every 50 steps, retain 5).
# Override PROMPT_DATA, GLOBAL_BATCH_SIZE, LR, SEQ_LENGTH or MAX_TOKENS_PER_GPU as needed.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RELAX_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
export RELAX="$RELAX_ROOT"
export RELAX_SKIP_CLUSTER_CLEANUP=1
MULTIMODAL=0
for arg in "$@"; do
  case "$arg" in
    --multimodal) MULTIMODAL=1 ;;
    *) echo "Unsupported argument: $arg (supported: --multimodal; tune documented environment variables)" >&2; exit 2 ;;
  esac
done
[[ "${MOE_FLEX_DISPATCHER_BACKEND:-deepep}" == deepep ]] || {
  echo 'This TP2/PP4/CP4/EP8/ETP1 preset requires deepep; HybridEP is unsupported.' >&2; exit 2;
}
: "${MODEL_DIR:?Set MODEL_DIR to the model root}"
: "${DATA_DIR:?Set DATA_DIR to the dataset root}"
: "${SAVE_DIR:?Set SAVE_DIR to a persistent checkpoint root}"
TRAIN_STEPS="${TRAIN_STEPS:-100}"
[[ "$TRAIN_STEPS" =~ ^[1-9][0-9]*$ ]] || { echo 'TRAIN_STEPS must be positive' >&2; exit 2; }
[[ -z "${NUM_EPOCH:-}" ]] || { echo 'Use TRAIN_STEPS; unset NUM_EPOCH' >&2; exit 2; }
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-1600}"
PROJECT_NAME="${PROJECT_NAME:-Relax/cpt/qwen3.5}"
INPUT_MODE=text
[[ "$MULTIMODAL" != 1 ]] || INPUT_MODE=multimodal
EXP_NAME="${EXP_NAME:-qwen3.5-397b-a17b-cpt-64xb300-$INPUT_MODE}"  # gitleaks:allow -- stable experiment/checkpoint identity
now="$(date +%Y%m%d-%H%M%S)"
export RAY_NO_WAIT="${RAY_NO_WAIT-}"
export RELAX_LOGPROB_CE_IMPL=native RELAX_MTP_CE_IMPL=native RELAX_LOGPROB_FAST_PATH=0
export RELAX_DISABLE_VISION_BRANCH_SYNC=1
if [[ "$MULTIMODAL" == 1 ]]; then
  export RELAX_DISABLE_VISION_BRANCH_SYNC=0
  PROMPT_DATA="${PROMPT_DATA:-$DATA_DIR/multimodal.jsonl}"
else
  PROMPT_DATA="${PROMPT_DATA:-$DATA_DIR/cpt.yaml}"
fi
if [[ -z "${RELAX_ENTRYPOINT_MODE:-}" && "${DRY_RUN:-0}" != 1 ]]; then
  source "$RELAX_ROOT/scripts/entrypoint/ray-job.sh"
fi
source "$RELAX_ROOT/scripts/models/qwen35-397B-A17B.sh"
CKPT_ARGS=(
  --resource "{\"sft\":[1,0],\"actor\":[1,64]}" --max-staleness 0 --num-data-storage-units 8
  --hf-checkpoint "${MODEL_DIR}/Qwen3.5-397B-A17B" --ref-load "${MODEL_DIR}/Qwen3.5-397B-A17B" --load "$SAVE_DIR/$EXP_NAME"
  --megatron-to-hf-mode bridge --warm-hf-checkpoint-page-cache --max-actor-ckpt-to-keep 5
)
SFT_ARGS=(
  --num-rollout "$TRAIN_STEPS" --loss-type sft --prompt-data "$PROMPT_DATA"
  --seq-length "${SEQ_LENGTH:-262144}" --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU:-65536}" --global-batch-size "$GLOBAL_BATCH_SIZE"
  --use-dynamic-batch-size --balance-data
  --sft-oversize-strategy "${SFT_OVERSIZE_STRATEGY:-skip}" --data-pad-size-multiplier "${DATA_PAD_SIZE_MULTIPLIER:-1024}"
  --sft-max-in-flight-steps "${SFT_MAX_IN_FLIGHT_STEPS:-3}"
  --sft-prefetch-buffer-size "${SFT_PREFETCH_BUFFER_SIZE:-2048}" --sft-prefetch-chunk-size 128
  --sft-prefetch-num-workers "${SFT_PREFETCH_NUM_WORKERS:-16}" --sft-tq-timeout-minutes 60
  --sft-chunked-logits --sft-logits-chunk-size "${SFT_LOGITS_CHUNK_SIZE:-1024}"
  --eval-size "${EVAL_SIZE:-0.01}" --eval-interval "${EVAL_INTERVAL:-0}"
)
PERF_ARGS=(
  --tensor-model-parallel-size "2" --pipeline-model-parallel-size "4"
  --context-parallel-size "4" --expert-model-parallel-size "8" --expert-tensor-parallel-size "1"
  --linear-cp-mode headwise --log-probs-chunk-size 2048 --recompute-loss-function
  --recompute-granularity full --recompute-method "uniform" --recompute-num-layers "1"
  --calculate-per-token-loss --moe-token-dispatcher-type flex --moe-flex-dispatcher-backend "deepep"
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
  --seed 42 --attention-dropout 0.0 --hidden-dropout 0.0 --accumulate-allreduce-grads-in-fp32
  --attention-softmax-in-fp32 --attention-backend auto --use-health-check
  --train-env-vars "{\"CUDA_DEVICE_MAX_CONNECTIONS\":\"32\"}"
)
TRACKING_ARGS=(
  --use-clearml --use-metrics-service --tb-project-name "$PROJECT_NAME"
  --tb-experiment-name "$EXP_NAME-${now}"
)
CKPT_ARGS+=(--save "$SAVE_DIR/$EXP_NAME" --save-interval "${SAVE_INTERVAL:-50}")
PERF_ARGS+=(--sequence-parallel --decoder-first-pipeline-num-layers 12 --decoder-last-pipeline-num-layers 12)
SFT_ARGS+=(--sft-training-mode cpt --mtp-num-layers 0 --mtp-loss-scaling-factor 0
  --tool-key '' --eval-tool-key '' --metadata-key '')
if [[ "$MULTIMODAL" == 1 ]]; then
  SFT_ARGS+=(--sft-cpt-template multimodal --input-key messages)
else
  SFT_ARGS+=(--sft-cpt-template "${CPT_TEMPLATE:-raw}" --input-key "${CPT_INPUT_KEY:-auto}")
fi
if [[ "$MULTIMODAL" == 1 ]]; then
  MULTIMODAL_KEYS="${MULTIMODAL_KEYS:-}"
  [[ -n "$MULTIMODAL_KEYS" ]] || MULTIMODAL_KEYS='{"image":"images"}'
  python3 - "$MULTIMODAL_KEYS" <<'PYJSON'
import json, sys
mapping = json.loads(sys.argv[1])
if not isinstance(mapping, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in mapping.items()):
    raise ValueError("MULTIMODAL_KEYS must map media types to dataset column names")
PYJSON
  SFT_ARGS+=(--multimodal-keys "$MULTIMODAL_KEYS"
    --sft-invalid-multimodal-strategy "${INVALID_MULTIMODAL_STRATEGY:-error}")
  if [[ -n "${CONVERSATION_KEY_MAP:-}" ]]; then SFT_ARGS+=(--conversation-key-map "$CONVERSATION_KEY_MAP"); fi
fi
ARGS=("${MODEL_ARGS[@]}" "${CKPT_ARGS[@]}" "${SFT_ARGS[@]}" "${PERF_ARGS[@]}" "${OPTIMIZER_ARGS[@]}" "${TRACKING_ARGS[@]}")
export TRITON_PTXAS_PATH="${TRITON_PTXAS_PATH:-/usr/local/cuda/bin/ptxas}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
if [[ -n "${NCCL_SOCKET_IFNAME:-}" ]]; then
  export NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME="${NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME:-$NCCL_SOCKET_IFNAME}"
fi
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
 'RELAX_SFT_DATA_NODE_IP','CUDA_HOME','CPATH','TORCHINDUCTOR_COMPILE_THREADS','TRITON_CACHE_DIR'):
 if key in os.environ: env[key]=os.environ[key]
print(json.dumps(runtime))
PY
)
RAY_ARGS=(job submit --address="${RAY_ADDRESS:-http://127.0.0.1:8265}" --runtime-env-json="$RUNTIME_ENV_JSON")
if [[ -n "${RELAX_SFT_DATA_NODE_IP:-}" ]]; then
  DATA_NODE_RESOURCES=$(python3 -c 'import json, os; print(json.dumps({"node:" + os.environ["RELAX_SFT_DATA_NODE_IP"]: 0.001}))')
  RAY_ARGS+=(--entrypoint-resources "$DATA_NODE_RESOURCES")
fi
[[ -z "${RAY_NO_WAIT}" ]] || RAY_ARGS+=(--no-wait)
[[ -z "${WORKING_DIR:-}" ]] || RAY_ARGS+=(--working-dir "$WORKING_DIR")
if [[ "${DRY_RUN:-0}" == 1 ]]; then
  python3 - "$RUNTIME_ENV_JSON" "${ARGS[@]}" <<'PYDRY'
import json, sys
print(json.dumps({"args": sys.argv[2:], "runtime_env": json.loads(sys.argv[1])}, indent=2))
PYDRY
  exit 0
fi
: "${RAY_ADDRESS:?Set RAY_ADDRESS to the Ray Jobs HTTP URL}"
mkdir -p "$SCRIPT_DIR/log"
ray "${RAY_ARGS[@]}" -- python3 -m relax.entrypoints.train "${ARGS[@]}" 2>&1 | tee "$SCRIPT_DIR/log/$EXP_NAME-$now.log"
