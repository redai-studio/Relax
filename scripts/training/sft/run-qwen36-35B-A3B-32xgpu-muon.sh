#!/usr/bin/env bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Qwen3.6-35B-A3B multimodal SFT on 4 x 8 Blackwell GPUs with Megatron Muon.
# Every training node needs the optimized Blackwell image with FlashQLA, patched FLA 0.5.2, DeepEP v2 and emerging-optimizers
# compatible with the image's Megatron. These packages must be available to the Ray training workers.
# Expanded DeepEP requires each EP8 group on one NVLink node. TP4 keeps long samples intact through 524288
# expanded tokens (131072 rows per rank) and skips longer samples before packing.
# Default Wholepack requires the TransferQueue assignment_transform patch in the Ray job driver's Python environment.
# Set QWEN35_WHOLEPACK=0 to use binpack with an unpatched TransferQueue.
# MODEL_DIR=/models/Qwen3.6-35B-A3B DATA_DIR=/data/parquet SAVE_DIR=/checkpoints \
#   bash scripts/training/sft/run-qwen36-35B-A3B-32xgpu-muon.sh
# Set DRY_RUN=1 to inspect the Ray job. Reuse EXP_NAME to resume the same run.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export RELAX="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
: "${MODEL_DIR:?Set MODEL_DIR to the Qwen3.6-35B-A3B HF checkpoint directory}"
: "${DATA_DIR:?Set DATA_DIR to the multimodal SFT dataset}"
: "${SAVE_DIR:?Set SAVE_DIR to a persistent shared checkpoint root}"
export RAY_ADDRESS="${RAY_ADDRESS:-http://127.0.0.1:8265}"
export RAY_NO_WAIT="${RAY_NO_WAIT-}"
export RELAX_SKIP_CLUSTER_CLEANUP="${RELAX_SKIP_CLUSTER_CLEANUP:-1}"
if [[ "${DRY_RUN:-0}" != 1 && -z "${RELAX_ENTRYPOINT_MODE:-}" ]]; then
    exec bash "$RELAX/scripts/entrypoint/ray-job.sh" "${BASH_SOURCE[0]}" "$@"
fi
source "$RELAX/scripts/models/qwen36-35B-A3B.sh"

PROJECT_NAME="${PROJECT_NAME:-Relax/sft/qwen36}"
EXP_NAME="${EXP_NAME:-qwen36-35B-A3B-muon-native-tp4-32gpu}"
RUN_NAME="$EXP_NAME-$(date +%Y%m%d-%H%M%S)-$$"
CHECKPOINT_DIR="$SAVE_DIR/$EXP_NAME"

ENV_JSON="$(python3 - <<'PY'
import json
import os
import sysconfig
from pathlib import Path

runtime = json.loads(os.environ.get("RUNTIME_ENV_JSON") or "{}")
env = runtime.setdefault("env_vars", {})
if "PYTHONPATH" in os.environ:
    env.setdefault("PYTHONPATH", os.environ["PYTHONPATH"])
site = Path(sysconfig.get_paths()["purelib"])
nccl = site / "nvidia/nccl"
cublas = Path("/opt/relax/cublas13.3/nvidia/cu13/lib")
preloads = os.environ.get("LD_PRELOAD", "").replace(":", " ").split()
libs = [cublas / "libcublasLt.so.13", cublas / "libcublas.so.13", nccl / "lib/libnccl.so.2"]
libs.extend(p for p in preloads if not Path(p).name.startswith(("libnccl.so", "libcublas.so", "libcublasLt.so")))
libdirs = [cublas, nccl / "lib", site / "nvidia/nvshmem/lib"]
train = {
    "LD_PRELOAD": ":".join(map(str, libs)),
    "LD_LIBRARY_PATH": ":".join([*map(str, libdirs), env.get("LD_LIBRARY_PATH", os.getenv("LD_LIBRARY_PATH", ""))]),
    "EP_NCCL_ROOT_DIR": str(nccl),
    "EP_JIT_CACHE_DIR": f"/tmp/relax-qwen35-{os.getuid()}/deepep",
    "EP_DISABLE_GIN": "1",
    "RELAX_FLA_CONV_NB_MODE": "none",
    "NCCL_CUMEM_ENABLE": "1",
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "NCCL_NVLS_ENABLE": "0",
    "NCCL_IGNORE_CPU_AFFINITY": "1",
    "NCCL_GRAPH_REGISTER": "0",
    "NVTE_GROUPED_LINEAR_USE_FUSED_GROUPED_GEMM": "0",
    "NVTE_CUTEDSL_FUSED_GROUPED_MLP": "0",
    "NVTE_USE_CUTLASS_GROUPED_GEMM": "1",
    "NVTE_BWD_LAYERNORM_SM_MARGIN": "20",
    "NVTE_FWD_LAYERNORM_SM_MARGIN": "20",
    "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
    "TORCH_DIST_INIT_BARRIER": "1",
    "TORCH_NCCL_BLOCKING_WAIT": "0",
    "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
    "TORCH_DISTRIBUTED_DEFAULT_TIMEOUT": "3600",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "256",
    "TORCHINDUCTOR_COMPILE_THREADS": "1",
    "FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED": "0",
    "RELAX_DISABLE_VISION_BRANCH_SYNC": "0",
    "RELAX_SFT_SELECTIVE_LOGITS": "1",
    "RELAX_LOGPROB_CE_IMPL": "native",
    "RELAX_MTP_CE_IMPL": "native",
    "RELAX_LOGPROB_FAST_PATH": "0",
    "RELAX_SKIP_NONFINITE_GRAD": os.getenv("RELAX_SKIP_NONFINITE_GRAD", "1"),
    "PYTORCH_ALLOC_CONF": "backend:native,expandable_segments:True",
    "TOKENIZERS_PARALLELISM": "false",
}
thread_limits = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
train.update(dict.fromkeys(thread_limits, "24"))
for key in ("CUDA_HOME", "CPATH", "TRITON_PTXAS_PATH", "TRITON_CACHE_DIR", "NCCL_SOCKET_IFNAME",
            "NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME", "CLEARML_OFFLINE_MODE", "CLEARML_CACHE_DIR"):
    if key in os.environ:
        train[key] = os.environ[key]
env.update(train)
# The data service uses one BLAS thread per process. Train actors get their own limits.
workers = dict.fromkeys(thread_limits, "1")
workers["TOKENIZERS_PARALLELISM"] = "false"
workers["RELAX_PROCESSOR_POOL_SIZE"] = os.getenv("RELAX_PROCESSOR_POOL_SIZE", "96")
data_node = os.getenv("RELAX_SFT_DATA_NODE_IP") or env.get("MASTER_ADDR") or os.getenv("MASTER_ADDR")
if data_node:
    workers["RELAX_SFT_DATA_NODE_IP"] = data_node
env.update(workers)
for key in ("NCCL_PROTO", "NCCL_ALGO"):
    env.pop(key, None)
forward = env.get("RELAX_PROPAGATE_ENV_VARS", os.getenv("RELAX_PROPAGATE_ENV_VARS", "")).split(",")
env["RELAX_PROPAGATE_ENV_VARS"] = ",".join(dict.fromkeys([*filter(None, forward), *workers]))
print(json.dumps(train))
print(json.dumps(runtime))
PY
)"
TRAIN_ENV_JSON="${ENV_JSON%%$'\n'*}"
RUNTIME_ENV_JSON="${ENV_JSON#*$'\n'}"

CKPT_ARGS=(
    --hf-checkpoint "$MODEL_DIR"
    --ref-load "$MODEL_DIR"
    --megatron-to-hf-mode bridge
    --warm-hf-checkpoint-page-cache
    --load "$CHECKPOINT_DIR"
    --save "$CHECKPOINT_DIR"
    --save-interval "${SAVE_INTERVAL:-50}"
    --max-actor-ckpt-to-keep 5
)

SFT_ARGS=(
    --loss-type sft
    --num-rollout "${TRAIN_STEPS:-50}"
    --prompt-data "$DATA_DIR"
    --input-key messages
    --metadata-key meta
    --tool-key tools
    --multimodal-keys '{"image":"images"}'
    --image-min-token-num 64
    --image-max-token-num 16384
    --apply-chat-template-kwargs '{"enable_thinking":false,"preserve_thinking":true,"add_non_thinking_prefix":true}'
    --sft-ignore-empty-think
    --freeze-vision-model
    --freeze-vision-projection
    --enable-mtp-training
    --mtp-num-layers 1
    --mtp-loss-scaling-factor 0.1
    --mtp-detach-paths none
    --global-batch-size "${GLOBAL_BATCH_SIZE:-800}"
    --seq-length 262144
    --max-tokens-per-gpu 262144
    --sft-oversize-strategy custom
    --sft-oversize-custom-function-path relax.utils.data.wholepack_policy.keep_within_tp4_deepep_capacity
    --sft-invalid-multimodal-strategy error
    --use-dynamic-batch-size
    --balance-data
    --per-rank-fetch
    --sft-train-data-prefetch
    --sft-max-in-flight-steps 3
    --sft-prefetch-buffer-size 2048
    --sft-prefetch-num-workers "${SFT_PREFETCH_NUM_WORKERS:-96}"
    --sft-prefetch-chunk-size 128
    --sft-tq-timeout-minutes 60
    --eval-size 0.01
    --eval-interval 0
)

case "${QWEN35_WHOLEPACK:-1}" in
    1) SFT_ARGS+=(--sft-wholepack) ;;
    0) SFT_ARGS+=(--sft-packing-strategy binpack) ;;
    *) echo "QWEN35_WHOLEPACK must be 0 or 1" >&2; exit 2 ;;
esac

# TP4 / PP1 / CP1 gives 8 dense data replicas, 100 samples each at GBS 800.
PARALLEL_ARGS=(
    --tensor-model-parallel-size 4
    --sequence-parallel
    --pipeline-model-parallel-size 1
    --context-parallel-size 1
    --linear-cp-mode headwise
    --expert-model-parallel-size 8
    --expert-tensor-parallel-size 1
    --moe-token-dispatcher-type flex
    --moe-flex-dispatcher-backend deepepv2
    --custom-megatron-init-path relax.models.qwen3_5.tilelang_nvcc.install
    --custom-megatron-before-train-step-hook-path relax.models.qwen3_5.recipe.before_step
    --moe-flex-dispatcher-num-sms 80
    --moe-router-fusion
    --moe-shared-expert-overlap
    --overlap-dispatch-backward-with-experts-wgrad
    --data-pad-size-multiplier 1024
    --recompute-granularity full
    --recompute-method block
    --recompute-num-layers 35
    --recompute-loss-function
    --sft-chunked-logits
    --sft-logits-chunk-size 1024
    --log-probs-chunk-size 2048
    --calculate-per-token-loss
    --cross-entropy-loss-fusion
    --cross-entropy-fusion-impl linear
    --no-rope-fusion
    --moe-router-load-balancing-type aux_loss
    --moe-aux-loss-coeff 1e-4
    --overlap-grad-reduce
    --overlap-param-gather
)

OPTIMIZER_ARGS=(
    --bf16
    --optimizer muon
    --use-distributed-optimizer
    --no-use-precision-aware-optimizer
    --optimizer-offload-fraction 0
    --muon-momentum 0.95
    --muon-nesterov
    --muon-coefficient-type polar_express
    --muon-num-ns-steps 8
    --muon-tp-mode duplicated
    --muon-scale-mode spectral
    --muon-extra-scale-factor 0.2
    --lr 2e-5
    --min-lr 1e-6
    --lr-decay-style cosine
    --lr-warmup-iters 10
    --lr-warmup-init 0
    --lr-decay-iters 200
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.98
    --adam-eps 1e-8
    --no-check-for-nan-in-loss-and-grad
    --clip-grad 1.0
    --accumulate-allreduce-grads-in-fp32
    --attention-softmax-in-fp32
    --attention-backend auto
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --seed 42
)

TRAIN_ARGS=(
    python3 -m relax.entrypoints.train
    --resource '{"sft":[1,0],"actor":[1,32]}'
    --num-data-storage-units 8
    --max-staleness 0
    --max-global-restart 0
    --train-env-vars "$TRAIN_ENV_JSON"
    --use-health-check
    --use-clearml
    --use-metrics-service
    --tb-project-name "$PROJECT_NAME"
    --tb-experiment-name "$RUN_NAME"
    "${MODEL_ARGS[@]}"
    "${CKPT_ARGS[@]}"
    "${SFT_ARGS[@]}"
    "${PARALLEL_ARGS[@]}"
    "${OPTIMIZER_ARGS[@]}"
)
# Older launch commands include this script-level flag. Multimodal SFT is the default.
for arg in "$@"; do
    if [[ "$arg" != --multimodal ]]; then TRAIN_ARGS+=("$arg"); fi
done

if [[ "${DRY_RUN:-0}" == 1 ]]; then
    python3 - "$RUNTIME_ENV_JSON" "${TRAIN_ARGS[@]}" <<'PY'
import json
import sys

print(json.dumps({"runtime_env": json.loads(sys.argv[1]), "argv": sys.argv[2:]}, indent=2))
PY
    exit 0
fi

RAY_ARGS=(
    job submit
    --address="${RAY_JOBS_ADDRESS:-$RAY_ADDRESS}"
    --submission-id="${SUBMISSION_ID:-$RUN_NAME}"
    --runtime-env-json="$RUNTIME_ENV_JSON"
)
if [[ -n "$RAY_NO_WAIT" ]]; then RAY_ARGS+=(--no-wait); fi
if [[ -n "${WORKING_DIR:-}" ]]; then RAY_ARGS+=(--working-dir "$WORKING_DIR"); fi
mkdir -p "$SCRIPT_DIR/log"
ray "${RAY_ARGS[@]}" -- "${TRAIN_ARGS[@]}" 2>&1 | tee "$SCRIPT_DIR/log/$RUN_NAME.log"
