#!/usr/bin/env bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Qwen3.8 Hopper CP/PLE and DoSA validation on two eight-H800 nodes.
# For the matched 16-H800 business arms set CP_SIZE=2 PP_SIZE=8 EP_SIZE=2,
# OPTIMIZER_OFFLOAD_MODE=cpu and VALIDATION_ARM=radix-adaptive or dosa-adaptive.
# Supply MODEL_DIR, DATA_DIR, SAVE_DIR and MEGATRON from the authorized environment.
# DRY_RUN=1 never submits a job. Use a shared source checkout and the patched Hopper image.
set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RELAX_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
export RELAX="${RELAX_ROOT}"
source "${RELAX_ROOT}/scripts/models/qwen38-flash-next.sh"
# Checkpoint paths must agree with the recipe's saved configuration.
for arg in "$@"; do
    case "${arg}" in
        --load|--load=*|--save|--save=*)
            echo 'Use SAVE_DIR, EXP_NAME and SAVE_INTERVAL instead of --load/--save' >&2
            exit 2 ;;
    esac
done
VALIDATION_ARM="${VALIDATION_ARM:-radix-full}"
enable_ple_halo=1
case "${VALIDATION_ARM}" in
    radix-native) selector=radix; transport=full; enable_ple_halo=0 ;;
    radix-full) selector=radix; transport=full ;;
    radix-adaptive) selector=radix; transport=adaptive ;;
    dosa-adaptive) selector=dosa; transport=adaptive ;;
    *) echo 'Invalid VALIDATION_ARM: choose radix-native, radix-full, radix-adaptive, or dosa-adaptive' >&2; exit 2 ;;
esac
SEQ_LENGTH="${SEQ_LENGTH:-32768}"
CP_SIZE="${CP_SIZE:-2}"
PP_SIZE="${PP_SIZE:-2}"
if [[ "${CP_SIZE}" != 2 && "${CP_SIZE}" != 4 ]]; then
    echo 'This 16-H800 recipe requires CP_SIZE=2 or 4' >&2
    exit 2
fi
case "${PP_SIZE}" in
    2) default_layout='Et*24|t*24mL'; default_ep=8 ;;
    4) default_layout='Et*13|t*13|t*12|t*10mL'; default_ep=4 ;;
    8) default_layout='Et*5|t*6|t*6|t*7|t*7|t*6|t*6|t*5mL'; default_ep=2 ;;
    *) echo 'This 16-H800 recipe requires PP_SIZE=2, 4 or 8' >&2; exit 2 ;;
esac
if [[ "${PP_SIZE}" == 8 && "${CP_SIZE}" != 2 ]]; then
    echo 'PP8 on 16 H800 with TP1 requires CP2' >&2
    exit 2
fi
EP_SIZE="${EP_SIZE:-${default_ep}}"
if [[ "${EP_SIZE}" != "${default_ep}" ]]; then
    echo 'This 16-H800 recipe requires PP2/EP8, PP4/EP4 or PP8/EP2 with TP1' >&2
    exit 2
fi
PIPELINE_LAYOUT="${PIPELINE_LAYOUT:-${default_layout}}"
export CONTEXT_LENGTH="${SEQ_LENGTH}-cp${CP_SIZE}-${VALIDATION_ARM}"
export SEQ_LENGTH CP_SIZE
export RELAX_QWEN38_QSA_SELECTOR_BACKEND="${selector}"
export QWEN38_PLE_INPUT_HALO_TRANSPORT="${transport}"
export QWEN38_DOSA_VARIANT=boundary
# Do not assume this image provides a working cuDNN DSA TopK wrapper on SM90.
export QWEN38_DOSA_INTERVAL_CONFIG='{"coarse_rows":64,"coarse_keys":128,"chunk_rows":512,"compact_tile":1024,"refine_tile":64,"capacity":256,"half_scores":true,"boundary_bins":2048}'
case "${RESUME:-0}" in
    0|1) ;;
    *) echo 'RESUME must be 0 or 1' >&2; exit 2 ;;
esac
if [[ "${RESUME:-0}" == 1 ]]; then
    : "${EXP_NAME:?Set EXP_NAME to the original experiment name for resume}"
fi
: "${MODEL_DIR:?Set MODEL_DIR to the HF model directory}"
: "${DATA_DIR:?Set DATA_DIR to the prompt-data directory or file}"
: "${SAVE_DIR:?Set SAVE_DIR to a persistent checkpoint root}"
: "${MEGATRON:?Set MEGATRON to the installed Megatron-LM repository}"
export MEGATRON
# Resolve shared libraries from the active Python environment without importing CUDA.
# Explicit library directories and inherited loader paths remain available.
LD_LIBRARY_PATH=$(python3 - <<'PYLIB'
import os
import site
from pathlib import Path

paths = []
for variable, relative in (("TORCH_LIB_PATH", "torch/lib"), ("NVSHMEM_LIB_PATH", "nvidia/nvshmem/lib")):
    directory = os.environ.get(variable)
    if not directory:
        directory = next((str(Path(root) / relative) for root in site.getsitepackages()
                          if (Path(root) / relative).is_dir()), None)
    if directory:
        paths.append(directory)
paths.extend(filter(None, os.environ.get("LD_LIBRARY_PATH", "").split(":")))
print(":".join(dict.fromkeys(paths)))
PYLIB
)
export LD_LIBRARY_PATH
if [[ ! "${SEQ_LENGTH}" =~ ^[1-9][0-9]*$ ]]; then
    echo 'SEQ_LENGTH must be a positive integer.' >&2
    exit 2
fi
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-$((SEQ_LENGTH / CP_SIZE))}"
if [[ ! "${SEQ_LENGTH}" =~ ^[1-9][0-9]*$ || ! "${MAX_TOKENS_PER_GPU}" =~ ^[1-9][0-9]*$ ]] ||
    (( MAX_TOKENS_PER_GPU * CP_SIZE > SEQ_LENGTH )); then
    echo "Require positive MAX_TOKENS_PER_GPU * CP_SIZE <= SEQ_LENGTH for the SFT sample limit" >&2
    exit 2
fi
export RAY_NO_WAIT="${RAY_NO_WAIT-}"
export RELAX_SKIP_CLUSTER_CLEANUP=1
export EXP_NAME="${EXP_NAME:-qwen38-${CONTEXT_LENGTH}-16xh800}"
export PROJECT_NAME="${PROJECT_NAME:-Relax/sft/qwen38-hopper-validation}"
now=$(date "+%Y%m%d-%H%M%S")
export SUBMISSION_ID="${SUBMISSION_ID:-${EXP_NAME}-${now}-$$}"

# RAY_API_SERVER_ADDRESS is shared by Ray node discovery and job submission.

# Use an exclusive Ray cluster; the operator must prevent concurrent training.
if [[ "${DRY_RUN:-0}" != 1 && -z "${RELAX_ENTRYPOINT_MODE:-}" ]]; then
    exec bash "${RELAX}/scripts/entrypoint/ray-job.sh" "${BASH_SOURCE[0]}" "$@"
fi

# Explicit driver/actor settings override entrypoint defaults (notably connections=1).
ENV_ARGS=(
    "QWEN48_PP_PAYLOAD_UNBATCHED=${QWEN48_PP_PAYLOAD_UNBATCHED:-1}"
    RELAX_QWEN38_PP_SHAPE_WARMUP=1
    "QWEN38_PLE_INPUT_HALO_CP=${enable_ple_halo}"
    "QWEN38_PLE_INPUT_HALO_TRANSPORT=${QWEN38_PLE_INPUT_HALO_TRANSPORT}"
    RELAX_QWEN38_QSA_CP=1
    RELAX_QWEN38_QSA_MIXED_LONG_BACKEND=grouped_cp
    "RELAX_QWEN38_GDN_CP_PREBUILD_ROUTE=${RELAX_QWEN38_GDN_CP_PREBUILD_ROUTE:-0}"
    RELAX_QWEN38_PLE_CP_FUSED_CONV=0
    RELAX_QWEN38_QSA_CP_TP_SELECTOR=0
    "RELAX_SKIP_NONFINITE_GRAD=${RELAX_SKIP_NONFINITE_GRAD:-0}"
    MCORE_QSA_SPARSE_BACKEND=id_sparse
    TORCHINDUCTOR_COMPILE_THREADS=1
    QWEN38_PLE_CPU_OFFLOAD=1
    "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}"
    RELAX_PROPAGATE_ENV_VARS=QWEN38_WORKSPACE,SAVE_DIR,RELAX_PROFILE_DIR,TENSORBOARD_DIR,CLEARML_CACHE_DIR,CLEARML_OFFLINE_MODE,PYTHONDONTWRITEBYTECODE,LD_LIBRARY_PATH,CUDA_DEVICE_MAX_CONNECTIONS,QWEN38_RUN_DIR,RELAX_SKIP_NONFINITE_GRAD,RELAX_QWEN38_QSA_SELECTOR_BACKEND,QWEN38_DOSA_VARIANT,QWEN38_DOSA_INTERVAL_CONFIG,QWEN38_PLE_INPUT_HALO_CP,QWEN38_PLE_INPUT_HALO_TRANSPORT,RELAX_QWEN38_QSA_MIXED_LONG_BACKEND
    NCCL_NVLS_ENABLE=0
    "PYTHONPATH=${RELAX}:${MEGATRON}${QWEN38_EXTRA_PYTHONPATH:+:${QWEN38_EXTRA_PYTHONPATH}}"
    "MEGATRON=${MEGATRON}"
    "RELAX=${RELAX}"
    PYTHONNOUSERSITE=1
    "CUDA_DEVICE_MAX_CONNECTIONS=${QWEN38_CUDA_DEVICE_MAX_CONNECTIONS:-1}"
    RELAX_FLA_CONV_NB_MODE=none
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.8
    TOKENIZERS_PARALLELISM=false
    RELAX_QWEN38_QSA_FUSED_ATTENTION=0
    RELAX_QWEN38_RETAIN_QSA_ATTENTION=0
    RELAX_QWEN38_QSA_FIXED_POOL=0
    RELAX_QWEN38_QSA_SPARSE_BLOCKMASK=1
    RELAX_QWEN38_RECORD_FUNCTION_TAGS=1
    RELAX_QWEN38_QSA_LOCAL_QUERY=1
    RELAX_QWEN38_ROUTER_SORTED_REPLAY=1
    RELAX_QWEN38_QSA_ROUTED_QKV=1
    RELAX_QWEN38_QSA_RECOMPUTE_REPLAY=1
    RELAX_QWEN38_QSA_CPU_LAYOUT=1
    RELAX_QWEN38_PLE_STREAMED=0
    RELAX_QWEN38_GDN_OUTPUT_SAFE_CONFIG=1
    RELAX_QWEN38_FLA_CONV_CACHE_REUSE=1
    RELAX_QWEN38_QSA_COMMUNITY_BACKEND=mixed
    "QWEN38_WORKSPACE=${RELAX}"
    "SAVE_DIR=${SAVE_DIR}"
    RELAX_SFT_RESIDENT_PARAM_OVERLAP=0
    RAY_OVERRIDE_JOB_RUNTIME_ENV=1
    "TENSORBOARD_DIR=${SAVE_DIR}/tensorboard"
    "CLEARML_CACHE_DIR=${SAVE_DIR}/clearml"
    "RELAX_PROFILE_DIR=${SAVE_DIR}/traces"
    PYTHONDONTWRITEBYTECODE=1
    RELAX_PROFILE_RANKS=0,8,16,24
    RELAX_PROFILE_GZIP_LEVEL=1
    FLA_CACHE_RESULTS=1
    TRITON_CACHE_AUTOTUNING=1
    TORCHINDUCTOR_FX_GRAPH_CACHE=1
    TORCHINDUCTOR_AUTOTUNE_LOCAL_CACHE=1
    "RELAX_AUTOTUNE_LOG_DIR=${SAVE_DIR}/autotune"
    RELAX_QWEN38_HC_FUSED_MIX=1
    # Recompute HC intermediates for long inputs and output heads.
    "RELAX_QWEN38_HC_RECOMPUTE=${RELAX_QWEN38_HC_RECOMPUTE:-1}"
    "RELAX_QWEN38_HC_OUTER_BYPASS_MAX_LOCAL_ROWS=${RELAX_QWEN38_HC_OUTER_BYPASS_MAX_LOCAL_ROWS:-65536}"
    RELAX_SFT_SELECTIVE_LOGITS=1
    RELAX_QWEN38_QSA_TP_SELECTOR=0
    RELAX_QWEN38_QSA_FUSED_SELECTOR=0
    RELAX_QWEN38_QSA_GROUPED_BACKWARD_WARPS=4
    RELAX_QWEN38_QSA_FLEX_SPARSE_MASK=1
    NVSHMEM_DISABLE_NCCL=1
    EP_DISABLE_GIN=1
    RELAX_QWEN38_HC_FUSED_COMBINE=1
    RELAX_QWEN38_PLE_FUSED_CONV=1
    RELAX_SFT_FUSED_LINEAR_CE=0
    "QWEN38_RUN_DIR=${SAVE_DIR}"
    RELAX_QWEN38_QSA_MIXED_FLEX_MAX_LEN=0
    RELAX_QWEN38_QSA_GROUPED_BACKWARD_STAGES=3
    RELAX_SFT_MTP_ACCEPTANCE_METRICS=0
    RELAX_QWEN38_QSA_GROUPED_BACKWARD_BLOCK_K=32
    RELAX_QWEN38_QSA_SELECTOR_COMPILE=1
)
unset NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN
export "${ENV_ARGS[@]}"
BASE_RUNTIME_ENV_JSON="${RUNTIME_ENV_JSON:-"{}"}"
TRAIN_ENV_JSON=$(python3 - "${BASE_RUNTIME_ENV_JSON}" "${ENV_ARGS[@]}" <<'PYENV'
import json
import os
import sys

base_env = json.loads(sys.argv[1]).get("env_vars", {})
train_env = dict(item.split("=", 1) for item in sys.argv[2:])
# Explicit shell overrides take priority over inherited Ray settings.
for key, default in (
    ("RELAX_QWEN38_QSA_SELECTOR_BACKEND", "torch"),
    ("QWEN38_DOSA_VARIANT", ""),
    ("QWEN38_DOSA_INTERVAL_CONFIG", "{}"),
):
    train_env[key] = os.environ.get(key, base_env.get(key)) or default
print(json.dumps(train_env))
PYENV
)
RUNTIME_ENV_JSON=$(python3 - "${BASE_RUNTIME_ENV_JSON}" "${TRAIN_ENV_JSON}" <<'PYRUNTIME'
import json
import sys
runtime = json.loads(sys.argv[1])
runtime.setdefault("env_vars", {}).pop("NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN", None)
runtime["env_vars"].update(json.loads(sys.argv[2]))
print(json.dumps(runtime))
PYRUNTIME
)

# PLE and vision stay on PP0. PP4 reduces PLE's replicated CPU table copies
# per host while keeping the same 16 GPUs and TP1 selector support.
PARALLEL_ARGS=(
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size "${PP_SIZE}"
    --context-parallel-size "${CP_SIZE}"
    --linear-cp-mode headwise
    --expert-model-parallel-size "${EP_SIZE}"
    --expert-tensor-parallel-size 1
    --pipeline-model-parallel-layout "${PIPELINE_LAYOUT}"
    # With 512 experts at EP2, the installed HybridEP intra-node dispatcher
    # rejects the 256 experts on each rank. Keep both selector arms on the
    # standard Megatron all-to-all dispatcher; no router/model changes.
    --moe-token-dispatcher-type alltoall
    --moe-grouped-gemm
    --colocate
    --no-rope-fusion
    --sft-chunked-logits
    # On the 80-GiB output stage, bound both the FP32 softmax temporary
    # and the retained vocab activations while preserving token-wise CE.
    --sft-logits-chunk-size 512
    --recompute-loss-function
    --recompute-granularity full
    --recompute-method uniform
    # Rebuild one decoder layer at a time; do not retain a second layer's
    # routed MoE tape while allocating the first layer's combine scratch.
    --recompute-num-layers 1
    --no-offload-train
    # Reclaim unused allocator segments before first synchronous DP gradient
    # reduction, so the lazy NCCL communicator can allocate on the 80-GiB GPU.
    --empty-unused-memory-level 1
)

# Relax takes an integer interval and always saves on the final step.
SAVE_INTERVAL="${SAVE_INTERVAL:-none}"
SAVE_ARGS=()
if [[ "${SAVE_INTERVAL}" != none ]]; then
    SAVE_ARGS=(--save "${SAVE_DIR}/${EXP_NAME}" --save-interval "${SAVE_INTERVAL}")
fi
EVAL_ARGS=()
if [[ "${EVAL_SIZE:-none}" != none ]]; then
    EVAL_ARGS=(--eval-size "${EVAL_SIZE}")
fi

CKPT_ARGS=(
    --hf-checkpoint "${MODEL_DIR}"
    --ref-load "${MODEL_DIR}"
    --megatron-to-hf-mode bridge
    --num-rollout "${NUM_STEPS:-8}"
    --load "${SAVE_DIR}/${EXP_NAME}"
    "${SAVE_ARGS[@]}"
    --ckpt-format torch_dist
    --max-actor-ckpt-to-keep 2
)

SFT_ARGS=(
    --loss-type sft
    --prompt-data "${DATA_DIR}"
    "${EVAL_ARGS[@]}"
    --input-key messages
    --tool-key tools
    --apply-chat-template
    --apply-chat-template-kwargs '{"add_generation_prompt":false,"preserve_thinking":true,"add_non_thinking_prefix":false}'
    --sft-invalid-multimodal-strategy error
    --sft-oversize-strategy custom
    --sft-oversize-custom-function-path relax.engine.sft.dataset.oversize.keep_within_sequence_limit
    --calculate-per-token-loss
    # The hook reads --seq-length after multimodal expansion. Do not set
    # rollout-max-context-len: this recipe deliberately permits long singletons.
    --global-batch-size "${GBS:-8}"
    --rollout-batch-size "${GBS:-8}"
    --n-samples-per-prompt 1
    --micro-batch-size 1
    --seq-length "${SEQ_LENGTH}"
    --use-dynamic-batch-size
    --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}"
    --data-pad-size-multiplier 4096
    --per-rank-fetch
    --sft-prefetch-num-workers 16
    --sft-train-data-prefetch
    --freeze-vision-model
    --freeze-vision-projection
    --freeze-params-name-list '(^|\.)per_layer_embedding\.|self_attention\.indexer\.'
    --enable-mtp-training
    --mtp-num-layers 1
    --mtp-loss-scaling-factor 0.1
    --mtp-detach-paths none
    --multimodal-keys '{"image":"images"}'
    --image-max-token-num "${IMAGE_MAX_TOKENS:-128}"
    --sft-image-preprocess-on-rank
)

# Keep the Adam/LR and FP32 update contract fixed; choose where state and the
# update run. The 80-GiB H800 fit check may require avoiding a whole GPU master
# restore window. Match the chosen mode across all benchmark arms.
case "${OPTIMIZER_OFFLOAD_MODE:-chunked}" in
    chunked)
        OPTIMIZER_OFFLOAD_ARGS=(--chunked-optimizer-state-offload --optimizer-state-offload-fraction 1.0
            --optimizer-state-offload-chunk-size-mb 512)
        ;;
    cpu)
        OPTIMIZER_OFFLOAD_ARGS=(--optimizer-cpu-offload --optimizer-offload-fraction 1.0)
        ;;
    *) echo 'OPTIMIZER_OFFLOAD_MODE must be chunked or cpu' >&2; exit 2 ;;
esac

OPTIMIZER_ARGS=(
    --optimizer adam
    --lr "${LR:-2e-5}"
    --lr-decay-style cosine
    --lr-decay-iters "${LR_DECAY_ITERS:-200}"
    --min-lr "${MIN_LR:-2e-6}"
    --lr-warmup-fraction 0.05
    --lr-warmup-init 0
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.98
    --adam-eps 1e-8
    --clip-grad 1.0
    --use-distributed-optimizer
    --use-precision-aware-optimizer
    --no-store-param-remainders
    "${OPTIMIZER_OFFLOAD_ARGS[@]}"
    --moe-aux-loss-coeff 0
    --moe-router-load-balancing-type aux_loss
)

MISC_ARGS=(
    --seed "${SEED:-9714}"
    --bf16
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --attention-softmax-in-fp32
    --attention-backend flash
    --accumulate-allreduce-grads-in-fp32
    --use-health-check
    --max-global-restart 0
    --train-env-vars "${TRAIN_ENV_JSON}"
    --moe-shared-expert-overlap
)

METRICS_ARGS=(
    --use-metrics-service
    --tb-project-name "${PROJECT_NAME}"
    --tb-experiment-name "${EXP_NAME}-${now}"
)

if [[ "${USE_CLEARML:-1}" == 1 ]]; then METRICS_ARGS+=(--use-clearml); fi
if [[ "${RESUME:-0}" == 1 ]]; then CKPT_ARGS+=(--use-checkpoint-opt_param-scheduler); fi

TRAIN_ARGS=(
    --resource '{"sft":[1,0],"actor":[1,16]}'
    --actor-num-nodes 2
    --actor-num-gpus-per-node 8
    --sft-max-in-flight-steps 4
    --sft-tq-timeout-minutes 120
    --num-data-storage-units 2
    "${MODEL_ARGS[@]}"
    "${CKPT_ARGS[@]}"
    "${SFT_ARGS[@]}"
    "${PARALLEL_ARGS[@]}"
    "${OPTIMIZER_ARGS[@]}"
    "${MISC_ARGS[@]}"
    "${METRICS_ARGS[@]}"
    "$@"
)

# Keep data semantics and parallel/optimizer settings consistent on resume.
# This is a recipe check; checkpoint loading and data-position recovery remain native.
python3 - "${SAVE_DIR}/${EXP_NAME}" "${RESUME:-0}" "${DRY_RUN:-0}" \
    "${CONTEXT_LENGTH}" "${MODEL_DIR}" "${DATA_DIR}" "${SEQ_LENGTH}" "${CP_SIZE}" \
    "${MAX_TOKENS_PER_GPU}" "${GBS:-8}" "${SEED:-9714}" \
    "${QWEN38_PLE_INPUT_HALO_TRANSPORT}" "${LR:-2e-5}" "${MIN_LR:-2e-6}" \
    "${PIPELINE_LAYOUT}" "${OPTIMIZER_OFFLOAD_MODE:-chunked}" "${PP_SIZE}" "${EP_SIZE}" "$@" <<'PYIDENTITY'
import json
import sys
from pathlib import Path

root, resume, dry = Path(sys.argv[1]), sys.argv[2] == "1", sys.argv[3] == "1"
keys = ("context_length", "model", "data", "seq_length", "cp", "packing_budget", "gbs", "seed",
        "ple_transport", "lr", "min_lr", "pipeline_layout", "optimizer_offload_mode", "pp", "ep")
identity = dict(zip(keys, sys.argv[4:19], strict=True))
extra = iter(sys.argv[19:])
identity["extra_args"] = []
for arg in extra:
    if arg == "--ckpt-step":
        next(extra)
    elif not arg.startswith("--ckpt-step=") and arg != "--use-checkpoint-opt_param-scheduler":
        identity["extra_args"].append(arg)
tracker = root / "latest_checkpointed_iteration.txt"
manifest = root / "recipe-config.json"
if resume:
    if not tracker.is_file() or not manifest.is_file():
        raise SystemExit("Resume requires the original checkpoint tracker and recipe-config.json")
    if json.loads(manifest.read_text()) != identity:
        raise SystemExit("Resume recipe differs from the saved configuration; keep context length, data and training settings unchanged")
elif tracker.exists():
    raise SystemExit("Checkpoint already exists: use RESUME=1 or choose a new EXP_NAME")
if not dry and not resume:
    root.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(identity, indent=2) + "\n")
PYIDENTITY

# Record the effective argv and worker environment for reproducibility.
if [[ "${DRY_RUN:-0}" != 1 ]]; then
    mkdir -p "${SAVE_DIR}/${EXP_NAME}/logs"
    python3 - "${SAVE_DIR}/${EXP_NAME}/logs/effective-config.json" "${RUNTIME_ENV_JSON}" "${TRAIN_ARGS[@]}" <<'PYCONFIG'
import json
import sys
from pathlib import Path

runtime = json.loads(sys.argv[2])
# Only record explicitly selected training settings, not inherited credentials.
selected = {key: value for key, value in runtime.get("env_vars", {}).items()
            if key.startswith(("RELAX_QWEN38_", "RELAX_SFT_", "QWEN38_PLE_", "QWEN38_DOSA_")) or key in
            ("MCORE_QSA_SPARSE_BACKEND", "CUDA_DEVICE_MAX_CONNECTIONS", "RELAX_HYBRID_EP_COMMIT")}
Path(sys.argv[1]).write_text(json.dumps({"argv": sys.argv[3:], "env_vars": selected}, indent=2) + "\n")
PYCONFIG
fi

if [[ "${DRY_RUN:-0}" == 1 ]]; then
    python3 - "${RUNTIME_ENV_JSON}" python3 -m relax.entrypoints.train "${TRAIN_ARGS[@]}" <<'PYDRY'
import json
import sys
print(json.dumps({"runtime_env": json.loads(sys.argv[1]), "argv": sys.argv[2:]}, indent=2))
PYDRY
    exit 0
fi

cd "${RELAX}"
mkdir -p "${SAVE_DIR}/${EXP_NAME}/logs"
ray job submit ${RAY_NO_WAIT:+--no-wait} \
    --submission-id "${SUBMISSION_ID}" \
    ${WORKING_DIR:+--working-dir "${WORKING_DIR}"} \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- python3 -m relax.entrypoints.train "${TRAIN_ARGS[@]}" \
    2>&1 | tee "${SAVE_DIR}/${EXP_NAME}/logs/${SUBMISSION_ID}.log"
