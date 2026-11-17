#!/bin/bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
TASK16_REPO="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${TASK16_REPO}"
: "${HF_CHECKPOINT:?Set HF_CHECKPOINT to the pinned local model directory}"
: "${WORKLOAD_DIR:?Set WORKLOAD_DIR to prepare_workload output}"
: "${RUN_DIR:?Set a new RUN_DIR for this independent run}"
TASK16_MODEL="${TASK16_MODEL:-dense}"
MODE="${MODE:-A}"
TASK16_PROFILE="${TASK16_PROFILE:-eight}"
OPTIMIZER_ARGS=()
CONTEXT_ARGS=()
case "${TASK16_PROFILE}:${TASK16_MODEL}" in
    eight:dense)
        source "${TASK16_REPO}/scripts/models/qwen35-27B.sh"
        DEFAULT_TP=2; DEFAULT_PP=2; DEFAULT_GPUS=8; EP_SIZE=1 ;;
    eight:moe)
        source "${TASK16_REPO}/scripts/models/qwen35-35B-A3B.sh"
        DEFAULT_TP=2; DEFAULT_PP=2; DEFAULT_GPUS=8; EP_SIZE="${EP_SIZE:-2}" ;;
    four:dense)
        source "${TASK16_REPO}/scripts/models/qwen35-4B.sh"
        FOUR_DENSE_ARGS=()
        for arg in "${MODEL_ARGS[@]}"; do
            if [[ "${arg}" != --untie-embeddings-and-output-weights ]]; then FOUR_DENSE_ARGS+=("${arg}"); fi
        done
        MODEL_ARGS=("${FOUR_DENSE_ARGS[@]}")
        DEFAULT_TP=2; DEFAULT_PP=1; DEFAULT_GPUS=4; EP_SIZE=1 ;;
    four:moe)
        source "${TASK16_REPO}/scripts/models/olmoe-7B.sh"
        DEFAULT_TP=1; DEFAULT_PP=2; DEFAULT_GPUS=4; EP_SIZE=2 ;;
    *) echo "TASK16_PROFILE must be eight/four and TASK16_MODEL dense/moe" >&2; exit 2 ;;
esac
TP_SIZE="${TP_SIZE:-${DEFAULT_TP}}"
PP_SIZE="${PP_SIZE:-${DEFAULT_PP}}"
CP_SIZE="${CP_SIZE:-1}"
NUM_GPUS="${NUM_GPUS:-${DEFAULT_GPUS}}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-64}"
for value in "${TP_SIZE}" "${PP_SIZE}" "${CP_SIZE}" "${NUM_GPUS}" "${GLOBAL_BATCH_SIZE}" "${EP_SIZE}"; do
    if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
        echo "Parallel sizes and batch size must be positive integers" >&2
        exit 2
    fi
done
if (( NUM_GPUS % (TP_SIZE * PP_SIZE * CP_SIZE) != 0 )); then
    echo "NUM_GPUS must be divisible by TP_SIZE*PP_SIZE*CP_SIZE" >&2
    exit 2
fi
DP_SIZE=$((NUM_GPUS / TP_SIZE / PP_SIZE / CP_SIZE))
if (( GLOBAL_BATCH_SIZE % DP_SIZE != 0 || (TP_SIZE * DP_SIZE * CP_SIZE) % EP_SIZE != 0 )); then
    echo "Batch size must divide across DP; EP must divide TP*DP*CP with expert TP=1" >&2
    exit 2
fi
if [[ "${TASK16_PROFILE}" == four ]]; then
    if (( NUM_GPUS != 4 || TP_SIZE != DEFAULT_TP || PP_SIZE != DEFAULT_PP || CP_SIZE != 1 )); then
        echo "The four-GPU profile requires its fixed TP/PP layout, CP1 and DP2" >&2
        exit 2
    fi
    python - "${HF_CHECKPOINT}" "${WORKLOAD_DIR}" "${TASK16_MODEL}" <<'PY_CHECK'
import json, pathlib, sys
model = json.loads((pathlib.Path(sys.argv[1]) / "config.json").read_text())
manifest = json.loads((pathlib.Path(sys.argv[2]) / "manifest.json").read_text())
expected = "qwen3_5" if sys.argv[3] == "dense" else "olmoe"
if model.get("model_type") != expected:
    raise SystemExit(f"Four-GPU {sys.argv[3]} profile requires model_type={expected}")
text = model.get("text_config", model)
shape = (text.get("hidden_size"), text.get("num_hidden_layers"))
if shape != ((2560, 32) if sys.argv[3] == "dense" else (2048, 16)):
    raise SystemExit("Model dimensions do not match the four-GPU model arguments")
if sys.argv[3] == "moe" and (text.get("num_experts"), text.get("num_experts_per_tok")) != (64, 8):
    raise SystemExit("Four-GPU OLMoE requires 64 experts with top-k 8")
if manifest["max_length"] > 4096 or max(manifest["total_lengths"]) > 4096:
    raise SystemExit("Four-GPU workloads must be frozen separately with --max-length 4096")
PY_CHECK
    OPTIMIZER_ARGS+=(--no-use-precision-aware-optimizer)
    CONTEXT_ARGS+=(--seq-length 4096 --max-position-embeddings 4096)
else
    OPTIMIZER_ARGS+=(--optimizer-cpu-offload --optimizer-offload-fraction 1.0 --use-precision-aware-optimizer)
fi
PARALLEL_ARGS=()
if (( TP_SIZE > 1 )); then PARALLEL_ARGS+=(--sequence-parallel); fi
SCHEDULE_ARGS=()
case "${MODE}" in
    A) ;;
    B) SCHEDULE_ARGS+=(--flops-microbatch-packing) ;;
    C) SCHEDULE_ARGS+=(--flops-dp-balancing) ;;
    D) SCHEDULE_ARGS+=(--flops-microbatch-packing --flops-dp-balancing) ;;
    *) echo "MODE must be A, B, C or D" >&2; exit 2 ;;
esac
RECOMPUTE_ARGS=()
case "${TASK16_RECOMPUTE:-full}" in
    full) RECOMPUTE_ARGS+=(--recompute-granularity full --recompute-method uniform --recompute-num-layers 1) ;;
    none) ;;
    *) echo "TASK16_RECOMPUTE must be full or none" >&2; exit 2 ;;
esac

TRAIN_ARGS=(
    "${MODEL_ARGS[@]}"
    --resource "{\"sft\": [1, 0], \"actor\": [1, ${NUM_GPUS}]}" --num-data-storage-units 1
    --hf-checkpoint "${HF_CHECKPOINT}" --megatron-to-hf-mode bridge
    --loss-type sft --prompt-data "${WORKLOAD_DIR}/train.jsonl"
    --input-key problem --label-key generated_solution
    --num-rollout "${NUM_ROLLOUT:-30}" --rollout-batch-size "${GLOBAL_BATCH_SIZE}"
    --global-batch-size "${GLOBAL_BATCH_SIZE}" --n-samples-per-prompt 1
    --micro-batch-size 1 --use-dynamic-batch-size --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU:-8192}"
    --balance-data --qkv-format thd --sft-oversize-strategy keep --sft-packing-strategy balanced
    --seed 1605 --rollout-seed 1605 --max-staleness 0
    --tensor-model-parallel-size "${TP_SIZE}" --pipeline-model-parallel-size "${PP_SIZE}"
    --context-parallel-size "${CP_SIZE}" --expert-model-parallel-size "${EP_SIZE}"
    --expert-tensor-parallel-size 1 "${PARALLEL_ARGS[@]}"
    --use-distributed-optimizer --accumulate-allreduce-grads-in-fp32
    "${RECOMPUTE_ARGS[@]}"
    --calculate-per-token-loss --sft-chunked-logits --sft-logits-chunk-size 1024
    --optimizer adam --lr 1e-5 --lr-decay-style constant --weight-decay 0.1
    "${OPTIMIZER_ARGS[@]}" "${CONTEXT_ARGS[@]}"
    --adam-beta1 0.9 --adam-beta2 0.98 --clip-grad 1.0
    --attention-dropout 0 --hidden-dropout 0 --attention-softmax-in-fp32
    --attention-backend flash --no-rope-fusion --cross-entropy-loss-fusion
    --mtp-num-layers 0
    "${SCHEDULE_ARGS[@]}"
)
if [[ "${TASK16_MODEL}" == moe ]]; then
    TRAIN_ARGS+=(--moe-router-load-balancing-type none --moe-aux-loss-coeff 0 --moe-z-loss-coeff 0)
fi
RECORD_ARGS=(--workload "${WORKLOAD_DIR}" --checkpoint "${HF_CHECKPOINT}" --run-dir "${RUN_DIR}" --mode "${MODE}")
if [[ "${TASK16_CAPTURE:-0}" == 1 ]]; then RECORD_ARGS+=(--capture); fi
if [[ "${TASK16_DRY_RUN:-0}" == 1 ]]; then RECORD_ARGS+=(--dry-run); fi
python -m examples.flops_balancing.record_run "${RECORD_ARGS[@]}" -- python3 -m relax.entrypoints.train "${TRAIN_ARGS[@]}"
if [[ "${TASK16_DRY_RUN:-0}" == 1 ]]; then exit 0; fi
if [[ "${RELAX_ENTRYPOINT_MODE:-}" != ray-job ]]; then
    echo "Launch this script through scripts/entrypoint/ray-job.sh" >&2
    exit 2
fi
: "${RAY_ADDRESS:?Set the actual Ray dashboard address}"
export TASK16_CAPTURE_PATH=""
if [[ "${TASK16_CAPTURE:-0}" == 1 ]]; then export TASK16_CAPTURE_PATH="${RUN_DIR}/capture"; fi
RUNTIME_ENV_JSON="$(python - <<'PY'
import json, os
env = json.loads(os.environ['RUNTIME_ENV_JSON'])
env.setdefault('env_vars', {})['RELAX_FLOPS_CAPTURE_DIR'] = os.environ['TASK16_CAPTURE_PATH']
for name in ('NVTE_ALLOW_NONDETERMINISTIC_ALGO', 'CUBLAS_WORKSPACE_CONFIG', 'NCCL_ALGO'):
    if name in os.environ:
        env['env_vars'][name] = os.environ[name]
print(json.dumps(env))
PY
)"
ray job submit --no-wait --address="${RAY_ADDRESS}" \
    ${WORKING_DIR:+--working-dir "${WORKING_DIR}"} --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- python3 -m relax.entrypoints.train "${TRAIN_ARGS[@]}" 2>&1 | tee "${RUN_DIR}/submit.log"
