#!/bin/bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Qwen3.5-4B full SFT comparison, complete multimodal tool trajectories, 8 GPUs.
# MODEL_DIR=/models DATA_DIR=/data SAVE_DIR=/checkpoints bash "$0" relax
# Modes: relax | evaluate <HF-checkpoint> <base|relax> | benchmark <simplevqa|vdr> <base|relax> | compare

set -eo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
RELAX_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
MODE="${1:-relax}"
PROJECT_NAME="${PROJECT_NAME:-Relax/sft/opensearch-vl}"
EXP_NAME="${EXP_NAME:-qwen3.5-4b-opensearch-vl-32k-8xgpu}"
now=$(date "+%Y%m%d-%H%M%S")
: "${MODEL_DIR:?Set MODEL_DIR to the initial model root}"
: "${DATA_DIR:?Set DATA_DIR to the prepared dataset root}"
: "${SAVE_DIR:?Set SAVE_DIR to a persistent checkpoint root}"
export RAY_NO_WAIT="${RAY_NO_WAIT-}"
HF_CHECKPOINT="${HF_CHECKPOINT:-${MODEL_DIR}/Qwen3.5-4B}"
NUM_EPOCHS="${NUM_EPOCHS:-2}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-32}"
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-32768}"
mkdir -p "${SCRIPT_DIR}/log" "${SAVE_DIR}/${EXP_NAME}"
cd "${RELAX_ROOT}"

case "${MODE}" in
    benchmark)
        : "${2:?Supply simplevqa or vdr}"
        : "${3:?Supply base or relax}"
        : "${BENCHMARK_DATA_ROOT:?Set the downloaded benchmark dataset root}"
        : "${BENCHMARK_OVERLAP_INDEX:?Set the SFT train overlap index}"
        case "$2" in
            simplevqa) benchmark_data="${BENCHMARK_DATA_ROOT}/simplevqa/test.parquet" ;;
            vdr) benchmark_data="${BENCHMARK_DATA_ROOT}/vdr/data/testmini500_v1.parquet" ;;
            vstar)
                python3 "${SCRIPT_DIR}/../tools/vstar_benchmark.py" \
                    --data "${BENCHMARK_DATA_ROOT}/vstar" --processor "${HF_CHECKPOINT}" \
                    --provenance "${AGENT_PROVENANCE_FILE:?Set verified model provenance}" --label "$3" \
                    --overlap-index "${BENCHMARK_OVERLAP_INDEX}" \
                    --output "${SAVE_DIR}/${EXP_NAME}/benchmark/$3/vstar" \
                    --workers "${BENCHMARK_WORKERS:-16}" --max-cases "${BENCHMARK_MAX_CASES:-0}"
                exit
                ;;
            *) echo "Benchmark must be simplevqa, vstar or vdr" >&2; exit 2 ;;
        esac
        benchmark_args=()
        if [[ "${BENCHMARK_MAX_CASES:-0}" != 0 ]]; then benchmark_args+=(--max-cases "${BENCHMARK_MAX_CASES}"); fi
        if [[ "${BENCHMARK_ALLOW_MISSING_TOOLS:-0}" == 1 ]]; then benchmark_args+=(--allow-missing-tools); fi
        python3 "${SCRIPT_DIR}/../tools/benchmark.py" \
            --workflow "${BENCHMARK_WORKFLOW:-direct}" --workers "${BENCHMARK_WORKERS:-1}" \
            --judge-qps "${JUDGE_QPS:-1}" \
            --dataset "$2" --label "$3" --data "${benchmark_data}" --processor "${HF_CHECKPOINT}" \
            --output "${SAVE_DIR}/${EXP_NAME}/benchmark/$3/$2" \
            --tool-cache "${BENCHMARK_CACHE_DIR:-${SAVE_DIR}/${EXP_NAME}/benchmark/tool-cache}" \
            --overlap-index "${BENCHMARK_OVERLAP_INDEX}" "${benchmark_args[@]}"
        exit
        ;;
    evaluate)
        : "${2:?Supply one explicit HF checkpoint directory}"
        : "${3:?Supply base or relax as the evaluation label}"
        case "$3" in base|relax) ;; *) echo "Evaluation label must be base or relax" >&2; exit 2 ;; esac
        python3 "${SCRIPT_DIR}/../tools/evaluate.py" \
            --model "$2" --processor "${HF_CHECKPOINT}" \
            --data "${DATA_DIR}/opensearch_vl/32k/test_steps.jsonl" \
            --output "${SAVE_DIR}/${EXP_NAME}/eval/$3" \
            2>&1 | tee "${SCRIPT_DIR}/log/${EXP_NAME}-eval-$3-${now}.log"
        exit
        ;;
    compare)
        if [[ -d "${SAVE_DIR}/${EXP_NAME}/benchmark" ]]; then
            python3 "${SCRIPT_DIR}/../tools/compare_benchmark.py" \
                --root "${SAVE_DIR}/${EXP_NAME}/benchmark" \
                --output "${SAVE_DIR}/${EXP_NAME}/benchmark-comparison.json"
            exit
        fi
        python3 "${SCRIPT_DIR}/../tools/compare.py" \
            --base "${SAVE_DIR}/${EXP_NAME}/eval/base/summary.json" \
            --relax "${SAVE_DIR}/${EXP_NAME}/eval/relax/summary.json" \
            --output "${SAVE_DIR}/${EXP_NAME}/comparison.json"
        exit
        ;;
    relax) ;;
    *) echo "Unknown mode: ${MODE}" >&2; exit 2 ;;
esac

# Do not source the training entrypoint during static checks.
if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    source "${RELAX_ROOT}/scripts/entrypoint/local.sh"
fi
source "${RELAX_ROOT}/scripts/models/${MODEL_CONFIG:-qwen35-4B}.sh"
# Match embedding sharing to the downloaded model (4B is tied; 27B is untied).
TIE_EMBEDDINGS=$(python3 - "${HF_CHECKPOINT}/config.json" <<'PY'
import json
import sys
config = json.load(open(sys.argv[1]))
print(int(config.get("tie_word_embeddings", config.get("text_config", {}).get("tie_word_embeddings", False))))
PY
)
TIED_MODEL_ARGS=()
for argument in "${MODEL_ARGS[@]}"; do
    if [[ "${TIE_EMBEDDINGS}" != 1 || "${argument}" != --untie-embeddings-and-output-weights ]]; then
        TIED_MODEL_ARGS+=("${argument}")
    fi
done
MODEL_ARGS=("${TIED_MODEL_ARGS[@]}")
python3 "${SCRIPT_DIR}/../tools/audit_data.py" \
    --model "${HF_CHECKPOINT}" --data-dir "${DATA_DIR}/opensearch_vl/32k" \
    --max-samples-per-split 8 \
    --output "${SAVE_DIR}/${EXP_NAME}/relax-audit.json"

CKPT_ARGS=(
    --hf-checkpoint "${HF_CHECKPOINT}"
    --ref-load "${HF_CHECKPOINT}"
    --megatron-to-hf-mode bridge
    --num-epoch "${NUM_EPOCHS}"
)
if [[ -n "${TRAIN_STEPS:-}" ]]; then CKPT_ARGS+=(--num-rollout "${TRAIN_STEPS}"); fi
# Preserve a resumable recovery point and retain only the latest checkpoint.
if [[ "${BENCHMARK_ONLY:-0}" != 1 ]]; then
    checkpoint_load="${RESUME_CHECKPOINT:-${SAVE_DIR}/${EXP_NAME}/relax}"
    CKPT_ARGS+=(
        --load "${checkpoint_load}"
        --save "${SAVE_DIR}/${EXP_NAME}/relax"
        --save-hf "${SAVE_DIR}/${EXP_NAME}/final-hf-{rollout_id}"
        --save-hf-post-hook-path examples.opensearch_vl.tools.checkpoint_retention.keep_latest_hf
        --save-interval "${SAVE_INTERVAL:-200}"
        --max-actor-ckpt-to-keep 1
    )
    if [[ -n "${RESUME_CHECKPOINT:-}" || -f "${checkpoint_load}/latest_checkpointed_iteration.txt" ]]; then
        resume_step=$(python3 - "${checkpoint_load}" <<'PY'
import sys

from relax.utils.utils import checkpoint_resume_step

print(checkpoint_resume_step(sys.argv[1]))
PY
)
        CKPT_ARGS+=(--start-rollout-id "${resume_step}")
    fi
else
    CKPT_ARGS+=(--num-rollout "${BENCHMARK_STEPS:-8}")
fi
SFT_ARGS=(
    --loss-type sft
    --prompt-data "${DATA_DIR}/opensearch_vl/32k/train.jsonl"
    --input-key messages
    --tool-key tools
    --multimodal-keys '{"image":"images"}'
    --apply-chat-template-kwargs '{"preserve_thinking":true}'
    --image-max-token-num 256
    --image-min-token-num 1
    --global-batch-size "${GLOBAL_BATCH_SIZE}"
    --use-dynamic-batch-size
    --seq-length "${MAX_TOKENS_PER_GPU}"
    --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}"
    --sft-oversize-strategy skip
    --balance-data
    --per-rank-fetch
    --sft-prefetch-num-workers "${SFT_PREFETCH_WORKERS:-8}"
    --sft-prefetch-buffer-size 128
    --sft-max-in-flight-steps 4
    --sft-train-data-prefetch
)
EVAL_ARGS=(
    --eval-prompt-data opensearch_val "${DATA_DIR}/opensearch_vl/32k/val.jsonl"
    --eval-interval "${EVAL_INTERVAL:-100}"
)
if [[ "${BENCHMARK_ONLY:-0}" == 1 ]]; then EVAL_ARGS=(); fi
PERF_ARGS=(
    --tensor-model-parallel-size "${TP_SIZE:-2}"
    --pipeline-model-parallel-size "${PP_SIZE:-1}"
    --context-parallel-size "${CP_SIZE:-1}"
    --expert-model-parallel-size 1
    --expert-tensor-parallel-size 1
    --calculate-per-token-loss
    --no-rope-fusion
    --colocate
    --cross-entropy-loss-fusion
    --sft-chunked-logits
    --sft-logits-chunk-size "${LOGITS_CHUNK_SIZE:-1024}"
    --recompute-loss-function
    --no-offload-train
)
if [[ "${TP_SIZE:-2}" -gt 1 ]]; then PERF_ARGS+=(--sequence-parallel); fi
if [[ "${RECOMPUTE:-full}" == full ]]; then
    PERF_ARGS+=(
        --recompute-granularity full
        --recompute-method uniform
        --recompute-num-layers 1
    )
fi
OPTIMIZER_ARGS=(
    --optimizer adam
    --lr 2e-5
    --min-lr 0
    --lr-decay-style cosine
    --lr-warmup-fraction 0.1
    --weight-decay 0
    --adam-beta1 0.9
    --adam-beta2 0.999
    --adam-eps 1e-8
    --clip-grad 1.0
    --use-distributed-optimizer
)
if [[ "${OPTIMIZER_OFFLOAD_FRACTION:-0}" != 0 ]]; then
    OPTIMIZER_ARGS+=(
        --optimizer-cpu-offload
        --optimizer-offload-fraction "${OPTIMIZER_OFFLOAD_FRACTION}"
        --overlap-cpu-optimizer-d2h-h2d
        --use-precision-aware-optimizer
    )
fi
WANDB_ARGS=(
    --use-clearml
    --use-metrics-service
    --tb-project-name "${PROJECT_NAME}"
    --tb-experiment-name "${EXP_NAME}-relax-${now}"
)
if [[ "${ENABLE_SWANLAB:-0}" == 1 ]]; then
    WANDB_ARGS+=(--use-tensorboard)
    export SWANLAB_PROJ_NAME="${SWANLAB_PROJ_NAME:-Relax-sft-opensearch-vl}"
    export SWANLAB_EXP_NAME="${EXP_NAME}-relax-${now}"
    export RUNTIME_ENV_JSON="$(python3 - <<'PY'
import json
import os

runtime = json.loads(os.environ['RUNTIME_ENV_JSON'])
runtime['worker_process_setup_hook'] = 'examples.opensearch_vl.tools.swanlab_hook.install'
runtime['env_vars'].update({key: value for key, value in os.environ.items() if key.startswith('SWANLAB_')})
print(json.dumps(runtime))
PY
)"
fi
MISC_ARGS=(
    --seed 42
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --accumulate-allreduce-grads-in-fp32
    --attention-softmax-in-fp32
    --attention-backend auto
    --use-health-check
)

ray job submit ${RAY_NO_WAIT:+--no-wait} --address="${RAY_JOBS_ADDRESS:-http://127.0.0.1:8265}" \
    --submission-id "${SUBMISSION_ID:-${EXP_NAME}-relax-${now}}" \
    ${WORKING_DIR:+--working-dir "${WORKING_DIR}"} \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- python3 -m relax.entrypoints.train \
    --resource '{"sft":[1,0],"actor":[1,8]}' \
    --num-data-storage-units 1 \
    "${MODEL_ARGS[@]}" "${CKPT_ARGS[@]}" "${SFT_ARGS[@]}" "${EVAL_ARGS[@]}" \
    "${PERF_ARGS[@]}" "${OPTIMIZER_ARGS[@]}" "${WANDB_ARGS[@]}" "${MISC_ARGS[@]}" \
    2>&1 | tee "${SCRIPT_DIR}/log/${EXP_NAME}-relax-${now}.log"
