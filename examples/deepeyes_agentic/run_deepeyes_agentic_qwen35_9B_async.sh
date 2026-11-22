#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Qwen3.5-9B 8xGPU single-node fully-async DeepEyes agentic training script.
#
# Resource layout (8 GPUs, fully-async):
#   actor:     4 GPUs (TP=4)
#   rollout:   2 GPUs (1 engine × 2 GPUs)
#   reference: 1 GPU  (TP=1, weight-only)
#   actor_fwd: 1 GPU

set -ex
set -o pipefail

###############################################################################
#                                 ENVIRONMENT                                 #
###############################################################################

TIMESTAMP=$(date "+%Y-%m-%d-%H:%M:%S")

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    source "${SCRIPT_DIR}/../../scripts/entrypoint/local.sh"
fi
source "${MODEL_CONFIG_DIR}/qwen35-9B.sh"

###############################################################################
#                                    DIRS                                     #
###############################################################################

PROJECT_NAME="${PROJECT_NAME:=Relax/dev/deepeyes}"
EXP_NAME="qwen35-9B-deepeyes-agentic-async-${TIMESTAMP}"

if [ -z "${MODEL_DIR:-}" ] || [ -z "${DATA_DIR:-}" ] || [ -z "${SAVE_DIR:-}" ]; then
    echo "ERROR: MODEL_DIR, DATA_DIR, and SAVE_DIR must be set."
    echo "Example: MODEL_DIR=/path/to/models DATA_DIR=/path/to/data SAVE_DIR=/path/to/save bash $0"
    exit 1
fi
mkdir -p ${SAVE_DIR}

###############################################################################
#                              JUDGE MODEL API                                #
###############################################################################

source "${SCRIPT_DIR}/sglang_judge_service.sh"

###############################################################################
#                                  MODEL CONFIG                               #
###############################################################################

CKPT_ARGS=(
    --hf-checkpoint ${MODEL_DIR}/Qwen3.5-9B
    --ref-load ${MODEL_DIR}/Qwen3.5-9B
    --save ${SAVE_DIR}/Qwen3.5-9B-DeepEyes-Checkpoint
    --megatron-to-hf-mode bridge
    --save-interval 100
    --max-actor-ckpt-to-keep 1
)

###############################################################################
#                                  DATASETS                                   #
###############################################################################

TRAIN_FILES=(
    "'${DATA_DIR}/deepeyes-v1/data_0.1.2_visual_toolbox_v2.parquet@[0:5000]'"
    "'${DATA_DIR}/deepeyes-v1/data_v0.8_visual_toolbox_v2.parquet@[0:5000]'"
)
TEST_FILES=("${DATA_DIR}/deepeyes-v1/data_thinklite_reasoning_acc.parquet@[0:256]")
PROMPT_SET="[$(IFS=,; echo "${TRAIN_FILES[*]}")]"

###############################################################################
#                               ROLLOUT CONFIG                                #
###############################################################################

NUM_ROLLOUT="${NUM_ROLLOUT:=2000}"

ROLLOUT_ARGS=(
    --prompt-data "${PROMPT_SET}"
    --input-key prompt
    --label-key reward_model
    --multimodal-keys '{"image":"images"}'
    --reward-key score
    --metadata-key extra_info
    --custom-rm-path examples.deepeyes_agentic.reward_deepeyes.reward_func
    --use-agentic-rollout
    --agent-command "bash ${SCRIPT_DIR}/run_agent_app.sh"
    --agent-cwd "${SCRIPT_DIR}"
    --num-rollout ${NUM_ROLLOUT}
    --rollout-batch-size 32
    --n-samples-per-prompt 8
    --rollout-max-response-len 2048
    --rollout-max-prompt-len 2048
    --rollout-temperature 1
    --global-batch-size 256
    --use-fault-tolerance
    --rollout-shuffle
    --use-streaming-dataset
)

###############################################################################
#                                EVAL CONFIG                                  #
###############################################################################

EVAL_ARGS=(
    --eval-interval 100
    --eval-prompt-data vstar ${TEST_FILES}
    --n-samples-per-eval-prompt 8
    --eval-max-response-len 2048
    --eval-top-p 0.7
)

###############################################################################
#                              ALGORITHM CONFIG                               #
###############################################################################

GRPO_ARGS=(
    --advantage-estimator grpo
    --kl-loss-coef 0.00
    --kl-loss-type low_var_kl
    --entropy-coef 0.00
    --eps-clip 0.2
    --eps-clip-high 0.28
    --eps-clip-c 3
    --use-tis
)

###############################################################################
#                              OPTIMIZER CONFIG                               #
###############################################################################

OPTIMIZER_ARGS=(
    --optimizer adam
    --lr 1e-6
    --lr-decay-style constant
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.98
    --optimizer-cpu-offload
    --overlap-cpu-optimizer-d2h-h2d
    --use-precision-aware-optimizer
)

###############################################################################
#                               SGLANG CONFIG                                 #
###############################################################################

SGLANG_ARGS=(
    --rollout-num-gpus-per-engine 2
    --sglang-mem-fraction-static 0.6
)

###############################################################################
#                               LOGGING CONFIG                                #
###############################################################################

LOG_ARGS=(
    --use-clearml
    --use-metrics-service
    --tb-project-name ${PROJECT_NAME}
    --tb-experiment-name ${EXP_NAME}
)

###############################################################################
#                              MEGATRON CONFIG                                #
###############################################################################

MEGATRON_ARGS=(
    --tensor-model-parallel-size 4
    --sequence-parallel
    --pipeline-model-parallel-size 1
    --context-parallel-size 1
    --expert-model-parallel-size 1
    --expert-tensor-parallel-size 1
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1
    --use-dynamic-batch-size
    --max-tokens-per-gpu 9216
    --no-rope-fusion
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --accumulate-allreduce-grads-in-fp32
    --attention-softmax-in-fp32
    --attention-backend flash
)

###############################################################################
#                              RESOURCE CONFIG                                #
###############################################################################

RAY_RESOURCE_ARGS=(
    --resource '{"actor": [1, 4], "rollout": [1, 2], "reference": [1, 1], "actor_fwd": [1, 1], "advantages": [1, 0]}'
    --max-staleness 2
    --num-data-storage-units 1
    --num-iters-per-train-update 8
    --ref-actor-config '{"tensor_model_parallel_size": 1, "max_tokens_per_gpu": 16384, "sequence_parallel": false, "only_load_weight": true}'
    --fully-async
    --use-health-check
)

###############################################################################
#                                 LAUNCH JOB                                  #
###############################################################################

mkdir -p logs

ray job submit ${RAY_NO_WAIT:+--no-wait} --address="http://127.0.0.1:8265" \
    -- python3 -m relax.entrypoints.train \
    "${RAY_RESOURCE_ARGS[@]}" \
    "${MODEL_ARGS[@]}" \
    "${CKPT_ARGS[@]}" \
    "${ROLLOUT_ARGS[@]}" \
    "${GRPO_ARGS[@]}" \
    "${OPTIMIZER_ARGS[@]}" \
    "${SGLANG_ARGS[@]}" \
    "${LOG_ARGS[@]}" \
    "${MEGATRON_ARGS[@]}" \
    "${EVAL_ARGS[@]}" \
    2>&1 | tee logs/${EXP_NAME}.log
