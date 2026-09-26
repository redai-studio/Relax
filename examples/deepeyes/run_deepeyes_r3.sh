#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.

set -ex
set -o pipefail

###############################################################################
#                                 ENVIRONMENT                                 #
###############################################################################

TIMESTAMP=$(date "+%Y-%m-%d-%H:%M:%S")

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
# Auto-source local environment when not launched via an external entrypoint
if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    source "${SCRIPT_DIR}/../../scripts/entrypoint/local.sh"
fi
source "${MODEL_CONFIG_DIR}/qwen3-vl-30B-A3B.sh"

###############################################################################
#                                    DIRS                                     #
###############################################################################

PROJECT_NAME="${PROJECT_NAME:=Relax/dev/deepeyes_r3}"
EXP_NAME="qwen3vl-deepeyes-r3-${TIMESTAMP}"

# Require MODEL_DIR, DATA_DIR, SAVE_DIR from environment or set defaults
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
    --hf-checkpoint ${MODEL_DIR}/Qwen3-VL-30B-A3B-Thinking
    --ref-load ${MODEL_DIR}/Qwen3-VL-30B-A3B-Thinking
    --save ${SAVE_DIR}/Qwen3-VL-30B-A3B-Thinking-Checkpoint
    --megatron-to-hf-mode bridge
    --save-interval 100
    --max-actor-ckpt-to-keep 3
    # --load ${SAVE_DIR}/Qwen3-VL-30B-A3B-Thinking-Checkpoint
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
    --apply-chat-template
    --custom-generate-function-path examples.deepeyes.rollout.generate
    --custom-rm-path examples.deepeyes.reward_deepeyes.reward_func
    --custom-config-path examples/deepeyes/deepeyes_config.yaml
    --num-rollout ${NUM_ROLLOUT}
    --rollout-batch-size 32
    --micro-batch-size 1
    --n-samples-per-prompt 8
    --rollout-max-response-len 2048
    --rollout-max-prompt-len 2048
    --rollout-temperature 1
    --global-batch-size 256
    --use-fault-tolerance
    --rollout-shuffle
)
###############################################################################
#                             ROUTING REPLAY CONFIG                            #
###############################################################################

ROUTING_REPLAY_ARGS=(
    --use-rollout-routing-replay
    --use-slime-router
)

###############################################################################
#                                EVAL CONFIG                                  #
###############################################################################

EVAL_ARGS=(
    --skip-eval-before-train
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
    --no-rope-fusion
)

###############################################################################
#                               SGLANG CONFIG                                 #
###############################################################################

SGLANG_ARGS=(
    --sglang-mem-fraction-static 0.8
    --sglang-external-model-package examples.deepeyes.sglang_patch
)

###############################################################################
#                               LOGGING CONFIG                                #
###############################################################################

LOG_ARGS=(
    --use-clearml
    --use-metrics-service
    --tb-project-name ${PROJECT_NAME}
    --tb-experiment-name ${EXP_NAME}
    # --dump-details dump_details_8k_0204
    # --use-wandb
    # --wandb-project slime-dev
    # --wandb-group qwen3-4B-test
    # --wandb-key ${WANDB_KEY}
)

###############################################################################
#                              MEGATRON CONFIG                                #
###############################################################################

MEGATRON_ARGS=(
    --tensor-model-parallel-size 4
    --sequence-parallel
    --pipeline-model-parallel-size 1
    --context-parallel-size 1
    --expert-model-parallel-size 8
    --expert-tensor-parallel-size 1
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1
    --max-tokens-per-gpu 8192
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --accumulate-allreduce-grads-in-fp32
    --attention-softmax-in-fp32
    --attention-backend flash
    --use-dynamic-batch-size
)

###############################################################################
#                              RESOURCE CONFIG                                #
###############################################################################

RAY_RESOURCE_ARGS=(
    --rollout-num-gpus-per-engine 1
    --resource '{"actor": [1, 8], "rollout": [1, 8]}'
    --max-staleness 0
    --num-data-storage-units 1
    --colocate
)

###############################################################################
#                                 LAUNCH JOB                                  #
###############################################################################

mkdir -p logs

ray job submit ${RAY_NO_WAIT:+--no-wait} --address="http://127.0.0.1:8265" \
    -- python3 relax/entrypoints/train.py \
    "${RAY_RESOURCE_ARGS[@]}" \
    "${MODEL_ARGS[@]}" \
    "${CKPT_ARGS[@]}" \
    "${ROLLOUT_ARGS[@]}" \
    "${ROUTING_REPLAY_ARGS[@]}" \
    "${GRPO_ARGS[@]}" \
    "${OPTIMIZER_ARGS[@]}" \
    "${SGLANG_ARGS[@]}" \
    "${LOG_ARGS[@]}" \
    "${MEGATRON_ARGS[@]}" \
    "${EVAL_ARGS[@]}" \
    2>&1 | tee logs/${EXP_NAME}.log
