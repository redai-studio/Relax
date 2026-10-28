#!/usr/bin/env bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.

# Run via scripts/entrypoint/ray-job.sh on an isolated two-node Ray cluster.
# This smoke objective verifies training execution, not model quality.
set -euo pipefail
: "${MODEL_DIR:?Set MODEL_DIR to the shared Qwen3-0.6B checkpoint}"
: "${INFERENCE_TRAIN_OUTPUT:?Set a new shared output directory}"
: "${RAY_DASHBOARD:?Set the current Ray dashboard HTTP URL}"
: "${RUNTIME_ENV_JSON:?Launch through scripts/entrypoint/ray-job.sh}"
source "${MODEL_CONFIG_DIR}/qwen3-0.6B.sh"
mkdir -p "$INFERENCE_TRAIN_OUTPUT"
PROMPT_DATA="$INFERENCE_TRAIN_OUTPUT/prompts.jsonl"
cat > "$PROMPT_DATA" <<'JSONL'
{"prompt":"Continue the sequence: one, two, three,", "label":"four"}
{"prompt":"What is two plus three? Explain briefly.", "label":"5"}
{"prompt":"Write a sentence about the moon.", "label":"moon"}
{"prompt":"Name a primary color and explain your choice.", "label":"red"}
JSONL

# torch_memory_saver (TMS) is the default colocated offload path.  Keep the
# application-level selective implementation available for backends that do
# not support TMS, but make it explicit so a validation run cannot silently
# claim default-TMS coverage.
OFFLOAD_ARGS=(--offload-train --offload-rollout)
if [[ "${INFERENCE_SELECTIVE_OFFLOAD:-0}" == 1 ]]; then
  OFFLOAD_ARGS+=(--selective-offload)
fi

TRAIN_ARGS=(
  --train-backend megatron
  --resource '{"actor":[1,2],"rollout":[1,2],"teacher":[1,2],"genrm":[1,2]}'
  --num-gpus-per-node 1 --actor-num-nodes 2 --actor-num-gpus-per-node 1
  --colocate "${OFFLOAD_ARGS[@]}"
  --max-staleness 0 --num-data-storage-units 1
  --hf-checkpoint "$MODEL_DIR" --load "$MODEL_DIR" --megatron-to-hf-mode bridge
  --save "$INFERENCE_TRAIN_OUTPUT/actor" --save-interval 1
  --save-hf "$INFERENCE_TRAIN_OUTPUT/hf/{rollout_id}"
  --prompt-data "$PROMPT_DATA" --input-key prompt --label-key label
  --apply-chat-template --apply-chat-template-kwargs '{"enable_thinking":false}'
  --num-rollout 2 --rollout-batch-size 2 --n-samples-per-prompt 2
  --num-steps-per-rollout 1 --global-batch-size 4 --micro-batch-size 1
  --rollout-max-response-len 32 --rollout-max-context-len 1024 --rollout-temperature 1
  --skip-eval-before-train
  --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1
  --context-parallel-size 1 --expert-model-parallel-size 1 --expert-tensor-parallel-size 1
  --seq-length 1024 --use-dynamic-batch-size --max-tokens-per-gpu 1024
  --log-probs-max-tokens-per-gpu 1024 --calculate-per-token-loss
  --advantage-estimator grpo --disable-rewards-normalization --use-rollout-logprobs
  --kl-coef 0 --entropy-coef 0 --eps-clip 0.2 --eps-clip-high 0.2
  --optimizer adam --lr 1e-5 --lr-decay-style constant --weight-decay 0
  --adam-beta1 0.9 --adam-beta2 0.999
  --use-opd --opd-type sglang --opd-token-selection student_sampled --opd-teacher-defer
  --teacher-hf-checkpoint "$MODEL_DIR" --teacher-num-gpus-per-engine 2
  --teacher-sglang-attention-backend triton --teacher-sglang-disable-cuda-graph
  --teacher-sglang-mem-fraction-static 0.5 --teacher-sglang-context-length 1024
  --teacher-sglang-max-total-tokens 2048 --opd-teacher-timeout-s 180
  --opd-kl-coef 0 --opd-loss-coef 0.1 --opd-kl-type low_var_kl
  --genrm-model-path "$MODEL_DIR" --genrm-num-gpus-per-engine 2
  --genrm-engine-config '{"attention_backend":"triton","disable_cuda_graph":true,"mem_fraction_static":0.5,"context_length":1024,"max_total_tokens":2048}'
  --defer-reward-to-post-process --rm-type dummy
  --custom-rm-path scripts.training.hpc.unified_inference_reward.score
  --reward-key score --reward-max-concurrency 2
  --rollout-num-gpus-per-engine 2 --sglang-load-format dummy
  --sglang-enable-weights-cpu-backup --sglang-attention-backend triton --sglang-disable-cuda-graph
  --sglang-mem-fraction-static 0.5 --sglang-context-length 1024
  --sglang-max-total-tokens 2048 --sglang-max-running-requests 4
  --attention-dropout 0 --hidden-dropout 0 --attention-backend fused
  --no-gradient-accumulation-fusion --no-rope-fusion
  --train-env-vars '{"NVTE_FLASH_ATTN":"0","NVTE_FUSED_ATTN":"1"}'
  --tb-project-name "$INFERENCE_TRAIN_OUTPUT/tensorboard" --tb-experiment-name unified-inference-defer
  --dump-details "$INFERENCE_TRAIN_OUTPUT/details"
)

if [[ "${INFERENCE_VALIDATE_ARGS_ONLY:-0}" == 1 ]]; then
  python -c 'from relax.utils.arguments import parse_args; parse_args(); print("TRAINING_ARGUMENTS_VALID")' \
    "${MODEL_ARGS[@]}" "${TRAIN_ARGS[@]}" "$@"
else
  RAY_NO_WAIT=1 ray job submit --no-wait --address="$RAY_DASHBOARD" \
    --runtime-env-json="$RUNTIME_ENV_JSON" \
    -- python -m relax.entrypoints.train "${MODEL_ARGS[@]}" "${TRAIN_ARGS[@]}" "$@"
fi
