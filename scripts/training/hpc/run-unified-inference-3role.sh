#!/usr/bin/env bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.

# Full three-role validation for RFC #71.  The same command exercises either
# independent (decoupled) placement groups or a disjoint shared-actor split.
# Teacher and GenRM are static: only Rollout receives dynamic weight updates.
#
# Required: MODEL_DIR, INFERENCE_TRAIN_OUTPUT, RAY_DASHBOARD, RUNTIME_ENV_JSON,
# and MODEL_CONFIG_DIR.  Set INFERENCE_LAYOUT=split for the shared-actor case.
# Split requires actor GPU capacity to equal rollout + teacher + GenRM GPU
# budgets; decoupled keeps actor capacity independent.
# Set INFERENCE_WAIT_FOR_JOB=1 when a caller must wait for Ray training to
# finish before running follow-up validation on the same cluster.

set -euo pipefail

: "${MODEL_DIR:?Set MODEL_DIR to the shared Qwen3-0.6B checkpoint}"
: "${INFERENCE_TRAIN_OUTPUT:?Set a new shared output directory}"
: "${RAY_DASHBOARD:?Set the current Ray dashboard HTTP URL}"
: "${RUNTIME_ENV_JSON:?Launch through scripts/entrypoint/ray-job.sh}"

source "${MODEL_CONFIG_DIR}/qwen3-0.6B.sh"

LAYOUT="${INFERENCE_LAYOUT:-decoupled}"
GPU_NODE_WIDTH="${INFERENCE_GPUS_PER_NODE:-8}"
ROLLOUT_GPUS="${INFERENCE_ROLLOUT_GPUS:-2}"
TEACHER_GPUS="${INFERENCE_TEACHER_GPUS:-2}"
GENRM_GPUS="${INFERENCE_GENRM_GPUS:-2}"
ACTOR_NODES="${INFERENCE_ACTOR_NODES:-1}"
ACTOR_GPUS_PER_NODE="${INFERENCE_ACTOR_GPUS_PER_NODE:-2}"
ACTOR_GPUS=$((ACTOR_NODES * ACTOR_GPUS_PER_NODE))
ROLLOUT_BATCH_SIZE="${INFERENCE_ROLLOUT_BATCH_SIZE:-2}"
N_SAMPLES_PER_PROMPT="${INFERENCE_N_SAMPLES_PER_PROMPT:-2}"
GLOBAL_BATCH_SIZE="${INFERENCE_GLOBAL_BATCH_SIZE:-4}"

case "$LAYOUT" in
  decoupled)
    # Separate pools must use the fully-async lifecycle.  The synchronous
    # path performs the initial Rollout weight update inline in the Actor
    # Serve replica; across nodes that blocks the Serve event loop while the
    # NCCL/DCS transaction completes and can trip the health probe.
    LAYOUT_ARGS=(--fully-async)
    if [[ $((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT)) -ne "$GLOBAL_BATCH_SIZE" ]]; then
      echo "decoupled validation requires global batch size must equal rollout batch size x samples per prompt" >&2
      exit 2
    fi
    ;;
  split)
    expected_actor_gpus=$((ROLLOUT_GPUS + TEACHER_GPUS + GENRM_GPUS))
    if [[ "$ACTOR_GPUS" -ne "$expected_actor_gpus" ]]; then
      echo "split layout requires actor GPUs=$expected_actor_gpus, got $ACTOR_GPUS" >&2
      exit 2
    fi
    LAYOUT_ARGS=(--colocate --offload-train --offload-rollout)
    ;;
  *)
    echo "INFERENCE_LAYOUT must be decoupled or split" >&2
    exit 2
    ;;
esac

# TMS is the default colocated offload mechanism.  Selective offload remains
# an explicit escape hatch for backends where torch_memory_saver is unavailable.
if [[ "${INFERENCE_SELECTIVE_OFFLOAD:-0}" == 1 ]]; then
  LAYOUT_ARGS+=(--selective-offload)
fi

# Deferred Teacher/GenRM scoring is the shared-bundle lifecycle used by split.
# Decoupled engines already have independent pools; the rollout pipeline
# pre-fills Teacher before invoking the custom GenRM reward in that mode.
if [[ "$LAYOUT" == split ]]; then
  REWARD_LIFECYCLE_ARGS=(--opd-teacher-defer --defer-reward-to-post-process --rm-type dummy)
else
  REWARD_LIFECYCLE_ARGS=(--rm-type dummy)
fi

mkdir -p "$INFERENCE_TRAIN_OUTPUT"
PROMPT_DATA="$INFERENCE_TRAIN_OUTPUT/prompts.jsonl"
cat > "$PROMPT_DATA" <<'JSONL'
{"prompt":"Continue the sequence: one, two, three,", "label":"four"}
{"prompt":"What is two plus three? Explain briefly.", "label":"5"}
{"prompt":"Write a sentence about the moon.", "label":"moon"}
{"prompt":"Name a primary color and explain your choice.", "label":"red"}
JSONL

if [[ "$LAYOUT" == decoupled ]]; then
  # Fully-async training consumes advantages from the dedicated CPU service.
  RESOURCE_JSON=$(printf '{"actor":[1,%s],"rollout":[1,%s],"teacher":[1,%s],"genrm":[1,%s],"advantages":[1,0]}' \
    "$ACTOR_GPUS" "$ROLLOUT_GPUS" "$TEACHER_GPUS" "$GENRM_GPUS")
else
  RESOURCE_JSON=$(printf '{"actor":[1,%s],"rollout":[1,%s],"teacher":[1,%s],"genrm":[1,%s]}' \
    "$ACTOR_GPUS" "$ROLLOUT_GPUS" "$TEACHER_GPUS" "$GENRM_GPUS")
fi

TRAIN_ARGS=(
  --train-backend megatron
  --resource "$RESOURCE_JSON"
  --num-gpus-per-node "$GPU_NODE_WIDTH"
  --actor-num-nodes "$ACTOR_NODES" --actor-num-gpus-per-node "$ACTOR_GPUS_PER_NODE"
  --max-staleness 0 --num-data-storage-units 1
  --hf-checkpoint "$MODEL_DIR" --load "$MODEL_DIR" --megatron-to-hf-mode bridge
  --save "$INFERENCE_TRAIN_OUTPUT/actor" --save-interval 1
  --save-hf "$INFERENCE_TRAIN_OUTPUT/hf/{rollout_id}"
  --prompt-data "$PROMPT_DATA" --input-key prompt --label-key label
  --apply-chat-template --apply-chat-template-kwargs '{"enable_thinking":false}'
  --num-rollout 2 --rollout-batch-size "$ROLLOUT_BATCH_SIZE" --n-samples-per-prompt "$N_SAMPLES_PER_PROMPT"
  --num-steps-per-rollout 1 --global-batch-size "$GLOBAL_BATCH_SIZE" --micro-batch-size 1
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
  --use-opd --opd-type sglang --opd-token-selection student_sampled
  --teacher-hf-checkpoint "$MODEL_DIR" --teacher-num-gpus-per-engine "$TEACHER_GPUS"
  --teacher-sglang-attention-backend triton --teacher-sglang-disable-cuda-graph
  --teacher-sglang-mem-fraction-static 0.5 --teacher-sglang-context-length 1024
  --teacher-sglang-max-total-tokens 2048 --opd-teacher-timeout-s 180
  --opd-kl-coef 0 --opd-loss-coef 0.1 --opd-kl-type low_var_kl
  --genrm-model-path "$MODEL_DIR" --genrm-num-gpus "$GENRM_GPUS"
  --genrm-num-gpus-per-engine "$GENRM_GPUS"
  --genrm-engine-config '{"attention_backend":"triton","disable_cuda_graph":true,"mem_fraction_static":0.5,"context_length":1024,"max_total_tokens":2048}'
  --custom-rm-path scripts.training.hpc.unified_inference_reward.score
  --reward-key score --reward-max-concurrency 2
  --rollout-num-gpus-per-engine "$ROLLOUT_GPUS" --sglang-load-format dummy
  --sglang-enable-weights-cpu-backup --sglang-attention-backend triton --sglang-disable-cuda-graph
  --sglang-mem-fraction-static 0.5 --sglang-context-length 1024
  --sglang-max-total-tokens 2048 --sglang-max-running-requests 4
  --attention-dropout 0 --hidden-dropout 0 --attention-backend fused
  --no-gradient-accumulation-fusion --no-rope-fusion
  --train-env-vars '{"NVTE_FLASH_ATTN":"0","NVTE_FUSED_ATTN":"1"}'
  --tb-project-name "$INFERENCE_TRAIN_OUTPUT/tensorboard" --tb-experiment-name "unified-inference-$LAYOUT"
  --dump-details "$INFERENCE_TRAIN_OUTPUT/details"
)

# Bash treats an empty array expansion as an unset variable under `set -u`.
# Append layout-specific flags only when the selected layout has any.
if ((${#LAYOUT_ARGS[@]})); then
  TRAIN_ARGS+=("${LAYOUT_ARGS[@]}")
fi
TRAIN_ARGS+=("${REWARD_LIFECYCLE_ARGS[@]}")

if [[ "${INFERENCE_VALIDATE_ARGS_ONLY:-0}" == 1 ]]; then
  python -c 'from relax.utils.arguments import parse_args; parse_args(); print("TRAINING_ARGUMENTS_VALID")' \
    "${MODEL_ARGS[@]}" "${TRAIN_ARGS[@]}" "$@"
else
  RAY_SUBMIT_ARGS=(ray job submit --address="$RAY_DASHBOARD")
  if [[ "${INFERENCE_WAIT_FOR_JOB:-0}" == 1 ]]; then
    # Acceptance runners must not start follow-up topology or fault tests while
    # the training job is still consuming the same Ray cluster. Keep the
    # historical asynchronous default for interactive launchers.
    RAY_SUBMIT_ARGS+=(--runtime-env-json="$RUNTIME_ENV_JSON")
  else
    RAY_SUBMIT_ARGS+=(--no-wait --runtime-env-json="$RUNTIME_ENV_JSON")
  fi
  "${RAY_SUBMIT_ARGS[@]}" \
    -- python -m relax.entrypoints.train "${MODEL_ARGS[@]}" "${TRAIN_ARGS[@]}" "$@"
fi
