#!/usr/bin/env bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Qwen3.5-2B distills from the math, code and instruction-following experts in
# https://huggingface.co/collections/XINLI1997/dn-mopd. By default, Student rollout
# and Teacher scoring use one GPU each. Teacher replicas switch checkpoints together.
# Student training uses all GPUs (TP2).
# Run on a dedicated single-node Ray cluster with RAY_ADDRESS set to its GCS.
#
# From the Relax checkout, with MODEL_DIR and DATA_DIR set:
#   hf download Qwen/Qwen3.5-2B --revision 15852e8c16360a2fea060d615a32b45270f8a8fc --local-dir "$MODEL_DIR/Qwen3.5-2B"
#   for item in math:2433fcdb07936644dc0d8ed88257e245f3b28f61 code:672d5947fadcef406af832e4913b08ce7daa0bec if:a346839dbd6197ec849ac14e9700328ecf321e70; do
#     name="DN-MOPD-Qwen3.5-2B-teacher-${item%%:*}"
#     hf download "XINLI1997/$name" --revision "${item#*:}" --local-dir "$MODEL_DIR/$name"
#   done
#   hf download --repo-type dataset XINLI1997/DN-MOPD-Data student/train_3domain.jsonl --revision 9f2424c297a610849eea2bc5006e19344bfc2280 --local-dir "$DATA_DIR"
#   bash examples/on_policy_distillation/mopd/run-mopd-qwen35-2b-2xgpu-shared.sh

set -eo pipefail

RELAX=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
export RELAX
MODEL_DIR="${MODEL_DIR:-$RELAX/../models}"
DATA_DIR="${DATA_DIR:-$RELAX/../data/DN-MOPD-Data}"
PROMPTS="$DATA_DIR/student/train_3domain.jsonl"
STUDENT="$MODEL_DIR/Qwen3.5-2B"
EXPERT="$MODEL_DIR/DN-MOPD-Qwen3.5-2B-teacher"
for path in "$PROMPTS" "$STUDENT/config.json" "$EXPERT-math/config.json" "$EXPERT-code/config.json" "$EXPERT-if/config.json"; do
    [[ -r "$path" ]] || { echo "Missing recipe input: $path" >&2; exit 1; }
done
: "${RAY_ADDRESS:?Set the GCS address of a dedicated Ray cluster for this run}"
export RELAX_SKIP_CLUSTER_CLEANUP=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export SGLANG_ENABLE_LOGITS_PROCESSER_CHUNK=1 SGLANG_LOGITS_PROCESSOR_CHUNK_SIZE=512
export RELAX_PROPAGATE_ENV_VARS="${RELAX_PROPAGATE_ENV_VARS:+${RELAX_PROPAGATE_ENV_VARS},}HF_HUB_OFFLINE,TRANSFORMERS_OFFLINE,SGLANG_ENABLE_LOGITS_PROCESSER_CHUNK,SGLANG_LOGITS_PROCESSOR_CHUNK_SIZE"
if [[ -z "${RELAX_ENTRYPOINT_MODE:-}" ]]; then
    source "$RELAX/scripts/entrypoint/ray-job.sh"
fi
source "$RELAX/scripts/models/qwen35-2B.sh"

NUM_ROLLOUT="${NUM_ROLLOUT:-2}"
ROLLOUT_GPUS="${ROLLOUT_GPUS:-1}"
TEACHER_GPUS="${TEACHER_GPUS:-1}"
ACTOR_GPUS=$((ROLLOUT_GPUS + TEACHER_GPUS))

CKPT_ARGS=(
    --hf-checkpoint "${STUDENT}"
    --megatron-to-hf-mode bridge
)

ROLLOUT_ARGS=(
    --prompt-data "${PROMPTS}"
    --input-key prompt
    --label-key label
    --metadata-key metadata
    --apply-chat-template
    --apply-chat-template-kwargs '{"enable_thinking":false}'

    --rm-type dummy
    --num-rollout "${NUM_ROLLOUT}"
    --rollout-batch-size 12
    --n-samples-per-prompt 2
    --rollout-max-prompt-len 2048
    --rollout-max-response-len 256
    --rollout-temperature 1
    --global-batch-size 24
)

# The dataset uses "ifeval" for the instruction-following Teacher.
TEACHER_ROUTES=$(jq -nc --arg p "${EXPERT}" '{math:($p+"-math"),code:($p+"-code"),ifeval:($p+"-if")}')

OPD_ARGS=(
    --use-opd
    --opd-type sglang
    --opd-shared-teacher
    --opd-teacher-key domain
    --opd-teacher-routes "${TEACHER_ROUTES}"
    --teacher-num-gpus-per-engine 1
    --teacher-sglang-mem-fraction-static 0.65
    --teacher-sglang-context-length 4096
    --teacher-sglang-attention-backend triton
    --teacher-sglang-disable-cuda-graph
    --opd-teacher-timeout-s 600
    --opd-only-reward
    --opd-kl-coef 1.0
    --opd-token-selection student_sampled
    --use-rollout-logprobs
)

GRPO_ARGS=(
    --advantage-estimator grpo
    --kl-loss-coef 0.0
)

OPTIMIZER_ARGS=(
    --optimizer adam
    --lr 1e-6
    --lr-decay-style constant
    --use-distributed-optimizer
)

EVAL_ARGS=(
    --skip-eval-before-train
)

MISC_ARGS=(
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --attention-softmax-in-fp32
    --no-rope-fusion
)

PERF_ARGS=(
    --tensor-model-parallel-size 2
    --sequence-parallel
    --pipeline-model-parallel-size 1
    --seq-length 4096
    --use-dynamic-batch-size
    --max-tokens-per-gpu 4096
    --log-probs-max-tokens-per-gpu 4096
)

SGLANG_ARGS=(
    --rollout-num-gpus-per-engine 1
    --sglang-mem-fraction-static 0.7
    --sglang-context-length 4096
    --sglang-attention-backend triton
    --sglang-load-format dummy
    --sglang-enable-weights-cpu-backup
)

RESOURCE_JSON="{\"actor\":[1,${ACTOR_GPUS}],\"rollout\":[1,${ROLLOUT_GPUS}],\"teacher\":[1,${TEACHER_GPUS}]}"
now=$(date "+%Y-%m-%d-%H:%M:%S")

python3 -m relax.entrypoints.train \
    --resource "${RESOURCE_JSON}" \
    --num-gpus-per-node "${ACTOR_GPUS}" \
    --rollout-num-gpus "${ROLLOUT_GPUS}" \
    --max-staleness 0 \
    --num-data-storage-units 1 \
    --colocate \
    --offload \
    "${MODEL_ARGS[@]}" \
    "${CKPT_ARGS[@]}" \
    "${ROLLOUT_ARGS[@]}" \
    "${OPD_ARGS[@]}" \
    "${GRPO_ARGS[@]}" \
    "${OPTIMIZER_ARGS[@]}" \
    "${PERF_ARGS[@]}" \
    "${SGLANG_ARGS[@]}" \
    "${EVAL_ARGS[@]}" \
    "${MISC_ARGS[@]}" 2>&1 | tee "mopd-qwen35-2b-${ACTOR_GPUS}xgpu-shared-${now}.log"
