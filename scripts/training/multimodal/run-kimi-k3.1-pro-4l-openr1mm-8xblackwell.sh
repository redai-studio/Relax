#!/bin/bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Reduced Kimi K3.1 SFT on one 8-GPU Blackwell node. The prepared HF checkpoint
# retains the first four layers: dense KDA, two MoE KDA, and MoE DSA.
# TP2/PP1/CP1/EP8/ETP1; frozen full vision tower; original 896 experts and
# top-2048 NoPE indexer. The indexer is frozen, while DSA projections train.
# MODEL_DIR points to the reduced checkpoint; PROMPT_DATA to SFT parquet with
# prompt=[user,...,assistant] and image=[bytes,...]. No label-key is needed.
# Submit through scripts/entrypoint/ray-job.sh. Set PROFILE_DSA=1 to capture
# the first training step on rank 0 and inspect SparseMLA forward/backward.
# Prepare the checkpoint/data with scripts/tools/prepare_kimi_k31_smoke.py.
# Include real samples longer than 2048 tokens to exercise sparse selection.
# SFT applies its own deterministic dataset/epoch shuffle with SEED.
set -eo pipefail
export RAY_NO_WAIT="${RAY_NO_WAIT-1}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
RELAX_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." &>/dev/null && pwd)"
source "${MODEL_CONFIG_DIR:-${RELAX_ROOT}/scripts/models}/kimi-k3.1-pro.sh"
: "${MODEL_DIR:?Set MODEL_DIR to the reduced K3.1 checkpoint directory}"
: "${PROMPT_DATA:?Set PROMPT_DATA to the converted SFT parquet path}"
NUM_LAYERS="$(python3 - "${MODEL_DIR}/config.json" <<'PY'
import json, sys
config = json.load(open(sys.argv[1]))
text = config['text_config']
assert config['model_type'] == 'kimi_k31'
assert 1 < text['num_hidden_layers'] < 93
assert text['linear_attn_config']['kda_layers']
assert text['dsa_config']['dsa_layers']
print(text['num_hidden_layers'])
PY
)"
MODEL_ARGS+=(--num-layers "${NUM_LAYERS}" --moe-layer-freq "[0]+[1]*$((NUM_LAYERS - 1))")
EXP_NAME="${EXP_NAME:-kimi-k3.1-pro-${NUM_LAYERS}l-openr1mm-8xblackwell-dsa}"
NUM_STEPS="${NUM_STEPS:-3}"
now=$(date "+%Y%m%d-%H%M%S")
unset RELAX_SFT_RAW_TEXT_CONCAT
CKPT_ARGS=(
    --hf-checkpoint "${MODEL_DIR}"
    --ref-load "${MODEL_DIR}"
    --trust-remote-code
    --megatron-to-hf-mode bridge
    --num-rollout "${NUM_STEPS}"
    --num-epoch 1
)
if [ -n "${SAVE_DIR:-}" ]; then
    CKPT_ARGS+=(--save "${SAVE_DIR}/${EXP_NAME}" --load "${SAVE_DIR}/${EXP_NAME}"
        --save-interval "${SAVE_INTERVAL:-2}" --max-actor-ckpt-to-keep 1)
fi
SFT_ARGS=(
    --loss-type sft
    --prompt-data "${PROMPT_DATA}"
    --input-key prompt
    --multimodal-keys '{"image":"image"}'
    --image-max-token-num 1024
    --global-batch-size "${GLOBAL_BATCH_SIZE:-4}"
    --use-dynamic-batch-size
    --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU:-8192}"
    --sft-oversize-strategy skip
    --sft-invalid-multimodal-strategy error
    --balance-data
    --per-rank-fetch
    --sft-prefetch-num-workers 4
    --sft-prefetch-buffer-size 16
    --sft-tq-timeout-minutes 60
)
PERF_ARGS=(
    --tensor-model-parallel-size 2
    --sequence-parallel
    --vision-dp-when-tp
    --freeze-vision-model
    --pipeline-model-parallel-size 1
    --context-parallel-size 1
    --expert-model-parallel-size 8
    --expert-tensor-parallel-size 1
    --moe-token-dispatcher-type alltoall
    --calculate-per-token-loss
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1
    --use-precision-aware-optimizer
    --main-grads-dtype bf16
    --disable-weights-backuper
    --cross-entropy-loss-fusion
    --sft-chunked-logits
    --sft-logits-chunk-size 256
)
# Use GPU optimizer by default; enable host offload only after checking
# available RAM and measured peak GPU memory on the node.
if [ "${OPTIMIZER_OFFLOAD_FRACTION:-0}" != 0 ]; then
    PERF_ARGS+=(--optimizer-cpu-offload --overlap-cpu-optimizer-d2h-h2d
        --optimizer-offload-fraction "${OPTIMIZER_OFFLOAD_FRACTION}")
fi
OPTIMIZER_ARGS=(
    --optimizer adam --lr "${LR:-1e-5}" --lr-decay-style constant
    --weight-decay 0.0 --adam-beta1 0.9 --adam-beta2 0.95 --clip-grad 1.0
)
PROFILE_ARGS=()
if [ "${PROFILE_DSA:-0}" = 1 ]; then
    : "${PROFILE_DIR:?Set PROFILE_DIR to a persistent profiler output directory}"
    # Capture immediately: a long compilation warmup can leave Kineto stopped
    # before the active window. Place a >2048-token sample in the first batch.
    PROFILE_ARGS=(--use-pytorch-profiler --profile-step-start 0 --profile-step-end 0
        --profile-target train_overall --profile-ranks 0 --profile-output-dir "${PROFILE_DIR}")
fi
TRAIN_ARGS=(
    --train-env-vars '{"FLA_TILELANG":"0","PYTORCH_CUDA_ALLOC_CONF":"expandable_segments:True"}'
    --resource '{"sft": [1, 0], "actor": [1, 8]}'
    --sft-max-in-flight-steps 1
    --num-data-storage-units 2
    "${MODEL_ARGS[@]}" "${CKPT_ARGS[@]}" "${SFT_ARGS[@]}"
    "${PERF_ARGS[@]}" "${OPTIMIZER_ARGS[@]}" "${PROFILE_ARGS[@]}"
    --attention-dropout 0.0 --hidden-dropout 0.0 --attention-backend flash
    --dsa-kernel-backend tilelang
    --tb-experiment-name "${EXP_NAME}-${now}"
    --seed "${SEED:-1234}"
    "$@"
)
if [ "${DRY_RUN:-0}" = 1 ]; then
    printf '%q ' python3 -m relax.entrypoints.train "${TRAIN_ARGS[@]}"
    printf '\n'
    exit 0
fi
: "${RAY_DASHBOARD_ADDRESS:?Set RAY_DASHBOARD_ADDRESS to the Ray Jobs HTTP address}"
if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    source "${RELAX_ROOT}/scripts/entrypoint/ray-job.sh"
fi
RUNTIME_ENV_JSON=$(python3 - <<'PY'
import json, os
runtime = json.loads(os.environ['RUNTIME_ENV_JSON'])
env = runtime.setdefault('env_vars', {})
env.pop('RELAX_SFT_RAW_TEXT_CONCAT', None)
env['RELAX_PROPAGATE_ENV_VARS'] = ','.join(dict.fromkeys(
    name for name in env.get('RELAX_PROPAGATE_ENV_VARS', '').split(',')
    if name and name != 'RELAX_SFT_RAW_TEXT_CONCAT'))
print(json.dumps(runtime))
PY
)
export RUNTIME_ENV_JSON
mkdir -p "${RELAX_ROOT}/log"
ray job submit ${RAY_NO_WAIT:+--no-wait} \
    --address="${RAY_DASHBOARD_ADDRESS}" \
    ${WORKING_DIR:+--working-dir "${WORKING_DIR}"} \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- python3 -m relax.entrypoints.train "${TRAIN_ARGS[@]}" \
    2>&1 | tee "${RELAX_ROOT}/log/${EXP_NAME}-${now}.log"
