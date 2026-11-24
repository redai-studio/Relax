#!/bin/bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Colocated GRPO with the original frozen vision tower. Shared by the OpenR1 hardware recipes.
# DATA_DIR is the directory containing multimodal-open-r1-8k-verified.
# Use scripts/entrypoint/ray-job.sh on an existing Ray cluster.
set -eo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)
: "${MODEL_DIR:?Set MODEL_DIR to the parent of the composite BF16 checkpoint}"
: "${DATA_DIR:?Set DATA_DIR to the dataset root}"
: "${SAVE_DIR:?Set SAVE_DIR to the experiment root}"
layers=${1:?Expected the decoder layer count from the hardware recipe}
shift
ep=16
update_weight_buffer_size=1073741824
mm_feature_transport=cpu
engram_config=
SGL_PARALLEL_ARGS=()
FULL_RUN_ARGS=()
OFFLOAD_ARGS=()
case "${layers}" in
    4)
        model_name=DeepSeek-V4.1-Flash-1017-4layer-bf16-vision
        model_script=deepseek-v4.1-4layer.sh
        nodes=4 pp=2 cp=2 context_len=4096 prompt_len=2048 response_len=256 token_budget=2048
        engine_gpus=8 attention_dp=1 mem_fraction=0.65 max_running=16
        ;;
    40)
        model_name=DeepSeek-V4.1-Flash-1017-bf16-vision
        model_script=deepseek-v4.1-flash.sh
        nodes=8 pp=4 cp=4 context_len=4096 prompt_len=2048 response_len=2048 token_budget=1024
        engine_gpus=64 attention_dp=64 mem_fraction=0.75 max_running=64
        server_concurrency=32
        chunked_prefill=$((attention_dp * 256))
        if [[ ${OPENR1_CLUSTER_GPUS:-64} == 128 ]]; then
            nodes=16 ep=32
            engine_gpus=${ROLLOUT_ENGINE_GPUS:-16}
            case "${engine_gpus}" in
                8)
                    attention_dp=1 mem_fraction=0.75 max_running=32
                    chunked_prefill=8192 server_concurrency=32
                    mm_feature_transport=cuda_ipc
                    engram_config='{"placement":"cpu"}'
                    SGL_PARALLEL_ARGS+=(--sglang-quantization fp8 --sglang-engram-config "${engram_config}")
                    export USE_ROLLOUT_ROUTING_REPLAY=1
                    ;;
                16)
                    # Eight-way attention TP and smaller IPC buckets leave room for the colocated weight push.
                    attention_dp=2 mem_fraction=0.94 max_running=16
                    chunked_prefill=4096 server_concurrency=16
                    update_weight_buffer_size=134217728
                    SGL_PARALLEL_ARGS+=(--sglang-max-total-tokens 32768)
                    # Only the selectively offloaded trainer uses expandable segments.
                    # SGLang retains its memory-saver allocator for weight/KV release.
                    FULL_RUN_ARGS+=(--train-env-vars '{"PYTORCH_ALLOC_CONF":"expandable_segments:True"}')
                    ;;
                32)
                    attention_dp=32 mem_fraction=0.75 max_running=32
                    chunked_prefill=8192 server_concurrency=32
                    ;;
                *) echo "Set ROLLOUT_ENGINE_GPUS to 8, 16 or 32 for this 128-GPU recipe" >&2; exit 2 ;;
            esac
            SGL_PARALLEL_ARGS+=(--sglang-router-policy round_robin)
        fi
        if (( attention_dp > 1 )); then
            SGL_PARALLEL_ARGS+=(--sglang-enable-dp-attention)
        fi
        SGL_PARALLEL_ARGS+=(
            --sglang-pp-size 1
            --sglang-moe-a2a-backend none
            --sglang-server-concurrency "${server_concurrency}"
            # SGLang divides this budget across DP ranks. Each rank needs one 256-token page.
            --sglang-chunked-prefill-size "${chunked_prefill}"
        )
        # GRPO has only the actor weight version here. Reuse its offloaded parameters for the
        # Bridge push and discard the flat gradient buffers while the trainer is asleep.
        OFFLOAD_ARGS=(
            --selective-offload
            --disable-weights-backuper
            --optimizer-offload-fraction 1.0
        )
        ;;
    *) echo "Expected 4 or 40 decoder layers" >&2; exit 2 ;;
esac
gpus=$((8 * nodes))
HF_CKPT=${HF_CKPT:-${MODEL_DIR}/${model_name}}
if [[ ${layers} == 40 && -n ${OPENR1_CLUSTER_GPUS:-} ]]; then
    split_dir="${DATA_DIR}/multimodal-open-r1-8k-verified/splits/stratified-eval128-v1"
    : "${PROMPT_DATA:=${split_dir}/train.parquet}"
    : "${NUM_ROLLOUT:=200}" "${ROLLOUT_BATCH_SIZE:=16}" "${N_SAMPLES_PER_PROMPT:=8}"
    : "${SAVE_INTERVAL:=50}" "${DUMP_DETAILS:=0}"
    if [[ ${USE_CLEARML:-1} == 1 ]]; then
        FULL_RUN_ARGS+=(
            --use-clearml
            --use-metrics-service
            --tb-project-name "${PROJECT_NAME:-Relax/rl/openr1mm}"
            --tb-experiment-name "${EXP_NAME:-deepseek-v41-1017-40layer-openr1mm-grpo}-$(date +%Y%m%d-%H%M%S)"
        )
    fi
    FULL_RUN_ARGS+=(
        --eval-prompt-data openr1_holdout "${EVAL_DATA:-${split_dir}/eval.parquet}"
        --eval-interval 10
        --n-samples-per-eval-prompt 1
        --eval-temperature 0
    )
fi
PROMPT_DATA=${PROMPT_DATA:-${DATA_DIR}/multimodal-open-r1-8k-verified/data/train-00000-of-00001_converted_noextract.parquet}
EXP_NAME=${EXP_NAME:-deepseek-v41-1017-${layers}layer-openr1mm-grpo}
RUN_DIR="${SAVE_DIR}/${EXP_NAME}"
NUM_ROLLOUT=${NUM_ROLLOUT:-2}
ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-8}
N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT:-4}
MAX_TOKENS_PER_GPU=${MAX_TOKENS_PER_GPU:-${token_budget}}
python3 - "${HF_CKPT}/config.json" "${layers}" <<'PY'
import json
import sys

with open(sys.argv[1]) as source:
    cfg = json.load(source)
if (cfg.get("model_type") != "deepseek_v41" or not cfg.get("vision_config")
        or cfg.get("quantization_config") or cfg["text_config"]["num_hidden_layers"] != int(sys.argv[2])):
    raise SystemExit(f"Expected the {sys.argv[2]}-layer 1017 composite BF16 checkpoint with its vision tower")
PY
source "${ROOT}/scripts/models/${model_script}"
CKPT_ARGS=(--hf-checkpoint "${HF_CKPT}" --megatron-to-hf-mode bridge)
if [[ ${SAVE_INTERVAL:-0} -gt 0 ]]; then
    CKPT_ARGS+=(--load "${RUN_DIR}/checkpoints" --save "${RUN_DIR}/checkpoints"
        --save-interval "${SAVE_INTERVAL}" --max-actor-ckpt-to-keep 2)
fi
# Retain the exact image tensors, response tokens, expert routes and logprobs for fixed-input replay.
if [[ ${DUMP_DETAILS:-1} == 1 ]]; then
    CKPT_ARGS+=(--save-debug-rollout-data "${RUN_DIR}/debug/rollout_data/{rollout_id}.pt"
        --save-debug-train-data "${RUN_DIR}/debug/train_data/{rollout_id}_{rank}.pt")
fi
ROLLOUT_ARGS=(
    --prompt-data "${PROMPT_DATA}" --input-key prompt --label-key label --multimodal-keys '{"image":"image"}'
    --apply-chat-template --apply-chat-template-kwargs '{"thinking_mode":"chat"}'
    --system-prompt 'Solve the problem and put your final answer inside <answer> </answer> tags.'
    --rollout-shuffle --rollout-seed 42 --rm-type openr1mm
    --num-rollout "${NUM_ROLLOUT}" --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
    --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}" --global-batch-size "$((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT))"
    --num-steps-per-rollout 1 --num-iters-per-train-update 1
    --rollout-max-prompt-len "${prompt_len}"
    --rollout-max-response-len "${RESPONSE_LEN:-${response_len}}"
    --rollout-max-context-len "${context_len}"
    --rollout-temperature 1 --image-max-token-num 1024
    --balance-data
)
if [[ ${USE_ROLLOUT_ROUTING_REPLAY:-1} == 1 ]]; then
    ROLLOUT_ARGS+=(--use-rollout-routing-replay)
fi
ACTOR_ARGS=(
    --qkv-format thd --use-dynamic-batch-size --cp-partition-mode contiguous
    --seq-length "${context_len}"
    --tensor-model-parallel-size 2
    --pipeline-model-parallel-size "${pp}"
    --context-parallel-size "${cp}" --expert-model-parallel-size "${ep}" --expert-tensor-parallel-size 1
    --sequence-parallel --allgather-cp --moe-token-dispatcher-type alltoall
    --micro-batch-size 1 --data-pad-size-multiplier 32 --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}"
    --log-probs-max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}" --calculate-per-token-loss
    --recompute-granularity full --recompute-method uniform --recompute-num-layers 1 --recompute-loss-function
    --bf16 --accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32
    --transformer-impl transformer_engine --attention-backend flash --attention-dropout 0 --hidden-dropout 0
    --advantage-estimator grpo --loss-type policy_loss --kl-coef 0 --kl-loss-coef 0
    --entropy-coef 0 --eps-clip 0.2 --eps-clip-high 0.28
    --optimizer adam --lr 1e-6 --lr-decay-style constant --weight-decay 0.1
    --adam-beta1 0.9 --adam-beta2 0.98 --adam-eps 1e-8 --clip-grad 1
    --optimizer-cpu-offload --overlap-cpu-optimizer-d2h-h2d --use-precision-aware-optimizer
    --no-pin-cpu-grads --no-pin-cpu-params --empty-unused-memory-level 1
)
SGL_ARGS=(
    --rollout-num-gpus-per-engine "${engine_gpus}"
    --sglang-tp-size "${engine_gpus}"
    --sglang-ep-size "${engine_gpus}"
    --sglang-dp-size "${attention_dp}"
    "${SGL_PARALLEL_ARGS[@]}"
    --sglang-external-model-package relax.backends.sglang.deepseekv41
    --sglang-load-format auto
    --sglang-context-length "${context_len}"
    --sglang-attention-backend dsv4 --sglang-moe-runner-backend auto
    --sglang-mm-feature-transport "${mm_feature_transport}"
    # Frozen image parameters are loaded from HF and must survive colocated weight release/resume.
    --sglang-enable-weights-cpu-backup
    --sglang-mem-fraction-static "${mem_fraction}"
    --sglang-max-running-requests "${max_running}"
    --sglang-watchdog-timeout 3600
)
TRAIN_ARGS=(
    --resource "{\"actor\":[1,${gpus}],\"rollout\":[1,${gpus}]}"
    --num-gpus-per-node 8
    --num-data-storage-units "${nodes}"
    --max-staleness 0 --colocate --offload-train --offload-rollout
    "${OFFLOAD_ARGS[@]}"
    "${MODEL_ARGS[@]}" "${CKPT_ARGS[@]}" "${ROLLOUT_ARGS[@]}" "${ACTOR_ARGS[@]}" "${SGL_ARGS[@]}"
    --seed 1234 --trust-remote-code --log-interval 1
    --update-weight-buffer-size "${update_weight_buffer_size}" --train-memory-margin-bytes 1073741824 "${FULL_RUN_ARGS[@]}" "$@"
)
if [[ ${DRY_RUN:-0} == 1 ]]; then
    printf '%q ' python3 -m relax.entrypoints.train "${TRAIN_ARGS[@]}"
    printf '\n'
    exit 0
fi
if [[ -z ${RELAX_ENTRYPOINT_MODE:-} ]]; then
    source "${ROOT}/scripts/entrypoint/local.sh"
fi
RUNTIME_ENV_JSON=$(python3 - "${RUN_DIR}/tensorboard" "${engine_gpus}" "${engram_config}" <<'PY'
import json
import os
import shutil
import sys
from pathlib import Path

runtime = json.loads(os.environ.get("RUNTIME_ENV_JSON") or "{}")
runtime.setdefault("env_vars", {}).update({
    "TENSORBOARD_DIR": sys.argv[1],
    "CUDA_DEVICE_MAX_CONNECTIONS": "1", "NCCL_CUMEM_ENABLE": "1", "NCCL_NVLS_ENABLE": "0",
    "RELAX_DSV41_CAPTURE_VISION_ROUTES": os.environ.get("USE_ROLLOUT_ROUTING_REPLAY", "1"),
    "SGLANG_DSV4_FP4_EXPERTS": "0", "SGLANG_OPT_FP8_WO_A_GEMM": "0",
    "SGLANG_OPT_DEEPGEMM_HC_PRENORM": "0", "SGLANG_OPT_FUSE_WQA_WKV": "0",
    "SGLANG_DISABLE_MULTIMEM_AG": "1", "SGLANG_DG_CACHE_DIR_PER_PROCESS": "1",
    "SGLANG_HEALTH_CHECK_TIMEOUT": "900", "TORCHINDUCTOR_COMPILE_THREADS": "1",
    "MEGATRON_SYNC_SAVE_BOUNDED_STAGING": "1",
})
env = runtime["env_vars"]
if sys.argv[3]:
    # The explicit SGLang config controls Engram placement for this recipe.
    env.pop("SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE", None)
    env.pop("SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT", None)
else:
    env["SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE"] = "1" if sys.argv[2] in ("8", "16") else "0"
    env["SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT"] = "shared" if sys.argv[2] == "8" else "per_rank"
if os.environ.get("OPENR1_CLUSTER_GPUS") == "128" and sys.argv[2] == "16":
    runtime["env_vars"]["RELAX_V41_LOW_MEMORY_WEIGHT_SYNC"] = "1"
    runtime["env_vars"]["RELAX_DSV41_SHARD_INPUT_EMBEDDING"] = "1"
    runtime["env_vars"]["RELAX_DSV41_STREAM_VOCAB_ROWS"] = "1"
    runtime["env_vars"]["RELAX_DSV41_ROUTE_BF16_EXPERTS"] = os.environ.get("RELAX_DSV41_ROUTE_BF16_EXPERTS", "1")
    runtime["env_vars"]["NCCL_BUFFSIZE"] = "1048576"
if os.environ.get("OPENR1_CLUSTER_GPUS") == "128":
    # A bare Ray worker on H800 can otherwise resolve the host CUDA 12 driver
    # before the CUDA 13 compatibility library supplied by the training image.
    nvcc = shutil.which("nvcc")
    cuda = os.environ.get("CUDA_HOME") or (str(Path(nvcc).parent.parent) if nvcc else None)
    if cuda and (Path(cuda) / "compat/libcuda.so.1").is_file():
        paths = [str(Path(cuda) / "compat"), *env.get("LD_LIBRARY_PATH", "").split(os.pathsep)]
        env["LD_LIBRARY_PATH"] = os.pathsep.join(dict.fromkeys(p for p in paths if p))
if sys.argv[2] == "64":
    # A 36-channel TP64 shard is not divisible by the BF16 clamp kernel's vector
    # width of eight. Replicate only the shared MLP; routed experts stay sharded.
    runtime["env_vars"]["SGLANG_SHARED_EXPERT_TP1"] = "1"
print(json.dumps(runtime))
PY
)
export RUNTIME_ENV_JSON
mkdir -p "${RUN_DIR}/logs"
ray job submit ${RAY_NO_WAIT:+--no-wait} \
    --address="${RAY_JOB_ADDRESS:-${RAY_DASHBOARD_ADDRESS:-http://127.0.0.1:8265}}" \
    ${WORKING_DIR:+--working-dir "${WORKING_DIR}"} --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- python3 -m relax.entrypoints.train "${TRAIN_ARGS[@]}" \
    2>&1 | tee "${RUN_DIR}/logs/submit-$(date +%Y%m%d-%H%M%S).log"
