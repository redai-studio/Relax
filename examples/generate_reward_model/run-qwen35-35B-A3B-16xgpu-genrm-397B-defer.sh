#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Qwen3.5-35B-A3B 16xGPU (2-node) fully colocate training script for DAPO math
# dataset, using Qwen3.5-397B-A17B as the Generative Reward Model (GenRM).
#
# Mode: DEFER / SWAP — rollout and GenRM share ALL 16 GPUs (shared-bundles).
# Inline reward is a no-op (--rm-type dummy); after all rollout finishes,
# custom_reward_post_process (post_process_genrm_swap.py) offloads rollout,
# onloads GenRM on the same 16 GPUs, batch-scores every sample, then offloads
# GenRM before training starts. Rollout runs at full 16-GPU width; GenRM also
# runs at full 16-GPU width. Serialized instead of overlapped.
#
# Suitable for RLVR / heavy-rollout workloads where the rollout kernel
# dominates and long-tail is minimal — the extra 8 GPUs during rollout more
# than pay for the lost overlap. See README for tradeoffs vs the split script.
#
# Usage:
#   bash examples/generate_reward_model/run-qwen35-35B-A3B-16xgpu-genrm-397B-defer.sh

set -ex
set -o pipefail

now=$(date "+%Y-%m-%d-%H:%M:%S")
echo "当前时间: $now"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
# Auto-source local environment when not launched via an external entrypoint
if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    source "${SCRIPT_DIR}/../../scripts/entrypoint/local.sh"
fi
source "${MODEL_CONFIG_DIR}/qwen35-35B-A3B.sh"

PROJECT_NAME="${PROJECT_NAME:=Relax/dev/dapo-math-genrm}"
EXP_DIR="${EXP_DIR:-${SCRIPT_DIR}/../../exps}"
MODEL_DIR="${MODEL_DIR:-${EXP_DIR}}"
DATA_DIR="${DATA_DIR:-${EXP_DIR}}"
NUM_ROLLOUT="${NUM_ROLLOUT:=1000}"

CKPT_ARGS=(
   --hf-checkpoint ${MODEL_DIR}/Qwen3.5-35B-A3B/
   --ref-load ${MODEL_DIR}/Qwen3.5-35B-A3B/
   --megatron-to-hf-mode bridge
   --warm-hf-checkpoint-page-cache
   # --load ${EXP_DIR}/Qwen3.5-35B-A3B_mcore_16xgpu_genrm397B_defer/
   --save ${EXP_DIR}/Qwen3.5-35B-A3B_mcore_16xgpu_genrm397B_defer/
   --save-interval 1000
   --max-actor-ckpt-to-keep 1
)

PROMPT_SET=${DATA_DIR}/dapo-math-17k/dapo-math-17k.jsonl

ROLLOUT_ARGS=(
   --prompt-data ${PROMPT_SET}
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type dummy
   --defer-reward-to-post-process
   --custom-reward-post-process-path examples.generate_reward_model.post_process_genrm_swap.custom_reward_post_process
   --reward-key score
   --num-rollout ${NUM_ROLLOUT}
   --rollout-batch-size 16
   --n-samples-per-prompt 8
   --rollout-max-response-len 8192
   --rollout-temperature 1
   --global-batch-size 128
   --use-fault-tolerance
   --balance-data
)

# Training reward is --rm-type dummy (real reward is deferred to the GenRM
# post-process pass). Eval runs inline, so without an explicit scorer it would
# inherit the global dummy rm_type and report a constant 0.0 (pass@k all zero).
# resolve_rm_type prefers the per-sample metadata rm_type over args.rm_type, and
# inject_metadata sources it from the eval dataset config's rm_type. --eval-config
# is the only path that can set a per-dataset rm_type (--eval-prompt-data only
# carries name/path), so pin the AIME eval to the real DAPO math verifier here.
EVAL_CONFIG=${EXP_DIR}/eval-aime-dapo-${now}.yaml
cat >"${EVAL_CONFIG}" <<EOF
eval:
  datasets:
    - name: aime
      path: ${DATA_DIR}/aime-2024/aime-2024.jsonl
      rm_type: dapo
EOF

EVAL_ARGS=(
   --log-passrate
   --skip-eval-before-train
   --eval-interval 20
   --eval-config ${EVAL_CONFIG}
   --n-samples-per-eval-prompt 8
   --eval-max-response-len 8192
   --eval-top-p 0.7
)

PERF_ARGS=(
   --tensor-model-parallel-size 4
   --sequence-parallel
   --pipeline-model-parallel-size 2
   --context-parallel-size 1
   --expert-model-parallel-size 8
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu 20480
   # --log-probs-max-tokens-per-gpu 40960

   --moe-flex-dispatcher-backend deepep
   --moe-token-dispatcher-type flex
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
   --use-tis
)

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

   # NOTE(wuhuan): to avoid algorithm performance degradation
   --no-rope-fusion
   --moe-router-load-balancing-type "none"
   --moe-aux-loss-coeff 0.0
)

# Rollout engine for the 35B-A3B actor. Shared-bundles colocate with GenRM:
# rollout owns all 16 GPUs during generate; GenRM is asleep. Between rollout
# and train, custom_reward_post_process (post_process_genrm_swap.py) offloads
# rollout, onloads GenRM on the same 16 GPUs, batch-scores, then offloads
# GenRM. mem_fraction can be generous because rollout and GenRM never hold
# GPU memory at the same time.
# rollout-num-gpus-per-engine 8 → 2 engines (TP=8 each) across 16 GPUs.
# sglang-server-concurrency 32 → 32 in-flight per engine × 2 engines = 64 global.
SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 2
   --sglang-mem-fraction-static 0.7
   --sglang-server-concurrency 32
)

# GenRM engine (Qwen3.5-397B-A17B FP8) on the SAME 16 GPUs as rollout.
# TP=8 is hard-capped by FP8 quantization: shared_expert per-partition must
# divide block_n=128, i.e. 1024/TP % 128 == 0 → TP must divide 8. So we run
# 2 engines of TP=DP=EP=8 across the 16 GPUs; router load-balances the batch.
# Onloaded only during the post-process batch pass; asleep during rollout
# and train.
GENRM_ARGS=(
   --genrm-model-path ${MODEL_DIR}/Qwen3.5-397B-A17B-FP8/
   --genrm-num-gpus 16
   --genrm-num-gpus-per-engine 8
   --genrm-engine-config '{"context_length": 10240, "mem_fraction_static": 0.85, "dp_size": 8, "ep_size": 8, "enable_dp_attention": true, "moe_dense_tp_size": 1, "enable_dp_lm_head": true, "disable_cuda_graph": true, "server_concurrency": 1024, "watchdog_timeout": 3600}'
   --genrm-sampling-config '{"temperature": 0.1, "top_p": 1.0, "top_k": -1, "max_response_len": 32, "chat_template_kwargs": {"enable_thinking": false}}'
)

WANDB_ARGS=(
   --use-clearml
   --use-metrics-service
   --tb-project-name  ${PROJECT_NAME}
   --tb-experiment-name qwen35-35B-A3B-GRPO-GenRM397B-defer-gpu16-${now}
)

MISC_ARGS=(
   # default dropout in megatron is 0.1
   --attention-dropout 0.0
   --hidden-dropout 0.0
   # should be good for model performance
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   # need to comment this when using model with MLA
   --attention-backend flash
)

# Tune NCCL/DeepEP timeouts for the heavier 397B GenRM engine.
RUNTIME_ENV_JSON=$(python3 -c '
import json, os
d = json.loads(os.environ["RUNTIME_ENV_JSON"])
d.setdefault("env_vars", {}).update({
    "TORCH_DIST_INIT_BARRIER": "1",
    "TORCH_NCCL_BLOCKING_WAIT": "0",
    "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
    "TORCH_DISTRIBUTED_DEFAULT_TIMEOUT": "3600",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "256",
})
print(json.dumps(d))
')
export RUNTIME_ENV_JSON

# ============================================
# Resource Configuration (DEFER / SWAP):
# 16 GPU (2 nodes). Actor / rollout / GenRM all bound to the SAME 16-GPU
# bundle (shared-bundles colocate: rollout_num_gpus == genrm_num_gpus ==
# actor_total triggers _genrm_colocate_with_rollout in
# relax/utils/arguments.py:2887).
# Two-phase execution (verl-style "reward loop colocate mode"):
#   Phase A: rollout awake (16 GPU, 2× TP=8 engines), GenRM asleep.
#            --rm-type dummy → inline reward is a no-op.
#   Phase B: after all rollout done, post_process_genrm_swap.py offloads
#            rollout, onloads GenRM on the same 16 GPUs (2× TP=8 engines),
#            batch-scores every sample, then offloads GenRM.
#   Phase C: actor wakes for train + weight update; GenRM stays offloaded
#            (--defer-reward-to-post-process makes actor.update_weights skip
#            its usual GenRM onload).
# ============================================

mkdir -p log
ray job submit ${RAY_NO_WAIT:+--no-wait} --address="http://${HOST_IP}:8265" \
   ${WORKING_DIR:+--working-dir "${WORKING_DIR}"} \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 -m relax.entrypoints.train \
   --resource '{"actor": [1, 16], "rollout": [1, 16], "genrm": [1, 16]}' \
   --colocate \
   --max-staleness 0 \
   --rollout-num-gpus 16 \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${GRPO_ARGS[@]}" \
   "${WANDB_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${EVAL_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${GENRM_ARGS[@]}" \
   "${MISC_ARGS[@]}"  2>&1 | tee log/qwen35-35B-A3B-GRPO-GenRM397B-defer-gpu16-${now}.log
