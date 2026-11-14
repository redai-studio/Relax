#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Qwen3.6-35B-A3B, 8xPPU (ZW-M890P), sync GRPO on dapo-math, on the
# default image (relax-sail:ppu-v0.1.0, VMM torch_memory_saver).
#
#   bash scripts/training/text/run-qwen36-35B-A3B-mtp-8xppu.sh
#   MODEL_DIR=/cpfs/models DATA_DIR=/cpfs/datasets bash ...   # this machine
#
# Shape: TP2 x PP1 x EP8 (ETP1), colocate, max-tokens 10240, mem-frac 0.7.
# PP=1 forbids --decoder-first-pipeline-num-layers (arguments.py asserts it).
#
# The train actor runs expandable_segments via --train-env-vars; rollout/sglang
# stay non-expandable.  Colocate sets offload_train=True, which raises on
# expandable unless --selective-offload is given.
#
# Every export below must precede the entrypoint: it freezes the ray runtime env.

set -ex
set -o pipefail

now=$(date "+%Y-%m-%d-%H:%M:%S")
echo "当前时间: $now"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
EXP_DIR="${EXP_DIR:-${SCRIPT_DIR}/../../../../exps}"

export HOST_IP="${HOST_IP:-127.0.0.1}"
export WORKDIR=/workspace
export MODEL_DIR="${MODEL_DIR:-${EXP_DIR}}"
export DATA_DIR="${DATA_DIR:-${EXP_DIR}}"
export PROJECT_NAME="${PROJECT_NAME:-Relax-Qwen3.6-35B-A3B}"
# local-ppu.sh puts this on PYTHONPATH.
export MEGATRON="${MEGATRON:-/root/Megatron-LM}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
# Pinned; the probe resolves to ppu as well, via /dev/alixpu.
export RELAX_DEVICE_TYPE=ppu
export RL_MODEL_TYPE="Qwen36_35B_text"

export NCCL_SOCKET_IFNAME=eth0
export GLOO_SOCKET_IFNAME=eth0
export TP_SOCKET_IFNAME=eth0

# Inherited by rollout/sglang; the train actor flips it back via --train-env-vars.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
# Decode-graph replay needs this; it must also ride in the ray runtime env.
export HGGC_DEVICE_GRAPH=2
export EXTRA_ENV_VARS_JSON='"HGGC_DEVICE_GRAPH": "2"'
export PYTORCH_ENABLE_PPU_ELEMENTWISE_OPT=1

# >= per-rank cap x draft tokens (640/8 x 4 = 320); undersized hangs engine init.
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=512

export RELAX_SET_CPU_AFFINITY=1
export RELAX_CPU_AFFINITY_MODE=node
export SGLANG_NUMA_BIND=1
# V2=1 (default) routes the binding through numactl, absent in this image -> unbound.
export SGLANG_NUMA_BIND_V2=0
# sglang maps gpu_id linearly onto sockets (wrong here) -> off.
export SGLANG_SET_CPU_AFFINITY=0

export RAY_DEDUP_LOGS=0
export RAY_DEDUP_LOGS_AGG_WINDOW_S=0

unset http_proxy
unset https_proxy

# PPU_CLEAR_CACHE=1 forces cold caches on a rerun; must run before local-ppu.sh,
# which starts the ray head into /tmp/ray.
if [ "${PPU_CLEAR_CACHE:-0}" = "1" ]; then
    rm -rf /tmp/ray /tmp/torchinductor_root /root/.cache/triton /root/.triton /root/.cache/nvte_gg_rtc
    echo "[cache] cleared triton/inductor/nvte-rtc caches + /tmp/ray (cold-cache run)"
fi

if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    source "${SCRIPT_DIR}/../../entrypoint/local-ppu.sh"
fi
source "${SCRIPT_DIR}/../../models/qwen36-35B-A3B.sh"

NUM_ROLLOUT="${NUM_ROLLOUT:=1000}"
TIMELINE="${DATA_DIR}/dapo-math-17k/dapo-math-17k.jsonl"

CKPT_ARGS=(
   --hf-checkpoint ${MODEL_DIR}/Qwen3.6-35B-A3B/
   --ref-load ${MODEL_DIR}/Qwen3.6-35B-A3B/
   --megatron-to-hf-mode bridge
   --warm-hf-checkpoint-page-cache

   --load ${EXP_DIR}/Qwen3.6-35B-A3B_mtp_mcore_8xppu/
   --save ${EXP_DIR}/Qwen3.6-35B-A3B_mtp_mcore_8xppu/
   --save-interval 100
   --max-actor-ckpt-to-keep 1
)

ROLLOUT_ARGS=(
   --prompt-data ${TIMELINE}
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type dapo
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

EVAL_ARGS=(
   --log-passrate
   --skip-eval-before-train
   --eval-interval 20
   --eval-prompt-data aime ${DATA_DIR}/aime-2024/aime-2024.jsonl
   --n-samples-per-eval-prompt 8
   --eval-max-response-len 8192
   --eval-top-p 0.7
)

PERF_ARGS=(
   --tensor-model-parallel-size 2
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 8
   --expert-tensor-parallel-size 1

   # max-tokens-per-gpu must divide TP*128 so the dynamic-batch packing is exact.
   --use-dynamic-batch-size
   --max-tokens-per-gpu 10240
   --moe-flex-dispatcher-backend deepep
   --moe-token-dispatcher-type flex
   # DeepEP SM budget, shared by dispatch and combine; 38 is the even-count ceiling.
   --moe-flex-dispatcher-num-sms 38
   --log-probs-chunk-size 4096

   # TE fused topk; --enable-experimental unlocks fused_indices_to_multihot.
   --enable-experimental
   --moe-router-fusion

   # Selective recompute; only fits because the train actor runs expandable_segments:True.
   --recompute-granularity selective
   --recompute-modules core_attn moe_act layernorm
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
   # TIS off: off-policy correction disabled for this run (upstream default is off).
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

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 8
   --sglang-mem-fraction-static 0.7
   --sglang-ep-size 8
   --sglang-enable-dp-attention
   --sglang-dp-size 8
   --sglang-enable-dp-lm-head
   --sglang-moe-dense-tp-size 1
   # 3200 mamba slots / 5 per request / dp 8 = 80 per rank, matched by 640/8.
   --sglang-max-mamba-cache-size 3200
   --sglang-cuda-graph-bs 1 2 4 8 16 32 64 128 256
   --sglang-server-concurrency 1024
   --sglang-max-running-requests 640
   --sglang-chunked-prefill-size 8192
   --sglang-page-size 64
   --sglang-disable-shared-experts-fusion
   --sglang-attention-backend fa3
   --sglang-disable-piecewise-cuda-graph
   --sglang-stream-interval 128
   --sglang-watchdog-timeout 3600
   --sglang-enable-metrics
   --sglang-router-policy round_robin

   # enable sglang MTP: the checkpoint already ships mtp.layers.0, no draft ckpt.
   # Keep draft = steps + 1 with eagle-topk 1.
   --sglang-speculative-algorithm EAGLE
   --sglang-speculative-num-steps 3
   --sglang-speculative-eagle-topk 1
   --sglang-speculative-num-draft-tokens 4

   # MambaRadixCache's default `no_buffer` breaks EAGLE/MTP -> `extra_buffer`
   # + spec-v2 overlap scheduler (see _handle_mamba_radix_cache).
   --sglang-mamba-scheduler-strategy extra_buffer

   # Inference-side MoE all-to-all over DeepEP's low-latency decode kernels.
   --sglang-moe-a2a-backend deepep
   --sglang-deepep-mode auto

   # VMM weight resume needs a CPU backup, else offloaded weights come back zero.
   --sglang-enable-weights-cpu-backup
)

# PPU NUMA binding needs the per-device sysfs node, which can only be read here.
PPU_NUMA_NODE_LIST=$(python3 - <<'PY'
import torch

nodes = []
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    bus = "%04x:%02x:%02x.0" % (p.pci_domain_id, p.pci_bus_id, p.pci_device_id)
    with open(f"/sys/bus/pci/devices/{bus}/numa_node") as f:
        nodes.append(f.read().strip())
print(" ".join(nodes))
PY
) || PPU_NUMA_NODE_LIST=""
if [ -n "${PPU_NUMA_NODE_LIST}" ]; then
   echo "[numa] sglang numa_node list: ${PPU_NUMA_NODE_LIST}"
   SGLANG_ARGS+=( --sglang-numa-node ${PPU_NUMA_NODE_LIST} )
fi

WANDB_ARGS=(
   --tb-project-name ${PROJECT_NAME}
   --tb-experiment-name qwen3.6-mtp-ppu8-vmm-${now}
   --no-use-metrics-service
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)


source "${SCRIPT_DIR}/../../entrypoint/runtime-env-ppu.sh"

mkdir -p log
ray job submit --address="http://${HOST_IP}:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 -m relax.entrypoints.train \
   --resource '{"actor": [1, 8], "rollout": [1, 8]}' \
   --colocate \
   --max-staleness 0 \
   --selective-offload \
   --train-env-vars '{"PYTORCH_CUDA_ALLOC_CONF":"expandable_segments:True"}' \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${GRPO_ARGS[@]}" \
   "${WANDB_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${EVAL_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${MISC_ARGS[@]}"  2>&1 | tee log/qwen36-35B-A3B-mtp-ppu8-vmm-${now}.log
