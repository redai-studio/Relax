#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Qwen3.6-35B-A3B 32xGPU DPO on UltraFeedback (Relax side of the swift 对齐实验).
#
# 拓扑 (32 GPU): TP1 / PP2 / CP16 / EP8, DP = 32 / (1*2*16) = 1
#   40 层 -> 每 PP stage 20 层. 数据集 UltraFeedback, 短句对,
#   目标是"每步 64k token": GBS=76 × ~840 tok/对 ≈ 63.8k.
#
# Usage:
#   bash scripts/entrypoint/ray-job.sh scripts/training/dpo/run-qwen36-35B-A3B-dpo-32xgpu.sh

set -ex
set -o pipefail

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

now=$(date "+%Y-%m-%d-%H:%M:%S")
echo "当前时间: $now"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
# Auto-source local environment when not launched via an external entrypoint
if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    source "${SCRIPT_DIR}/../../entrypoint/local.sh"
fi
source "${MODEL_CONFIG_DIR}/qwen36-35B-A3B.sh"

# ---------------------------------------------------------------------------
# 路径
# ---------------------------------------------------------------------------
PROJECT_NAME="${PROJECT_NAME:-Relax/dpo/qwen3.6-35B-A3B}"
EXP_NAME="${EXP_NAME:-qwen36-35b-a3b-dpo-uf64ktok-32xgpu}"
EXP_DIR="${EXP_DIR:?Set EXP_DIR to a shared experiment root}"
MODEL_DIR="${MODEL_DIR:-${EXP_DIR}/model}"
DATA_DIR="${DATA_DIR:-${EXP_DIR}/data}"
SAVE_DIR="${SAVE_DIR:-${EXP_DIR}/checkpoints}"
CHECKPOINT_DIR="${SAVE_DIR}/${EXP_NAME}"

mkdir -p "${EXP_DIR}/logs"

# ---------------------------------------------------------------------------
# 集群与 ClearML
# ---------------------------------------------------------------------------
RAY_ADDRESS="${RAY_ADDRESS:-http://${HOST_IP:-127.0.0.1}:8265}"

# MetricsService 可能落在别的节点, 必须用共享路径下的 clearml.conf.
export CLEARML_CONFIG_FILE="${CLEARML_CONFIG_FILE:-${EXP_DIR}/clearml.conf}"
export RELAX_PROPAGATE_ENV_VARS="${RELAX_PROPAGATE_ENV_VARS:+${RELAX_PROPAGATE_ENV_VARS},}CLEARML_CONFIG_FILE"

RUNTIME_ENV_JSON="$(python3 -c '
import json, os
runtime = json.loads(os.environ["RUNTIME_ENV_JSON"])
env = runtime.setdefault("env_vars", {})
names = {
    name.strip()
    for source in (env.get("RELAX_PROPAGATE_ENV_VARS", ""), os.environ.get("RELAX_PROPAGATE_ENV_VARS", ""))
    for name in source.split(",")
    if name.strip()
}
names.update(("PYTORCH_CUDA_ALLOC_CONF", "CLEARML_CONFIG_FILE"))
env.update({name: os.environ[name] for name in ("PYTORCH_CUDA_ALLOC_CONF", "CLEARML_CONFIG_FILE")})
env["RELAX_PROPAGATE_ENV_VARS"] = ",".join(sorted(names))
print(json.dumps(runtime))
')"
export RUNTIME_ENV_JSON

# ---------------------------------------------------------------------------
# 模型 / checkpoint
# ---------------------------------------------------------------------------
HF_CHECKPOINT="${HF_CHECKPOINT:-${MODEL_DIR}/Qwen3.6-35B-A3B}"

CKPT_ARGS=(
   --hf-checkpoint "${HF_CHECKPOINT}"
   # DPO 需要冻结的 reference policy 算 ref logprob; --ref-load 复用现成的
   # ActorFwd / ROLES.reference 机制 (relax/components/actor_fwd.py).
   --ref-load "${HF_CHECKPOINT}"
   --megatron-to-hf-mode bridge
   --warm-hf-checkpoint-page-cache
   --enable-weights-backuper
   # 首次从 HF 初始化，后续从同一实验的 checkpoint 恢复。
   # 新实验或改变训练总步数时更换 EXP_NAME，避免复用旧调度器状态。
   --load "${CHECKPOINT_DIR}"
   --save "${CHECKPOINT_DIR}"
   --save-interval "${SAVE_INTERVAL:-25}"
   --max-actor-ckpt-to-keep "${MAX_ACTOR_CKPT_TO_KEEP:-2}"
)

# ---------------------------------------------------------------------------
# 偏好数据 (chosen / rejected 成对样本)
# ---------------------------------------------------------------------------
# 使用 Relax 的显式偏好 schema (prompt / chosen / rejected), 故 --input-key prompt.
PREFERENCE_DATA="${DATA_DIR}/ultrafeedback/ultrafeedback_train.jsonl"

# ---------------------------------------------------------------------------
# DPO 超参
# ---------------------------------------------------------------------------
# DPO 由 `38db5feb 【Task.31】feat(dpo): add pair-aware DPO training (#257)` 引入,
# 已内建: relax/engine/sft/dataset/preference.py 数据集、relax/backends/megatron/loss.py
# 的 dpo_loss_function、以及 --loss-type dpo / --dpo-beta.
# 64k 长序列所需的 CP/PP 支持在本分支补齐 (loss.py 的 `_preference_branch_sums`
# 与 runtime.py 的 `validate_preference_args`).
DPO_ARGS=(
   --loss-type dpo
   --dpo-beta 0.1
   --prompt-data "${PREFERENCE_DATA}"
   --input-key prompt
   # 精度对齐: 两侧都按文件顺序取前 N 行, 关闭 shuffle (镜像 swift --dataset_shuffle false).
   --no-dataset-shuffle
   --seed 42
   # UltraFeedback 短句对 (最长 ~5.2k), 8192 足够不截断.
   --preference-max-length 8192
   --preference-max-completion-length 4096
   --use-gloo-process-groups
)

# ---------------------------------------------------------------------------
# 并行配置 (32 GPU)
# ---------------------------------------------------------------------------
PERF_ARGS=(
   --tensor-model-parallel-size 1
   --pipeline-model-parallel-size 2
   --context-parallel-size 16
   --expert-model-parallel-size 8
   --sequence-parallel

   --actor-num-nodes 4
   --actor-num-gpus-per-node 8

   # 离线目标必须显式声明 resource: sft producer 是 CPU-only, actor 占满 32 卡.
   --resource '{"sft":[1,0],"actor":[1,32]}'
   # 每步的偏好"对"数 (一对 = chosen+rejected); DP=2, 每 rank 4 对.
   --global-batch-size 64
   # 正式跑 1 个 epoch (4096 对 / batch 64 = 64 步).
   --num-epoch 1

   # UltraFeedback 短句对 (平均 ~840 tok/对); 每步 76 对 ≈ 64k token.
   --seq-length 8192
   --micro-batch-size 1
   --use-dynamic-batch-size
   # 偏好目标按"对"计: 一对 = chosen + rejected, prompt 计两次.
   # --max-tokens-per-gpu 是"每 rank"预算, microbatch 总预算 = 该值 * CP16.
   # 8192 * 16 = 131072 >= 最大的一对 (120.6k) → 没有样本被判 oversize.
   --max-tokens-per-gpu 4096

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --no-rope-fusion
   --attention-backend flash
)

OPTIMIZER_ARGS=(
   --optimizer adam
   # 1e-6 是仓库对大模型训练的惯例 (scripts/training/text/ 下 51/53 用 1e-6);
   # 5e-7 是 0.6B DPO 示例的默认值, 对 35B 偏小.
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
   --clip-grad 1.0
)

# ---------------------------------------------------------------------------
# 指标 (ClearML)
# ---------------------------------------------------------------------------
WANDB_ARGS=(
   --use-tensorboard
   --use-clearml
   --use-metrics-service
   --tb-project-name "${PROJECT_NAME}"
   --tb-experiment-name "${EXP_NAME}-${now}"
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
)

# ---------------------------------------------------------------------------
# 提交
# ---------------------------------------------------------------------------
ray job submit ${RAY_NO_WAIT:+--no-wait} --address="${RAY_ADDRESS}" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 -m relax.entrypoints.train \
   --colocate \
   --use-health-check \
   --selective-offload \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${DPO_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${WANDB_ARGS[@]}" \
   "${MISC_ARGS[@]}" 2>&1 | tee "${EXP_DIR}/logs/${EXP_NAME}-${now}.log"
