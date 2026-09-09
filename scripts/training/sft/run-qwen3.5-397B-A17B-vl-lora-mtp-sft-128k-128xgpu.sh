#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Qwen3.5-397B-A17B (VLM) LoRA SFT — multimodal + tool-calling, 128xGPU (16-node), ray-submit launch.
#
# Relax-side 128-GPU adaptation of the ms-swift launcher
#   run_qwen35_397b_sft_v3_vit_lora.sh
# The model, dataset and no-MTP objective follow that committed launcher. Parallel settings follow the
# 128-GPU adaptation of the env-overridden `v12-20260810-221352` run (baseline captured in baseline_msswift_0625/):
#   - TP=4 / EP=8  (committed script hardcodes TP=8 / EP=16)
#   - 128 GPU (committed script targets 256)
# For the TP=8/EP=16, no-MTP, 256-GPU faithful port see
# run-qwen3.5-397B-A17B-vl-lora-sft-128k-256xgpu.sh.
#
# Flag mapping (ms-swift v12 -> Relax):
#   tuner_type lora / rank32 / alpha64 / dropout0.05 / target_modules all-linear -> LORA_ARGS
#   freeze_vit true / freeze_llm false / freeze_aligner false                    -> LoRA scoped to LLM decoder
#                                                                                   (vision tower stays frozen)
#   mtp_num_layers 3 / mtp_shared_weights true / mtp_loss_scaling_factor 0.1     -> MTP_ARGS
#   TP=4 / EP=8 / CP=1 / PP=8 / ETP=1 (+ VPP=4 in v12)                           -> PERF_ARGS (see VPP note)
#   micro_batch 1 / global_batch 64                                             -> --global-batch-size 64 + dyn batch
#   max_length 131072                                                           -> --max-tokens-per-gpu 131072 (CP=1)
#   train_iters 1200                                                            -> --num-rollout 1200
#   lr 2e-5 / lr_warmup_fraction 0.05 / min_lr 1e-6 / cosine / wd 0.1           -> OPTIMIZER_ARGS
#   adam_beta1 0.9 / adam_beta2 0.95 (ms-swift default; NOT 0.98)               -> --adam-beta2 0.95
#   moe_aux_loss_coeff 1e-6 / use_precision_aware_optimizer / cross_entropy_fusion
#   optimizer_cpu_offload false (LoRA state is tiny)                            -> CPU-offload flags omitted
#   packing false / sequence_parallel / attention_backend flash / bf16
#
# Parallelism (actual, see PERF_ARGS): TP=4, PP=4, CP=2, EP=16, ETP=1.
#   TP*PP*CP = 4*4*2 = 32 GPU / replica, DP = 128/32 = 4.
#   Expert region TP*CP*DP = 4*2*4 = 32; EP*ETP*EDP = 16*1*2 = 32 (整除 OK).
#
# ── MTP + pipeline layout note ───────────────────────────────────────────────────────────────────
#   v12 ran VPP=4 with an explicit uneven layout string "E|tt|...|t|tmL" (the `m` = MTP layer).
#   Relax drives MTP through its own flags (--mtp-num-layers / --enable-mtp-training), and the proven
#   Relax MTP path (run-qwen3.5-397B-A17B-mtp-sft-128k-128xgpu.sh) uses an uneven decoder split with
#   VPP disabled. We default to that proven path (decoder-first=5 / decoder-last=1, VPP off). The
#   v12-faithful VPP=4 layout is provided below as opt-in (PIPELINE_LAYOUT_ARGS) — combining an explicit
#   layout string with the MTP flags is unvalidated in this Relax build, so validate before relying on it.
# ─────────────────────────────────────────────────────────────────────────────────────────────────
#
# ── Assumptions to validate on cluster ───────────────────────────────────────────────────────────
#   * MTP is intentionally disabled to match the committed MS 0625 launcher and v5 args (mtp_num_layers=null).
#   * tool-calling: the 0625 source is consumed through its pre-converted Qwen3.5 copy; source rows and images
#     are unchanged, while standalone role="tool_call" messages become assistant.tool_calls.
#   * "all-linear" LoRA is scoped to LLM decoder linears (qkv/proj/fc1/fc2) so the vision tower stays frozen;
#     confirm MoE-expert (grouped-GEMM) LoRA support — fall back to qkv/proj only if unsupported.
#   * ms-swift loss_scale=last_round+ignore_empty_think / enable_thinking=false have no 1:1 Relax knob;
#     Relax trains all assistant turns by default.
#   * Dataset has 26,334 rows; 1,200 steps at GBS=64 process 76,800 examples (~2.92 epochs), matching the
#     committed ms-swift launcher's train_iters/global_batch_size rather than the earlier 0730 memorization run.
# ─────────────────────────────────────────────────────────────────────────────────────────────────
#
# Usage:
#   bash scripts/training/sft/run-qwen3.5-397B-A17B-vl-lora-mtp-sft-128k-128xgpu.sh

set -ex
set -o pipefail

now=$(date "+%Y-%m-%d-%H:%M:%S")
echo "Current time: $now"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
# Auto-source local environment when not launched via an external entrypoint
if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    source "${SCRIPT_DIR}/../../entrypoint/local.sh"
fi
source "${MODEL_CONFIG_DIR}/qwen35-397B-A17B.sh"

PROJECT_NAME="${PROJECT_NAME:-Relax/sft/vl-lora-ms0625-align}"
# Use a fresh run name so an older MTP checkpoint cannot be resumed accidentally.
EXP_NAME="${EXP_NAME:-qwen3.5-397B-A17B-vl-lora-sft-task2-pp4-ep16-deepep-v1-gpu128}"

# Real on-cluster paths (model/data mirror the committed ms-swift launcher); all overridable via env.
HF_CKPT="${HF_CKPT:?Set HF_CKPT to the Qwen3.5-397B-A17B checkpoint directory}"
PROMPT_DATA="${PROMPT_DATA:?Set PROMPT_DATA to the converted task2 JSONL file}"
SAVE_DIR="${SAVE_DIR:?Set SAVE_DIR to the checkpoint output directory}"
LOAD_DIR="${LOAD_DIR:-${SAVE_DIR}/${EXP_NAME}}"
RAY_ADDRESS="${RAY_ADDRESS:-http://${HOST_IP:-127.0.0.1}:8265}"

CKPT_ARGS=(
   --hf-checkpoint ${HF_CKPT}
   --ref-load ${HF_CKPT}
   --megatron-to-hf-mode bridge
   --warm-hf-checkpoint-page-cache

   --load ${LOAD_DIR}
   --save ${SAVE_DIR}/${EXP_NAME}
   --save-interval ${SAVE_INTERVAL:-100}
   --max-actor-ckpt-to-keep 4
   # ms-swift train_iters=1200 @ GBS=64. --num-rollout pins the exact step count (~2.92 epochs on 26,334 rows).
   --num-rollout ${NUM_ROLLOUT:-1200}
)

# LoRA (ms-swift: rank=32 / alpha=64 / dropout=0.05 / target_modules=all-linear,
# freeze_vit=true / freeze_aligner=false). Scope=all injects both language and visual linears; the
# post-injection freeze regex keeps the vision tower frozen while leaving vision_model.merger LoRA trainable.
# "all-linear" over the LLM decoder = attention (linear_qkv/linear_proj) + GDN linear-attention
# (in_proj/out_proj) + MLP/shared/routed experts (linear_fc1/linear_fc2).
# NOTE: confirm the checkpoint has GDN layers (ms-swift sets SWIFT_USE_MCORE_GDN=1); drop in_proj/out_proj if not.
LORA_ARGS=(
   --lora-rank ${LORA_RANK:-32}
   --lora-alpha ${LORA_ALPHA:-64}
   --lora-scope all
   --lora-target-modules linear_qkv linear_proj in_proj out_proj linear_fc1 linear_fc2
   --lora-dropout ${LORA_DROPOUT:-0.05}
   --lora-merge-mode
   # Bridge applies this regex after LoRA injection. Freeze every visual parameter except the aligner/merger.
   --freeze-params-name-list '^(vision_model|visual)\.(?!merger\.)'
)

SFT_ARGS=(
   --loss-type sft
   --prompt-data "${PROMPT_DATA}"
   # MS: split_dataset_ratio=0.01, dataset_shuffle=true, data_seed=42. Relax holds out the split once
   # and automatically runs its deterministic dataset shuffle plus Megatron sampler shuffle each epoch.
   --eval-size ${EVAL_SIZE:-0.01}
   --input-key ${INPUT_KEY:-messages}
   # 完整 messages 模式 (不设 --label-key)
   --tool-key ${TOOL_KEY:-tools}
   # 多模态: 数据集 images 字段 -> image 媒体
   --multimodal-keys '{"image": "images"}'
   # 关键: Relax --image-max-token-num 默认 16384(!), 而 ms-swift IMAGE_MAX_TOKEN_NUM=1024。视觉编码器在
   # PP stage 0, 每图 16384 vision token 是巨量激活 -> 直接喂爆 stage 0 热点, 且与 baseline 不一致。
   # 1024(对齐 ms-swift) 后视觉 OOM 从 1.2GiB 降到仅 400MiB(仍在 vision_model forward, stage 0),
   # 只差 ~156MiB。再降到 256 (用户建议) 给足余量、决定性通过 step 0。文本/工具为主的语料视觉保真影响小,
   # 稳定后可回调 384/512 找平衡。(无 vit-gradient-checkpointing flag 可用。)
   --image-max-token-num ${IMAGE_MAX_TOKEN_NUM:-1024}
   --global-batch-size ${GLOBAL_BATCH_SIZE:-64}
   --use-dynamic-batch-size
   # 128K 上下文, CP=2: 每个 DP replica 容量 131072 token，单 CP rank 最多承载 65536 token。
   # 相比 CP4/DP2 保持相同的 128K 样本可见性，同时用 DP4 减少梯度累积并提高吞吐。
   --max-tokens-per-gpu ${MAX_TOKENS_PER_GPU:-65536}
   # 数据流水线预取深度。in-flight=2 (max_staleness=1) 下 producer 只领先 1 步, 而 producer 产一个
   # partition (~110-125s) ≈ compute (~110-117s), 势均力敌 -> 缓冲振荡: consumer 快(命中 108s)时
   # producer 备不出 N+1 -> 下步 miss (_agree_on_fetch 同步重取 ~20-30s), 出现命中/未命中交替
   # (108s / 138s, 均值 ~130s)。in-flight=3 (max_staleness=2) 让 producer 领先 2 步, 多一格缓冲吸收抖动,
   # 使 sft_{N+1} 在预取 RPC 时可靠已备好 -> 每步命中 -> 稳态 ~108s < ms 112s。TQ 存储按 max_staleness+1
   # 自动扩容 (controller.py:250)。built-in 单步预取仍只取 rollout_id+1 (消费者下一步), 与 in-flight 深度
   # 无关; 若 producer 领先致预取槽不匹配, _take 回退同步取数 (miss, 不崩)。
   --sft-max-in-flight-steps ${SFT_MAX_IN_FLIGHT_STEPS:-3}
   --sft-tq-timeout-minutes ${SFT_TQ_TIMEOUT_MINUTES:-60}
   # 生产者吞吐 (本轮性能主力): 实测稳态每步 train_get_data≈46s "empty meta, retrying" —— trainer 要
   # rollout N 时 TQ 里 sft_N 还没生产好。SFT producer 单 replica 渲染 64 样本 (含多模态图像解码),
   # 每 partition 耗时 110-128s ≈ compute 115s, 生产与训练势均力敌, prefetch 总是差一口气 (数据还没
   # 进 TQ)。--sft-prefetch-num-workers 是 PrefetchBuffer 内 I/O-bound 图像解码的线程数, 从 4 提到 16
   # 加大解码并行度, 让 partition 提前备好 -> prefetch 命中 -> 压掉 46s wait。buffer/chunk 同步加深保证
   # 流水线不空。纯数据侧, 零精度/显存风险 (producer 在 CPU-only replica)。
   --sft-prefetch-num-workers ${SFT_PREFETCH_NUM_WORKERS:-16}
   --sft-prefetch-buffer-size ${SFT_PREFETCH_BUFFER_SIZE:-512}
   --sft-prefetch-chunk-size ${SFT_PREFETCH_CHUNK_SIZE:-64}
   --balance-data
   --sft-oversize-strategy ${SFT_OVERSIZE_STRATEGY:-skip}
   --sft-invalid-multimodal-strategy skip
   --data-pad-size-multiplier 8192
   # ── ms-swift loss 对齐开关 (默认关, 这里显式开启以对齐 v12 baseline) ──────────────────────────
   # loss_scale=last_round: 每条 ~8 轮 assistant, ms 只训最后一轮。这是 iter1 差 0.5 (我们 1.53 vs
   # ms 0.987) 的主因。离线验证: 13 轮样本 loss token 5042->794。
   --sft-loss-last-turn-only
   # ignore_empty_think: 剔除空 <think></think> (模板注入的空 think, 每轮 ~3 token)。
   --sft-ignore-empty-think
   # MS also uses add_non_thinking_prefix=true. The empty think block is masked, but remains in the
   # attention context and therefore must be rendered for absolute first-step loss alignment.
   --apply-chat-template-kwargs '{"enable_thinking": false, "preserve_thinking": false, "add_non_thinking_prefix": true}'
   # 长序列 (49k-121k token) + vocab 248320 时, 最后一个 PP stage 会 materialize 满 [S,V] logits
   # 会造成 ~50GiB 级不均衡 OOM (其余 stage <30%)。--sft-chunked-logits 把
   # lm_head+CE 延迟进 loss 并按 chunk 计算, 不再一次性生成整块 logits。
   # 要求 untie-embeddings (已满足), 且与 --overlap-moe-expert-parallel-comm 互斥 (未启用)。
   --sft-chunked-logits
   --sft-logits-chunk-size ${SFT_LOGITS_CHUNK_SIZE:-1024}
)

# MTP intentionally disabled: MS 0625 v5 has mtp_num_layers=null and no MTP objective/metrics.
# MTP_ARGS=(
#    --mtp-num-layers ${MTP_NUM_LAYERS:-3}
#    --enable-mtp-training
#    --mtp-use-repeated-layer
#    --mtp-loss-scaling-factor ${MTP_LOSS_SCALING_FACTOR:-0.1}
# )

# 性能/并行参数: TP=4, PP=4, CP=2, EP=16, ETP=1，128 卡上 DP=4。
PERF_ARGS=(
   --tensor-model-parallel-size ${TP_SIZE:-4}
   --sequence-parallel
   --pipeline-model-parallel-size ${PP_SIZE:-4}
   --context-parallel-size ${CP_SIZE:-2}
   --expert-model-parallel-size ${EP_SIZE:-16}
   --expert-tensor-parallel-size ${ETP_SIZE:-1}
   # TP4/PP4/CP2/EP16 -> DP=4 on 128 GPUs. CP2 峰值由本轮严格对齐任务验证；若最长样本 OOM 则回退 CP4。
   # 历史 CP4/DP2 实测 (1/24/24/11) 各 PP stage 稳态峰值:
   #   pp0=46GiB(58%,1层但含vision+embedding+warmup) pp1=47(59%) pp2=47(59%) pp3=71GiB(90%!)
   # 旧 MTP 试验的 pp3 末段峰值来自全词表 logits + MTP head + loss；关闭 MTP 后仍保留余量。
   # 60 decoder layers across PP4: 2 / 20 / 20 / 18. With chunked logits, the last stage can safely
   # carry more decoder layers; this lowers the two middle stages from 27 to 20 layers each.
   --decoder-first-pipeline-num-layers ${DECODER_FIRST:-2}
   --decoder-last-pipeline-num-layers ${DECODER_LAST:-18}

   --log-probs-chunk-size 2048
   --recompute-loss-function

   # recompute: full/uniform/1 = 每层整层重算 (显存最省, 算力最贵)。ms v12 用 selective (只重算
# core_attn), 但 CP4 长序列下 selective 保留全部层激活仍有 OOM 风险，故保持 full recompute。full 下 step_time≈113.7s, 已基本
   # 追平 ms 112s/it, 且 wait 已被 prefetch 完全掩盖 (1.2s, ratio 1%)。
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --calculate-per-token-loss
   
   --freeze-vision-model
   --use-precision-aware-optimizer
   --cross-entropy-loss-fusion

   --moe-token-dispatcher-type ${MOE_TOKEN_DISPATCHER_TYPE:-flex}
   --moe-flex-dispatcher-backend ${MOE_FLEX_DISPATCHER_BACKEND:-deepep}

   # per-rank-fetch: 每步 wait 的 45s 几乎全是数据广播 (perf: tgd_bcast_pp ~20s + tgd_bcast_mm ~22s) —
   # 默认 rank0 pickle 多模态 pixel_values (几百 MB) 再 PP/TP 广播。开启后每个 TP/PP rank 并行自取,
   # 消除 rank0 pickle + 广播。与 rollout_routed_experts 不兼容 (SFT 无此字段, 自动生效)。
   # 要求 --num-data-storage-units >= TP world size (16 >= 4 ✓)。
   --per-rank-fetch
   # 在当前 step 计算时，让每 rank 在 CPU 后台预取下一 step 的 get_meta/get_data；TP/CP/PP 一致性
   # 检查和 GPU 搬运仍留在主训练线程，不改变 collective 或 batch 顺序。in-flight=3 提供 next partition。
   --sft-train-data-prefetch

   # 视觉编码器在 PP stage 0, 是该 stage 热点的一大来源。vision-dp-when-tp 把视觉计算按 TP rank 数据并行
   # 切分 (各 TP rank 处理部分图, 再 all-reduce 汇总 embedding), 降低单卡视觉激活。loss 数学等价。
   # (--vision-dp-when-cp 此 build 不存在, 仅 -when-tp。)
   --vision-dp-when-tp
)

# opt-in: v12-faithful VPP=4 显式布局 (embedding chunk + 60 层 + 末尾 mtp/loss)。
# 若启用: 追加 "${PIPELINE_LAYOUT_ARGS[@]}" 到 train 命令, 并从 PERF_ARGS 移除 decoder-first/last。
# 注意: 显式 layout 与 MTP flags 组合在当前 Relax 构建未验证 (layout 中的 `m` 语义 vs --mtp-num-layers)。
# PIPELINE_LAYOUT_ARGS=(
#    --virtual-pipeline-model-parallel-size 4
#    --pipeline-model-parallel-layout "E|tt|tt|tt|tt|tt|tt|tt|tt|tt|tt|tt|tt|tt|tt|tt|tt|tt|tt|tt|tt|tt|tt|tt|tt|tt|tt|tt|tt|tt|t|tmL"
# )

# 优化器参数 (对齐 ms-swift v12): lr=2e-5, warmup_fraction=0.05, cosine, min-lr=1e-6, beta2=0.95.
# LoRA optimizer state 极小, 不启用 CPU offload (ms-swift optimizer_cpu_offload=false)。
OPTIMIZER_ARGS=(
   --optimizer adam
   --lr ${LR:-2e-5}
   --lr-decay-style cosine
   --lr-warmup-fraction ${LR_WARMUP_FRACTION:-0.05}
   --min-lr ${MIN_LR:-1e-6}
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.95
   --clip-grad 1.0

   # --use-precision-aware-optimizer

   --no-rope-fusion
   --moe-aux-loss-coeff 1e-6
)

WANDB_ARGS=(
   --use-clearml
   --use-metrics-service
   --tb-project-name ${PROJECT_NAME}
   --tb-experiment-name ${EXP_NAME}-${now}
)

MISC_ARGS=(
   --seed ${SEED:-42}
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   --use-health-check
)

RUNTIME_ENV_JSON=$(python3 -c '
import json, os
d = json.loads(os.environ["RUNTIME_ENV_JSON"])
d.setdefault("env_vars", {}).update({
    "TORCH_DIST_INIT_BARRIER": "1",
    "TORCH_NCCL_BLOCKING_WAIT": "0",
    "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
    "TORCH_DISTRIBUTED_DEFAULT_TIMEOUT": "3600",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "256",
    # MS PEFT uses Kaiming-uniform LoRA A initialization and distinct adapters for every routed expert.
    # Bridge defaults to Xavier + one adapter shared by local experts; that leaves step-0 loss unchanged but
    # changes grad_norm and the complete optimization trajectory.
    "RELAX_LORA_A_INIT_METHOD": "kaiming",
    # MS injects LoRA into language + merger only. Exclude the frozen vision tower before PEFT injection;
    # freezing its adapter parameters after injection still executes dropout and advances training RNG.
    "RELAX_LORA_EXCLUDE_FROZEN_MODULES": "true",
    "RELAX_LORA_SHARE_EXPERT_ADAPTERS": "false",
    # The Ray job driver receives the variables above from this runtime env; explicitly forward them again
    # to the nested Ray Serve / Megatron actors created by relax.entrypoints.train.
    "RELAX_PROPAGATE_ENV_VARS": "RELAX_LORA_A_INIT_METHOD,RELAX_LORA_EXCLUDE_FROZEN_MODULES,RELAX_LORA_SHARE_EXPERT_ADAPTERS",
    # NVLS (NVLink SHARP) 在本集群会在首个 all_reduce 崩溃 (nvls.cc Cuda failure 999);
    # ms-swift baseline 也显式 export NCCL_NVLS_ENABLE=0。覆盖 ray-job.sh 的 HAS_NVLINK 自动值。
    "NCCL_NVLS_ENABLE": "0",
    # 长序列 (单样本可达 121k token, CP=1) 在 MoE linear_fc2 处 OOM, 且有 ~12GiB reserved-but-unallocated
    # 碎片。ms-swift v12 baseline 用 expandable_segments:True 且同为 TP=4 峰值 ~98GiB。对齐之。
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    # 某些节点 NCCL/NVSHMEM(DeepEP) bootstrap 未固定网卡时会连到公网 IP (misc/socket.cc
    # "Connection closed by remote peer <public IP>") 导致集合通信 hang。ms-swift baseline 也 export
    # NCCL_SOCKET_IFNAME=eth0 (其日志 Bootstrap: Using eth0:10.144.x)。固定到内网 eth0。
    "NCCL_SOCKET_IFNAME": "eth0",
    "NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME": "eth0",
})
print(json.dumps(d))
')
export RUNTIME_ENV_JSON

mkdir -p log

# MTP is disabled above; intentionally do not append "${MTP_ARGS[@]}" to this command.
ray job submit ${RAY_NO_WAIT:+--no-wait} --address="${RAY_ADDRESS}" \
   ${WORKING_DIR:+--working-dir "${WORKING_DIR}"} \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 -m relax.entrypoints.train \
   --resource '{"sft": [1, 0], "actor": [1, 128]}' \
   --max-staleness 0 \
   --num-data-storage-units 16 \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${LORA_ARGS[@]}" \
   "${SFT_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${WANDB_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${MISC_ARGS[@]}"  2>&1 | tee log/${EXP_NAME}-${now}.log
