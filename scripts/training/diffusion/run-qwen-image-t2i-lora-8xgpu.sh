#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Qwen-Image T2I 8xGPU colocate native-generation RL training script, LoRA variant
# (FlowGRPO + FSDP2 + PEFT + SGLang native diffusion).
#
# STATUS: this is the VALIDATED adapter-sync recipe. It has a completed
# 100-rollout alignment run (final eval 0.8643 vs the reference baseline's
# 0.8620). Prefer it over the full-FT sibling unless you specifically need
# full-parameter updates.
#
# Sibling of run-qwen-image-t2i-8xgpu.sh. Kept separate rather than branched into
# it because the optimizer settings are not shared: the full-FT script's
# --lr 3e-5 / --adam-eps 1e-15 were bracketed empirically for full-parameter
# updates, and a rank-64 adapter needs roughly an order of magnitude more lr
# (the reference qwen_image LoRA recipe runs 3e-4). Where a setting IS shared, the
# full-FT script carries the long-form rationale and this one cross-references
# it; the two must not disagree.
#
# WHAT LORA BUYS: memory, not bandwidth. Freezing the base removes its gradients
# and AdamW moments (~25 GB/rank on a 20B DiT), which is what lets you raise
# N_SAMPLES and the number of trained SDE steps. In merge mode the weight sync
# still streams the whole transformer every step; this script
# defaults to ADAPTER_MODE=1 to stream only the adapter through the combined
# SGLang patch. Set ADAPTER_MODE=0 only when validating merge-mode sync itself.
#
# Usage:
#   bash scripts/training/diffusion/run-qwen-image-t2i-lora-8xgpu.sh
#   ADAPTER_MODE=0 bash scripts/training/diffusion/run-qwen-image-t2i-lora-8xgpu.sh

set -ex
set -o pipefail

now=$(date "+%Y-%m-%d-%H:%M:%S")
echo "当前时间: $now"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
# Auto-source local environment when not launched via an external entrypoint
if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    source "${SCRIPT_DIR}/../../entrypoint/local.sh"
fi

PROJECT_NAME="${PROJECT_NAME:=Relax/dev/native-generation}"
# Experiment root. Override per-user/per-cluster, e.g.
#   EXP_DIR=/your/shared/fs/native-generation bash scripts/training/diffusion/run-qwen-image-t2i-lora-8xgpu.sh
# Defaults to a repo-local directory so the script has no machine-specific path.
EXP_DIR="${EXP_DIR:-${SCRIPT_DIR}/../../../exps/native-generation}"
MODEL_DIR="${MODEL_DIR:-${EXP_DIR}/models}"
DATA_DIR="${DATA_DIR:-${EXP_DIR}/data}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-${EXP_DIR}/artifacts/t2i-lora}"
NUM_ROLLOUT="${NUM_ROLLOUT:=10000}"
ROLLOUT_GPUS="${ROLLOUT_GPUS:-8}"
ROLLOUT_GPUS_PER_ENGINE="${ROLLOUT_GPUS_PER_ENGINE:-1}"
SGLANG_ATTENTION_BACKEND="${SGLANG_ATTENTION_BACKEND-}"

MODEL_ARGS=(
   --train-backend fsdp
   --generation-task t2i
   --model-path ${MODEL_DIR}/Qwen-Image
   --model-adapter-path relax.models.qwen_image.adapter.QwenImageAdapter
   --rollout-engine-class-path relax.backends.sglang.diffusion_engine.SGLangNativeGenerationEngine
   --rollout-function-path relax.engine.rollout.native_generation.generate_rollout
   --custom-convert-samples-to-train-data-path relax.engine.rollout.native_generation.convert_samples_to_train_data
   --custom-reward-post-process-path relax.engine.rewards.generative.post_process
)

CKPT_ARGS=(
   --save ${SAVE_DIR:-${EXP_DIR}/runs/qwen-image-t2i-lora}/
   --save-interval ${SAVE_INTERVAL:-20}
   # A LoRA checkpoint stores ONLY the adapter (the base is reloaded from
   # --model-path on resume), so it is under 1 GB instead of the full-FT ~115 GB.
   # Retention can therefore be much looser. Retention globs iter_* under the save
   # dir, so give each experiment its own SAVE_DIR.
   --max-actor-ckpt-to-keep ${KEEP_CKPT:-10}
)

# Prompt sets. Same Pick-a-Pic alignment split as the full-FT script
# (pickapic_v1 unique captions filtered to >=6 words, 2048 held out as test);
# see that script for why the unfiltered captions starve GRPO of signal. Produce these exact
# files with:
#   python3 examples/diffusion/curate_data.py --input <unified.jsonl> \
#       --out-dir ${DATA_DIR}/processed/t2i --prefix pickapic_ \
#       --min-prompt-words 6 --eval-size 2048 --eval-subset-sizes 64 256 \
#       --no-check-media
PROMPT_SET="${PROMPT_SET:-${DATA_DIR}/processed/t2i/pickapic_train.jsonl}"
# 256 held-out prompts is the number the published comparison was measured on;
# pickapic_eval64.jsonl is the ~2 min variant for short diagnostic runs.
EVAL_SET="${EVAL_SET:-${DATA_DIR}/processed/t2i/pickapic_eval256.jsonl}"
N_PROMPTS="${N_PROMPTS:-32}"
N_SAMPLES="${N_SAMPLES:-8}"
ADVANTAGE_STD_MODE="${ADVANTAGE_STD_MODE:-group}"
REFERENCE_DATALOADER_ORDER="${REFERENCE_DATALOADER_ORDER:-1}"
REFERENCE_SHUFFLE_SEED="${REFERENCE_SHUFFLE_SEED:-42}"

if [ "${REFERENCE_DATALOADER_ORDER}" = "1" ]; then
   # Reference alignment uses torch DataLoader(shuffle=True, generator=manual_seed(42),
   # drop_last=True). Materialize that first-epoch order into a prompt manifest
   # and keep Relax rollout_shuffle disabled, so metadata/sample_id fields still
   # point back to the original prompt rows.
   ORDERED_PROMPT_SET="${ARTIFACT_ROOT}/prompt_order/reference_dataloader_seed${REFERENCE_SHUFFLE_SEED}_batch${N_PROMPTS}.jsonl"
   mkdir -p "$(dirname "${ORDERED_PROMPT_SET}")"
   python3 - "${PROMPT_SET}" "${ORDERED_PROMPT_SET}" "${N_PROMPTS}" "${REFERENCE_SHUFFLE_SEED}" <<'PY'
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
batch_size = int(sys.argv[3])
seed = int(sys.argv[4])

lines = src.read_text(encoding="utf-8").splitlines()
loader = DataLoader(
    TensorDataset(torch.arange(len(lines))),
    batch_size=batch_size,
    shuffle=True,
    generator=torch.Generator().manual_seed(seed),
    num_workers=0,
    drop_last=True,
)
indices = []
for (batch_indices,) in loader:
    indices.extend(int(i) for i in batch_indices.tolist())

tmp = dst.with_suffix(dst.suffix + ".tmp")
with tmp.open("w", encoding="utf-8") as f:
    for idx in indices:
        f.write(lines[idx])
        f.write("\n")
tmp.replace(dst)
print(
    f"Wrote reference DataLoader prompt order to {dst} "
    f"({len(indices)}/{len(lines)} prompts, first_indices={indices[:8]})"
)
PY
   PROMPT_SET="${ORDERED_PROMPT_SET}"
fi

ROLLOUT_ARGS=(
   --prompt-data ${PROMPT_SET}
   --input-key prompt
   --artifact-root ${ARTIFACT_ROOT}

   --num-rollout ${NUM_ROLLOUT}
   # FlowGRPO group geometry, matching the reference launch this run
   # reproduces: 32 prompts x 8 candidates = a 256-sample global batch. One
   # optimizer step covers the WHOLE global batch (each prompt group is a
   # micro-batch scaled by 1/num_groups and accumulated), firing once per PPO
   # mini-epoch -- see --num-updates-per-batch.
   --rollout-batch-size ${N_PROMPTS}
   --n-samples-per-prompt ${N_SAMPLES}
   --rollout-seed ${REFERENCE_SHUFFLE_SEED}

   --global-batch-size $(( ${N_PROMPTS} * ${N_SAMPLES} ))
   # At n=8 over dp8 this rank owns 1 candidate per group, so there is nothing
   # left to sub-split; see the full-FT script for why you otherwise want this
   # as LARGE as memory allows (every extra micro-batch is another FSDP
   # all-gather and reduce-scatter for identical flops).
   --micro-batch-size 1
)

EVAL_ARGS=(
   # Held-out PickScore eval. The rollout loop runs one pass at step 0 BEFORE the
   # first weight update (components/rollout.py: `step == 0 and not
   # skip_eval_before_train`), then every --eval-interval rollouts, so the
   # base-vs-RL delta is measured by one harness with one set of seeds.
   --eval-prompt-data pickscore ${EVAL_SET}
   --eval-interval ${EVAL_INTERVAL:-10}
   --n-samples-per-eval-prompt ${EVAL_N:-2}
   # SKIP_EVAL0=1 drops the step-0 baseline pass (~14 min on 256x2). Only for
   # short diagnostic runs -- a real run needs it as the before/after reference.
   ${SKIP_EVAL0:+--skip-eval-before-train}
)

SAMPLING_CONFIG="{\"height\":${RES:-384},\"width\":${RES:-384},\"num_inference_steps\":12,\"guidance_scale\":1.0,\"eta\":0.7,\"sde_type\":\"${SDE_TYPE:-sde}\",\"sde_indices\":${SDE_IDX:-[3,4,5]},\"sde_resample_per_rollout\":${SDE_RESAMPLE:-true},\"num_sde_steps\":3,\"sde_pool\":${SDE_POOL:-[1,2,3,4,5]},\"driver_xt\":${DRIVER_XT:-true},\"sample_id_mode\":\"${SAMPLE_ID_MODE:-metadata}\"}"

SAMPLING_ARGS=(
   # Same geometry as the reference trainside Qwen-Image run: FlowGRPO, 12 steps,
   # eta=0.7, and three SDE steps sampled from the first half of the schedule.
   # SDE steps are REDRAWN every rollout from default_rng(rollout_id) over
   # sde_pool -- the reference scheduler's AllSDEScheduler.get_sde_indices, which is what this run
   # is reproducing. These indices decide both where exploration noise is injected
   # at generation time and which steps receive gradient, so a fixed set trains 3
   # of 12 steps forever. In SGLang/Relax, the standard FlowSDEStrategy is named
   # "sde"; "dance" is only an override for DanceGRPO experiments.
   #
   # The pool EXCLUDES step 0, exactly as in the full-FT script (the reference pool
   # includes it). At index 0 sigma == 1, so sqrt(sigma/(1-sigma)) is singular
   # and only the arbitrary sigma_max clamp keeps it finite: the replayed
   # Gaussian is then not the one the sample was drawn from (2.4x off on the std
   # and a sign flip on the prev_sample_mean 'sample' coefficient), so the
   # gradient on the most influential trained step is wrong. Preflight rejects a
   # trained step 0. Dropping 0 does not perturb this recipe: default_rng(0)'s
   # first draw over [1,2,3,4,5] is still [3,4,5], the deterministic eval
   # fallback below, so eval step 0 remains comparable to the reference baseline.
   --sampling-config "${SAMPLING_CONFIG}"
   --generation-seed 42
)

LORA_ARGS=(
   # rank/alpha from the reference qwen_image recipe (scale = alpha/r = 2.0).
   --lora-rank ${LORA_RANK:-64}
   --lora-alpha ${LORA_ALPHA:-128}
   # Must stay 0. The FlowGRPO pi_old anchor is a REPLAY through the live model
   # under model.train(); a stochastic forward makes the anchor differ from the
   # update, the ratio drifts off 1 on an on-policy step, and at --eps-clip 1e-4
   # essentially everything clips — the run trains and goes nowhere, silently.
   # Preflight rejects a nonzero value.
   --lora-dropout 0.0
   # Keep the adapter's eight Qwen-Image attention projections explicit here so
   # launch-time validation does not depend on backend-deferred defaults.
   --lora-target-modules \
      attn.to_q attn.to_k attn.to_v attn.to_out.0 \
      attn.add_q_proj attn.add_k_proj attn.add_v_proj attn.to_add_out
   #
   # Rollout path. adapter (default): stream only LoRA tensors. merge folds B@A
   # into the base trainer-side and reuses the native-diffusion full-weight
   # transport. Adapter sync is the validated path and is ~100x smaller. The two
   # modes must never be mixed: once SGLang converts to LoRA
   # layers its DiT parameters are renamed and a full-weight sync silently drops
   # every tensor.
   $([ "${ADAPTER_MODE:-1}" = "1" ] && echo "--lora-adapter-mode" || echo "--lora-merge-mode")
)

FSDP_ARGS=(
   --fsdp-trainable-mode lora
   --fsdp-trainable-attr transformer
   ${DEBUG_FP:+--fsdp-debug-fingerprint}
   --fsdp-param-dtype bf16
   --fsdp-reduce-dtype fp32
   # fp32 optimizer masters for the adapter only (the frozen base stays bf16, and
   # --fsdp-param-dtype still governs the all-gathered compute copy, so the forward
   # is numerically identical to full FT). Without this the ~1e-4 relative updates
   # round off the bf16 mantissa: grad norm and loss stay healthy while the policy
   # stops moving — the same symptom as the lr=1e-5 FROZEN run documented in the
   # full-FT script, and easily misdiagnosed as an lr problem.
   --fsdp-master-dtype fp32
   # resharding after forward is the default; opt out with
   # --no-fsdp-reshard-after-forward if you would rather trade memory for speed.
   --fsdp-activation-checkpointing
   # Load the 20B base model 2 ranks at a time so 8 colocated ranks don't each
   # allocate a full CPU copy + hammer the shared model dir at once (launch OOM/hang).
   --fsdp-load-wave-size 2
   # Colocate time-share: offload the actor (model + optimizer) to CPU during
   # rollout so the diffusion engine has the GPU to generate; wake it for training
   # + weight sync. --fsdp-cpu-offload is NOT used and is rejected at preflight for
   # LoRA: every FSDP collective over CPU-offloaded DTensors (clip_grad_norm_)
   # fails with "No backend type associated with device type cpu". Freezing the
   # base already removes its grads and AdamW moments (~25 GB/rank), which was the
   # binding constraint in the full-FT script.
   --offload-train
   # Offload the diffusion engine's movable modules to CPU during training (the
   # engine also self-offloads dit/text_encoder). Weight sync onloads only the
   # engine transformer, then fully reloads before the next rollout.
   --offload-rollout
)

GRPO_ARGS=(
   --advantage-estimator grpo
   # FlowGRPO uses a tiny PPO clip range (reference clip_range=1e-4), not the text-RL 0.2.
   --eps-clip 1e-4
   # Explicit reward-scale semantics. `group` matches canonical Relax/Text GRPO;
   # set ADVANTAGE_STD_MODE=batch for the reference diffusion global-std divisor,
   # or none for Dr.GRPO.
   --generative-advantage-std-mode ${ADVANTAGE_STD_MODE}
   # PPO mini-epochs per rollout batch (reference num_updates_per_batch=2). With 1 the
   # ratio is always 1 -> the clipped loss is 0 by construction (group-centered
   # advantages) and clipping never engages; 2 makes the 2nd update diverge from
   # the frozen old_logp, giving a nonzero loss and real PPO clipping.
   --num-updates-per-batch 2
)

OPTIMIZER_ARGS=(
   # 3e-4 is the reference qwen_image LoRA lr. It is ~10x the full-FT 3e-5 because a
   # rank-64 adapter starts from B=0 and has far fewer degrees of freedom; the
   # quantity to watch is still the logp ratio (target ~7e-4 so the second PPO
   # mini-epoch clips partially rather than totally), not the parameter delta.
   --lr ${LR:-3e-4}
   # The reference run uses warmup_steps: 0 and adam_epsilon: 1.0e-8, and its
   # reward curve climbs monotonically (0.8152 -> 0.8620 over 100 rollouts,
   # measured on this node). Both values are therefore matched EXACTLY rather than
   # carried over from the full-FT script, whose 1e-15 eps was bracketed for a
   # different optimization problem (20B dense params, where eps dominated the
   # AdamW denominator). Deviating from a recipe that is known to converge, on a
   # run whose whole purpose is to reproduce that convergence, would make any
   # difference in the curve un-attributable.
   --lr-warmup-iters ${LR_WARMUP:-0}
   --adam-eps ${ADAM_EPS:-1e-8}
   --weight-decay ${WEIGHT_DECAY:-0.0}
)

REWARD_ARGS=(
   # PickScore (CLIP-H, ~2B). `colocate` scores in-process on the rollout GPU
   # while the actor is offloaded (see generative._get_manager); the ~5.6 GB it
   # needs fits comfortably. On CPU it dominates post-processing -- measured
   # ~104 s/rollout for reward+convert+TQ at 256 samples.
   --reward-runtime ${REWARD_RUNTIME:-colocate}
   --reward-scorer-path relax.engine.rewards.pickscore.PickScoreScorer
   --reward-model-path ${MODEL_DIR}/PickScore_v1
   --reward-required-components '["pickscore"]'
   --reward-component-weights '{"pickscore":1.0}'
)

WEIGHT_SYNC_ARGS=(
   # weight-sync-mode is derived from the LoRA rollout path at preflight
   # (adapter mode forces 'adapter'), so it is not set here.
   --weight-sync-wire-dtype bf16
   # Bucket size drives the RPC COUNT, and a merge-mode sync's cost is almost
   # entirely per-RPC overhead rather than payload: the [wsync time] breakdown
   # measured gather=0.5s / serialize=0.3s / transport=38.8s for 40.9 GB, i.e.
   # ~0.5 s per bucket to open an on-device CUDA-IPC handle. 512MB = 80 buckets,
   # 2048MB = 20. The bucket is materialized on GPU, so this also adds ~1.5 GB to
   # the transient sync footprint. Largely moot under ADAPTER_MODE=1, which ships
   # ~377 MB instead of the whole transformer.
   --weight-sync-bucket-size-mb ${WSYNC_BUCKET_MB:-2048}
)

SGLANG_CONFIG_PATH="${SGLANG_CONFIG_PATH:-${ARTIFACT_ROOT}/sglang-rollout-${now}.yaml}"
mkdir -p "$(dirname "${SGLANG_CONFIG_PATH}")"
if [ -n "${SGLANG_ATTENTION_BACKEND}" ]; then
   cat > "${SGLANG_CONFIG_PATH}" <<EOF
sglang:
  - name: default
    num_gpus_per_engine: ${ROLLOUT_GPUS_PER_ENGINE}
    engine_groups:
      - worker_type: regular
        num_gpus: ${ROLLOUT_GPUS}
        num_gpus_per_engine: ${ROLLOUT_GPUS_PER_ENGINE}
        overrides:
          attention_backend: ${SGLANG_ATTENTION_BACKEND}
EOF
else
   cat > "${SGLANG_CONFIG_PATH}" <<EOF
sglang:
  - name: default
    num_gpus_per_engine: ${ROLLOUT_GPUS_PER_ENGINE}
    engine_groups:
      - worker_type: regular
        num_gpus: ${ROLLOUT_GPUS}
        num_gpus_per_engine: ${ROLLOUT_GPUS_PER_ENGINE}
EOF
fi

VALIDATION_RECIPE_PATH="${VALIDATION_RECIPE_PATH:-${ARTIFACT_ROOT}/recipes/qwen-image-t2i-lora-${now}.json}"
mkdir -p "$(dirname "${VALIDATION_RECIPE_PATH}")"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
GIT_REV="$(git -C "${REPO_ROOT}" rev-parse HEAD 2>/dev/null || echo unknown)"
RELAX_VALIDATION_TIMESTAMP="${now}" \
RELAX_VALIDATION_GIT_REV="${GIT_REV}" \
RELAX_VALIDATION_PROMPT_SET="${PROMPT_SET}" \
RELAX_VALIDATION_EVAL_SET="${EVAL_SET}" \
RELAX_VALIDATION_N_PROMPTS="${N_PROMPTS}" \
RELAX_VALIDATION_N_SAMPLES="${N_SAMPLES}" \
RELAX_VALIDATION_ADVANTAGE_STD_MODE="${ADVANTAGE_STD_MODE}" \
RELAX_VALIDATION_ADAPTER_MODE="${ADAPTER_MODE:-1}" \
RELAX_VALIDATION_REFERENCE_ORDER="${REFERENCE_DATALOADER_ORDER}" \
RELAX_VALIDATION_REFERENCE_SEED="${REFERENCE_SHUFFLE_SEED}" \
RELAX_VALIDATION_SAMPLING_CONFIG="${SAMPLING_CONFIG}" \
RELAX_VALIDATION_SGLANG_CONFIG="${SGLANG_CONFIG_PATH}" \
RELAX_VALIDATION_MODEL_PATH="${MODEL_DIR}/Qwen-Image" \
RELAX_VALIDATION_REWARD_MODEL_PATH="${MODEL_DIR}/PickScore_v1" \
python3 - "${VALIDATION_RECIPE_PATH}" <<'PY'
import json
import os
import sys
from pathlib import Path

recipe = {
    "script": "scripts/training/diffusion/run-qwen-image-t2i-lora-8xgpu.sh",
    "timestamp": os.environ["RELAX_VALIDATION_TIMESTAMP"],
    "git_rev": os.environ["RELAX_VALIDATION_GIT_REV"],
    "model_path": os.environ["RELAX_VALIDATION_MODEL_PATH"],
    "reward_model_path": os.environ["RELAX_VALIDATION_REWARD_MODEL_PATH"],
    "prompt_set": os.environ["RELAX_VALIDATION_PROMPT_SET"],
    "eval_set": os.environ["RELAX_VALIDATION_EVAL_SET"],
    "n_prompts": int(os.environ["RELAX_VALIDATION_N_PROMPTS"]),
    "n_samples_per_prompt": int(os.environ["RELAX_VALIDATION_N_SAMPLES"]),
    "reference_dataloader_order": os.environ["RELAX_VALIDATION_REFERENCE_ORDER"] == "1",
    "reference_shuffle_seed": int(os.environ["RELAX_VALIDATION_REFERENCE_SEED"]),
    "generation_seed": 42,
    "advantage_std_mode": os.environ["RELAX_VALIDATION_ADVANTAGE_STD_MODE"],
    "lora_adapter_mode": os.environ["RELAX_VALIDATION_ADAPTER_MODE"] == "1",
    "sampling_config": json.loads(os.environ["RELAX_VALIDATION_SAMPLING_CONFIG"]),
    "sglang_config": os.environ["RELAX_VALIDATION_SGLANG_CONFIG"],
    "sglang_patch": "docker/patch/latest/sglang.patch",
    "reward_scorer": "relax.engine.rewards.pickscore.PickScoreScorer",
}

path = Path(sys.argv[1])
path.write_text(json.dumps(recipe, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(f"Wrote validation recipe manifest to {path}")
PY

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine ${ROLLOUT_GPUS_PER_ENGINE}
   --sglang-config ${SGLANG_CONFIG_PATH}
)

WANDB_ARGS=(
   --use-clearml
   --use-metrics-service
   --tb-project-name ${PROJECT_NAME}
   --tb-experiment-name qwen-image-t2i-lora-gpu8-${now}
)

mkdir -p log
ray job submit ${RAY_NO_WAIT:+--no-wait} --address="http://127.0.0.1:8265" \
   ${WORKING_DIR:+--working-dir "${WORKING_DIR}"} \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 -m relax.entrypoints.train \
   --resource "{\"actor\": [1, ${ROLLOUT_GPUS}], \"rollout\": [1, ${ROLLOUT_GPUS}]}" \
   --colocate \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${EVAL_ARGS[@]}" \
   "${SAMPLING_ARGS[@]}" \
   "${LORA_ARGS[@]}" \
   "${FSDP_ARGS[@]}" \
   "${GRPO_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${REWARD_ARGS[@]}" \
   "${WEIGHT_SYNC_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${WANDB_ARGS[@]}" 2>&1 | tee log/qwen-image-t2i-lora-gpu8-${now}.log

exit ${PIPESTATUS[0]}
