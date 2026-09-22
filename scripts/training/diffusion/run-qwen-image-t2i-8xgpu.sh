#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Qwen-Image T2I 8xGPU colocate native-generation RL training script
# (FlowGRPO + FSDP2 + SGLang native diffusion). Mirrors
# scripts/training/text/run-qwen3-4B-8xgpu.sh; Megatron args are replaced with
# the FSDP / diffusion equivalents.
#
# STATUS: full fine-tune reference. The VALIDATED 100-rollout recipe is the
# sibling `run-qwen-image-t2i-lora-8xgpu.sh` in its default adapter-sync mode.
# This script is where the empirically bracketed FULL-FT settings (lr, adam-eps, offload,
# weight-sync bucket size) are documented; the LoRA script keeps the alignment
# defaults and calls out each full-FT divergence.
#
# Usage:
#   bash scripts/training/diffusion/run-qwen-image-t2i-8xgpu.sh

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
#   EXP_DIR=/your/shared/fs/native-generation bash scripts/training/diffusion/run-qwen-image-t2i-8xgpu.sh
# Defaults to a repo-local directory so the script has no machine-specific path.
EXP_DIR="${EXP_DIR:-${SCRIPT_DIR}/../../../exps/native-generation}"
MODEL_DIR="${MODEL_DIR:-${EXP_DIR}/models}"
DATA_DIR="${DATA_DIR:-${EXP_DIR}/data}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-${EXP_DIR}/artifacts/t2i}"
NUM_ROLLOUT="${NUM_ROLLOUT:=10000}"

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
   --save ${SAVE_DIR:-${EXP_DIR}/runs/qwen-image-t2i}/
   --save-interval ${SAVE_INTERVAL:-20}
   # A full-FT 20B DCP checkpoint (weights + fp32 AdamW state) is ~115 GB, so an
   # unbounded run fills the FS in a handful of saves. Retention globs iter_* under
   # the save dir, so give each experiment its own SAVE_DIR — otherwise this prunes
   # a previous run's checkpoints too.
   --max-actor-ckpt-to-keep ${KEEP_CKPT:-2}
)

# Prompt sets. Default to the Pick-a-Pic alignment split (pickapic_v1 unique
# captions filtered to >=6 words, 2048 held out as test). Unfiltered one-word
# captions score with almost no intra-group variance and starve GRPO of signal.
# Produce these exact files with:
#   python3 examples/diffusion/curate_data.py --input <unified.jsonl> \
#       --out-dir ${DATA_DIR}/processed/t2i --prefix pickapic_ \
#       --min-prompt-words 6 --eval-size 2048 --eval-subset-sizes 64 256 \
#       --no-check-media
PROMPT_SET="${PROMPT_SET:-${DATA_DIR}/processed/t2i/pickapic_train.jsonl}"
# Held-out eval set. 64 prompts keeps a pass to ~2 min; use
# pickapic_eval256.jsonl / the full pickapic_eval.jsonl for the final before/after number.
EVAL_SET="${EVAL_SET:-${DATA_DIR}/processed/t2i/pickapic_eval64.jsonl}"
N_PROMPTS="${N_PROMPTS:-8}"
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
   # FlowGRPO group geometry. The actor forwards a whole group at once (one
   # micro-batch == one group of n candidates, stacked on dim 0). Full-FT caps n
   # by memory, but activation checkpointing
   # (recompute) collapses the n-dependent activation cost to ~block-inputs, so
   # n=8 fits the 96GB card (vs the bare GRPO minimum of 2, whose group_std_mean
   # was only ~0.01 — too weak to learn from). Raise N_SAMPLES when memory allows.
   # global-batch-size = rollout-batch-size * n-samples-per-prompt = 64.
   # One optimizer step now covers this WHOLE global batch: each prompt group is a
   # micro-batch whose loss is scaled by 1/num_groups and accumulated, and the step
   # fires once per PPO mini-epoch (see --num-updates-per-batch).
   --rollout-batch-size ${N_PROMPTS}
   --n-samples-per-prompt ${N_SAMPLES}
   --rollout-seed ${REFERENCE_SHUFFLE_SEED}

   --global-batch-size $(( ${N_PROMPTS} * ${N_SAMPLES} ))
   # Replay sub-batching. This rank owns n-samples-per-prompt / dp_world candidates
   # of each group; MICRO_BATCH splits that slice further (it must divide it evenly
   # or the split is skipped, see _candidate_chunks). Keep it as LARGE as memory
   # allows: the flops are identical either way, but every extra micro-batch is
   # another FSDP all-gather round and another 20B reduce-scatter. At n=16/dp8 the
   # slice is 2, so 1 doubles the micro-batch count for nothing -- measured
   # mem_peak_gb 63.6/96 at B=1, so B=2 has ample headroom. Drop back to 1 only if
   # mem_peak_gb approaches the card limit.
   --micro-batch-size ${MICRO_BATCH:-2}
)

EVAL_ARGS=(
   # Held-out PickScore eval. The rollout loop runs one pass at step 0 BEFORE the
   # first weight update (components/rollout.py: `step == 0 and not
   # skip_eval_before_train`), then every --eval-interval rollouts, so the
   # base-vs-RL delta is measured by one harness with one set of seeds — no
   # separate before/after generation pass to keep in sync.
   --eval-prompt-data pickscore ${EVAL_SET}
   --eval-interval ${EVAL_INTERVAL:-10}
   --n-samples-per-eval-prompt ${EVAL_N:-2}
   # SKIP_EVAL0=1 drops the step-0 baseline pass (~14 min on 256x2). Only for
   # short diagnostic runs — a real run needs it as the before/after reference.
   ${SKIP_EVAL0:+--skip-eval-before-train}
)

SAMPLING_ARGS=(
   # Qwen-Image alignment preset: 12 inference steps, guidance_scale=1.0 (CFG off, so
   # single-GPU engines accept it), eta=0.7, 384x384. RES=256 halves the
   # activation cost if the 384 replay does not fit.
   # TRAINED SDE STEPS: the actor replays the transformer once PER trained step
   # (_track_logp + flow_grpo_update each loop over sde_indices). The replay grad
   # graph is ACTIVATION-CHECKPOINTED (--fsdp-activation-checkpointing, wired in
   # fsdp2_wrap) — each transformer block recomputes its forward in backward
   # instead of storing it — and the AdamW states stream to GPU only around
   # optimizer.step. Generation still uses all 12 denoising steps; only these 3
   # are trained.
   # Default [3,4,5] — three stochastic steps in the early high-sigma half,
   # mirroring the reference config (num_sde_steps=3, timestep_fraction=[0,0.5]), but SKIPPING
   # index 0. At index 0 sigma == 1 exactly, where the SDE coefficient
   # sqrt(sigma/(1-sigma)) diverges and is only finite because of the arbitrary
   # sigma_max clamp. Measured against the SGLang engine's own sampling log-prob:
   # steps away from that boundary agree to 1e-4, but step 0 disagrees 2.4x on the
   # std AND flips the sign of the prev_sample_mean 'sample' coefficient — i.e. the
   # replayed Gaussian is not the one the sample was drawn from, so the policy
   # gradient on the single most influential trained step is wrong (k3 ~= 0.20 vs
   # an on-policy target of ~0). The engine does not report its own noise std, so
   # the coefficient cannot be matched exactly; avoiding the singular step is the
   # fix that does not rely on guessing engine internals. Override with SDE_IDX.
   #
   # RESAMPLING (sde_resample_per_rollout): these indices decide BOTH where SDE
   # noise is injected at generation time and which steps get replayed for the
   # gradient, so pinning one set means exploration never leaves that subspace and
   # the other steps never receive any gradient. Redraw every rollout from
   # default_rng(rollout_id) over sde_pool. The pool deliberately
   # EXCLUDES step 0 for the singularity above. The explicit
   # sde_indices above stays as the deterministic fallback and is what held-out
   # eval uses -- eval must keep one fixed set or its number is not comparable
   # across steps. This matches the LoRA recipe's default_rng(0) draw over the
   # same pool.
   --sampling-config "{\"height\":${RES:-384},\"width\":${RES:-384},\"num_inference_steps\":12,\"guidance_scale\":1.0,\"eta\":0.7,\"sde_type\":\"${SDE_TYPE:-sde}\",\"sde_indices\":${SDE_IDX:-[3,4,5]},\"sde_resample_per_rollout\":${SDE_RESAMPLE:-true},\"num_sde_steps\":3,\"sde_pool\":${SDE_POOL:-[1,2,3,4,5]},\"driver_xt\":${DRIVER_XT:-true},\"sample_id_mode\":\"${SAMPLE_ID_MODE:-metadata}\"}"
   --generation-seed 42
)

FSDP_ARGS=(
   --fsdp-trainable-mode full
   --fsdp-trainable-attr transformer
   # DEBUG_FP=1 enables rank-0 debug diagnostics: the trainable-param fingerprint
   # and the [logp parity] line (replayed pi_old anchor vs the engine's own
   # sampling log-prob). Costs extra param syncs — diagnostic runs only.
   ${DEBUG_FP:+--fsdp-debug-fingerprint}
   --fsdp-param-dtype bf16
   --fsdp-reduce-dtype fp32
   # resharding after forward is the default; opt out with
   # --no-fsdp-reshard-after-forward if you would rather trade memory for speed.
   --fsdp-activation-checkpointing
   # Load the 20B base model 2 ranks at a time so 8 colocated ranks don't each
   # allocate a full CPU copy + hammer the shared model dir at once (launch OOM/hang).
   --fsdp-load-wave-size 2
   # Colocate time-share: offload the actor (model + optimizer) to CPU during
   # rollout so the diffusion engine has the GPU to generate; wake it for training
   # + weight sync. With the 3-step SDE subset (SAMPLING_ARGS) the training-step
   # peak (~50GB: sharded params/grads/AdamW + 3-step replay activations) fits
   # beside the offloaded engine's ~6GB context, so full-FT trains WITHOUT
   # --fsdp-cpu-offload. (cpu-offload was tried but every FSDP collective on the
   # CPU-offloaded DTensors — full_tensor, clip_grad_norm_ — fails "No backend
   # type associated with device type cpu"; not worth it once the 3-step subset
   # fits on GPU. To train all 12 steps at higher n, use the LoRA script
   # (run-qwen-image-t2i-lora-8xgpu.sh): it frees the grads + AdamW moments of
   # the frozen base (~25GB/rank), which is the binding constraint here. Note it
   # does NOT shrink the weight sync — merge mode still streams the full
   # transformer every step.)
   --offload-train
   # Offload the diffusion engine's movable modules to CPU during training (the
   # engine also self-offloads dit/text_encoder). Weight sync onloads only the
   # engine transformer, then fully reloads before the next rollout.
   --offload-rollout
)

GRPO_ARGS=(
   --advantage-estimator grpo
   # FlowGRPO uses a tiny PPO clip range (alignment recipe clip_range=1e-4), not the text-RL 0.2.
   --eps-clip 1e-4
   # Full-FT uses canonical Relax/Text GRPO scaling by default. Use batch only
   # when explicitly reproducing a reference recipe that uses global std.
   --generative-advantage-std-mode ${ADVANTAGE_STD_MODE}
   # PPO mini-epochs per rollout batch (alignment recipe num_updates_per_batch=2). With 1 the
   # ratio is always 1 → clipped loss is 0 by construction (group-centered advantages)
   # and clipping never engages; 2 makes the 2nd update diverge → nonzero, meaningful
   # loss + real PPO clipping (reuses the rollout data against the frozen old_logp).
   --num-updates-per-batch 2
)

OPTIMIZER_ARGS=(
   # NOTE: with gradient accumulation the rollout now takes num-updates-per-batch
   # optimizer steps (2) instead of one per prompt group (was 8 groups x 2 = 16),
   # so the policy moves ~8x less per rollout at the same lr. If convergence looks
   # slow, raise --lr.
   #
   # LR was bracketed empirically on the 384/8x8 config against the held-out
   # 256-prompt PickScore eval (baseline eval@0 = 0.7818, eval noise ~0.0015):
   #   lr=1e-5, adam_eps=1e-8  -> FROZEN. 129 rollouts, eval flat 0.7818->0.7826;
   #     diffing the DCP ckpts showed only ~5e-5 RELATIVE weight drift per 100
   #     optimizer steps; ratio in [0.9999,1.0002], approx_kl 0, clip_frac ~0.
   #   lr=1e-4, adam_eps=1e-15 -> OVERSHOOT. eval 0.7818 -> 0.7220 @rollout 19
   #     (-0.06 = 37x the eval noise; damage lands in the first ~20 rollouts).
   #     ratio reached 2.5e-3 = 25x the eps-clip range, so u1 was 100% clipped
   #     (clip_fraction pinned at 0.50 = u0 0.0 + u1 1.0) and grad_norm spiked to
   #     2.03 (vs 0.015 at lr=1e-5). NOTE the two changes compound: dropping
   #     adam_eps out of the eps-dominated regime is itself worth ~2x, so 1e-4
   #     + 1e-15 was ~20-25x the original step, not 10x.
   # 3e-5 is the midpoint of that bracket. What matters is movement in FUNCTION
   # space (the logp ratio), not parameter space: the reference LoRA recipe uses
   # clip_range=1e-4 with lr=3e-4, so a full-FT lr is NOT simply "30x smaller".
   # Target ratio ~7e-4 so u1 still clips
   # partially instead of totally.
   --lr ${LR:-3e-5}
   # Warmup over optimizer steps (num-updates-per-batch per rollout, so 20 steps
   # ~= 10 rollouts). The lr=1e-4 run took its damage almost entirely in the first
   # 20 rollouts; ramping in avoids that initial shock.
   --lr-warmup-iters ${LR_WARMUP:-20}
   # AdamW is only scale-invariant while sqrt(v) >> eps. The mean-reduced flow-SDE
   # logp yields per-element grads ~1e-8 (measured: optim exp_avg_sq ~1e-16), so the
   # default eps=1e-8 DOMINATES the denominator — it silently damps the step and makes
   # it resolution-dependent (grad_norm fell 0.111 -> 0.015 going 256 -> 384). A tiny
   # eps restores scale invariance so --lr means what it says.
   --adam-eps ${ADAM_EPS:-1e-15}
   # The reference Qwen-Image LoRA recipe uses weight_decay 0.0; decay is pure drag on an RL fine-tune.
   --weight-decay ${WEIGHT_DECAY:-0.0}
)

REWARD_ARGS=(
   # PickScore (CLIP-H, ~2B) scoring device. `cpu` was chosen back when the
   # full-FT actor was fighting the engine for the card, but that was before
   # activation checkpointing landed: the measured train peak is now 63.5 GB of a
   # 96 GB card, and the scorer runs while the actor is offloaded anyway, so the
   # ~5.6 GB it needs fits comfortably. On CPU it dominates the rollout's
   # post-processing — 104 s/rollout for reward+convert+TQ at 256 samples, ~20% of
   # the 523 s cycle. `colocate` scores in-process on the rollout GPU instead
   # (see generative._get_manager). `reward/score_time` in the reward log line
   # now measures the scorer alone, so the win is verifiable rather than assumed.
   --reward-runtime ${REWARD_RUNTIME:-colocate}
   --reward-scorer-path relax.engine.rewards.pickscore.PickScoreScorer
   --reward-model-path ${MODEL_DIR}/PickScore_v1
   --reward-required-components '["pickscore"]'
   --reward-component-weights '{"pickscore":1.0}'
)

WEIGHT_SYNC_ARGS=(
   --weight-sync-mode full
   --weight-sync-wire-dtype bf16
   # Bucket size drives the RPC COUNT, and a sync's measured cost is almost
   # entirely per-RPC overhead rather than payload: the [wsync time] breakdown
   # reads gather=0.5s / serialize=0.3s / transport=38.8s for 40.9GB, i.e. ~0.5s
   # per bucket for what is an on-device CUDA-IPC handle open. 512MB = 80 buckets,
   # 2048MB = 20. The bucket is materialized on GPU, so this also raises the
   # transient sync footprint by ~1.5GB (train peak is ~64GB of 96GB).
   --weight-sync-bucket-size-mb ${WSYNC_BUCKET_MB:-2048}
)

SGLANG_ARGS=(
   # Colocate layout (mirrors the text scripts): ONE single-GPU diffusion server
   # per GPU. RolloutManager creates world (=8) independent engines, each pinned to
   # its own physical card. The engine reuses the text SGLangEngine launch path —
   # SGLang's own launch_server runs inside a framework-managed mp.Process, so
   # kill_process_tree reaps the whole tree (uvicorn + sgl_diffusion::scheduler_*
   # workers) cleanly on any teardown; there is no ad-hoc subprocess to orphan.
   # num-gpus-per-engine is a real config: 1 fits Qwen-Image (20B) on one card;
   # a larger diffusion model can set N (the server is then pinned to a contiguous
   # base..base+N-1 GPU block via CUDA_VISIBLE_DEVICES). Weight sync is
   # per-rank->per-engine CUDA IPC (rank j -> engine j, same GPU); see
   # FSDPTrainRayActor._run_weight_transaction.
   --rollout-num-gpus-per-engine 1
)

WANDB_ARGS=(
   # Metrics service collects metrics and fans them out to ClearML (service.py
   # dispatches to the _ClearMLAdapter). Both flags on together — same pattern as
   # the text scripts. clearml reuses --tb-project-name / --tb-experiment-name as
   # its project/task (creds from env CLEARML_* or ~/clearml.conf).
   --use-clearml
   --use-metrics-service
   --tb-project-name ${PROJECT_NAME}
   --tb-experiment-name qwen-image-t2i-gpu8-${now}
)

mkdir -p log
ray job submit ${RAY_NO_WAIT:+--no-wait} --address="http://127.0.0.1:8265" \
   ${WORKING_DIR:+--working-dir "${WORKING_DIR}"} \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 -m relax.entrypoints.train \
   --resource '{"actor": [1, 8], "rollout": [1, 8]}' \
   --colocate \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${EVAL_ARGS[@]}" \
   "${SAMPLING_ARGS[@]}" \
   "${FSDP_ARGS[@]}" \
   "${GRPO_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${REWARD_ARGS[@]}" \
   "${WEIGHT_SYNC_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${WANDB_ARGS[@]}" 2>&1 | tee log/qwen-image-t2i-gpu8-${now}.log
