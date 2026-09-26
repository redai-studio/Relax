#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Qwen3-0.6B SFT, exactly 4 local RTX 4090s, data-parallel size 4
# (TP = PP = CP = 1), single node, ray-submit launch.
#
# Purpose: a *control-variable* recipe for the Task 11 straggler profiler. All
# four ranks are identical data-parallel replicas of the same role, so they form
# one valid comparison cohort: same model, same chunk, same micro-batch budget,
# differing only in the DP index. That is the precondition the observer/detector
# needs to call a rank a straggler instead of merely "a different rank".
#
# Derived from scripts/training/sft/run-qwen3-0.6B-math-8xgpu.sh with only the
# topology, dataset and run-length changed. See the ARGUMENT NOTES section at the
# bottom of this header for one line per change and why.
#
# Intentionally NOT enabled: wandb (the 8-GPU original never enabled it either;
# --use-wandb is absent), ClearML (no ClearML server is available locally), and
# the straggler profiler (default off; see the opt-in block further down).
#
# Data: the OpenMathReasoning-mini parquet the 8-GPU script expects does not
# exist on this machine. A 256-row JSONL derived from the local
# dapo-math-17k.jsonl is used instead (tools/straggler/make_sft_dataset.py).
#
# ── environment contract ───────────────────────────────────────────────────
# Every knob below can be overridden from the environment; the defaults are the
# verified paths of this machine.
#
#   TRAIN_VENV      native training venv, must contain bin/python
#                   (default /root/autodl-tmp/megatron-stack/venv)
#   MEGATRON        Megatron-LM source tree
#                   (default /root/autodl-tmp/megatron-stack/Megatron-LM)
#   RELAX           Relax repo root; derived from this script's location
#   MODEL_PATH      Qwen3-0.6B HF weights
#                   (default the verified hf-cache snapshot)
#   PROMPT_SET      SFT JSONL        (default scripts/training/sft/data/dapo-math-17k-sft-256.jsonl)
#   NUM_GPUS        GPUs Ray may see (default 4)
#   SAVE_DIR        checkpoint root  (default /root/autodl-tmp/...)
#   NUM_ROLLOUT     optimizer steps  (default 48; rollout_batch_size == global_batch_size)
#   GLOBAL_BATCH_SIZE  samples per step (default 32)
#   SAVE_INTERVAL   periodic save interval (default 1000000 == effectively off)
#   SAVE=0          omit --save entirely. The finish path performs one forced
#                   synchronous checkpoint even with --save-interval off, which
#                   would dominate a short whole-run wall-time comparison; the
#                   timing arms set SAVE=0 in BOTH arms so only the observer
#                   differs.
#   DRY_RUN=1       print the resolved runtime env + command and exit without
#                   starting Ray or submitting anything
#
# ── straggler profiler (OPT-IN, default OFF) ───────────────────────────────
# Uncomment these six export lines to enable the profiler. The injection block
# right below them then copies the variables into the Ray runtime env so every
# worker process (not just the submitter shell) sees them. With them commented
# out RELAX_STRAGGLER_ENABLE is unset and the injection block is a no-op, so the
# default recipe is byte-for-byte the observer-off arm.
#
# export RELAX_STRAGGLER_ENABLE=1
# export RELAX_STRAGGLER_COLLECTOR_ADDR=127.0.0.1:29741
# export RELAX_STRAGGLER_OUTPUT_DIR=/root/autodl-tmp/relax-work/task11_evidence/sft-dp4-observer
# export RELAX_STRAGGLER_WINDOW_S=5.0
# export RELAX_STRAGGLER_WARMUP_WINDOWS=2
# export RELAX_STRAGGLER_REPORT_INTERVAL_S=10.0
#
# ── usage ──────────────────────────────────────────────────────────────────
#   bash scripts/training/sft/run-qwen3-0.6B-4xgpu-dp4-observer.sh
#   DRY_RUN=1 bash scripts/training/sft/run-qwen3-0.6B-4xgpu-dp4-observer.sh
#
# ── ARGUMENT NOTES (change vs. run-qwen3-0.6B-math-8xgpu.sh) ───────────────
#   --resource {"sft":[1,0],"actor":[1,4]}  actor DP=4 on 4 GPUs; no separate
#       rollout service because loss_type=sft only ever binds {sft, actor}
#       (relax/core/registry.py ROLES_SFT_ONLY). Dropping rollout also drops
#       the rollout->actor weight-sync wiring (_needs_rollout_manager_setup).
#   --num-data-storage-units 4             scaled from 8 for 4 DP ranks.
#   --num-rollout 48                       48 optimizer steps after warm-up
#       (train_iters = num_rollout * rollout_batch_size // global_batch_size).
#       --num-epoch is deliberately NOT set: on a 256-row set with
#       global_batch_size=32 one epoch is only 8 steps, and resolve_sft_num_rollout
#       caps num_rollout by num_epoch, so --num-epoch 1 would silently cut the
#       run to 8 steps.
#   --prompt-data/--input-key/--label-key  point at the generated JSONL; the
#       SFT loader requires a *string* input when --label-key is set, so the
#       dapo {prompt: [messages], label: str} rows were flattened to
#       {problem: str, generated_solution: str}.
#   --save-interval 1000000 (+ --no-save-rng)  periodic saving off; a run this
#       short would otherwise pay ~11 s/step and destroy the measurement.
#   --no-use-wandb                          never passed (wandb off).
#   --save/-hf-checkpoint/-ref-load         retargeted from the missing
#       ${EXP_DIR}/Qwen3-0.6B and the missing OpenMathReasoning parquet to the
#       verified local snapshot and generated JSONL.
#   dropped --load                          so each launch deterministically
#       starts from --ref-load (bridge mode) instead of resuming a stale
#       checkpoint left in SAVE_DIR by a previous run.
#   dropped --eval-size/--eval-interval     a 48-step timing run never reaches
#       eval_interval=1000; removing both also avoids carving rows out of the
#       tiny train pool. (arguments.py requires eval-size XOR eval-prompt-data
#       whenever eval-interval is set, so they must be dropped together.)

set -ex
set -o pipefail

# ── environment contract (exported before the entrypoint is sourced) ────────
TRAIN_VENV="${TRAIN_VENV:-/root/autodl-tmp/megatron-stack/venv}"
MEGATRON="${MEGATRON:-/root/autodl-tmp/megatron-stack/Megatron-LM}"
MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/hf-cache/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca}"
NUM_GPUS="${NUM_GPUS:-4}"
SAVE_DIR="${SAVE_DIR:-/root/autodl-tmp/relax-checkpoints/qwen3-0.6B-sft-dp4-observer}"
NUM_ROLLOUT="${NUM_ROLLOUT:-48}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-32}"
SAVE_INTERVAL="${SAVE_INTERVAL:-1000000}"
SAVE="${SAVE:-1}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
RELAX="${RELAX:-$(cd -- "${SCRIPT_DIR}/../../.." &>/dev/null && pwd)}"
# Mandatory: submit with an explicit working directory so the Ray driver and its
# actors import THIS repo's `relax` package. Without --working-dir the Ray daemon
# resolves `relax` from its own cwd, which can be another worktree entirely.
WORKING_DIR="${WORKING_DIR:-${RELAX}}"
MODEL_CONFIG_DIR="${MODEL_CONFIG_DIR:-${SCRIPT_DIR}/../../models}"
PROMPT_SET="${PROMPT_SET:-${SCRIPT_DIR}/data/dapo-math-17k-sft-256.jsonl}"
LOG_DIR="${LOG_DIR:-${RELAX}/log}"

PROJECT_NAME="${PROJECT_NAME:-Relax/sft/math-dp4-observer}"
EXP_NAME="${EXP_NAME:-qwen3-0.6b-sft-math-dp4-observer}"
NOW="$(date "+%Y-%m-%d-%H:%M:%S")"
LOG_PATH="${LOG_DIR}/${EXP_NAME}-${NOW}.log"

TRAIN_PYTHON="${TRAIN_VENV}/bin/python"
if [ ! -x "${TRAIN_PYTHON}" ]; then
    echo "TRAIN_VENV has no executable bin/python: ${TRAIN_VENV}" >&2
    exit 2
fi

# ── entrypoint environment ──────────────────────────────────────────────────
# Same PYTHONPATH / RUNTIME_ENV_JSON pattern as scripts/entrypoint/local.sh.
# DRY_RUN sources the very same file, but shadows the destructive Ray/process
# commands with shell functions so nothing is started or killed.
export MEGATRON RELAX NUM_GPUS MODEL_CONFIG_DIR
if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    if [ "${DRY_RUN:-0}" = "1" ]; then
        echo "=== DRY_RUN: sourcing scripts/entrypoint/local.sh with Ray/process commands shadowed ==="
        export RAY_ADDRESS=""
        ray() { echo "[dry-run] ray $*"; return 0; }
        pkill() { echo "[dry-run] pkill $*"; return 0; }
        sleep() { return 0; }
        nvidia-smi() { return 1; }
        # shellcheck source=../../entrypoint/local.sh
        source "${SCRIPT_DIR}/../../entrypoint/local.sh"
        unset -f ray pkill sleep nvidia-smi
    else
        # shellcheck source=../../entrypoint/local.sh
        source "${SCRIPT_DIR}/../../entrypoint/local.sh"
    fi
fi

# Ray's system interpreter owns worker processes while Transformer Engine and the
# native extensions live in TRAIN_VENV; carry its site-packages into the submitted
# runtime environment (same trick as the Task 11 observer smoke script).
TRAIN_SITE="$("${TRAIN_PYTHON}" -c 'import site; print(site.getsitepackages()[0])')"
if [ ! -d "${TRAIN_SITE}" ]; then
    echo "Could not resolve training site-packages: ${TRAIN_SITE}" >&2
    exit 2
fi
export RUNTIME_ENV_JSON="$(TRAIN_SITE="${TRAIN_SITE}" "${TRAIN_PYTHON}" - <<'PY'
import json
import os

runtime_env = json.loads(os.environ["RUNTIME_ENV_JSON"])
env_vars = runtime_env.setdefault("env_vars", {})
existing = env_vars.get("PYTHONPATH", "")
env_vars["PYTHONPATH"] = f"{os.environ['TRAIN_SITE']}:{existing}" if existing else os.environ["TRAIN_SITE"]
print(json.dumps(runtime_env, separators=(",", ":")))
PY
)"

# ── optional straggler profiler (default OFF) ───────────────────────────────
# The exports at the top of this file are commented out, so this block does
# nothing unless the operator uncomments them or exports the variables itself.
if [ -n "${RELAX_STRAGGLER_ENABLE:-}" ]; then
    export RELAX_STRAGGLER_ENABLE=1
    export RELAX_STRAGGLER_COLLECTOR_ADDR="${RELAX_STRAGGLER_COLLECTOR_ADDR:-127.0.0.1:29741}"
    export RELAX_STRAGGLER_OUTPUT_DIR="${RELAX_STRAGGLER_OUTPUT_DIR:-/root/autodl-tmp/relax-work/task11_evidence/sft-dp4-observer/${NOW}}"
    export RELAX_STRAGGLER_WINDOW_S="${RELAX_STRAGGLER_WINDOW_S:-5.0}"
    export RELAX_STRAGGLER_WARMUP_WINDOWS="${RELAX_STRAGGLER_WARMUP_WINDOWS:-2}"
    export RELAX_STRAGGLER_REPORT_INTERVAL_S="${RELAX_STRAGGLER_REPORT_INTERVAL_S:-10.0}"
    mkdir -p "${RELAX_STRAGGLER_OUTPUT_DIR}"
    export RUNTIME_ENV_JSON="$("${TRAIN_PYTHON}" - <<'PYINNER'
import json
import os

runtime_env = json.loads(os.environ["RUNTIME_ENV_JSON"])
env_vars = runtime_env.setdefault("env_vars", {})
for name in (
    "RELAX_STRAGGLER_ENABLE",
    "RELAX_STRAGGLER_COLLECTOR_ADDR",
    "RELAX_STRAGGLER_OUTPUT_DIR",
    "RELAX_STRAGGLER_WINDOW_S",
    "RELAX_STRAGGLER_WARMUP_WINDOWS",
    "RELAX_STRAGGLER_REPORT_INTERVAL_S",
):
    env_vars[name] = os.environ[name]
print(json.dumps(runtime_env, separators=(",", ":")))
PYINNER
)"
    echo "straggler profiler ENABLED: window=${RELAX_STRAGGLER_WINDOW_S}s warmup=${RELAX_STRAGGLER_WARMUP_WINDOWS} collector=${RELAX_STRAGGLER_COLLECTOR_ADDR}"
    echo "straggler output dir: ${RELAX_STRAGGLER_OUTPUT_DIR}"
else
    echo "straggler profiler disabled (RELAX_STRAGGLER_ENABLE unset); pass the commented exports to enable"
fi

source "${MODEL_CONFIG_DIR}/qwen3-0.6B.sh"

CKPT_ARGS=(
   --hf-checkpoint "${MODEL_PATH}"
   --ref-load "${MODEL_PATH}"

   --megatron-to-hf-mode bridge
)

if [ "${SAVE}" = "1" ]; then
   CKPT_ARGS+=(
      --save "${SAVE_DIR}/sft/${EXP_NAME}"
      --save-interval "${SAVE_INTERVAL}"
      --no-save-rng
   )
fi

SFT_ARGS=(
   --loss-type sft
   --prompt-data "${PROMPT_SET}"
   --input-key problem
   --label-key generated_solution
   --global-batch-size "${GLOBAL_BATCH_SIZE}"
   --use-dynamic-batch-size
   --max-tokens-per-gpu 20480
   --sft-oversize-strategy skip
   --balance-data
   --per-rank-fetch
   --sft-async-prepack
   --sft-prefetch-num-workers 8
   --sft-prefetch-buffer-size 256
   --num-rollout "${NUM_ROLLOUT}"
)

PERF_ARGS=(
   --tensor-model-parallel-size 1
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --no-rope-fusion

   --colocate
   --cross-entropy-loss-fusion
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-5
   --lr-decay-style cosine
   --min-lr 1e-6
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
   --clip-grad 1.0
)

METRICS_ARGS=(
   --use-metrics-service
   --tb-project-name "${PROJECT_NAME}"
   --tb-experiment-name "${EXP_NAME}-${NOW}"
   # --use-wandb is intentionally NOT passed: wandb stays disabled.
   # --use-clearml is intentionally NOT passed: no ClearML server locally.
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   --use-health-check
)

# Full argument list after `-m relax.entrypoints.train`; built once so the
# dry-run printer and the real submission cannot drift apart.
TRAIN_ARGS=(
   --resource '{"sft": [1, 0], "actor": [1, 4]}'
   --sft-max-in-flight-steps 4
   --num-data-storage-units 4
   "${MODEL_ARGS[@]}"
   "${CKPT_ARGS[@]}"
   "${SFT_ARGS[@]}"
   "${OPTIMIZER_ARGS[@]}"
   "${METRICS_ARGS[@]}"
   "${PERF_ARGS[@]}"
   "${MISC_ARGS[@]}"
)

RAY_ADDR="${RAY_ADDRESS:-http://127.0.0.1:8265}"

if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "=== DRY_RUN resolved RUNTIME_ENV_JSON (not submitted) ==="
    if command -v jq >/dev/null 2>&1; then
        printf '%s' "${RUNTIME_ENV_JSON}" | jq .
    else
        printf '%s\n' "${RUNTIME_ENV_JSON}"
    fi
    echo "=== DRY_RUN final command line (not submitted) ==="
    printf 'ray job submit'
    if [ -n "${RAY_NO_WAIT:-}" ]; then printf ' --no-wait'; fi
    printf ' --address=%q' "${RAY_ADDR}"
    if [ -n "${WORKING_DIR:-}" ]; then printf ' --working-dir %q' "${WORKING_DIR}"; fi
    printf ' --runtime-env-json=%q' "${RUNTIME_ENV_JSON}"
    printf ' -- %q -m relax.entrypoints.train' "${TRAIN_PYTHON}"
    printf ' %q' "${TRAIN_ARGS[@]}"
    printf '\n'
    echo "=== DRY_RUN log path (real run would tee here) ==="
    echo "${LOG_PATH}"
    if [ -n "${TRAIN_ARGS_DUMP:-}" ]; then
        # NUL-separated argv (after `-m relax.entrypoints.train`) so an
        # argument-parse-only validation can feed it straight to parse_args()
        # without re-typing the list.
        printf '%s\0' "${TRAIN_ARGS[@]}" > "${TRAIN_ARGS_DUMP}"
        echo "=== DRY_RUN wrote ${#TRAIN_ARGS[@]} argv entries to ${TRAIN_ARGS_DUMP} ==="
    fi
    exit 0
fi

mkdir -p "${LOG_DIR}" "${SAVE_DIR}"

ray job submit ${RAY_NO_WAIT:+--no-wait} --address="${RAY_ADDR}" \
   ${WORKING_DIR:+--working-dir "${WORKING_DIR}"} \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- "${TRAIN_PYTHON}" -m relax.entrypoints.train \
   "${TRAIN_ARGS[@]}"  2>&1 | tee "${LOG_PATH}"
