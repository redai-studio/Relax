#!/usr/bin/env bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
set +x
set -euo pipefail
umask 077
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.."
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
unset RAY_ADDRESS RAY_JOB_SUBMISSION_ID RELAX_ENTRYPOINT_MODE WORKING_DIR RAY_NO_WAIT SSH_AUTH_SOCK
: "${RETOOL_OUTPUT_ROOT:?Choose a fresh output root}"
root_parent=$(cd -- "$(dirname -- "$RETOOL_OUTPUT_ROOT")" && pwd -P)
export ROOT="$root_parent/$(basename -- "$RETOOL_OUTPUT_ROOT")"
resume=()
continuation=()
if [[ -n "${RETOOL_RESUME_FROM:-}" ]]; then
  : "${RETOOL_OFFICIAL_MODEL:?Set the official base model for comparison}"
  resume=(--resume-from "$RETOOL_RESUME_FROM")
  if [[ -d "$ROOT" ]]; then
    continuation=(--continue-evaluation)
  else
    mkdir -- "$ROOT"
    ln -s "$(readlink -f "$RETOOL_RESUME_FROM/data")" "$ROOT/data"
    ln -s "$(readlink -f "$RETOOL_RESUME_FROM/model")" "$ROOT/model"
  fi
else
  mkdir -- "$ROOT"
  export RETOOL_OFFICIAL_MODEL="$ROOT/model"
  unset RETOOL_EVAL_CONFIG
fi
runtime_parent=${RETOOL_RUNTIME_PARENT:-/tmp}
RT=${RETOOL_PRECHECK_RT:-$(mktemp -d "$runtime_parent/rt.XXXXXX")}
SANDBOX_BASE=$(mktemp -d "$runtime_parent/rs.XXXXXX")
export RT SANDBOX_BASE
chmod 0711 "$SANDBOX_BASE"
export CUDA_VISIBLE_DEVICES= CUDA_DEVICE_MAX_CONNECTIONS=1
export WANDB_MODE=disabled WANDB_DISABLED=true CLEARML_OFFLINE_MODE=1
export RELAX_SKIP_CLUSTER_CLEANUP=1 RAY_USAGE_STATS_ENABLED=0 RAY_OVERRIDE_JOB_RUNTIME_ENV=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD${MEGATRON_PATH:+:$MEGATRON_PATH}${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$RT/tmp"
export TMPDIR="$RT/tmp"
exec > >(tee -a "$ROOT/launch.log") 2>&1
export RETOOL_SOURCE_COMMIT=${RETOOL_SOURCE_COMMIT:-$(git rev-parse HEAD)}
if [[ ${#continuation[@]} == 0 ]]; then
  printf '%s\n' "$RETOOL_SOURCE_COMMIT" > "$ROOT/source_commit.txt"
fi
if [[ ${#resume[@]} == 0 ]]; then
  python3 -m examples.retool.prepare_run "$ROOT"
fi
if [[ -z "${RETOOL_EVAL_CONFIG:-}" ]]; then
  env HTTP_PROXY="${RETOOL_DOWNLOAD_PROXY:-}" HTTPS_PROXY="${RETOOL_DOWNLOAD_PROXY:-}" \
    http_proxy="${RETOOL_DOWNLOAD_PROXY:-}" https_proxy="${RETOOL_DOWNLOAD_PROXY:-}" \
    python3 -m examples.retool.prepare_evaluation --model "$RETOOL_OFFICIAL_MODEL" --out "$ROOT/benchmarks"
  export RETOOL_EVAL_CONFIG=$ROOT/benchmarks/manifest.json
fi
extra=()
if [[ "${RETOOL_SMOKE:-0}" == 1 ]]; then
  : "${RETOOL_STOP_AFTER_UPDATES:?Smoke requires an explicit bounded optimizer boundary}"
  extra=(--smoke --stop-after-updates "$RETOOL_STOP_AFTER_UPDATES")
fi
exec python3 -m examples.retool.workflow --root "$ROOT" --temp "$RT" --model "$ROOT/model" \
  ${resume[@]+"${resume[@]}"} --deadline "$(( $(date +%s) + ${RETOOL_TIMEOUT_SECONDS:-172800} ))" \
  --sandbox-base "$SANDBOX_BASE" \
  --eval-config "$RETOOL_EVAL_CONFIG" ${extra[@]+"${extra[@]}"} ${continuation[@]+"${continuation[@]}"}
