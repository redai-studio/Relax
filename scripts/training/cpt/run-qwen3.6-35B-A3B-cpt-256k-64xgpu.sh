#!/usr/bin/env bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Qwen3.6 raw-text CPT, 256K, TP4/CP1/PP1/EP8/ETP1, on 64 Blackwell GPUs.
# MODEL_DIR points to the HF model, DATA_DIR to the input data, SAVE_DIR to checkpoints.
# Requires the optimized runtime dependencies used by the Qwen3.5 image recipe.
# Example: MODEL_DIR=/models/Qwen3.6 DATA_DIR=/data/train.jsonl SAVE_DIR=/checkpoints \
#   bash scripts/training/cpt/run-qwen3.6-35B-A3B-cpt-256k-64xgpu.sh --dry-run
# DRY_RUN=1 (or --dry-run) renders arguments and worker environment without submission.
# Set RELAX_SKIP_NONFINITE_GRAD=0 for fail-fast diagnostics.
set -euo pipefail
if [[ "${1:-}" == --dry-run || "${1:-}" == --render-env ]]; then
    export DRY_RUN=1
    shift
fi
# Capture the outer Ray Job before the entrypoint or tee can redirect its output.
if [[ ! -v CPT_PARENT_SUBMISSION_ID ]]; then
    export CPT_PARENT_SUBMISSION_ID="${RAY_JOB_SUBMISSION_ID:-}"
    if [[ -z "$CPT_PARENT_SUBMISSION_ID" ]]; then
        for fd in 1 2; do
            parent_log="$(readlink -f "/proc/$$/fd/$fd" 2>/dev/null || true)"
            if [[ "$parent_log" =~ /job-driver-(.+)\.(log|out|err)$ ]]; then
                export CPT_PARENT_SUBMISSION_ID="${BASH_REMATCH[1]}"
                break
            fi
        done
    fi
fi
export RELAX_SKIP_CLUSTER_CLEANUP=1

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export RELAX="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
: "${MODEL_DIR:?Set MODEL_DIR}"
: "${DATA_DIR:?Set DATA_DIR}"
: "${SAVE_DIR:?Set SAVE_DIR to a persistent shared checkpoint root}"
export NUM_GPUS=64
export RAY_ADDRESS="${RAY_ADDRESS:-http://127.0.0.1:8265}"
export RAY_JOBS_ADDRESS="${RAY_JOBS_ADDRESS:-$RAY_ADDRESS}"
export RAY_NO_WAIT="${RAY_NO_WAIT-}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$RELAX:${PYTHONPATH:-}"
if [[ "${DRY_RUN:-0}" != 1 && -z "${RELAX_ENTRYPOINT_MODE:-}" ]]; then
    exec bash "$RELAX/scripts/entrypoint/ray-job.sh" "${BASH_SOURCE[0]}" "$@"
fi
source "${MODEL_CONFIG_DIR:-$RELAX/scripts/models}/qwen36-35B-A3B.sh"

PROJECT_NAME="${PROJECT_NAME:-Relax/cpt/qwen36_optimized}"
EXP_NAME="${EXP_NAME:-qwen36-cpt-256k-64xB300}"  # gitleaks:allow -- stable experiment/checkpoint identity
now="$(date +%Y%m%d-%H%M%S)"
TASK_NAME="${EXP_NAME}-${now}-$$"
export SUBMISSION_ID="${SUBMISSION_ID:-$TASK_NAME}"
if [[ -n "$CPT_PARENT_SUBMISSION_ID" && "$SUBMISSION_ID" == "$CPT_PARENT_SUBMISSION_ID" ]]; then
    echo "Training submission ID must differ from the parent Ray Job ID" >&2
    exit 1
fi
CHECKPOINT_DIR="${SAVE_DIR}/${EXP_NAME}"
TRAIN_STEPS="${TRAIN_STEPS:-1000}"
LR="${LR:-2e-5}"
MIN_LR="${MIN_LR:-1e-6}"
SAVE_INTERVAL="${SAVE_INTERVAL:-100}"
CPT_TEMPLATE="${CPT_TEMPLATE:-raw}"
INPUT_KEY="${INPUT_KEY:-text}"
[[ "$TRAIN_STEPS" =~ ^[1-9][0-9]*$ ]] || { echo "TRAIN_STEPS must be positive" >&2; exit 1; }
[[ "$CPT_TEMPLATE" == raw || "$CPT_TEMPLATE" == qwen3_5 ]] || {
    echo "CPT_TEMPLATE must be raw or qwen3_5" >&2; exit 1;
}
if [[ "$SAVE_INTERVAL" == last ]]; then
    SAVE_EVERY="$TRAIN_STEPS"
else
    [[ "$SAVE_INTERVAL" =~ ^[1-9][0-9]*$ ]] || { echo "SAVE_INTERVAL must be positive or last" >&2; exit 1; }
    SAVE_EVERY="$SAVE_INTERVAL"
fi

# Serialize the same native-library environment for the Ray driver and actors.
ENV_JSON="$(python3 - <<'PYENV'
import json
import os
from pathlib import Path

def build_environment() -> dict[str, str]:
    """Select only the optimized job's native libraries, including Ray
    workers."""
    import sysconfig

    site = Path(sysconfig.get_paths()["purelib"])
    nccl = site / "nvidia/nccl"
    cublas = Path("/opt/relax/cublas13.3/nvidia/cu13/lib")
    kept = [
        p
        for p in os.environ.get("LD_PRELOAD", "").replace(":", " ").split()
        if not Path(p).name.startswith(("libnccl.so", "libcublas.so", "libcublasLt.so"))
    ]
    env = {key: os.environ[key] for key in ("PYTHONPATH", "PYTHONDONTWRITEBYTECODE") if key in os.environ}
    env.update(
        {
            "LD_PRELOAD": ":".join(
                [
                    str(cublas / "libcublasLt.so.13"),
                    str(cublas / "libcublas.so.13"),
                    str(nccl / "lib/libnccl.so.2"),
                    *kept,
                ]
            ),
            "LD_LIBRARY_PATH": ":".join(
                [
                    str(cublas),
                    str(nccl / "lib"),
                    str(site / "nvidia/nvshmem/lib"),
                    os.environ.get("LD_LIBRARY_PATH", ""),
                ]
            ),
            "RELAX_FLA_CONV_NB_MODE": "none",
            "EP_NCCL_ROOT_DIR": str(nccl),
            "EP_JIT_CACHE_DIR": f"/tmp/relax-qwen35-{os.getuid()}/deepep",
        }
    )
    env.update(
        {
            "EP_DISABLE_GIN": "1",
            "NCCL_CUMEM_ENABLE": "1",
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
            "FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED": "0",
            "NVTE_GROUPED_LINEAR_USE_FUSED_GROUPED_GEMM": "0",
            "NVTE_CUTEDSL_FUSED_GROUPED_MLP": "0",
            "RELAX_SFT_SELECTIVE_LOGITS": "1",
            "RELAX_DISABLE_VISION_BRANCH_SYNC": "1",
            "RELAX_LOGPROB_CE_IMPL": "native",
            "RELAX_MTP_CE_IMPL": "native",
            "RELAX_LOGPROB_FAST_PATH": "1",
            "OMP_NUM_THREADS": "24",
            "MKL_NUM_THREADS": "24",
            "OPENBLAS_NUM_THREADS": "24",
            "PYTORCH_ALLOC_CONF": "backend:native,expandable_segments:True",
            "PYTORCH_CUDA_ALLOC_CONF": "backend:native,expandable_segments:True",
            "NCCL_NVLS_ENABLE": "0",
            "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
            "NVTE_BWD_LAYERNORM_SM_MARGIN": "20",
            "NVTE_FWD_LAYERNORM_SM_MARGIN": "20",
            "NCCL_IGNORE_CPU_AFFINITY": "1",
            "NCCL_GRAPH_REGISTER": "0",
            "NVTE_USE_CUTLASS_GROUPED_GEMM": "1",
            "TORCH_DIST_INIT_BARRIER": "1",
            "TORCH_NCCL_BLOCKING_WAIT": "0",
            "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
            "TORCH_DISTRIBUTED_DEFAULT_TIMEOUT": "3600",
            "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "256",
            "TORCHINDUCTOR_COMPILE_THREADS": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    data_node = os.environ.get("RELAX_SFT_DATA_NODE_IP") or os.environ.get("MASTER_ADDR")
    if data_node:
        env["RELAX_SFT_DATA_NODE_IP"] = data_node
    elif os.environ.get("DRY_RUN") != "1":
        raise RuntimeError("The standard Ray entrypoint must resolve MASTER_ADDR before submission")
    for key in (
        "CUDA_HOME",
        "CPATH",
        "TRITON_PTXAS_PATH",
        "TRITON_CACHE_DIR",
        "NCCL_SOCKET_IFNAME",
        "NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME",
        "CLEARML_OFFLINE_MODE",
        "CLEARML_CACHE_DIR",
        "RELAX_SKIP_NONFINITE_GRAD",
    ):
        if key in os.environ:
            env[key] = os.environ[key]
    env["RELAX_SKIP_NONFINITE_GRAD"] = os.environ.get("RELAX_SKIP_NONFINITE_GRAD", "1")
    return env


if os.environ.get("DRY_RUN") != "1":
    manifest = json.loads(Path("/etc/relax-qwen35-runtime.json").read_text())
    assert manifest["packages"]["relax-deep-ep-v2"] == "2.1.0"
    mode = os.environ.get("RELAX_FLA_CONV_NB_MODE", "original")
    assert mode == "none" or (mode == "auto" and os.environ.get("RELAX_GPU_ARCH") in {"blackwell", "b300"})  # gitleaks:allow -- existing runtime environment compatibility key
env = build_environment()
runtime = json.loads(os.environ.get("RUNTIME_ENV_JSON") or "{}")
runtime.setdefault("env_vars", {}).update(env)
for key in ("NCCL_PROTO", "NCCL_ALGO"):
    runtime["env_vars"].pop(key, None)
print(json.dumps(env))
print(json.dumps(runtime))
PYENV
)"
TRAIN_ENV_JSON="${ENV_JSON%%$'\n'*}"
RUNTIME_ENV_JSON="${ENV_JSON#*$'\n'}"

CKPT_ARGS=(
    --hf-checkpoint "${MODEL_DIR}"
    --ref-load "${MODEL_DIR}"
    --load "${CHECKPOINT_DIR}"
    --megatron-to-hf-mode bridge
    --warm-hf-checkpoint-page-cache
    --max-actor-ckpt-to-keep 1
    --num-rollout "${TRAIN_STEPS}"
    --save "${CHECKPOINT_DIR}"
    --save-interval "${SAVE_EVERY}"
)

CPT_ARGS=(
    --loss-type sft
    --prompt-data "${DATA_DIR}"
    --seq-length 262144
    --global-batch-size 1600
    --use-dynamic-batch-size
    --sft-packing-strategy binpack
    --balance-data
    --sft-oversize-strategy skip
    --data-pad-size-multiplier 1024
    --sft-max-in-flight-steps 2
    --sft-prefetch-buffer-size 128
    --sft-prefetch-chunk-size 128
    --sft-prefetch-num-workers 16
    --sft-tq-timeout-minutes 60
    --sft-chunked-logits
    --sft-logits-chunk-size 1024
    --eval-size 0.01
    --eval-interval 0
    --per-rank-fetch
    --sft-train-data-prefetch
    --input-key "${INPUT_KEY}"
    --sft-training-mode cpt
    --sft-cpt-template "${CPT_TEMPLATE}"
    --metadata-key ""
    --tool-key ""
    --eval-tool-key ""
    --sft-invalid-multimodal-strategy skip
)

MTP_ARGS=(
    --mtp-num-layers 1
    --enable-mtp-training
    --mtp-loss-scaling-factor 0.1
    --mtp-detach-paths none
)

OPTIMIZER_ARGS=(
    --optimizer adam
    --lr "${LR}"
    --lr-decay-style cosine
    --lr-warmup-iters 10
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.98
    --adam-eps 1e-8
    --clip-grad 1.0
    --optimizer-cpu-offload
    --optimizer-offload-fraction 0
    --overlap-cpu-optimizer-d2h-h2d
    --use-precision-aware-optimizer
    --lr-warmup-init 0
    --lr-decay-iters 200
)

PERF_ARGS=(
    --linear-cp-mode headwise
    --recompute-loss-function
    --recompute-granularity full
    --recompute-method block
    --calculate-per-token-loss
    --moe-token-dispatcher-type flex
    --cross-entropy-loss-fusion
    --moe-shared-expert-overlap
    --overlap-grad-reduce
    --overlap-param-gather
    --no-rope-fusion
    --moe-router-load-balancing-type aux_loss
    --moe-aux-loss-coeff 1e-4
    --moe-router-dtype fp32
    --cross-entropy-fusion-impl linear
    --sequence-parallel
    --recompute-num-layers 35
    --moe-flex-dispatcher-backend deepepv2
    --moe-flex-dispatcher-num-sms 80
    --moe-router-fusion
    --overlap-dispatch-backward-with-experts-wgrad
    --tensor-model-parallel-size 4
    --pipeline-model-parallel-size 1
    --context-parallel-size 1
    --expert-model-parallel-size 8
    --expert-tensor-parallel-size 1
    --max-tokens-per-gpu 262144
    --sft-wholepack
    --custom-megatron-before-train-step-hook-path relax.models.qwen3_5.recipe.before_step
)

MONITOR_ARGS=(
    --use-clearml
    --use-metrics-service
    --tb-project-name "${PROJECT_NAME}"
    --tb-experiment-name "${TASK_NAME}"
)

MISC_ARGS=(
    --resource '{"sft":[1,0],"actor":[1,64]}'
    --max-staleness 0
    --num-data-storage-units 8
    --log-probs-chunk-size 2048
    --min-lr "${MIN_LR}"
    --seed 42
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --accumulate-allreduce-grads-in-fp32
    --attention-softmax-in-fp32
    --attention-backend auto
    --use-health-check
    --train-env-vars "${TRAIN_ENV_JSON}"
    --max-global-restart 0
    --freeze-vision-model
    --freeze-vision-projection
    --no-check-for-nan-in-loss-and-grad
)

TRAIN_ARGS=(
    python3 -m relax.entrypoints.train
    "${MODEL_ARGS[@]}" "${CKPT_ARGS[@]}" "${CPT_ARGS[@]}" "${MTP_ARGS[@]}"
    "${OPTIMIZER_ARGS[@]}" "${PERF_ARGS[@]}" "${MONITOR_ARGS[@]}" "${MISC_ARGS[@]}"
    "$@"
)
if [[ "${DRY_RUN:-0}" == 1 ]]; then
    python3 - "$RUNTIME_ENV_JSON" "${TRAIN_ARGS[@]}" <<'PYRENDER'
import json
import sys

print(json.dumps({"runtime_env": json.loads(sys.argv[1]), "argv": sys.argv[2:]}, indent=2))
PYRENDER
    exit 0
fi

# Refuse concurrent training, preserving the parent job and existing cluster.
python3 -B - <<'PY_JOBS'
import json
import os
import urllib.request
with urllib.request.urlopen(os.environ["RAY_JOBS_ADDRESS"].rstrip("/") + "/api/jobs/", timeout=10) as response:
    jobs = json.load(response)
if not isinstance(jobs, list) or any(not isinstance(job, dict) for job in jobs):
    raise RuntimeError("Unexpected Ray jobs API response")
parent = os.environ["CPT_PARENT_SUBMISSION_ID"]
active = [job.get("submission_id") for job in jobs if job.get("status") in ("RUNNING", "PENDING")
          and not (parent and job.get("submission_id") == parent)
          and job.get("entrypoint", "").strip() not in ("sleep inf", "sleep infinity")]
if active:
    raise RuntimeError(f"Cluster has active jobs; refusing submission: {active}")
PY_JOBS
GPU_COMPUTE_PIDS="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)"
if [[ -n "${GPU_COMPUTE_PIDS//[[:space:]]/}" ]]; then
    echo "GPU compute processes remain; refusing submission: $GPU_COMPUTE_PIDS" >&2
    exit 1
fi
RAY_ARGS=(job submit --address="$RAY_JOBS_ADDRESS" --submission-id="$SUBMISSION_ID"
    --runtime-env-json="$RUNTIME_ENV_JSON")
# An outer Ray Job must stay alive and report the child's actual terminal status.
if [[ -n "$RAY_NO_WAIT" || -n "$CPT_PARENT_SUBMISSION_ID" ]]; then RAY_ARGS+=(--no-wait); fi
if [[ -n "${WORKING_DIR:-}" ]]; then RAY_ARGS+=(--working-dir "$WORKING_DIR"); fi
mkdir -p "$SCRIPT_DIR/log"
submit_result=0
ray "${RAY_ARGS[@]}" -- "${TRAIN_ARGS[@]}" 2>&1 | tee "$SCRIPT_DIR/log/$TASK_NAME.log" || submit_result=$?
if (( submit_result != 0 )); then
    [[ -n "$CPT_PARENT_SUBMISSION_ID" ]] || exit "$submit_result"
    echo "Submission acknowledgement failed ($submit_result); checking the fixed child ID before exiting" >&2
fi

if [[ -n "$CPT_PARENT_SUBMISSION_ID" ]]; then
    echo "Training logs: ray job logs --address $RAY_JOBS_ADDRESS $SUBMISSION_ID --follow"
    python3 -B - <<'PY_WAIT'
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request

submission = os.environ["SUBMISSION_ID"]
url = (os.environ["RAY_JOBS_ADDRESS"].rstrip("/") + "/api/jobs/"
       + urllib.parse.quote(submission, safe=""))
print(f"Waiting for training Job {submission}; parent={os.environ['CPT_PARENT_SUBMISSION_ID']}", flush=True)
previous = None
last_report = 0.0
failures = 0
while True:
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            job = json.load(response)
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as error:
        failures += 1
        print(f"Training status unavailable ({failures}/60): {error}", flush=True)
        if failures >= 60:
            raise RuntimeError("Cannot determine training outcome; inspect the child Ray Job") from error
        time.sleep(15)
        continue
    failures = 0
    if job.get("submission_id") != submission:
        raise RuntimeError("Unexpected Ray Job identity in status response")
    status = job.get("status")
    if status != previous or time.monotonic() - last_report >= 60:
        print(f"Training Job {submission}: {status}", flush=True)
        previous = status
        last_report = time.monotonic()
    if status in ("SUCCEEDED", "FAILED", "STOPPED"):
        print(job.get("message") or "", flush=True)
        raise SystemExit(0 if status == "SUCCEEDED" else 1)
    if status not in ("PENDING", "RUNNING"):
        raise RuntimeError(f"Unexpected Ray Job status: {status}")
    time.sleep(15)
PY_WAIT
fi
