#!/bin/bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.

# Run through scripts/entrypoint/ray-job.sh inside an isolated GPU runtime.
# SMOKE_MODE: single (default), dp2 (colocate), async2 (one GPU per role).
set -euo pipefail

: "${MODEL_PATH:?Set the existing Qwen2.5-7B-Instruct directory}"
: "${OUTPUT_DIR:?Set a task-specific output directory}"
: "${RAY_ADDRESS:?Set the isolated Ray dashboard address}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
source "${REPO_ROOT}/scripts/models/qwen25-7B.sh"

RESOURCE='{"actor": [1, 1], "rollout": [1, 1]}'
MODE_ARGS=(--colocate --num-gpus-per-node 1)
PROMPT_BATCH=1
case "${SMOKE_MODE:-single}" in
    single) ;;
    dp2)
        RESOURCE='{"actor": [1, 2], "rollout": [1, 2]}'
        MODE_ARGS=(--colocate --num-gpus-per-node 2)
        PROMPT_BATCH=2
        ;;
    async2)
        RESOURCE='{"actor": [1, 1], "rollout": [1, 1], "advantages": [1, 0]}'
        MODE_ARGS=(--fully-async --num-gpus-per-node 2 --num-iters-per-train-update 1 --update-weights-interval 1)
        if [[ "${PARTIAL_SMOKE:-0}" == 1 ]]; then
            echo "PARTIAL_SMOKE file gates require colocate; run async2 without them." >&2
            exit 2
        fi
        ;;
    *) echo "Unknown SMOKE_MODE: ${SMOKE_MODE}" >&2; exit 2 ;;
esac

mkdir -p "${OUTPUT_DIR}"
python - "${PROMPT_BATCH}" <<'PY'
import json
import os
import sys
from pathlib import Path

batch = int(sys.argv[1])
rows = [
    {
        "prompt": [{"role": "user", "content": (
            "Invent a short, unusual nickname for a doubling tool and briefly introduce it. "
            f"Then say you will use it on the number {i + 1}."
        )}],
        "metadata": {
            "value": i + 1,
            "wait_for": ("train-1-0.pt", "long-ready", "train-0-0.pt", "release-extra")[min(i // batch, 3)],
            "signal_first_turn": i < batch,
            "first_turn_barrier_size": batch * 2,
        },
    }
    for i in range(8)
]
(Path(os.environ["OUTPUT_DIR"]) / "prompts.jsonl").write_text("\n".join(map(json.dumps, rows)) + "\n")
PY

PARTIAL_ARGS=()
DEFAULT_NUM_ROLLOUT=2
if [[ "${PARTIAL_SMOKE:-0}" == 1 ]]; then
    DEFAULT_NUM_ROLLOUT=3
    PARTIAL_ARGS+=(--partial-rollout --over-sampling-batch-size "$((PROMPT_BATCH * 3))" --partial-rollout-max-aborted-count 10)
    PARTIAL_ARGS+=(--agent-env "TASK15_CONTROL_DIR=${OUTPUT_DIR}")
fi

RAY_NO_WAIT=1 ray job submit --no-wait --address="${RAY_ADDRESS}" \
    --runtime-env-json="${RUNTIME_ENV_JSON:-\{\}}" \
    -- python3 -m relax.entrypoints.train \
    --resource "${RESOURCE}" "${MODE_ARGS[@]}" \
    --max-staleness 0 --num-data-storage-units 1 \
    "${MODEL_ARGS[@]}" \
    --hf-checkpoint "${MODEL_PATH}" --megatron-to-hf-mode bridge \
    --prompt-data "${OUTPUT_DIR}/prompts.jsonl" --input-key prompt --metadata-key metadata \
    --use-agentic-rollout \
    --agent-command "python -m examples.agentic_weight_audit.agent" --agent-cwd "${REPO_ROOT}" \
    --agent-timeout 600 --agentic-concurrency "$((PROMPT_BATCH * 3))" --sglang-server-concurrency "$((PROMPT_BATCH * 6))" \
    --num-rollout "${NUM_ROLLOUT:-${DEFAULT_NUM_ROLLOUT}}" --rollout-batch-size "${PROMPT_BATCH}" --n-samples-per-prompt 2 \
    --global-batch-size "$((PROMPT_BATCH * 2))" --micro-batch-size 1 --rollout-max-response-len 64 --rollout-max-prompt-len 256 \
    --rollout-temperature 1.0 --advantage-estimator grpo --kl-coef 0 --entropy-coef 0 \
    --optimizer adam --optimizer-cpu-offload --optimizer-offload-fraction 1 --use-precision-aware-optimizer \
    --lr 1e-6 --lr-decay-style constant --weight-decay 0 \
    --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 --context-parallel-size 1 \
    --use-dynamic-batch-size --max-tokens-per-gpu 512 --log-probs-max-tokens-per-gpu 512 \
    --seq-length 512 --attention-backend flash --attention-dropout 0 --hidden-dropout 0 \
    --rollout-num-gpus-per-engine 1 --sglang-mem-fraction-static 0.35 \
    --sglang-context-length 1024 --sglang-disable-cuda-graph \
    --skip-eval-before-train \
    --save-debug-rollout-data "${OUTPUT_DIR}/rollout-{rollout_id}.pt" \
    --save-debug-train-data "${OUTPUT_DIR}/train-{rollout_id}-{rank}.pt" \
    "${PARTIAL_ARGS[@]}" "$@"
