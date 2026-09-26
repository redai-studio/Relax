#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

prepare_inside_container() {
    local gym_root="${GYM_ROOT:-/opt/nemo-gym}"
    local data_dir="${DATA_DIR:-/data/nemo-gym/multienv}"
    local math_raw="${data_dir}/math_with_judge_train.jsonl"
    local workplace_raw="${data_dir}/workplace_assistant_train.jsonl"
    local merged="${NEMO_GYM_SOURCE_DATA:-${data_dir}/multienv_train_raw.jsonl}"

    mkdir -p "${data_dir}"
    if [ ! -s "${math_raw}" ]; then
        "${gym_root}/.venv/bin/gym" dataset download \
            --repo-id nvidia/Nemotron-RL-math-OpenMathReasoning \
            --artifact train.jsonl \
            --output "${math_raw}"
    fi
    if [ ! -s "${workplace_raw}" ]; then
        "${gym_root}/.venv/bin/gym" dataset download \
            --repo-id nvidia/Nemotron-RL-agent-workplace_assistant \
            --artifact train.jsonl \
            --output "${workplace_raw}"
    fi

    /usr/bin/python3 - \
        "${math_raw}" \
        "${workplace_raw}" \
        "${merged}" \
        "${MULTIENV_MATH_TARGET_LINES:-2400}" \
        "${MULTIENV_WORKPLACE_TARGET_LINES:-0}" <<'PY'
import json
import os
import random
import sys
from pathlib import Path

math_path, workplace_path, output_path = map(Path, sys.argv[1:4])
math_limit, workplace_limit = map(int, sys.argv[4:6])
rng = random.Random(20260829)


def load(path, *, environment, config, limit, require_ground_truth=False):
    rows = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError(f"{path}:{line_number} must contain a JSON object")
            if require_ground_truth and not row.get("ground_truth"):
                continue
            row["environment"] = environment
            row["config"] = config
            row["data_source"] = environment
            rows.append(row)
    return rows[:limit] if limit > 0 else rows


math_rows = load(
    math_path,
    environment="math_with_judge",
    config="math-with-judge-v1",
    limit=math_limit,
)
workplace_rows = load(
    workplace_path,
    environment="workplace_assistant",
    config="workplace-assistant-v1",
    limit=workplace_limit,
    require_ground_truth=True,
)
if not math_rows or not workplace_rows:
    raise RuntimeError(
        f"both environments need rows: math={len(math_rows)} workplace={len(workplace_rows)}"
    )

merged = math_rows + workplace_rows
rng.shuffle(merged)

temporary = output_path.with_suffix(output_path.suffix + ".tmp")
with temporary.open("w", encoding="utf-8") as destination:
    for row in merged:
        destination.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
os.replace(temporary, output_path)
print(f"prepared math={len(math_rows)} workplace={len(workplace_rows)} total={len(merged)}")
PY
}

if [ "${MULTIENV_PREPARE_IN_CONTAINER:-0}" = "1" ]; then
    prepare_inside_container
    exit 0
fi

: "${NEMO_GYM_IMAGE:?NEMO_GYM_IMAGE must name the pinned NeMo Gym image}"
DATA_DIR="${DATA_DIR:-/data/nemo-gym/multienv}"
NEMO_GYM_SOURCE_DATA="${NEMO_GYM_SOURCE_DATA:-${DATA_DIR}/multienv_train_raw.jsonl}"
case "${NEMO_GYM_SOURCE_DATA}" in
    /*/multienv_train_raw.jsonl) ;;
    *)
        echo "NEMO_GYM_SOURCE_DATA must be an absolute path ending in /multienv_train_raw.jsonl" >&2
        exit 2
        ;;
esac
DATA_DIR="$(dirname -- "${NEMO_GYM_SOURCE_DATA}")"
REPO_ROOT="${RELAX_REPO_ROOT:-$(cd -- "${SCRIPT_DIR}/../../../.." &>/dev/null && pwd)}"
mkdir -p "${DATA_DIR}"

docker image inspect "${NEMO_GYM_IMAGE}" >/dev/null
docker run --rm \
    --network host \
    --entrypoint bash \
    --mount "type=bind,source=${REPO_ROOT},target=/opt/relax-integration,readonly" \
    --mount "type=bind,source=${DATA_DIR},target=${DATA_DIR}" \
    --env MULTIENV_PREPARE_IN_CONTAINER=1 \
    --env DATA_DIR="${DATA_DIR}" \
    --env NEMO_GYM_SOURCE_DATA="${NEMO_GYM_SOURCE_DATA}" \
    --env MULTIENV_MATH_TARGET_LINES="${MULTIENV_MATH_TARGET_LINES:-2400}" \
    --env MULTIENV_WORKPLACE_TARGET_LINES="${MULTIENV_WORKPLACE_TARGET_LINES:-0}" \
    --env HTTP_PROXY="${http_proxy:-${HTTP_PROXY:-}}" \
    --env HTTPS_PROXY="${https_proxy:-${HTTPS_PROXY:-}}" \
    --env NO_PROXY="${no_proxy:-${NO_PROXY:-127.0.0.1,localhost}}" \
    "${NEMO_GYM_IMAGE}" \
    /opt/relax-integration/examples/nemo_gym_agentic/recipes/multienv-math-workplace/prepare_multienv.sh

test -s "${NEMO_GYM_SOURCE_DATA}"
