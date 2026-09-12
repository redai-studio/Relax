#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.

# Prepare reasoning-gym-cc data:
#
#   * Materialize a NeMo Gym reasoning_gym JSONL at ${NEMO_GYM_SOURCE_DATA}.
#     The default train split downloads the NeMo Gym HF artifact and is the
#     only split intended for training. The optional example split is shipped
#     in environments/reasoning_gym/data/example.jsonl (5 smoke rows, no
#     download required).
#   * Strip the create_dataset.py hardcoded ``agent_ref`` field if present
#     (Hydra would otherwise fail with ``Missing key
#     reasoning_gym_simple_agent`` at rollout time).
#   * Convert to the Relax rollout schema for spot-checks.
#
# Two invocations:
#   * On the host: launches a fresh, network-host, bind-mount container
#     (mirrors the calendar recipe's shape).
#   * Inside the container: called by the host wrapper via
#     REASONING_GYM_CC_PREPARE_IN_CONTAINER=1.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

prepare_inside_container() {
    GYM_ROOT="${GYM_ROOT:-/opt/nemo-gym}"
    OUTPUT_DIR="${OUTPUT_DIR:-/data/nemo-gym}"
    RELAX_INTEGRATION_ROOT="${RELAX_INTEGRATION_ROOT:-/opt/relax-integration}"
    EXAMPLE_DIR="${RELAX_INTEGRATION_ROOT}/examples/nemo_gym_agentic"
    SPLIT="${REASONING_GYM_CC_SPLIT:-train}"

    case "${SPLIT}" in
        example)
            SOURCE_DATA="${OUTPUT_DIR}/reasoning_gym_cc_example.jsonl"
            SRC_JSONL="${GYM_ROOT}/environments/reasoning_gym/data/example.jsonl"
            ;;
        train)
            SOURCE_DATA="${OUTPUT_DIR}/reasoning_gym_cc_train.jsonl"
            SRC_JSONL=""
            ;;
        *)
            echo "REASONING_GYM_CC_SPLIT must be example or train" >&2
            exit 2
            ;;
    esac

    RELAX_DATA="${OUTPUT_DIR}/reasoning_gym_cc_${SPLIT}_relax.jsonl"
    mkdir -p "${OUTPUT_DIR}"

    RAW_DATA="${OUTPUT_DIR}/.reasoning_gym_cc_${SPLIT}_raw.jsonl"
    if [ "${SPLIT}" = "example" ]; then
        cp "${SRC_JSONL}" "${RAW_DATA}"
    else
        # Same repo pinned by resources_servers/reasoning_gym/configs/;
        # override via env if you need a private mirror.
        REPO_ID="${REASONING_GYM_CC_DATASET_REPO:-nvidia/Nemotron-RL-ReasoningGym-v1}"
        ARTIFACT="${REASONING_GYM_CC_DATASET_ARTIFACT:-data/train.jsonl}"
        "${GYM_ROOT}/.venv/bin/gym" dataset download \
            --repo-id "${REPO_ID}" \
            --artifact "${ARTIFACT}" \
            --output "${RAW_DATA}"
    fi

    # Drop the create_dataset.py hardcoded agent_ref key. Validate the exact
    # verifier contract before atomically publishing the raw training file.
    SOURCE_TMP="${SOURCE_DATA}.tmp.${BASHPID}.${RANDOM}"
    /usr/bin/python3 - "${RAW_DATA}" "${SOURCE_TMP}" <<'PY'
import json
import sys

src, dst = sys.argv[1], sys.argv[2]
kept = 0
with open(src, encoding="utf-8") as fin, open(dst, "w", encoding="utf-8") as fout:
    for line_number, line in enumerate(fin, start=1):
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"row {line_number} must be a JSON object")
        responses_params = row.get("responses_create_params")
        metadata = row.get("metadata")
        if not isinstance(responses_params, dict) or not isinstance(responses_params.get("input"), list):
            raise ValueError(f"row {line_number} is missing responses_create_params.input")
        if not isinstance(row.get("question"), str) or "answer" not in row:
            raise ValueError(f"row {line_number} is missing question/answer")
        if not isinstance(metadata, dict) or not isinstance(metadata.get("source_dataset"), str):
            raise ValueError(f"row {line_number} is missing metadata.source_dataset")
        row.pop("agent_ref", None)
        fout.write(json.dumps(row, ensure_ascii=False) + "\n")
        kept += 1
if kept == 0:
    raise ValueError(f"no non-empty rows found in {src}")
print(f"stripped agent_ref from {kept} rows -> {dst}")
PY
    mv "${SOURCE_TMP}" "${SOURCE_DATA}"
    rm -f "${RAW_DATA}"

    /usr/bin/python3 \
        "${EXAMPLE_DIR}/scripts/convert_dataset.py" \
        --input "${SOURCE_DATA}" \
        --output "${RELAX_DATA}"

    SOURCE_ROWS="$(awk 'NF { count++ } END { print count+0 }' "${SOURCE_DATA}")"
    RELAX_ROWS="$(awk 'NF { count++ } END { print count+0 }' "${RELAX_DATA}")"
    if [ "${SOURCE_ROWS}" -ne "${RELAX_ROWS}" ]; then
        echo "row-count mismatch: raw=${SOURCE_ROWS} converted=${RELAX_ROWS}" >&2
        exit 2
    fi
    echo "validated ${SOURCE_ROWS} reasoning-gym-cc rows"
}

if [ "${REASONING_GYM_CC_PREPARE_IN_CONTAINER:-0}" = "1" ]; then
    prepare_inside_container
    exit 0
fi

: "${DATA_DIR:?Set DATA_DIR to the output directory for this recipe}"
DATA_DIR="${DATA_DIR%/}"
SPLIT="${REASONING_GYM_CC_SPLIT:-train}"
NEMO_GYM_SOURCE_DATA="${DATA_DIR}/reasoning_gym_cc_${SPLIT}.jsonl"
: "${NEMO_GYM_IMAGE:=relax-nemo-gym:a85670e}"

case "${DATA_DIR}" in /*) ;;
    *) echo "DATA_DIR must be absolute: ${DATA_DIR}" >&2; exit 2 ;;
esac

OUTPUT_DIR="${DATA_DIR}"
command -v docker >/dev/null
docker image inspect "${NEMO_GYM_IMAGE}" >/dev/null
mkdir -p "${OUTPUT_DIR}"

# REPO_ROOT is opt-in: if set (via env or --repo-dir-style export), bind-mount
# this Relax checkout at /opt/relax-integration so recipe edits take effect
# without rebuilding the image. It MUST be the docker daemon's absolute path
# (differs from the shell path in dev containers). If unset, the container
# uses the recipe scripts baked into the image by the Dockerfile's COPY.
repo_mount_args=()
if [ -n "${REPO_ROOT:-}" ]; then
    case "${REPO_ROOT}" in /*) ;;
        *) echo "REPO_ROOT must be absolute: ${REPO_ROOT}" >&2; exit 2 ;;
    esac
    if ! docker run --rm \
        --network none \
        --entrypoint bash \
        --mount "type=bind,source=${REPO_ROOT},target=/repo,readonly" \
        "${NEMO_GYM_IMAGE}" \
        -ceu 'test -s /repo/examples/nemo_gym_agentic/recipes/reasoning-gym-cc/prepare_reasoning_gym_cc.sh'; then
        echo "REPO_ROOT ${REPO_ROOT} is not visible to the Docker daemon." >&2
        echo "Pass the daemon-visible absolute path (outer host, not dev container), or unset REPO_ROOT to use the image's baked-in recipe files." >&2
        exit 2
    fi
    repo_mount_args+=(--mount "type=bind,source=${REPO_ROOT},target=/opt/relax-integration,readonly")
fi

docker run --rm \
    --network host \
    --entrypoint bash \
    --mount "type=bind,source=${OUTPUT_DIR},target=${OUTPUT_DIR}" \
    "${repo_mount_args[@]}" \
    -e REASONING_GYM_CC_PREPARE_IN_CONTAINER=1 \
    -e REASONING_GYM_CC_SPLIT="${SPLIT}" \
    -e REASONING_GYM_CC_DATASET_REPO="${REASONING_GYM_CC_DATASET_REPO:-}" \
    -e REASONING_GYM_CC_DATASET_ARTIFACT="${REASONING_GYM_CC_DATASET_ARTIFACT:-}" \
    -e OUTPUT_DIR="${OUTPUT_DIR}" \
    -e HTTP_PROXY="${http_proxy:-${HTTP_PROXY:-}}" \
    -e HTTPS_PROXY="${https_proxy:-${HTTPS_PROXY:-}}" \
    -e NO_PROXY="${no_proxy:-${NO_PROXY:-127.0.0.1,localhost}}" \
    "${NEMO_GYM_IMAGE}" \
    /opt/relax-integration/examples/nemo_gym_agentic/recipes/reasoning-gym-cc/prepare_reasoning_gym_cc.sh

RELAX_DATA="${OUTPUT_DIR}/reasoning_gym_cc_${SPLIT}_relax.jsonl"
for required_path in "${NEMO_GYM_SOURCE_DATA}" "${RELAX_DATA}"; do
    if [ ! -s "${required_path}" ]; then
        echo "ERROR: prepared file is missing or empty: ${required_path}" >&2
        exit 2
    fi
done
echo "reasoning-gym-cc data is ready:"
wc -l "${NEMO_GYM_SOURCE_DATA}" "${RELAX_DATA}"
