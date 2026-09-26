#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.

set -euo pipefail

GYM_ROOT="${GYM_ROOT:-/opt/nemo-gym}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/nemo-gym}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
EXAMPLE_DIR="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"
DATASET_REPO="${WORKPLACE_ASSISTANT_DATASET_REPO:-nvidia/Nemotron-RL-agent-workplace_assistant}"
DATASET_SPLIT="${WORKPLACE_ASSISTANT_SPLIT:-train}"

case "${DATASET_SPLIT}" in
    train | validation)
        SOURCE_DATA="${OUTPUT_DIR}/workplace_assistant_${DATASET_SPLIT}.jsonl"
        ;;
    example)
        SOURCE_DATA="${OUTPUT_DIR}/workplace_assistant_example.jsonl"
        ;;
    *)
        echo "WORKPLACE_ASSISTANT_SPLIT must be train, validation, or example" >&2
        exit 2
        ;;
esac

RELAX_DATA="${OUTPUT_DIR}/workplace_assistant_${DATASET_SPLIT}_relax.jsonl"
mkdir -p "${OUTPUT_DIR}"

if [ "${DATASET_SPLIT}" = "example" ]; then
    cp "${GYM_ROOT}/resources_servers/workplace_assistant/data/example.jsonl" "${SOURCE_DATA}"
else
    "${GYM_ROOT}/.venv/bin/gym" dataset download \
        --repo-id "${DATASET_REPO}" \
        --artifact "${DATASET_SPLIT}.jsonl" \
        --output "${SOURCE_DATA}"
fi

/usr/bin/python3 \
    "${EXAMPLE_DIR}/scripts/convert_dataset.py" \
    --input "${SOURCE_DATA}" \
    --output "${RELAX_DATA}"

awk 'NF { count++ } END { print count, FILENAME }' "${SOURCE_DATA}"
awk 'NF { count++ } END { print count, FILENAME }' "${RELAX_DATA}"
