#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

prepare_inside_container() {
    GYM_ROOT="${GYM_ROOT:-/opt/nemo-gym}"
    OUTPUT_DIR="${OUTPUT_DIR:-/data/nemo-gym}"
    CALENDAR_DATASET_REPO="${CALENDAR_DATASET_REPO:-nvidia/Nemotron-RL-agent-calendar_scheduling}"
    CALENDAR_SPLIT="${CALENDAR_SPLIT:-train}"
    RELAX_INTEGRATION_ROOT="${RELAX_INTEGRATION_ROOT:-/opt/relax-integration}"
    EXAMPLE_DIR="${RELAX_INTEGRATION_ROOT}/examples/nemo_gym_agentic"

    case "${CALENDAR_SPLIT}" in
        train | validation)
            SOURCE_DATA="${OUTPUT_DIR}/calendar_${CALENDAR_SPLIT}.jsonl"
            ;;
        example)
            SOURCE_DATA="${OUTPUT_DIR}/calendar_example.jsonl"
            ;;
        *)
            echo "CALENDAR_SPLIT must be train, validation, or example" >&2
            exit 2
            ;;
    esac

    RELAX_DATA="${OUTPUT_DIR}/calendar_${CALENDAR_SPLIT}_relax.jsonl"
    mkdir -p "${OUTPUT_DIR}"

    if [ "${CALENDAR_SPLIT}" = "example" ]; then
        cp "${GYM_ROOT}/environments/calendar/data/example.jsonl" "${SOURCE_DATA}"
    else
        "${GYM_ROOT}/.venv/bin/gym" dataset download \
            --repo-id "${CALENDAR_DATASET_REPO}" \
            --artifact "${CALENDAR_SPLIT}.jsonl" \
            --output "${SOURCE_DATA}"
    fi

    /usr/bin/python3 \
        "${EXAMPLE_DIR}/scripts/convert_dataset.py" \
        --input "${SOURCE_DATA}" \
        --output "${RELAX_DATA}"

    awk 'NF { count++ } END { print count, FILENAME }' "${SOURCE_DATA}"
    awk 'NF { count++ } END { print count, FILENAME }' "${RELAX_DATA}"
}

if [ "${CALENDAR_PREPARE_IN_CONTAINER:-0}" = "1" ]; then
    prepare_inside_container
    exit 0
fi

if [ -f "${SCRIPT_DIR}/env.sh" ]; then
    # shellcheck disable=SC1091
    source "${SCRIPT_DIR}/env.sh"
fi

: "${NEMO_GYM_SOURCE_DATA:?Set NEMO_GYM_SOURCE_DATA in ${SCRIPT_DIR}/env.sh}"
NEMO_GYM_IMAGE="${NEMO_GYM_IMAGE:-relax-nemo-gym:calendar-local-20260809}"
OUTPUT_DIR="$(dirname -- "${NEMO_GYM_SOURCE_DATA}")"
EXPECTED_SOURCE_DATA="${OUTPUT_DIR}/calendar_train.jsonl"
RELAX_DATA="${OUTPUT_DIR}/calendar_train_relax.jsonl"

if [ "${NEMO_GYM_SOURCE_DATA}" != "${EXPECTED_SOURCE_DATA}" ]; then
    echo "NEMO_GYM_SOURCE_DATA must end with /calendar_train.jsonl" >&2
    exit 2
fi

command -v docker >/dev/null
docker image inspect "${NEMO_GYM_IMAGE}" >/dev/null
mkdir -p "${OUTPUT_DIR}"

docker run --rm \
    --network host \
    --entrypoint bash \
    --mount "type=bind,source=${OUTPUT_DIR},target=${OUTPUT_DIR}" \
    -e CALENDAR_PREPARE_IN_CONTAINER=1 \
    -e CALENDAR_SPLIT=train \
    -e OUTPUT_DIR="${OUTPUT_DIR}" \
    -e HTTP_PROXY="${http_proxy:-${HTTP_PROXY:-}}" \
    -e HTTPS_PROXY="${https_proxy:-${HTTPS_PROXY:-}}" \
    -e NO_PROXY="${no_proxy:-${NO_PROXY:-127.0.0.1,localhost}}" \
    "${NEMO_GYM_IMAGE}" \
    /opt/relax-integration/examples/nemo_gym_agentic/recipes/calendar/prepare_calendar.sh

test -s "${NEMO_GYM_SOURCE_DATA}"
test -s "${RELAX_DATA}"
echo "Calendar data is ready:"
wc -l "${NEMO_GYM_SOURCE_DATA}" "${RELAX_DATA}"
