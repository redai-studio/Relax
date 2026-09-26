#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
RELAX_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." &>/dev/null && pwd)"

gateway_url="$1"
source_data="$2"
prompt_data="$3"
data_limit="$4"
shift 4

until curl --noproxy "*" --fail --silent "${gateway_url}/readyz" >/dev/null; do
    echo "Waiting for NeMo Gym Gateway at ${gateway_url}"
    sleep 2
done

/usr/bin/python3 "${SCRIPT_DIR}/convert_dataset.py" \
    --input "${source_data}" \
    --output "${prompt_data}" \
    --limit "${data_limit}"

cd "${RELAX_ROOT}"
exec /usr/bin/python3 -m relax.entrypoints.train "$@"
