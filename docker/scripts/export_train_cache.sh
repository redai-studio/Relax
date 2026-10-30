#!/usr/bin/env bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
set -euo pipefail

if [[ $# != 2 ]]; then
    echo "Usage: $0 TRAIN_IMAGE OUTPUT.tar" >&2
    exit 2
fi

image="$1"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$(dirname "$2")"
output="$(cd "$(dirname "$2")" && pwd)/$(basename "$2")"
if [[ -e "$output" ]]; then
    echo "Output already exists: $output" >&2
    exit 1
fi

staging="$(mktemp -d "$(dirname "$output")/.train-cache.XXXXXX")"
container=""
cleanup() {
    if [[ -n "$container" ]]; then
        docker rm -f "$container" >/dev/null
    fi
    rm -rf "$staging"
}
trap cleanup EXIT

# docker cp also works when the Docker daemon runs on another host.
container="$(docker create --entrypoint /bin/bash "$image" -lc '
    set -euo pipefail
    python /tmp/export_train_wheels.py export /tmp/train-cache
    python /tmp/export_train_wheels.py check /tmp/train-cache
    tar -C /tmp/train-cache -cf /tmp/train-cache.tar .
')"
docker cp "$script_dir/export_train_wheels.py" "$container:/tmp/export_train_wheels.py"
docker start --attach "$container"
test "$(docker inspect --format '{{.State.ExitCode}}' "$container")" = 0
docker cp "$container:/tmp/train-cache.tar" "$staging/cache.tar"
mv "$staging/cache.tar" "$output"
sha256sum "$output"
