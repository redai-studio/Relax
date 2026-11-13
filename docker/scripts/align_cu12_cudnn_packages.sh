#!/usr/bin/env bash
set -euo pipefail

version="$1"
updates=()
held=()
for package in libcudnn9-cuda-12 libcudnn9-dev-cuda-12 libcudnn9-headers-cuda-12; do
    selection="$(dpkg-query -W -f='${db:Status-Want}:${db:Status-Status}:${Version}' "$package" 2>/dev/null || true)"
    IFS=: read -r want status installed_version <<< "$selection"
    if [[ "$status" == installed && "$installed_version" != "$version" ]]; then
        updates+=("$package=$version")
        if [[ "$want" == hold ]]; then
            held+=("$package")
        fi
    fi
done

if (("${#updates[@]}")); then
    apt-get update
    apt-get install -y --no-install-recommends --allow-change-held-packages "${updates[@]}"
    if (("${#held[@]}")); then
        apt-mark hold "${held[@]}"
    fi
    rm -rf /var/lib/apt/lists/*
fi
