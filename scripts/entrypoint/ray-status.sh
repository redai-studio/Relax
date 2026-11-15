#!/usr/bin/env bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Source after runtime-env construction: wrap local CLI calls, not Ray workers.
_RELAX_RAY_BIN="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/ray-bin" && pwd)"
if [[ "$(type -P ray)" != "$_RELAX_RAY_BIN/ray" ]]; then
    # A parent shell may already use the wrapper from another checkout.
    if [[ "$(type -P ray)" != */ray-bin/ray ]]; then
        RELAX_REAL_RAY="$(type -P ray)"
    elif [[ ! -x "${RELAX_REAL_RAY:-}" || "$RELAX_REAL_RAY" == */ray-bin/ray ]]; then
        echo 'ERROR: existing Ray wrapper has no native executable' >&2
        return 1
    fi
    if [[ -z "$RELAX_REAL_RAY" ]]; then
        echo 'ERROR: ray executable not found in PATH' >&2
        return 1
    fi
    RELAX_REAL_RAY="$(cd -- "$(dirname -- "$RELAX_REAL_RAY")" && pwd)/$(basename -- "$RELAX_REAL_RAY")"
    export RELAX_REAL_RAY
    export PATH="$_RELAX_RAY_BIN:$PATH"
    hash -r
fi
unset _RELAX_RAY_BIN
