#!/usr/bin/env bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.

set -euo pipefail

exec python3 "$(dirname -- "${BASH_SOURCE[0]}")/agent_client.py" "$@"
