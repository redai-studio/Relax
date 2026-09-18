#!/bin/bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Example environment. Copy to ignored env.local.sh and fill in your paths.
# Both entrypoints source env.local.sh first, then fill defaults from env.sh.

export MEGATRON="${MEGATRON:-/path/to/Megatron-LM}"
export MODEL_DIR="${MODEL_DIR:-/path/to/models}"  # contains Qwen3-VL-8B-Thinking
export DATA_DIR="${DATA_DIR:-/path/to/rl_data}"
export SAVE_DIR="${SAVE_DIR:-/path/to/checkpoints/osworld_agentic}"  # fresh run directory
export RAY_JOB_ADDRESS="${RAY_JOB_ADDRESS:-http://127.0.0.1:8265}"

# Nex-enabled OSWorld with HybridAgentLocal; see README for the sibling folders.
export OSWORLD_REPO="${OSWORLD_REPO:-/path/to/OSWorld-main-nex}"
export OSWORLD_CACHE_DIR="${OSWORLD_CACHE_DIR:-${OSWORLD_REPO}/cache}"
export MCP_SRC_ROOT="${MCP_SRC_ROOT:-/path/to/toolcua_mcp}"
export NEX_API_BASE="${NEX_API_BASE:-http://nex.example.internal/api/v1}"
export NEX_WORKSPACE_ID="${NEX_WORKSPACE_ID:-workspace-xxxxxxxx}"
export NEX_TEMPLATE="${NEX_TEMPLATE:-osworld-guest}"
# Provide NEX_API_KEY in env.local.sh or source your private credentials there.
# If needed, set download_proxy for evaluator downloads in env.local.sh.
# Guest uploads use the Nex intranet; avoid http_proxy / https_proxy for them.
