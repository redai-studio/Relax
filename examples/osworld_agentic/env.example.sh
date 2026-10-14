#!/bin/bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Copy to env.local.sh, fill in deployment-specific settings, and chmod 600.
# Never commit env.local.sh. MODEL_DIR, DATA_DIR and SAVE_DIR are supplied at launch.

export OSWORLD_REPO="/path/to/OSWorld-main-nex"
export OSWORLD_CACHE_DIR="${OSWORLD_REPO}/cache"
# Repository root containing mcp/, NOT the mcp/ directory itself.
export MCP_SRC_ROOT="${OSWORLD_REPO}"
# Use a separate OSWorld environment to avoid replacing training dependencies.
export OSWORLD_AGENT_PYTHON="/path/to/osworld-venv/bin/python"

export NEX_API_BASE="https://nex.example.com/api/v1"
export NEX_WORKSPACE_ID="your-workspace-id"
export NEX_TEMPLATE="your-osworld-template"
# Alternatively provide NEX_TEMPLATE_ID if supported by your Nex deployment.
# Set a real key locally, or source it from your secret manager before launch.
export NEX_API_KEY="${NEX_API_KEY:-}"

# Optional evaluator download proxy; guest traffic must remain directly reachable.
# export download_proxy="http://your-proxy:port"
# Configure ClearML on all training nodes (clearml-init or your secret manager).
# Do not add its credentials to the committed template or runtime-env JSON.
