#!/usr/bin/env bash

LOG_PATH="/tmp/osworld_mcp_server.log"

: > "${LOG_PATH}"
nohup python server.py >>"${LOG_PATH}" 2>&1 &
echo "Started OSWorld MCP server. Logs: ${LOG_PATH}"
