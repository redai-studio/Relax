#!/bin/bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# End-to-end single-VM smoke test — no Relax rollout worker involved.
#
# The point is to catch problems in the session subprocess itself before
# hitting the Ray/agentic path:
#
#   * Vendor imports resolve on PYTHONPATH.
#   * DesktopEnv can pull the docker image + mount the qcow2.
#   * HybridAgentLocal + our RelaxHybridAgent talk to whatever OpenAI-shaped
#     endpoint you point OPENAI_BASE_URL at (any OpenAI-compatible server works —
#     we simulate Relax's AgenticChatAPIService by exporting the same env vars).
#   * The reward extracted from evaluate() flows through the output json.
#
# Usage:
#   MODEL_DIR=... OSWORLD_TASK_JSON=/path/to/a/task.json \
#   OPENAI_BASE_URL=http://... OPENAI_API_KEY=... \
#   bash examples/osworld_agentic/scripts/smoke_test.sh

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
APP_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

: "${OSWORLD_TASK_JSON:?Set OSWORLD_TASK_JSON to a single OSWorld task JSON.}"
: "${OPENAI_BASE_URL:?Set OPENAI_BASE_URL to a chat-completions endpoint.}"
: "${OPENAI_API_KEY:?Set OPENAI_API_KEY for the endpoint.}"

# Mimic the env Relax injects into the session subprocess. The trailing-slash
# strip + /v1 postfix live in run_agent_app.sh proper; here we let the caller
# hand us the exact URL to reduce moving parts.
export RELAX_BASE_URL="${OPENAI_BASE_URL%/v1}"    # run_agent_app.sh reappends /v1
export RELAX_SESSION_ID="${RELAX_SESSION_ID:-smoke-$(date +%s)}"
export OPENAI_MODEL="${OPENAI_MODEL:-Qwen3-VL-8B-Thinking}"
export OSWORLD_CONFIG_PATH="${OSWORLD_CONFIG_PATH:-${APP_DIR}/app/osworld_config.yaml}"
export OSWORLD_CACHE_DIR="${OSWORLD_CACHE_DIR:-/tmp/osworld_cache}"

TMPDIR_SMOKE=$(mktemp -d -t osworld-smoke-XXXXXX)
trap 'rm -rf "${TMPDIR_SMOKE}"' EXIT

INPUT_JSON="${TMPDIR_SMOKE}/input.json"
OUTPUT_JSON="${TMPDIR_SMOKE}/output.json"

# Build the session-input envelope agent.py expects. task_config lives at the
# top level so the second read-path (metadata.task_config) doesn't need to fire.
python3 - "${OSWORLD_TASK_JSON}" "${INPUT_JSON}" <<'PY'
import json, sys
task_cfg = json.loads(open(sys.argv[1], encoding="utf-8").read())
payload = {
    "task_config": task_cfg,
    "metadata": {
        "data_source": "osworld",
        "instruction": task_cfg.get("instruction", ""),
        "task_config": task_cfg,
    },
    "prompt": [{"role": "user", "content": task_cfg.get("instruction", "")}],
}
open(sys.argv[2], "w", encoding="utf-8").write(json.dumps(payload))
PY

export RELAX_INPUT_JSON="${INPUT_JSON}"
export RELAX_OUTPUT_JSON="${OUTPUT_JSON}"

echo "== running one session =="
echo "  task json:      ${OSWORLD_TASK_JSON}"
echo "  base url:       ${RELAX_BASE_URL}"
echo "  osworld config: ${OSWORLD_CONFIG_PATH}"
echo "  session id:     ${RELAX_SESSION_ID}"

# `.` (source) matches how Relax launches the agent command, so we exercise
# the same PYTHONPATH+exec plumbing.
(
    cd "${APP_DIR}"
    . ./run_agent_app.sh
)

echo "== session output =="
python3 - "${OUTPUT_JSON}" <<'PY'
import json, sys
raw = open(sys.argv[1], encoding="utf-8").read().strip()
n_records = 0
try:
    data = json.loads(raw)
except json.JSONDecodeError:
    # Explicit export: one JSONL record per agent turn (PITFALLS.md section 11).
    records = [json.loads(line) for line in raw.splitlines() if line.strip()]
    n_records = len(records)
    data = {"reward": records[0].get("reward"), "metadata": records[0].get("metadata", {})}
summary = {k: v for k, v in data.items() if k != "messages"}
print(json.dumps(summary, indent=2, ensure_ascii=False)[:2000])
print(f"export_records={n_records}")
reward = data.get("reward")
meta = data.get("metadata", {})
turns = meta.get("agent_turns")
stop = meta.get("stop_reason")
last_err = meta.get("last_error")
if last_err:
    print(f"\n!! last_error: {last_err}", file=sys.stderr)
    sys.exit(1)
if reward is None or turns is None:
    print("!! missing reward or agent_turns", file=sys.stderr)
    sys.exit(1)
print(f"\nOK — reward={reward} turns={turns} stop_reason={stop}")
PY
