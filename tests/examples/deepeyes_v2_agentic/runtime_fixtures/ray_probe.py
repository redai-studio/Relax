# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


parser = argparse.ArgumentParser()
job = parser.add_subparsers(required=True).add_parser("job")
submit = job.add_subparsers(required=True).add_parser("submit")
submit.add_argument("--runtime-env-json", required=True)
submit.add_argument("--address")
submit.add_argument("--no-wait", action="store_true")
submit.add_argument("entrypoint", nargs=argparse.REMAINDER)
args = parser.parse_args()
entry_parser = argparse.ArgumentParser()
entry_parser.add_argument("--agent-command", required=True)
entry_parser.add_argument("--agent-cwd", required=True)
entry_parser.add_argument("--agent-env", action="append", nargs="+", required=True)
entry_args, _ = entry_parser.parse_known_args(args.entrypoint[1:])
assert len(entry_args.agent_env) == 1
agent_env = dict(value.split("=", 1) for value in entry_args.agent_env[0])
assert set(agent_env) == {"AGENT_DEBUG_LOG_DIR", "DEEPEYES_V2_SEARCH_CONFIG_PATH"}
runtime = json.loads(args.runtime_env_json)
assert agent_env["DEEPEYES_V2_SEARCH_CONFIG_PATH"] == runtime["env_vars"]["DEEPEYES_V2_SEARCH_CONFIG_PATH"]
Path(os.environ["SEARCH_RUNTIME_TEST_RUNTIME_REPORT"]).write_text(json.dumps(runtime), encoding="utf-8")
environment = {
    key: os.environ[key] for key in ("PATH", "TMPDIR", "SEARCH_RUNTIME_TEST_PYTHON", "PYTHONDONTWRITEBYTECODE")
}
environment.update(runtime["env_vars"])
environment.update(agent_env)
environment.update(
    RELAX_BASE_URL="https://model.example.test/v1",
    RELAX_SESSION_ID="fixture-session",
    RELAX_INPUT_JSON=os.environ["SEARCH_RUNTIME_TEST_AGENT_INPUT"],
    RELAX_OUTPUT_JSON=os.environ["SEARCH_RUNTIME_TEST_AGENT_OUTPUT"],
)
process = subprocess.run(
    ["bash", "-c", entry_args.agent_command],
    cwd=entry_args.agent_cwd,
    env=environment,
    timeout=20,
)
sys.exit(process.returncode)
