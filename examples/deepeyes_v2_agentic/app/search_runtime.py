# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from pathlib import Path
from typing import Any

from pydantic import ValidationError


EXAMPLE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EXAMPLE_DIR))

from app.search_config import SEARCH_CONFIG_ENV, ExternalSearchConfig, SearchError, load_search_config  # noqa: E402


EXAMPLE_ENV_NAMES = (
    "DEEPEYES_V2_SEARCH_CACHE_PATHS",
    "DEEPEYES_JUDGE_BASE_URL",
    "DEEPEYES_JUDGE_MODELS",
    "DEEPEYES_JUDGE_API_KEY",
    "APPTAINER_IMAGE_PATH",
    "DEEPEYES_V2_APP_PYTHON",
)
RESERVED_AUTH_NAMES = {
    SEARCH_CONFIG_ENV,
    *EXAMPLE_ENV_NAMES,
    "SANDBOX_BACKEND",
    "SANDBOX_CONFIG_PATH",
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "OPENAI_MODEL",
    "AGENT_DEBUG_LOG_DIR",
    "RUNTIME_ENV_JSON",
    "DEEPEYES_V2_BASE_RUNTIME_ENV_JSON",
    "XMLIR_ENABLE_H2D_SSE_COPY",
    "USE_CAST_FC_FUSION",
    "WANDB_API_KEY",
}


def prepare_search_environment() -> dict[str, str]:
    config_path = os.environ.get(SEARCH_CONFIG_ENV, str(EXAMPLE_DIR / "search_config.mock.yaml"))
    if not config_path.strip():
        raise SearchError("invalid_config_file")
    try:
        absolute_path = str(Path(config_path).resolve(strict=True))
    except (OSError, RuntimeError, ValueError):
        raise SearchError("invalid_config_file") from None
    os.environ[SEARCH_CONFIG_ENV] = absolute_path
    try:
        config = load_search_config()
    except (SearchError, ValidationError):
        raise SearchError("invalid_search_config") from None

    values = {SEARCH_CONFIG_ENV: absolute_path}
    if isinstance(config, ExternalSearchConfig) and config.auth is not None:
        name = config.auth.env
        if name in RESERVED_AUTH_NAMES or name.startswith("RELAX_") or "=" in name or "\0" in name:
            raise SearchError("invalid_search_auth_env")
        value = os.environ.get(name)
        if value is None or not value.strip():
            raise SearchError("missing_search_auth")
        values[name] = value
    return values


def _read_runtime_environment(name: str) -> dict[str, Any]:
    raw = os.environ.get(name)
    if raw is None:
        return {}
    try:
        value = json.loads(raw)
    except (ValueError, RecursionError):
        raise SearchError("invalid_runtime_environment") from None
    if not isinstance(value, dict):
        raise SearchError("invalid_runtime_environment")
    env_vars = value.get("env_vars", {})
    if not isinstance(env_vars, dict) or any(
        not isinstance(key, str) or not isinstance(item, str) for key, item in env_vars.items()
    ):
        raise SearchError("invalid_runtime_environment")
    return value


def build_runtime_environment(profile: str) -> dict[str, Any]:
    if profile not in {"standard", "klx"}:
        raise SearchError("invalid_runtime_profile")
    base = _read_runtime_environment("DEEPEYES_V2_BASE_RUNTIME_ENV_JSON")
    current = _read_runtime_environment("RUNTIME_ENV_JSON")
    env_vars = {**base.get("env_vars", {}), **current.get("env_vars", {})}
    env_vars.update({name: os.environ.get(name, "") for name in EXAMPLE_ENV_NAMES})
    env_vars.update(
        SANDBOX_BACKEND="apptainer_jupyter",
        SANDBOX_CONFIG_PATH=str(EXAMPLE_DIR / "apptainer_env" / "apptainer_config.yaml"),
    )
    if profile == "klx":
        for name in ("XMLIR_ENABLE_H2D_SSE_COPY", "USE_CAST_FC_FUSION"):
            env_vars[name] = os.environ.get(name, "1")
        wandb_key = os.environ.get("WANDB_API_KEY", "")
        if wandb_key and wandb_key != "YOUR-KEY":
            env_vars["WANDB_API_KEY"] = wandb_key
    env_vars.update(prepare_search_environment())
    return {**base, **current, "env_vars": env_vars}


def main() -> int:
    parser = argparse.ArgumentParser(description="生成 DeepEyes-V2 示例的搜索运行环境。")
    parser.add_argument("command", choices=("prepare", "runtime", "agent-command"))
    parser.add_argument("--profile", choices=("standard", "klx"), default="standard")
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            output = prepare_search_environment()[SEARCH_CONFIG_ENV]
        elif args.command == "runtime":
            output = json.dumps(build_runtime_environment(args.profile), ensure_ascii=False, allow_nan=False)
        else:
            output = shlex.join([".", str(EXAMPLE_DIR / "run_agent_app.sh")])
    except (SearchError, ValueError, RecursionError):
        sys.stderr.write("DeepEyes-V2 search runtime configuration is invalid.\n")
        return 1
    sys.stdout.write(output + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
