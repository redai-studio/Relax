# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Side-effect-free configuration preflight for Relax training commands."""

import argparse
import json
import pprint
import shlex
import sys
import tempfile
from collections.abc import Mapping
from contextlib import redirect_stderr, redirect_stdout


_SENSITIVE_VALUE_OPTIONS = {"--agent-env", "--train-env-vars"}
_SECRET_NAMES = {
    "api_key",
    "access_key",
    "auth_token",
    "bearer_token",
    "credential",
    "credentials",
    "key",
    "password",
    "private_key",
    "secret",
    "secret_key",
    "token",
    "wandb_key",
}
_SECRET_SUFFIXES = (
    "_api_key",
    "_access_key",
    "_secret_key",
    "_auth_token",
    "_token",
    "_password",
    "_secret",
    "_credential",
    "_credentials",
)


def _is_secret(name: object) -> bool:
    normalized = str(name).lower().lstrip("-").replace("-", "_")
    return normalized in _SECRET_NAMES or normalized.endswith(_SECRET_SUFFIXES)


def _is_sensitive_option(option: str) -> bool:
    return option in _SENSITIVE_VALUE_OPTIONS or _is_secret(option)


def _redact(value, key=None):
    if _is_secret(key):
        return "<redacted>"
    if isinstance(value, Mapping):
        return {str(k): _redact(v, k) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_redact(item) for item in value]
    if isinstance(value, str) and "=" in value:
        env_key, env_value = value.split("=", 1)
        if _is_secret(env_key):
            return f"{env_key}=<redacted>"
        return f"{env_key}={env_value}"
    return value


def _redact_argv(argv: list[str]) -> list[str]:
    result = []
    redact_next = False
    for token in argv:
        if redact_next:
            result.append("<redacted>")
            redact_next = False
            continue
        if token.startswith("-"):
            option, separator, _attached = token.partition("=")
            if _is_sensitive_option(option):
                if separator:
                    result.append(f"{option}=<redacted>")
                else:
                    result.append(option)
                    redact_next = True
                continue
        try:
            parsed = json.loads(token)
        except (json.JSONDecodeError, TypeError):
            parsed = token
        sanitized = _redact(parsed)
        result.append(json.dumps(sanitized, sort_keys=True) if parsed is not token else sanitized)
    return result


def _secret_values(value, key=None) -> list[str]:
    if _is_secret(key):
        return [str(value)]
    if isinstance(value, Mapping):
        return [secret for k, item in value.items() for secret in _secret_values(item, k)]
    if isinstance(value, (list, tuple)):
        return [secret for item in value for secret in _secret_values(item)]
    if isinstance(value, str) and "=" in value:
        env_key, env_value = value.split("=", 1)
        return [env_value] if _is_secret(env_key) else []
    return []


def _redact_error(message: str, argv: list[str]) -> str:
    secrets = []
    redact_next = False
    for token in argv:
        if redact_next:
            secrets.append(token)
            try:
                parsed = json.loads(token)
            except (json.JSONDecodeError, TypeError):
                parsed = token
            secrets.extend(_secret_values(parsed))
            redact_next = False
            continue
        if token.startswith("-"):
            option, separator, attached = token.partition("=")
            if _is_sensitive_option(option):
                if separator:
                    secrets.append(attached)
                else:
                    redact_next = True
                continue
        try:
            value = json.loads(token)
        except (json.JSONDecodeError, TypeError):
            value = token
        secrets.extend(_secret_values(value))
    for secret in sorted(set(secrets), key=len, reverse=True):
        if secret:
            message = message.replace(secret, "<redacted>")
    return message


def _error_report(error: BaseException, argv: list[str]) -> dict:
    message = _redact_error(f"{type(error).__name__}: {error}", argv)
    error_text, marker, suggestion = message.partition(" Fix: ")
    if isinstance(error, ModuleNotFoundError):
        missing = getattr(error, "name", None) or "the missing module"
        suggestion = f"install {missing!r} and the project requirements in the active environment"
    elif "Upgrade with:" in message:
        error_text, suggestion = message.split("Upgrade with:", 1)
        suggestion = f"upgrade with:{suggestion}"
    elif not marker:
        suggestion = "correct the invalid option described above and rerun the check"
    return {"status": "error", "error": error_text.rstrip(), "suggestion": suggestion.strip().rstrip(".")}


def _active_roles(args) -> list[str]:
    from relax.utils.arguments import resolve_configured_roles

    return resolve_configured_roles(args)


def _resource_summary(args, roles: list[str]) -> dict:
    from relax.utils.opd.opd_utils import is_managed_opd_teacher_colocate

    resource = getattr(args, "resource", {})
    active = {role: resource[role] for role in roles if role in resource}
    shared_roles = {"actor", "rollout", "genrm"}
    if getattr(args, "advantage_estimator", None) == "ppo" and resource.get("critic") == resource.get("actor"):
        shared_roles.add("critic")
    if is_managed_opd_teacher_colocate(args):
        shared_roles.add("teacher")
    shares_gpus = bool(
        getattr(args, "colocate", False)
        and not getattr(args, "hybrid", False)
        and {"actor", "rollout"}.issubset(active)
    )
    if shares_gpus:
        shared_gpu = max((spec[1] for role, spec in active.items() if role in shared_roles), default=0)
        independent_gpu = sum(spec[1] for role, spec in active.items() if role not in shared_roles)
        total_gpu = shared_gpu + independent_gpu
    else:
        total_gpu = sum(spec[1] for spec in active.values())
    return {"roles": active, "shares_actor_rollout_gpus": shares_gpus, "total_gpus": total_gpu}


def _report(args, training_argv: list[str]) -> dict:
    config = {key: value for key, value in vars(args).items() if not key.startswith("_")}
    roles = _active_roles(args)
    safe_argv = _redact_argv(training_argv)
    return {
        "status": "ok",
        "config": _redact(config),
        "roles": roles,
        "resources": _resource_summary(args, roles),
        "expected_command": shlex.join(["python", "-m", "relax.entrypoints.train", *safe_argv]),
    }


def _print_report(report: dict, output_format: str) -> None:
    if output_format == "json":
        print(json.dumps(report, indent=2, sort_keys=True, default=str))
        return
    print("Relax configuration check: OK")
    print("\nMerged config:")
    print(pprint.pformat(report["config"], sort_dicts=True))
    print(f"\nRole topology: {', '.join(report['roles'])}")
    print("\nResource requirement:")
    print(pprint.pformat(report["resources"], sort_dicts=True))
    print(f"\nExpected launch command:\n{report['expected_command']}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", choices=("text", "json"), default="text")
    options, training_argv = parser.parse_known_args(argv)
    if training_argv and training_argv[0] == "--":
        training_argv = training_argv[1:]
    if not training_argv:
        parser.error("pass the complete training arguments after '--'")

    original_argv = sys.argv
    try:
        sys.argv = ["relax.entrypoints.train", *training_argv]
        # Some backend parsers query fileno(), so use real temporary files
        # rather than StringIO while suppressing their verbose argument dumps.
        with tempfile.TemporaryFile(mode="w+") as stdout, tempfile.TemporaryFile(mode="w+") as stderr:
            with redirect_stdout(stdout), redirect_stderr(stderr):
                from relax.utils.arguments import parse_args

                args = parse_args(strict=True)
        _print_report(_report(args, training_argv), options.format)
        return 0
    except (Exception, SystemExit) as error:
        payload = _error_report(error, training_argv)
        if options.format == "json":
            print(json.dumps(payload, indent=2, sort_keys=True), file=sys.stderr)
        else:
            print(
                f"Relax configuration check: FAILED\n{payload['error']}\nFix: {payload['suggestion']}",
                file=sys.stderr,
            )
        return 1
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    raise SystemExit(main())
