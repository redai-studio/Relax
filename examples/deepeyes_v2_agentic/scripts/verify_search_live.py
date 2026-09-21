# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import re
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, NoReturn
from urllib.parse import quote, quote_plus

import httpx
from pydantic import ValidationError


EXAMPLE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EXAMPLE_DIR))
sys.path.insert(1, str(EXAMPLE_DIR.parent.parent))

from app import search_http, search_utils  # noqa: E402
from app.search_config import (  # noqa: E402
    SEARCH_CONFIG_ENV,
    ExternalSearchConfig,
    RetrieverSearchConfig,
    SearchError,
    load_search_config,
)


LiveConfig = RetrieverSearchConfig | ExternalSearchConfig


class VerificationError(ValueError):
    pass


class VerificationParser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        self.exit(2, "verify_search_live: invalid_arguments\n")


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = VerificationParser(description="调用已部署的搜索服务并保存脱敏验收证据。", add_help=False)
    parser.add_argument("-h", "--help", action="help", help="显示帮助并退出")
    parser.add_argument("--config", type=Path, required=True, help="显式指定 retriever 或 external YAML 配置")
    parser.add_argument("--service-version", required=True, help="部署版本或可核查版本说明，由执行者提供")
    parser.add_argument("--query", action="append", required=True, help="搜索查询，至少提供两条不同的非空查询")
    parser.add_argument("--output-dir", type=Path, required=True, help="保存证据的新目录，已有目录将拒绝覆盖")
    parser.add_argument(
        "--sensitive-query-param", action="append", default=[], help="需要脱敏的 endpoint 查询参数名，可重复指定"
    )
    parser.add_argument(
        "--sensitive-header", action="append", default=[], help="需要脱敏的自定义 header 名，可重复指定"
    )
    return parser.parse_args(argv)


@contextmanager
def selected_config(path: Path) -> Iterator[LiveConfig]:
    previous = os.environ.get(SEARCH_CONFIG_ENV)
    os.environ[SEARCH_CONFIG_ENV] = str(path)
    try:
        try:
            config = load_search_config()
        except (SearchError, ValidationError):
            raise VerificationError("invalid_config") from None
        if not isinstance(config, (RetrieverSearchConfig, ExternalSearchConfig)):
            raise VerificationError("mock_backend_rejected")
        if isinstance(config, ExternalSearchConfig) and config.auth is not None:
            token = os.environ.get(config.auth.env)
            if token is None or not token.strip():
                raise VerificationError("missing_auth")
        yield config
    finally:
        if previous is None:
            os.environ.pop(SEARCH_CONFIG_ENV, None)
        else:
            os.environ[SEARCH_CONFIG_ENV] = previous


def secret_values(
    config: LiveConfig,
    *,
    sensitive_query_params: list[str] | tuple[str, ...] = (),
    sensitive_headers: list[str] | tuple[str, ...] = (),
) -> set[str]:
    url = httpx.URL(str(config.endpoint))
    values = {url.username, url.password, str(url)}
    for name in sensitive_query_params:
        if name not in url.params:
            raise VerificationError("unknown_sensitive_query_param")
        values.update(url.params.get_list(name))
    headers = (
        {name.lower(): value for name, value in config.headers.items()}
        if isinstance(config, ExternalSearchConfig)
        else {}
    )
    for name in sensitive_headers:
        if name.lower() not in headers:
            raise VerificationError("unknown_sensitive_header")
        values.add(headers[name.lower()])
    if url.path != "/":
        values.add(url.path)
        values.add(str(url.copy_with(userinfo=b"", query=None, fragment=None)))
    if isinstance(config, ExternalSearchConfig):
        if config.auth is not None:
            token = os.environ[config.auth.env]
            values.update((token, config.auth.prefix + token))
    return {variant for value in values if value for variant in (value, quote(value, safe=""), quote_plus(value))}


def redact(value: Any, secrets: set[str]) -> Any:
    pattern = re.compile("|".join(re.escape(item) for item in sorted(secrets, key=len, reverse=True)))

    def clean(item: Any) -> Any:
        if isinstance(item, str):
            return pattern.sub("[REDACTED]", item) if secrets else item
        if isinstance(item, list):
            return [clean(child) for child in item]
        if isinstance(item, dict):
            result: dict[str, Any] = {}
            for key, child in item.items():
                sanitized_key = clean(key)
                if sanitized_key in result:
                    raise VerificationError("redacted_key_collision")
                result[sanitized_key] = clean(child)
            return result
        return item

    return clean(value)


def read_field(value: object, path: list[str], *, optional: bool = False) -> object:
    for key in path:
        if not isinstance(value, dict):
            raise VerificationError("source_validation_failed")
        if key not in value:
            if optional:
                return None
            raise VerificationError("source_validation_failed")
        value = value[key]
    return value


def verify_source(payload: object, normalized: object, config: LiveConfig) -> bool:
    if not isinstance(normalized, dict) or not isinstance(normalized.get("data"), list):
        raise VerificationError("invalid_normalized_response")
    elapsed = normalized.get("elapsed_time")
    if isinstance(elapsed, bool) or not isinstance(elapsed, (int, float)) or not math.isfinite(elapsed) or elapsed < 0:
        raise VerificationError("invalid_normalized_response")
    actual = normalized["data"]
    if not actual:
        raise VerificationError("empty_results")
    expected: list[dict[str, Any]] = []
    if isinstance(config, RetrieverSearchConfig):
        batches = read_field(payload, ["result"])
        if not isinstance(batches, list) or len(batches) != 1 or not isinstance(batches[0], list):
            raise VerificationError("source_validation_failed")
        for row in batches[0]:
            if not isinstance(row, dict):
                raise VerificationError("source_validation_failed")
            document = row["document"] if "document" in row else row
            if not isinstance(document, dict):
                raise VerificationError("source_validation_failed")
            contents = document.get("contents")
            if not isinstance(contents, str) or not contents.strip():
                raise VerificationError("source_validation_failed")
            lines = contents.split("\n", 1)
            for name in ("title", "link", "url"):
                if name in document and not isinstance(document[name], str):
                    raise VerificationError("source_validation_failed")
            expected.append(
                {
                    "title": document.get("title", lines[0]),
                    "link": document.get("link", document.get("url", "")),
                    "snippet": lines[1] if len(lines) == 2 else contents,
                    "date": document.get("date"),
                }
            )
    else:
        rows = read_field(payload, config.response.items_path)
        if not isinstance(rows, list):
            raise VerificationError("source_validation_failed")
        fields = config.response.fields
        for row in rows:
            if not isinstance(row, dict):
                raise VerificationError("source_validation_failed")
            converted = {name: read_field(row, getattr(fields, name)) for name in ("title", "link", "snippet")}
            if any(not isinstance(value, str) for value in converted.values()):
                raise VerificationError("source_validation_failed")
            converted["date"] = None if fields.date is None else read_field(row, fields.date, optional=True)
            expected.append(converted)
    if any(row["date"] is not None and not isinstance(row["date"], str) for row in expected):
        raise VerificationError("source_validation_failed")
    if actual != expected[: config.topk]:
        raise VerificationError("source_mismatch")
    return True


@contextmanager
def observe_requests(attempts: list[dict[str, Any]]) -> Iterator[None]:
    original_factory = search_http._create_client

    def request_hook(request: httpx.Request) -> None:
        attempts.append({"status_code": None, "raw_response": None, "capture_error": None})

    def response_hook(response: httpx.Response) -> None:
        attempt = attempts[-1]
        attempt["status_code"] = response.status_code
        if not response.is_success:
            return
        try:
            response.read()
        except httpx.RequestError:
            attempt["capture_error"] = "response_read_failed"
            raise
        try:
            attempt["raw_response"] = response.json(parse_constant=reject_json_constant)
        except (ValueError, RecursionError):
            attempt["capture_error"] = "invalid_json"

    def create_client(config: LiveConfig) -> httpx.Client:
        client = original_factory(config)
        client.event_hooks["request"].append(request_hook)
        client.event_hooks["response"].append(response_hook)
        return client

    search_http._create_client = create_client
    try:
        yield
    finally:
        search_http._create_client = original_factory


def reject_json_constant(value: str) -> NoReturn:
    raise ValueError("invalid_json_constant")


def verify_query(query: str, config: LiveConfig) -> dict[str, Any]:
    attempts: list[dict[str, Any]] = []
    started = time.monotonic()
    normalized: object = "Error"
    reason = None
    matched = False
    try:
        with observe_requests(attempts):
            normalized = search_utils.search(query)
        if normalized == "Error":
            raise VerificationError("search_failed")
        successful = [
            attempt
            for attempt in attempts
            if attempt["status_code"] is not None and 200 <= attempt["status_code"] < 300
        ]
        if not successful or successful[-1]["capture_error"] is not None:
            raise VerificationError("missing_response_evidence")
        matched = verify_source(successful[-1]["raw_response"], normalized, config)
    except VerificationError as exc:
        reason = str(exc)
    except Exception:
        reason = "search_exception"
    return {
        "query": query,
        "passed": matched,
        "error": reason,
        "request_count": len(attempts),
        "attempts": attempts,
        "elapsed_time": time.monotonic() - started,
        "normalized": normalized,
        "field_source_matches": matched,
    }


def implementation_hashes() -> dict[str, str]:
    names = ("search_config.py", "search_utils.py", "search_http.py", "search_retriever.py", "search_external.py")
    values = {f"app/{name}": hashlib.sha256((EXAMPLE_DIR / "app" / name).read_bytes()).hexdigest() for name in names}
    values["scripts/verify_search_live.py"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return values


def redact_report(result: dict[str, Any], secrets: set[str]) -> dict[str, Any]:
    normalized = result["normalized"]
    if isinstance(normalized, dict):
        if set(normalized) != {"elapsed_time", "data"} or not isinstance(normalized["data"], list):
            raise VerificationError("invalid_normalized_response")
        rows = []
        for row in normalized["data"]:
            if not isinstance(row, dict) or set(row) != {"title", "link", "snippet", "date"}:
                raise VerificationError("invalid_normalized_response")
            rows.append({name: redact(value, secrets) for name, value in row.items()})
        normalized = {"elapsed_time": normalized["elapsed_time"], "data": rows}
    elif normalized != "Error":
        raise VerificationError("invalid_normalized_response")
    return {
        **result,
        "query": redact(result["query"], secrets),
        "attempts": [
            {**attempt, "raw_response": redact(attempt["raw_response"], secrets)} for attempt in result["attempts"]
        ],
        "normalized": normalized,
    }


def write_json(path: Path, value: object) -> None:
    data = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(data)


def read_artifact(path: Path, expected: object) -> None:
    with path.open(encoding="utf-8") as handle:
        actual = json.load(handle, parse_constant=reject_json_constant)
    if json.dumps(actual, sort_keys=True, allow_nan=False) != json.dumps(expected, sort_keys=True, allow_nan=False):
        raise VerificationError("artifact_validation_failed")


def run_verification(args: argparse.Namespace) -> bool:
    queries = [query.strip() for query in args.query]
    if len(queries) < 2 or any(not query for query in queries) or len(set(queries)) != len(queries):
        raise VerificationError("invalid_queries")
    if not args.service_version.strip():
        raise VerificationError("invalid_service_version")
    with selected_config(args.config) as config:
        secrets = secret_values(
            config, sensitive_query_params=args.sensitive_query_param, sensitive_headers=args.sensitive_header
        )
        config_sha256 = hashlib.sha256(args.config.read_bytes()).hexdigest()
        try:
            args.output_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
        except FileExistsError:
            raise VerificationError("output_exists") from None
        except OSError:
            raise VerificationError("output_unavailable") from None
        url = httpx.URL(str(config.endpoint))
        results: list[dict[str, Any]] = []
        artifacts: dict[str, dict[str, Any]] = {}
        previous_logging = logging.root.manager.disable
        logging.disable(logging.CRITICAL)
        try:
            for index, query in enumerate(queries, 1):
                result = redact_report(verify_query(query, config), secrets)
                filename = f"query-{index:03d}.json"
                write_json(args.output_dir / filename, result)
                artifacts[filename] = result
                results.append(
                    {
                        "query": result["query"],
                        "passed": result["passed"],
                        "error": result["error"],
                        "evidence": filename,
                    }
                )
        finally:
            logging.disable(previous_logging)
        passed = all(result["passed"] for result in results)
        summary = {
            "passed": passed,
            "backend": config.backend,
            "service_version": redact(args.service_version.strip(), secrets),
            "service_version_source": "operator_supplied",
            "checked_at_utc": datetime.now(timezone.utc).isoformat(),
            "endpoint_origin": {"scheme": url.scheme, "host": redact(url.host, secrets), "port": url.port},
            "config_sha256": config_sha256,
            "search_options": {
                "topk": config.topk,
                "timeout_s": config.timeout_s,
                "max_retries": config.max_retries,
                "retry_delay_s": config.retry_delay_s,
                "retry_max_delay_s": config.retry_max_delay_s,
                "trust_env": config.trust_env,
            },
            "implementation_sha256": implementation_hashes(),
            "queries": results,
        }
        summary_path = args.output_dir / "summary.json"
        try:
            write_json(summary_path, summary)
            for filename, expected in artifacts.items():
                read_artifact(args.output_dir / filename, expected)
            read_artifact(summary_path, summary)
        except BaseException:
            summary_path.unlink(missing_ok=True)
            raise
        return passed


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        passed = run_verification(args)
    except VerificationError as exc:
        sys.stderr.write(f"verify_search_live: {exc}\n")
        return 1
    except Exception:
        sys.stderr.write("verify_search_live: verification_failed\n")
        return 1
    sys.stdout.write(json.dumps({"passed": passed, "query_count": len(args.query)}) + "\n")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
