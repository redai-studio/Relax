# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Text search adapters configured by DEEPEYES_V2_SEARCH_CONFIG."""

from __future__ import annotations

import math
import os
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit


DEFAULTS = {
    "backend": "mock",
    "top_k": 5,
    "timeout_s": 10.0,
    "max_retries": 2,
    "backoff_s": 0.5,
    "trust_env": False,
}


class SearchError(ValueError):
    """Search failure whose message is safe to log."""


def _mapping(value: Any, name: str, allowed: set[str]) -> dict:
    if not isinstance(value, dict):
        raise SearchError(f"{name} must be a mapping")
    if value.keys() - allowed:
        raise SearchError(f"{name} contains unknown fields")
    return value


def _text(value: Any, name: str, *, empty: bool = False) -> str:
    if not isinstance(value, str) or (not empty and not value.strip()):
        raise SearchError(f"{name} must be a {'non-empty ' if not empty else ''}string")
    return value


def _integer(value: Any, name: str, minimum: int) -> int:
    if type(value) is not int or value < minimum:
        raise SearchError(f"{name} must be an integer >= {minimum}")
    return value


def _load_config() -> dict:
    config = dict(DEFAULTS)
    path = os.environ.get("DEEPEYES_V2_SEARCH_CONFIG", "").strip()
    if path:
        import yaml

        try:
            source = yaml.safe_load(Path(path).expanduser().read_text(encoding="utf-8"))
        except (OSError, UnicodeError, yaml.YAMLError) as exc:
            raise SearchError(f"cannot read DEEPEYES_V2_SEARCH_CONFIG ({type(exc).__name__})") from None
        config.update(_mapping(source, "search config", set(DEFAULTS) | {"retriever", "external"}))
    if config["backend"] not in ("mock", "retriever", "external"):
        raise SearchError("backend must be mock, retriever or external")
    _integer(config["top_k"], "top_k", 1)
    _integer(config["max_retries"], "max_retries", 0)
    for name in ("timeout_s", "backoff_s"):
        value = config[name]
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise SearchError(f"{name} must be a finite non-negative number")
    if config["timeout_s"] == 0:
        raise SearchError("timeout_s must be positive")
    if not isinstance(config["trust_env"], bool):
        raise SearchError("trust_env must be boolean")
    return config


def _endpoint(value: Any, name: str) -> str:
    value = _text(value, name).strip()
    try:
        parsed = urlsplit(value)
        valid = parsed.scheme in ("http", "https") and parsed.hostname and not parsed.username and not parsed.password
        parsed.port
    except ValueError:
        valid = False
    if not valid:
        raise SearchError(f"{name} must be an HTTP(S) URL without credentials")
    return value


def _request(config: dict, url: str, method: str, payload: dict, headers: dict) -> Any:
    import requests

    delay = min(config["backoff_s"], 2.0)
    attempts = config["max_retries"] + 1
    with requests.Session() as session:
        session.trust_env = config["trust_env"]
        for attempt in range(attempts):
            try:
                with session.request(
                    method,
                    url,
                    headers=headers,
                    **{"params" if method == "GET" else "json": payload},
                    timeout=config["timeout_s"],
                    allow_redirects=False,
                ) as response:
                    status = response.status_code
                    if 200 <= status < 300:
                        try:
                            return response.json()
                        except ValueError:
                            raise SearchError("search response is not valid JSON") from None
                    if status not in (408, 429) and not 500 <= status < 600:
                        raise SearchError(f"search HTTP status {status}")
                    reason = f"HTTP {status}"
            except (requests.Timeout, requests.ConnectionError) as exc:
                reason = type(exc).__name__
            except requests.RequestException as exc:
                raise SearchError(f"search request failed ({type(exc).__name__})") from None
            if attempt + 1 < attempts:
                time.sleep(delay)
                delay = min(delay * 2, 2.0)
    raise SearchError(f"search failed after {attempts} attempts ({reason})")


def _date(value: Any) -> str | None:
    return None if value is None else _text(value, "result.date", empty=True)


def _retriever_rows(body: Any, size: int) -> list[dict]:
    batches = body.get("result") if isinstance(body, dict) else None
    if not isinstance(batches, list) or len(batches) != 1 or not isinstance(batches[0], list):
        raise SearchError("retriever response must contain one result batch")
    rows = []
    for item in batches[0][:size]:
        doc = item.get("document", item) if isinstance(item, dict) else None
        if not isinstance(doc, dict):
            raise SearchError("retriever document must be a mapping")
        content = _text(doc.get("contents"), "retriever document.contents")
        title = doc.get("title")
        if title is not None:
            title = _text(title, "result.title", empty=True)
        title = title or content.strip().splitlines()[0].strip().strip('"')
        link = doc.get("url")
        if link is None:
            link = doc.get("link", "")
        rows.append(
            {
                "title": title,
                "link": _text("" if link is None else link, "result.link", empty=True),
                "snippet": content,
                "date": _date(doc.get("date")),
            }
        )
    return rows


def _field(row: Any, path: str) -> Any:
    for key in path.split("."):
        if not isinstance(row, dict):
            return None
        row = row.get(key)
    return row


def _external(config: dict, query: str, size: int) -> list[dict]:
    external = _mapping(
        config.get("external"), "external", {"endpoint", "method", "auth", "request_map", "response_map"}
    )
    url = _endpoint(external.get("endpoint"), "external.endpoint")
    method = _text(external.get("method", "POST"), "external.method").upper()
    if method not in ("GET", "POST"):
        raise SearchError("external.method must be GET or POST")
    request_map = _mapping(external.get("request_map"), "external.request_map", {"query", "size"})
    response_map = _mapping(
        external.get("response_map"), "external.response_map", {"results", "title", "link", "snippet", "date"}
    )
    for name in ("query", "size"):
        _text(request_map.get(name), f"external.request_map.{name}")
    if request_map["query"] == request_map["size"]:
        raise SearchError("external.request_map fields must be distinct")
    for name in ("results", "title", "link", "snippet"):
        _text(response_map.get(name), f"external.response_map.{name}")
    if "date" in response_map:
        _text(response_map["date"], "external.response_map.date")
    headers = {"Accept": "application/json"}
    if "auth" in external:
        auth = _mapping(external["auth"], "external.auth", {"header", "prefix"})
        header = _text(auth.get("header"), "external.auth.header")
        prefix = _text(auth.get("prefix", ""), "external.auth.prefix", empty=True)
        secret = os.environ.get("DEEPEYES_V2_SEARCH_API_KEY", "")
        if not secret.strip() or any(c in header + prefix + secret for c in "\r\n"):
            raise SearchError("external authentication is missing or invalid")
        headers[header] = prefix + secret
    body = _request(config, url, method, {request_map["query"]: query, request_map["size"]: size}, headers)
    items = _field(body, response_map["results"])
    if not isinstance(items, list):
        raise SearchError("external results must be a list")
    rows = []
    for item in items[:size]:
        row = {
            name: _text(_field(item, response_map[name]), f"result.{name}", empty=True)
            for name in ("title", "link", "snippet")
        }
        row["date"] = _date(_field(item, response_map["date"])) if "date" in response_map else None
        rows.append(row)
    return rows


def run_search(query: str, size: int | None = None) -> dict:
    """Return normalized search results, or raise SearchError on failure."""
    query = _text(query, "query").strip()
    config = _load_config()
    size = config["top_k"] if size is None else _integer(size, "size", 1)
    if config["backend"] == "mock":
        return {
            "elapsed_time": 0.0,
            "data": [
                {
                    "title": f"Offline mock result {i + 1}",
                    "link": f"https://example.invalid/{i + 1}",
                    "snippet": f"Deterministic offline result for: {query}",
                    "date": None,
                }
                for i in range(size)
            ],
        }
    started = time.monotonic()
    if config["backend"] == "retriever":
        retriever = _mapping(config.get("retriever"), "retriever", {"url"})
        body = _request(
            config,
            _endpoint(retriever.get("url"), "retriever.url"),
            "POST",
            {"queries": [query], "topk": size, "return_scores": True},
            {"Accept": "application/json"},
        )
        rows = _retriever_rows(body, size)
    else:
        rows = _external(config, query, size)
    return {"elapsed_time": time.monotonic() - started, "data": rows}
