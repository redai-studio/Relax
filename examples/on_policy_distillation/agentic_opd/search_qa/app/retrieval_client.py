# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Thin HTTP client for the external, stateless retrieval service.

Wire protocol (matches the vendored ``retriever/retrieval_server.py``):
    POST {url}  {"queries": [<str>], "topk": <int>, "return_scores": true}
    -> {"result": [[ {"document": {"contents": <str>, ...}, "score": <float>}, ... ]]}
The endpoint is batch-oriented; we send one query per call and read ``result[0]``,
formatting docs as ``Doc {i}: {contents}`` for the environment's ``<information>``
block. A single pooled ``requests.Session`` is reused across calls.
"""

from __future__ import annotations

import logging
import os
import time

import requests


logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = float(os.getenv("SEARCH_RETRIEVAL_TIMEOUT_SECONDS", "30"))
MAX_RETRIES = int(os.getenv("SEARCH_RETRIEVAL_MAX_RETRIES", "50"))
INITIAL_RETRY_DELAY = float(os.getenv("SEARCH_RETRIEVAL_INITIAL_RETRY_DELAY_SECONDS", "1"))
DEFAULT_RETRY_BUDGET = float(os.getenv("SEARCH_RETRIEVAL_RETRY_BUDGET_SECONDS", "120"))
DEFAULT_TRUST_ENV = os.getenv("SEARCH_RETRIEVAL_TRUST_ENV", "0").lower() in {"1", "true", "yes"}


def _passages2string(retrieval_result: list[dict]) -> str:
    format_reference = ""
    for idx, doc_item in enumerate(retrieval_result):
        content = doc_item["document"]["contents"].strip()
        format_reference += f"Doc {idx + 1}: {content}\n"
    return format_reference


class RetrievalClient:
    """Single-query client against a stateless retrieval endpoint."""

    def __init__(
        self,
        url: str,
        *,
        topk: int = 3,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = MAX_RETRIES,
        retry_budget: float = DEFAULT_RETRY_BUDGET,
        trust_env: bool = DEFAULT_TRUST_ENV,
    ) -> None:
        self.url = url
        self.topk = topk
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_budget = retry_budget

        session = requests.Session()
        session.trust_env = trust_env
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=512,
            pool_maxsize=512,
            max_retries=0,  # retries are handled explicitly below
            pool_block=False,
        )
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        self.session = session

    def _post(self, payload: dict) -> dict:
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        last_error: str | None = None
        deadline = time.monotonic() + self.retry_budget
        attempts = 0
        while attempts < max(1, self.max_retries):
            attempts += 1
            try:
                resp = self.session.post(self.url, headers=headers, json=payload, timeout=self.timeout)
                if resp.status_code in (500, 502, 503, 504):
                    last_error = f"server error {resp.status_code}"
                else:
                    resp.raise_for_status()
                    try:
                        data = resp.json()
                    except ValueError as exc:
                        preview = resp.text[:200]
                        last_error = (
                            f"invalid JSON response status={resp.status_code} "
                            f"bytes={len(resp.content)} body={preview!r}: {exc}"
                        )
                    else:
                        if not isinstance(data, dict):
                            last_error = (
                                f"invalid JSON response type={type(data).__name__} "
                                f"status={resp.status_code} bytes={len(resp.content)}"
                            )
                        else:
                            return data
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as exc:
                last_error = str(exc)
            except requests.exceptions.RequestException as exc:
                last_error = str(exc)
                break

            remaining = deadline - time.monotonic()
            if attempts >= max(1, self.max_retries) or remaining <= 0:
                break
            time.sleep(min(INITIAL_RETRY_DELAY * attempts, 5.0, remaining))
        raise RuntimeError(f"retrieval request failed after {attempts} attempts: {last_error}")

    def search(self, query: str | None) -> str:
        """Return formatted documents for ``query`` (``""`` when query is
        empty).

        Raises ``RuntimeError`` if the service is unreachable after retries;
        the caller (environment) decides how to handle that.
        """
        if not query:
            return ""
        query = query.strip()
        api_response = self._post({"queries": [query], "topk": self.topk, "return_scores": True})
        raw_results = api_response.get("result", [])
        if not raw_results:
            return ""
        # Batch endpoint: one docs-list per query. We sent a single query.
        return _passages2string(raw_results[0])
