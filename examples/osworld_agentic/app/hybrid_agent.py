# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Relax-side subclass of the vendored HybridAgentLocal.

The upstream `HybridAgentLocal.call_llm` POSTs to a local vLLM endpoint via
raw httpx. When integrating with Relax we want it to talk to the
per-session AgenticChatAPIService — the URL is passed in via `OPENAI_BASE_URL`
(already `/v1`-suffixed and trailing-slash-stripped by `run_agent_app.sh`)
and the API key is the session id.

Everything else — prompt building, tool retriever, image history, parse
retries — is inherited unchanged. Keeping the diff surface tiny is the point
of the vendor snapshot.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import httpx


_VENDOR_ROOT = Path(__file__).resolve().parent.parent / "vendor"
for _p in (_VENDOR_ROOT, _VENDOR_ROOT / "agents"):
    p_str = str(_p)
    if p_str not in sys.path:
        sys.path.insert(0, p_str)

from mm_agents.hybrid_agent_local import HybridAgentLocal  # noqa: E402


logger = logging.getLogger("osworld_agentic.hybrid_agent")


class SessionDiscardedError(RuntimeError):
    """Relax pipeline dropped this session mid-flight (sync-mode tail
    discard)."""


class ContextLengthExceededError(RuntimeError):
    """Prompt exceeded the served model's context window."""


class RelaxHybridAgent(HybridAgentLocal):
    """HybridAgentLocal wired to Relax's AgenticChatAPIService.

    Reads:
      * ``OPENAI_BASE_URL`` — full base including ``/v1`` (see run_agent_app.sh).
      * ``OPENAI_API_KEY``  — RELAX_SESSION_ID (used to key session state
        inside AgenticChatAPIService).

    Bounded retry with exponential backoff on:
      * HTTP 5xx / connection errors.
      * ``finish_reason != "stop"`` and no partial content.
    """

    def __init__(
        self,
        *,
        llm_call_timeout_s: float = 600.0,
        max_llm_call_retries: int = 3,
        backoff_seconds: float = 5.0,
        stop_token_ids: list[int] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._llm_call_timeout_s = float(llm_call_timeout_s)
        self._max_llm_call_retries = int(max_llm_call_retries)
        self._backoff_seconds = float(backoff_seconds)
        self._stop_token_ids = list(stop_token_ids) if stop_token_ids else []
        # One entry per committed turn; app/agent.py turns these into the
        # explicit export records Relax needs (see PITFALLS.md section 11).
        self.turn_records: list[dict[str, Any]] = []

        base_url = os.environ.get("OPENAI_BASE_URL", "").rstrip("/")
        if not base_url:
            raise RuntimeError("OPENAI_BASE_URL is not set. run_agent_app.sh should export it from RELAX_BASE_URL.")
        self._base_url = base_url
        self._api_key = os.environ.get("OPENAI_API_KEY", "")

    def call_llm(self, payload: dict[str, Any], model: str) -> str:  # noqa: D401
        """Send `payload` to Relax's chat completions endpoint and return
        content."""
        url = f"{self._base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }
        # Force stop_token_ids from config every call — the upstream payload
        # already sets `stop=["</tool_call>"]`, but SGLang honors stop_token_ids
        # too, and the tool-call parser memory (effb0f58) tells us we cannot rely
        # on structured `tool_calls` output; explicit stop ids reduce noise past
        # the closing tag.
        if self._stop_token_ids and "stop_token_ids" not in payload:
            payload = dict(payload)
            payload["stop_token_ids"] = list(self._stop_token_ids)

        last_error = None
        for attempt in range(self._max_llm_call_retries):
            try:
                resp = httpx.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=self._llm_call_timeout_s,
                )
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_error = f"transport: {exc}"
                logger.warning(
                    "call_llm attempt %d/%d transport error: %s",
                    attempt + 1,
                    self._max_llm_call_retries,
                    exc,
                )
                time.sleep(self._backoff_seconds)
                continue

            if resp.status_code == 400:
                # Distinguish context-length-exceeded (fatal, exit cleanly) from
                # generic 400 (transient / retry). Relax reports the former via
                # `{"error": {"code": "context_length_exceeded"}}`; fall back on
                # substring match for older service versions.
                if self._is_error_code(resp, "context_length_exceeded") or "context length" in resp.text.lower():
                    raise ContextLengthExceededError(resp.text[:400])

            if resp.status_code == 404 and self._is_error_code(resp, "session_discarded"):
                # Sync-mode tail discard by the Relax pipeline. Not an error;
                # surface so the driver can exit the loop cleanly.
                raise SessionDiscardedError("session discarded by pipeline")

            if resp.status_code != 200:
                last_error = f"HTTP {resp.status_code}: {resp.text[:400]}"
                logger.warning(
                    "call_llm attempt %d/%d bad status: %s",
                    attempt + 1,
                    self._max_llm_call_retries,
                    last_error,
                )
                time.sleep(self._backoff_seconds)
                continue

            data = resp.json()
            choice = (data.get("choices") or [{}])[0]
            content = (choice.get("message") or {}).get("content") or ""
            finish_reason = choice.get("finish_reason")
            usage = data.get("usage") or {}

            if content:
                self._input_toks_per_step.append(int(usage.get("prompt_tokens", 0)))
                self._output_toks_per_step.append(int(usage.get("completion_tokens", 0)))
                if finish_reason not in ("stop", "length"):
                    logger.info(
                        "call_llm finish_reason=%s (accepting content)",
                        finish_reason,
                    )
                self.turn_records.append(
                    {
                        "messages": [
                            *payload["messages"],
                            {"role": "assistant", "content": content},
                        ],
                        "tools": payload.get("tools") or [],
                        "chat_template_kwargs": payload.get("chat_template_kwargs") or {},
                    }
                )
                return content

            last_error = f"empty content (finish_reason={finish_reason})"
            logger.warning(
                "call_llm attempt %d/%d %s",
                attempt + 1,
                self._max_llm_call_retries,
                last_error,
            )
            time.sleep(self._backoff_seconds)

        raise RuntimeError(f"call_llm failed after {self._max_llm_call_retries} attempts to {url}: {last_error}")

    @staticmethod
    def _is_error_code(resp: httpx.Response, code: str) -> bool:
        try:
            body = resp.json()
        except Exception:
            return False
        err = body.get("error") if isinstance(body, dict) else None
        return isinstance(err, dict) and err.get("code") == code
