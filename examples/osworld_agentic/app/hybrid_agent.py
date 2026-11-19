# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Relax-side subclass of the external HybridAgentLocal.

The upstream `HybridAgentLocal.call_llm` POSTs to a local vLLM endpoint via
raw httpx. When integrating with Relax we want it to talk to the
per-session AgenticChatAPIService — the URL is passed in via `OPENAI_BASE_URL`
(already `/v1`-suffixed and trailing-slash-stripped by `run_agent_app.sh`)
and the API key is the session id.

Everything else — prompt building, tool retriever, image history, parse
retries — is inherited from the external OSWorld installation.
"""

from __future__ import annotations

import copy
import logging
import os
import time
from typing import Any

import httpx
from mm_agents.hybrid_agent_local import HybridAgentLocal

from .visual_context import VisualContext


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
        context_mode: str = "baseline",
        vcs_keep_images: int = 2,
        vcs_prune_images: int = 2,
        **kwargs: Any,
    ) -> None:
        if context_mode not in ("baseline", "vcs"):
            raise ValueError(f"Unknown OSWorld context_mode: {context_mode}")
        self.visual_context = VisualContext.from_env(
            enabled=context_mode == "vcs",
            keep_images=vcs_keep_images,
            prune_images=vcs_prune_images,
        )
        if self.visual_context is not None and kwargs.get("context_policy", "baseline") != "baseline":
            raise ValueError("VCS requires context_policy=baseline (no screenshot skipping)")
        super().__init__(**kwargs)
        self._llm_call_timeout_s = float(llm_call_timeout_s)
        self._max_llm_call_retries = int(max_llm_call_retries)
        self._backoff_seconds = float(backoff_seconds)
        self._stop_token_ids = list(stop_token_ids) if stop_token_ids else []
        # One entry per committed turn; app/agent.py turns these into the
        # explicit export records Relax needs (see README.md).
        self.turn_records: list[dict[str, Any]] = []
        self.model_request_s = 0.0
        self.transport_retries = 0

        base_url = os.environ.get("OPENAI_BASE_URL", "").rstrip("/")
        if not base_url:
            raise RuntimeError("OPENAI_BASE_URL is not set. run_agent_app.sh should export it from RELAX_BASE_URL.")
        self._base_url = base_url
        self._api_key = os.environ.get("OPENAI_API_KEY", "")

    def predict(self, instruction: str, obs: dict[str, Any], **kwargs: Any) -> Any:
        if os.environ.get("RELAX_DECISION_HIDDEN_DIR"):
            self._hidden_turn_id = int(kwargs.get("step_idx", len(self.actions)))
        if self.visual_context is not None:
            self.visual_context.begin_turn(int(kwargs.get("step_idx", len(self.actions))))
        return super().predict(instruction, obs, **kwargs)

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

        visual_context = getattr(self, "visual_context", None)
        if visual_context is not None:
            payload = visual_context.prepare(payload)
        if os.environ.get("RELAX_DECISION_HIDDEN_DIR"):
            payload = {
                **payload,
                "relax_hidden_context": {"task_id": self._hidden_task_id, "turn_id": self._hidden_turn_id},
            }
        archive = getattr(self, "archive", None)
        last_error = None
        for attempt in range(self._max_llm_call_retries):
            call_id = archive.model_start(payload, model, attempt) if archive else None
            request_start = time.monotonic()
            try:
                resp = httpx.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=self._llm_call_timeout_s,
                )
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                if archive:
                    archive.model_result(call_id, None, None, f"{type(exc).__name__}: {exc}")
                self.transport_retries += 1
                last_error = f"transport: {exc}"
                logger.warning(
                    "call_llm attempt %d/%d transport error: %s",
                    attempt + 1,
                    self._max_llm_call_retries,
                    exc,
                )
                time.sleep(self._backoff_seconds)
                continue
            finally:
                self.model_request_s += time.monotonic() - request_start

            if archive:
                archive.model_result(call_id, resp.text, resp.status_code)

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
                        "messages": copy.deepcopy([*payload["messages"], {"role": "assistant", "content": content}]),
                        "tools": payload.get("tools") or [],
                        "chat_template_kwargs": payload.get("chat_template_kwargs") or {},
                    }
                )
                if visual_context is not None:
                    # Preserve the service's canonical assistant fields, including
                    # reasoning/tool calls if a parser is explicitly configured.
                    message = choice["message"]
                    self.turn_records[-1]["messages"][-1] = {
                        "role": "assistant",
                        "content": content,
                        **{
                            key: copy.deepcopy(message[key])
                            for key in ("reasoning_content", "tool_calls")
                            if key in message
                        },
                    }
                    visual_context.commit(self.turn_records[-1], usage)
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
