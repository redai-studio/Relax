# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Shared request-time visual segmentation; SessionForest owns training tokens.

Adapters call prepare before generation, commit with the exact reply, and
export records after finish. Full-history mode accepts user and tool
observations alike. The latest-observation mode preserves OSWorld's rebuilt-
prompt contract.
"""

from __future__ import annotations

import copy
from typing import Any

from relax.utils.env import Envs, is_env_set


def image_count(messages: list[dict[str, Any]]) -> int:
    return sum(
        block.get("type") == "image_url"
        for message in messages
        if isinstance(message.get("content"), list)
        for block in message["content"]
    )


class VisualContext:
    """Export only committed leaves, never reconstructed or concatenated
    responses.

    A segment marker before the first observation prevents Forest's longest-
    prefix lookup from reusing an older response ancestor. Adapters must use
    one instance per sequential conversation, archive calls, and export
    records.
    """

    def __init__(
        self,
        keep_images: int = 2,
        prune_images: int = 2,
        *,
        history_mode: str = "full_history",
        metric_prefix: str = "agentic",
    ) -> None:
        if keep_images < 1 or prune_images < 1:
            raise ValueError("VCS keep_images and prune_images must be positive")
        self.keep_images = keep_images
        self.prune_images = prune_images
        if history_mode not in ("full_history", "latest_observation"):
            raise ValueError(f"Unknown VCS history mode: {history_mode}")
        self.history_mode = history_mode
        self.metric_prefix = metric_prefix
        self._source_request: list[dict[str, Any]] | None = None
        self._source_committed: list[dict[str, Any]] = []
        self.records: list[dict[str, Any]] = []
        self.calls: list[dict[str, Any]] = []
        self._history: list[dict[str, Any]] = []
        self._request: dict[str, Any] | None = None
        self._upstream_messages: list[dict[str, Any]] | None = None
        self._systems: list[dict[str, Any]] = []
        self._tools: list[dict[str, Any]] = []
        self._template: dict[str, Any] = {}
        self._start_call = 0
        self._segment_id = 0
        self._turn = 0
        self._attempt = 0

    @classmethod
    def from_env(cls, *, enabled: bool = False, **kwargs: Any) -> VisualContext | None:
        """Explicit RELAX_VCS overrides an adapter's legacy configuration."""
        if not (Envs.RELAX_VCS if is_env_set("RELAX_VCS") else enabled):
            return None
        if is_env_set("RELAX_VCS_KEEP_IMAGES"):
            kwargs["keep_images"] = Envs.RELAX_VCS_KEEP_IMAGES
        if is_env_set("RELAX_VCS_PRUNE_IMAGES"):
            kwargs["prune_images"] = Envs.RELAX_VCS_PRUNE_IMAGES
        return cls(**kwargs)

    def begin_turn(self, turn: int) -> None:
        self._turn = turn
        self._attempt = 0
        self._request = None
        self._upstream_messages = None

    def _new_context(self, body: list[dict[str, Any]]) -> list[dict[str, Any]]:
        marker = {"role": "user", "content": f"[Visual context segment {self._segment_id}]"}
        self._segment_id += 1
        return [*copy.deepcopy(self._systems), marker, *body]

    def prepare(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Called before the actual model request, once per upstream parse
        attempt."""
        if self.history_mode == "full_history":
            return self._prepare_full_history(payload)
        request = copy.deepcopy(payload)
        messages = request["messages"]
        systems = [m for m in messages if m["role"] == "system"]
        # Deliberately support the pinned HybridAgentLocal prompt shape only.
        if not messages or messages[-1]["role"] != "user" or not systems:
            raise ValueError("VCS requires HybridAgentLocal system + history + current user observation")
        if messages[: len(systems)] != systems or image_count([messages[-1]]) != 1:
            raise ValueError("VCS requires leading system rules and exactly one current screenshot")
        if self._request is not None and [m for m in messages if m["role"] != "system"] != [
            m for m in self._upstream_messages or [] if m["role"] != "system"
        ]:
            raise ValueError("VCS pinned upstream contract changed: parsing retries must reuse the same messages")
        self._upstream_messages = copy.deepcopy(messages)
        tools = request.get("tools") or []
        template = request.get("chat_template_kwargs") or {}
        changed = (systems, tools, template) != (self._systems, self._tools, self._template)
        old_system_count = len(self._systems)
        self._systems, self._tools, self._template = systems, tools, template

        reasons = []
        if self._request is not None:
            # Parsing retries resample the same causal input; the rejected answer
            # remains trainable in the preceding segment, as in baseline exports.
            reasons.append("parse_retry")
            body = copy.deepcopy(self._request["messages"][old_system_count + 1 :])
        elif not self._history:
            body = [messages[-1]]
        else:
            body = [*copy.deepcopy(self._history[old_system_count + 1 :]), messages[-1]]
        if changed and self._history:
            reasons.append("tools_or_system_or_template_changed")

        count = image_count(body)
        # Initial and current screenshots both count. Check the *upcoming*
        # request, but export the preceding committed response, not this new obs.
        prune = count > self.keep_images + self.prune_images
        if prune:
            reasons.append("image_window")
        if reasons:
            self.finish(
                "+".join(reasons),
                pruned_images=self.prune_images if prune else 0,
                trigger_images=count if prune else None,
            )
        if prune:
            remaining = self.prune_images
            for message in body:
                if not isinstance(message.get("content"), list):
                    continue
                blocks = []
                for block in message["content"]:
                    if block.get("type") == "image_url" and remaining:
                        remaining -= 1
                    else:
                        blocks.append(block)
                message["content"] = blocks or [{"type": "text", "text": "[Screenshot removed by VCS]"}]
        request["messages"] = (
            self._new_context(body)
            if reasons or not self._history
            else [*systems, copy.deepcopy(self._history[old_system_count]), *body]
        )
        self._request = copy.deepcopy(request)
        return request

    def _prepare_full_history(self, payload: dict[str, Any]) -> dict[str, Any]:
        request = copy.deepcopy(payload)
        messages = request["messages"]
        if not messages or messages[-1]["role"] == "assistant":
            raise ValueError("VCS expects a nonempty full history ending in an observation")
        systems = [m for m in messages if m["role"] == "system"]
        if messages[: len(systems)] != systems:
            raise ValueError("VCS requires system messages before conversation history")
        tools = request.get("tools") or []
        template = request.get("chat_template_kwargs") or {}
        changed = (systems, tools, template) != (self._systems, self._tools, self._template)
        old_system_count = len(self._systems)
        reasons = []
        # Match the source history, but extend the exact committed model history.
        # Never reconstruct a response from parsed actions or stripped reasoning.
        if self._source_committed and messages[: len(self._source_committed)] == self._source_committed:
            body = [
                *copy.deepcopy(self._history[old_system_count + 1 :]),
                *copy.deepcopy(messages[len(self._source_committed) :]),
            ]
        elif self._source_request == messages and self._request is not None:
            body = copy.deepcopy(self._request["messages"][old_system_count + 1 :])
            reasons.append("parse_retry")
        else:
            # A correction, reset, or intentional branch keeps the supplied text.
            # Earlier answers on this new root are observations, not new targets.
            body = copy.deepcopy(messages[len(systems) :])
            if self._history:
                reasons.append("history_changed")
        if changed and self._history:
            reasons.append("tools_or_system_or_template_changed")
        count = image_count(body)
        pruned = 0
        while count - pruned > self.keep_images + self.prune_images:
            pruned += self.prune_images
        if pruned:
            reasons.append("image_window")
        if reasons:
            self.finish("+".join(reasons), pruned_images=pruned, trigger_images=count if pruned else None)
        if pruned:
            remaining = pruned
            for message in body:
                if not isinstance(message.get("content"), list):
                    continue
                blocks = []
                for block in message["content"]:
                    if block.get("type") == "image_url" and remaining:
                        remaining -= 1
                    else:
                        blocks.append(block)
                message["content"] = blocks or [{"type": "text", "text": "[Screenshot removed by VCS]"}]
        self._systems, self._tools, self._template = systems, tools, template
        request["messages"] = (
            self._new_context(body)
            if reasons or not self._history
            else [*systems, copy.deepcopy(self._history[old_system_count]), *body]
        )
        self._source_request = copy.deepcopy(messages)
        self._request = copy.deepcopy(request)
        return request

    def commit(self, record: dict[str, Any], usage: dict[str, Any]) -> None:
        if self.history_mode == "full_history":
            if self._source_request is None:
                raise ValueError("VCS commit requires a prepared request")
            self._source_committed = [*copy.deepcopy(self._source_request), copy.deepcopy(record["messages"][-1])]
        record.update(
            turn_index=self._turn,
            parse_attempt=self._attempt,
            input_tokens=int(usage.get("prompt_tokens", 0)),
            output_tokens=int(usage.get("completion_tokens", 0)),
        )
        self._attempt += 1
        self.calls.append(record)
        self._history = record["messages"]

    def finish(self, reason: str, *, pruned_images: int = 0, trigger_images: int | None = None) -> None:
        if len(self.calls) == self._start_call:
            return
        calls = self.calls[self._start_call :]
        # Keep the exact committed snapshot, including tools and template kwargs.
        record = dict(calls[-1])
        record["vcs_segment"] = {
            "index": len(self.records),
            "reason": reason,
            "call_indices": list(range(self._start_call, len(self.calls))),
            "environment_turns": [c["turn_index"] for c in calls],
            "parse_attempts": [c["parse_attempt"] for c in calls],
            "new_answers": len(calls),
            "image_count": image_count(record["messages"]),
            "pruned_images": pruned_images,
            "trigger_images": trigger_images,
            "input_tokens": sum(c["input_tokens"] for c in calls),
            "answer_tokens": sum(c["output_tokens"] for c in calls),
        }
        self.records.append(record)
        self._start_call = len(self.calls)

    def metrics(self) -> dict[str, int]:
        return {
            f"{self.metric_prefix}/vcs_segments": len(self.records),
            f"{self.metric_prefix}/vcs_pruned_images": sum(r["vcs_segment"]["pruned_images"] for r in self.records),
            f"{self.metric_prefix}/input_tokens": sum(c["input_tokens"] for c in self.calls),
            f"{self.metric_prefix}/answer_tokens": sum(c["output_tokens"] for c in self.calls),
        }
