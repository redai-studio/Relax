# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""OSWorld request-time visual segmentation; token ownership stays in
SessionForest."""

from __future__ import annotations

import copy
from typing import Any


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

    The pinned upstream retries parsing with the same request, without the
    failed answer. Keep that branch policy. A segment marker before the first
    observation prevents Forest's longest-prefix lookup from reusing an older
    response ancestor (also when tools change back, or two retries generate
    identical answers).
    """

    def __init__(self, keep_images: int = 2, prune_images: int = 2) -> None:
        if keep_images < 1 or prune_images < 1:
            raise ValueError("VCS keep_images and prune_images must be positive")
        self.keep_images = keep_images
        self.prune_images = prune_images
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

    def commit(self, record: dict[str, Any], usage: dict[str, Any]) -> None:
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
            "osworld/vcs_segments": len(self.records),
            "osworld/vcs_pruned_images": sum(r["vcs_segment"]["pruned_images"] for r in self.records),
            "osworld/input_tokens": sum(c["input_tokens"] for c in self.calls),
            "osworld/answer_tokens": sum(c["output_tokens"] for c in self.calls),
        }
