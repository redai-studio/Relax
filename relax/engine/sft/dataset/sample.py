# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Canonical SFT sample contract."""

from dataclasses import dataclass
from typing import Any


VALID_ROLES = {"system", "user", "assistant", "tool", "function_call"}


@dataclass
class CanonicalMessage:
    """One turn in a conversation."""

    role: str
    content: str | list[dict]
    learn: bool
    # OpenAI-style assistant tool calls, e.g.
    # [{"type": "function", "function": {"name": ..., "arguments": {...}}}].
    # Passed through to the chat template unchanged.
    tool_calls: list[dict] | None = None
    reasoning_content: str | None = None

    def __post_init__(self) -> None:
        if self.role not in VALID_ROLES:
            raise ValueError(f"CanonicalMessage.role must be one of {VALID_ROLES}, got {self.role!r}")


@dataclass
class CanonicalSample:
    """One SFT training sample after row normalization."""

    messages: list[CanonicalMessage]
    metadata: dict[str, Any]
    tools: list[dict] | None = None
    images: list[Any] | None = None
    videos: list[Any] | None = None
    audios: list[Any] | None = None

    def __post_init__(self) -> None:
        if "source_dataset" not in self.metadata:
            raise ValueError("CanonicalSample.metadata must include 'source_dataset'")
        if "row_index" not in self.metadata:
            raise ValueError("CanonicalSample.metadata must include 'row_index'")
