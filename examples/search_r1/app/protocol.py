# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Search-R1 action parsing and Chat Completions requests."""

from __future__ import annotations

import re
from typing import Any


ACTION_STOPS = ["</search>", "</answer>"]
INVALID_ACTION_OBSERVATION = (
    "\nMy previous action is invalid. If I want to search, I should put the query between <search> and </search>. "
    "If I want to give the final answer, I should put the answer between <answer> and </answer>. Let me try again.\n"
)


_ACTION_RE = re.compile(r"<(search|answer)>(.*?)</\1>", re.DOTALL)


def parse_action(text: str) -> tuple[str | None, str]:
    action = _ACTION_RE.search(text)
    if action is None:
        return None, ""
    return action.group(1), action.group(2).strip()


async def request_action(
    *,
    client: Any,
    history: list[dict[str, Any]],
    observation: str | None,
    final_turn: bool,
) -> tuple[str, str | None, str | None]:
    tool_message = None if observation is None else {"role": "tool", "content": observation}
    request_messages = history if tool_message is None else [*history, tool_message]
    try:
        response = await client.chat.completions.create(model="model", messages=request_messages, stop=ACTION_STOPS)
    except Exception as exc:
        body = getattr(exc, "body", None)
        error = body.get("error", body) if isinstance(body, dict) else {}
        if getattr(exc, "status_code", None) == 400 and error.get("code") == "context_length_exceeded":
            return "", "context_length_exceeded", None
        raise

    choice = response.choices[0]
    message = choice.message.model_dump()
    if tool_message is not None:
        history.append(tool_message)
    history.append(message)
    action, content = parse_action(choice.message.content)
    if action == "answer":
        return content, "answer" if content else "invalid_action", None
    if final_turn:
        return content, "invalid_action" if action is None else "search_budget_exhausted", None
    if action is None:
        return content, None, INVALID_ACTION_OBSERVATION
    return content, None, None
