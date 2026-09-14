# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Check installed OpenHands tool-error history without model or sandbox calls.

Run with the OpenHands virtualenv Python from its checkout directory. The
verifier imports the installed ConversationMemory directly; it applies no
patch.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any


def build_events(kinds: tuple[str, ...], response_id: str) -> tuple[list[Any], Any]:
    from litellm import ModelResponse
    from openhands.agenthub.codeact_agent.function_calling import response_to_actions
    from openhands.events.action import AgentThinkAction, FunctionCallNotExistsAction, ValidationFailureAction
    from openhands.events.event import EventSource
    from openhands.events.observation.agent import (
        AgentThinkObservation,
        FunctionCallNotExistsObservation,
        ValidationFailureObservation,
    )

    definitions = {
        "invalid": ("str_replace_editor", {"command": "view"}, ValidationFailureAction),
        "unknown": ("unknown_regression_tool", {}, FunctionCallNotExistsAction),
        "valid": ("think", {"thought": "Check the next step."}, AgentThinkAction),
    }
    calls = []
    for index, kind in enumerate(kinds):
        name, arguments, _ = definitions[kind]
        calls.append(
            {
                "id": f"{response_id}-call-{index}",
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments)},
            }
        )
    response = ModelResponse(
        id=response_id,
        model="regression-model",
        choices=[
            {
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {"role": "assistant", "content": "Inspect the file.", "tool_calls": calls},
            }
        ],
    )
    actions = response_to_actions(response)
    assert len(actions) == len(kinds)
    events = []
    for kind, action in zip(kinds, actions):
        assert isinstance(action, definitions[kind][2]), f"Unexpected action for {kind}"
        action._source = EventSource.AGENT
        if isinstance(action, ValidationFailureAction):
            observation = ValidationFailureObservation(
                content="validation failed", function_name=action.function_name, error_message=action.error_message
            )
        elif isinstance(action, FunctionCallNotExistsAction):
            observation = FunctionCallNotExistsObservation(
                content="unknown tool", function_name=action.function_name, error_message=action.error_message
            )
        else:
            observation = AgentThinkObservation(content="Thought recorded.")
        observation.tool_call_metadata = action.tool_call_metadata
        events.extend([action, observation])
    return events, response


def render(events: list[Any]) -> list[Any]:
    from openhands.core.config.agent_config import AgentConfig
    from openhands.events.action import MessageAction
    from openhands.events.action.message import SystemMessageAction
    from openhands.events.event import EventSource
    from openhands.memory.conversation_memory import ConversationMemory

    user = MessageAction(content="Inspect the repository.")
    user._source = EventSource.USER
    # Explicit system/user events avoid prompt construction or external resources.
    history = [SystemMessageAction(content="You are a coding assistant."), user, *copy.deepcopy(events)]
    return ConversationMemory(AgentConfig(), None).process_events(history, user, max_message_chars=10000)


def verify_case(kinds: tuple[str, ...]) -> dict[str, Any]:
    events, response = build_events(kinds, "response-1")
    messages = render(events)
    assistants = [message for message in messages if message.role == "assistant"]
    tools = [message for message in messages if message.role == "tool"]
    assert len(assistants) == 1, f"{kinds}: expected one assistant response, got {len(assistants)}"
    expected = response.choices[0].message
    assert [call.model_dump() for call in assistants[0].tool_calls] == [
        call.model_dump() for call in expected.tool_calls
    ], f"{kinds}: original tool calls changed"
    assert assistants[0].content[0].text == expected.content, f"{kinds}: assistant content changed"
    assert [message.tool_call_id for message in tools] == [call.id for call in expected.tool_calls], (
        f"{kinds}: missing or reordered tool feedback"
    )
    for kind, tool in zip(kinds, tools):
        if kind == "invalid":
            assert "path" in tool.content[0].text, "Missing parameter error was not replayed"
        if kind == "unknown":
            assert "unknown_regression_tool" in tool.content[0].text, "Unknown tool error was not replayed"
    next_events, _ = build_events(("valid",), "response-2")
    continued = render([*events, *next_events])
    assert [message.model_dump() for message in continued[: len(messages)]] == [
        message.model_dump() for message in messages
    ], f"{kinds}: subsequent request changed earlier history"
    return {"tools": kinds, "roles": [message.role for message in messages], "prefix_preserved": True}


def main() -> None:
    import openhands.memory.conversation_memory as memory_module

    cases = (
        ("invalid",),
        ("unknown",),
        ("invalid", "unknown"),
        ("valid", "invalid"),
        ("invalid", "valid"),
        ("valid",),
    )
    results = [verify_case(kinds) for kinds in cases]
    source = Path(memory_module.__file__).resolve()
    print(
        json.dumps(
            {
                "status": "PASS",
                "passed": len(results),
                "module_file": str(source),
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "scope": "installed parser and history serializer; synthetic observations; no model or sandbox calls",
                "cases": results,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
