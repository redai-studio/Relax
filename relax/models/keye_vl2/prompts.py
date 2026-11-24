# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from typing import Any


def openr1_prompt(prompt: list[dict[str, Any]], data: dict[str, Any]) -> list[dict[str, Any]]:
    """Keye's native template requires a string system message; put the OpenR1
    formatting instruction in the user turn instead."""
    messages = [dict(message) for message in prompt]
    messages[-1]["content"] += "\nPut your reasoning in <think> </think> and your final answer in <answer> </answer>."
    return messages
