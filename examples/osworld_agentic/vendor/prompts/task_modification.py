"""task_modification.py.

Prompt templates for the co-evolution environment model: generates harder /
easier variants of OSWorld tasks, now extended with a "tool hint dimension" —
newly generated tasks can require more or fewer MCP tool calls depending on the
desired difficulty direction.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Prompt templates
# ─────────────────────────────────────────────────────────────────────────────

TASK_MOD_SYSTEM = """\
You are an expert at designing computer-use benchmark tasks.
Given an existing OSWorld task and the agent's recent performance, generate a
modified version of the task at the requested difficulty level.

Consider two dimensions when adjusting difficulty:
  1. GUI complexity  – more menus, nested dialogs, precise coordinates
  2. MCP tool usage  – whether the task requires structured API calls

Output ONLY valid JSON — no explanation outside the JSON block.
"""

TASK_MOD_USER_TEMPLATE = """\
ORIGINAL TASK:
{original_task_json}

AGENT PERFORMANCE: accuracy = {accuracy:.2f}  (over {num_trials} trials)
TARGET DIFFICULTY: {difficulty}  (harder | easier)
AVAILABLE MCP TOOLS: {tools_summary}

Generate a modified task JSON with these fields:
{{
  "instruction": "<modified natural-language task instruction>",
  "app":         "<application name>",
  "difficulty_direction": "{difficulty}",
  "gui_steps_estimate":   <int>,   // expected number of GUI actions
  "mcp_steps_estimate":   <int>,   // expected MCP tool calls
  "key_tool_hints":       ["<MCP tool names most likely needed>"],
  "rationale":            "<why this is {difficulty} than the original>"
}}
"""

# ─────────────────────────────────────────────────────────────────────────────
# Builder
# ─────────────────────────────────────────────────────────────────────────────


def build_task_modification_messages(
    original_task: Dict[str, Any],
    accuracy: float,
    difficulty: str,
    num_trials: int = 8,
    available_tools: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Build the message list for the task-modification (environment) model.

    Parameters
    ----------
    original_task : dict
        The current_task dict from the training example JSON.
    accuracy : float
        Agent's recent accuracy on this task (0.0–1.0).
    difficulty : "harder" | "easier"
        Direction for the modification.
    num_trials : int
        How many rollouts the accuracy was measured over.
    available_tools : list | None
        MCP tool schemas to provide as context.

    Returns
    -------
    OpenAI-style messages list.
    """
    tools_summary = _format_tools_for_prompt(available_tools)
    task_json_str = json.dumps(original_task, ensure_ascii=False, indent=2)

    user_text = TASK_MOD_USER_TEMPLATE.format(
        original_task_json=task_json_str,
        accuracy=accuracy,
        num_trials=num_trials,
        difficulty=difficulty,
        tools_summary=tools_summary,
    )

    return [
        {"role": "system", "content": TASK_MOD_SYSTEM.strip()},
        {"role": "user", "content": user_text},
    ]


def parse_task_modification(response_text: str) -> Optional[Dict[str, Any]]:
    """Parse the environment model's task modification JSON.

    Returns None if parsing fails.
    """
    if not response_text:
        return None

    start = response_text.find("{")
    end = response_text.rfind("}") + 1
    if start == -1 or end == 0:
        return None

    try:
        return json.loads(response_text[start:end])
    except json.JSONDecodeError:
        return None


def merge_task_modification(
    original_task: Dict[str, Any],
    modification: Dict[str, Any],
) -> Dict[str, Any]:
    """Merge a parsed modification dict back into the task config format
    expected by OSWorld's evaluation_examples JSON."""
    updated = dict(original_task)
    if "instruction" in modification:
        # Try to set the instruction in the standard OSWorld location
        if isinstance(updated.get("instruction"), str):
            updated["instruction"] = modification["instruction"]
        else:
            updated["instruction"] = modification["instruction"]
    updated["_hybrid_meta"] = {
        "difficulty_direction": modification.get("difficulty_direction", ""),
        "gui_steps_estimate": modification.get("gui_steps_estimate", 0),
        "mcp_steps_estimate": modification.get("mcp_steps_estimate", 0),
        "key_tool_hints": modification.get("key_tool_hints", []),
        "rationale": modification.get("rationale", ""),
    }
    return updated


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _format_tools_for_prompt(tools: Optional[List[Dict[str, Any]]]) -> str:
    if not tools:
        return "none"
    names = [t.get("name", "") for t in tools[:20]]
    return ", ".join(names)
