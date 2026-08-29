"""error_summary.py.

Critic feedback prompt for the RLAnything error-summarization pipeline.
Extends the original OSWorld error summarizer with tool-usage analysis:
  - Did the agent fail to use an available MCP tool?
  - Did it get stuck in a GUI loop when a single tool call would suffice?
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Prompt templates
# ─────────────────────────────────────────────────────────────────────────────

ERROR_SUMMARY_SYSTEM = """\
You are a critic analysing why an AI agent failed to complete a computer task.
You will receive the full trajectory and must produce a structured failure analysis.
Focus especially on ROUTING MISTAKES — choosing the wrong action type — in addition to
standard logical errors.
"""

ERROR_SUMMARY_USER_TEMPLATE = """\
TASK: {task_description}
OUTCOME: {outcome}  (1.0 = success, 0.0 = failure)
TOTAL STEPS: {total_steps}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TRAJECTORY SUMMARY:
{trajectory_summary}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
AVAILABLE MCP TOOLS (during this rollout):
{tools_summary}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Analyse the failure and output a JSON object with these fields:
{{
  "primary_error_type":    "<one of: wrong_routing | wrong_tool | wrong_params | gui_stuck | task_misunderstood | env_error | other>",
  "routing_mistakes":      ["<describe each step where the wrong action type was chosen>"],
  "missed_tool_opportunities": ["<describe steps where GUI was used but an MCP tool was available>"],
  "suggested_corrections": ["<concrete correction for each mistake>"],
  "tool_hint_for_next_attempt": "<which MCP tool, if any, should be tried first in the next rollout>",
  "summary": "<1-2 sentence diagnosis>"
}}
"""

# ─────────────────────────────────────────────────────────────────────────────
# Builder
# ─────────────────────────────────────────────────────────────────────────────


def build_error_summary_messages(
    task_description: str,
    trajectory: List[Dict[str, Any]],
    outcome: float,
    available_tools: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Build the message list for the critic / error-summarization model.

    Parameters
    ----------
    task_description : str
    trajectory : list of step dicts
        Each dict should have: step_index, action_type, action_str, exec_ok, exec_result
    outcome : float
        Task success score (0.0 = failure, 1.0 = full success).
    available_tools : list | None
        MCP tools that were available during the rollout.

    Returns
    -------
    OpenAI-style messages list.
    """
    traj_lines = _format_trajectory(trajectory)
    tools_summary = _format_tools_summary(available_tools)

    user_text = ERROR_SUMMARY_USER_TEMPLATE.format(
        task_description=task_description,
        outcome=f"{outcome:.2f}",
        total_steps=len(trajectory),
        trajectory_summary=traj_lines,
        tools_summary=tools_summary,
    )

    return [
        {"role": "system", "content": ERROR_SUMMARY_SYSTEM.strip()},
        {"role": "user", "content": user_text},
    ]


def parse_error_summary(response_text: str) -> Dict[str, Any]:
    """Parse the critic's JSON output.

    Returns a default dict on failure.
    """
    defaults: Dict[str, Any] = {
        "primary_error_type": "other",
        "routing_mistakes": [],
        "missed_tool_opportunities": [],
        "suggested_corrections": [],
        "tool_hint_for_next_attempt": "",
        "summary": "",
    }
    if not response_text:
        return defaults

    start = response_text.find("{")
    end = response_text.rfind("}") + 1
    if start == -1 or end == 0:
        return defaults

    try:
        data = json.loads(response_text[start:end])
        for k in defaults:
            if k in data:
                defaults[k] = data[k]
    except json.JSONDecodeError:
        pass

    return defaults


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _format_trajectory(trajectory: List[Dict[str, Any]]) -> str:
    if not trajectory:
        return "(empty)"
    lines: List[str] = []
    for step in trajectory:
        idx = step.get("step_index", "?")
        atype = step.get("action_type", "gui")
        ok = "✓" if step.get("exec_ok", True) else "✗"
        action_str = _truncate(str(step.get("action_str", "")), 120)
        result_str = _truncate(str(step.get("exec_result", "")), 80)
        lines.append(
            f"  Step {idx:>2} [{atype}] {ok}  {action_str}"
            + (f"\n           result: {result_str}" if result_str else "")
        )
    return "\n".join(lines)


def _format_tools_summary(tools: Optional[List[Dict[str, Any]]]) -> str:
    if not tools:
        return "none"
    lines = []
    for t in tools[:20]:
        name = t.get("name", "")
        desc = _truncate(t.get("description", ""), 80)
        lines.append(f"  - {name}: {desc}")
    return "\n".join(lines)


def _truncate(text: str, max_len: int) -> str:
    return text if len(text) <= max_len else text[: max_len - 3] + "…"
