"""reward_hybrid.py.

Reward model prompt templates for multi-signal evaluation of the two action
types (gui / mcp).

Signal routing:
  gui  → screenshot diff (same as original OSWorld reward)
  mcp  → primarily tool return value JSON, screenshot as secondary signal

Additionally scores ROUTING QUALITY:
  - Penalise using GUI when an MCP tool was available (wasted opportunity)
  - Penalise using MCP when the only sensible option was GUI
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Shared header
# ─────────────────────────────────────────────────────────────────────────────

_SHARED_HEADER = """\
You are a reward model evaluating an AI agent that controls a computer desktop.
Score the agent's action for this task step.

TASK: {task_description}
STEP: {step_index}
ACTION TYPE CHOSEN: {action_type}
"""

_SCORE_FORMAT = """\
Output a JSON object with EXACTLY these fields:
{{
  "task_progress":   <float 0.0-1.0>,  // how much this step advanced the task
  "action_quality":  <float 0.0-1.0>,  // correctness of the chosen action
  "routing_quality": <float 0.0-1.0>,  // was action_type the right choice?
  "overall":         <float 0.0-1.0>,  // weighted combination
  "rationale":       "<one sentence>"
}}
Weights: overall = 0.5*task_progress + 0.3*action_quality + 0.2*routing_quality
"""

# ─────────────────────────────────────────────────────────────────────────────
# GUI prompt
# ─────────────────────────────────────────────────────────────────────────────

GUI_REWARD_PROMPT = (
    _SHARED_HEADER
    + """
GUI ACTION: {action_str}

BEFORE screenshot: [attached image 1]
AFTER  screenshot: [attached image 2]

Evaluate:
1. Did the GUI action produce a visible, meaningful change on screen?
2. Does the screen state after the action look closer to task completion?
3. Was GUI the right choice? Was an MCP tool available that would have been faster?
   Available MCP tools: {available_tools_summary}

"""
    + _SCORE_FORMAT
)

# ─────────────────────────────────────────────────────────────────────────────
# MCP prompt
# ─────────────────────────────────────────────────────────────────────────────

MCP_REWARD_PROMPT = (
    _SHARED_HEADER
    + """
MCP TOOL CALLED: {tool_name}
PARAMS: {params_json}
TOOL RETURN VALUE:
{tool_result_json}

AFTER screenshot: [attached image]

Evaluate:
1. Did the tool call succeed (ok=true in the return value)?
2. Does the return value indicate the intended operation was performed correctly?
3. Does the screenshot confirm the expected state change in the application?
4. Was MCP the right choice for this step, or would GUI have been better?

"""
    + _SCORE_FORMAT
)

# ─────────────────────────────────────────────────────────────────────────────
# Builder functions
# ─────────────────────────────────────────────────────────────────────────────


def build_reward_messages(
    task_description: str,
    step_index: int,
    action_type: str,
    action_dict: Dict[str, Any],
    exec_result: Dict[str, Any],
    screenshot_before_b64: Optional[str] = None,
    screenshot_after_b64: Optional[str] = None,
    available_tools: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Build the reward model message list for one agent step.

    Parameters
    ----------
    task_description : str
    step_index : int
    action_type : "gui" | "mcp"
    action_dict : Dict  – the parsed action
    exec_result : Dict  – the execution result
    screenshot_before_b64 : str | None  – base64 PNG before action (gui only)
    screenshot_after_b64  : str | None  – base64 PNG after  action
    available_tools : list | None       – tools the agent had access to

    Returns
    -------
    OpenAI-style messages list for the reward model call.
    """
    tools_summary = _format_tools_summary(available_tools)

    header_vars = dict(
        task_description=task_description,
        step_index=step_index,
        action_type=action_type,
    )

    if action_type == "mcp":
        prompt_text = MCP_REWARD_PROMPT.format(
            tool_name=action_dict.get("tool_name", ""),
            params_json=json.dumps(action_dict.get("params", {}), indent=2),
            tool_result_json=json.dumps(exec_result.get("_raw", exec_result), indent=2),
            **header_vars,
        )
    else:
        prompt_text = GUI_REWARD_PROMPT.format(
            action_str=action_dict.get("action", ""),
            available_tools_summary=tools_summary,
            **header_vars,
        )

    content: List[Dict[str, Any]] = [{"type": "text", "text": prompt_text}]

    # Attach screenshots as image blocks
    if action_type == "gui" and screenshot_before_b64:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{screenshot_before_b64}"},
            }
        )
    if screenshot_after_b64:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{screenshot_after_b64}"},
            }
        )

    return [{"role": "user", "content": content}]


def parse_reward_response(response_text: str) -> Dict[str, float]:
    """Parse the reward model's JSON output into a scores dict.

    Returns default zeros on any parse failure.
    """
    defaults = {
        "task_progress": 0.0,
        "action_quality": 0.0,
        "routing_quality": 0.0,
        "overall": 0.0,
        "rationale": "",
    }
    if not response_text:
        return defaults

    # Find JSON block
    start = response_text.find("{")
    end = response_text.rfind("}") + 1
    if start == -1 or end == 0:
        return defaults

    try:
        data = json.loads(response_text[start:end])
    except json.JSONDecodeError:
        return defaults

    for k in ("task_progress", "action_quality", "routing_quality", "overall"):
        try:
            defaults[k] = float(data.get(k, 0.0))
        except (TypeError, ValueError):
            pass
    defaults["rationale"] = str(data.get("rationale", ""))
    return defaults


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _format_tools_summary(tools: Optional[List[Dict[str, Any]]]) -> str:
    if not tools:
        return "none"
    names = [t.get("name", "") for t in tools[:10]]
    return ", ".join(names)
