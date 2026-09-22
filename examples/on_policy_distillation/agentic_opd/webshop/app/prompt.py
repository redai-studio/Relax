# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""WebShop prompt templates, observation formatting, and action parsing.

This module is the single source of truth for the WebShop rollout protocol and
is imported by both sides of the shared-server design:

* ``server.py`` uses :func:`flatten_actions` to turn WebShop's
  ``get_available_actions()`` dict into the admissible-action list returned over
  HTTP.
* ``agent.py`` uses :func:`extract_task`, :func:`format_obs`,
  :func:`build_prompt` (multi-turn history lives agent-side) and
  :func:`extract_action` (parse the model's ``<action>``).

The templates, the >13000-char history fallback, and the validity rule are kept
identical to the original single-process ``env_webshop.py`` (which mirrored
SDAR's ``WebshopEnvironmentManager`` / ``webshop_projection``), so a model
trained/evaluated here stays comparable to the SDAR recipe.
"""

from __future__ import annotations

import re
from typing import Any


# --------------------------------------------------------------------------- #
# Prompt templates (kept identical to SDAR's prompts/webshop.py).
# --------------------------------------------------------------------------- #
WEBSHOP_TEMPLATE_NO_HIS = """
You are an expert autonomous agent operating in the WebShop e‑commerce environment.
Your task is to: {task_description}.
Your current observation is: {current_observation}.
Your admissible actions of the current situation are:
[
{available_actions}
].

Now it's your turn to take one action for the current step.
You should first reason step-by-step about the current situation, then think carefully which admissible action best advances the shopping goal. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""

WEBSHOP_TEMPLATE = """
You are an expert autonomous agent operating in the WebShop e‑commerce environment.
Your task is to: {task_description}.
Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}.
Your admissible actions of the current situation are:
[
{available_actions}
].

Now it's your turn to take one action for the current step.
You should first reason step-by-step about the current situation, then think carefully which admissible action best advances the shopping goal. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""

# Above this rendered length SDAR falls back to the no-history template to avoid
# blowing the prompt budget (see env_manager.py:496-502).
_MAX_OBS_CHARS = 13000

_ACTION_RE = re.compile(r"<action>(.*?)</action>", re.DOTALL | re.IGNORECASE)
_CHINESE_RE = re.compile("[一-鿿]")


def extract_action(response_text: str) -> tuple[str, bool]:
    """Extract the WebShop action from a model response.

    Returns ``(action, is_valid)``. The action is the lower-cased content inside
    the *first* ``<action>`` block; a response is "valid" when an ``<action>``
    block is present and it contains no Chinese characters.

    NB: unlike SDAR's ``webshop_projection`` we do NOT require a response-side
    ``<think>``. SDAR runs Qwen3 with ``enable_thinking=False``, so its
    response-only ``<think>`` check marks every action invalid and (with
    invalid_action_penalty on) systematically penalizes the policy (see
    ZJU-REAL/SDAR issues #21, #34). We run non-thinking mode too, so
    validity = action parsed + no Chinese, matching the ALFWorld recipe.
    """
    match = _ACTION_RE.search(response_text)
    if match is None:
        # No parsable action: send a harmless tail so the env replies with an
        # unchanged page and the episode can continue / time out.
        return response_text.strip().lower()[-20:], False
    action = match.group(1).strip().lower()
    is_valid = _CHINESE_RE.search(response_text) is None
    return action, is_valid


def extract_task(raw_obs: str) -> str:
    """Pull the instruction/goal text out of the reset observation.

    Layout: "... [SEP] Instruction: [SEP] <task> [SEP] ..." (env_manager.py:426-432).
    """
    parts = raw_obs.split(" [SEP] ")
    if len(parts) >= 3 and parts[1] == "Instruction:":
        return parts[2]
    return raw_obs


def format_obs(raw_obs: str, task: str) -> str:
    """Format a raw WebShop text observation into the prompt's observation
    slot.

    Drops everything up to and including the task instruction, quoting each
    remaining ``[SEP]``-separated fragment (mirrors env_manager.py:434-441).
    """
    parts = raw_obs.split(" [SEP] ")
    try:
        index = parts.index(task)
        return " [SEP] ".join(f"'{p}'" for p in parts[index + 1 :])
    except ValueError:
        return raw_obs


def flatten_actions(available_actions: dict[str, Any]) -> list[str]:
    """Turn WebShop's ``get_available_actions()`` dict into admissible strings.

    ``{"has_search_bar": bool, "clickables": [str]}`` ->
    ``["search[<your query>]", "click[<text>]", ...]`` (mirrors
    env_manager.py ``format_avail_actions``). Used server-side so the HTTP
    contract carries a plain ``list[str]`` (DESIGN §6.1).
    """
    for key in available_actions.keys():
        if key not in ("has_search_bar", "clickables"):
            raise ValueError(f"Unknown key in available actions: {key}")
    actions: list[str] = []
    if available_actions.get("has_search_bar"):
        actions.append("search[<your query>]")
    for txt in available_actions.get("clickables", []):
        actions.append(f"click[{txt}]")
    return actions


def _format_admissible(available_actions: list[str]) -> str:
    return "\n".join(f"'{s}'," for s in available_actions)


def build_prompt(
    *,
    task: str,
    obs: str,
    available_actions: list[str],
    history: list[tuple[str, str]],
    history_length: int,
    init: bool = False,
) -> str:
    """Render the per-turn prompt from the task, current obs and (obs, action)
    history.

    ``history`` is the list of past ``(formatted_observation, action)`` pairs
    the agent maintains; only the last ``history_length`` are shown. Falls back
    to the no-history template on the first turn, when history is
    disabled/empty, or when the rendered prompt would exceed
    ``_MAX_OBS_CHARS``.
    """
    admissible = _format_admissible(available_actions)
    if init or history_length <= 0 or not history:
        return WEBSHOP_TEMPLATE_NO_HIS.format(
            task_description=task,
            current_observation=obs,
            available_actions=admissible,
        )
    recent = history[-history_length:]
    start = len(history) - len(recent)
    action_history = "".join(
        f"\n[Observation {start + j + 1}: '{h_obs}', Action {start + j + 1}: '{act}']"
        for j, (h_obs, act) in enumerate(recent)
    )
    prompt = WEBSHOP_TEMPLATE.format(
        task_description=task,
        step_count=len(history),
        history_length=len(recent),
        action_history=action_history,
        current_step=len(history) + 1,
        current_observation=obs,
        available_actions=admissible,
    )
    if len(prompt) > _MAX_OBS_CHARS:
        return WEBSHOP_TEMPLATE_NO_HIS.format(
            task_description=task,
            current_observation=obs,
            available_actions=admissible,
        )
    return prompt
