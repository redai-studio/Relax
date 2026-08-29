"""policy_hybrid.py — owl-style prompt templates for hybrid GUI + MCP agent.

Uses the NousFnCall protocol that Qwen was trained on: tools are serialized into
the system message by qwen_agent.NousFnCallPrompt, and the model responds with
  <thinking>...</thinking>
  <tool_call>{"name": ..., "arguments": {...}}</tool_call>
  <conclusion>...</conclusion>

This module only owns the text templates. Message assembly (NousFnCallPrompt +
screenshots) lives in OSWorld-main/mm_agents/hybrid_agent_local.py.
"""

from __future__ import annotations


SYSTEM_TEXT_OLD = "You are a helpful assistant."

# Soft guidance: prefer MCP tools for efficiency but allow GUI fallback at any step.
# The previous "Do NOT mix pyautogui" hard restriction caused base models to attempt
# MCP-only paths they couldn't complete, collapsing pure-GUI accuracy from ~28% to ~14%.
SYSTEM_TEXT_NEW = "You are a helpful assistant."

USER_PROMPT_TEMPLATE = (
    "Please generate the next move according to the UI screenshot, instruction and previous actions.\n"
    "Instruction: {instruction}\n"
    "Previous actions:\n"
    "{history}\n"
    "Before answering, explain your reasoning step-by-step in <thinking></thinking> tags, "
    "and insert them before the <tool_call></tool_call> XML tags.\n"
    "After answering, summarize your action in <conclusion></conclusion> tags, "
    "and insert them after the <tool_call></tool_call> XML tags."
)


def build_system_text(variant: str = "old") -> str:
    """variant="old" → bare "You are a helpful assistant." (no tool guidance)
    variant="new" → soft MCP preference (GUI fallback always allowed)"""
    return SYSTEM_TEXT_NEW if variant == "new" else SYSTEM_TEXT_OLD


def build_user_prompt_text(instruction: str, history_str: str) -> str:
    return USER_PROMPT_TEMPLATE.format(
        instruction=instruction,
        history=history_str if history_str else "(none)",
    )
