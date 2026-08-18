# tool_doc_hints.py — prompt-side tool-doc enhancement (decision gate, 2026-07-06)
#
# Background: the tool-effectiveness study (b2_think_base, 5 repeats / 448 calls)
# traced every failure to parameter semantics:
#   find_and_replace 0/23 (regex over-escaping → "0 replacements" still reported
#   success / wild regex guesses), pure read-only trajectories 0/39,
#   convert_to_docx 0/16, etc. Infrastructure was not at fault (exec_ok 98-100%).
# This module injects "correct usage / common pitfalls" into tool descriptions as
# documentation — lifting the sampling probability of a correct call from ~0 to a
# level RL can see (exploration seeding on the prompt side, without touching SFT).
# Called by hybrid_agent_local only when OSWORLD_TOOL_DOC_ENHANCE=1; zero impact by default.
# NOTE: the MCP server side is NOT modified (DON'T) — enhancement happens only on
# the <tools> text injected on the agent side.

from typing import Any, Dict, List

# key = last segment of the tool name (suffix match); value = USAGE NOTES appended to the description
TOOL_HINTS: Dict[str, str] = {
    "find_and_replace": (
        " USAGE NOTES: 'pattern' is a REGULAR EXPRESSION, not plain text. Plain "
        "words need no escaping — e.g. to replace every 'colour' with 'color' "
        "across the whole document call "
        '{"pattern": "colour", "replacement": "color"} '
        "and OMIT paragraph_indices entirely (omitting = all paragraphs; do NOT "
        "pass an empty list). To match a regex special character (. * + ? ( ) [ ]) "
        "literally, escape it with a backslash (doubled inside JSON). Prefer the "
        "simplest pattern that matches. ALWAYS read the returned replacement "
        "count: '0 replacements' means the document was NOT changed and your "
        "pattern is wrong — fix it or switch to the GUI."
    ),
    "env_info": (
        " USAGE NOTES: read-only. Call at most once or twice to inspect state, then "
        "act with an effectful tool or the GUI — repeated info calls make no progress."
    ),
    "get_workbook_info": (
        " USAGE NOTES: read-only. Inspect once, then act — this call by itself never "
        "completes any task."
    ),
    "convert_to_docx": (
        " USAGE NOTES: you MUST pass output_path (full absolute path ending in "
        ".docx, e.g. /home/user/xxx.docx) — if omitted, the converted document is "
        "kept in memory and NOTHING is saved to disk. Verify afterwards that the "
        "file exists where the task requires it."
    ),
    "sort_column": (
        " USAGE NOTES: column_name is a letter ('A','B',...). start_index defaults "
        "to 2 (data starts at row 2, header row untouched) — pass start_index=1 "
        "ONLY if there is no header. Verify the visible result on the next "
        "screenshot before moving on."
    ),
    "reorder_columns": (
        " USAGE NOTES: list ALL columns in the desired final order, not just the ones "
        "that move. Verify the visible result on the next screenshot before moving on."
    ),
    "set_slide_background": (
        " USAGE NOTES: omit slide_index only when the task wants ALL slides changed; "
        "otherwise pass the exact 1-based slide index. Verify visually afterwards."
    ),
}

# Generic discipline section appended after mcp_hint (only when tools are available)
GENERIC_DISCIPLINE = (
    "\n**Tool usage discipline**:\n"
    "1. A tool result reports EXECUTION success, not task progress. Read the result "
    "text: counts like '0 replacements' or an unchanged value mean nothing was "
    "modified — your parameters were wrong.\n"
    "2. After any state-changing MCP call, verify the effect on the next screenshot "
    "before proceeding.\n"
    "3. Read-only tools (env_info / get_*) only gather information; they never "
    "complete a task by themselves.\n"
    "4. If a tool call did not visibly achieve the goal after one retry, switch to "
    "GUI actions instead of repeating the call.\n"
)


def enhance_tool_defs(all_tool_defs: List[Dict[str, Any]]) -> int:
    """Append USAGE NOTES to matching tool descriptions in-place.

    all_tool_defs: the qwen-format list built in hybrid_agent_local (computer_use
    first, then MCP tools). Matching is by the last dot-segment of the tool name.
    Returns the number of tools enhanced (for logging).
    """
    n = 0
    for td in all_tool_defs:
        fn = td.get("function") or {}
        name = fn.get("name") or ""
        suffix = name.rsplit(".", 1)[-1]
        hint = TOOL_HINTS.get(suffix)
        if hint and isinstance(fn.get("description"), str):
            if "USAGE NOTES:" not in fn["description"]:
                fn["description"] = fn["description"].rstrip() + hint
                n += 1
    return n


def zero_effect_warning(feedback: str) -> str:
    """Detect silent no-op results in MCP feedback and append an explicit warning.

    From the tool-effectiveness study: find_and_replace returns '"success":true' +
    'Successfully made 0 replacements' — the model treats execution success as goal
    success. This appends one explicit warning line to the feedback text on the
    agent side (the server is untouched). Non-no-op feedback passes through as-is.
    """
    if not feedback:
        return feedback
    low = feedback.lower()
    if "0 replacements" in low or "made 0 " in low or "0 matches" in low:
        return (
            feedback
            + "\n[WARNING] The call executed but changed NOTHING (0 replacements/"
            "matches). Do not proceed as if it worked — fix the parameters or use the GUI."
        )
    return feedback
