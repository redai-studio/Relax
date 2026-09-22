# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Prompt templates, action parsing, and EM scoring for the Search-QA agent.

Ported from SDAR / verl-agent (GiGPO): prompt templates, action projection, and
EM scoring utilities.
"""

from __future__ import annotations

import os
import re
import string


# Prompt templates (verbatim from SDAR prompts/search.py).
SEARCH_TEMPLATE_NO_HIS = """
You are an expert agent tasked with answering the given question step-by-step.
Your question: {task_description}

Now it's your turn to respond for the current step.
You should first conduct reasoning process. This process MUST be enclosed within <think> </think> tags.
After completing your reasoning, choose only one of the following actions (do not perform both):
(1) If you find you lack some knowledge, you can call a search engine to get more external information using format: <search> your query </search>.
(2) If you have enough knowledge to answer the question confidently, provide your final answer within <answer> </answer> tags, without detailed illustrations. For example, <answer>Beijing</answer>.
"""

SEARCH_TEMPLATE = """
You are an expert agent tasked with answering the given question step-by-step.
Your question: {task_description}

Prior to this step, you have already taken {step_count} step(s). Below is the interaction history where <search> </search> wrapped your past search queries and <information> </information> wrapped the corresponding search results returned by the external search engine. History:
{memory_context}

Now it's your turn to respond for the current step.
You should first conduct reasoning process. This process MUST be enclosed within <think> </think> tags.
After completing your reasoning, choose only one of the following actions (do not perform both):
(1) If you find you lack some knowledge, you can call a search engine to get more external information using format: <search> your query </search>.
(2) If you have enough knowledge to answer the question confidently, provide your final answer within <answer> </answer> tags, without detailed illustrations. For example, <answer>Beijing</answer>.
"""


# Action parsing (ported from projection.py, specialized to a single action).
_SEARCH_BLOCK_RE = re.compile(r"<search>(.*?)</search>", re.IGNORECASE | re.DOTALL)
_ANSWER_BLOCK_RE = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)
_SEARCH_TAG_RE = re.compile(r"<search>", re.IGNORECASE)
_ANSWER_TAG_RE = re.compile(r"<answer>", re.IGNORECASE)


def _postprocess_action(action: str) -> str:
    """Trim everything after the first closing ``</search>`` /
    ``</answer>``."""
    if "</search>" in action:
        return action.split("</search>", 1)[0] + "</search>"
    if "</answer>" in action:
        return action.split("</answer>", 1)[0] + "</answer>"
    return action


def extract_action(response_text: str) -> tuple[str, bool]:
    """Project a raw model response into ``(action, is_valid)``.

    ``action`` is the first complete ``<search>...</search>`` block, else the
    first ``<answer>...</answer>`` block, else ``""``. Validity flips to False
    when no block is found, when both a search and an answer tag are present,
    or when either tag appears more than once (mirrors ``search_projection``).
    """
    trimmed = _postprocess_action(response_text)

    m = _SEARCH_BLOCK_RE.search(trimmed)
    if m:
        action = f"<search>{m.group(1).strip()}</search>"
    else:
        m = _ANSWER_BLOCK_RE.search(trimmed)
        action = f"<answer>{m.group(1).strip()}</answer>" if m else ""

    is_valid = action != ""
    n_search = len(_SEARCH_TAG_RE.findall(response_text))
    n_answer = len(_ANSWER_TAG_RE.findall(response_text))
    if n_search and n_answer:
        is_valid = False
    elif n_search > 1 or n_answer > 1:
        is_valid = False
    return action, is_valid


def parse_search_query(action: str) -> str | None:
    """Return the query inside a projected ``<search>...</search>`` action."""
    m = _SEARCH_BLOCK_RE.search(action)
    return m.group(1).strip() if m else None


def is_answer_action(action: str) -> bool:
    return action.startswith("<answer>") and action.endswith("</answer>")


# Prompt rendering.
#
# Render-time reasoning-tag rewrite: under ``enable_thinking=false`` Qwen3's chat
# template pre-fills an empty ``<think></think>``, so we rewrite the prompt's
# ``<think>`` to a neutral tag to avoid colliding with that reserved block.
# ``SEARCH_THINKING_TAG`` defaults to ``think`` (no rewrite).
THINKING_TAG = os.environ.get("SEARCH_THINKING_TAG", "think")


def _retag_reasoning(text: str) -> str:
    if THINKING_TAG == "think":
        return text
    return text.replace("<think>", f"<{THINKING_TAG}>").replace("</think>", f"</{THINKING_TAG}>")


def build_prompt(
    *,
    task: str,
    history: list[tuple[str, str]],
    history_length: int,
    init: bool = False,
) -> str:
    """Render a fresh single-turn prompt with an embedded history window.

    ``history`` is a list of ``(action, information)`` pairs, oldest first.
    Only the most recent ``history_length`` steps are embedded, each as ``Step
    {n}:{action} {information}`` (SDAR memory_context format).
    """
    if init or history_length <= 0 or not history:
        return _retag_reasoning(SEARCH_TEMPLATE_NO_HIS.format(task_description=task))
    recent = history[-history_length:]
    start = len(history) - len(recent)
    memory_context = "\n".join(
        f"Step {start + j + 1}:{action} {information}" for j, (action, information) in enumerate(recent)
    )
    return _retag_reasoning(
        SEARCH_TEMPLATE.format(
            task_description=task,
            step_count=len(history),
            memory_context=memory_context,
        )
    )


# Exact-match scoring (verbatim from SDAR utils.py; golds-list contract).
def normalize_answer(s: str) -> str:
    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def em_check(prediction: str, golden_answers: str | list[str]) -> int:
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    for golden_answer in golden_answers:
        if normalize_answer(golden_answer) == normalized_prediction:
            return 1
    return 0


def extract_solution(solution_str: str) -> str | None:
    """Return the content of the *last* ``<answer>...</answer>`` block, or
    None."""
    matches = list(re.finditer(r"<answer>(.*?)</answer>", solution_str, re.DOTALL))
    if not matches:
        return None
    return matches[-1].group(1).strip()


def compute_score(
    solution_str: str,
    golds: str | list[str],
    *,
    format_score: float = 0.0,
    score: float = 1.0,
) -> float:
    """Binary EM: 1.0 iff the last ``<answer>`` normalizes-equal to any
    gold."""
    answer = extract_solution(solution_str)
    if answer is None:
        return 0.0
    return score if em_check(answer, golds) else format_score
