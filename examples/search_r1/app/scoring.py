# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Pure Search-R1 answer scoring used by the managed agent."""

from __future__ import annotations

import re
import string
from dataclasses import dataclass
from typing import Any, Sequence

from .protocol import parse_action


_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_ASCII_PUNCTUATION = set(string.punctuation)


def normalize_answer(text: str) -> str:
    lowered = text.lower()
    without_punctuation = "".join(character for character in lowered if character not in _ASCII_PUNCTUATION)
    without_articles = re.sub(r"\b(a|an|the)\b", " ", without_punctuation)
    return " ".join(without_articles.split())


def extract_last_answer(response: str) -> str | None:
    matches = list(_ANSWER_RE.finditer(response))
    if not matches:
        return None
    return matches[-1].group(1).strip()


def exact_match(prediction: str | None, aliases: Sequence[str]) -> float:
    if prediction is None:
        return 0.0
    normalized_prediction = normalize_answer(prediction)
    return float(any(normalize_answer(alias) == normalized_prediction for alias in aliases))


def action_protocol_is_valid(messages: list[dict[str, Any]], prompt_message_count: int) -> bool:
    awaiting_information = False
    finished = False
    for message in messages[prompt_message_count:]:
        role = message.get("role")
        if role == "assistant":
            if awaiting_information or finished:
                return False
            action, content = parse_action(message.get("content") or "")
            if not content or action is None:
                return False
            if action == "search":
                awaiting_information = True
            else:
                finished = True
        elif role == "tool":
            if not awaiting_information:
                return False
            awaiting_information = False
        else:
            return False
    return finished and not awaiting_information


@dataclass(frozen=True)
class SearchR1Score:
    outcome_em: float
    shaped_reward: float
    final_answer_valid: float
    action_protocol_valid: float


def compute_score(
    messages: list[dict[str, Any]],
    prompt_message_count: int,
    aliases: Sequence[str],
    *,
    structure_weight: float,
    final_weight: float,
) -> SearchR1Score:
    response = "\n".join(
        message["content"] for message in messages[prompt_message_count:] if message.get("role") == "assistant"
    )
    action_protocol_valid = action_protocol_is_valid(messages, prompt_message_count)
    answer = extract_last_answer(response)
    outcome_em = exact_match(answer, aliases)
    final_answer_valid = float(bool(answer))
    protocol_valid = float(action_protocol_valid)
    if outcome_em:
        shaped_reward = 1.0 if action_protocol_valid else 1.0 - structure_weight
    elif action_protocol_valid:
        shaped_reward = structure_weight
    elif final_answer_valid:
        shaped_reward = final_weight
    else:
        shaped_reward = 0.0
    return SearchR1Score(
        outcome_em=outcome_em,
        shaped_reward=shaped_reward,
        final_answer_valid=final_answer_valid,
        action_protocol_valid=protocol_valid,
    )


def search_count(messages: list[dict[str, Any]], prompt_message_count: int) -> int:
    count = 0
    for message in messages[prompt_message_count:]:
        if message.get("role") == "assistant":
            action, content = parse_action(message.get("content") or "")
            count += action == "search" and bool(content)
    return count


def main_metadata(
    *,
    messages: list[dict[str, Any]],
    prompt_message_count: int,
    score: SearchR1Score,
    termination_reason: str,
) -> dict[str, Any]:
    return {
        "role": "main",
        "score/main": score.outcome_em,
        "reward/shaped": score.shaped_reward,
        "format/action_protocol_valid": score.action_protocol_valid,
        "format/final_answer_valid": score.final_answer_valid,
        "search/count": search_count(messages, prompt_message_count),
        "termination/reason": termination_reason,
        "termination/context_length_exceeded": float(termination_reason == "context_length_exceeded"),
        "termination/search_budget_exhausted": float(termination_reason == "search_budget_exhausted"),
        "termination/invalid_action": float(termination_reason == "invalid_action"),
    }
