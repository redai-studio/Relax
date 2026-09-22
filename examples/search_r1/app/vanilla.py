# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Vanilla Search-R1 agent loop."""

from __future__ import annotations

from typing import Any

from .protocol import request_action
from .retriever import format_information
from .scoring import compute_score, main_metadata


async def run_vanilla_session(
    *,
    messages: list[dict[str, Any]],
    aliases: list[str],
    client: Any,
    retriever: Any,
    max_observation_chars: int,
    max_search_turns: int,
    structure_weight: float,
    final_weight: float,
    rollout_mode: str,
) -> dict[str, Any] | list[dict[str, Any]]:
    prompt_message_count = len(messages)
    history = list(messages)
    pending_observation: str | None = None
    termination_reason = "search_budget_exhausted"

    for turn in range(max_search_turns + 1):
        content, termination, pending_observation = await request_action(
            client=client,
            history=history,
            observation=pending_observation,
            final_turn=turn == max_search_turns,
        )
        if termination is not None:
            termination_reason = termination
            break
        if pending_observation is not None:
            continue
        documents = await retriever.retrieve(content)
        pending_observation = format_information(documents, max_observation_chars)

    score = compute_score(
        history,
        prompt_message_count,
        aliases,
        structure_weight=structure_weight,
        final_weight=final_weight,
    )
    metadata = main_metadata(
        messages=history,
        prompt_message_count=prompt_message_count,
        score=score,
        termination_reason=termination_reason,
    )
    reward = score.outcome_em if rollout_mode == "eval" else score.shaped_reward
    if termination_reason == "context_length_exceeded" and len(history) > prompt_message_count:
        return [{"name": "main", "messages": history, "metadata": metadata, "reward": reward}]
    return {"metadata": metadata, "reward": reward}
