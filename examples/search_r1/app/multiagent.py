# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Multi-agent Search-R1 main and searcher loops."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from .protocol import request_action
from .retriever import format_information
from .scoring import compute_score, main_metadata, search_count


@dataclass
class SearcherRun:
    messages: list[dict[str, Any]]
    finding: str | None
    termination_reason: str


async def _run_searcher(
    *,
    need: str,
    hop_index: int,
    client: Any,
    retriever: Any,
    max_observation_chars: int,
    max_search_turns: int,
) -> SearcherRun:
    history = [
        {
            "role": "system",
            "content": (
                "You are a Search-R1 searcher. Resolve one delegated fact. Emit <search>query</search> when "
                "retrieval is needed. After receiving <information>, emit one concise <answer>finding</answer>."
            ),
        },
        {"role": "user", "content": f"Delegated hop {hop_index}: {need}"},
    ]
    finding: str | None = None
    termination_reason = "search_budget_exhausted"
    pending_observation: str | None = None

    for turn in range(max_search_turns + 1):
        content, termination, pending_observation = await request_action(
            client=client,
            history=history,
            observation=pending_observation,
            final_turn=turn == max_search_turns,
        )
        if termination is not None:
            if termination == "answer":
                finding = content
            termination_reason = termination
            break
        if pending_observation is not None:
            continue
        documents = await retriever.retrieve(content)
        pending_observation = format_information(documents, max_observation_chars)

    return SearcherRun(
        messages=history,
        finding=finding,
        termination_reason=termination_reason,
    )


async def run_multiagent_session(
    *,
    messages: list[dict[str, Any]],
    aliases: list[str],
    client: Any,
    retriever: Any,
    max_observation_chars: int,
    max_search_turns: int,
    searcher_max_search_turns: int,
    structure_weight: float,
    final_weight: float,
    rollout_mode: str,
) -> list[dict[str, Any]]:
    prompt_message_count = len(messages)
    main_history = list(messages)
    searchers: list[SearcherRun] = []
    pending_observation: str | None = None
    termination_reason = "search_budget_exhausted"

    for hop_index in range(max_search_turns + 1):
        content, termination, pending_observation = await request_action(
            client=client,
            history=main_history,
            observation=pending_observation,
            final_turn=hop_index == max_search_turns,
        )
        if termination is not None:
            termination_reason = termination
            break
        if pending_observation is not None:
            continue
        documents, searcher = await asyncio.gather(
            retriever.retrieve(content),
            _run_searcher(
                need=content,
                hop_index=hop_index,
                client=client,
                retriever=retriever,
                max_observation_chars=max_observation_chars,
                max_search_turns=searcher_max_search_turns,
            ),
        )
        searchers.append(searcher)
        observation = format_information(documents, max_observation_chars)
        if searcher.finding is not None:
            observation += f"\n\n<information>Searcher finding: {searcher.finding}</information>\n\n"
        pending_observation = observation

    score = compute_score(
        main_history,
        prompt_message_count,
        aliases,
        structure_weight=structure_weight,
        final_weight=final_weight,
    )
    records: list[dict[str, Any]] = [
        {
            "name": "main",
            "messages": main_history,
            "metadata": main_metadata(
                messages=main_history,
                prompt_message_count=prompt_message_count,
                score=score,
                termination_reason=termination_reason,
            ),
            "reward": score.outcome_em if rollout_mode == "eval" else score.shaped_reward,
        }
    ]
    if rollout_mode == "train":
        records.extend(
            {
                "name": f"searcher_{searcher_index}",
                "messages": searcher.messages,
                "metadata": {
                    "role": "searcher",
                    "searcher/action/search_count": search_count(searcher.messages, 2),
                    "searcher/termination/reason": searcher.termination_reason,
                },
            }
            for searcher_index, searcher in enumerate(searchers)
        )
    return records
