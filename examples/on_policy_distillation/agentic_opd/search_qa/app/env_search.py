# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""In-process Search-QA environment (one instance per rollout session).

State lives in the agent process; the "world" is a stateless retrieval service
reached over HTTP. The episode is a ``max_turns``-step loop where each model
reply either issues a ``<search>`` (retrieve, continue) or an ``<answer>``
(terminal). Reward is a binary exact-match over the transcript at episode end.
Ported from SDAR ``.../envs/search/env.py``.
"""

from __future__ import annotations

import logging
from typing import Any

from app.prompt import (
    build_prompt,
    compute_score,
    extract_action,
    is_answer_action,
    parse_search_query,
)
from app.retrieval_client import RetrievalClient


logger = logging.getLogger(__name__)


def _as_gold_list(ground_truth: Any) -> list[str]:
    """Normalize the stored ground truth into a flat list of gold strings.

    ``prepare_data.py`` writes a list; tolerate a bare string or SDAR's
    ``{"target": [...]}`` dict shape rather than crash.
    """
    if isinstance(ground_truth, dict) and "target" in ground_truth:
        ground_truth = ground_truth["target"]
    if isinstance(ground_truth, str):
        return [ground_truth]
    if ground_truth is None:
        return []
    return [str(g) for g in ground_truth]


class SearchEnv:
    """Thin, single-episode driver around the Search-QA state machine."""

    def __init__(
        self,
        *,
        question: str,
        ground_truth: Any,
        max_turns: int,
        history_length: int,
        retrieval_url: str,
        topk: int = 3,
    ) -> None:
        self.question = question
        self.golds = _as_gold_list(ground_truth)
        self.max_turns = max_turns
        self.history_length = history_length
        self.client = RetrievalClient(retrieval_url, topk=topk)

        self.turn = 0
        # (projected action, information block) pairs, oldest first.
        self.history: list[tuple[str, str]] = []
        # Full transcript (assistant actions + observations) for terminal EM.
        self._transcript = ""

    def reset(self) -> str:
        self.turn = 0
        self.history = []
        self._transcript = ""
        return build_prompt(
            task=self.question,
            history=self.history,
            history_length=self.history_length,
            init=True,
        )

    def step(self, response_text: str) -> tuple[str | None, bool, dict[str, Any]]:
        action, is_valid = extract_action(response_text)
        self._transcript += response_text
        self.turn += 1

        answered = is_answer_action(action)
        reached_max = self.turn >= self.max_turns
        done = answered or reached_max

        info: dict[str, Any] = {
            "won": False,
            "action": action,
            "is_action_valid": is_valid,
            "env_done": answered,
            "score": 0.0,
            "is_search": False,
            "retrieval_error": False,
        }

        if done:
            score = compute_score(self._transcript, self.golds)
            info["score"] = score
            info["won"] = score >= 1.0
            return None, True, info

        # Not terminal: a valid turn should carry a <search> query. Anything
        # else (invalid / bare answer-less text) yields an empty observation.
        information = ""
        query = parse_search_query(action)
        if query is not None:
            info["is_search"] = True
            try:
                docs = self.client.search(query)
                if docs:
                    information = f"\n<information>{docs}</information>\n"
            except RuntimeError as exc:
                # Retrieval service unreachable after retries: waste this turn
                # (empty obs, flag it) rather than failing the whole episode.
                info["retrieval_error"] = True
                logger.warning("retrieval failed for query %r: %s", query, exc)

        self.history.append((action, information))
        self._transcript += information
        next_prompt = build_prompt(
            task=self.question,
            history=self.history,
            history_length=self.history_length,
            init=False,
        )
        return next_prompt, False, info
