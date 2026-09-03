# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Lifecycle holders for generative reward scorers (design doc 7.3 / 11.2).

Two managers, one ``score(requests) -> {component: [score, ...]}`` contract,
both with a finally-offload guarantee so scoring never leaves a model resident
on a GPU the diffusion pipeline needs back:

* :class:`LocalGenerativeRewardManager` — in-process scoring. This is what
  ``--reward-runtime cpu`` (CUDA disabled in the scorer) and ``colocate``
  (shares the rollout worker's GPU) both use, and what
  ``examples/diffusion/evaluate.py`` uses offline.
* :class:`GenerativeRewardManager` — HTTP proxy to an external scorer service
  (``--reward-runtime remote``).

**Deleted: the Ray scorer worker pool.** This module used to also own a
placement-group-scheduled pool (``_create_workers`` / ``_RayScorerWorker`` /
``onload`` / ``offload``) for a ``dedicated`` runtime. It was unreachable in
every configuration: ``rewards.generative._get_manager`` is the only caller and
it routed everything except ``remote`` to the in-process manager, while
``remote`` skipped worker creation by construction. Scoring is called from the
rollout worker's post-process hook, which has no controller-created placement
group to schedule a pool into — so a pool would have to be created and owned by
the controller, not resurrected here. The ``--reward-num-gpus`` knob went with
it.
"""

from __future__ import annotations

from typing import Any, Dict, List

from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)

__all__ = ["GenerativeRewardManager", "LocalGenerativeRewardManager"]


def _load_scorer(args):
    from relax.utils.utils import load_function

    scorer_cls = load_function(args.reward_scorer_path)
    # Scorers take the args namespace so they can read model_path / frame
    # selection / sample rate from the reward config.
    return scorer_cls(args)


class LocalGenerativeRewardManager:
    """In-process (non-Ray) scorer holder with finally-offload semantics."""

    def __init__(self, args) -> None:
        self.args = args
        self._scorer = None

    def _ensure(self):
        if self._scorer is None:
            if not getattr(self.args, "reward_scorer_path", None):
                raise ValueError("reward_scorer_path is required to score generative rewards.")
            self._scorer = _load_scorer(self.args)
        return self._scorer

    def score(self, requests: List[Dict[str, Any]]) -> Dict[str, List[float]]:
        scorer = self._ensure()
        scorer.onload()
        try:
            return scorer.score_batch(requests)
        finally:
            scorer.offload()


class GenerativeRewardManager:
    """HTTP proxy to an external scorer service (``--reward-runtime remote``).

    Holds no model and needs no placement group, which is why it is the one
    runtime that works unchanged from inside the rollout worker.
    """

    def __init__(self, args) -> None:
        self.args = args

    @classmethod
    def local(cls, args) -> LocalGenerativeRewardManager:
        """Return the in-process manager (kept here so callers have a single
        import for both scoring modes)."""
        return LocalGenerativeRewardManager(args)

    def score(self, requests: List[Dict[str, Any]]) -> Dict[str, List[float]]:
        import requests as http

        endpoint = getattr(self.args, "reward_endpoint", None)
        if not endpoint:
            raise ValueError("reward_runtime=remote requires reward_endpoint.")
        resp = http.post(endpoint, json={"requests": requests}, timeout=600)
        resp.raise_for_status()
        return resp.json()
