# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Reward function for the WebShop agentic OPD recipe.

The WebShop outcome is produced by the environment during rollout and carried
back through ``sample.metadata`` (written by ``app/agent.py``). WebShop yields a
continuous task-score in ``[0, 1]`` (partial credit for matching attributes /
price) plus a binary ``won`` (perfect completion). We drive the policy gradient
with the continuous ``task_score`` (denser signal, matching SDAR's episode reward
of ``10 * score``) and track ``won`` as the success-rate metric.
"""

from __future__ import annotations

from relax.utils.types import Sample


def compute_score(metadata: dict) -> dict:
    won = float(metadata.get("success", 1.0 if metadata.get("won") else 0.0))
    task_score = float(metadata.get("task_score", won))
    return {
        "score": task_score,
        "acc": won,
        "won": won,
        "task_score": task_score,
    }


async def reward_func(args, sample: Sample, **kwargs):
    if not isinstance(sample, Sample):
        raise TypeError("Sample must be an instance of Sample class.")
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    return compute_score(metadata)
