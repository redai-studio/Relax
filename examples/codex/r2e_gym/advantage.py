# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""One GRPO advantage per Codex Session, shared by all of its contexts."""

from __future__ import annotations

import numpy as np


def session_grpo(sessions: list[dict[str, dict]]) -> list[dict[str, float]]:
    rewards = np.asarray(
        [
            next(context["session_reward"] for context in contexts.values() if "session_reward" in context)
            for contexts in sessions
        ],
        dtype=np.float32,
    )
    advantages = ((rewards - rewards.mean()) / (rewards.std(ddof=1) + 1e-6)).tolist()
    return [{name: advantage for name in contexts} for contexts, advantage in zip(sessions, advantages)]
