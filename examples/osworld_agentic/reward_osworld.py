# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""OSWorld agentic reward function.

The heavy lifting — running the VM, grading via OSWorld's postconfig / metrics
— already happened inside the session subprocess (``app/agent.py``), which
writes the ``pass``/``fail`` outcome into ``sample.reward`` and mirrors
per-step context onto ``sample.metadata``.

This function *shapes* that raw outcome into the training signal:

* ``outcome`` : ±1 outcome-only signal (upstream ``outcome_only_rl``).
* ``length_penalty`` : gentle nudge toward terminating early once the task is
  solved (per-turn).
* ``cap_penalty`` : hard bonus for staying under ``max_turns / 2``.
* ``mcp_bonus`` : off by default. Set ``OSWORLD_MCP_BONUS=1`` to enable a
  binary +0.05 for at least one successful MCP call — this diverges from
  upstream but tracks the diagnostic tool_reward that upstream left as dead
  code (see the DeepEyes V2 memory ``cb9b2665``).

Final training reward is a scalar in roughly ``[-1.1, 1.1]``.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from relax.utils.types import Sample


logger = logging.getLogger(__name__)


# ── knobs (defaults match outcome_only_rl) ─────────────────────────────────

DEFAULT_LENGTH_PENALTY = 0.05          # per turn, subtracted from reward
DEFAULT_CAP_PENALTY_THRESHOLD_FRAC = 0.5   # fraction of max_turns to earn the bonus
DEFAULT_CAP_BONUS = 0.2
DEFAULT_MAX_TURNS = 50


def _outcome_from_sample(sample: Sample) -> float:
    """Extract the raw ±1 outcome the session driver already computed."""
    reward = getattr(sample, "reward", None)
    if isinstance(reward, dict):
        raw = reward.get("outcome")
        if raw is None:
            raw = reward.get("score", 0.0)
    elif isinstance(reward, (int, float)):
        raw = float(reward)
    else:
        raw = 0.0
    if raw is None:
        return 0.0
    return 1.0 if float(raw) >= 0.5 else -1.0


def _turn_count(sample: Sample) -> int:
    meta = sample.metadata or {}
    # ``rollout_turns`` is the canonical field written by relax/agentic (see
    # memory 168f2c3b — some readers still get 1 from the agentic path). Fall
    # back to our own ``agent_turns`` written in app/agent.py.
    for key in ("agent_turns", "rollout_turns", "num_turns"):
        val = meta.get(key)
        if isinstance(val, (int, float)) and val > 0:
            return int(val)
    trace = meta.get("agentic_trace") or {}
    tc = trace.get("turn_count") if isinstance(trace, dict) else None
    if isinstance(tc, (int, float)) and tc > 0:
        return int(tc)
    return 1


def _has_successful_mcp_call(sample: Sample) -> bool:
    meta = sample.metadata or {}
    action_types = meta.get("action_types") or []
    if isinstance(action_types, list) and any(t == "mcp" for t in action_types):
        exec_oks = meta.get("action_exec_oks") or []
        if isinstance(exec_oks, list) and exec_oks:
            return any(
                bool(ok) and (t == "mcp")
                for ok, t in zip(exec_oks, action_types)
            )
        # Fallback: presence-only signal
        return True
    return False


def _shape(args: Any, sample: Sample) -> dict[str, float]:
    outcome = _outcome_from_sample(sample)
    turns = _turn_count(sample)

    length_penalty_coeff = float(getattr(args, "osworld_length_penalty",
                                         DEFAULT_LENGTH_PENALTY))
    cap_threshold_frac = float(getattr(args, "osworld_cap_threshold_frac",
                                       DEFAULT_CAP_PENALTY_THRESHOLD_FRAC))
    cap_bonus = float(getattr(args, "osworld_cap_bonus", DEFAULT_CAP_BONUS))
    max_turns = int(getattr(args, "osworld_max_turns", DEFAULT_MAX_TURNS))

    length_penalty = length_penalty_coeff * turns
    cap_penalty = cap_bonus if turns <= (cap_threshold_frac * max_turns) else 0.0

    mcp_bonus = 0.0
    if os.environ.get("OSWORLD_MCP_BONUS") == "1":
        mcp_bonus = 0.05 if _has_successful_mcp_call(sample) else 0.0

    total = outcome - length_penalty + cap_penalty + mcp_bonus
    return {
        "score": float(total),
        "outcome": float(outcome),
        "length_penalty": float(length_penalty),
        "cap_penalty": float(cap_penalty),
        "mcp_bonus": float(mcp_bonus),
        "turns": float(turns),
    }


async def reward_func(args: Any, sample: Sample, **_: Any) -> dict[str, float]:
    """Async entry point registered via ``--custom-rm-path``.

    ``sample.reward`` at call time is whatever the session driver wrote
    (typically ``{"outcome": 0.0/1.0}`` from OSWorld's evaluator). This
    function shapes it into the RL signal.
    """
    if not isinstance(sample, Sample):
        raise TypeError("`sample` must be a relax.utils.types.Sample instance.")
    return _shape(args, sample)
