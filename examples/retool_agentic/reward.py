# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Preserve ReTool's episode status and reward after Agentic export."""

from typing import Any

from examples.retool.reward import reward_func as retool_reward
from relax.agentic.profile import TRACE_KEY
from relax.utils.types import Sample


async def reward_func(args: Any, sample: Sample, **kwargs: Any) -> dict:
    # A native stop at </code> ends an API response, not necessarily the
    # episode. Agent-side turn/observation budgets can also end an episode.
    if sample.metadata["retool_stop_reason"] != "final_answer":
        sample.status = Sample.Status.TRUNCATED
        sample.metadata[TRACE_KEY]["terminal_status"] = sample.status.value
    if "policy_outer" in sample.metadata:
        expected = sample.metadata["policy_outer"] + 1
        if not sample.weight_versions or any(
            not str(version).isdigit() or int(version) + (args.start_rollout_id or 0) != expected
            for version in sample.weight_versions
        ):
            raise RuntimeError("Rollout policy version differs from the current training batch")
    return await retool_reward(args, sample, **kwargs)
