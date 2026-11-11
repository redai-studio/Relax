# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Shared agentic-observability test args.

The rollout sampling defaults live here in exactly one place and are asserted
against the real argument parser by
``tests/cpu/test_agentic_obs_args_binding.py``. Individual tests only override
what differs via ``**overrides`` — no test copies this table a second time.
"""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

# Single source of truth for the rollout sampling params read directly by
# ``relax.agentic.inventory.session_sampling_config``. These mirror the argparse
# defaults in ``relax/utils/arguments.py``; the binding test pins them together.
ROLLOUT_SAMPLING_DEFAULTS: dict = {
    "rollout_temperature": 1.0,
    "rollout_top_p": 1.0,
    "rollout_top_k": -1,
    "rollout_max_response_len": None,
    "rollout_stop": None,
    "rollout_stop_token_ids": None,
    "rollout_skip_special_tokens": False,
}

# ``sglang_enable_deterministic_inference`` is declared by the SGLang ServerArgs
# mirror (--sglang-enable-deterministic-inference, store_true, default False),
# not by the slime parser; it only exists when sglang args were merged
# (not skip_sglang). Tests keep it present so the direct read never raises.
_SGLANG_FLAG_DEFAULT = False


def make_agentic_obs_args(
    tmp_path: Path,
    *,
    run_id: str = "run",
    generation_id: str = "generation",
    dump_media: bool = False,
    **overrides,
) -> Namespace:
    """Build a Namespace for agentic-observability code paths.

    Tests pass only the fields that differ from this baseline via overrides.
    """
    data: dict = {
        "dump_agentic_obs": str(tmp_path),
        "agentic_obs_run_id": run_id,
        "agentic_obs_train_generation_id": generation_id,
        "dump_agentic_obs_media": dump_media,
        "reward_key": None,
        "eval_reward_key": None,
        "sglang_enable_deterministic_inference": _SGLANG_FLAG_DEFAULT,
    }
    data.update(ROLLOUT_SAMPLING_DEFAULTS)
    data.update(overrides)
    return Namespace(**data)
