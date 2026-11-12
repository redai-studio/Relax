# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Session accounting must compose with existing CLI mode contracts."""

import argparse
from types import SimpleNamespace

import pytest

from tests.cpu.algorithms.test_arguments_spec_driven import arguments_module as _arguments_module


@pytest.fixture
def arguments_module(monkeypatch):
    return _arguments_module.__wrapped__(monkeypatch)


@pytest.mark.parametrize(
    "mode_flags",
    [
        ["--partial-rollout"],
        ["--agentic-prelaunch"],
        ["--partial-rollout", "--agentic-prelaunch"],
        ["--partial-rollout", "--use-dynamic-global-batch-size"],
        ["--partial-rollout", "--agentic-prelaunch", "--use-dynamic-global-batch-size"],
        ["--fully-async", "--agentic-prelaunch"],
    ],
)
def test_capacity_accepts_composable_modes_without_changing_them(arguments_module, tmp_path, mode_flags):
    arguments_module.RouterArgs = SimpleNamespace(add_cli_args=lambda parser, **_kwargs: parser)
    parser = arguments_module.get_slime_extra_args_provider()(argparse.ArgumentParser())
    args = parser.parse_args(
        [
            "--use-agentic-rollout",
            "--agent-command",
            "python agent.py",
            "--agent-cwd",
            str(tmp_path),
            "--agentic-session-capacity-per-node",
            "4",
            "--agentic-concurrency",
            "3",
            "--n-samples-per-prompt",
            "2",
            *mode_flags,
        ]
    )

    arguments_module._validate_agentic_rollout_args(args)

    assert args.agentic_session_capacity_per_node == 4
    assert args.agentic_concurrency == 3
    for flag in mode_flags:
        assert getattr(args, flag.removeprefix("--").replace("-", "_")) is True
    assert args.rollout_function_path == "relax.agentic.rollout.generate_rollout"
    assert args.eval_function_path == args.rollout_function_path


def test_capacity_still_requires_room_for_a_whole_group(arguments_module, tmp_path):
    arguments_module.RouterArgs = SimpleNamespace(add_cli_args=lambda parser, **_kwargs: parser)
    parser = arguments_module.get_slime_extra_args_provider()(argparse.ArgumentParser())
    args = parser.parse_args(
        [
            "--use-agentic-rollout",
            "--agent-command",
            "python agent.py",
            "--agent-cwd",
            str(tmp_path),
            "--agentic-session-capacity-per-node",
            "1",
            "--n-samples-per-prompt",
            "2",
            "--partial-rollout",
            "--agentic-prelaunch",
        ]
    )
    with pytest.raises(ValueError, match="fit one complete Group"):
        arguments_module._validate_agentic_rollout_args(args)
