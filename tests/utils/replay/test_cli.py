# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""CLI entry point tests.

Happy-path replay, sample/batch selection and PR #65 localization are covered
by the runner tests. This module only checks argv wiring, exit codes and the
--step resolver unique to the CLI.
"""

from __future__ import annotations

from relax.tools.trajectory_replay.cli import main
from relax.utils.replay.capture import build_bundle_from_record
from tests.utils.replay.helpers import build_grpo_bundle, make_capture_record, make_rollout_capture_record


def test_cli_inspect_valid(tmp_path):
    bundle, _, _ = build_grpo_bundle(tmp_path / "bundle")
    assert main(["inspect", str(bundle)]) == 0


def test_cli_inspect_missing(tmp_path):
    assert main(["inspect", str(tmp_path / "nope")]) == 1


def test_cli_validate_valid(tmp_path):
    bundle, _, _ = build_grpo_bundle(tmp_path / "bundle")
    assert main(["validate", str(bundle)]) == 0


def test_cli_validate_invalid(tmp_path):
    bundle, _, _ = build_grpo_bundle(tmp_path / "bundle")
    (bundle / "COMPLETE").unlink()
    assert main(["validate", str(bundle)]) == 1


def test_cli_replay_divergent(tmp_path):
    bundle, _, _ = build_grpo_bundle(tmp_path / "bundle", pr65_bug=True)
    assert main(["replay", str(bundle)]) == 1


def test_cli_replay_step_assert_mismatch(tmp_path):
    bundle, _, _ = build_grpo_bundle(tmp_path / "bundle")
    assert main(["replay", str(bundle), "--step", "999:0"]) == 1


def test_cli_replay_step_on_rollout_bundle(tmp_path):
    bundle = build_bundle_from_record(make_rollout_capture_record("b-rollout"), tmp_path)
    assert main(["replay", str(bundle), "--step", "120:0"]) == 1
    assert main(["replay", str(bundle), "--rollout", "120"]) == 0
    assert main(["replay", str(bundle), "--rollout", "121"]) == 1


def test_cli_replay_step_picks_from_capture_dir(tmp_path):
    build_grpo_bundle(tmp_path / "replay-120-0")
    build_bundle_from_record(make_rollout_capture_record("replay-rollout-120"), tmp_path)
    assert main(["replay", str(tmp_path), "--step", "120:0"]) == 0
    assert main(["replay", str(tmp_path), "--rollout", "120"]) == 0
    assert main(["replay", str(tmp_path), "--step", "999:0"]) == 1


def test_cli_replay_step_and_rollout_exclusive(tmp_path):
    bundle, _, _ = build_grpo_bundle(tmp_path / "bundle")
    assert main(["replay", str(bundle), "--step", "120:0", "--rollout", "120"]) == 1


def test_cli_replay_requested_unsupported_stage(tmp_path):
    bundle, _, _ = build_grpo_bundle(tmp_path / "bundle")
    assert main(["replay", str(bundle), "--stage", "loss.value"]) == 1


def test_cli_replay_step_requires_cohort_complete(tmp_path):
    record0 = make_capture_record("replay-120-0")
    record0.capture_rank = 0
    record0.expected_ranks = [0, 1]
    rank0 = build_bundle_from_record(record0, tmp_path)
    assert main(["replay", str(tmp_path), "--step", "120:0"]) == 1
    assert main(["replay", str(rank0)]) == 1
    assert main(["replay", str(tmp_path)]) == 1
    assert main(["validate", str(rank0)]) == 1
    assert main(["inspect", str(rank0)]) == 0

    record1 = make_capture_record("replay-120-0")
    record1.capture_rank = 1
    record1.expected_ranks = [0, 1]
    rank1 = build_bundle_from_record(record1, tmp_path)
    assert main(["replay", str(tmp_path), "--step", "120:0"]) == 0
    assert main(["replay", str(rank1), "--step", "120:0"]) == 0
    assert main(["replay", str(rank0)]) == 0
    assert main(["validate", str(rank0)]) == 0
