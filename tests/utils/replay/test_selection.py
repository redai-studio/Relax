# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Selection replay tests: single sample / group / micro-batch and closure.

A partial selection expands to its semantic-group closure and replays the per-
sample/per-token stages; cohort-level stages (loss.policy) are skipped rather
than comparing a subset scalar against a full-cohort expected value.
"""

from __future__ import annotations

import pytest

from relax.utils.replay.bundle import BundleReader
from relax.utils.replay.identity import ClosureError, expand_selection
from relax.utils.replay.report import StageStatus
from relax.utils.replay.runner import replay
from relax.utils.replay.selection import select_bundle
from tests.utils.replay.helpers import build_grpo_bundle


def _stage(report, name: str):
    return next(stage for stage in report.stages if stage.stage == name)


def test_select_single_sample_closure(tmp_path):
    bundle, index, _ = build_grpo_bundle(tmp_path / "b")
    closure = expand_selection(index, sample_ids=["s-0"])
    assert set(closure.sample_ids) == {"s-0", "s-1"}
    assert closure.group_ids == {"g-0"}


def test_select_batch_closure(tmp_path):
    bundle, index, _ = build_grpo_bundle(tmp_path / "b")
    closure = expand_selection(index, batch_ids=["mb-0008"])
    assert set(closure.sample_ids) == {"s-2", "s-3"}
    assert closure.group_ids == {"g-1"}


def test_replay_single_sample_passes_and_skips_loss(tmp_path):
    bundle, _, _ = build_grpo_bundle(tmp_path / "b")
    report = replay(bundle, sample_ids=["s-0"])
    assert report.passed is True
    assert _stage(report, "advantage.estimate").status == StageStatus.PASS
    assert _stage(report, "loss.policy").status == StageStatus.SKIPPED


def test_replay_single_batch_passes(tmp_path):
    bundle, _, _ = build_grpo_bundle(tmp_path / "b")
    report = replay(bundle, batch_ids=["mb-0008"])
    assert report.passed is True
    assert _stage(report, "reward.post_process").status == StageStatus.PASS


def test_replay_full_selection_still_replays_loss(tmp_path):
    bundle, _, _ = build_grpo_bundle(tmp_path / "b")
    report = replay(bundle, group_ids=["g-0", "g-1"])
    assert report.passed is True
    assert _stage(report, "loss.policy").status == StageStatus.PASS


def test_select_missing_sample_fails_closed(tmp_path):
    bundle, index, _ = build_grpo_bundle(tmp_path / "b")
    with pytest.raises(ClosureError):
        expand_selection(index, sample_ids=["s-99"])


def test_select_missing_batch_fails_closed(tmp_path):
    bundle, index, _ = build_grpo_bundle(tmp_path / "b")
    with pytest.raises(ClosureError):
        expand_selection(index, batch_ids=["mb-9999"])


def test_select_missing_group_fails_closed(tmp_path):
    bundle, index, _ = build_grpo_bundle(tmp_path / "b")
    index.samples[0].group_index = None
    with pytest.raises(ClosureError):
        expand_selection(index, sample_ids=["s-0"])


def test_select_bundle_slices_tensors_and_expected(tmp_path):
    bundle, _, _ = build_grpo_bundle(tmp_path / "b")
    loaded = BundleReader(bundle).load()
    subset = select_bundle(loaded, sample_ids=["s-0"])
    assert len(subset.index.samples) == 2
    assert subset.tensors["old_log_probs"].numel() == 4
    assert len(subset.expected["reward.raw"]["raw_rewards"]) == 2
