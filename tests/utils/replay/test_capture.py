# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Tests for the PR B production-capture data plane.

These exercise CaptureRecord / build_bundle_from_record (the pure
capture→bundle bridge) and CaptureManager (the async, bounded, failure-
isolated sink) entirely on CPU — the round-trip parity test proves the captured
bundle is replayable by the PR A runner.
"""

from __future__ import annotations

import json
import threading

import pytest
import torch

from relax.utils.replay import capture, capture_hooks
from relax.utils.replay.bundle import BundleReader, IncompleteBundleError
from relax.utils.replay.capture import (
    CaptureConfig,
    CaptureManager,
    CaptureRecord,
    active,
    begin_rollout,
    begin_step,
    build_bundle_from_record,
    build_manifest_for_record,
    config_from_env,
    current_step,
    disable,
    enable,
    end_rollout,
    end_step,
    maybe_enable_from_env,
    should_capture,
    should_capture_rollout,
)
from relax.utils.replay.runner import replay
from relax.utils.replay.schema import (
    ActorStepId,
    Identity,
    RecomputeConfig,
    StageCapability,
    StageId,
)
from tests.utils.replay.helpers import make_capture_record, make_rollout_capture_record


@pytest.fixture(autouse=True)
def _reset_capture_global():
    yield
    disable()


def test_capture_snapshot_isolates_storage():
    from relax.utils.replay.capture import _snapshot_record

    record = make_capture_record("b-clone")
    snapshot = _snapshot_record(record)
    record.tensors["log_probs"].fill_(99.0)
    assert torch.equal(snapshot.tensors["log_probs"], torch.zeros_like(record.tensors["log_probs"]))


def test_capture_roundtrip_parity(tmp_path):
    record = make_capture_record("b-capture")
    bundle_path = build_bundle_from_record(record, tmp_path)

    # The manifest must be derived from the frozen V1 capability matrix, not
    # hard-coded by the capture layer.
    manifest = BundleReader(bundle_path).load().manifest
    for stage in (
        StageId.SAMPLE,
        StageId.REWARD_RAW,
        StageId.REWARD_POST_PROCESS,
        StageId.ADVANTAGE_KL,
        StageId.ADVANTAGE_ESTIMATE,
        StageId.LOSS_POLICY,
    ):
        assert manifest.stage_contracts[stage].capability == StageCapability.RECOMPUTE
    assert manifest.stage_contracts[StageId.LOSS_VALUE].capability == StageCapability.UNSUPPORTED

    report = replay(bundle_path)
    assert report.passed
    assert report.first_divergent_stage is None


def test_capture_manifest_reimplemented_marker():
    manifest = build_manifest_for_record(make_capture_record("b-marker"))
    assert manifest.stage_contracts[StageId.REWARD_POST_PROCESS].implementation == "reimplemented"
    assert manifest.stage_contracts[StageId.ADVANTAGE_ESTIMATE].implementation == "reuse"


def test_capture_manager_disabled_noop(tmp_path):
    manager = CaptureManager(CaptureConfig(enabled=False, output_dir=str(tmp_path)))
    assert manager.submit(make_capture_record("b-disabled")) is False
    assert manager.dropped_count == 0
    assert not (tmp_path / "b-disabled").exists()


def test_capture_manager_enabled_writes_bundle(tmp_path):
    manager = CaptureManager(CaptureConfig(enabled=True, output_dir=str(tmp_path)))
    assert manager.submit(make_capture_record("b-async")) is True
    manager.flush()
    manager.close()
    assert (tmp_path / "b-async" / "COMPLETE").exists()
    assert replay(tmp_path / "b-async").passed


def test_capture_manager_queue_overflow_drops(tmp_path, monkeypatch):
    entered = threading.Event()
    release = threading.Event()

    def slow_build(record, output_dir):
        entered.set()
        release.wait(timeout=10)
        return build_bundle_from_record(record, output_dir)

    monkeypatch.setattr(capture, "build_bundle_from_record", slow_build)
    manager = CaptureManager(CaptureConfig(enabled=True, output_dir=str(tmp_path), queue_capacity=1))

    assert manager.submit(make_capture_record("b-1")) is True
    assert entered.wait(timeout=5)
    # Writer is blocked inside build, so the queue is empty. Fill it, then
    # overflow it: the third submit must be dropped, never back-pressure.
    assert manager.submit(make_capture_record("b-2")) is True
    assert manager.submit(make_capture_record("b-3")) is False
    assert manager.dropped_count == 1

    release.set()
    manager.close()


def test_capture_failure_isolation(tmp_path, monkeypatch):
    def failing_build(record, output_dir):
        raise RuntimeError("boom")

    monkeypatch.setattr(capture, "build_bundle_from_record", failing_build)
    manager = CaptureManager(CaptureConfig(enabled=True, output_dir=str(tmp_path)))

    assert manager.submit(make_capture_record("b-fail")) is True  # never raises
    manager.flush()
    assert manager.error_count == 1
    manager.close()


def test_capture_selected_steps_and_rollouts(tmp_path):
    manager = CaptureManager(
        CaptureConfig(
            enabled=True,
            output_dir=str(tmp_path),
            selected_steps={(120, 0)},
            selected_rollouts={120},
        )
    )
    assert manager.submit(make_capture_record("b-selected")) is True
    skipped_step = make_capture_record("b-skipped-step")
    skipped_step.actor_step_id = (121, 0)
    assert manager.submit(skipped_step) is False

    assert manager.submit(make_rollout_capture_record("b-rselected")) is True
    skipped_rollout = make_rollout_capture_record("b-rskipped")
    skipped_rollout.rollout_id = 121
    assert manager.submit(skipped_rollout) is False

    manager.flush()
    manager.close()
    assert (tmp_path / "b-selected" / "COMPLETE").exists()
    assert (tmp_path / "b-rselected" / "COMPLETE").exists()
    assert not (tmp_path / "b-skipped-step").exists()
    assert not (tmp_path / "b-rskipped").exists()


def _step_identity() -> Identity:
    return Identity(actor_step_id=ActorStepId(rollout_id=120, step_id=0), rank={"dp": 0, "tp": 0, "pp": 0, "cp": 1})


def _step_config() -> RecomputeConfig:
    return RecomputeConfig(advantage_estimator="grpo", n_samples_per_prompt=2)


def test_capture_enable_disable_active(tmp_path):
    assert active() is None
    manager = enable(CaptureConfig(enabled=True, output_dir=str(tmp_path)))
    assert active() is manager
    assert manager.enabled
    disable()
    assert active() is None


def test_capture_step_lifecycle(tmp_path):
    enable(CaptureConfig(enabled=True, output_dir=str(tmp_path)))
    record = make_capture_record("b-step")

    begin_step((120, 0), identity=record.identity, config=record.config, bundle_id="b-step")
    step = current_step()
    assert step is not None and step.actor_step_id == (120, 0)
    step.samples = record.samples
    step.tensors = record.tensors
    step.expected = record.expected
    # end_step drops accumulators that never received a hook payload (stages
    # stays None). The production hooks always set this; the test must too.
    step.stages = {
        StageId.SAMPLE,
        StageId.REWARD_RAW,
        StageId.REWARD_POST_PROCESS,
        StageId.ADVANTAGE_KL,
        StageId.ADVANTAGE_ESTIMATE,
        StageId.LOSS_POLICY,
    }
    end_step()
    disable()

    assert (tmp_path / "b-step" / "COMPLETE").exists()
    assert replay(tmp_path / "b-step").passed


def test_group_indices_from_rollout_uses_tq_or_synthesizes():
    class _Args:
        n_samples_per_prompt = 2

    from_tq = capture_hooks._group_indices_from_rollout({"group_index": [3, 3, 9, 9]}, _Args())
    assert list(from_tq) == [3, 3, 9, 9]

    synthesized = capture_hooks._group_indices_from_rollout({"response_lengths": [1, 1, 1, 1]}, _Args())
    assert synthesized == [0, 0, 1, 1]


def test_capture_hooks_noop_when_disabled():
    disable()
    assert should_capture((0, 0)) is False
    assert should_capture_rollout(120) is False
    begin_step((120, 0), identity=_step_identity(), config=_step_config(), bundle_id="b-x")
    assert current_step() is None
    begin_rollout(120, identity=Identity(rollout_id=120, rank={"cp": 1}), config=_step_config(), bundle_id="b-x")
    assert current_step() is None


def test_capture_unselected_or_empty_does_not_write(tmp_path):
    enable(CaptureConfig(enabled=True, output_dir=str(tmp_path), selected_steps={(0, 0)}, selected_rollouts={0}))
    begin_step((120, 0), identity=_step_identity(), config=_step_config(), bundle_id="b-x")
    assert current_step() is None
    begin_rollout(120, identity=Identity(rollout_id=120, rank={"cp": 1}), config=_step_config(), bundle_id="b-x")
    assert current_step() is None

    enable(CaptureConfig(enabled=True, output_dir=str(tmp_path)))
    begin_step((120, 0), identity=_step_identity(), config=_step_config(), bundle_id="b-empty")
    end_step()
    begin_rollout(120, identity=Identity(rollout_id=120, rank={"cp": 1}), config=_step_config(), bundle_id="b-empty-r")
    end_rollout()
    disable()
    assert not (tmp_path / "b-empty").exists()
    assert not (tmp_path / "b-empty-r").exists()


def test_capture_loss_only_roundtrip(tmp_path):
    # A partial (loss-only) capture, as produced by capture_hooks.capture_policy_loss:
    # positional sample ids built from raw metadata, advantages recorded as a
    # tensor (not recomputed upstream), and only loss.policy declared.
    record = make_capture_record("b-loss-only")
    loss_only = CaptureRecord(
        actor_step_id=record.actor_step_id,
        identity=record.identity,
        samples=[],
        config=record.config,
        tensors={name: record.tensors[name] for name in ("old_log_probs", "log_probs", "entropy", "advantages")},
        expected={StageId.LOSS_POLICY.value: record.expected[StageId.LOSS_POLICY.value]},
        bundle_id="b-loss-only",
        stages={StageId.LOSS_POLICY},
        response_lengths=[sample.response_length for sample in record.samples],
        total_lengths=[sample.total_length for sample in record.samples],
        loss_masks_tensor=torch.tensor(
            [value for sample in record.samples for value in sample.loss_mask], dtype=torch.float32
        ),
    )

    bundle_path = build_bundle_from_record(loss_only, tmp_path)
    loaded = BundleReader(bundle_path).load()
    assert set(loaded.manifest.stage_contracts) == {StageId.LOSS_POLICY}
    assert loaded.index.identity.micro_batch_ids == ["mb-0000"]
    assert all(sample.micro_batch_id == "mb-0000" for sample in loaded.index.samples)

    assert replay(bundle_path).passed
    assert replay(bundle_path, batch_ids=["mb-0000"]).passed


def test_capture_rollout_roundtrip_parity(tmp_path):
    record = make_rollout_capture_record("b-rollout")
    bundle_path = build_bundle_from_record(record, tmp_path)

    loaded = BundleReader(bundle_path).load()
    # A rollout-level bundle declares only the reward → advantage chain; the
    # per-step loss stages are absent because their payloads are not captured.
    assert set(loaded.manifest.stage_contracts) == {
        StageId.REWARD_RAW,
        StageId.REWARD_POST_PROCESS,
        StageId.ADVANTAGE_KL,
        StageId.ADVANTAGE_ESTIMATE,
    }
    for contract in loaded.manifest.stage_contracts.values():
        assert contract.capability == StageCapability.RECOMPUTE
    assert loaded.index.identity.rollout_id == 120
    assert loaded.index.identity.actor_step_id is None

    report = replay(bundle_path)
    assert report.passed
    assert report.first_divergent_stage is None


def test_capture_rollout_lifecycle(tmp_path):
    enable(CaptureConfig(enabled=True, output_dir=str(tmp_path)))
    record = make_rollout_capture_record("b-rollout-step")

    begin_rollout(120, identity=record.identity, config=record.config, bundle_id="b-rollout-step")
    step = current_step()
    assert step is not None and step.rollout_id == 120 and step.actor_step_id is None
    step.samples = record.samples
    step.tensors = record.tensors
    step.expected = record.expected
    step.stages = record.stages
    step.response_lengths = record.response_lengths
    step.total_lengths = record.total_lengths
    step.loss_masks_tensor = record.loss_masks_tensor
    step.group_indices_tensor = record.group_indices_tensor
    step.raw_rewards_tensor = record.raw_rewards_tensor
    step.rewards_tensor = record.rewards_tensor
    end_rollout()
    disable()

    assert (tmp_path / "b-rollout-step" / "COMPLETE").exists()
    assert replay(tmp_path / "b-rollout-step").passed


@pytest.mark.parametrize("set_capture", [False, True])
def test_config_from_env_stays_disabled_without_dir(monkeypatch, set_capture):
    if set_capture:
        monkeypatch.setenv("RELAX_REPLAY_CAPTURE", "1")
    else:
        monkeypatch.delenv("RELAX_REPLAY_CAPTURE", raising=False)
    monkeypatch.delenv("RELAX_REPLAY_CAPTURE_DIR", raising=False)
    assert config_from_env() is None


def test_config_from_env_parses_selectors(monkeypatch, tmp_path):
    monkeypatch.setenv("RELAX_REPLAY_CAPTURE", "1")
    monkeypatch.setenv("RELAX_REPLAY_CAPTURE_DIR", str(tmp_path))
    monkeypatch.setenv("RELAX_REPLAY_CAPTURE_STEPS", "0:0, 1:2")
    monkeypatch.setenv("RELAX_REPLAY_CAPTURE_ROLLOUTS", "0,3")
    config = config_from_env()
    assert config is not None
    assert config.enabled
    assert config.output_dir == str(tmp_path)
    assert config.selected_steps == {(0, 0), (1, 2)}
    assert config.selected_rollouts == {0, 3}


def test_config_from_env_rejects_bad_steps(monkeypatch, tmp_path):
    monkeypatch.setenv("RELAX_REPLAY_CAPTURE", "1")
    monkeypatch.setenv("RELAX_REPLAY_CAPTURE_DIR", str(tmp_path))
    monkeypatch.setenv("RELAX_REPLAY_CAPTURE_STEPS", "not-a-step")
    with pytest.raises(ValueError, match="RELAX_REPLAY_CAPTURE_STEPS"):
        config_from_env()


def test_maybe_enable_from_env_records_rank(monkeypatch, tmp_path):
    monkeypatch.setenv("RELAX_REPLAY_CAPTURE", "1")
    monkeypatch.setenv("RELAX_REPLAY_CAPTURE_DIR", str(tmp_path))
    manager = maybe_enable_from_env(rank=3, expected_ranks=[1, 3])
    assert manager is not None
    assert manager.enabled
    assert active() is manager
    assert manager.output_dir == str(tmp_path)
    assert manager.rank == 3
    assert manager.expected_ranks == [1, 3]


def test_maybe_enable_for_actor_only_producers(monkeypatch, tmp_path):
    monkeypatch.setenv("RELAX_REPLAY_CAPTURE", "1")
    monkeypatch.setenv("RELAX_REPLAY_CAPTURE_DIR", str(tmp_path))
    monkeypatch.setattr(capture_hooks, "capture_producer_ranks", lambda: [1, 3])
    monkeypatch.setattr("torch.distributed.get_rank", lambda: 2)
    assert capture_hooks.maybe_enable_for_actor() is None
    disable()
    monkeypatch.setattr("torch.distributed.get_rank", lambda: 3)
    manager = capture_hooks.maybe_enable_for_actor()
    assert manager is not None
    assert manager.rank == 3
    assert manager.expected_ranks == [1, 3]


def _policy_reported_loss(old, logp, entropy, advantages, samples, config):
    from relax.utils.training.ppo_utils import compute_policy_loss

    lengths = [sample.response_length for sample in samples]
    masks = [sample.loss_mask for sample in samples]
    ppo_kl = old - logp
    pg_loss, clipfrac = compute_policy_loss(ppo_kl, advantages, config.eps_clip, config.eps_clip_high)

    def reduce(values: torch.Tensor) -> torch.Tensor:
        total = values.new_zeros(())
        for chunk, mask in zip(torch.split(values, lengths), masks, strict=False):
            mask_t = torch.tensor(mask, dtype=values.dtype)
            total = total + (chunk * mask_t).sum() / torch.clamp_min(mask_t.sum(), 1)
        return total

    pg = reduce(pg_loss)
    entropy_loss = reduce(entropy)
    return {
        "loss": pg - config.entropy_coef * entropy_loss,
        "pg_loss": pg,
        "entropy_loss": entropy_loss,
        "pg_clipfrac": reduce(clipfrac),
        "ppo_kl": reduce(ppo_kl),
    }


def _capture_loss_slice(record, start: int, end: int, micro_batch_index: int) -> None:
    samples = record.samples[start:end]
    token_start = sum(sample.response_length for sample in record.samples[:start])
    token_end = token_start + sum(sample.response_length for sample in samples)
    sl = slice(token_start, token_end)
    capture_hooks.capture_policy_loss(
        old_log_probs=record.tensors["old_log_probs"][sl],
        log_probs=record.tensors["log_probs"][sl],
        entropy=record.tensors["entropy"][sl],
        advantages=record.tensors["advantages"][sl],
        loss_masks=[torch.tensor(sample.loss_mask) for sample in samples],
        response_lengths=[sample.response_length for sample in samples],
        total_lengths=[sample.total_length for sample in samples],
        reported_loss=_policy_reported_loss(
            record.tensors["old_log_probs"][sl],
            record.tensors["log_probs"][sl],
            record.tensors["entropy"][sl],
            record.tensors["advantages"][sl],
            samples,
            record.config,
        ),
        micro_batch_index=micro_batch_index,
    )


def test_capture_policy_loss_records_data_iterator_micro_batches(tmp_path):
    record = make_capture_record("b-mb")
    enable(CaptureConfig(enabled=True, output_dir=str(tmp_path)))
    begin_step((120, 0), identity=record.identity, config=record.config, bundle_id="b-mb")

    _capture_loss_slice(record, 0, 2, micro_batch_index=7)
    _capture_loss_slice(record, 2, 4, micro_batch_index=8)
    # Activation-checkpoint recompute of mb 7 must not double-count.
    _capture_loss_slice(record, 0, 2, micro_batch_index=7)
    end_step()
    disable()

    loaded = BundleReader(tmp_path / "b-mb").load()
    assert loaded.index.identity.micro_batch_ids == ["mb-0007", "mb-0008"]
    assert [sample.micro_batch_id for sample in loaded.index.samples] == ["mb-0007", "mb-0007", "mb-0008", "mb-0008"]
    assert loaded.tensors["old_log_probs"].numel() == 8
    assert replay(tmp_path / "b-mb").passed
    assert replay(tmp_path / "b-mb", batch_ids=["mb-0008"]).passed


def test_capture_cohort_try_finalize(tmp_path):
    record0 = make_capture_record("b-cohort")
    record0.capture_rank = 0
    record0.expected_ranks = [0, 1]
    path0 = build_bundle_from_record(record0, tmp_path)
    cohort = tmp_path / "b-cohort"
    assert path0 == cohort / "rank-0"
    assert (cohort / "COMPLETE.0").is_file()
    assert not (cohort / "COMPLETE").is_file()
    rank0_complete = json.loads((path0 / "COMPLETE").read_text(encoding="utf-8"))
    assert rank0_complete["ranks"] == [0]
    assert rank0_complete["expected_ranks"] == [0, 1]
    with pytest.raises(IncompleteBundleError, match="incomplete"):
        BundleReader(path0).load()

    record1 = make_capture_record("b-cohort")
    record1.capture_rank = 1
    record1.expected_ranks = [0, 1]
    path1 = build_bundle_from_record(record1, tmp_path)
    assert path1 == cohort / "rank-1"
    complete = json.loads((cohort / "COMPLETE").read_text(encoding="utf-8"))
    assert complete["ranks"] == [0, 1]
    assert replay(path0).passed
    assert replay(path1).passed
