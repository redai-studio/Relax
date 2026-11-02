# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from relax.distributed.ray import training_phase
from relax.engine.inference.ownership import ActivationLease
from relax.engine.inference.phase_client import current_phase


@pytest.fixture
def actor(monkeypatch):
    events = []
    leases = []

    def acquire(owner, operation, claims, *, role=None):
        lease = ActivationLease("epoch", owner, operation, len(leases) + 1, tuple(claims), role)
        leases.append(lease)
        events.append("acquire")
        return lease

    authority = SimpleNamespace(
        claims_for=SimpleNamespace(remote=lambda role: ("shared-bundles",)),
        acquire=SimpleNamespace(remote=acquire),
        command=SimpleNamespace(remote=lambda *args: events.append("train handoff")),
        release=SimpleNamespace(remote=lambda *args: events.append("release")),
        block=SimpleNamespace(remote=lambda *args: events.append("blocked")),
    )
    actor = SimpleNamespace(
        args=SimpleNamespace(_inference_authority=authority),
        role="actor",
        _inference_train_resident=False,
    )
    monkeypatch.setattr(training_phase.ray, "get", lambda value: value)
    group = object()
    monkeypatch.setattr(training_phase, "get_gloo_group", lambda: group)
    monkeypatch.setattr(training_phase.dist, "get_rank", lambda g: 0 if g is group else pytest.fail("missing group"))
    monkeypatch.setattr(training_phase.dist, "get_world_size", lambda g: 2)
    monkeypatch.setattr(training_phase.dist, "broadcast_object_list", lambda *args, **kwargs: None)

    def receipts(output, resident, group):
        events.append("all-rank receipt")
        output[:] = [resident, getattr(actor, "peer_resident", resident)]

    monkeypatch.setattr(training_phase.dist, "all_gather_object", receipts)
    return actor, events, leases


def test_training_handoff_precedes_work_and_waits_for_every_rank(actor):
    instance, events, _ = actor
    instance.peer_resident = True

    @training_phase.leased_training
    def train(actor):
        assert current_phase() is actor._training_session
        events.append("train")

    train(instance)
    assert events == ["acquire", "train handoff", "train", "all-rank receipt"]
    assert not instance._training_session.closed
    instance.peer_resident = False
    train(instance)
    assert events[-3:] == ["train", "all-rank receipt", "release"]
    assert instance._training_session.closed


def test_sft_predict_resume_rebinds_current_phase_to_the_new_generation(actor):
    instance, events, leases = actor

    @training_phase.leased_training
    def train(actor):
        old = current_phase()
        training_phase.release_training_phase(actor)
        assert old.closed
        training_phase.resume_training_phase(actor)
        assert current_phase() is actor._training_session
        assert current_phase().lease.generation > old.lease.generation

    train(instance)
    assert len(leases) == 2
    assert events.count("release") == 2
    assert training_phase._CURRENT.get() is None


def test_training_failure_quarantines_without_waiting_on_a_failed_peer(actor, monkeypatch):
    instance, events, _ = actor
    collective = Mock(side_effect=AssertionError("must not wait for a failed peer"))
    monkeypatch.setattr(training_phase.dist, "all_gather_object", collective)

    @training_phase.leased_training
    def train(actor):
        raise RuntimeError("NCCL peer lost")

    with pytest.raises(RuntimeError, match="NCCL peer lost"):
        train(instance)
    assert events[-1] == "blocked"
    assert "release" not in events
    assert not instance._training_session.closed
