# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Unit tests for the training-loop context attached to straggler evidence."""

import json

import pytest

import relax.utils.straggler as straggler_package
from relax.utils.straggler import context


@pytest.fixture(autouse=True)
def _reset_context():
    context.reset_training_context_for_tests()
    yield
    context.reset_training_context_for_tests()


def test_training_context_defaults_to_none() -> None:
    assert context.get_training_context() is None
    assert context.snapshot() == {}


def test_set_and_get_round_trip() -> None:
    context.set_training_context(3, 5)

    stored = context.get_training_context()

    assert stored is not None
    assert stored.rollout_id == 3
    assert stored.optimizer_step == 5
    assert stored.sample_seq is None
    # The step index is in-rollout only; no run-wide step is fabricated.
    assert stored.global_step is None
    assert stored.step_ordinal == 1
    assert stored.updated_at > 0.0


def test_a_rollout_length_never_synthesises_a_global_step() -> None:
    """The platform arithmetic is not a run-wide identity and is not used.

    ``rollout_id * num_steps_per_rollout + optimizer_step`` is inherited from
    the platform's ``accumulated_step_id``, which is documented as non-monotonic
    under dynamic batching. The profiler must not reproduce it.
    """
    context.set_training_context(2, 3, num_steps_per_rollout=4)

    stored = context.get_training_context()

    assert stored is not None
    assert stored.global_step is None
    assert stored.optimizer_step == 3


def test_the_varying_rollout_length_case_is_strictly_increasing_and_collision_free() -> None:
    """A four-step rollout then a one-step rollout must not go backwards.

    The platform scalar for these coordinates would be ``[0, 1, 2, 3, 1]``:
    ``(0, 3)`` and ``(1, 0)`` do not increase. The profiler assigns its own
    ``step_ordinal`` per ``(rollout_id, optimizer_step)``, so the sequence is
    strictly increasing and collision-free.
    """
    observed = {}
    for step in range(4):
        context.set_training_context(0, step, num_steps_per_rollout=4)
        observed[(0, step)] = context.get_training_context().step_ordinal
    context.set_training_context(1, 0, num_steps_per_rollout=1)
    observed[(1, 0)] = context.get_training_context().step_ordinal

    values = list(observed.values())
    assert values == [1, 2, 3, 4, 5]
    assert values == sorted(values)
    assert len(set(values)) == len(values), "the ordinals must be collision-free"

    # The arithmetic this replaces is retained only as a documented counterexample.
    platform_arithmetic = [0 * 4 + step for step in range(4)] + [1 * 1 + 0]
    assert platform_arithmetic == [0, 1, 2, 3, 1]
    assert platform_arithmetic != sorted(platform_arithmetic)


def test_sample_seq_is_recorded_when_given() -> None:
    context.set_training_context(1, 0, sample_seq=7)

    stored = context.get_training_context()

    assert stored is not None
    assert stored.sample_seq == 7


def test_record_optimizer_step_is_a_noop_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(straggler_package, "is_straggler_profiler_enabled", lambda: False)
    before = context.training_context_stats()["updates"]

    context.record_optimizer_step(4, 9)

    assert context.get_training_context() is None
    assert context.training_context_stats()["updates"] == before


def test_record_optimizer_step_sets_the_context_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(straggler_package, "is_straggler_profiler_enabled", lambda: True)

    context.record_optimizer_step(4, 9)

    stored = context.get_training_context()
    assert stored is not None
    assert (stored.rollout_id, stored.optimizer_step) == (4, 9)


def test_record_optimizer_step_never_derives_a_global_step(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(straggler_package, "is_straggler_profiler_enabled", lambda: True)

    context.record_optimizer_step(4, 9, num_steps_per_rollout=12)

    stored = context.get_training_context()
    assert stored is not None
    assert stored.global_step is None
    assert stored.optimizer_step == 9


def test_record_optimizer_step_without_a_length_is_in_rollout_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """The no-length case must not emit the bare step index as if it were global.

    The field is typed ``Optional[int]`` and documented as in-rollout only, and
    the snapshot omits ``global_step`` entirely when no run-wide value exists.
    """
    from typing import Optional

    monkeypatch.setattr(straggler_package, "is_straggler_profiler_enabled", lambda: True)

    context.record_optimizer_step(4, 9)

    stored = context.get_training_context()
    assert stored is not None
    assert stored.global_step is None, "the bare optimizer step must not be presented as global"
    assert stored.optimizer_step == 9
    assert "global_step" not in context.snapshot()

    # Typed and documented: the annotation is Optional[int] and the hook says
    # the step index is in-rollout only.
    assert context.TrainingContext.__annotations__["global_step"] == Optional[int]
    assert context.TrainingContext.__annotations__["optimizer_step"] is int
    assert "in-rollout" in (context.record_optimizer_step.__doc__ or "")


def test_record_optimizer_step_never_raises_when_the_switch_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom() -> bool:
        raise RuntimeError("switch exploded")

    monkeypatch.setattr(straggler_package, "is_straggler_profiler_enabled", boom)
    before = context.training_context_stats()["failures"]

    context.record_optimizer_step(1, 2)

    assert context.get_training_context() is None
    assert context.training_context_stats()["failures"] == before + 1


def test_a_raising_builder_does_not_escape_set(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_args, **_kwargs):
        raise RuntimeError("builder exploded")

    monkeypatch.setattr(context, "_build_context", boom)
    before = context.training_context_stats()["failures"]

    context.set_training_context(1, 2)

    assert context.get_training_context() is None
    assert context.training_context_stats()["failures"] == before + 1


def test_snapshot_is_json_serialisable() -> None:
    context.set_training_context(6, 1, sample_seq=2, num_steps_per_rollout=3)

    payload = json.loads(json.dumps(context.snapshot()))

    assert payload == {"rollout_id": 6, "optimizer_step": 1, "sample_seq": 2, "step_ordinal": 1}


def test_many_updates_do_not_grow_the_module_state() -> None:
    for index in range(10_000):
        context.set_training_context(index, index)

    stats = context.training_context_stats()

    assert stats["updates"] == 10_000
    assert stats["failures"] == 0
    assert stats["stored"] == 1
    assert stats["step_ordinal"] == 10_000


def test_the_producer_does_no_io_and_no_socket_on_the_training_thread(monkeypatch: pytest.MonkeyPatch) -> None:
    """Invariant A, demonstrated: publication adds no file or socket work.

    ``publish_step_workload`` runs on the training thread, so the whole path is
    exercised here with ``open`` and ``socket.socket`` spied; neither may fire.
    """
    import builtins
    import socket

    events: list = []
    real_open = builtins.open

    def spy_open(*args, **kwargs):
        events.append("open")
        return real_open(*args, **kwargs)

    monkeypatch.setattr(builtins, "open", spy_open)
    monkeypatch.setattr(socket, "socket", lambda *args, **kwargs: events.append("socket"))

    context.publish_step_workload(42, [(10, 2, 1)])
    context.set_training_context(42, 0, num_steps_per_rollout=1)

    assert context.snapshot()["tokens"] == 10
    assert context.snapshot()["sequences"] == 2
    assert events == [], f"the producer touched I/O on the training thread: {events}"


def test_the_producer_source_has_no_device_sync_or_collective() -> None:
    """Invariant A, source-level: no sync primitive can hide in this module.

    ``context.py`` is the only training-thread file the producer adds, so a
    token scan over its source is a complete check for the forbidden calls.
    """
    import pathlib

    source = pathlib.Path(context.__file__).read_text(encoding="utf-8")
    for token in ("all_reduce", "all_gather", "broadcast(", "synchronize", ".item()", ".cpu()", "torch.", "dist."):
        assert token not in source, f"{token} must not appear in the training-thread producer path"


def test_publication_is_bounded_and_evicts_old_rollouts() -> None:
    """A prefetch that runs far ahead cannot grow memory here."""
    for rollout_id in range(context.MAX_ROLLOUT_WORKLOADS + 5):
        context.publish_step_workload(rollout_id, [(rollout_id, 1, 1)])

    stats = context.training_context_stats()
    assert stats["workload_rollouts"] == context.MAX_ROLLOUT_WORKLOADS
    assert stats["workload_evictions"] == 5
