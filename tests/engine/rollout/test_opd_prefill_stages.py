# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""``OpdManager.prefill`` as three stages -- teacher, student, assemble -- that
generation runs back to back and deferred scoring runs one at a time."""

from __future__ import annotations

from argparse import Namespace
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from relax.engine.rollout import on_policy_distillation as opd
from relax.utils.types import Sample


@pytest.fixture
def manager(monkeypatch):
    """An OPD manager whose requests are recorded instead of sent."""
    calls: list[str] = []
    sessions: list[object] = []

    @asynccontextmanager
    async def fake_session(args):
        session = object()
        sessions.append(session)
        yield session

    async def teacher_prefill(sample, session) -> bool:
        calls.append(f"teacher {sample.index}")
        return True

    async def student_prefill(sample, session, encode_multimodal_inputs) -> None:
        calls.append(f"student {sample.index}")

    monkeypatch.setattr(opd, "_create_teacher_client_session", fake_session)
    manager = object.__new__(opd.OpdManager)
    manager.args = Namespace(opd_teacher_url="http://teacher:1/generate")
    manager.opsd_worker = None
    manager.sampled_worker = None
    manager.topk_worker = SimpleNamespace(spec=SimpleNamespace(student_at_teacher=True))
    manager._teacher_prefill = teacher_prefill
    manager._student_prefill = student_prefill
    manager._assemble_transfer = lambda samples: calls.append(f"assemble {[sample.index for sample in samples]}")
    return SimpleNamespace(manager=manager, calls=calls, sessions=sessions)


def _samples() -> list[Sample]:
    return [Sample(index=0, response_length=4), Sample(index=1, response_length=4)]


async def test_opd_prefill_inline_runs_stages_in_original_order(manager):
    await manager.manager.prefill(_samples())

    # Every teacher request, then every student request, then one assembly --
    # over a single HTTP session, as before the split.
    assert manager.calls == ["teacher 0", "teacher 1", "student 0", "student 1", "assemble [0, 1]"]
    assert len(manager.sessions) == 1


async def test_opd_prefill_accepts_a_single_sample(manager):
    await manager.manager.prefill(Sample(index=5, response_length=4))

    assert manager.calls == ["teacher 5", "student 5", "assemble [5]"]


async def test_opd_prefill_teacher_stage_does_not_contact_student(manager):
    await manager.manager.teacher_stage(_samples())

    assert manager.calls == ["teacher 0", "teacher 1"]


async def test_opd_prefill_student_stage_only_runs_when_the_token_selection_needs_it(manager):
    assert manager.manager.needs_student_stage is True
    await manager.manager.student_stage(_samples())
    assert manager.calls == ["student 0", "student 1"]

    manager.calls.clear()
    manager.manager.topk_worker.spec.student_at_teacher = False
    assert manager.manager.needs_student_stage is False
    await manager.manager.student_stage(_samples())
    assert manager.calls == []

    # Sampled-token distillation has no top-k worker at all.
    manager.manager.topk_worker = None
    assert manager.manager.needs_student_stage is False


async def test_opd_prefill_teacher_stage_raises_when_every_request_failed(manager):
    async def failing_teacher_prefill(sample, session) -> bool:
        return False

    manager.manager._teacher_prefill = failing_teacher_prefill

    with pytest.raises(RuntimeError, match="All OPD teacher fetches failed"):
        await manager.manager.prefill(_samples())

    # Nothing after the teacher stage ran.
    assert manager.calls == []
