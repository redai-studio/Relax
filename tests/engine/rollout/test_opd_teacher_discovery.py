# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Managed OPD uses per-request admission; external teacher URLs stay
usable."""

from argparse import Namespace
from contextlib import asynccontextmanager

import pytest

from relax.engine.inference.discovery import EngineState, RoleSnapshot, build_model_snapshot
from relax.engine.inference.routing import ModelUnavailableError
from relax.engine.rollout import on_policy_distillation as opd
from relax.utils.types import Sample
from tests.engine.inference.helpers import AdmissionManager


class Session:
    def __init__(self):
        self.urls = []
        self.closed = 0

    @asynccontextmanager
    async def post(self, url, **kwargs):
        self.urls.append(url)
        try:
            yield self
        finally:
            self.closed += 1

    status = 200

    async def json(self):
        return {"meta_info": {"input_token_logprobs": [[None, 1, None], [-0.5, 2, None]]}}


def snapshot(*names, host="engine", state=EngineState.READY):
    return RoleSnapshot(
        "teacher", 1, models=tuple(build_model_snapshot(name, [(0, f"http://{host}:1", state)]) for name in names)
    )


async def test_managed_teacher_rechecks_state_and_never_falls_back_to_startup(monkeypatch):
    owner = AdmissionManager(snapshot("__default__"))
    client = owner.client("teacher")
    monkeypatch.setattr(opd, "get_request_client", lambda role: client)
    manager = object.__new__(opd.OpdManager)
    manager.args = Namespace(opd_teacher_timeout_s=0.01)
    session = Session()
    sample = Sample(index=0)

    async def send():
        return await manager._post_logprob(
            session, "http://startup:9/generate", {}, sample, "teacher", inference={"role": "teacher"}
        )

    await send()
    owner.snapshot = snapshot("__default__", state=EngineState.SLEEPING)
    with pytest.raises(ModelUnavailableError):
        await send()
    owner.snapshot = snapshot("__default__", host="replacement")
    await send()
    assert session.urls == ["http://engine:1/generate", "http://replacement:1/generate"]
    assert session.closed == 2
    assert owner.acquired == owner.released


async def test_mopd_routes_with_affinity_through_admission(monkeypatch):
    current = RoleSnapshot(
        "teacher",
        1,
        models=(
            build_model_snapshot("math", [(0, "http://math:1", EngineState.READY)]),
            build_model_snapshot(
                "code", [(0, "http://code-0:1", EngineState.READY), (1, "http://code-1:1", EngineState.READY)]
            ),
        ),
    )
    owner = AdmissionManager(current)
    monkeypatch.setattr(opd, "get_request_client", lambda role: owner.client("teacher"))
    manager = object.__new__(opd.OpdManager)
    manager.args = Namespace(opd_teacher_timeout_s=0.01)
    session = Session()
    args = Namespace(opd_teacher_routes_map={"math": ["old"], "code": ["old"]})
    for source, group in (("math", 0), ("code", 1), ("code", 1), ("code", 2)):
        sample = Sample(index=group, group_index=group, metadata={"data_source": source})
        route_key = opd._teacher_route_key(args, sample)
        await manager._post_logprob(
            session,
            "",
            {},
            sample,
            "teacher",
            inference={"role": "teacher", "route_key": route_key, "affinity_key": sample.group_index},
        )
    assert session.urls == [
        "http://math:1/generate",
        "http://code-0:1/generate",
        "http://code-0:1/generate",
        "http://code-1:1/generate",
    ]


async def test_external_teacher_uses_explicit_url_without_managed_admission(monkeypatch):
    monkeypatch.setattr(opd, "get_request_client", lambda role: pytest.fail("external service has no managed permit"))
    args = Namespace(opd_teacher_url="http://external:9/generate")
    manager = object.__new__(opd.OpdManager)
    session = Session()
    await manager._post_logprob(session, opd._pick_teacher_url(args), {}, Sample(index=0), "teacher")
    assert session.urls == [args.opd_teacher_url]


def test_managed_teacher_discovery_cannot_be_used_as_permission_to_send():
    with pytest.raises(ModelUnavailableError):
        opd._pick_teacher_url(
            Namespace(opd_teacher_discovery_url="http://serve:1/engines", opd_teacher_url="http://startup:1/generate")
        )


async def test_opd_connection_failure_notifies_owner_without_waiting_for_recovery(monkeypatch):
    from types import SimpleNamespace

    import aiohttp

    owner = AdmissionManager(snapshot("__default__"))
    hints = []
    owner.report_failure = SimpleNamespace(remote=lambda url: hints.append(url) or object())
    monkeypatch.setattr(opd, "get_request_client", lambda role: owner.client("teacher"))

    class DisconnectedSession:
        @asynccontextmanager
        async def post(self, *args, **kwargs):
            raise aiohttp.ClientConnectionError("teacher vanished")
            yield  # pragma: no cover

    manager = object.__new__(opd.OpdManager)
    manager.args = Namespace(opd_teacher_timeout_s=0.25)
    clock = [0.0]

    async def backoff(delay):
        assert hints == ["http://engine:1"]
        assert owner.released == owner.acquired
        clock[0] += delay

    monkeypatch.setattr(opd, "monotonic", lambda: clock[0])
    monkeypatch.setattr(opd, "sleep", backoff)
    with pytest.raises(ModelUnavailableError, match="retry budget.*exhausted"):
        await manager._post_logprob(
            DisconnectedSession(), "", {}, Sample(index=0), "teacher", inference={"role": "teacher"}
        )
    assert hints == ["http://engine:1"]
    assert owner.released == owner.acquired
