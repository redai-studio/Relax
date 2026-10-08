# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Where OPD gets a managed teacher's replica addresses from.

A managed teacher publishes its topology through the ``/teacher`` gateway; OPD
follows it so a replica rebuilt at another address is used without a restart.
The URLs captured in ``args`` at startup stay the fallback, and an external
teacher never involves discovery at all.
"""

from __future__ import annotations

from argparse import Namespace
from contextlib import asynccontextmanager

import pytest

from relax.engine.inference.discovery import EngineState, RoleSnapshot, build_model_snapshot
from relax.engine.rollout import on_policy_distillation as opd
from relax.utils.types import Sample


DISCOVERY_URL = "http://serve:8000/teacher/engines?schema_version=2"
STARTUP_URL = "http://startup:1/generate"


class _Discovery:
    """Stands in for the teacher gateway's discovery endpoint.

    Serves a scripted sequence of replies (the last one repeats); an exception
    in the sequence is raised, as an unreachable gateway would.
    """

    def __init__(self, monkeypatch, *replies):
        self.replies = list(replies)
        self.fetches: list[str] = []
        monkeypatch.setattr(opd, "_fetch_teacher_topology", self._fetch)

    def _fetch(self, discovery_url: str) -> dict:
        reply = self.replies[min(len(self.fetches), len(self.replies) - 1)]
        self.fetches.append(discovery_url)
        if isinstance(reply, Exception):
            raise reply
        return reply


def _topology(revision: int, **models: list[tuple[str, EngineState]]) -> dict:
    return RoleSnapshot(
        role="teacher",
        topology_revision=revision,
        default_model="__default__" if "__default__" in models else None,
        models=tuple(
            build_model_snapshot(name, [(index, url, state) for index, (url, state) in enumerate(replicas)])
            for name, replicas in models.items()
        ),
    ).to_dict()


def _ready(*hosts: str) -> list[tuple[str, EngineState]]:
    return [(f"http://{host}:1", EngineState.READY) for host in hosts]


_MODULE_STATE = (
    opd._TEACHER_URL_RR,
    opd._TEACHER_GROUP_REPLICA,
    opd._TEACHER_TOPOLOGY,
    opd._TEACHER_DISCOVERY_WARNED,
)


@pytest.fixture(autouse=True)
def _isolate_module_state(monkeypatch):
    """Routing and topology state is module-level; isolate every test."""
    for state in _MODULE_STATE:
        state.clear()
    # A failed request must be able to trigger a refresh right away in tests.
    monkeypatch.setattr(opd, "_TEACHER_DISCOVERY_COOLDOWN_S", 0.0)
    yield
    for state in _MODULE_STATE:
        state.clear()


def _managed_args(**overrides) -> Namespace:
    values = dict(
        opd_teacher_url=STARTUP_URL,
        opd_teacher_urls=[STARTUP_URL],
        opd_teacher_discovery_url=DISCOVERY_URL,
    )
    values.update(overrides)
    return Namespace(**values)


async def test_opd_teacher_url_uses_discovery_for_managed_teacher(monkeypatch):
    discovery = _Discovery(monkeypatch, _topology(1, __default__=_ready("r0", "r1")))
    args = _managed_args()

    await opd._refresh_teacher_topology(args)
    picks = [opd._pick_teacher_url(args, Sample(group_index=group)) for group in (0, 0, 1, 1)]

    # Discovered replicas replace the startup URL, with the usual group affinity.
    assert picks == ["http://r0:1/generate"] * 2 + ["http://r1:1/generate"] * 2
    assert discovery.fetches == [DISCOVERY_URL]


async def test_opd_teacher_url_follows_a_rebuilt_replica(monkeypatch):
    discovery = _Discovery(
        monkeypatch, _topology(1, __default__=_ready("old")), _topology(2, __default__=_ready("new"))
    )
    args = _managed_args()

    await opd._refresh_teacher_topology(args)
    assert opd._pick_teacher_url(args) == "http://old:1/generate"

    # Nothing changes until a request to the old address fails...
    await opd._refresh_teacher_topology(args)
    assert len(discovery.fetches) == 1
    # ...which makes the next batch look at the topology again.
    opd._report_teacher_failure(args)
    await opd._refresh_teacher_topology(args)

    assert opd._pick_teacher_url(args) == "http://new:1/generate"


async def test_opd_teacher_url_uses_discovery_per_mopd_teacher(monkeypatch):
    _Discovery(monkeypatch, _topology(1, math=_ready("math-new"), code=_ready("code-0", "code-1")))
    args = _managed_args(
        opd_teacher_routes_map={"math": ["http://math-old:1/generate"], "code": ["http://code-old:1/generate"]},
        opd_teacher_key="data_source",
    )

    await opd._refresh_teacher_topology(args)

    def pick(source: str, group: int) -> str:
        return opd._pick_teacher_url(args, Sample(group_index=group, metadata={"data_source": source}))

    assert pick("math", 0) == "http://math-new:1/generate"
    assert [pick("code", group) for group in (1, 2)] == ["http://code-0:1/generate", "http://code-1:1/generate"]


async def test_opd_teacher_url_falls_back_to_static_urls(monkeypatch):
    monkeypatch.setattr(opd, "_TEACHER_DISCOVERY_COOLDOWN_S", 60.0)
    discovery = _Discovery(monkeypatch, ConnectionError("gateway unreachable"))
    args = _managed_args()

    await opd._refresh_teacher_topology(args)
    assert opd._pick_teacher_url(args) == STARTUP_URL

    # An unreachable gateway is not hammered once per batch.
    await opd._refresh_teacher_topology(args)
    assert len(discovery.fetches) == 1


async def test_opd_teacher_url_falls_back_when_no_replica_is_ready(monkeypatch):
    _Discovery(monkeypatch, _topology(1, __default__=[("http://r0:1", EngineState.SLEEPING)]))
    args = _managed_args()

    await opd._refresh_teacher_topology(args)

    assert opd._pick_teacher_url(args) == STARTUP_URL


async def test_opd_teacher_url_external_url_bypasses_discovery(monkeypatch):
    discovery = _Discovery(monkeypatch, AssertionError("an external teacher must not be discovered"))
    args = Namespace(opd_teacher_url="http://external:9000/generate")

    await opd._refresh_teacher_topology(args)

    assert opd._pick_teacher_url(args) == "http://external:9000/generate"
    assert discovery.fetches == []
    assert opd._TEACHER_TOPOLOGY == {}


async def test_opd_prefill_recovers_after_teacher_replica_moves(monkeypatch):
    _Discovery(monkeypatch, _topology(1, __default__=_ready("old")), _topology(2, __default__=_ready("new")))
    args = _managed_args()
    requested: list[str] = []

    @asynccontextmanager
    async def fake_session(args):
        yield None

    async def fake_teacher_prefill(sample, session) -> bool:
        url = opd._pick_teacher_url(args, sample)
        requested.append(url)
        return "old" not in url  # the old replica is gone

    manager = object.__new__(opd.OpdManager)
    manager.args, manager.opsd_worker, manager.topk_worker = args, None, None
    manager._teacher_prefill = fake_teacher_prefill
    monkeypatch.setattr(opd, "_create_teacher_client_session", fake_session)
    sample = Sample(index=0, response_length=4)

    with pytest.raises(RuntimeError, match="All OPD teacher fetches failed"):
        await manager.prefill(sample)
    await manager.prefill(sample)

    assert requested == ["http://old:1/generate", "http://new:1/generate"]
