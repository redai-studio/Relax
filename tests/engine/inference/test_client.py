# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""``InferenceClient``: snapshot caching, refresh triggers, shared routing."""

import pytest

from relax.engine.inference.client import InferenceClient
from relax.engine.inference.discovery import EngineState, RoleSnapshot, build_model_snapshot
from relax.engine.inference.routing import RoutingState, candidate_replicas, select_model, select_target


READY = EngineState.READY


def _snapshot(revision, urls, name="math"):
    model = build_model_snapshot(name, [(i, url, READY) for i, url in enumerate(urls)])
    return RoleSnapshot(role="teacher", topology_revision=revision, models=(model,))


class _Source:
    """Serves a scripted sequence of snapshots; the last one repeats."""

    def __init__(self, *snapshots):
        self.snapshots = list(snapshots)
        self.fetches = 0

    def __call__(self):
        reply = self.snapshots[min(self.fetches, len(self.snapshots) - 1)]
        self.fetches += 1
        if isinstance(reply, Exception):
            raise reply
        return reply


class _Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


def test_client_fetches_once_and_reuses_the_snapshot():
    source = _Source(_snapshot(1, ["http://a:1"]))
    client = InferenceClient(source)

    assert [client.resolve().base_url for _ in range(3)] == ["http://a:1"] * 3
    assert source.fetches == 1


def test_client_accepts_snapshot_dicts():
    client = InferenceClient(_Source(_snapshot(1, ["http://a:1"]).to_dict()))

    assert client.resolve(model="math").engine_id == "math/0"


def test_client_refreshes_after_connection_failure():
    source = _Source(_snapshot(1, ["http://old:1"]), _snapshot(2, ["http://new:1"]))
    client = InferenceClient(source)
    assert client.resolve().base_url == "http://old:1"

    client.report_failure()

    assert client.resolve().base_url == "http://new:1"
    assert source.fetches == 2


def test_client_drops_addresses_missing_from_new_snapshot():
    source = _Source(_snapshot(1, ["http://a:1", "http://gone:1"]), _snapshot(2, ["http://a:1", "http://b:1"]))
    client = InferenceClient(source)
    assert {client.resolve().base_url for _ in range(2)} == {"http://a:1", "http://gone:1"}

    client.report_failure()

    assert {client.resolve().base_url for _ in range(4)} == {"http://a:1", "http://b:1"}


def test_client_respects_refresh_cooldown():
    clock = _Clock()
    source = _Source(_snapshot(1, ["http://a:1"]), _snapshot(2, ["http://b:1"]))
    client = InferenceClient(source, refresh_cooldown_s=5.0, clock=clock)
    client.resolve()

    # A burst of failures right after a fetch does not trigger another one.
    for _ in range(10):
        client.report_failure()
        client.resolve()
    assert source.fetches == 1

    clock.now += 5.0
    client.report_failure()
    assert client.resolve().base_url == "http://b:1"
    assert source.fetches == 2


def test_client_notices_revision_change_when_snapshot_ages_out():
    clock = _Clock()
    source = _Source(_snapshot(1, ["http://a:1"]), _snapshot(2, ["http://b:1"]))
    client = InferenceClient(source, max_age_s=2.0, clock=clock)
    assert client.resolve().base_url == "http://a:1"

    clock.now += 1.0
    assert client.resolve().base_url == "http://a:1"
    clock.now += 1.0
    assert client.resolve().base_url == "http://b:1"
    assert client.snapshot().topology_revision == 2


def test_client_keeps_last_snapshot_when_refresh_fails():
    source = _Source(_snapshot(1, ["http://a:1"]), ConnectionError("discovery down"))
    client = InferenceClient(source)
    client.resolve()

    client.report_failure()

    assert client.resolve().base_url == "http://a:1"


def test_client_raises_when_no_snapshot_was_ever_fetched():
    client = InferenceClient(_Source(ConnectionError("discovery down")))

    with pytest.raises(ConnectionError):
        client.resolve()


def test_client_and_gateway_select_same_candidates():
    """A gateway routes with ``select_target`` on the snapshot it holds; the
    client must reach the same model and choose from the same replicas."""
    snapshot = RoleSnapshot(
        role="genrm",
        topology_revision=1,
        route_keys={"judge": "quality"},
        models=(
            build_model_snapshot("quality", [(0, "http://q0:1", READY), (1, "http://q1:1", EngineState.SLEEPING)]),
            build_model_snapshot("safety", [(0, "http://s0:1", READY)]),
        ),
    )
    client = InferenceClient(lambda: snapshot)

    for kwargs in ({"model": "safety"}, {"route_key": "judge"}, {"route_key": "safety"}):
        gateway_model = select_model(snapshot, kwargs.get("model"), kwargs.get("route_key"))
        gateway_target = select_target(snapshot, RoutingState(), **kwargs)
        client_target = client.resolve(**kwargs)

        assert client_target.model == gateway_target.model == gateway_model.name
        assert client_target.base_url in {replica.base_url for replica in candidate_replicas(gateway_model)}
        assert client_target == gateway_target
