# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""``FSDPTrainRayActor`` attaches its lazily created TransferQueue client
through the bounded lifecycle helper and detaches it on teardown."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

from relax.backends.fsdp import actor as fsdp_actor
from relax.backends.fsdp.actor import FSDPTrainRayActor


def _make_actor(tq_config: Any) -> FSDPTrainRayActor:
    actor = FSDPTrainRayActor.__new__(FSDPTrainRayActor)
    actor.args = SimpleNamespace(tq_config=tq_config)
    actor._data_system_client = None
    return actor


def test_tq_client_attaches_through_bounded_helper(monkeypatch) -> None:
    sentinel = object()
    attach = MagicMock(return_value=sentinel)
    monkeypatch.setattr(fsdp_actor, "attach_tq_client", attach)

    actor = _make_actor(SimpleNamespace(storage_backend="SimpleStorage"))

    assert actor._tq_client() is sentinel
    attach.assert_called_once()
    assert attach.call_args.args[0] is actor.args.tq_config
    assert attach.call_args.kwargs["role"] == "fsdp_actor"

    # The attached client is cached: a second read must not re-attach.
    assert actor._tq_client() is sentinel
    attach.assert_called_once()


def test_tq_client_without_config_reuses_existing_client(monkeypatch) -> None:
    import transfer_queue as tq

    client = object()
    get_client = MagicMock(return_value=client)
    attach = MagicMock()
    monkeypatch.setattr(tq, "get_client", get_client)
    monkeypatch.setattr(fsdp_actor, "attach_tq_client", attach)

    actor = _make_actor(None)

    assert actor._tq_client() is client
    get_client.assert_called_once()
    attach.assert_not_called()


def test_destructor_detaches_once_and_never_raises(monkeypatch) -> None:
    detach = MagicMock()
    monkeypatch.setattr(fsdp_actor, "detach_tq_client", detach)

    actor = _make_actor(SimpleNamespace())
    actor._data_system_client = object()

    actor.__del__()
    detach.assert_called_once()
    assert actor._data_system_client is None

    # Already detached: the destructor must stay a no-op instead of re-closing.
    actor.__del__()
    detach.assert_called_once()

    # A raising helper is swallowed: destructors run during interpreter shutdown.
    actor._data_system_client = object()
    monkeypatch.setattr(fsdp_actor, "detach_tq_client", MagicMock(side_effect=RuntimeError("boom")))
    actor.__del__()
