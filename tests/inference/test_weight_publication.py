# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import ast
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Optional
from unittest.mock import MagicMock

import pytest
import ray
import requests

from relax.distributed.ray.inference_manager import _InferenceObservation
from relax.utils.logging_utils import get_logger


@pytest.fixture
def publication(monkeypatch):
    # Load the real method without importing the optional SGLang runtime.
    path = Path(__file__).parents[2] / "relax/distributed/ray/rollout.py"
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "RolloutManager")
    method = next(
        node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "mark_inference_weights_ready"
    )
    namespace = {"Optional": Optional, "ray": ray, "logger": get_logger(__name__)}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    monkeypatch.setattr(ray, "get", lambda ref, timeout: ref)
    engine = MagicMock()
    observation = _InferenceObservation("rollout")
    key = "default/group-0/replica-0"
    observation.initialized(key, [engine], weights_ready=False)
    record = SimpleNamespace(pending_router_registration=False)
    group = SimpleNamespace(all_engines=[engine], engine_init_records={0: record})
    replicas = {key: [engine]}
    manager = SimpleNamespace(
        _engine_lifecycle_lock=threading.RLock(),
        _get_server=lambda _: SimpleNamespace(model_name="default", engine_groups=[group]),
        _get_inference_observation=lambda: observation,
        _inference_replicas=lambda _: replicas,
    )
    return SimpleNamespace(
        publish=lambda **kwargs: namespace["mark_inference_weights_ready"](manager, **kwargs),
        record=record,
        replicas=replicas,
        engine=engine,
        observation=observation,
        key=key,
        logger=namespace["logger"],
    )


@pytest.mark.parametrize(
    "error", [requests.HTTPError("404 Not Found"), ray.exceptions.GetTimeoutError("query timeout")]
)
def test_weight_publication_failure_fences_logs_and_propagates(publication, monkeypatch, caplog, error):
    monkeypatch.setattr(publication.logger, "handlers", [caplog.handler])
    publication.engine.get_weight_version.remote.side_effect = error
    revision = publication.observation.registry.snapshot()["topology_revision"]

    with pytest.raises(type(error)) as caught:
        publication.publish()

    assert caught.value is error
    entry = publication.observation.entries[publication.key]
    assert entry["state"] == "FAILED"
    assert not entry["admission"] and not entry["weights_ready"] and not entry["initial_weight_probe"]
    assert publication.observation.registry.snapshot()["topology_revision"] > revision
    assert any(
        publication.key in record.getMessage() and record.exc_info and record.exc_info[1] is error
        for record in caplog.records
    )

    publication.engine.get_weight_version.remote.side_effect = None
    publication.engine.get_weight_version.remote.return_value = "8"
    publication.publish()
    assert entry["state"] == "READY" and entry["weight_version"] == "8"


@pytest.mark.parametrize("version", [None, "", "default"])
def test_weight_publication_rejects_unpublished_version(publication, version):
    publication.engine.get_weight_version.remote.return_value = version

    with pytest.raises(RuntimeError, match="no published weight version"):
        publication.publish()

    assert publication.observation.entries[publication.key]["state"] == "FAILED"


def test_weight_publication_accepts_zero_version(publication):
    publication.engine.get_weight_version.remote.return_value = 0

    publication.publish()

    entry = publication.observation.entries[publication.key]
    assert entry["state"] == "READY"
    assert entry["weight_version"] == "0"


@pytest.mark.parametrize("synced", [[], ["different-generation"]])
def test_weight_publication_excludes_unsynced_heads(publication, synced):
    publication.engine._actor_id.hex.return_value = "current-generation"
    publication.engine.get_weight_version.remote.return_value = "old-version"
    publication.publish(synced_actor_ids=synced)
    entry = publication.observation.entries[publication.key]
    assert not entry["weights_ready"] and not entry["admission"]
    publication.engine.get_weight_version.remote.assert_not_called()


@pytest.mark.parametrize("synced", [None, ["current-generation"]])
def test_weight_publication_waits_for_router_registration(publication, synced):
    publication.engine._actor_id.hex.return_value = "current-generation"
    publication.record.pending_router_registration = True
    publication.publish(synced_actor_ids=synced)
    assert not publication.observation.entries[publication.key]["weights_ready"]
    publication.engine.get_weight_version.remote.assert_not_called()
    publication.record.pending_router_registration = False
    publication.engine.get_weight_version.remote.return_value = "9"
    publication.publish(synced_actor_ids=synced)
    assert publication.observation.entries[publication.key]["weights_ready"]


def test_weight_publication_requires_only_multinode_head_receipt(publication):
    publication.engine._actor_id.hex.return_value = "head-generation"
    publication.engine.get_weight_version.remote.return_value = "9"
    publication.replicas[publication.key].append(MagicMock())
    publication.publish(synced_actor_ids=["head-generation"])
    assert publication.observation.entries[publication.key]["weights_ready"]
