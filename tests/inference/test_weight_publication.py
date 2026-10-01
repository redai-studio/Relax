# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import ast
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
    manager = SimpleNamespace(
        _get_server=lambda _: SimpleNamespace(model_name="default"),
        _get_inference_observation=lambda: observation,
        _inference_replicas=lambda _: {key: [engine]},
    )
    return SimpleNamespace(
        publish=lambda: namespace["mark_inference_weights_ready"](manager),
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
