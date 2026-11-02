# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import importlib
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from relax.distributed.ray import placement_group as placement


@pytest.mark.parametrize("stage", ["ready", "probe"])
def test_pending_placement_or_gpu_probe_timeout_releases_attempt_resources(monkeypatch, stage):
    ready, probe = object(), object()
    pg = SimpleNamespace(ready=lambda: ready)
    actor = SimpleNamespace(get_ip_and_gpu_id=SimpleNamespace(remote=lambda: probe))
    cls = MagicMock()
    cls.options.return_value.remote.return_value = actor
    removed, killed = [], []
    monkeypatch.setattr(placement, "placement_group", lambda *a, **kw: pg)
    monkeypatch.setattr(placement, "InfoActor", cls)
    monkeypatch.setattr(placement, "PlacementGroupSchedulingStrategy", lambda **kw: None)
    monkeypatch.setattr(placement, "get_ray_accelerator_kwargs", lambda _: {})
    monkeypatch.setattr(placement.device_utils, "get_ray_accelerator_name", lambda: "GPU")
    monkeypatch.setattr(placement.ray, "kill", killed.append)
    monkeypatch.setattr(importlib.import_module("ray.util.placement_group"), "remove_placement_group", removed.append)

    def get(ref, *, timeout=None):
        assert timeout is not None and timeout > 0
        if stage == "ready" or ref is not ready:
            raise placement.ray.exceptions.GetTimeoutError("capacity unavailable")
        return True

    monkeypatch.setattr(placement.ray, "get", get)
    with pytest.raises(placement.ray.exceptions.GetTimeoutError):
        placement.create_placement_group(1, node_group_affinity=False)
    assert removed == [pg]
    assert killed == ([] if stage == "ready" else [actor])
