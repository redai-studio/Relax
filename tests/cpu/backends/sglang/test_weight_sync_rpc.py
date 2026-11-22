# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import subprocess
import sys
from types import SimpleNamespace

import torch
from ray import cloudpickle

from relax.distributed.checkpoint_service.backends import device_direct
from relax.distributed.checkpoint_service.backends.device_direct import RolloutEngine, _ShardRolloutEngine


def test_shard_http_actor_and_native_payload_deserialize_without_training_runtime(tmp_path, monkeypatch):
    plain_module = "relax.backends.sglang.weight_sync_rpc"
    assert _ShardRolloutEngine.__ray_metadata__.modified_class.__module__ == plain_module
    assert RolloutEngine.__ray_metadata__.modified_class.__module__.endswith(".device_direct")
    backend = object.__new__(device_direct.DeviceDirectBackend)
    backend.weight_version, backend._group_name, backend._model_update_groups = 1, "seed", None
    backend.rollout_topology = {0: {}, 1: {"metadata": {"sync_nextn_draft": True}}}
    endpoint = SimpleNamespace(remote=lambda: True)
    backend.lock = SimpleNamespace(acquire=endpoint, release=endpoint)
    captured = []

    def capture(endpoint, body, *, get_rank):
        assert endpoint == "/update_weights_from_distributed" and get_rank
        captured.append(body)
        return []

    backend._batch_request = capture
    with monkeypatch.context() as patched:
        patched.setattr(device_direct.ray, "get", lambda value: value)
        patched.setattr(device_direct.dist, "broadcast", lambda *args, **kwargs: SimpleNamespace(wait=lambda: None))
        backend._update_bucket_weights_from_distributed([("weight", torch.empty(2, 3, dtype=torch.bfloat16))])
    assert len(captured) == 1
    payload = tmp_path / "actor.pkl"
    payload.write_bytes(cloudpickle.dumps((_ShardRolloutEngine.__ray_metadata__.modified_class, captured[0])))
    code = """import sys, ray
from ray import cloudpickle
from pathlib import Path
cls, payload = cloudpickle.loads(Path(sys.argv[1]).read_bytes())
assert cls.__module__ == 'relax.backends.sglang.weight_sync_rpc'
assert 'torch' not in sys.modules
assert 'relax.distributed.checkpoint_service.backends.device_direct' not in sys.modules
assert payload[0]['shapes'] == [[2, 3]] and payload[0]['dtypes'] == ['bfloat16']
assert 'selector' not in payload[0] and payload[1] == {**payload[0], 'selector': 'all'}
assert all(callable(getattr(cls, name)) for name in ('health', 'make_request', 'flush_cache', 'update_weights_from_tensor'))
"""
    result = subprocess.run([sys.executable, "-c", code, str(payload)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
