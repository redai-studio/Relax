import json
from types import SimpleNamespace

import pytest
import torch.distributed as distributed

from relax.distributed.weight_sync.fence import WorkerFence
from relax.distributed.weight_sync.runtime import ConsumptionError
from relax.distributed.weight_sync.sglang_plugin import worker_rpc


def test_consumption_error_preserves_recovery_state_when_serialized():
    import pickle

    state = {"stage": "INSTALLING", "reason": "worker busy", "allowed_actions": ["retry"]}
    original = ConsumptionError(state)
    restored = pickle.loads(pickle.dumps(original))
    assert restored.state == state
    assert str(restored) == str(original)


def test_ray_task_error_preserves_consumer_recovery_state():
    from ray import cloudpickle
    from ray.exceptions import RayTaskError

    state = {"stage": "INSTALLING", "reason": "worker busy", "allowed_actions": ["retry"]}
    task_error = RayTaskError("consume_weight_version", "traceback", ConsumptionError(state))
    restored = cloudpickle.loads(cloudpickle.dumps(task_error)).as_instanceof_cause()
    assert isinstance(restored, ConsumptionError)
    assert restored.cause.state == state


@pytest.mark.parametrize("stale", [False, True])
def test_busy_worker_receipt_preserves_identity_failures(monkeypatch, tmp_path, stale):
    fence = WorkerFence()
    identity = fence.begin(fence.instance, 1, "one", "root")
    scheduler = SimpleNamespace(
        ps=SimpleNamespace(pp_rank=0, tp_size=1, tp_rank=0),
        tp_group=SimpleNamespace(cpu_group=object()),
        _relax_weight_fence=fence,
        is_fully_idle=lambda: False,
    )
    monkeypatch.setattr(distributed, "get_world_size", lambda group: 1)
    monkeypatch.setattr(distributed, "all_gather_object", lambda output, value, group: output.__setitem__(0, value))
    if stale:
        identity = {**identity, "epoch": 0}
    worker_rpc(
        scheduler,
        json.dumps(
            {
                "request_id": "request",
                "receipts": str(tmp_path),
                "phase": "flush",
                "command_id": "flush",
                "workers": {"0": identity},
            }
        ),
    )
    receipt = json.loads((tmp_path / "rank-0.json").read_bytes())
    assert not receipt["success"]
    assert receipt["error_code"] == ("WORKER_ERROR" if stale else "WORKER_BUSY")
    assert fence.phase == "PREPARED"


@pytest.mark.parametrize("cache_enabled,flush_ok", [(True, True), (False, True), (True, False)])
def test_version_flush_invalidates_visual_features(monkeypatch, tmp_path, cache_enabled, flush_ok):
    import torch
    from sglang.srt.managers import mm_utils
    from sglang.srt.mem_cache.multimodal_cache import EmbeddingResult, MultiModalStaticCache

    cache = MultiModalStaticCache(1024)
    assert cache.set(17, EmbeddingResult(embedding=torch.ones(8)))
    monkeypatch.setattr(mm_utils, "embedding_cache", cache if cache_enabled else None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(distributed, "get_world_size", lambda group: 1)
    monkeypatch.setattr(distributed, "all_gather_object", lambda output, value, group: output.__setitem__(0, value))
    fence = WorkerFence()
    identity = fence.begin(fence.instance, 1, "one", "root")
    fence.phase = "verify"
    scheduler = SimpleNamespace(
        ps=SimpleNamespace(pp_rank=0, tp_size=1, tp_rank=0),
        tp_group=SimpleNamespace(cpu_group=object()),
        _relax_weight_fence=fence,
        is_fully_idle=lambda: True,
        flush_cache=lambda: flush_ok,
    )
    worker_rpc(
        scheduler,
        json.dumps(
            {
                "request_id": "request",
                "receipts": str(tmp_path),
                "phase": "flush",
                "command_id": "flush",
                "workers": {"0": identity},
            }
        ),
    )
    receipt = json.loads((tmp_path / "rank-0.json").read_bytes())
    assert receipt["success"] == flush_ok
    if cache_enabled and flush_ok:
        assert cache.get_single(17) is None
        assert cache.current_size == 0 and cache.max_size == 1024
        assert receipt["result"]["multimodal_cache"] == {"entries_cleared": 1, "bytes_cleared": 32}
    else:
        assert cache.get_single(17) is not None
        assert cache.current_size == 32
