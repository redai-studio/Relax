# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from relax.utils.inference_sync import notify_inference_weight_update


torch = pytest.importorskip("torch")
ray = pytest.importorskip("ray")


@pytest.mark.parametrize("rank,rpc_fails,peer_fails", [(0, False, False), (0, True, False), (1, False, True)])
def test_inference_weight_notification_agrees_on_rpc_failure(monkeypatch, rank, rpc_fails, peer_fails):
    group = object()
    dist = torch.distributed
    monkeypatch.setattr(dist, "get_rank", lambda actual: rank if actual is group else pytest.fail("wrong group"))
    manager = SimpleNamespace(inference_weight_update=SimpleNamespace(remote=MagicMock(return_value="ref")))

    def resolve(ref, timeout):
        assert ref == "ref" and timeout == 30
        if rpc_fails:
            raise RuntimeError("dead manager")
        return {"serial": 1}

    def agree(tensor, op, group):
        assert tensor.device.type == "cpu"
        if peer_fails:
            tensor[0] = 1

    monkeypatch.setattr(ray, "get", resolve)
    monkeypatch.setattr(dist, "all_reduce", agree)
    if rpc_fails or peer_fails:
        with pytest.raises(RuntimeError, match="publish inference"):
            notify_inference_weight_update(manager, group)
    else:
        assert notify_inference_weight_update(manager, group) == {"serial": 1}
    assert manager.inference_weight_update.remote.call_count == int(rank == 0)


def test_inference_async_completion_confirms_every_pipeline_stage(monkeypatch):
    dist = torch.distributed
    group = object()
    monkeypatch.setattr(dist, "get_rank", lambda actual: 0)
    monkeypatch.setattr(dist, "get_world_size", lambda actual: 3)
    monkeypatch.setattr(dist, "all_reduce", lambda *args, **kwargs: None)

    def gather(memberships, local, **kwargs):
        assert kwargs["group"] is group
        memberships[:] = [{"http://a", "http://b"}, None, {"http://a"}]

    monkeypatch.setattr(dist, "all_gather_object", gather)
    remote = MagicMock(side_effect=lambda token: token)
    monkeypatch.setattr(ray, "get", lambda ref, **kwargs: ref)
    manager = SimpleNamespace(inference_weight_update=SimpleNamespace(remote=remote))
    token = {"serial": 1, "engines": ["a", "b"], "urls": {"a": "http://a", "b": "http://b"}}
    completed = notify_inference_weight_update(
        manager, group, token, confirm_topology=True, participating_urls={"http://a", "http://b"}
    )
    assert completed["engines"] == ["a"]
    assert completed["unconfirmed_engines"] == ["b"]
    assert token["engines"] == ["a", "b"]


def _notification_failure_worker(rank, rendezvous):
    from datetime import timedelta

    dist = torch.distributed
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=20))
    try:
        remote = MagicMock(return_value="ref")
        manager = SimpleNamespace(inference_weight_update=SimpleNamespace(remote=remote))
        with patch.object(ray, "get", side_effect=RuntimeError("manager unreachable")):
            with pytest.raises(RuntimeError, match="publish inference"):
                notify_inference_weight_update(manager, dist.group.WORLD)
        assert remote.call_count == int(rank == 0)
    finally:
        dist.destroy_process_group(dist.group.WORLD)


def test_inference_notification_failure_reaches_real_gloo_peers(tmp_path):
    torch.multiprocessing.spawn(_notification_failure_worker, args=(f"file://{tmp_path / 'rendezvous'}",), nprocs=2)
