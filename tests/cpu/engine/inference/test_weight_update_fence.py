# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from datetime import timedelta
from pathlib import Path
from time import monotonic
from types import SimpleNamespace

import pytest
import ray
import torch.distributed as dist
import torch.multiprocessing as mp

from relax.distributed.ray.inference_handle import ManagedRolloutHandle
from relax.engine.inference.lifecycle import managed_weight_update


def _rank(rank: int, rendezvous: str, directory: str, fail: bool | str) -> None:
    from relax.utils.distributed_utils import init_gloo_group

    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=30))
    group = init_gloo_group(distributed_timeout_minutes=1)
    original_get = ray.get
    ray.get = lambda value: value
    entered = False

    def begin(source):
        if fail is True:
            raise RuntimeError("manager rejected admission")
        return "weight-token"

    class Backend:
        args = SimpleNamespace(rank=rank)
        weight_updater = SimpleNamespace(weight_version="transport-1")
        rollout_manager = ManagedRolloutHandle(
            SimpleNamespace(
                begin_weight_update=SimpleNamespace(remote=begin),
                end_weight_update=SimpleNamespace(
                    remote=lambda token, version: Path(directory, f"end-{rank}.txt").write_text(str(version))
                ),
            ),
            None,
        )

        @managed_weight_update
        def update(self):
            nonlocal entered
            entered = True
            dist.all_reduce(value, group=group)
            if fail == "body" and rank == 1:
                raise RuntimeError("rank-local post-sync failed")

    import torch

    value = torch.ones(1)
    try:
        if fail in {"scaled", "lock_acquire", "lock_release"}:
            import asyncio

            from relax.backends.megatron import actor as actor_module

            async def transfer(*args):
                Path(directory, f"transfer-{rank}.txt").touch()
                return []

            def acquire():
                if fail == "lock_acquire":
                    raise RuntimeError("lock unavailable")
                return True

            def release():
                if fail == "lock_release":
                    raise RuntimeError("lock release unavailable")

            def scaled():
                Path(directory, f"scaled-{rank}.txt").touch()
                return {"success": False}

            actor_module.print_memory = lambda *args, **kwargs: None
            actor_module.run = asyncio.run
            train_actor = object.__new__(actor_module.MegatronTrainRayActor)
            train_actor.args = SimpleNamespace(rollout_http_timeout=5)
            train_actor._weight_sync_lock = SimpleNamespace(
                acquire=SimpleNamespace(remote=acquire), release=SimpleNamespace(remote=release)
            )
            train_actor.checkpoint_engine_client = SimpleNamespace(
                backend=SimpleNamespace(weight_version="1"), update_weights_for_rollout=transfer
            )
            train_actor.rollout_manager = SimpleNamespace(
                register_recovered_engines=SimpleNamespace(remote=lambda *args, **kwargs: None),
                sync_weights_for_scaled_out_engines=SimpleNamespace(remote=scaled),
            )
            entered = True
            train_actor.update_weights_fully_async(0, rollout_only=True)
        else:
            Backend().update()
        result = f"entered={entered},sum={int(value[0])}"
    except RuntimeError as exc:
        result = f"entered={entered},error={exc}"
    finally:
        ray.get = original_get
        dist.destroy_process_group(group)
        dist.destroy_process_group(dist.group.WORLD)
    Path(directory, f"rank-{rank}.txt").write_text(result)


@pytest.mark.skipif(not dist.is_gloo_available(), reason="requires CPU Gloo collectives")
@pytest.mark.parametrize("fail", [False, True])
def test_weight_fence_outcome_reaches_all_ranks_before_collectives(tmp_path, fail):
    rendezvous = f"file://{tmp_path / 'rendezvous'}"
    context = mp.spawn(_rank, args=(rendezvous, str(tmp_path), fail), nprocs=2, join=False)
    deadline = monotonic() + 45
    try:
        while not context.join(timeout=1):
            if monotonic() >= deadline:
                raise TimeoutError("CPU weight-fence ranks did not finish within 45 seconds")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
    results = [(tmp_path / f"rank-{rank}.txt").read_text() for rank in range(2)]
    if fail:
        assert results == ["entered=False,error=Inference weight-update fence failed: RuntimeError"] * 2
    else:
        assert results == ["entered=True,sum=2"] * 2


@pytest.mark.parametrize("transport", ["megatron", "dcs", "fsdp"])
def test_managed_fence_publishes_transport_version_after_sync(monkeypatch, transport):
    events = []
    owner = SimpleNamespace(
        begin_weight_update=SimpleNamespace(remote=lambda source: "transaction"),
        end_weight_update=SimpleNamespace(remote=lambda token, version: events.append((token, version))),
    )
    monkeypatch.setattr(ray, "get", lambda value: value)
    monkeypatch.setattr(dist, "is_initialized", lambda: False)

    class Backend:
        args = SimpleNamespace(rank=0, rollout_id=9999)
        rollout_manager = ManagedRolloutHandle(owner, None)

        @managed_weight_update
        def sync(self):
            if transport == "megatron":
                self.weight_updater.weight_version = "tensor-13"
            elif transport == "dcs":
                self.checkpoint_engine_client.backend.weight_version = "dcs-17"
            else:
                self.policy_version = 23
            return "completed"

    backend = Backend()
    if transport == "megatron":
        backend.weight_updater = SimpleNamespace(weight_version="tensor-12")
        expected = "tensor-13"
    elif transport == "dcs":
        backend._use_dcs_hybrid_weight_sync = True
        backend.weight_updater = SimpleNamespace(weight_version="unrelated")
        backend.checkpoint_engine_client = SimpleNamespace(backend=SimpleNamespace(weight_version="dcs-16"))
        expected = "dcs-17"
    else:
        backend.policy_version = 22
        expected = "23"
    assert backend.sync() == "completed"
    assert events == [("transaction", expected)]


@pytest.mark.skipif(not dist.is_gloo_available(), reason="requires CPU Gloo collectives")
@pytest.mark.parametrize(
    ("failure", "message", "transferred", "scaled"),
    [
        ("body", "Policy weight synchronization failed on ranks [1]", False, False),
        ("scaled", "Scaled-out replica weight sync failed", True, True),
        ("lock_acquire", "DCS weight-sync lock acquisition failed: RuntimeError: lock unavailable", False, False),
        ("lock_release", "DCS weight-sync cleanup failed: RuntimeError: lock release unavailable", True, False),
    ],
)
def test_rank_local_sync_failure_never_strands_peers(tmp_path, failure, message, transferred, scaled):
    """Two bounded CPU ranks exercise the real collective error handshake.

    Local mocks cannot reveal a peer stuck waiting for the end broadcast. This
    reuses the existing Gloo harness (two processes, 45-second parent timeout).
    """
    rendezvous = f"file://{tmp_path / 'post-sync-rendezvous'}"
    context = mp.spawn(_rank, args=(rendezvous, str(tmp_path), failure), nprocs=2, join=False)
    deadline = monotonic() + 45
    try:
        while not context.join(timeout=1):
            if monotonic() >= deadline:
                pytest.fail("rank-local sync failure stranded a peer")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
    for rank in range(2):
        result = (tmp_path / f"rank-{rank}.txt").read_text()
        assert message in result, f"rank {rank} reported {result!r}"
        assert (tmp_path / f"transfer-{rank}.txt").exists() is transferred
        assert (tmp_path / f"scaled-{rank}.txt").exists() is (scaled and rank == 0)
        if failure == "body":
            assert "rank-local post-sync failed" in result
    assert not list(tmp_path.glob("end-*.txt"))
