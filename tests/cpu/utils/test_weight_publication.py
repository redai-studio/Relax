# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""CPU transport tests: peer rejection and publisher restart must stay
unknown."""

import asyncio
import threading
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch.distributed as dist
import torch.multiprocessing as mp

from relax.utils.agentic_obs import weight_publication as publication
from relax.utils.agentic_obs.weight_audit import PublicationLedger, attempt_source


def _args():
    return SimpleNamespace(dump_agentic_obs="unused", agentic_obs_run_id="run", use_agentic_rollout=True)


def _publication_worker(rank, rendezvous):
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=20))
    ledger = PublicationLedger()
    coordinator = SimpleNamespace(
        begin_weight_publication=SimpleNamespace(remote=ledger.begin),
        commit_weight_publication=SimpleNamespace(remote=ledger.commit),
    )
    try:
        with (
            patch("relax.utils.distributed_utils.get_gloo_group", return_value=dist.group.WORLD),
            patch("ray.get", side_effect=lambda result, **kwargs: result),
            patch.object(publication, "_resolve_coordinator", return_value=coordinator),
        ):
            owner = SimpleNamespace(args=_args(), weight_version=1)
            publication.begin_publication(owner)
            publication.commit_publication(owner)
            first = publication.consumption_snapshot(owner.args)
            assert first["ordinal"] == 1
            assert first["raw_version"] == publication.publication_label(owner)
            owner.weight_version = 2
            publication.begin_publication(owner)
            publication.observe_install_result(owner, [{"success": rank == 0}])
            publication.commit_publication(owner)
            assert publication.consumption_snapshot(owner.args) is None
            owner.weight_version = 90
            publication.begin_publication(owner)
            publication.commit_publication(owner)
            assert publication.consumption_snapshot(owner.args)["ordinal"] == 2
            stale = ledger.snapshot()
            # Restart with reused numeric counter, then lose the begin RPC.
            restarted = SimpleNamespace(args=_args(), weight_version=90)
            with patch.object(publication, "_resolve_coordinator", side_effect=TimeoutError("lost begin")):
                publication.begin_publication(restarted)
            publication.commit_publication(restarted)
            assert publication.consumption_snapshot(restarted.args) is None
            assert publication.publication_label(restarted) != publication.publication_label(owner)
            assert attempt_source(publication.publication_label(restarted), stale, stale) is None
    finally:
        dist.destroy_process_group()


def test_publication_rejection_on_peer_and_restart_with_lost_begin(tmp_path):
    # Two ranks are necessary: a rank-zero-only mock cannot expose a lost PP ACK.
    mp.spawn(_publication_worker, args=((tmp_path / "gloo").as_uri(),), nprocs=2, join=True)


def _divergent_receiver_worker(rank, rendezvous):
    """Two PP source ranks report different receiver sets (PP0={A,B}, PP1={B});

    the intersection {B} is non-empty yet engine A only installed PP0's stage,
    so committing a shared new label would certify a partial install.
    """
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=20))
    ledger = PublicationLedger()
    coordinator = SimpleNamespace(
        begin_weight_publication=SimpleNamespace(remote=ledger.begin),
        commit_weight_publication=SimpleNamespace(remote=ledger.commit),
    )
    try:
        with (
            patch("relax.utils.distributed_utils.get_gloo_group", return_value=dist.group.WORLD),
            patch("ray.get", side_effect=lambda result, **kwargs: result),
            patch.object(publication, "_resolve_coordinator", return_value=coordinator),
        ):
            # Rank 0 kept receivers {0, 1}; rank 1 pruned receiver 0 after its
            # proxy died (the SGLang engine behind it stays healthy).
            engines = {0: object(), 1: object()} if rank == 0 else {1: object()}
            owner = SimpleNamespace(args=_args(), weight_version=1, rollout_engines=engines)
            publication.begin_publication(owner)
            publication.commit_publication(owner)
            assert publication.consumption_snapshot(owner.args) is None
            # With identical receiver sets the same flow certifies normally.
            owner = SimpleNamespace(args=_args(), weight_version=2, rollout_engines={0: object(), 1: object()})
            publication.begin_publication(owner)
            publication.commit_publication(owner)
            assert publication.consumption_snapshot(owner.args)["ordinal"] == 1
    finally:
        dist.destroy_process_group()


def test_publication_rejects_divergent_pp_receiver_sets(tmp_path):
    # A rank-zero-only mock cannot exercise a cross-PP receiver-set mismatch.
    mp.spawn(_divergent_receiver_worker, args=((tmp_path / "gloo").as_uri(),), nprocs=2, join=True)


def test_coordinator_lookup_is_bounded_even_when_handle_discovery_blocks(monkeypatch):
    from ray import serve

    release = threading.Event()
    entered = threading.Event()

    def stalled_lookup(*args):
        entered.set()
        release.wait(2)
        raise RuntimeError("released")

    monkeypatch.setattr(serve, "get_app_handle", stalled_lookup)
    monkeypatch.setattr(publication, "_TIMEOUT", 0.1)
    try:
        with pytest.raises(TimeoutError):
            publication._resolve_coordinator()
        assert entered.is_set()
    finally:
        release.set()


def test_coordinator_lookup_from_running_async_loop_uses_worker_thread(monkeypatch):
    from ray import serve

    coordinator = object()

    def lookup(*args):
        with pytest.raises(RuntimeError, match="no running event loop"):
            asyncio.get_running_loop()
        return SimpleNamespace(
            runtime_resources=SimpleNamespace(
                remote=lambda: SimpleNamespace(result=lambda **kwargs: ((), coordinator))
            )
        )

    monkeypatch.setattr(serve, "get_app_handle", lookup)

    async def caller():
        assert publication._resolve_coordinator() is coordinator

    asyncio.run(caller())


def test_custom_router_keeps_original_label_and_disables_certification():
    args = _args()
    args.use_slime_router = True
    args.slime_router_middleware_paths = ["relax.engine.router.middleware.radix_tree_middleware.RadixTreeMiddleware"]
    owner = SimpleNamespace(args=args, weight_version=3)
    publication.begin_publication(owner)
    publication.commit_publication(owner)
    assert publication.publication_label(owner) == "3"
    assert not publication.weight_audit_enabled(args)
