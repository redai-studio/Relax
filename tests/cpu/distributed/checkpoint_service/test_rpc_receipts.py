# Copyright (c) 2026 Relax Authors. All Rights Reserved.
import copy
import json
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from relax.distributed.checkpoint_service.backends.shard_p2p.remote.plan import (
    RPC_RECEIPT_DELIVERY,
    ReceiptValidator,
    fingerprint,
)
from relax.distributed.checkpoint_service.backends.shard_p2p.remote.rpc_receipts import (
    index_receipts,
    read_commit_receipts,
    route_receipts,
)
from relax.distributed.checkpoint_service.backends.shard_p2p.remote.sender import _ExpertTransferProtocol
from relax.distributed.checkpoint_service.backends.shard_p2p.remote.synchronizer import P2PSynchronizer


def fixture_plans(world_size=8, engines=4):
    # Nonuniform target ownership, including owners with no writes to a target.
    return {
        f"{engine}-{tp}": dict(
            receipt_delivery=RPC_RECEIPT_DELIVERY,
            writes=[
                dict(source_rank=rank, id=f"{rank}:{part}")
                for rank in range(world_size)
                if rank % 2 == tp
                for part in range(rank + 1)
            ],
        )
        for engine in range(engines)
        for tp in range(2)
    }


def fixture_sender(rank, plans, group, world_size=8):
    sync = P2PSynchronizer.__new__(P2PSynchronizer)
    sync.rank, sync.world_size, sync.group = rank, world_size, group
    sync._phase_error_flag = torch.zeros(1, dtype=torch.int32)
    direct = _ExpertTransferProtocol.__new__(_ExpertTransferProtocol)
    direct.sync, direct.plans = sync, plans
    direct.hashes = {key: fingerprint(plan) for key, plan in plans.items()}
    direct.rpc_validators = {key: ReceiptValidator(plan) for key, plan in plans.items()}
    direct.manifests = {key: dict(generation=f"generation:{key}") for key in plans}
    return direct


def fixture_receipts(plans, world_size=8, version=4):
    return [
        json.dumps(
            {
                key: dict(
                    source_rank=rank,
                    version=version,
                    generation=f"generation:{key}",
                    plan_hash=fingerprint(plan),
                    writes=[w["id"] for w in plan["writes"] if w["source_rank"] == rank],
                    extra="original-owner-data",
                )
                for key, plan in plans.items()
                if any(w["source_rank"] == rank for w in plan["writes"])
            }
        )
        for rank in range(world_size)
    ]


@pytest.mark.parametrize("world_size,engines", [(3, 1), (4, 3), (8, 4)])
def test_router_preserves_owners_originals_and_target_sets(world_size, engines):
    plans = fixture_plans(world_size, engines)
    validators = {key: ReceiptValidator(plan) for key, plan in plans.items()}
    generations = {key: f"generation:{key}" for key in plans}
    payloads = fixture_receipts(plans, world_size)
    routed = route_receipts(payloads, validators, generations, 4)
    for key, receipts in routed.items():
        for receipt in receipts:
            assert receipt == json.loads(payloads[receipt["source_rank"]])[key]
    assert not any(isinstance(value, torch.Tensor) for value in routed.values())


@pytest.mark.parametrize(
    "fault",
    [
        "missing_owner",
        "wrong_slot",
        "missing_target",
        "extra_target",
        "version",
        "generation",
        "plan_hash",
        "writes",
        "duplicate_write",
        "source_rank",
        "malformed",
    ],
)
def test_router_refuses_incomplete_or_stale_owner_data(fault):
    plans = fixture_plans()
    payloads = fixture_receipts(plans)
    local = json.loads(payloads[0])
    key = next(iter(local))
    if fault == "missing_owner":
        payloads.pop()
    elif fault == "wrong_slot":
        payloads[0], payloads[2] = payloads[2], payloads[0]
    elif fault == "missing_target":
        local.pop(key)
    elif fault == "extra_target":
        local["foreign-0"] = local[key]
    elif fault == "duplicate_write":
        local[key]["writes"] *= 2
    elif fault == "malformed":
        local[key] = None
    else:
        local[key][fault] = [] if fault == "writes" else "stale"
    if fault not in ("missing_owner", "wrong_slot"):
        payloads[0] = json.dumps(local)
    with pytest.raises(RuntimeError):
        route_receipts(
            payloads,
            {key: ReceiptValidator(p) for key, p in plans.items()},
            {key: f"generation:{key}" for key in plans},
            4,
        )


def test_duplicate_owner_not_hidden_by_dictionary_index():
    with pytest.raises(RuntimeError, match="Duplicate"):
        index_receipts([dict(source_rank=2), dict(source_rank=2)])


def test_cold_file_receipts_reject_unexpected_rpc(tmp_path):
    plan = dict(writes=[dict(source_rank=0, id="x")], receipt_delivery=RPC_RECEIPT_DELIVERY)
    state = dict(
        plan=plan,
        receipt_validator=ReceiptValidator(plan),
        generation="G",
        version=3,
        initial_version=3,
        receipt_tp_size=2,
    )
    value = dict(source_rank=0, generation="G", version=3, plan_hash=fingerprint(plan), writes=["x"])
    (tmp_path / "0-0.source-0.done.json").write_text(json.dumps(value))
    assert read_commit_receipts(state, {}, tmp_path, "0-0") == {0: value}
    with pytest.raises(RuntimeError, match="Unexpected"):
        read_commit_receipts(state, dict(source_receipts={}), tmp_path, "0-0")


@pytest.mark.parametrize("fault", [None, "missing", "target", "duplicate", "version", "generation", "plan_hash"])
def test_warm_receiver_requires_fresh_rpc_even_if_valid_files_exist(tmp_path, fault):
    plans = fixture_plans()
    routed = route_receipts(
        fixture_receipts(plans),
        {k: ReceiptValidator(p) for k, p in plans.items()},
        {k: f"generation:{k}" for k in plans},
        4,
    )
    key = "0-0"
    state = dict(
        plan=plans[key],
        receipt_validator=ReceiptValidator(plans[key]),
        generation=f"generation:{key}",
        version=4,
        initial_version=3,
        receipt_tp_size=2,
    )
    command = dict(engine="0", source_receipts={k: v for k, v in routed.items() if k.startswith("0-")})
    for receipt in routed[key]:
        (tmp_path / f"{key}.source-{receipt['source_rank']}.done.json").write_text(json.dumps(receipt))
    if fault == "missing":
        command.pop("source_receipts")
    elif fault == "target":
        command["source_receipts"]["foreign-0"] = command["source_receipts"].pop("0-1")
    elif fault == "duplicate":
        command["source_receipts"][key].append(copy.deepcopy(command["source_receipts"][key][0]))
    elif fault:
        command["source_receipts"][key][0][fault] = "stale"
    if fault:
        with pytest.raises(RuntimeError):
            read_commit_receipts(state, command, tmp_path, key)
    else:
        assert read_commit_receipts(state, command, tmp_path, key) == index_receipts(routed[key])


def _gloo_worker(rank, rendezvous, directory):
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=8, timeout=timedelta(seconds=60))
    try:
        # Also exercise a subgroup whose group root is not global rank zero.
        for ranks in (list(range(8)), [1, 3, 5, 7]):
            group = dist.new_group(ranks, backend="gloo", timeout=timedelta(seconds=60))
            if rank not in ranks:
                continue
            local_rank = dist.get_rank(group)
            plans = fixture_plans(len(ranks))
            direct = fixture_sender(local_rank, plans, group, len(ranks))
            sync = direct.sync
            for version, fault in ((3, None), (4, None), (5, "serialize"), (6, "route"), (7, None)):
                path = Path(directory) / f"{len(ranks)}-{version}"
                sync._phase(lambda: path.mkdir(exist_ok=True) if local_rank == 0 else None)
                direct.rpc_receipts_active = version != 3
                direct.receipt_version = version
                direct.local_rpc_receipts = direct.gathered_rpc_receipts = direct.routed_rpc_receipts = None
                direct.hashes = {key: fingerprint(plan) for key, plan in plans.items()}
                if fault == "serialize" and local_rank == 0:
                    direct.hashes[next(iter(plans))] = object()
                if fault == "serialize":
                    with pytest.raises(RuntimeError, match="JSON serializable"):
                        sync._phase(lambda: direct.finish(path, version))
                    assert direct.gathered_rpc_receipts is None if local_rank == 0 else True
                else:
                    sync._phase(lambda: direct.finish(path, version))
                    if direct.rpc_receipts_active:
                        if fault == "route" and local_rank == 1:
                            # Intentionally corrupt only actual serialized owner data.
                            local = json.loads(direct.local_rpc_receipts)
                            next(iter(local.values()))["version"] -= 1
                            direct.local_rpc_receipts = json.dumps(local)
                        sync._phase(direct.gather_receipts)
                        if fault == "route":
                            with pytest.raises(RuntimeError, match="stale or incomplete"):
                                sync._phase(direct.route_receipts)
                        else:
                            sync._phase(direct.route_receipts)
                            if local_rank == 0:
                                assert not list(path.glob("*.source-*.done.json"))
                                for engine in range(4):
                                    command = dict(
                                        engine=str(engine), source_receipts=direct.commit_receipts(engine, version)
                                    )
                                    assert set(command["source_receipts"]) == {f"{engine}-0", f"{engine}-1"}
                                    for key in command["source_receipts"]:
                                        state = dict(
                                            plan=plans[key],
                                            generation=f"generation:{key}",
                                            version=version,
                                            initial_version=3,
                                            receipt_tp_size=2,
                                            receipt_validator=ReceiptValidator(plans[key]),
                                        )
                                        read_commit_receipts(state, command, path, key)
                                with pytest.raises(RuntimeError, match="this version"):
                                    direct.commit_receipts(0, version - 1)
                marker = torch.ones(1, dtype=torch.int32)
                dist.all_reduce(marker, group=group)
                assert marker.item() == len(ranks)
    finally:
        dist.destroy_process_group()


def test_eight_rank_gloo_fresh_versions_errors_and_nonzero_global_root(tmp_path):
    mp.spawn(_gloo_worker, args=((tmp_path / "gloo-init").as_uri(), str(tmp_path)), nprocs=8, join=True)
