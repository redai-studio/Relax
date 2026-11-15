# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Reconstruct resident replica outputs after their NIXL chunks are DONE."""

from datetime import timedelta

import torch
import torch.distributed as dist

from relax.distributed.checkpoint_service.backends.shard_p2p.common import validate_worker_identity, worker_identity
from relax.distributed.checkpoint_service.backends.shard_p2p.plan import fingerprint
from relax.distributed.checkpoint_service.backends.shard_p2p.remote.plan import ReceiptValidator
from relax.distributed.checkpoint_service.backends.shard_p2p.remote.rpc_receipts import index_receipts
from relax.distributed.checkpoint_service.backends.shard_p2p.replicas import partition_writes, wire_writes
from relax.utils.distributed_utils import init_process_group


def replica_parameters(parameters: dict) -> list[str]:
    """Aliases and overlapping native views retain their original transfer."""
    spans = {
        name: (value.data_ptr(), value.data_ptr() + value.numel() * value.element_size())
        for name, value in parameters.items()
    }
    return sorted(
        name
        for name, value in parameters.items()
        if value.is_cuda
        and value.dtype == torch.bfloat16
        and value.is_contiguous()
        and not any(
            other != name and max(begin, spans[name][0]) < min(end, spans[name][1])
            for other, (begin, end) in spans.items()
        )
    )


def retire_replica_group(state: dict) -> None:
    replica = state.pop("replicas", None)
    if replica is not None:
        dist.destroy_process_group(replica["group"])
    state.pop("replica_store", None)
    state.pop("replica_config_hash", None)


def control_replica(model: torch.nn.Module, command: dict, parameters: dict) -> dict | None:
    state = model._relax_expert_ipc
    if (
        not state.get("persistent_targets")
        or state["failed"]
        or any(command.get(key) != state[key] for key in ("session", "engine", "tp_rank"))
        or command["version"] != state["version"] + 1
    ):
        raise RuntimeError("Replica command has no matching resident transfer session")
    rank = state["engine"]
    if command["op"] in ("replica_bootstrap", "replica_configure"):
        config = command["config"]
        members = config["members"]
        size = len(members)
        for member in members:
            validate_worker_identity(member)
        if (
            size < 2
            or not 0 <= rank < size
            or state.get("epoch") is not None
            or len({(member["hostname"], member["boot_id"]) for member in members}) != 1
            or len({member["pid"] for member in members}) != size
            or len({member["gpu_uuid"] for member in members}) != size
            or members[rank] != state["delivery"]["receiver"]
            or worker_identity(members[rank]["generation"], torch.device("cuda", torch.cuda.current_device()))
            != members[rank]
            or config["logical_plan_hash"] != state["plan_hash"]
            or not set(config["parameters"]) <= set(replica_parameters(parameters))
            or config["layouts"] != {name: dict(nbytes=value.numel() * 2) for name, value in parameters.items()}
            or any(
                name not in config["parameters"] or list(parameters[name].shape) != shape or len(shape) != 2
                for name, shape in config.get("transpose_shapes", {}).items()
            )
            or state["delivery"]["routes"]["cuda_ipc"]
        ):
            raise RuntimeError("Replica configuration differs from actual native workers or allocations")
        digest = fingerprint(config)
        if command["op"] == "replica_bootstrap":
            if "replica_config_hash" in state:
                raise RuntimeError("Replica bootstrap is already bound")
            state["replica_config_hash"] = digest
            bootstrap = None
            if rank == 0:
                store = dist.TCPStore(
                    "127.0.0.1", 0, size, True, timeout=timedelta(seconds=120), wait_for_workers=False
                )
                state["replica_store"] = store
                bootstrap = dict(
                    address="127.0.0.1",
                    port=store.port,
                    world_size=size,
                    group_name=f"relax-replica-{state['session']}-{state['tp_rank']}",
                )
            return dict(bootstrap=bootstrap)
        bootstrap = command["bootstrap"]
        if (
            state.get("replica_config_hash") != digest
            or "replicas" in state
            or bootstrap["address"] != "127.0.0.1"
            or type(bootstrap["port"]) is not int
            or not 0 < bootstrap["port"] < 65536
            or bootstrap["world_size"] != size
            or bootstrap["group_name"] != f"relax-replica-{state['session']}-{state['tp_rank']}"
        ):
            raise RuntimeError("Replica group bootstrap changed")
        store = state.get("replica_store")
        if store is None:
            store = dist.TCPStore(bootstrap["address"], bootstrap["port"], size, False, timeout=timedelta(seconds=120))
        group = init_process_group(
            "nccl",
            store=dist.PrefixStore(bootstrap["group_name"], store),
            world_size=size,
            rank=rank,
            group_name=bootstrap["group_name"],
            timeout=timedelta(seconds=120),
        )
        state["replicas"] = dict(group=group, store=store, config=config, digest=digest)
        dist.barrier(group=group, device_ids=[torch.cuda.current_device()])
        partial = partition_writes(
            wire_writes(state["writes"], config.get("transpose_shapes", {})),
            config["layouts"],
            config["parameters"],
            rank,
            size,
        )
        state["remote_receipt_validator"] = ReceiptValidator(dict(writes=partial))
        return None
    replica, epoch = state["replicas"], state["epoch"]
    if epoch.get("ready") or command.get("epoch_generation") != epoch["generation"]:
        raise RuntimeError("Replica collective has no active allocation epoch")
    if command["op"] == "replica_ready":
        if "replica_ready" in epoch:
            raise RuntimeError("Replica epoch was already armed")
        receipts = command["source_receipts"]
        state["remote_receipt_validator"].validate(index_receipts(receipts), epoch["generation"], command["version"])
        names, size = replica["config"]["parameters"], len(replica["config"]["members"])
        views = []
        for name in names:
            output = parameters[name].detach().view(-1)
            width = output.numel() // size
            if output.numel() != width * size or width == 0:
                raise RuntimeError("Replica input no longer aliases an equal native output chunk")
            views.append((output, output.narrow(0, rank * width, width)))
        torch.cuda.synchronize()
        ready = dict(
            target=f"{rank}-{state['tp_rank']}",
            session=state["session"],
            version=command["version"],
            generation=epoch["generation"],
            config_hash=replica["digest"],
            receipt_hash=fingerprint(receipts),
        )
        epoch.update(replica_views=views, replica_ready=ready)
        return ready
    ready = command["ready"]
    if (
        command["op"] != "replica_gather"
        or len(ready) != len(replica["config"]["members"])
        or ready[rank] != epoch.get("replica_ready")
        or epoch.get("replica_complete")
        or any(
            value["target"] != f"{index}-{state['tp_rank']}"
            or value["session"] != state["session"]
            or value["version"] != command["version"]
            or value["config_hash"] != replica["digest"]
            for index, value in enumerate(ready)
        )
    ):
        raise RuntimeError("Replica gather omitted a ready participant or changed its version")
    for output, local in epoch["replica_views"]:
        dist.all_gather_into_tensor(output, local, group=replica["group"])
    for name, (rows, columns) in replica["config"].get("transpose_shapes", {}).items():
        output = parameters[name].detach()
        output.copy_(output.view(columns, rows).t().contiguous())
    torch.cuda.synchronize()
    epoch["replica_complete"] = fingerprint(ready)
    epoch.pop("replica_views")
    return None


def validate_replica_completion(state: dict, receipts: list[dict]) -> None:
    if "replicas" in state:
        epoch = state["epoch"]
        if not epoch.get("replica_complete") or epoch["replica_ready"]["receipt_hash"] != fingerprint(receipts):
            raise RuntimeError("Replica output has not completed its matching GPU collective")
