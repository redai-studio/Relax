# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Partition expert delivery across replicas and coordinate their local NCCL
AllGather."""

import json
import time
import uuid
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist

from relax.utils.distributed_utils import init_process_group

from .common import atomic_json, identity, layout_of
from .plan import COLLECTIVE_DELIVERY, ReceiptValidator, collective_peer_plans, fingerprint
from .rpc_receipts import index_receipts, read_commit_receipts


# Moe collective.


def configure(state: dict, groups: list[list[str]], key: str, tp_rank: int) -> dict | None:
    """Bind group membership at receiver prepare, before publishing a
    manifest."""
    if (
        not isinstance(groups, list)
        or not groups
        or any(not isinstance(group, list) or len(group) < 2 for group in groups)
    ):
        raise RuntimeError("Missing collective receiver groups")
    flat = [peer for group in groups for peer in group]
    if len(set(flat)) != len(flat) or key not in flat:
        raise RuntimeError("Collective groups must partition distinct receiver keys")
    keys = next(group for group in groups if key in group)
    if any(peer.rsplit("-", 1)[1] != str(tp_rank) for peer in keys):
        raise RuntimeError("Collective participants must have the same TP rank")
    if "collective_groups" in state and state["collective_groups"] != groups:
        raise RuntimeError("Collective group membership changed")
    if state.get("collective_failed"):
        raise RuntimeError("A failed collective session cannot be reused")
    state.update(collective_groups=groups, collective_keys=keys)
    return coordinator_bootstrap(state, len(keys)) if key == keys[0] else None


def cache_plan(state: dict) -> None:
    """Cache whole-plan validators, not versioned completion evidence."""
    peers = collective_peer_plans(state["plan"])
    state["collective_validators"] = {key: ReceiptValidator(plan) for key, plan in peers.items()}
    state["collective_plan_hashes"] = {key: fingerprint(plan) for key, plan in peers.items()}
    state["collective_schema_hash"] = fingerprint(state["plan"]["collective"])


def _receipt_hash(receipts: dict[int, dict]) -> str:
    return fingerprint([receipts[rank] for rank in sorted(receipts)])


def _own(state: dict, key: str, peers: dict) -> None:
    expected = peers[key]
    if expected["identity"] != identity(state) or expected["layout_hash"] != fingerprint(
        {name: state["layouts"][name] for name in state["direct_params"]}
    ):
        raise RuntimeError("Collective sender metadata does not match the actual receiver")


def _wait(store: dist.Store, names: list[str], failures: list[str], deadline: float) -> dict[str, dict]:
    """Check errors while waiting; no rank can synthesize another rank's
    ACK."""
    while True:
        if time.monotonic() >= deadline:
            raise TimeoutError("Collective peer did not publish current-version readiness or completion")
        for name in failures:
            if store.check([name]):
                raise RuntimeError("Collective peer failed: " + store.get(name).decode())
        if store.check(names):
            values = {name: json.loads(store.get(name)) for name in names}
            if time.monotonic() >= deadline:
                raise TimeoutError("Collective metadata arrived after the operation deadline")
            return values
        time.sleep(0.005)


def validate_ready(state: dict, values: dict[str, dict], peers: dict) -> None:
    keys, version = state["collective_keys"], state["version"]
    if set(values) != set(keys):
        raise RuntimeError("Collective ready set is incomplete")
    for key in keys:
        ready = values[key]
        if (
            ready["key"] != key
            or ready["version"] != version
            or ready["identity"] != peers[key]["identity"]
            or ready["layout_hash"] != peers[key]["layout_hash"]
            or ready["plan_hash"] != state["collective_plan_hashes"][key]
            or ready["schema_hash"] != state["collective_schema_hash"]
            or ready["gpu_ready"] is not True
        ):
            raise RuntimeError("Stale or mismatched collective readiness")
        state["collective_validators"][key].validate(
            index_receipts(ready["source_receipts"]), peers[key]["identity"]["generation"], version
        )


def validate_acks(state: dict, values: dict[str, dict], peers: dict, ready_hash: str) -> None:
    if set(values) != set(state["collective_keys"]):
        raise RuntimeError("Collective GPU completion set is incomplete")
    for key in state["collective_keys"]:
        ack = values[key]
        if (
            ack["key"] != key
            or ack["version"] != state["version"]
            or ack["identity"] != peers[key]["identity"]
            or ack["plan_hash"] != state["collective_plan_hashes"][key]
            or ack["ready_hash"] != ready_hash
            or ack["gpu_completed"] is not True
        ):
            raise RuntimeError("Stale or mismatched collective GPU completion")


def execute(state: dict, command: dict, directory: Path, key: str) -> None:
    """Only enter NCCL after every same-TP participant validates all originals.

    Each TP group may progress independently. The sender waits for every engine
    before commit, so failure of either group prevents global publication. This
    is fail-closed publication, not rollback of partially written GPU storage.
    """
    if state.get("collective_failed"):
        raise RuntimeError("A failed collective session cannot be reused")
    store = None
    version = state["version"]
    prefix = f"v{version}/"
    started = time.monotonic()
    deadline, prepare_deadline = started + 120, started + 60
    try:
        plan, keys = state["plan"], state["collective_keys"]
        if plan["delivery"] != COLLECTIVE_DELIVERY or plan["collective"]["keys"] != keys:
            raise RuntimeError("Unexpected collective delivery plan")
        if state.get("collective_last_version", state["initial_version"] - 1) + 1 != version:
            raise RuntimeError("Duplicate or discontinuous collective version")
        all_keys = {peer for group in state["collective_groups"] for peer in group}
        peers, bootstraps = command["collective_peers"], command["collective_bootstraps"]
        if set(peers) != all_keys or set(bootstraps) != {group[0] for group in state["collective_groups"]}:
            raise RuntimeError("Incomplete collective peer or coordinator metadata")
        _own(state, key, peers)
        bootstrap = bootstraps[keys[0]]
        initialize_group(
            state, bootstrap, {peer: peers[peer]["identity"] for peer in keys}, keys, key, timeout_seconds=30
        )
        if "collective_control_store" not in state:
            # A separate client keeps control/error operations bounded without
            # changing the timeout of the NCCL rendezvous connection.
            state["collective_control_store"] = dist.TCPStore(
                bootstrap["address"], bootstrap["port"], len(keys), False, timeout=timedelta(seconds=5)
            )
        store = dist.PrefixStore(bootstrap["group_name"] + "/control", state["collective_control_store"])
        failures = [prefix + "failed/" + peer for peer in keys]
        receipts = read_commit_receipts(state, command, directory, key)
        views = prepare_views(state, keys.index(key), len(keys))
        torch.cuda.synchronize()
        ready = dict(
            key=key,
            version=version,
            identity=identity(state),
            layout_hash=peers[key]["layout_hash"],
            plan_hash=state["validated_plan_hash"],
            schema_hash=state["collective_schema_hash"],
            source_receipts=[receipts[rank] for rank in sorted(receipts)],
            gpu_ready=True,
        )
        store.set(prefix + "ready/" + key, json.dumps(ready, allow_nan=False))
        raw = _wait(store, [prefix + "ready/" + peer for peer in keys], failures, prepare_deadline)
        ready_set = {peer: raw[prefix + "ready/" + peer] for peer in keys}
        validate_ready(state, ready_set, peers)
        if ready_set[key] != ready:
            raise RuntimeError("Collective readiness no longer matches this receiver's original")
        ready_hash = fingerprint(ready_set)
        # Every participant validates the whole set before voting to enter.
        store.set(prefix + "armed/" + key, json.dumps(dict(key=key, version=version, ready_hash=ready_hash)))
        armed = _wait(store, [prefix + "armed/" + peer for peer in keys], failures, prepare_deadline)
        if any(
            armed[prefix + "armed/" + peer] != dict(key=peer, version=version, ready_hash=ready_hash) for peer in keys
        ):
            raise RuntimeError("Collective participants armed different readiness sets")
        if time.monotonic() >= prepare_deadline:
            raise TimeoutError("Collective preparation exhausted its bounded execution budget")
        allgather_parameters(views, state["collective_group"])
        ack = dict(
            key=key,
            version=version,
            identity=identity(state),
            plan_hash=state["validated_plan_hash"],
            ready_hash=ready_hash,
            gpu_completed=True,
        )
        store.set(prefix + "ack/" + key, json.dumps(ack))
        raw = _wait(store, [prefix + "ack/" + peer for peer in keys], failures, deadline)
        acknowledgements = {peer: raw[prefix + "ack/" + peer] for peer in keys}
        validate_acks(state, acknowledgements, peers, ready_hash)
        if acknowledgements[key] != ack:
            raise RuntimeError("Collective completion no longer matches this receiver's original")
        if time.monotonic() >= deadline:
            raise TimeoutError("Collective completion exceeded the operation deadline")
        completion = dict(
            version=version, ready_hash=ready_hash, readiness=ready_set, acknowledgements=acknowledgements
        )
        if state["retain_source_receipts"]:
            atomic_json(directory / f"{key}.collective-ready.json", ready_set)
        state.update(
            collective_last_version=version,
            collective_receipt_hash=_receipt_hash(receipts),
            collective_completion=completion,
            collective_peers=peers,
        )
        # A globally committed previous update precedes the next one. Remove
        # only our own old records; never alter a peer's evidence.
        if version > state["initial_version"] + 1:
            for phase in ("ready", "armed", "ack", "failed"):
                store.delete_key(f"v{version - 2}/{phase}/{key}")
    except Exception as error:
        state["collective_failed"] = repr(error)
        if store is not None:
            try:
                store.set(prefix + "failed/" + key, repr(error))
            except Exception:
                pass  # Original failure remains authoritative; never retry NCCL.
        raise


def verify_commit(state: dict, receipts: dict[int, dict]) -> None:
    completion = state.get("collective_completion")
    if (
        state.get("collective_failed")
        or state.get("collective_last_version") != state["version"]
        or completion is None
        or completion["version"] != state["version"]
        or _receipt_hash(receipts) != state["collective_receipt_hash"]
    ):
        raise RuntimeError("Collective commit requires this version's actual GPU completion and source receipts")
    validate_acks(state, completion["acknowledgements"], state["collective_peers"], completion["ready_hash"])
    validate_ready(state, completion["readiness"], state["collective_peers"])
    if fingerprint(completion["readiness"]) != completion["ready_hash"]:
        raise RuntimeError("Collective completion is not bound to its original readiness set")


# Moe collective nccl.


def coordinator_bootstrap(state: dict, world_size: int) -> dict:
    """Create a nonblocking loopback store without changing the default PG."""
    if type(world_size) is not int or world_size < 2:
        raise ValueError("Local AllGather requires multiple replica participants")
    if "collective_bootstrap" not in state:
        store = dist.TCPStore("127.0.0.1", 0, world_size, True, timeout=timedelta(seconds=120), wait_for_workers=False)
        state["collective_store"] = store
        state["collective_bootstrap"] = dict(
            address="127.0.0.1",
            port=store.port,
            world_size=world_size,
            group_name="moe-allgather-" + str(uuid.uuid4()),
        )
    if state["collective_bootstrap"]["world_size"] != world_size:
        raise RuntimeError("Local AllGather participant count changed")
    return state["collective_bootstrap"]


def prepare_views(state: dict, rank: int, world_size: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Validate every output and its exact in-place input chunk before NCCL.

    Only detach/narrow views are allowed here: no clone, contiguous copy or
    output scratch. In particular, storage_offset on the original native
    parameter is preserved rather than assuming a zero-offset allocation.
    """
    if type(world_size) is not int or world_size < 2 or type(rank) is not int or rank not in range(world_size):
        raise ValueError("Invalid local AllGather rank or participant count")
    names = sorted(state["direct_params"])
    if not names:
        raise RuntimeError("Local AllGather has no native parameters")
    views = []
    devices = set()
    for name in names:
        parameter = state["params"][name]
        if (
            not parameter.is_cuda
            or parameter.ndim < 2
            or not parameter.is_contiguous()
            or parameter.shape[0] % world_size
            or layout_of(parameter) != state["layouts"][name]
        ):
            raise RuntimeError(f"Invalid local AllGather output storage: {name}")
        output = parameter.detach()
        rows = output.shape[0] // world_size
        if rows == 0:
            raise RuntimeError("Local AllGather input chunk is empty")
        local = output.narrow(0, rank * rows, rows)
        offset = rank * local.numel() * local.element_size()
        if (
            not local.is_contiguous()
            or local.data_ptr() != output.data_ptr() + offset
            or local.untyped_storage().data_ptr() != output.untyped_storage().data_ptr()
            or local.numel() * world_size != output.numel()
        ):
            raise RuntimeError("Local AllGather input must alias its exact output chunk")
        devices.add(output.device)
        views.append((output, local))
    if len(devices) != 1 or next(iter(devices)).index != torch.cuda.current_device():
        raise RuntimeError("Local AllGather parameters must use the current CUDA device")
    return views


def initialize_group(
    state: dict, bootstrap: dict, identities: dict, keys: list[str], key: str, *, timeout_seconds: int = 120
) -> None:
    """Cache a private group bound to ordered, immutable receiver
    identities."""
    world_size = len(keys)
    if (
        type(timeout_seconds) is not int
        or not 1 <= timeout_seconds <= 120
        or world_size < 2
        or len(set(keys)) != world_size
        or set(identities) != set(keys)
        or key not in keys
        or bootstrap["address"] != "127.0.0.1"
        or type(bootstrap["port"]) is not int
        or not 0 < bootstrap["port"] < 65536
        or bootstrap["world_size"] != world_size
        or not bootstrap["group_name"].startswith("moe-allgather-")
    ):
        raise RuntimeError("Invalid local AllGather bootstrap or participant set")
    peers = list(identities.values())
    if (
        len({peer["hostname"] for peer in peers}) != 1
        or len({peer["tp_rank"] for peer in peers}) != 1
        or len({peer["gpu_id"] for peer in peers}) != world_size
        or len({peer["pid"] for peer in peers}) != world_size
        or any(not peer["generation"] for peer in peers)
    ):
        raise RuntimeError("Local AllGather requires distinct same-host, same-TP receivers")
    own = identity(state)
    if identities[key] != own:
        raise RuntimeError("Local AllGather participant identity does not match this receiver")
    rank = keys.index(key)
    signature = fingerprint(
        dict(bootstrap=bootstrap, rank=rank, identities=identities, keys=keys, timeout_seconds=timeout_seconds)
    )
    if "collective_bootstrap" in state and state["collective_bootstrap"] != bootstrap:
        raise RuntimeError("Local AllGather coordinator store identity changed")
    if "collective_group" in state:
        if signature != state["collective_signature"]:
            raise RuntimeError("Local AllGather group identity changed")
        return
    store = state.get("collective_store")
    if store is None:
        store = dist.TCPStore(
            bootstrap["address"], bootstrap["port"], world_size, False, timeout=timedelta(seconds=timeout_seconds)
        )
    else:
        store.set_timeout(timedelta(seconds=timeout_seconds))
    group = init_process_group(
        "nccl",
        store=dist.PrefixStore(bootstrap["group_name"], store),
        world_size=world_size,
        rank=rank,
        group_name=bootstrap["group_name"],
        timeout=timedelta(seconds=timeout_seconds),
    )
    state.update(collective_store=store, collective_group=group, collective_signature=signature)


def allgather_parameters(views: list[tuple[torch.Tensor, torch.Tensor]], group: dist.ProcessGroup) -> None:
    """Execute already-agreed views; return only after real GPU completion."""
    for output, local in views:
        dist.all_gather_into_tensor(output, local, group=group)
    torch.cuda.synchronize()
