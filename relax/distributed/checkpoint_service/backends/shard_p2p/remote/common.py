# Copyright (c) 2026 Relax Authors. All Rights Reserved.
import base64
import hashlib
import io
import json
import os
import pickle
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch


REMOTE_SHARD_LOADER = "relax.backends.sglang.p2p_weight_loader.control"
NATIVE_LEASE_PROTOCOL = "relax.p2p-native-weight-lease.v1"


def serialize_control(command: dict, empty: torch.Tensor) -> str:
    """Encode zero CPU storage with a local, unpatched CPU tensor reducer."""
    from multiprocessing.reduction import ForkingPickler

    from torch.multiprocessing import reductions

    if empty.device.type != "cpu" or empty.numel() != 0 or empty.untyped_storage().nbytes() != 0:
        raise ValueError("Control serialization requires empty CPU storage")
    buffer = io.BytesIO()
    pickler = ForkingPickler(buffer, pickle.HIGHEST_PROTOCOL)
    # SG's process-wide reducer assumes CUDA. Standard Tensor pickle instead
    # embeds Megatron's storage loader, which SG's SafeUnpickler rejects. Only
    # this zero-storage control pickler selects Torch's native CPU reducer.
    pickler.dispatch_table = dict(pickler.dispatch_table)
    pickler.dispatch_table[torch.Tensor] = getattr(reductions, "_reduce_tensor_original", reductions.reduce_tensor)
    pickler.dump([(json.dumps(command), empty)])
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def weight_update_modes(topology: dict) -> dict:
    """Require explicit, generation-bound capabilities from each real
    engine."""
    if not topology:
        raise NotImplementedError("P2P requires registered receiver capabilities")
    modes = {}
    for engine, info in topology.items():
        if not isinstance(info, dict):
            raise NotImplementedError("P2P requires generation-bound receiver metadata")
        metadata = info.get("metadata") or {}
        mode = metadata.get("weight_update_session")
        if type(mode) is not bool or metadata.get("native_weight_lease_protocol") != NATIVE_LEASE_PROTOCOL:
            raise NotImplementedError("P2P receiver did not advertise a checked native allocation lease")
        modes[engine] = mode
    return modes


def identity(manifest: dict) -> dict:
    return {name: manifest[name] for name in ("generation", "hostname", "pid", "tp_rank", "gpu_id")}


def enabled(args: Any) -> bool:
    return bool(getattr(args, "use_shard_p2p_weight_sync", False))


def atomic_json(path: Path, data: Any) -> None:
    temporary = path.with_suffix(".tmp")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as stream:
        json.dump(data, stream)
    os.replace(temporary, path)


def read_manifest(path: Path, version: int, previous: dict | None = None) -> dict:
    """Resolve a versioned receipt against a previously validated static
    layout."""
    return resolve_manifest(json.loads(path.read_text()), version, previous)


def resolve_manifest(current: dict, version: int, previous: dict | None = None) -> dict:
    """Validate the original receiver metadata, whether read or broadcast."""
    if current.get("version") != version:
        raise RuntimeError("Stale P2P manifest version")
    if previous is None:
        if current.get("cached_layout"):
            raise RuntimeError("P2P manifest needs an initial full layout")
        return current
    identity = ("generation", "hostname", "pid", "tp_rank", "gpu_id")
    dynamic = {"version", "cached_layout"}
    if (
        current.get("cached_layout") is not True
        or set(current) != set(identity) | dynamic
        or version != previous["version"] + 1
        or any(current[key] != previous[key] for key in identity)
    ):
        raise RuntimeError("P2P cached manifest identity or layout changed")
    return previous | current


def layout_of(tensor: torch.Tensor) -> dict:
    if not tensor.is_contiguous():
        raise NotImplementedError("P2P currently requires contiguous target parameters")
    return dict(
        shape=list(tensor.shape),
        stride=list(tensor.stride()),
        dtype=str(tensor.dtype),
        nbytes=tensor.numel() * tensor.element_size(),
        address=tensor.data_ptr(),
    )


def digest(tensor: torch.Tensor) -> str:
    # Full CPU copies are only used by the initial parity audit, never steady sync.
    value = tensor.detach().contiguous().view(torch.uint8).cpu().numpy()
    return hashlib.sha256(memoryview(value)).hexdigest()


def create_agent() -> Any:
    import uuid

    from nixl._api import nixl_agent, nixl_agent_config, nixl_thread_sync_t

    agent = nixl_agent(
        f"relax-p2p-{uuid.uuid4()}",
        nixl_agent_config(backends=["UCX"], sync_mode=nixl_thread_sync_t.NIXL_THREAD_SYNC_STRICT),
    )
    if "UCX" not in agent.get_plugin_list():
        raise RuntimeError("NIXL UCX plugin is required for P2P")
    return agent


def named_sources(sync: Any) -> Iterator[tuple[str, torch.Tensor]]:
    """Read the current source parameters and persistent buffers for this
    phase."""
    backend = sync.backend
    return backend._named_params_and_buffers()
