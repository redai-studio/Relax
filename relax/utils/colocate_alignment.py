# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Check that colocated training ranks and rollout engines share physical GPUs.

Colocated weight sync hands each engine the CUDA-IPC handles gathered from the
training ranks ``[offset, offset + count)``, and engine worker ``k`` opens the
handle from rank ``offset + k``. That is only correct when rank ``offset + k``
runs on exactly the GPU worker ``k`` drives. Both sides here report what their
own process observes -- a device UUID where the accelerator exposes one -- so
the check compares observations instead of two re-derived index layouts.
"""

import os
from collections.abc import Mapping, Sequence

from relax.utils import device as device_utils


# (Ray node id, device identity) of one GPU.
DeviceIdentity = tuple[str, str]


class ColocateAlignmentError(RuntimeError):
    """Colocated training ranks and rollout engines are on different GPUs."""


def local_device_identity(device_index: int) -> str:
    """Name the physical device behind ``device_index`` in this process.

    A UUID names one device on every node and under any visible-devices
    remapping. An accelerator without one falls back to the index the visible-
    devices list maps ``device_index`` to, which only identifies a device when
    both sides see the same node-wide numbering.
    """
    properties = device_utils.get_torch_device_module().get_device_properties(device_index)
    uuid = getattr(properties, "uuid", None)
    if uuid is not None:
        return f"uuid:{uuid}"
    visible = [item.strip() for item in os.environ.get(device_utils.get_visible_devices_env_var(), "").split(",")]
    visible = [item for item in visible if item]
    return f"index:{visible[device_index] if device_index < len(visible) else device_index}"


def check_colocate_alignment(
    rank_devices: Sequence[DeviceIdentity],
    engine_identities: Sequence[Mapping],
    offsets: Sequence[int],
    counts: Sequence[int],
) -> tuple[list[str], list[int]]:
    """Return the alignment problems and the engines that could not be checked.

    ``rank_devices[r]`` is training rank ``r``'s device. ``engine_identities[i]``
    is ``{"node_id", "devices"}`` reported by colocated engine ``i``, whose
    ``devices`` list its workers' devices in worker order on its own node, or is
    ``None`` for an engine that does not know them (an external engine). A
    multi-node engine reports its own node only, so only those ranks are
    checked.

    The function is pure: every rank calls it with the same gathered inputs and
    reaches the same verdict, so all ranks fail together rather than leaving
    some blocked in a later collective.
    """
    problems: list[str] = []
    unverified: list[int] = []
    world = len(rank_devices)

    seen: dict[DeviceIdentity, int] = {}
    for rank, device in enumerate(rank_devices):
        device = tuple(device)
        if device in seen:
            problems.append(f"training ranks {seen[device]} and {rank} share device {device}")
        else:
            seen[device] = rank

    claimed: dict[int, int] = {}
    for index, (identity, offset, count) in enumerate(zip(engine_identities, offsets, counts, strict=True)):
        if offset < 0 or count <= 0 or offset + count > world:
            problems.append(f"engine {index} claims ranks [{offset}, {offset + count}) outside world size {world}")
            continue
        for rank in range(offset, offset + count):
            if rank in claimed:
                problems.append(f"engines {claimed[rank]} and {index} both claim training rank {rank}")
            claimed.setdefault(rank, index)
        devices = identity.get("devices")
        if devices is None:
            unverified.append(index)
            continue
        if not devices or len(devices) > count:
            problems.append(f"engine {index} reports {len(devices)} device(s) for {count} rank(s)")
            continue
        expected = [(identity.get("node_id", ""), device) for device in devices]
        actual = [tuple(rank_devices[rank]) for rank in range(offset, offset + len(devices))]
        if actual != expected:
            pairs = ", ".join(
                f"rank {offset + k}: {got} vs worker {k}: {want}"
                for k, (got, want) in enumerate(zip(actual, expected, strict=True))
                if got != want
            )
            problems.append(f"engine {index} workers are not on their training ranks' GPUs ({pairs})")
    return problems, unverified
