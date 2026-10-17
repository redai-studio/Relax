# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Unified placement planning for inference roles.

A planner resolves and validates GPU/bundle slices; it never creates or
destroys Ray resources. Each control domain owns exactly one planner instance,
and that instance holds the allocation ledger for every role in the domain, so
an allocation is always discoverable by whoever is allowed to release it. No
allocation state lives on the class: separate role processes must not discover
each other's resources through shared class state.

The ledger is keyed by a *stable* placement-group identity rather than by
``id()`` of a Python object, because a view crosses a Ray RPC boundary by value
and the receiving process therefore sees a distinct object for the same
placement group.

Allocations in the same phase may never share GPUs; different phases may
reuse one slice because the Manager runs them one at a time.

A view built from a probed :class:`PlacementBinding` also carries where every
bundle physically landed, and planning then checks the resolved slots against
that binding instead of trusting node boundaries inferred from offsets.
"""

from dataclasses import dataclass, field
from enum import Enum
from threading import Lock
from typing import Any, Iterable, Mapping, NamedTuple, Sequence


class PlacementOwner(str, Enum):
    """Who is responsible for the lifetime of the placement group itself."""

    MANAGER = "manager"
    CONTROLLER = "controller"
    EXTERNAL = "external"

    @property
    def removable(self) -> bool:
        """Only a placement group this control plane created may be removed."""
        return self is PlacementOwner.MANAGER


class PlacementConflictError(ValueError):
    """A request cannot be satisfied alongside the recorded allocations."""


def placement_group_key(placement_group: object) -> str:
    """Return a placement-group identity that survives serialization.

    Ray placement groups expose a hex ``id``, which is identical in every
    process that holds the group. Anything else (a test double, a plain label)
    falls back to a process-local key, which is only ever correct because such
    objects never travel between processes.
    """
    identifier = getattr(placement_group, "id", None)
    to_hex = getattr(identifier, "hex", None)
    if callable(to_hex):
        value = to_hex()
        if isinstance(value, str):
            return f"pg:{value}"
    if isinstance(identifier, (bytes, bytearray)):
        return f"pg:{identifier.hex()}"
    if isinstance(placement_group, (str, bytes, int)):
        return f"label:{placement_group!s}"
    return f"local:{id(placement_group):x}"


class PlacementBinding(NamedTuple):
    """A ready placement group and where each of its reordered bundles landed.

    Entry ``i`` of ``bundle_indices``, ``gpu_ids`` and ``node_ids`` describes
    logical slot ``i``. ``node_ids`` holds Ray node ids, which -- unlike node
    IPs or hostnames -- name one machine exactly once.
    """

    pg: Any
    bundle_indices: tuple[int, ...]
    gpu_ids: tuple[int, ...]
    node_ids: tuple[str, ...]
    # The device index an engine on that node addresses each GPU by, under the
    # visible devices it inherits; ``None`` when the probe did not report it.
    local_gpu_ids: tuple[int, ...] | None = None

    @classmethod
    def require(cls, value: Any) -> "PlacementBinding":
        """Return ``value`` if it is a probed binding, else raise."""
        if not isinstance(value, cls):
            raise TypeError(f"Expected a probed PlacementBinding, got {type(value).__name__}")
        return value


@dataclass(frozen=True)
class PlacementGroupView:
    """A reordered, node-grouped view of one placement group's bundles.

    ``node_ids`` and ``bundle_count`` are known only for a probed placement
    group; without them planning can check the logical layout alone.
    """

    bundle_indices: tuple[int, ...]
    gpu_ids: tuple[int, ...]
    owner: PlacementOwner
    identity: object = field(default_factory=object, repr=False, compare=False)
    node_ids: tuple[str, ...] | None = None
    bundle_count: int | None = None
    local_gpu_ids: tuple[int, ...] | None = None

    @classmethod
    def from_binding(cls, binding: PlacementBinding, owner: PlacementOwner) -> "PlacementGroupView":
        bundle_count = getattr(binding.pg, "bundle_count", None)
        return cls(
            tuple(binding.bundle_indices),
            tuple(binding.gpu_ids),
            owner,
            identity=binding.pg,
            node_ids=tuple(binding.node_ids),
            bundle_count=bundle_count if isinstance(bundle_count, int) else None,
            local_gpu_ids=None if binding.local_gpu_ids is None else tuple(binding.local_gpu_ids),
        )

    @property
    def key(self) -> str:
        return placement_group_key(self.identity)


@dataclass(frozen=True)
class PlacementRequest:
    group_id: str
    worker_type: str
    num_gpus: int
    num_gpus_per_engine: int
    num_gpus_per_node: int
    phase: str = "inference"
    bundle_offset: int | None = None
    active: bool = True


@dataclass(frozen=True)
class PlacementSlice:
    """A resolved slice: engine pools consume this instead of deriving
    offsets."""

    group_id: str
    worker_type: str
    phase: str
    owner: PlacementOwner
    reserved_offset: int
    reserved_size: int
    referenced_offsets: tuple[int, ...]
    bundle_indices: tuple[int, ...]
    gpu_ids: tuple[int, ...]
    placement_group_key: str = ""
    gpus_per_slot: int = 1
    active: bool = True


@dataclass(frozen=True)
class ModelPlacement:
    """Where one model's engines run, aggregated over its group slices.

    ``activation_group`` names the GPUs the model takes turns on (its placement
    group) and ``activation_phase`` the phase it holds them in; both are
    ``None`` for a model that never yields its GPUs to another role.
    """

    pg_owner: PlacementOwner
    bundle_indices: tuple[int, ...]
    gpu_ids: tuple[int, ...]
    activation_group: str | None
    activation_phase: str | None

    @classmethod
    def from_slices(cls, slices: Sequence[PlacementSlice], *, shared_phase: str) -> "ModelPlacement":
        """Aggregate a model's startup slices; ``shared_phase`` is the phase in
        which roles keep their GPUs instead of taking turns."""
        if not slices:
            raise ValueError("A model placement needs at least one slice")
        phases = {item.phase for item in slices}
        if len(phases) != 1:
            raise ValueError(f"A model must be placed in one phase, got {sorted(phases)}")
        phase = phases.pop()
        deferred = phase != shared_phase
        return cls(
            pg_owner=slices[0].owner,
            bundle_indices=tuple(index for item in slices for index in item.bundle_indices),
            gpu_ids=tuple(gpu for item in slices for gpu in item.gpu_ids),
            activation_group=slices[0].placement_group_key if deferred else None,
            activation_phase=phase if deferred else None,
        )


@dataclass(frozen=True)
class PlacementRelease:
    """The outcome of releasing ledger entries.

    ``remove_placement_group`` is the only sanctioned answer to "may I destroy
    this placement group": it is true only for a group this control plane
    created and whose last allocation was just released.
    """

    placement_group_key: str
    slices: tuple[PlacementSlice, ...] = ()
    remove_placement_group: bool = False


class PlacementPlanner:
    def __init__(self) -> None:
        self._lock = Lock()
        # {placement group key: {group_id: (request, resolved slice)}}
        self._allocations: dict[str, dict[str, tuple[PlacementRequest, PlacementSlice]]] = {}

    # ------------------------------------------------------------------
    # Planning.
    # ------------------------------------------------------------------
    def plan(
        self,
        requests: Sequence[PlacementRequest],
        placement_group: PlacementGroupView,
        *,
        dry_run: bool = False,
    ) -> tuple[PlacementSlice, ...]:
        """Resolve ``requests`` against ``placement_group`` and record them.

        Re-planning a ``group_id`` with an identical request replays the
        recorded slice, so a retried startup is idempotent; a different request
        for the same ``group_id`` is a conflict rather than a silent overwrite.
        ``dry_run`` validates a whole layout without reserving it, which is how
        a launcher checks a plan before it spawns anything.
        """
        self._validate_view(placement_group)
        key = placement_group.key
        with self._lock:
            recorded = dict(self._allocations.get(key, {}))
            resolved: list[PlacementSlice] = []
            for request in requests:
                previous = recorded.get(request.group_id)
                if previous is not None:
                    if previous[0] != request:
                        raise PlacementConflictError(
                            f"Placement request conflict for {request.group_id}: {previous[0]} was already allocated"
                        )
                    resolved.append(previous[1])
                    continue
                item = self._resolve(request, placement_group, recorded)
                recorded[request.group_id] = (request, item)
                resolved.append(item)
            if not dry_run:
                self._allocations[key] = recorded
            return tuple(resolved)

    def _resolve(
        self,
        request: PlacementRequest,
        placement_group: PlacementGroupView,
        recorded: Mapping[str, tuple[PlacementRequest, PlacementSlice]],
    ) -> PlacementSlice:
        capacity = len(placement_group.bundle_indices)
        slot_size = self._validate_parallel_layout(request, capacity)
        offset = request.bundle_offset
        if offset is None:
            # Auto-placement appends within the requesting phase only: other
            # phases deliberately reuse the same GPUs at a different time.
            offset = max(
                (
                    item.reserved_offset + item.reserved_size
                    for _, item in recorded.values()
                    if item.phase == request.phase
                ),
                default=0,
            )
        end = offset + request.num_gpus
        if offset < 0 or end > capacity:
            raise ValueError(
                f"Placement group is too small for {request.group_id}: "
                f"bundles [{offset}, {end}) exceed capacity {capacity}"
            )
        # An inactive request only reserves its region -- it runs no engine, so
        # it occupies no slot and has no parallel layout to honor.
        referenced = (
            tuple(offset + index * slot_size for index in range(request.num_gpus // slot_size))
            if request.active
            else ()
        )
        self._validate_node_boundaries(request, referenced, slot_size)
        if placement_group.node_ids is not None:
            self._validate_binding(request, placement_group, referenced, slot_size)
        for other_request, other in recorded.values():
            if other.phase != request.phase:
                continue
            if offset < other.reserved_offset + other.reserved_size and other.reserved_offset < end:
                raise PlacementConflictError(
                    f"Placement overlap in phase {request.phase!r}: {request.group_id} overlaps {other.group_id}"
                )
        return PlacementSlice(
            group_id=request.group_id,
            worker_type=request.worker_type,
            phase=request.phase,
            owner=placement_group.owner,
            reserved_offset=offset,
            reserved_size=request.num_gpus,
            referenced_offsets=referenced,
            bundle_indices=tuple(placement_group.bundle_indices[index] for index in referenced),
            gpu_ids=tuple(placement_group.gpu_ids[index] for index in referenced),
            placement_group_key=placement_group.key,
            gpus_per_slot=slot_size,
            active=request.active,
        )

    # ------------------------------------------------------------------
    # Validation.
    # ------------------------------------------------------------------
    @staticmethod
    def _validate_view(placement_group: PlacementGroupView) -> None:
        bundles = placement_group.bundle_indices
        if len(bundles) != len(placement_group.gpu_ids):
            raise ValueError(
                f"Placement group view is inconsistent: {len(bundles)} bundles, {len(placement_group.gpu_ids)} GPU ids"
            )
        local = placement_group.local_gpu_ids
        if local is not None and len(local) != len(bundles):
            raise ValueError(
                f"Placement group view is inconsistent: {len(bundles)} bundles, {len(local)} local device indices"
            )
        if len(set(bundles)) != len(bundles):
            raise ValueError(f"Placement group view repeats a bundle index: {bundles}")
        count = placement_group.bundle_count
        if count is not None and any(index < 0 or index >= count for index in bundles):
            raise ValueError(f"Placement group view names bundles outside [0, {count}): {bundles}")
        node_ids = placement_group.node_ids
        if node_ids is None:
            return
        if len(node_ids) != len(bundles):
            raise ValueError(f"Placement group view is inconsistent: {len(bundles)} bundles, {len(node_ids)} node ids")
        if not all(isinstance(node_id, str) and node_id for node_id in node_ids):
            raise ValueError(f"Placement group view has an unknown node id: {node_ids}")
        devices = [(node_id, int(gpu_id)) for node_id, gpu_id in zip(node_ids, placement_group.gpu_ids, strict=True)]
        if len(set(devices)) != len(devices):
            duplicated = sorted({device for device in devices if devices.count(device) > 1})
            raise ValueError(f"Placement group bundles map to the same physical GPU: {duplicated}")

    @staticmethod
    def _validate_binding(
        request: PlacementRequest,
        placement_group: PlacementGroupView,
        referenced: Sequence[int],
        slot_size: int,
    ) -> None:
        """Check resolved slots against where the bundles physically are.

        One slot is one engine actor on one node, and the engine drives the
        local devices ``base + k`` under the visible devices it inherits, so a
        slot must sit on one node over consecutive local device indices: then
        each device it drives is a GPU the slot reserved. Without those indices
        the physical IDs stand in for them. A multi-node engine must put each
        of its slots on a distinct node, or it would run fewer nodes than its
        TP/PP layout assumes.
        """
        node_ids = placement_group.node_ids
        assert node_ids is not None
        local = placement_group.local_gpu_ids
        gpu_ids = placement_group.gpu_ids if local is None else local
        for start in referenced:
            slot = range(start, start + slot_size)
            nodes = {node_ids[index] for index in slot}
            if len(nodes) != 1:
                raise ValueError(
                    f"An engine slot of {request.group_id} spans nodes {sorted(nodes)}: "
                    f"bundles [{start}, {start + slot_size})"
                )
            base = int(gpu_ids[start])
            actual = [int(gpu_ids[index]) for index in slot]
            if actual != list(range(base, base + slot_size)):
                physical = [int(placement_group.gpu_ids[index]) for index in slot]
                raise ValueError(
                    f"An engine slot of {request.group_id} needs consecutive GPUs from {base}, "
                    f"got {actual} (physical {physical}) on node {node_ids[start]}"
                )
        nodes_per_engine = max(1, request.num_gpus_per_engine // request.num_gpus_per_node)
        for first in range(0, len(referenced), nodes_per_engine):
            replica = referenced[first : first + nodes_per_engine]
            nodes = [node_ids[start] for start in replica]
            if len(set(nodes)) != nodes_per_engine:
                raise ValueError(
                    f"A {nodes_per_engine}-node engine of {request.group_id} lands on nodes {nodes}: "
                    f"bundles {list(replica)}"
                )

    @staticmethod
    def _validate_parallel_layout(request: PlacementRequest, capacity: int) -> int:
        """Return the per-slot GPU count implied by the model-parallel layout.

        One slot is one node actor, so a single-node engine is one slot and a
        multi-node engine is one slot per node.
        """
        if request.num_gpus <= 0 or request.num_gpus > capacity:
            raise ValueError(
                f"Invalid placement size for {request.group_id}: {request.num_gpus} GPU(s) against capacity {capacity}"
            )
        if request.num_gpus_per_engine <= 0 or request.num_gpus_per_node <= 0:
            raise ValueError(
                f"Invalid parallel layout for {request.group_id}: "
                f"{request.num_gpus_per_engine} GPU(s)/engine, {request.num_gpus_per_node} GPU(s)/node"
            )
        slot_size = min(request.num_gpus_per_engine, request.num_gpus_per_node)
        if not request.active:
            return slot_size
        if request.num_gpus_per_engine > request.num_gpus_per_node:
            if request.num_gpus_per_engine % request.num_gpus_per_node:
                raise ValueError(
                    f"A multi-node engine must use whole nodes for {request.group_id}: "
                    f"{request.num_gpus_per_engine} GPU(s)/engine, {request.num_gpus_per_node} GPU(s)/node"
                )
        if request.num_gpus % slot_size:
            raise ValueError(
                f"Placement size does not fit the parallel layout of {request.group_id}: "
                f"{request.num_gpus} GPU(s) is not a multiple of {slot_size}"
            )
        return slot_size

    @staticmethod
    def _validate_node_boundaries(request: PlacementRequest, referenced: Sequence[int], slot_size: int) -> None:
        """Reject layouts that straddle nodes.

        The view is ordered so that consecutive indices belong to the same
        node, so a node is identified by ``offset // num_gpus_per_node``.
        """
        per_node = request.num_gpus_per_node
        for start in referenced:
            if request.num_gpus_per_engine > per_node:
                if start % per_node:
                    raise ValueError(
                        f"A multi-node engine must start on a node boundary for {request.group_id}: "
                        f"bundle {start} with {per_node} GPU(s)/node"
                    )
            elif start // per_node != (start + slot_size - 1) // per_node:
                raise ValueError(
                    f"An engine must not span a node boundary for {request.group_id}: "
                    f"bundles [{start}, {start + slot_size}) with {per_node} GPU(s)/node"
                )

    # ------------------------------------------------------------------
    # Release and queries.
    # ------------------------------------------------------------------
    def release(self, placement: PlacementSlice | PlacementGroupView, group_id: str | None = None) -> PlacementRelease:
        """Release recorded allocations and report whether the group may go.

        A slice releases itself; a view releases ``group_id``, or the whole
        group when no ``group_id`` is given. Releasing something that is no
        longer recorded is not an error, but it also never authorizes removing
        the placement group again.
        """
        if isinstance(placement, PlacementSlice):
            key = placement.placement_group_key
            targets: tuple[str, ...] | None = (placement.group_id,)
        else:
            key = placement.key
            targets = None if group_id is None else (group_id,)
        with self._lock:
            recorded = self._allocations.get(key)
            if recorded is None:
                return PlacementRelease(key)
            names = tuple(recorded) if targets is None else targets
            released = tuple(entry[1] for entry in (recorded.pop(name, None) for name in names) if entry is not None)
            remaining = bool(recorded)
            if not recorded:
                self._allocations.pop(key, None)
        remove = bool(released) and not remaining and all(item.owner.removable for item in released)
        return PlacementRelease(key, released, remove)

    def allocations(self, placement_group: PlacementGroupView | None = None) -> tuple[PlacementSlice, ...]:
        with self._lock:
            if placement_group is not None:
                groups: Iterable[dict[str, tuple[PlacementRequest, PlacementSlice]]] = (
                    self._allocations.get(placement_group.key, {}),
                )
            else:
                groups = tuple(self._allocations.values())
            return tuple(entry[1] for group in groups for entry in group.values())
