# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Build and validate weight-transfer plans without sending payloads or
creating process groups."""

import copy
import math
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ..layout import ExpertGeometry
from ..plan import fingerprint as fingerprint
from ..plan import region_write as region_write
from ..plan import validate_coverage as validate_coverage
from .layout import validate_dense_topology, validate_source_topology


# Rpc receipts.

RPC_RECEIPT_DELIVERY = "commit-rpc-v1"

# Expert routing.


def build_expert_routing(
    records: list[dict], actors: tuple[int, ...], engines: list[str], tp: int, experts: int
) -> dict:
    """Use actual dense and expert groups; never assume replica bytes are
    equal.

    This describes ETP1 ownership across actual pipeline stages. Native byte
    plans must separately prove that each selected owner's Bridge output covers
    its expert range. Single-engine replica groups cannot use AllGather.
    """
    dense = validate_dense_topology(records, actors, engines, tp)
    expert = validate_source_topology(records, actors, engines, tp, partition_axis="ep", replica_axis="edp")
    ep_size = len(records[0]["ep_members"])
    if type(experts) is not int or experts <= 0 or experts % ep_size:
        raise ValueError("Experts must divide the actual expert parallel group")
    for record in records:
        if (
            type(record.get("etp_rank")) is not int
            or record["etp_rank"] != 0
            or record.get("etp_members") != [record["global_rank"]]
        ):
            raise ValueError("Expert routing currently requires actual ETP1 groups")
    slots = {rank: index for index, rank in enumerate(actors)}
    by_global = {record["global_rank"]: record for record in records}
    sources = {}
    for index, engine in enumerate(engines):
        members = [member for peer in records[index * tp]["pp_members"] for member in by_global[peer]["ep_members"]]
        sources[engine] = dict(
            anchor=index * tp,
            dense_replica=dense[engine],
            expert_replica=expert[engine],
            owners=[slots[rank] for rank in members],
            expert_ranges={
                str(slots[rank]): [by_global[rank]["ep_rank"] * (experts // ep_size), experts // ep_size]
                for rank in members
            },
        )
    groups = [
        [f"{engine}-{rank}" for engine in engines if expert[engine] == replica]
        for replica in sorted(set(expert.values()))
        for rank in range(tp)
    ]
    fields = ["rank", "global_rank"] + [
        axis + suffix for axis in ("tp", "dp", "ep", "edp", "etp", "pp", "cp") for suffix in ("_rank", "_members")
    ]
    if "dp_cp_members" in records[0]:
        fields.extend(("dp_cp_rank", "dp_cp_members"))
    return dict(
        source_world_size=len(actors),
        experts=experts,
        engine_sources=sources,
        collective_groups=groups,
        coordinates=[
            {field: list(record[field]) if field.endswith("_members") else record[field] for field in fields}
            for record in records
        ],
    )


def stage_expert_owners(route: dict, engine: str, stage: int) -> list[int]:
    """Follow the engine's actual PP lane to its expert replica at a stage."""
    coordinates = route["coordinates"]
    by_global = {record["global_rank"]: record for record in coordinates}
    anchor = route["engine_sources"][engine]["anchor"]
    if type(anchor) is not int or not 0 <= anchor < len(coordinates):
        raise ValueError("Invalid expert engine source anchor")
    pipeline = coordinates[anchor]["pp_members"]
    if type(stage) is not int or not 0 <= stage < len(pipeline):
        raise ValueError("Native expert parameter has an invalid pipeline stage")
    peer = by_global[pipeline[stage]]
    if peer["pp_rank"] != stage:
        raise ValueError("Native parameter stage differs from the engine's pipeline lane")
    return [by_global[rank]["rank"] for rank in peer["ep_members"]]


def owned_expert_engines(route: dict, rank: int, canonical_names: Iterable[str]) -> list[str]:
    """Check exact per-layer/projection IDs before authorizing target
    writes."""
    engines = []
    expected = None
    for engine, source in route["engine_sources"].items():
        if rank not in source["owners"]:
            continue
        begin, count = source["expert_ranges"][str(rank)]
        owned = set(range(begin, begin + count))
        if expected is not None and owned != expected:
            raise ValueError("One source cannot own different expert ranges for its target replicas")
        engines.append(engine)
        expected = owned
    if not engines or not expected:
        raise ValueError("Expert source has no authorized inference engine")
    actual = {}
    for name in canonical_names:
        match = re.fullmatch(r"(.*\.experts)\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight", name)
        if match is None:
            raise ValueError("Bridge did not expose a canonical expert matrix")
        prefix, expert, projection = match.groups()
        ids = actual.setdefault((prefix, projection), set())
        if int(expert) in ids:
            raise ValueError("Duplicate canonical expert identity")
        ids.add(int(expert))
    layers = {prefix for prefix, _ in actual}
    if (
        not layers
        or set(actual)
        != {(prefix, projection) for prefix in layers for projection in ("gate_proj", "up_proj", "down_proj")}
        or any(ids != expected for ids in actual.values())
    ):
        raise ValueError("Bridge canonical expert ownership differs from actual MPU routing")
    return engines


# Expert catalog.


def _shape(value: object) -> bool:
    return isinstance(value, (tuple, list)) and bool(value) and all(type(n) is int and n > 0 for n in value)


def build_expert_catalog(routing: dict, inventories: list[dict], dependencies: dict) -> dict:
    """Check canonical coverage independently for every actual EP replica.

    Full HF-to-raw-source records are cold inputs only. Plans retain compact
    owner references and schema hashes, never model tensors or addresses.
    """
    coordinates = routing["coordinates"]
    world = routing["source_world_size"]
    if len(inventories) != world or len(coordinates) != world or not dependencies:
        raise ValueError("Incomplete original expert source inventory")
    slots = {record["global_rank"]: rank for rank, record in enumerate(coordinates)}
    if len(slots) != world:
        raise ValueError("Duplicate source Actor in expert catalog")
    stage_owners = defaultdict(set)
    groups = defaultdict(set)
    for rank, (record, inventory) in enumerate(zip(coordinates, inventories, strict=True)):
        stage = record["pp_rank"]
        if (
            record["rank"] != rank
            or type(stage) is not int
            or stage < 0
            or record["etp_members"] != [record["global_rank"]]
        ):
            raise ValueError("Expert catalog requires actual ETP1 stage coordinates")
        stage_owners[stage].add(rank)
        groups[stage].add(tuple(record["ep_members"]))
        if not isinstance(inventory, dict) or set(inventory) != {"sources", "parameters"}:
            raise ValueError("Missing local original expert source metadata")
        sources, parameters = inventory["sources"], inventory["parameters"]
        if (
            not isinstance(sources, dict)
            or not isinstance(parameters, dict)
            or not set(parameters) <= set(dependencies)
        ):
            raise ValueError("Unexpected native parameter in local expert catalog")
        for name, spec in sources.items():
            if (
                not isinstance(name, str)
                or not name
                or not isinstance(spec, dict)
                or set(spec) != {"shape", "dtype"}
                or not _shape(spec["shape"])
                or not isinstance(spec["dtype"], str)
                or not spec["dtype"]
            ):
                raise ValueError("Invalid original expert source schema")
        used = set()
        for name, outputs in parameters.items():
            if not isinstance(outputs, dict) or not outputs or not set(outputs) <= set(dependencies[name]):
                raise ValueError("Canonical outputs differ from actual native loader dependencies")
            for value in outputs.values():
                if (
                    not isinstance(value, dict)
                    or set(value) != {"source", "shape", "dtype"}
                    or value["source"] not in sources
                    or not _shape(value["shape"])
                    or value["dtype"] != sources[value["source"]]["dtype"]
                ):
                    raise ValueError("Canonical expert output references a nonlocal source")
                used.add(value["source"])
        if used != set(sources):
            raise ValueError("Original expert inventory contains unused or missing source parameters")
    params = {}
    for name, expected in sorted(dependencies.items()):
        if not expected or len(set(expected)) != len(expected):
            raise ValueError("Invalid native expert dependencies")
        owners = {rank for rank, record in enumerate(inventories) if name in record["parameters"]}
        stages = {coordinates[rank]["pp_rank"] for rank in owners}
        if len(stages) != 1:
            raise ValueError("A native expert parameter must belong to exactly one pipeline stage")
        stage = stages.pop()
        if owners != stage_owners[stage]:
            raise ValueError("An expert parameter is missing from one of its stage's original owners")
        reference = None
        visited = set()
        for group in sorted(groups[stage]):
            if not group or any(member not in slots for member in group):
                raise ValueError("Expert catalog references an unknown EP group member")
            members = {slots[member] for member in group}
            if len(members) != len(group) or not members <= owners or visited & members:
                raise ValueError("Expert groups do not partition the original stage owners")
            visited.update(members)
            outputs = {}
            for rank in members:
                if tuple(coordinates[rank]["ep_members"]) != group:
                    raise ValueError("Expert group membership disagrees between original owners")
                local = inventories[rank]["parameters"][name]
                if outputs.keys() & local.keys():
                    raise ValueError("Two original owners in one EP replica emit the same canonical expert")
                outputs.update(
                    {hf: dict(shape=list(value["shape"]), dtype=value["dtype"]) for hf, value in local.items()}
                )
            if set(outputs) != set(expected) or (reference is not None and outputs != reference):
                raise ValueError("An EP replica does not cover the same complete native expert schema")
            reference = outputs
        if visited != owners:
            raise ValueError("An original expert owner has no complete EP replica")
        bindings = {}
        for rank in sorted(owners):
            local = inventories[rank]["parameters"][name]
            raw = sorted({value["source"] for value in local.values()})
            bindings[str(rank)] = dict(
                sources=raw,
                schema_hash=fingerprint(
                    dict(canonical=local, sources={s: inventories[rank]["sources"][s] for s in raw})
                ),
            )
        params[name] = dict(pp_stage=stage, owners=bindings, hf_schema_hash=fingerprint(reference))
    return dict(format="native-expert-sources-v1", routing_hash=fingerprint(routing), params=params)


def validate_expert_catalog(
    plan: dict, key: str, layouts: dict, writes: list[dict] | None = None, *, required: bool = False
) -> None:
    """Reject valid byte coverage supplied by a different stage or replica."""
    catalog = plan.get("source_catalog")
    routing = plan.get("source_routing")
    if catalog is None:
        if required or (
            routing is not None and any(record.get("pp_rank", 0) for record in routing.get("coordinates", []))
        ):
            raise ValueError("Pipeline expert writes require an original source catalog")
        return
    if (
        routing is None
        or catalog["format"] != "native-expert-sources-v1"
        or catalog["routing_hash"] != fingerprint(routing)
    ):
        raise ValueError("Expert source catalog differs from the qualified routing")
    coordinates = routing["coordinates"]
    if len(coordinates) != plan["source_world_size"]:
        raise ValueError("Expert source catalog has a different Actor group")
    writes = plan["writes"] if writes is None else writes
    by_parameter = defaultdict(set)
    for write in writes:
        by_parameter[write["param"]].add(write["source_rank"])
    if set(by_parameter) != set(catalog["params"]):
        raise ValueError("Expert writes differ from the complete original parameter catalog")
    engine = key.rsplit("-", 1)[0]
    for name, item in catalog["params"].items():
        stage = item["pp_stage"]
        owners = {rank for rank, record in enumerate(coordinates) if record["pp_rank"] == stage}
        if (
            type(stage) is not int
            or not owners
            or set(item["owners"]) != {str(rank) for rank in owners}
            or by_parameter[name] != set(stage_expert_owners(routing, engine, stage))
        ):
            raise ValueError("Expert writes use the wrong original pipeline stage or EP replica")
        for binding in item["owners"].values():
            raw = binding["sources"]
            if (
                not isinstance(raw, list)
                or not raw
                or any(not isinstance(source, str) or not source for source in raw)
                or raw != sorted(set(raw))
                or not isinstance(binding["schema_hash"], str)
                or len(binding["schema_hash"]) != 64
            ):
                raise ValueError("Invalid original source reference in expert catalog")
    for write in writes:
        record = coordinates[write["source_rank"]]
        experts = routing["experts"]
        ep = len(record["ep_members"])
        layout = layouts[write["param"]]
        if (
            not ep
            or type(record["ep_rank"]) is not int
            or not 0 <= record["ep_rank"] < ep
            or record["ep_members"][record["ep_rank"]] != record["global_rank"]
            or experts % ep
            or layout["shape"][0] != experts
            or layout["nbytes"] % experts
        ):
            raise ValueError("Expert source coordinates differ from the native expert axis")
        expert_bytes = layout["nbytes"] // experts
        begin = record["ep_rank"] * (experts // ep) * expert_bytes
        end = begin + (experts // ep) * expert_bytes
        if (
            write["rows"] != 1
            or write["dst_stride"] != write["row_bytes"]
            or write["dst_offset"] % expert_bytes
            or write["row_bytes"] % expert_bytes
            or not begin <= write["dst_offset"] < write["dst_offset"] + write["row_bytes"] <= end
        ):
            raise ValueError("Expert byte interval differs from its original EP owner")


class ReceiptValidator:
    """Pin a validated static plan, while checking every version's real
    receipts."""

    def __init__(self, plan: dict) -> None:
        expected = defaultdict(set)
        for write in plan["writes"]:
            expected[write["source_rank"]].add(write["id"])
        self.expected: dict[int, frozenset[str]] = {rank: frozenset(ids) for rank, ids in expected.items()}
        self.plan_hash: str = fingerprint(plan)

    def validate(self, receipts: dict[int, dict], generation: str, version: int) -> None:
        if set(receipts) != set(self.expected):
            raise RuntimeError("Remote shard update is missing a source receipt")
        for rank, ids in self.expected.items():
            receipt = receipts[rank]
            completed = receipt["writes"]
            if (
                receipt["source_rank"] != rank
                or receipt["version"] != version
                or receipt["generation"] != generation
                or receipt["plan_hash"] != self.plan_hash
                or set(completed) != ids
                or len(completed) != len(ids)
            ):
                raise RuntimeError("Remote shard update has a stale or incomplete source receipt")


_EXPERT = re.compile(r"^(.*\.experts)\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$")


@dataclass(frozen=True)
class OwnerBlock:
    parameter: str
    projection: str
    experts: tuple[int, ...]
    hf_names: tuple[str, ...]
    sources: tuple[str, ...]
    hidden: int
    intermediate: int
    tp_size: int

    @property
    def geometry(self) -> ExpertGeometry:
        return ExpertGeometry(self.experts[-1] + 1, self.hidden, self.intermediate, self.tp_size)

    @property
    def canonical_shape(self) -> tuple[int, int, int]:
        return self.geometry.canonical_shape(self.projection, len(self.experts))

    @property
    def native_shape(self) -> tuple[int, int, int]:
        return self.geometry.native_shape(self.projection, len(self.experts))

    def write(self, *, rank: int, target: str, tp_rank: int) -> dict:
        if not 0 <= tp_rank < self.tp_size:
            raise ValueError("Invalid target TP rank")
        # The producer is the bounded native block, not a raw Megatron tensor.
        source, destination = self.geometry.native_block_regions(self.projection, self.experts[0], len(self.experts))
        return region_write(
            source,
            destination,
            id=f"{rank}:{self.parameter}:{self.experts[0]}:{len(self.experts)}:{tp_rank}",
            source=f"{self.parameter}:{self.experts[0]}:{len(self.experts)}",
            source_rank=rank,
            target=target,
            param=self.parameter,
            tp_rank=tp_rank,
        )


def plan_owner_blocks(
    canonical: Mapping[str, tuple[str, Sequence[int]]],
    dependencies: Mapping[str, Sequence[str]],
    layouts: Mapping[str, Mapping[str, Any]],
    *,
    tp_size: int,
    max_bytes: int,
    expert_boundaries: Sequence[int] = (),
) -> tuple[OwnerBlock, ...]:
    """Split actual global expert IDs into contiguous, budgeted ranges.

    ``canonical`` maps HF names to current raw source names and shapes. Both
    canonical scratch and each native write fit ``max_bytes``. Native
    dependencies come from the real loader probe, not name rewriting. Optional
    global expert boundaries keep each block inside one collective seed's range
    without changing its source identity or packing semantics.
    """
    if tp_size <= 0 or max_bytes <= 0 or not canonical:
        raise ValueError("Positive TP size, byte budget and nonempty sources are required")
    if any(type(boundary) is not int or boundary <= 0 for boundary in expert_boundaries):
        raise ValueError("Expert boundaries must be positive integer indices")
    boundaries = frozenset(expert_boundaries)
    layers = defaultdict(dict)
    for name, (source, shape) in canonical.items():
        match = _EXPERT.fullmatch(name)
        if match is None or not source or len(shape) != 2 or any(n <= 0 for n in shape):
            raise ValueError(f"Unsupported canonical expert: {name}")
        prefix, expert, projection = match.groups()
        key = (int(expert), projection)
        if key in layers[prefix]:
            raise ValueError("Duplicate canonical expert identity")
        targets = tuple(dependencies.get(name, ()))
        if len(targets) != 1 or targets[0] not in layouts:
            raise ValueError("Expert output must have exactly one probed native target")
        layers[prefix][key] = (name, source, tuple(shape), targets[0])
    result = []
    for prefix, entries in sorted(layers.items()):
        experts = sorted({expert for expert, _ in entries})
        projections = ("gate_proj", "up_proj", "down_proj")
        if set(entries) != {(expert, projection) for expert in experts for projection in projections}:
            raise ValueError("Local expert gate/up/down ownership differs")
        intermediate, hidden = entries[experts[0], "gate_proj"][2]
        if intermediate % tp_size:
            raise ValueError("Unpadded intermediate dimension must divide target TP size")
        width = intermediate // tp_size
        target_count = None
        for projection, pieces in (("gate_up", projections[:2]), ("down", projections[2:])):
            target = entries[experts[0], pieces[0]][3]
            layout = layouts[target]
            shape = tuple(layout["shape"])
            expected_tail = (2 * width, hidden) if projection == "gate_up" else (hidden, width)
            if (
                len(shape) != 3
                or shape[0] <= 0
                or shape[1:] != expected_tail
                or tuple(layout["stride"]) != (math.prod(expected_tail), expected_tail[1], 1)
                or layout["nbytes"] != math.prod(shape) * 2
                or layout["dtype"] != "torch.bfloat16"
            ):
                raise ValueError("Unsupported native expert layout")
            if target_count is not None and target_count != shape[0]:
                raise ValueError("Native gate/up and down expert counts differ")
            target_count = shape[0]
            if any(boundary >= target_count for boundary in boundaries):
                raise ValueError("Expert boundary exceeds the native expert axis")
            if experts[-1] >= target_count:
                raise ValueError("Owner expert index exceeds native expert storage")
            for expert in experts:
                for piece in pieces:
                    _, _, actual_shape, actual_target = entries[expert, piece]
                    expected = (hidden, intermediate) if piece == "down_proj" else (intermediate, hidden)
                    if actual_shape != expected or actual_target != target:
                        raise ValueError("Canonical shapes or target dependencies differ within one projection")
            canonical_expert_bytes = hidden * intermediate * len(pieces) * 2
            limit = max_bytes // canonical_expert_bytes
            if not limit:
                raise ValueError("One canonical expert exceeds the scratch budget")
            ranges = []
            for expert in experts:
                if ranges and expert not in boundaries and expert == ranges[-1][-1] + 1 and len(ranges[-1]) < limit:
                    ranges[-1].append(expert)
                else:
                    ranges.append([expert])
            for owned in ranges:
                values = [entries[expert, piece] for expert in owned for piece in pieces]
                result.append(
                    OwnerBlock(
                        target,
                        projection,
                        tuple(owned),
                        tuple(value[0] for value in values),
                        tuple(dict.fromkeys(value[1] for value in values)),
                        hidden,
                        intermediate,
                        tp_size,
                    )
                )
    return tuple(result)


# Tp plan.


# Moe collective plan.

COLLECTIVE_DELIVERY = "nccl_allgather"


def _expert_layouts(layouts: dict, names: set[str]) -> tuple[dict, int]:
    result, counts = {}, set()
    if not names or not names.issubset(layouts):
        raise ValueError("Missing native expert layouts")
    for name in sorted(names):
        layout = layouts[name]
        shape, stride = tuple(layout["shape"]), tuple(layout["stride"])
        if (
            len(shape) < 2
            or any(type(n) is not int or n <= 0 for n in shape)
            or stride != tuple(math.prod(shape[index + 1 :]) for index in range(len(shape)))
            or type(layout["nbytes"]) is not int
            or layout["nbytes"] % math.prod(shape)
            or layout["nbytes"] // math.prod(shape) not in (1, 2, 4, 8)
        ):
            raise ValueError("Collective experts require contiguous unpadded native storage")
        result[name] = dict(shape=list(shape), stride=list(stride), dtype=layout["dtype"], nbytes=layout["nbytes"])
        counts.add(shape[0])
    if len(counts) != 1:
        raise ValueError("Native expert counts differ between parameters")
    return result, counts.pop()


def _validate_interval(writes: list[dict], layouts: dict, begin: int, count: int, world_size: int) -> None:
    narrowed = {name: dict(nbytes=layout["nbytes"] // layout["shape"][0] * count) for name, layout in layouts.items()}
    rebased = []
    for write in writes:
        expert_bytes = layouts[write["param"]]["nbytes"] // layouts[write["param"]]["shape"][0]
        rebased.append(dict(write, dst_offset=write["dst_offset"] - begin * expert_bytes))
    if validate_coverage(rebased, narrowed, source_world_size=world_size) != set(layouts):
        raise ValueError("An expert owner or collective shard omits native parameters")


def _owner_ranges(writes: list[dict], layouts: dict, world_size: int, experts: int) -> list[dict]:
    if type(world_size) is not int or world_size <= 0:
        raise ValueError("Unsupported or nonuniform expert ownership")
    by_owner = defaultdict(list)
    by_parameter = defaultdict(lambda: defaultdict(list))
    for write in writes:
        if write["param"] not in layouts or type(write["source_rank"]) is not int:
            raise ValueError("Invalid expert owner or parameter")
        owner = write["source_rank"]
        expert_bytes = layouts[write["param"]]["nbytes"] // experts
        if (
            owner not in range(world_size)
            or write["rows"] != 1
            or write["dst_stride"] != write["row_bytes"]
            or write["dst_offset"] % expert_bytes
            or write["row_bytes"] % expert_bytes
        ):
            raise ValueError("Owner writes must be contiguous whole-expert blocks")
        by_owner[owner].append(write)
        by_parameter[write["param"]][owner].append(
            (write["dst_offset"] // expert_bytes, (write["dst_offset"] + write["row_bytes"]) // expert_bytes)
        )
    if not by_owner or set(by_parameter) != set(layouts):
        raise ValueError("Missing expert owner or parameter")
    # Pipeline stages own disjoint parameters. Count the participating owners
    # per parameter, not the union of senders across all stages or replicas.
    if any(experts % len(owners) for owners in by_parameter.values()):
        raise ValueError("Nonuniform expert ownership within a parameter")
    result = []
    for owner in sorted(by_owner):
        intervals = []
        owned_layouts = {name: layouts[name] for name, owners in by_parameter.items() if owner in owners}
        counts = {experts // len(by_parameter[name]) for name in owned_layouts}
        if len(counts) != 1:
            raise ValueError("Expert ownership differs between native parameters")
        count = counts.pop()
        for name in owned_layouts:
            ranges = sorted(by_parameter[name][owner])
            if not ranges or any(left[1] != right[0] for left, right in zip(ranges, ranges[1:])):
                raise ValueError("Expert owner has a hole or overlapping blocks")
            intervals.append((ranges[0][0], ranges[-1][1]))
        if len(set(intervals)) != 1:
            raise ValueError("Expert ownership differs between native parameters")
        begin, end = intervals[0]
        if end - begin != count or begin % count or begin < 0 or end > experts:
            raise ValueError("Nonuniform or misaligned expert owner interval")
        _validate_interval(by_owner[owner], owned_layouts, begin, count, world_size)
        result.append(dict(owner=owner, begin=begin, count=count))
    return result


def _plans(base: dict, schema: dict) -> dict[str, dict]:
    width = schema["experts"] // len(schema["keys"])
    return {
        key: dict(
            base,
            target=key,
            delivery=COLLECTIVE_DELIVERY,
            collective=schema,
            collective_rank=rank,
            expert_range=[rank * width, (rank + 1) * width],
            params=sorted(schema["layouts"]),
            writes=schema["shards"][key],
        )
        for rank, key in enumerate(schema["keys"])
    }


def partition_expert_plans(plans: dict[str, dict], layouts: dict, target_groups: list[list[str]]) -> dict[str, dict]:
    """Route complete source blocks to seeds in each ordered TP group.

    Ownership comes from the original Bridge-derived byte ranges, not rank
    order. One owner may span multiple seeds, but its blocks must already
    respect seed boundaries. Unequal or noncontiguous ownership is rejected.
    """
    keys = [key for group in target_groups for key in group]
    if set(keys) != set(plans) or len(keys) != len(set(keys)):
        raise ValueError("Collective groups must partition the exact target set")
    result = {}
    for group in target_groups:
        if len(group) < 2:
            raise ValueError("Collective delivery requires multiple replicas")
        first = plans[group[0]]
        if first.get("delivery") or first.get("relay_consumers") or "collective" in first:
            raise ValueError("Collective partitioning requires full network write plans")
        base = copy.deepcopy({name: value for name, value in first.items() if name not in {"writes", "target"}})
        world_size = first["source_world_size"]
        names = validate_coverage(first["writes"], layouts, source_world_size=world_size)
        native, experts = _expert_layouts(layouts, names)
        owners = _owner_ranges(first["writes"], native, world_size, experts)
        if experts % len(group):
            raise ValueError("Expert axis cannot evenly cover the replica group")
        for write in first["writes"]:
            seed_bytes = native[write["param"]]["nbytes"] // len(group)
            if write["dst_offset"] // seed_bytes != (write["dst_offset"] + write["row_bytes"] - 1) // seed_bytes:
                raise ValueError("Expert source block crosses a collective seed boundary")
        reference = [{name: value for name, value in write.items() if name != "target"} for write in first["writes"]]
        reference.sort(key=lambda write: write["id"])
        tp_ranks = {write["tp_rank"] for write in first["writes"]}
        if len(tp_ranks) != 1:
            raise ValueError("Collective targets must have one TP rank")
        shards = {}
        for rank, key in enumerate(group):
            plan = plans[key]
            if {name: value for name, value in plan.items() if name not in {"writes", "target"}} != base:
                raise ValueError("Replica network plan modes differ")
            if plan.get("target", key) != key or any(write["target"] != key for write in plan["writes"]):
                raise ValueError("Misrouted full replica write plan")
            normalized = [
                {name: value for name, value in write.items() if name != "target"} for write in plan["writes"]
            ]
            normalized.sort(key=lambda write: write["id"])
            if normalized != reference:
                raise ValueError("Replica native write layouts differ")
            shards[key] = sorted(
                [
                    copy.deepcopy(write)
                    for write in plan["writes"]
                    if write["dst_offset"] // (native[write["param"]]["nbytes"] // len(group)) == rank
                ],
                key=lambda write: write["id"],
            )
        schema = dict(
            keys=list(group), experts=experts, tp_rank=tp_ranks.pop(), owners=owners, layouts=native, shards=shards
        )
        generated = _plans(base, schema)
        for key, plan in generated.items():
            validate_collective_plan(plan, layouts, key, schema["tp_rank"])
        result.update(generated)
    return result


def validate_collective_plan(plan: dict, layouts: dict, key: str, tp_rank: int) -> set[str]:
    """Cold receiver gate: individual owners, local shard and full output."""
    schema = plan["collective"]
    keys, shards, world_size = schema["keys"], schema["shards"], plan["source_world_size"]
    if (
        plan["delivery"] != COLLECTIVE_DELIVERY
        or plan.get("target", key) != key
        or len(keys) < 2
        or len(set(keys)) != len(keys)
        or set(shards) != set(keys)
        or key not in keys
        or plan["collective_rank"] != keys.index(key)
        or schema["tp_rank"] != tp_rank
        or plan["writes"] != shards[key]
        or plan["params"] != sorted(schema["layouts"])
    ):
        raise ValueError("Invalid or misrouted collective plan")
    native, experts = _expert_layouts(layouts, set(plan["params"]))
    if native != schema["layouts"] or experts != schema["experts"] or experts % len(keys):
        raise ValueError("Collective native layouts differ")
    full = [write for target in keys for write in shards[target]]
    if validate_coverage(full, native, source_world_size=world_size) != set(native):
        raise ValueError("Collective shards do not cover all native parameters")
    validate_expert_catalog(plan, key, native, full)
    owners = _owner_ranges(full, native, world_size, experts)
    if owners != schema["owners"]:
        raise ValueError("Collective owner ranges differ from actual writes")
    routing = plan.get("source_routing")
    if routing is not None:
        if (
            routing["source_world_size"] != world_size
            or routing["experts"] != experts
            or keys not in routing["collective_groups"]
        ):
            raise ValueError("Collective group differs from qualified source routing")
        for peer in keys:
            source = routing["engine_sources"][peer.rsplit("-", 1)[0]]
            expected = [
                dict(
                    owner=owner,
                    begin=source["expert_ranges"][str(owner)][0],
                    count=source["expert_ranges"][str(owner)][1],
                )
                for owner in sorted(source["owners"])
            ]
            if owners != expected:
                raise ValueError("Collective bytes belong to a different qualified expert replica")
    width = experts // len(keys)
    rank = keys.index(key)
    if plan["expert_range"] != [rank * width, (rank + 1) * width]:
        raise ValueError("Collective input is not its own output chunk")
    for index, target in enumerate(keys):
        expected = {
            item["owner"]
            for item in owners
            if item["begin"] < (index + 1) * width and item["begin"] + item["count"] > index * width
        }
        if {write["source_rank"] for write in shards[target]} != expected or any(
            write["target"] != target or write["tp_rank"] != tp_rank for write in shards[target]
        ):
            raise ValueError("Wrong collective shard owner or target")
        _validate_interval(shards[target], native, index * width, width, world_size)
    return set(native)


def collective_plan_hashes(plan: dict) -> dict[str, str]:
    """Bind every peer's whole plan without a circular peer-hash field."""
    return {key: fingerprint(value) for key, value in collective_peer_plans(plan).items()}


def collective_peer_plans(plan: dict) -> dict[str, dict]:
    """Reconstruct read-only peer plans from the agreed common schema."""
    base = {
        name: value
        for name, value in plan.items()
        if name not in {"delivery", "collective", "collective_rank", "expert_range", "params", "writes"}
    }
    return _plans(base, plan["collective"])
