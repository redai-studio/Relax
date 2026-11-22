# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Exact byte coverage is independent of Actor count; malformed owners fail."""

import copy
import pickle

import pytest

from relax.distributed.checkpoint_service.backends.shard_p2p.plan import (
    TransferPlan,
    coalesce_writes,
    fingerprint,
    validate_coverage,
)


def _writes(world):
    return [
        dict(id=f"owner-{rank}", source_rank=rank, param="p", dst_offset=rank * 8, rows=2, row_bytes=4, dst_stride=4)
        for rank in range(world)
    ]


@pytest.mark.parametrize("world", [1, 3, 8, 32])
def test_coverage_reconstructs_every_byte_for_actual_owner_count(world):
    writes = _writes(world)
    layout = {"p": dict(nbytes=world * 8)}
    assert validate_coverage(list(reversed(writes)), layout, source_world_size=world) == {"p"}
    counts = [0] * (world * 8)
    for write in writes:
        for row in range(write["rows"]):
            for offset in range(write["row_bytes"]):
                counts[write["dst_offset"] + row * write["dst_stride"] + offset] += 1
    assert set(counts) == {1}
    for damaged in [writes[:-1], writes + [dict(writes[0], id="duplicate-region")]]:
        # An empty write set has no covered parameters; the caller requires
        # equality with its expected native parameter set.
        if not damaged:
            assert validate_coverage(damaged, layout, source_world_size=world) == set()
        else:
            with pytest.raises(ValueError):
                validate_coverage(damaged, layout, source_world_size=world)


@pytest.mark.parametrize("world", [0, -1, True, 2.0, "2"])
def test_invalid_owner_count_is_rejected_even_without_writes(world):
    with pytest.raises(ValueError, match="world size"):
        validate_coverage([], {}, source_world_size=world)


@pytest.mark.parametrize(
    "field,value",
    [
        ("source_rank", True),
        ("source_rank", 0.0),
        ("source_rank", -1),
        ("source_rank", 2),
        ("id", ""),
        ("id", []),
        ("param", []),
        ("dst_offset", False),
        ("dst_offset", 0.0),
        ("rows", True),
        ("rows", 2.0),
        ("rows", 10**12),
        ("row_bytes", 4.0),
        ("dst_stride", False),
    ],
)
def test_malformed_descriptor_cannot_pass_integer_or_bounds_checks(field, value):
    writes = copy.deepcopy(_writes(2))
    writes[0][field] = value
    with pytest.raises(ValueError):
        validate_coverage(writes, {"p": dict(nbytes=16)}, source_world_size=2)


@pytest.mark.parametrize("corruption", ["hole", "overlap", "overflow", "duplicate", "owner"])
def test_coverage_rejects_invalid_write_plan(corruption):
    writes = _writes(2)
    if corruption == "hole":
        writes.pop()
    elif corruption == "overlap":
        writes[1]["dst_offset"] = 0
    elif corruption == "overflow":
        writes[-1]["rows"] += 1
    elif corruption == "duplicate":
        writes[1]["id"] = writes[0]["id"]
    else:
        writes[0]["source_rank"] = 4
    with pytest.raises(ValueError):
        validate_coverage(writes, {"p": {"nbytes": 16}})


@pytest.mark.parametrize(
    "capacity,rectangles",
    [
        (131072, [(0, 16384, 4, 8), (4, 16384, 4, 8)]),
        (48, [(0, 2, 4, 8), (4, 1, 4, 4), (12, 1, 24, 24), (36, 1, 4, 4), (40, 1, 8, 8)]),
        (24, [(0, 6, 2, 4), (2, 3, 2, 8), (6, 3, 2, 8)]),
        (18, [(4, 2, 4, 6), (0, 1, 4, 4), (8, 1, 2, 2), (14, 1, 4, 4)]),
    ],
)
def test_coverage_strided_rows_mixed_contiguous_edges_and_multiple_strides(capacity, rectangles):
    writes = [
        dict(id=str(index), source_rank=0, param="p", dst_offset=offset, rows=rows, row_bytes=width, dst_stride=stride)
        for index, (offset, rows, width, stride) in enumerate(rectangles)
    ]
    layout = {"p": dict(nbytes=capacity)}
    assert validate_coverage(list(reversed(writes)), layout) == {"p"}
    for damaged in (writes[1:], writes + [dict(writes[0], id="overlapping-copy")]):
        with pytest.raises(ValueError):
            validate_coverage(damaged, layout)


def _tiles():
    # Four tiles of a 4x8 byte matrix; the source starts inside a larger
    # allocation. Independent expected output is source[5:37].
    return [
        dict(
            id=f"tile-{row}-{col}",
            source="raw",
            source_rank=0,
            target="engine-0",
            param="p",
            src_offset=5 + row * 8 + col,
            dst_offset=row * 8 + col,
            rows=2,
            row_bytes=4,
            src_stride=8,
            dst_stride=8,
        )
        for row in (0, 2)
        for col in (0, 4)
    ]


def _copy_bytes(writes, source, size):
    result = bytearray([255] * size)
    for write in writes:
        for row in range(write["rows"]):
            src = write["src_offset"] + row * write["src_stride"]
            dst = write["dst_offset"] + row * write["dst_stride"]
            width = write["row_bytes"]
            result[dst : dst + width] = source[src : src + width]
    return result


def test_delivery_builder_owns_sources_and_immutable_write_identity():
    writes = _tiles()
    source = dict(
        boot_id="00000000-0000-0000-0000-000000000001",
        hostname="source",
        pid=1,
        gpu_uuid="source-0",
        device_index=0,
        generation="session",
    )
    sources = {0: source}
    receiver = dict(source, pid=2, gpu_uuid="receiver-0")
    plan = TransferPlan("engine-0", 1, tuple(writes))
    geometry = {"source_world_size": 1, "layout": {"张量": [1, 8]}}
    builder = plan.bind_sources(sources, geometry=geometry)
    expected = builder(receiver)
    assert expected == plan.delivery(sources, receiver)
    assert expected["logical_hash"] == fingerprint(plan.as_dict())
    assert builder.geometry_hash == fingerprint(dict(geometry=geometry, writes=writes))
    for write in writes:
        write.update(id="changed", source_rank=99)
    source["generation"] = "changed"
    assert builder(receiver) == expected
    response = builder(receiver)
    response["routes"]["cuda_ipc"].clear()
    response["receiver"]["pid"] = 99
    response["sources"][0]["generation"] = "changed"
    assert builder(receiver) == expected
    assert pickle.loads(pickle.dumps(builder))(receiver) == expected


@pytest.mark.parametrize("max_bytes,expected_count", [(None, 1), (32, 1), (16, 2), (8, 4)])
def test_common_plan_merges_tiles_without_changing_bytes_or_exceeding_budget(max_bytes, expected_count):
    original = _tiles()
    untouched = copy.deepcopy(original)
    plan = TransferPlan.create(
        "engine-0",
        original,
        {"p": {"nbytes": 32}},
        source_world_size=1,
        parameters={"p"},
        max_bytes=max_bytes,
    ).as_dict()
    assert original == untouched
    assert len(plan["writes"]) == expected_count
    source = bytes(range(64))
    assert _copy_bytes(plan["writes"], source, 32) == source[5:37]
    assert sum(w["rows"] * w["row_bytes"] for w in plan["writes"]) == 32
    # Existing oversize fragments are retained for the producer's normal
    # capacity rejection; the optimizer must never create a larger one.
    assert all(w["rows"] * w["row_bytes"] <= max(16, max_bytes or 32) for w in plan["writes"])
    reversed_plan = TransferPlan.create(
        "engine-0",
        list(reversed(original)),
        {"p": {"nbytes": 32}},
        source_world_size=1,
        parameters={"p"},
        max_bytes=max_bytes,
    ).as_dict()
    assert {w["id"] for w in plan["writes"]} == {w["id"] for w in reversed_plan["writes"]}


@pytest.mark.parametrize("axis", ["rows", "columns", "reverse"])
def test_strided_coalescing_preserves_gaps_and_nonzero_offsets(axis):
    first = dict(
        id="a",
        source="raw",
        source_rank=0,
        target="engine-0",
        param="p",
        src_offset=3,
        dst_offset=1,
        rows=2,
        row_bytes=2,
        src_stride=8,
        dst_stride=6,
    )
    second = dict(first, id="b")
    second.update(src_offset=19 if axis == "rows" else 5, dst_offset=13 if axis == "rows" else 3)
    if axis == "reverse":
        first["dst_offset"] = 9
        second.update(src_offset=19, dst_offset=1)
    original = [first, second]
    merged = coalesce_writes(original, max_bytes=16)
    assert len(merged) == (2 if axis == "reverse" else 1)
    source = bytes(range(64))
    assert _copy_bytes(merged, source, 32) == _copy_bytes(original, source, 32)
    assert sum(write["rows"] * write["row_bytes"] for write in merged) == 8


@pytest.mark.parametrize(
    "boundary", ["source", "source_rank", "target", "param", "tp_rank", "dtype", "src_offset", "dst_offset"]
)
def test_merge_never_crosses_producer_owner_target_or_adapter_boundary(boundary):
    first = dict(
        id="a",
        source="raw",
        source_rank=0,
        target="engine-0",
        param="p",
        tp_rank=0,
        dtype="bf16",
        src_offset=0,
        dst_offset=0,
        rows=1,
        row_bytes=4,
        src_stride=4,
        dst_stride=4,
    )
    second = dict(first, id="b", src_offset=4, dst_offset=4)
    if boundary in {"src_offset", "dst_offset"}:
        second[boundary] += 1
    else:
        second[boundary] = 1 if boundary in {"source_rank", "tp_rank"} else "different"
    assert len(coalesce_writes([first, second])) == 2


@pytest.mark.parametrize(
    "first_metadata,second_metadata,count", [(True, 1, 2), ({"axes": [0, 1]}, {"axes": (0, 1)}, 1)]
)
def test_merge_preserves_json_adapter_identity(first_metadata, second_metadata, count):
    first = dict(
        _tiles()[0],
        id="a",
        src_offset=0,
        dst_offset=0,
        rows=1,
        row_bytes=2,
        src_stride=2,
        dst_stride=2,
        adapter=first_metadata,
    )
    second = dict(first, id="b", src_offset=2, dst_offset=2, adapter=second_metadata)
    merged = coalesce_writes([first, second])
    assert len(merged) == count
    assert _copy_bytes(merged, bytes(range(4)), 4) == bytes(range(4))
