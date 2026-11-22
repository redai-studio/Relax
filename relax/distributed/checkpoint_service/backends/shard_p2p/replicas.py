# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Partition identical receiver outputs without changing logical shard
plans."""

from math import prod

from .plan import fingerprint


def replica_write_signature(writes: list[dict]) -> str:
    """Compare logical source rectangles without receiver coordinates."""
    return fingerprint(
        sorted(
            ({key: value for key, value in write.items() if key not in ("id", "target")} for write in writes),
            key=fingerprint,
        )
    )


def tp_replica_configurations(commands: list[list[dict]], capabilities: list[list[str]]) -> dict:
    """Partition only identical, entirely remote ordinary outputs within a
    host."""
    result = {}
    for engine, pair in enumerate(commands):
        size = len(pair)
        members = [command["delivery"]["delivery"]["receiver"] for command in pair]
        if (
            size < 2
            or len({(member["hostname"], member["boot_id"]) for member in members}) != 1
            or len({member["pid"] for member in members}) != size
            or len({member["gpu_uuid"] for member in members}) != size
        ):
            continue
        candidates = set(pair[0]["geometry"].get("ordinary_targets", {}))
        for tp in range(size):
            values = capabilities[engine * size + tp]
            if not isinstance(values, list) or any(not isinstance(name, str) for name in values):
                raise ValueError("Invalid native replica capability")
            candidates.intersection_update(values)
        selected, signatures = {}, {}
        for name in sorted(candidates):
            shapes, signatures_by_rank = [], []
            for command in pair:
                shape = command["geometry"].get("ordinary_targets", {}).get(name)
                writes = [write for write in command["writes"] if write["param"] == name]
                remote = set(command["delivery"]["delivery"]["routes"]["nixl"])
                if not shape or not writes or any(write["id"] not in remote for write in writes):
                    break
                shapes.append(shape)
                signatures_by_rank.append(replica_write_signature(writes))
            if len(shapes) != size or any(shape != shapes[0] for shape in shapes) or len(set(signatures_by_rank)) != 1:
                continue
            nbytes = prod(shapes[0]) * 2
            if nbytes < 2 * size or nbytes % (2 * size):
                continue
            selected[name] = dict(nbytes=nbytes)
            signatures[name] = signatures_by_rank[0]
        if selected:
            config = dict(members=members, layouts=selected, signatures=signatures, parameters=sorted(selected))
            for tp in range(size):
                result[f"{engine}-{tp}"] = config
    return result


def partition_writes(writes: list[dict], layouts: dict, names: list[str], rank: int, size: int) -> list[dict]:
    """Clip rectangles to this replica's contiguous BF16 output chunk.

    Unqualified outputs retain their complete NIXL writes. Fragments preserve
    source strides and offsets, including partial first and last rows.
    """
    if size < 2 or not 0 <= rank < size:
        raise ValueError("Invalid receiver replica partition")
    selected = set(names)
    if any(name not in layouts or layouts[name]["nbytes"] % (2 * size) for name in selected):
        raise ValueError("Replica output cannot be split into equal BF16 chunks")
    result = []
    for write in writes:
        if write["param"] not in selected:
            result.append(dict(write))
            continue
        chunk = layouts[write["param"]]["nbytes"] // size
        low, high = rank * chunk, (rank + 1) * chunk
        offset, stride, width = write["dst_offset"], write["dst_stride"], write["row_bytes"]
        first = max(0, (low - offset - width) // stride + 1)
        stop = min(write["rows"], (high - offset + stride - 1) // stride)
        if first >= stop:
            continue

        def emit(row, rows, column, count):
            source_column = column // 2 * write["src_column_stride"] if "src_column_stride" in write else column
            fragment = dict(
                write,
                src_offset=write["src_offset"] + row * write["src_stride"] + source_column,
                dst_offset=offset + row * stride + column,
                rows=rows,
                row_bytes=count,
            )
            fragment["id"] = fingerprint(dict(parent=write["id"], replica=rank, fragment=fragment))
            result.append(fragment)

        if first == stop - 1:
            start = max(low - offset - first * stride, 0)
            emit(first, 1, start, min(width, high - offset - first * stride) - start)
            continue
        if low > offset + first * stride:
            column = low - offset - first * stride
            emit(first, 1, column, width - column)
            first += 1
        last = stop - 1
        tail = high < offset + last * stride + width
        if first < stop - int(tail):
            emit(first, stop - int(tail) - first, 0, width)
        if tail:
            emit(last, 1, 0, high - offset - last * stride)
    return result
