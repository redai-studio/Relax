# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Partition identical receiver outputs without changing logical shard
plans."""

from .plan import fingerprint


def wire_writes(writes: list[dict], shapes: dict) -> list[dict]:
    """Pack full-height column shards in transposed wire order.

    The native allocation is scratch until the replica collective restores row-
    major order. Logical plans and their native coverage stay unchanged.
    """
    result = []
    for write in writes:
        shape = shapes.get(write["param"])
        if shape is None:
            result.append(dict(write))
            continue
        rows, columns = shape
        if (
            write["rows"] != rows
            or write["dst_stride"] != columns * 2
            or not 0 <= write["dst_offset"] < columns * 2
            or write["row_bytes"] % 2
            or write["dst_offset"] % 2
        ):
            raise ValueError("Wire transpose requires full-height BF16 column shards")
        value = dict(
            write,
            src_column_stride=write["src_stride"],
            src_stride=2,
            rows=write["row_bytes"] // 2,
            row_bytes=rows * 2,
            dst_stride=rows * 2,
            dst_offset=write["dst_offset"] * rows,
        )
        value["id"] = fingerprint(value)
        result.append(value)
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


def replica_configurations(
    bindings: list[dict],
    capabilities: list[list[str]],
    layouts: dict,
    tp_size: int,
    *,
    ordinary_shapes: list[dict],
    plans: dict,
    staging_bytes: int,
) -> dict:
    """Use real same-host workers; other topologies retain direct transfers."""
    size = len(bindings) // tp_size
    peers = [binding["delivery"]["receiver"] for binding in bindings]
    if (
        size < 2
        or len({(peer["hostname"], peer["boot_id"]) for peer in peers}) != 1
        or any(binding["delivery"]["routes"]["cuda_ipc"] for binding in bindings)
    ):
        return {}
    result = {}
    for tp in range(tp_size):
        keys = [f"{engine}-{tp}" for engine in range(size)]
        slots = [engine * tp_size + tp for engine in range(size)]
        names = set(layouts[keys[0]])
        for slot in slots:
            values = capabilities[slot]
            if not isinstance(values, list) or any(not isinstance(name, str) for name in values):
                raise ValueError("Invalid native replica capability")
            names.intersection_update(values)
        names = sorted(
            name
            for name in names
            if layouts[keys[0]][name]["nbytes"] >= 2 * size and layouts[keys[0]][name]["nbytes"] % (2 * size) == 0
        )
        if not names:
            return {}
        transposed = {}
        for name, shape in ordinary_shapes[tp].items():
            writes = [write for write in plans[keys[0]]["writes"] if write["param"] == name]
            if (
                name in names
                and len(shape) == 2
                and layouts[keys[0]][name]["nbytes"] <= staging_bytes
                and staging_bytes % 2 == 0
                and writes
                and all(
                    write["rows"] == shape[0]
                    and write["dst_stride"] == shape[1] * 2
                    and write["dst_offset"] < shape[1] * 2
                    and write["row_bytes"] < shape[1] * 2
                    for write in writes
                )
            ):
                transposed[name] = list(shape)
        config = dict(
            members=[peers[slot] for slot in slots],
            parameters=names,
            layouts=layouts[keys[0]],
            logical_plan_hash=bindings[slots[0]]["logical_plan_hash"],
            transpose_shapes=transposed,
        )
        for key in keys:
            if layouts[key] != config["layouts"]:
                raise ValueError("Replica output layouts disagree")
            result[key] = config
    return result
