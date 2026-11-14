# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Send only cross-node colocated shards to a fresh receiver allocation."""

from concurrent.futures import Future

import torch

from .remote.transport import BufferedWriter


def send_remote_shards(
    sources: dict[str, torch.Tensor],
    plans: dict[str, dict],
    manifests: dict[str, dict],
    *,
    rank: int,
    version: int,
    writer: BufferedWriter | None,
    agent_future: Future | None = None,
    buffer_bytes: int = 64 << 20,
) -> tuple[dict[str, dict], BufferedWriter | None]:
    """Return receipts only after every original owner's native write is
    DONE."""
    if not plans:
        if writer is not None and writer.targets:
            writer.disconnect_targets()
        return {}, writer
    if writer is None:
        agent = agent_future.result() if agent_future is not None else None
        pending_bytes = sum(write["rows"] * write["row_bytes"] for plan in plans.values() for write in plan["writes"])
        writer = BufferedWriter(
            min(buffer_bytes, pending_bytes), device=torch.device("cuda", torch.cuda.current_device()), agent=agent
        )
    try:
        persistent = all(manifests[target].get("persistent_registration_key") for target in plans)
        connected = getattr(writer, "targets", {})
        identities = getattr(writer, "target_registration_keys", {})
        if connected and (
            not persistent
            or set(connected) != set(plans)
            or any(
                identities.get(target)
                != (manifests[target]["persistent_registration_key"], manifests[target]["agent_metadata"])
                for target in plans
            )
        ):
            writer.disconnect_targets()
        for target in plans:
            if target not in manifests or not manifests[target].get("agent_metadata"):
                raise RuntimeError("Remote shard target has no registered NIXL allocation")
            writer.connect_target(target, manifests[target])
        index, buffer = writer.acquire()
        cursor, batch = 0, []

        def flush() -> None:
            nonlocal index, buffer, cursor, batch
            if batch:
                writer.submit_ranges(index, batch, manifests)
                index, buffer = writer.acquire()
                cursor, batch = 0, []

        receipts, flats = {}, {}
        for target, plan in sorted(plans.items()):
            completed, seen = [], set()
            for write in plan["writes"]:
                if write["source_rank"] != rank or write["id"] in seen:
                    raise RuntimeError("Remote shard owner or rectangle identity changed")
                name = write["source"]
                if name not in flats:
                    source = sources[name]
                    if not source.is_cuda or source.dtype != torch.bfloat16 or not source.is_contiguous():
                        raise RuntimeError("Remote shard source storage changed")
                    flats[name] = source.view(torch.uint8).view(-1)
                flat = flats[name]
                column_stride = write.get("src_column_stride", 1)
                if (
                    any(type(write[key]) is not int for key in ("src_offset", "src_stride", "rows", "row_bytes"))
                    or write["src_offset"] < 0
                    or write["rows"] <= 0
                    or write["row_bytes"] <= 0
                    or type(column_stride) is not int
                    or (column_stride == 1 and write["src_stride"] < write["row_bytes"])
                    or (
                        column_stride != 1
                        and (
                            column_stride <= 0
                            or column_stride % 2
                            or write["src_stride"] != 2
                            or write["src_offset"] % 2
                            or write["row_bytes"] % 2
                            or buffer.numel() % 2
                            or write["row_bytes"] > buffer.numel()
                        )
                    )
                ):
                    raise RuntimeError("Remote shard source rectangle is invalid")
                extent = (
                    write["row_bytes"] if column_stride == 1 else (write["row_bytes"] // 2 - 1) * column_stride + 2
                )
                end = write["src_offset"] + (write["rows"] - 1) * write["src_stride"] + extent
                if end > flat.numel():
                    raise RuntimeError("Remote shard source rectangle exceeds live storage")
                row = 0
                while row < write["rows"]:
                    width = write["row_bytes"]
                    if width <= buffer.numel():
                        count_rows = min(write["rows"] - row, (buffer.numel() - cursor) // width)
                        if not count_rows:
                            flush()
                            continue
                        size = count_rows * width
                        payload = buffer[cursor : cursor + size]
                        offset = flat.storage_offset() + write["src_offset"] + row * write["src_stride"]
                        if column_stride == 1:
                            payload.view(count_rows, width).copy_(
                                flat.as_strided((count_rows, width), (write["src_stride"], 1), storage_offset=offset)
                            )
                        else:
                            payload.view(torch.bfloat16).view(count_rows, width // 2).copy_(
                                flat.view(torch.bfloat16).as_strided(
                                    (count_rows, width // 2), (1, column_stride // 2), storage_offset=offset // 2
                                )
                            )
                        batch.append(
                            (
                                dict(
                                    target=target,
                                    param=write["param"],
                                    dst_offset=write["dst_offset"] + row * write["dst_stride"],
                                    dst_stride=write["dst_stride"],
                                    rows=count_rows,
                                    row_bytes=width,
                                ),
                                payload,
                            )
                        )
                        cursor += size
                    else:
                        for column in range(0, width, buffer.numel()):
                            size = min(buffer.numel(), width - column)
                            if cursor + size > buffer.numel():
                                flush()
                            payload = buffer[cursor : cursor + size]
                            offset = write["src_offset"] + row * write["src_stride"] + column
                            payload.copy_(flat.narrow(0, offset, size))
                            batch.append(
                                (
                                    dict(
                                        target=target,
                                        param=write["param"],
                                        dst_offset=write["dst_offset"] + row * write["dst_stride"] + column,
                                        dst_stride=size,
                                        rows=1,
                                        row_bytes=size,
                                    ),
                                    payload,
                                )
                            )
                            cursor += size
                    row += 1 if width > buffer.numel() else count_rows
                completed.append(write["id"])
                seen.add(write["id"])
            receipts[target] = dict(
                source_rank=rank,
                version=version,
                generation=manifests[target]["generation"],
                plan_hash=plan["plan_hash"],
                writes=completed,
            )
        flush()
        writer.finish()
        if not persistent:
            writer.disconnect_targets()
        return receipts, writer
    except BaseException:
        # The session retains both this writer and the live source snapshot.
        writer._retain_failure()
        raise
