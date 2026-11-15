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
) -> tuple[dict[str, dict], BufferedWriter | None]:
    """Return receipts only after every original owner's native write is
    DONE."""
    if not plans:
        if writer is not None and writer.targets:
            writer.disconnect_targets()
        return {}, writer
    if writer is None:
        agent = agent_future.result() if agent_future is not None else None
        writer = BufferedWriter(64 << 20, device=torch.device("cuda", torch.cuda.current_device()), agent=agent)
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
                if (
                    any(type(write[key]) is not int for key in ("src_offset", "src_stride", "rows", "row_bytes"))
                    or write["src_offset"] < 0
                    or write["rows"] <= 0
                    or write["row_bytes"] <= 0
                    or write["src_stride"] < write["row_bytes"]
                ):
                    raise RuntimeError("Remote shard source rectangle is invalid")
                end = write["src_offset"] + (write["rows"] - 1) * write["src_stride"] + write["row_bytes"]
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
                        payload.view(count_rows, width).copy_(
                            flat.as_strided(
                                (count_rows, width),
                                (write["src_stride"], 1),
                                storage_offset=flat.storage_offset() + write["src_offset"] + row * write["src_stride"],
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
