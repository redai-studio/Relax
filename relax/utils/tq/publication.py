# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Confirm deferred writes against consumer-visible controller state."""

import asyncio
from typing import Any

from relax.engine.rollout.deferred import PublicationReceipt, PublicationState


async def publish_confirmed(
    client: Any,
    data: Any,
    *,
    partition_id: str,
    operation_id: str,
    custom_meta: Any = None,
    is_last: bool = False,
    timeout_s: float = 120,
) -> PublicationReceipt:
    """Write once, then verify every returned row and terminal marker.

    A lost write reply, query failure or cancellation leaves the outcome
    unknown. Callers must stop the affected attempt; this function never
    replays writes or deletes a partition while consumers may still use it. The
    caller owns operation deduplication via DeferredExecutor.
    """
    if not operation_id or not partition_id or timeout_s <= 0:
        raise ValueError("Publication requires identity, partition and positive timeout")

    async def commit() -> None:
        metadata = await client.async_put(
            data=data, partition_id=partition_id, custom_meta=custom_meta, is_last=is_last
        )
        expected = set(metadata.global_indexes)
        if not expected:
            raise RuntimeError("Publication returned no sample identities")
        while True:
            indexes, status = await client.async_get_production_status(list(data.keys()), partition_id)
            if indexes is not None and status is not None:
                observed = indexes.tolist()
                positions = [i for i, index in enumerate(observed) if index in expected]
                visible = expected.issubset(set(observed)) and bool(status[positions].all())
                if visible and (not is_last or await client.async_check_production_completed(partition_id)):
                    return
            await asyncio.sleep(0.05)

    await asyncio.wait_for(commit(), timeout=timeout_s)
    return PublicationReceipt(operation_id, PublicationState.CONFIRMED)
