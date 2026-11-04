# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Strict SimpleStorage deletion: every storage ACK precedes index release."""

import asyncio
import inspect
from typing import Any


def require_strict_clear_contract(client: Any) -> None:
    from transfer_queue.storage.managers.simple_storage_manager import AsyncSimpleStorageManager

    manager = client.storage_manager
    if not isinstance(manager, AsyncSimpleStorageManager):
        raise RuntimeError("recovery settlement is qualified only for SimpleStorage")
    required = (
        (manager, "_group_by_hash"),
        (manager, "_clear_single_storage_unit"),
        (client, "_get_partition_meta"),
        (client, "_clear_partition_in_controller"),
    )
    if any(not callable(getattr(owner, name, None)) for owner, name in required):
        raise RuntimeError("incompatible TQ strict-clear contract")
    source = inspect.getsource(manager._clear_single_storage_unit)
    if "CLEAR_DATA_RESPONSE" not in source or "raise" not in source:
        raise RuntimeError("TQ storage clear does not expose a verified ACK")
    parameters = inspect.signature(client.async_put).parameters
    if not {"custom_meta", "is_last"}.issubset(parameters):
        raise RuntimeError("TQ lacks atomic metadata/readiness and EOS contract")


async def strict_clear_partition(client: Any, partition: str, expected_rows: list[int]) -> None:
    """Clear a settled partition only after every storage unit acknowledges
    deletion.

    The caller must first confirm old writers have finished. Do not use
    manager.clear_data: it catches per-storage failures and returns.
    """
    metadata = await client._get_partition_meta(partition)
    if metadata is None or not metadata.global_indexes:
        raise RuntimeError("strict clear requires an existing nonempty partition")
    rows = list(metadata.global_indexes)
    if len(set(rows)) != len(rows) or set(rows) != set(expected_rows):
        raise RuntimeError("strict clear partition rows differ from settled manifest")
    manager = client.storage_manager
    routing = manager._group_by_hash(rows)
    routed = [row for group in routing.values() for row in group.global_indexes]
    if len(routed) != len(rows) or set(routed) != set(rows):
        raise RuntimeError("strict clear routing does not cover exactly the settled rows")
    results = await asyncio.gather(
        *(
            manager._clear_single_storage_unit(group.global_indexes, target_storage_unit=unit)
            for unit, group in routing.items()
        ),
        return_exceptions=True,
    )
    errors = [result for result in results if isinstance(result, BaseException)]
    if errors:
        raise RuntimeError("storage clear failed or ACK unknown; controller indexes retained") from errors[0]
    await client._clear_partition_in_controller(partition)
