# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Check CPU body transport failure boundaries without external services."""

import asyncio
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from relax.distributed.replay_manager.storage import TQBodyBackend, _read_simple_storage_bodies


@pytest.mark.parametrize("failure", ["missing", "duplicate", "rpc"])
def test_direct_cpu_body_reads_drain_all_units_and_reject_incomplete_batches(failure):
    completed = []
    routing = {
        "a": SimpleNamespace(global_indexes=[11], batch_positions=[0]),
        "b": SimpleNamespace(global_indexes=[12], batch_positions=[0 if failure == "duplicate" else 1]),
    }

    async def receive(indexes, fields, *, target_storage_unit):
        if target_storage_unit == "b":
            await asyncio.sleep(0)
        completed.append(target_storage_unit)
        if target_storage_unit == "a" and failure == "rpc":
            raise RuntimeError("injected storage RPC failure")
        values = (
            [] if target_storage_unit == "a" and failure == "missing" else [torch.zeros((2, 1), dtype=torch.int32)]
        )
        return fields, {"replay_body": values}

    manager = SimpleNamespace(_group_by_hash=lambda indexes: routing, _get_from_single_storage_unit=receive)
    metadata = SimpleNamespace(global_indexes=[11, 12], field_names=["replay_body"], size=2)
    with pytest.raises(RuntimeError):
        asyncio.run(_read_simple_storage_bodies(manager, metadata))
    assert sorted(completed) == ["a", "b"]


def test_encoded_unknown_manager_uses_existing_generic_get_data():
    class Metadata:
        size = 1

        def select_fields(self, fields):
            return self

        def select_samples(self, samples):
            return self

    calls = []
    client = SimpleNamespace(
        storage_manager=object(),
        kv_retrieve_meta=lambda **kwargs: Metadata(),
        get_data=lambda metadata: calls.append(metadata) or {"replay_body": torch.arange(4).reshape(1, 2, 2)},
        close=lambda: None,
    )
    backend = TQBodyBackend("a", lambda: client)
    try:
        np.testing.assert_array_equal(backend.read_encoded(["key"])[0], np.arange(4).reshape(2, 2))
        assert len(calls) == 1
    finally:
        backend.close()
