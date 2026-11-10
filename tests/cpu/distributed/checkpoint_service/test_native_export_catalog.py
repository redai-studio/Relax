# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Reuse only a complete reference schema and retain uncertain
registrations."""

import gc
import weakref

import pytest
import torch

from relax.distributed.checkpoint_service.backends.shard_p2p.remote.loader_probe import NativeExportCatalog
from relax.distributed.checkpoint_service.backends.shard_p2p.remote.transport import RegisteredMemory


def test_native_reference_catalog_retains_only_schema_and_requires_completion():
    catalog = NativeExportCatalog(4)
    weight = torch.ones(3, 2, dtype=torch.bfloat16)
    reference = weakref.ref(weight)
    catalog.record(4, [("weight", weight)])
    catalog.record(4, [("weight", weight)])
    del weight
    gc.collect()
    assert reference() is None
    with pytest.raises(RuntimeError, match="incomplete or stale"):
        catalog.resolve(4)
    catalog.complete(4)
    assert catalog.resolve(4)["weight"] == ((3, 2), torch.bfloat16)
    with pytest.raises(RuntimeError, match="stale"):
        catalog.resolve(5)


@pytest.mark.parametrize("fault", ["version", "shape", "dtype", "late"])
def test_native_reference_capture_defers_failures_until_collective_boundary(fault):
    catalog = NativeExportCatalog(4)
    value = torch.ones(2, dtype=torch.bfloat16)
    catalog.record(4, [("weight", value)])
    if fault == "late":
        catalog.complete(4)
    # Capture must finish the native pipeline's collective schedule, even when
    # a schema is invalid. resolve later runs inside all-rank error agreement.
    catalog.record(
        5 if fault == "version" else 4,
        [("weight", torch.ones(3) if fault == "shape" else value.float() if fault == "dtype" else value)],
    )
    catalog.complete(4)
    with pytest.raises(RuntimeError, match="incomplete or stale"):
        catalog.resolve(4)


def test_completed_empty_reference_cannot_select_a_direct_payload():
    catalog = NativeExportCatalog(4)
    catalog.complete(4)
    with pytest.raises(RuntimeError, match="did not export"):
        catalog.resolve(4)


def test_failed_batched_registration_keeps_its_owner_and_allocations():
    class Agent:
        def get_reg_descs(self, values):
            assert len(values) == 2
            return object()

        def register_memory(self, descriptors):
            raise TimeoutError("registration outcome unknown")

    registration = RegisteredMemory(Agent(), {"a": torch.ones(2), "b": torch.ones(3)})
    owner = weakref.ref(registration)
    with pytest.raises(TimeoutError, match="unknown"):
        registration.register(batched=True)
    with pytest.raises(RuntimeError, match="inactive or uncertain"):
        registration.release()
    del registration
    gc.collect()
    assert owner() is not None
    assert set(owner().parameters) == {"a", "b"}
    assert owner().registrations
