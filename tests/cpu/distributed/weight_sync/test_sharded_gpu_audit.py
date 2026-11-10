# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import pytest
import torch
from torch import nn

from task_delta_weight_sync.sharded_gpu_audit import AuditSnapshot, AuditState


class _TiedModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.left = nn.Parameter(torch.tensor([0, 1, -2], dtype=torch.bfloat16))
        self.right = nn.Parameter(self.left.detach())


def test_tied_distinct_parameter_objects_are_complemented_once() -> None:
    model = _TiedModel()
    assert model.left is not model.right
    assert model.left.untyped_storage()._cdata == model.right.untyped_storage()._cdata
    audit = AuditSnapshot(model, lease_bytes=64)
    audit.capture("u0")
    original = model.left.view(torch.int16).clone()

    audit.complement("u0")

    torch.testing.assert_close(model.left.view(torch.int16), torch.bitwise_not(original))
    assert audit.state == AuditState.REFERENCE_PENDING
    with torch.no_grad():
        model.left.view(torch.int16).copy_(original)
    audit.compare("u0")
    assert audit.state == AuditState.CAPTURED


def test_missing_full_element_fails_after_poisoning() -> None:
    model = nn.Linear(4, 2, bias=False, dtype=torch.bfloat16)
    audit = AuditSnapshot(model, lease_bytes=1024)
    audit.capture("u1")
    audit.complement("u1")
    with torch.no_grad():
        model.weight.view(-1)[0].copy_(torch.tensor(0, dtype=torch.bfloat16))

    with pytest.raises(AssertionError, match="Raw GPU audit mismatch"):
        audit.compare("u1")
    assert audit.state == AuditState.POISONED


def test_ordinary_full_restore_matches_captured_snapshot() -> None:
    model = nn.Linear(4, 2, bias=False, dtype=torch.float32)
    audit = AuditSnapshot(model, lease_bytes=1024)
    audit.capture("u2")
    expected = model.weight.detach().clone()
    audit.complement("u2")
    with torch.no_grad():
        model.weight.copy_(expected)

    result = audit.compare("u2")
    assert result["mismatched_bytes"] == 0
    assert audit.state == AuditState.CAPTURED


def test_snapshot_buffers_are_reused_across_updates() -> None:
    model = nn.Linear(2, 2, bias=False, dtype=torch.bfloat16)
    audit = AuditSnapshot(model, lease_bytes=1024)
    storage_id = id(audit._entries[0].snapshot)
    audit.capture("first")
    with torch.no_grad():
        model.weight.add_(1)
    audit.capture("second")
    assert id(audit._entries[0].snapshot) == storage_id
    assert audit.compare("second")["compared_bytes"] == model.weight.numel() * 2


def test_unsupported_dtype_and_partial_alias_are_rejected() -> None:
    class HalfModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(2, dtype=torch.float16))

    with pytest.raises(ValueError, match="Unsupported audit dtype"):
        AuditSnapshot(HalfModel(), lease_bytes=16)

    class PartialAliasModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            base = torch.zeros(8, dtype=torch.bfloat16)
            self.left = nn.Parameter(base[:5])
            self.right = nn.Parameter(base[4:])

    with pytest.raises(ValueError, match="Partially overlapping"):
        AuditSnapshot(PartialAliasModel(), lease_bytes=64)


def test_persistent_buffer_and_lease_overflow_are_rejected() -> None:
    class BufferedModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(4, dtype=torch.bfloat16))
            self.register_buffer("running", torch.zeros(1, dtype=torch.bfloat16))

    with pytest.raises(ValueError, match="Persistent buffers"):
        AuditSnapshot(BufferedModel(), lease_bytes=64)

    model = nn.Linear(4, 2, bias=False, dtype=torch.bfloat16)
    with pytest.raises(MemoryError, match="lease allows"):
        AuditSnapshot(model, lease_bytes=1)


def test_nonpersistent_buffer_is_listed_in_inventory() -> None:
    class ModelWithBuffer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(2, dtype=torch.bfloat16))
            self.register_buffer("scratch", torch.zeros(3), persistent=False)

    audit = AuditSnapshot(ModelWithBuffer(), lease_bytes=64)
    assert audit.inventory()["buffers"] == [
        {"name": "scratch", "shape": [3], "dtype": "torch.float32", "device": "cpu", "persistent": False}
    ]


def test_lease_reserves_snapshot_and_scratch_until_close() -> None:
    model = nn.Linear(2, 2, bias=False, dtype=torch.bfloat16)
    allocated = 0

    def reserve(size: int) -> None:
        nonlocal allocated
        allocated += size

    def release(size: int) -> None:
        nonlocal allocated
        allocated -= size

    audit = AuditSnapshot(
        model,
        lease_bytes=1024,
        reserve=reserve,
        allocated=lambda: allocated,
        release=release,
    )
    assert allocated == audit.snapshot_bytes + 16
    audit.close()
    assert allocated == 0


def test_rebound_model_storage_cannot_be_compared_or_complemented():
    model = nn.Linear(2, 2, bias=False, dtype=torch.bfloat16)
    audit = AuditSnapshot(model, lease_bytes=1024)
    audit.capture("u")
    model.weight = nn.Parameter(model.weight.detach().clone())
    with pytest.raises(ValueError, match="storage or schema changed"):
        audit.complement("u")
    assert audit.state == AuditState.POISONED
