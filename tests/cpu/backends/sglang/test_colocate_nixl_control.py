# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from relax.backends.sglang import shard_p2p_weight_loader as loader
from relax.distributed.checkpoint_service.backends.shard_p2p.remote import common


@pytest.mark.parametrize("enabled", [False, True])
def test_consistency_policy_accepts_the_selected_numerical_audit(monkeypatch, enabled):
    monkeypatch.setenv("RELAX_P2P_WEIGHT_CONSISTENCY_CHECK", str(int(enabled)))
    common.validate_consistency_check(dict(consistency_check=enabled, audit_native=enabled), audit_key="audit_native")


def test_cold_plan_registers_only_remote_span_and_reuses_it(monkeypatch):
    monkeypatch.setenv("RELAX_COLOCATE_PINNED_NIXL_TARGET", "1")
    original_empty = torch.empty
    monkeypatch.setattr(torch, "empty", lambda *a, **kw: original_empty(*a, **(kw | {"pin_memory": False})))
    agent = SimpleNamespace(get_reg_descs=lambda values: values, register_memory=Mock(), deregister_memory=Mock())
    state = dict(
        nixl_agent=agent,
        delivery=dict(routes=dict(nixl=["r"])),
        writes=[dict(id="r", param="w", dst_offset=8, rows=2, dst_stride=8, row_bytes=4)],
    )
    loader._prefetch_pinned_targets(state, {"w": torch.empty(12, dtype=torch.bfloat16)})
    pinned = loader._finish_pinned_prefetch(state)
    assert pinned.sizes == {"w": 24} and pinned.spans == {"w": (8, 20)}
    assert pinned.buffer.numel() == 12
    assert loader._finish_pinned_prefetch(state) is pinned
    agent.register_memory.assert_called_once()
    assert "pinned_future" not in state and "pinned_pool" not in state
    pinned.registration.release()
    agent.deregister_memory.assert_called_once()


@pytest.mark.parametrize("enabled,remote", [(False, True), (True, False)])
def test_unused_cold_plan_does_not_start_allocation(monkeypatch, enabled, remote):
    monkeypatch.setenv("RELAX_COLOCATE_PINNED_NIXL_TARGET", str(int(enabled)))
    state = dict(nixl_agent=object(), delivery=dict(routes=dict(nixl=["remote"] if remote else [])))
    loader._prefetch_pinned_targets(state, {})
    assert "pinned_future" not in state and "pinned_pool" not in state


def test_resident_gpu_targets_do_not_allocate_pinned_memory(monkeypatch):
    monkeypatch.setenv("RELAX_COLOCATE_PINNED_NIXL_TARGET", "1")
    state = dict(persistent_targets=True, nixl_agent=object())
    loader._prefetch_pinned_targets(state, {})
    assert "pinned_future" not in state and "pinned_targets" not in state


@pytest.mark.parametrize("busy", [False, True])
def test_persistent_registration_retires_only_after_completion(busy):
    registration = Mock()
    model = torch.nn.Linear(2, 2)
    model._relax_expert_ipc = dict(
        persistent_targets=True,
        failed=False,
        epoch=object() if busy else None,
        persistent_registration=registration,
        persistent_parameters={"weight": model.weight},
        persistent_storage_ids={"weight": model.weight.untyped_storage()._cdata},
    )
    if busy:
        with pytest.raises(RuntimeError, match="pending"):
            loader.retire_persistent_targets(model)
        registration.release.assert_not_called()
        assert model._relax_expert_ipc["persistent_registration"] is registration
    else:
        loader.retire_persistent_targets(model)
        loader.retire_persistent_targets(model)
        registration.release.assert_called_once()
        assert "persistent_parameters" not in model._relax_expert_ipc


def test_replica_completion_requires_matching_receipts_and_gpu_completion():
    from relax.backends.sglang.replica_weight_sync import validate_replica_completion
    from relax.distributed.checkpoint_service.backends.shard_p2p.plan import fingerprint

    state = dict(replicas={}, epoch=dict(replica_ready=dict(receipt_hash=fingerprint([]))))
    with pytest.raises(RuntimeError, match="GPU collective"):
        validate_replica_completion(state, [])
    state["epoch"]["replica_complete"] = "completed"
    validate_replica_completion(state, [])
    with pytest.raises(RuntimeError, match="GPU collective"):
        validate_replica_completion(state, [dict(version=2)])
