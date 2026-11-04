import json

import pytest
import torch
from experiments.copy_audit import (
    LogicalCopyAudit,
    condition_copy_phase,
    merge_repeated_transfer_evidence,
    suspend_logical_copy_audit,
)


def test_actual_copy_operations_report_direction_dtype_and_bytes():
    source = torch.arange(6, dtype=torch.float32)
    target = torch.empty_like(source)
    with LogicalCopyAudit() as audit:
        cloned = source.clone()
        converted = source.to(dtype=torch.float64)
        returned = target.copy_(source)
    result = audit.result()
    assert returned is target and torch.equal(cloned, source) and torch.equal(converted, source)
    assert result["calls"] == 3 and result["logical_bytes"] == 24 + 48 + 24
    assert result["logical_bytes_by_direction"] == {"cpu->cpu": 96}
    assert {row["operator"] for row in result["operations"]} == {
        "aten.clone.default",
        "aten._to_copy.default",
        "aten.copy_.default",
    }
    cast = next(row for row in result["operations"] if row["operator"] == "aten._to_copy.default")
    assert cast["source_equivalent_bytes"] == 24 and cast["destination_bytes"] == 48
    assert result["completed_without_exception"]
    json.dumps(result, allow_nan=False)


def test_views_identity_to_and_overlapping_self_copy_are_not_counted():
    source = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    with LogicalCopyAudit() as audit:
        view = source.view_as(source)
        source.transpose(0, 1)
        source[:, :1]
        assert source.to(dtype=source.dtype, device=source.device) is source
        source.copy_(view)
        torch.empty(0).clone()
    assert audit.result()["calls"] == 0
    assert audit.result()["skipped"]["empty_or_self_copy"] == 2


def test_distinct_nonoverlapping_views_still_copy_and_broadcast_counts_destination():
    source = torch.arange(8, dtype=torch.float32)
    with LogicalCopyAudit() as audit:
        source[4:].copy_(source[:4])
        torch.empty(3).copy_(torch.ones(1))
    assert audit.result()["logical_bytes"] == 4 * 4 + 3 * 4
    assert torch.equal(source[:4], source[4:])


def test_nested_suspension_excludes_evidence_copies_without_changing_operations():
    source = torch.arange(5)
    with suspend_logical_copy_audit():
        pass
    with LogicalCopyAudit() as audit:
        first = source.clone()
        with suspend_logical_copy_audit():
            second = source.clone()
            with suspend_logical_copy_audit():
                third = source.clone()
        fourth = source.clone()
    assert all(torch.equal(source, value) for value in (first, second, third, fourth))
    assert audit.result()["calls"] == 2
    assert audit.result()["skipped"]["suspended_copy_calls"] == 2


def test_native_exception_and_return_semantics_are_preserved_and_scope_is_removed():
    source = torch.arange(5)
    audit = LogicalCopyAudit()
    with pytest.raises(RuntimeError, match="single memory location"):
        with audit:
            source[1:].copy_(source[:-1])
    assert not audit.result()["completed_without_exception"]
    assert audit.result()["calls"] == 0
    with LogicalCopyAudit() as next_audit:
        source.clone()
    assert next_audit.result()["calls"] == 1


def test_observer_preserves_exact_output_and_rng_without_value_reads():
    source = torch.arange(6, dtype=torch.float32).requires_grad_()
    rng_before = torch.get_rng_state().clone()
    reference = source.clone().to(dtype=torch.float64)
    with LogicalCopyAudit() as audit:
        actual = source.clone().to(dtype=torch.float64)
    assert torch.equal(actual, reference)
    assert actual.requires_grad and actual.grad_fn is not None
    assert torch.equal(
        torch.autograd.grad(actual.sum(), source)[0],
        torch.autograd.grad(reference.sum(), source)[0],
    )
    assert torch.equal(torch.get_rng_state(), rng_before)
    assert audit.result()["calls"] == 2


def cpu_fingerprint_audit(**kwargs):
    # CPU test direction is explicit; this is not a CUDA transfer test.
    return LogicalCopyAudit(
        collect_repeated_transfers=True,
        fingerprint_directions=(("cpu", "cpu"),),
        **kwargs,
    )


def test_repeated_bytes_require_actual_equal_source_content_and_encoding():
    source = torch.arange(4, dtype=torch.float32)
    with cpu_fingerprint_audit() as audit:
        with condition_copy_phase("frozen_encoder"):
            one, two = source.clone(), source.clone()
            cast = source.double()
            source.add_(1)
            different = source.clone()
    result = audit.result()["repeated_transfers"]
    assert result["repeated_transfer_calls"] == 1
    assert result["repeated_destination_bytes"] == 16
    assert len(result["fingerprints"]) == 3
    assert result["cuda_value_readbacks_for_audit"] == 0
    assert not result["complete_traffic_measurement"]
    assert torch.equal(one, two) and torch.equal(cast.float(), one) and not torch.equal(different, one)
    json.dumps(result, allow_nan=False)


def test_fingerprint_scope_suspension_and_bounds_are_explicit():
    source = torch.arange(2, dtype=torch.float32)
    with cpu_fingerprint_audit(max_total_hashed_bytes=16, max_fingerprints=1) as audit:
        source.clone()
        with condition_copy_phase("cache_read"):
            with suspend_logical_copy_audit():
                source.clone()
            source.clone()
            source.clone()
            source.clone()
    result = audit.result()["repeated_transfers"]
    assert result["hashed_cpu_source_bytes"] == 16
    assert len(result["fingerprints"]) == 1 and result["fingerprints"][0]["calls"] == 2
    assert result["repeated_destination_bytes"] == 8
    assert result["unfingerprinted"]["outside_condition_phase"]["calls"] == 1
    assert result["unfingerprinted"]["total_hashed_byte_limit"]["destination_bytes"] == 8
    with pytest.raises(ValueError, match="Unknown"):
        with condition_copy_phase("arbitrary-unbounded-label"):
            pass


def test_cross_run_online_repeat_evidence_uses_retained_content_fingerprints():
    source = torch.arange(2)
    records = []
    for _ in range(2):
        with cpu_fingerprint_audit() as audit:
            with condition_copy_phase("frozen_encoder"):
                source.clone()
        records.append(audit.result()["repeated_transfers"])
    merged = merge_repeated_transfer_evidence(records)
    assert merged["runs"] == 2 and merged["repeated_transfer_calls"] == 1
    assert merged["repeated_destination_bytes"] == 16


def test_fingerprinting_keeps_rng_autograd_values_and_default_h2d_scope():
    source = torch.arange(3, dtype=torch.float32).requires_grad_()
    rng = torch.get_rng_state().clone()
    with cpu_fingerprint_audit() as audit:
        with condition_copy_phase("frozen_encoder"):
            result = source.clone()
    assert len(audit.result()["repeated_transfers"]["fingerprints"]) == 1
    assert torch.equal(result, source)
    assert torch.equal(torch.autograd.grad(result.sum(), source)[0], torch.ones_like(source))
    assert torch.equal(torch.get_rng_state(), rng)
    with LogicalCopyAudit(collect_repeated_transfers=True) as production_scope:
        with condition_copy_phase("frozen_encoder"):
            source.clone()
    measured = production_scope.result()["repeated_transfers"]
    assert not measured["fingerprints"]
    assert measured["unfingerprinted"]["outside_declared_cpu_source_directions"]["calls"] == 1
