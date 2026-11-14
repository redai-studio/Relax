"""Exact windowed verification must detect bit differences on real CUDA."""

import pytest
import torch

from relax.distributed.weight_sync.sglang_loader import _compare_bytes


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a leased CUDA device")
@pytest.mark.parametrize("fault", [None, "signed_zero", "nan_payload", "shape", "dtype"])
def test_device_comparison_preserves_bits_and_schema(fault):
    # Float equality is unsuitable: +0/-0 compare equal and NaN != NaN.
    bits = torch.tensor([0, -2147483648, 0x7FC00001, 0x7FC00002], dtype=torch.int32)
    expected = bits.view(torch.float32)
    actual = expected.cuda()
    if fault == "signed_zero":
        actual.view(torch.int32)[0] = -2147483648
    elif fault == "nan_payload":
        actual.view(torch.int32)[2] = 0x7FC00002
    elif fault == "shape":
        actual = actual[:-1]
    elif fault == "dtype":
        actual = actual.to(torch.bfloat16)
    mismatches = []
    _compare_bytes(actual, expected, "bits", mismatches, chunk_bytes=8)
    assert bool(mismatches) == (fault is not None)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a leased CUDA device")
def test_device_comparison_strided_windows_and_late_corruption():
    base = torch.arange(4096, dtype=torch.float32).reshape(64, 64)
    expected = base[:, ::2]
    physical = base.cuda()
    actual = physical[:, ::2]
    mismatches = []
    _compare_bytes(actual, expected, "strided", mismatches, chunk_bytes=256)
    assert mismatches == []
    actual[-1, -1] += 1
    _compare_bytes(actual, expected, "strided", mismatches, chunk_bytes=256)
    assert len(mismatches) == 1 and "row window [62, 64)" in mismatches[0]
