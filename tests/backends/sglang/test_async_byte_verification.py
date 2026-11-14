"""Leased-CUDA checks for bounded asynchronous exact-byte verification."""

import json

import pytest
import torch
from safetensors.torch import save_file

from relax.distributed.weight_sync.sglang_loader import (
    _AsyncByteComparator,
    _compare_bytes,
    _verification_comparator,
    verify_qwen3_dense,
)


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a leased CUDA device")


@pytest.mark.parametrize("fault", [None, "signed_zero", "nan_payload", "last_window"])
def test_two_slot_pipeline_keeps_bits_and_all_windows(fault):
    bits = torch.tensor([0, -2147483648, 0x7FC00001, 0x7FC00002] * 257, dtype=torch.int32)
    wanted = bits.view(torch.float32).reshape(257, 4)
    actual = wanted.cuda()
    if fault == "signed_zero":
        actual.view(torch.int32)[0, 0] = -2147483648
    elif fault == "nan_payload":
        actual.view(torch.int32)[0, 2] = 0x7FC00002
    elif fault == "last_window":
        actual.view(torch.int32)[-1, -1] ^= 1
    comparator = _AsyncByteComparator(actual.device, chunk_bytes=64, batch_size=3)
    token = _verification_comparator.set(comparator)
    failures = []
    try:
        _compare_bytes(actual, wanted, "bits", failures, chunk_bytes=64)
        comparator.finish(failures)
    finally:
        comparator.close()
        _verification_comparator.reset(token)
    assert bool(failures) == bool(fault)
    assert comparator.metrics["h2d_bytes"] == wanted.numel() * wanted.element_size()
    assert comparator.metrics["windows"] == 65
    assert comparator.metrics["host_staging_bytes"] == 128


@pytest.mark.parametrize("fault", [False, True])
@pytest.mark.parametrize("reference_device", ["cpu", "cuda"])
def test_strided_device_views_and_late_tail(fault, reference_device):
    base = torch.arange(4096, dtype=torch.float32).reshape(64, 64)
    wanted = base.to(reference_device)[:, ::2]
    actual = base.cuda()[:, ::2]
    if fault:
        actual[-1, -1] += 1
    comparator = _AsyncByteComparator(actual.device, chunk_bytes=256)
    token = _verification_comparator.set(comparator)
    failures = []
    try:
        _compare_bytes(actual, wanted, "strided", failures, chunk_bytes=256)
        comparator.finish(failures)
    finally:
        comparator.close()
        _verification_comparator.reset(token)
    assert len(failures) == int(fault)
    if fault:
        assert "row window [62, 64)" in failures[0]


def test_full_verification_cleans_context_and_does_not_cache_device_bytes(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(dict(model_type="qwen3", num_attention_heads=1, num_key_value_heads=1))
    )
    wanted = torch.arange(32768, dtype=torch.float32)
    save_file({"model.norm.weight": wanted}, tmp_path / "model.safetensors")
    actual = wanted.cuda()
    physical = {(0, "model.norm.weight"): actual}
    statistics = {}
    proof = verify_qwen3_dense(tmp_path, None, physical_tensors=physical, pipeline_metrics=statistics)
    assert proof["bytes_checked"] == wanted.numel() * 4 and statistics["h2d_bytes"] == proof["bytes_checked"]
    assert _verification_comparator.get() is None
    actual[-1] += 1
    with pytest.raises(ValueError, match="Physical weight mismatch"):
        verify_qwen3_dense(tmp_path, None, physical_tensors=physical)
    assert _verification_comparator.get() is None
