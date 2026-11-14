"""Bounded expert staging must preserve every bit across TP layouts."""

import pytest
import torch

from relax.distributed.weight_sync.sglang_loader import (
    _compare_packed_expert,
    _verification_timings,
)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("fused_gate", [False, True])
@pytest.mark.parametrize("tp_size", [1, 2])
@pytest.mark.parametrize("chunk_bytes", [32, 8 * 1024**2])
def test_packed_expert_bits_partitions_and_fresh_reads(device, fused_gate, tp_size, chunk_bytes):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires a leased CUDA device")
    # Include distinct NaN payloads, signed zeros and normal finite values.
    bits = torch.tensor([0, -2147483648, 0x7FC00001, 0x7FC00002, 0x3F800000, -1], dtype=torch.int32)
    storage = bits.repeat(64).reshape(16, 24).view(torch.float32)
    wanted = storage.T
    for rank in range(tp_size):
        parts = [p.chunk(tp_size, dim=0)[rank] for p in wanted.chunk(2, dim=0)]
        reference = torch.cat(parts) if fused_gate else wanted.chunk(tp_size, dim=1)[rank]
        actual = reference.contiguous().to(device)
        phases = {}
        token = _verification_timings.set(phases)
        try:

            def compare():
                mismatches = []
                _compare_packed_expert(
                    actual,
                    wanted,
                    "expert",
                    mismatches,
                    fused_gate=fused_gate,
                    tp_size=tp_size,
                    tp_rank=rank,
                    chunk_bytes=chunk_bytes,
                )
                return mismatches

            assert compare() == []
            assert ("packed_expert_transfer" in phases) == (device == "cuda" and chunk_bytes >= storage.numel() * 4)
            actual.view(torch.int32)[-1, -1] ^= 1
            assert compare()  # Detect a last-byte corruption in each TP rank.
            actual.view(torch.int32)[-1, -1] ^= 1
            assert compare() == []
            # Changing the source between verifications must never trust a
            # previously staged expert. This element is in every TP partition.
            before = storage.view(torch.int32).clone()
            storage.view(torch.int32).bitwise_xor_(1)
            assert compare()
            storage.view(torch.int32).copy_(before)
            assert compare() == []
        finally:
            _verification_timings.reset(token)


@pytest.mark.parametrize("fault", ["shape", "dtype"])
def test_packed_expert_schema_only_never_transfers(fault):
    wanted = torch.empty((16, 24), device="meta").T
    actual = torch.empty((24, 16), device="meta")
    actual = actual[:-1] if fault == "shape" else actual.to(torch.bfloat16)
    mismatches = []
    _compare_packed_expert(
        actual, wanted, "schema", mismatches, fused_gate=True, tp_size=1, tp_rank=0, schema_only=True
    )
    assert mismatches
