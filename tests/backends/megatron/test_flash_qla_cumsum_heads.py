# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Check small-head FlashQLA scans against an independent FP64 reference."""

import pytest


@pytest.mark.parametrize("heads", [1, 2, 3, 4, 5, 6, 9, 48])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("variable", [False, True])
def test_chunk64_cumsum_small_heads(heads, reverse, variable):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] not in (10, 11):
        pytest.skip("Requires a Blackwell GPU and the patched FlashQLA runtime")
    cumsum = pytest.importorskip("flash_qla.ops.utils.cumsum")
    generator = torch.Generator(device="cuda").manual_seed(815)
    values = torch.randn((1, 256, heads), device="cuda", generator=generator, dtype=torch.float32) * 0.1
    boundaries = [0, 63, 192] if variable else [0, 256]
    offsets = torch.tensor(boundaries, device="cuda", dtype=torch.int32) if variable else None
    expected = torch.zeros_like(values, dtype=torch.float64)
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        for left in range(start, end, 64):
            right = min(left + 64, end)
            part = values[:, left:right].double()
            expected[:, left:right] = part.flip(1).cumsum(1).flip(1) if reverse else part.cumsum(1)
    actual = cumsum.chunk_local_cumsum(
        values, chunk_size=64, cu_seqlens=offsets, reverse=reverse, zero_last_chunk=variable
    )
    torch.testing.assert_close(actual.double(), expected, atol=2e-6, rtol=2e-6)
    assert actual.shape == values.shape
    assert actual.dtype == values.dtype
