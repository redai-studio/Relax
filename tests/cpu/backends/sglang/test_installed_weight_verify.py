# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import pytest
import torch

from relax.backends.sglang.installed_weight_verify import InstalledWeightVerifier


def test_final_raw_state_checks_indexed_writes_tied_views_nan_signed_zero_and_buffers(tmp_path, monkeypatch):
    monkeypatch.setenv("RELAX_DELTA_VERIFY_TMPDIR", str(tmp_path))
    module = torch.nn.Module()
    bits = torch.arange(80, dtype=torch.int32).reshape(8, 10)
    bits[0, 0], bits[1, 0] = 0x7FC01234, -2147483648
    module.register_parameter("weight", torch.nn.Parameter(bits.view(torch.float32), requires_grad=False))
    module.register_buffer("view", bits.T)
    module.register_buffer("scalar", torch.tensor(13, dtype=torch.int16))
    module.register_buffer("empty", torch.empty(0))
    check = InstalledWeightVerifier(module, chunk_bytes=16)
    # A native loader can write the same destination twice, then restore the original raw NaN.
    bits[0, 0], bits[0, 0] = 19, 0x7FC01234
    assert check.finish(module) == 2 * bits.nbytes + 2
    assert check.file.closed
    for name in ("weight", "scalar", "schema"):
        check = InstalledWeightVerifier(module, chunk_bytes=16)
        if name == "weight":
            bits[1, 0] = 0
        elif name == "scalar":
            module.scalar.add_(1)
        else:
            module.register_buffer("extra", torch.ones(1))
        with pytest.raises(ValueError, match="differs from native FULL|registered parameter"):
            check.finish(module)
        assert check.file.closed
