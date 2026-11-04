# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import pytest
import torch

from relax.backends.megatron.chunkflow.metadata import int32_metadata


@pytest.mark.parametrize("enabled", ["0", "1"])
def test_chunk_metadata_preserves_values_and_owns_storage(monkeypatch, enabled):
    monkeypatch.setenv("RELAX_CHUNKFLOW_EXECUTION_OPT", enabled)
    values = [0, 8192, 16384, 16512]
    result = int32_metadata(values, torch.device("cpu"))
    values[-1] = 0
    torch.testing.assert_close(result, torch.tensor([0, 8192, 16384, 16512], dtype=torch.int32))
    assert result.is_contiguous()
