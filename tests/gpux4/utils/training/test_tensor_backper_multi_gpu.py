# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import pytest
import torch

from tests.utils.training.test_tensor_backper import _assert_backup_on_device_preserves_each_source_device


# Creating CUDA tensor views in FakeTensorMode can still touch the real device.
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires two visible CUDA devices")
def test_backup_on_device_preserves_each_source_device(monkeypatch):
    _assert_backup_on_device_preserves_each_source_device(monkeypatch, ("cuda:0", "cuda:1"))
