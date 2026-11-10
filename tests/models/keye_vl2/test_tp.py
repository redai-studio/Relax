# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="SGLang model imports require CUDA")
def test_replicated_expert_loader_updates_full_down_weights():
    from relax.models.keye_vl2.sglang.model import _replicated_expert_weight_loader

    parameter = torch.nn.Parameter(torch.zeros(3, 7, 8), requires_grad=False)
    address = parameter.data_ptr()
    weight = torch.randn(7, 8)
    _replicated_expert_weight_loader(parameter, weight, shard_id="w2", expert_id=2)
    torch.testing.assert_close(parameter[2], weight, atol=0, rtol=0)
    assert torch.count_nonzero(parameter[:2]) == 0
    replacement = torch.randn_like(parameter)
    _replicated_expert_weight_loader(parameter, replacement)
    torch.testing.assert_close(parameter, replacement, atol=0, rtol=0)
    assert parameter.data_ptr() == address
