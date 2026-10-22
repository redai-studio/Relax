# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Exercise HC derived weights across reloads and real B300 graph replay."""

import importlib.util
import os
from pathlib import Path

import pytest
import torch


@pytest.fixture
def hc_module():
    source = os.environ.get("SGLANG_SOURCE_DIR")
    if not source:
        pytest.skip("Set SGLANG_SOURCE_DIR to the patched SGLang checkout")
    path = Path(source) / "python/sglang/srt/layers/hyperconnection.py"
    spec = importlib.util.spec_from_file_location("qwen38_hc_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="HC CuTe decode kernels require a Blackwell GPU",
)
def test_qwen38_hc_reload_refreshes_captured_cache_in_place(hc_module):
    from sglang.kernels.ops.elementwise.hc_mix import permute_pad_up_weight

    config = hc_module.HyperConnectionConfig(hidden_size=2560, hc_count=4, hc_lowrank=320, hc_per_branch_norm=True)
    with torch.device("cuda"):
        model = hc_module.GatedResidual(config, use_combine=False).to(dtype=torch.bfloat16)
    x = torch.randn(8, 10240, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        for param in model.parameters():
            param.uniform_(-0.02, 0.02)
        # The model post-load hook runs before graph capture/offload enrollment.
        model.finalize_mix_weights()
        cache = model._mix_up_weight_padded
        pointer = cache.data_ptr()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                model.mix(x)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = model.mix(x)[0]
        before = output.clone()
        param = model.input_mix_weight_up.weight
        updated = torch.randn_like(param) * 0.2
        param.weight_loader(param, updated)
        assert torch.equal(param, updated)
        assert cache.data_ptr() == pointer
        assert torch.equal(cache, permute_pad_up_weight(updated, config.hc_count))
        expected = model.mix(x)[0]
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(output, expected)
        assert not torch.equal(output, before)
        # A subsequent synchronization must preserve the same graph storage too.
        param.weight_loader(param, -updated)
        expected = model.mix(x)[0]
        graph.replay()
        torch.cuda.synchronize()
        assert cache.data_ptr() == pointer
        assert torch.equal(output, expected)
