# Copyright (c) 2026 Relax Authors. All Rights Reserved.
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch


@pytest.fixture
def vision(monkeypatch):
    # Isolate Transformers decorators/position APIs; execute the real adapter.
    for name, members in {
        "transformers.modeling_outputs": {"BaseModelOutputWithPooling": SimpleNamespace},
        "transformers.utils.generic": {"merge_with_config_defaults": lambda fn: fn},
        "transformers.utils.output_capturing": {"capture_outputs": lambda fn: fn},
        "transformers.vision_utils": {
            "get_vision_interpolation_indices_and_weights": lambda *a, **kw: None,
            "get_vision_position_ids": lambda *a, **kw: None,
        },
    }.items():
        module = ModuleType(name)
        module.__dict__.update(members)
        monkeypatch.setitem(sys.modules, name, module)
    path = Path(__file__).resolve().parents[4] / "relax/backends/fsdp_turbo/performance_ops/vision.py"
    spec = importlib.util.spec_from_file_location("vision_adapter_test", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("device", ["cpu", "meta"])
def test_frame_boundaries_are_cpu_metadata(vision, device):
    calls = []

    def forward(hidden, cu, positions, **kwargs):
        assert cu.device.type == "cpu"
        assert cu.dtype == torch.int32
        calls.append((cu.tolist(), kwargs["max_seqlen"], positions[0].shape[0]))
        return hidden.square()

    hidden = torch.randn(7, 3, device=device, requires_grad=True)
    result = vision._vision_block(
        SimpleNamespace(_relax_block_forward=forward),
        hidden,
        None,
        position_embeddings=(hidden.detach(), hidden.detach()),
        _relax_frame_lengths=[3, 4],
    )
    assert calls == [([0, 3], 3, 3), ([0, 4], 4, 4)]
    if device == "meta":
        assert result.shape == hidden.shape
        return
    torch.testing.assert_close(result, hidden.square())
    result.sum().backward()
    torch.testing.assert_close(hidden.grad, 2 * hidden.detach())


def test_merger_runs_once_for_multiple_frames(vision, monkeypatch):
    calls = []
    monkeypatch.setattr(
        vision,
        "get_vision_interpolation_indices_and_weights",
        lambda *a, **kw: (torch.zeros(8, 1, dtype=torch.long), torch.ones(8, 1)),
    )
    monkeypatch.setattr(vision, "get_vision_position_ids", lambda *a, **kw: torch.arange(8))

    def merge(hidden):
        calls.append(hidden.shape)
        return hidden.reshape(-1, 4, 3).mean(1)

    visual = SimpleNamespace(
        num_grid_per_side=2,
        interpolation_mode="bilinear",
        interpolation_align_corners=False,
        spatial_merge_size=2,
        patch_embed=lambda hidden, **kw: hidden,
        pos_embed=lambda ids: torch.zeros(8, 1, 3),
        rotary_pos_emb=lambda *a: None,
        blocks=[],
        merger=merge,
    )
    hidden = torch.randn(8, 3, requires_grad=True)
    result = vision._vision_forward(visual, hidden, None, _relax_vision_grid=[[2, 2, 2]])
    assert calls == [torch.Size([8, 3])]
    torch.testing.assert_close(result.pooler_output, hidden.reshape(2, 4, 3).mean(1))
    result.pooler_output.sum().backward()
    torch.testing.assert_close(hidden.grad, torch.full_like(hidden, 0.25))
