# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from types import SimpleNamespace

import pytest
import torch


def test_vision_replay_captures_actual_text_and_image_experts(monkeypatch):
    vl_routing = pytest.importorskip("sglang.srt.multimodal.dsv41.vl_routing", exc_type=ImportError)
    from sglang.srt.layers.moe import topk
    from sglang.srt.models import deepseek_v2

    from relax.backends.sglang.deepseekv41.patch import apply_vision_route_capture_patch

    # Exercise the actual CPU gate with opposite text and image selection biases.
    monkeypatch.setattr(vl_routing, "is_cuda", lambda: False)
    config = topk.TopKConfig(top_k=2)
    moe = SimpleNamespace(
        layer_id=3,
        config=SimpleNamespace(image_token_id=99),
        topk=SimpleNamespace(topk_config=config),
        gate=SimpleNamespace(
            e_score_correction_bias=torch.tensor([4, 3, 2, 1.0]),
            e_score_correction_bias_vl=torch.tensor([1, 2, 3, 4.0]),
        ),
    )
    logits, tokens = torch.ones(2, 4), torch.tensor([7, 99])
    original = vl_routing.vision_topk(moe, logits, tokens)
    captured = {}
    monkeypatch.setattr(topk, "get_global_experts_capturer", lambda: SimpleNamespace(capture=captured.update))
    monkeypatch.setattr(deepseek_v2, "vision_topk", vl_routing.vision_topk)
    apply_vision_route_capture_patch()
    actual = deepseek_v2.vision_topk(moe, logits, tokens)
    expected = torch.tensor([[0, 1], [3, 2]], dtype=torch.int32)
    assert captured["layer_id"] == 3
    torch.testing.assert_close(captured["topk_indices"], expected)
    torch.testing.assert_close(actual.topk_ids, original.topk_ids)
    torch.testing.assert_close(actual.topk_weights, original.topk_weights)

    config.allow_routed_experts_capture = False
    captured.clear()
    deepseek_v2.vision_topk(moe, logits, tokens)
    assert not captured
