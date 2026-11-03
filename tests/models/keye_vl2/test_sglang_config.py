# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from types import SimpleNamespace

import pytest
import torch


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="SGLang imports require the CUDA runtime")


@pytest.mark.parametrize("context_length, expected_capacity", [(None, 128), (64, 64), (256, 128)])
def test_keye_decode_accepts_default_context_length(monkeypatch, context_length, expected_capacity):
    from relax.models.keye_vl2.sglang import model

    attention = model.KeyeAttention.__new__(model.KeyeAttention)
    torch.nn.Module.__init__(attention)
    attention.qkv_proj = lambda hidden: (hidden, None)
    attention.q_size, attention.kv_size = 256, 128
    attention.num_heads, attention.num_kv_heads, attention.head_dim = 2, 1, 128
    attention.q_norm, attention.k_norm = torch.nn.Identity(), torch.nn.Identity()
    attention.rope_theta = 1e7
    attention.keye_multimodal_enabled = False
    attention.sa_indexer = SimpleNamespace(
        q_proj=lambda hidden: hidden.new_zeros((len(hidden), 1040)),
        k_proj=lambda hidden: hidden.new_zeros((len(hidden), 64)),
        q_norm=torch.nn.Identity(),
        k_norm=torch.nn.Identity(),
    )
    attention.attn = SimpleNamespace(layer_id=0)
    attention.o_proj = lambda hidden: (hidden, None)
    pool = SimpleNamespace(
        set_kv_buffer=lambda *args: None,
        keye_index_cache={0: torch.zeros(128, 64, device="cuda")},
        get_key_buffer=lambda layer: torch.zeros(128, 1, 128, device="cuda"),
        get_value_buffer=lambda layer: torch.zeros(128, 1, 128, device="cuda"),
    )
    monkeypatch.setattr(model, "get_model", lambda: SimpleNamespace(context_length=context_length))
    monkeypatch.setattr(model, "get_token_to_kv_pool", lambda: pool)
    monkeypatch.setattr(
        model, "get_req_to_token_pool", lambda: SimpleNamespace(req_to_token=torch.zeros(1, 128, device="cuda"))
    )
    monkeypatch.setattr(model, "rope", lambda x, *args: x)
    capacities = []

    def decode(query, *args):
        capacities.append(args[-1])
        return query.flatten(1)

    monkeypatch.setattr(model, "paged_decode", decode)
    batch = SimpleNamespace(
        out_cache_loc=torch.tensor([0], device="cuda"),
        req_pool_indices=torch.tensor([0], device="cuda"),
        seq_lens=torch.tensor([1], device="cuda"),
        seq_lens_cpu=[1],
        forward_mode=SimpleNamespace(is_decode=lambda: True),
    )
    output = attention(torch.tensor([0], device="cuda"), torch.ones(1, 512, device="cuda"), batch)
    assert output.shape == (1, 256)
    assert capacities == [expected_capacity]


@pytest.mark.parametrize(
    "enable_multimodal, language_only, resolved_multimodal",
    [(None, False, True), (True, False, True), (False, False, False), (None, True, False)],
)
def test_keye_vision_uses_resolved_model_capability(
    monkeypatch, enable_multimodal, language_only, resolved_multimodal
):
    from relax.models.keye_vl2.sglang import model

    def language_init(self, config, *args):
        torch.nn.Module.__init__(self)
        self.config = config
        self.lm_head = torch.nn.Linear(2, 2)
        self.model = torch.nn.Module()
        self.model.norm = torch.nn.Identity()
        self.model.layers = []

    monkeypatch.setattr(model.Qwen3MoeForCausalLM, "__init__", language_init)
    monkeypatch.setattr(model.KeyeRMSNorm, "convert", lambda norm: None)
    monkeypatch.setattr(model, "get_mm", lambda: SimpleNamespace(enable_multimodal=enable_multimodal), raising=False)
    monkeypatch.setattr(model, "get_model", lambda: SimpleNamespace(model_path="unused", language_only=language_only))
    monkeypatch.setattr(
        model, "process_model_config", lambda: SimpleNamespace(is_multimodal=resolved_multimodal), raising=False
    )
    monkeypatch.setattr(model, "build_vision", lambda *args: (torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)))
    instance = model.KeyeVL2MoeForConditionalGeneration(SimpleNamespace())
    assert instance.multimodal_enabled is resolved_multimodal
    assert hasattr(instance, "visual") is resolved_multimodal

    monkeypatch.setattr(model.Qwen3MoeForCausalLM, "load_weights", lambda self, weights, **kwargs: list(weights))
    instance.load_weights([("visual.weight", torch.full((2, 2), 3.0))])
    if resolved_multimodal:
        torch.testing.assert_close(instance.visual.weight, torch.full((2, 2), 3.0))

    monkeypatch.setattr(model.Qwen3MoeForCausalLM, "forward", lambda *args: "language")
    monkeypatch.setattr(model, "general_mm_embed_routine", lambda *args, **kwargs: "image")
    instance.logits_processor = lambda ids, hidden, *args: hidden
    assert instance(None, None, None) == ("image" if resolved_multimodal else "language")
