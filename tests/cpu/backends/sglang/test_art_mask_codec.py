# Copyright (c) 2026 Relax Authors. All Rights Reserved.
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from relax.backends.sglang.diffusion_engine import _deserialize


def _overlay_module(name):
    path = "python/" + name.replace(".", "/") + ".py"
    patch = (Path(__file__).resolve().parents[4] / "docker/patch/sglang/v0.5.21-sensenova-art.patch").read_text()
    section = patch.split(f"diff --git a/{path} b/{path}\n", 1)[1].split("diff --git ", 1)[0]
    source = "\n".join(
        line[1:] for line in section.splitlines() if line.startswith("+") and not line.startswith("+++")
    )
    module = types.ModuleType(name)
    exec(compile(source, path, "exec"), module.__dict__)
    return module


@pytest.fixture(autouse=True)
def install_codec(monkeypatch):
    # Execute the actual new-file implementation shipped in the image overlay.
    global codec
    name = "sglang.multimodal_gen.runtime.entrypoints.post_training.mask_codec"
    codec = _overlay_module(name)
    monkeypatch.setitem(sys.modules, name, codec)


def _maybe_serialize(value):
    if isinstance(value, torch.Tensor):
        from safetensors.torch import save

        return {"__tensor__": True, "data": save({"t": value.detach().contiguous()})}
    if isinstance(value, dict):
        return {key: _maybe_serialize(item) for key, item in value.items()}
    return value


def mask():
    value = torch.zeros(1, 1, 513, 513)
    value[:, :, :250] = -float("inf")
    return value


def test_exact_owned_mask_and_original_unchanged():
    for value in [mask(), torch.zeros_like(mask()), torch.full_like(mask(), -float("inf"))]:
        saved = value.clone()
        envelope = codec.encode_mask(value)
        restored = _deserialize(envelope)
        assert torch.equal(restored.view(torch.uint8), value.view(torch.uint8))
        restored.zero_()
        assert torch.equal(value, saved)


@pytest.mark.parametrize("bad", [-0.0, float("nan"), float("inf"), -1.0])
def test_unsupported_bit_pattern_falls_back(bad):
    value = mask()
    value[0, 0, -1, -1] = bad
    assert codec.encode_mask(value) is None
    restored = _deserialize(_maybe_serialize(value))
    assert torch.equal(restored.view(torch.uint8), value.view(torch.uint8))


def test_unsupported_layout_and_dtype_fall_back():
    value = mask()
    for v in [value.bfloat16(), value.transpose(-1, -2), value.requires_grad_(), torch.empty(0)]:
        assert codec.encode_mask(v) is None


@pytest.mark.parametrize(
    "field,value",
    [
        ("__art_mask_bitpack__", 2),
        ("shape", [1, 1, 513, 514]),
        ("numel", 1),
        ("bitorder", "big"),
        ("dtype", "torch.bfloat16"),
        ("data", b""),
    ],
)
def test_invalid_envelopes_fail(field, value):
    encoded = codec.encode_mask(mask())
    encoded[field] = value
    with pytest.raises(ValueError):
        _deserialize(encoded)


def test_padding_and_extra_bytes_fail():
    encoded = codec.encode_mask(mask())
    for data in [encoded["data"] + b"\0", encoded["data"][:-1] + b"\x80"]:
        with pytest.raises(ValueError):
            codec.decode_mask({**encoded, "data": data})


def test_only_art_leaf_encoded_without_mutating_environment():
    value = mask()
    conditions = {"prefix_attention_mask": value, "other": torch.arange(4.0)}
    env = SimpleNamespace(
        pos_cond_kwargs=conditions,
        image_kwargs={"sensenova_rollout": {"capture": "ode_states", "protocol": "sensenova_i2i_flow_v1"}},
    )
    encoded = codec.serialize_art_conditions(env, _maybe_serialize)
    assert codec.MARKER in encoded["prefix_attention_mask"]
    assert conditions["prefix_attention_mask"] is value
    assert torch.equal(_deserialize(encoded)["prefix_attention_mask"], value)
    env.image_kwargs["sensenova_rollout"]["capture"] = "terminal"
    assert "__tensor__" in codec.serialize_art_conditions(env, _maybe_serialize)["prefix_attention_mask"]


def test_rope_reuse_preserves_results_and_clears_between_forwards():
    from copy import deepcopy

    cache = _overlay_module("sglang.multimodal_gen.runtime.models.sensenova_u1.rope_cache")

    class Rotary(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("inv_freq", torch.tensor([0.5, 1.0]))
            self.attention_scaling, self.rope_type, self.calls = 1.0, "default", 0

        def forward(self, x, position_ids):
            self.calls += 1
            return (position_ids[..., None] * self.inv_freq).cos().to(x.dtype)

    class Decoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList(
                [torch.nn.ModuleDict({"rotary_emb": Rotary(), "rotary_emb_hw": Rotary()}) for _ in range(2)]
            )
            for layer in self.layers:
                layer.self_attn = SimpleNamespace(rotary_emb=layer["rotary_emb"], rotary_emb_hw=layer["rotary_emb_hw"])

        def forward(self, x, positions):
            return sum(
                layer["rotary_emb"](x, positions) + layer["rotary_emb_hw"](x, positions) for layer in self.layers
            )

    decoder = Decoder()
    reference = deepcopy(decoder)
    cache.install_rope_cache(SimpleNamespace(language_model=SimpleNamespace(model=decoder)))
    x, positions = torch.ones(1, dtype=torch.bfloat16), torch.tensor([[0.0, 1.0]])
    torch.testing.assert_close(decoder(x, positions), reference(x, positions), rtol=0, atol=0)
    assert [layer["rotary_emb"].calls for layer in decoder.layers] == [1, 0]
    positions.add_(1)  # Same storage address, different forward: the cache must expire.
    torch.testing.assert_close(decoder(x, positions), reference(x, positions), rtol=0, atol=0)
    assert [layer["rotary_emb"].calls for layer in decoder.layers] == [2, 0]
