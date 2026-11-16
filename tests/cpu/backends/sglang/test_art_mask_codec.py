# Copyright (c) 2026 Relax Authors. All Rights Reserved.
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from relax.backends.sglang.diffusion_engine import _deserialize


@pytest.fixture(autouse=True)
def install_codec(monkeypatch):
    # Exercise the shipped patch, including its actual import path at the receiver.
    global codec
    name = "sglang.multimodal_gen.runtime.entrypoints.post_training.mask_codec"
    path = "python/" + name.replace(".", "/") + ".py"
    patch = (Path(__file__).resolve().parents[4] / "docker/patch/sglang/v0.5.21.patch").read_text()
    section = patch.split(f"diff --git a/{path} b/{path}\n", 1)[1].split("diff --git ", 1)[0]
    source = "\n".join(
        line[1:] for line in section.splitlines() if line.startswith("+") and not line.startswith("+++")
    )
    codec = types.ModuleType(name)
    exec(compile(source, path, "exec"), codec.__dict__)
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
