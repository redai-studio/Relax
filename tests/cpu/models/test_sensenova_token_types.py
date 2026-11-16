# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Homogeneous token dispatch must avoid scalar reads without changing model
math."""

from copy import deepcopy

import pytest
import torch


@pytest.fixture
def token_model(monkeypatch):
    pytest.importorskip("torchvision")
    from relax.models.sensenova.model import SenseNovaU1Model

    monkeypatch.setenv("RELAX_ATTENTION_IMPLEMENTATION", "torch")
    return SenseNovaU1Model(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=8,
        pad_token_id=0,
    )


def _forbid_scalar_read(tensor, *args, **kwargs):
    raise AssertionError("Homogeneous token dispatch must not read a tensor scalar")


@pytest.mark.parametrize("token_type", ["understanding", "generation"])
@pytest.mark.parametrize("checkpointing", [False, True])
def test_token_type_matches_legacy_outputs_cache_and_gradients(token_model, monkeypatch, token_type, checkpointing):
    reference = deepcopy(token_model)
    embeddings = torch.randn(1, 3, 32, requires_grad=True)
    actual_embeddings = embeddings.detach().clone().requires_grad_(True)
    indexes = torch.tensor([[0, 1, 2], [0, 0, 1], [0, 1, 0]])
    kwargs = {"indexes": indexes, "use_gradient_checkpointing": checkpointing}
    expected, expected_cache = reference(
        inputs_embeds=embeddings,
        image_gen_indicators=torch.full((1, 3), token_type == "generation", dtype=torch.bool),
        **kwargs,
    )
    expected.square().sum().backward()

    with monkeypatch.context() as context:
        context.setattr(torch.Tensor, "item", _forbid_scalar_read)
        actual, actual_cache = token_model(inputs_embeds=actual_embeddings, token_type=token_type, **kwargs)
        actual.square().sum().backward()

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual_embeddings.grad, embeddings.grad, rtol=0, atol=0)
    for actual_layer, expected_layer in zip(actual_cache.layers, expected_cache.layers, strict=True):
        torch.testing.assert_close(actual_layer.keys, expected_layer.keys, rtol=0, atol=0)
        torch.testing.assert_close(actual_layer.values, expected_layer.values, rtol=0, atol=0)
    for actual_parameter, expected_parameter in zip(token_model.parameters(), reference.parameters(), strict=True):
        assert (actual_parameter.grad is None) == (expected_parameter.grad is None)
        if expected_parameter.grad is not None:
            torch.testing.assert_close(actual_parameter.grad, expected_parameter.grad, rtol=0, atol=0)


def test_prefix_without_indicators_remains_understanding(token_model, monkeypatch):
    embeddings = torch.randn(1, 2, 32)
    indexes = torch.zeros(3, 2, dtype=torch.long)
    expected, _ = token_model(inputs_embeds=embeddings, indexes=indexes, token_type="understanding", use_cache=False)
    with monkeypatch.context() as context:
        context.setattr(torch.Tensor, "item", _forbid_scalar_read)
        actual, cache = token_model(inputs_embeds=embeddings, indexes=indexes, use_cache=False)
    assert cache is None
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_legacy_mixed_indicators_still_reject_unsupported_attention(token_model):
    with pytest.raises(NotImplementedError, match="Mixed understanding/generation decoder layer"):
        token_model(
            inputs_embeds=torch.randn(1, 2, 32),
            indexes=torch.zeros(3, 2, dtype=torch.long),
            image_gen_indicators=torch.tensor([[False, True]]),
        )


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"token_type": "mixed"}, "token_type must be"),
        (
            {"token_type": "generation", "image_gen_indicators": torch.tensor([[False, True]])},
            "Specify either token_type or image_gen_indicators",
        ),
    ],
)
def test_explicit_token_type_rejects_ambiguous_or_invalid_dispatch(token_model, kwargs, match):
    with pytest.raises(ValueError, match=match):
        token_model(inputs_embeds=torch.randn(1, 2, 32), indexes=torch.zeros(3, 2, dtype=torch.long), **kwargs)


def test_dit_generation_uses_known_type_and_preserves_prefix_cache(token_model, monkeypatch):
    from relax.models.sensenova.model import SenseNovaU1DiT

    # Exercise the production generation entry point with a small decoder and
    # image head; its vision encoders are not involved after image embeddings.
    dit = SenseNovaU1DiT.__new__(SenseNovaU1DiT)
    torch.nn.Module.__init__(dit)
    dit.language_model = torch.nn.Module()
    dit.language_model.model = token_model
    dit.fm_modules = torch.nn.ModuleDict({"fm_head": torch.nn.Conv2d(32, 12, 1)})
    dit.patch_size, dit.merge_size = 2, 1
    indexes = torch.zeros(3, 2, dtype=torch.long)
    with torch.no_grad():
        _, prefix = token_model(inputs_embeds=torch.randn(1, 2, 32), indexes=indexes)
    prefix_before = [(layer.keys.clone(), layer.values.clone()) for layer in prefix.layers]
    embeddings = torch.randn(1, 2, 32, requires_grad=True)

    with monkeypatch.context() as context:
        context.setattr(torch.Tensor, "item", _forbid_scalar_read)
        image = dit(embeddings, indexes, prefix, (4, 2), use_gradient_checkpointing=True)
        image.square().sum().backward()

    assert image.shape == (1, 2, 12)
    assert embeddings.grad is not None
    assert torch.isfinite(embeddings.grad).all()
    assert prefix.get_seq_length() == 2
    for layer, (key_before, value_before) in zip(prefix.layers, prefix_before, strict=True):
        torch.testing.assert_close(layer.keys, key_before, rtol=0, atol=0)
        torch.testing.assert_close(layer.values, value_before, rtol=0, atol=0)
