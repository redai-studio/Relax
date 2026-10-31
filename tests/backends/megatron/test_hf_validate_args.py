# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Comparison scope of the HF parameter validation (``_hf_validate_args``).

The function runs before Megatron's default derivation and the Bridge
provider: fields not given explicitly are still ``None``, and ``--num-layers``
still allows the layer-reduced override. Validation must therefore only
compare fields that "already have a meaning at this point", otherwise it
would reject launch configurations that are otherwise legal. This file pins
that scope so later refactors cannot quietly widen it.
"""

from argparse import Namespace

import pytest


pytest.importorskip("megatron.training.arguments")

from relax.backends.megatron import arguments as megatron_arguments  # noqa: E402


HF_CONFIG = Namespace(
    hidden_size=2560,
    num_attention_heads=32,
    num_hidden_layers=4,
    intermediate_size=10240,
    tie_word_embeddings=False,
    rope_theta=1_000_000,
    rms_norm_eps=1e-6,
)


def _legacy_args(**overrides):
    """Shape of the pinned mirror (MCore 4a4de865): only layernorm_epsilon."""
    values = {
        "hidden_size": 2560,
        "num_attention_heads": 32,
        "num_layers": 2,  # layer-reduced training override
        "ffn_hidden_size": None,  # --ffn-hidden-size omitted, derived later by the Bridge
        "untie_embeddings_and_output_weights": False,
        "rotary_base": 10000,
        "layernorm_epsilon": 1e-6,
        "moe_layer_freq": None,
    }
    values.update(overrides)
    return Namespace(**values)


def _canonical_args(**overrides):
    """A Megatron version that exposes norm_epsilon."""
    values = {
        "hidden_size": 2560,
        "num_attention_heads": 32,
        "num_layers": 4,
        "ffn_hidden_size": 10240,
        "untie_embeddings_and_output_weights": True,
        "rotary_base": 1_000_000,
        "norm_epsilon": 1e-6,
        "moe_layer_freq": None,
    }
    values.update(overrides)
    return Namespace(**values)


def test_legacy_epsilon_name_only_compares_epsilon():
    """Underived fields (ffn_hidden_size=None) and the layer override must not
    be flagged as inconsistent."""
    megatron_arguments._hf_validate_args(_legacy_args(), HF_CONFIG)


def test_legacy_epsilon_mismatch_is_still_rejected():
    with pytest.raises(AssertionError, match="rms_norm_eps"):
        megatron_arguments._hf_validate_args(_legacy_args(layernorm_epsilon=1e-5), HF_CONFIG)


def test_canonical_epsilon_name_compares_the_model_table():
    """Versions exposing norm_epsilon still compare per-field with the original
    semantics."""
    megatron_arguments._hf_validate_args(_canonical_args(), HF_CONFIG)


def test_canonical_epsilon_name_reports_mismatched_field():
    with pytest.raises(AssertionError, match="hidden_size"):
        megatron_arguments._hf_validate_args(_canonical_args(hidden_size=4096), HF_CONFIG)


def test_missing_epsilon_field_is_an_explicit_error():
    """Neither name exists, meaning the Megatron version is unsupported: it
    must fail explicitly."""
    args = _legacy_args()
    del args.layernorm_epsilon
    with pytest.raises(AttributeError, match="neither 'norm_epsilon'"):
        megatron_arguments._hf_validate_args(args, HF_CONFIG)
