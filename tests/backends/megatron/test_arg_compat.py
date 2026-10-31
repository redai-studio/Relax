# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from types import SimpleNamespace

from relax.backends.megatron.compat import (
    ensure_megatron_rope_type,
    megatron_context_parallel_size,
    megatron_dynamic_context_parallel,
    megatron_norm_epsilon_arg_name,
)


def test_megatron_norm_epsilon_prefers_new_name():
    assert megatron_norm_epsilon_arg_name(SimpleNamespace(norm_epsilon=1e-5)) == "norm_epsilon"


def test_megatron_norm_epsilon_falls_back_to_legacy_name():
    assert megatron_norm_epsilon_arg_name(SimpleNamespace(layernorm_epsilon=1e-5)) == "layernorm_epsilon"


def test_ensure_megatron_rope_type_fills_none():
    args = SimpleNamespace(rope_type=None, multi_latent_attention=True)
    ensure_megatron_rope_type(args)
    assert args.rope_type == "yarn"

    args = SimpleNamespace(rope_type=None, multi_latent_attention=False)
    ensure_megatron_rope_type(args)
    assert args.rope_type == "rope"


def test_ensure_megatron_rope_type_noop_without_field():
    args = SimpleNamespace(multi_latent_attention=False)
    ensure_megatron_rope_type(args)
    assert not hasattr(args, "rope_type")


def test_megatron_dynamic_context_parallel_defaults_false():
    assert megatron_dynamic_context_parallel(SimpleNamespace()) is False
    assert megatron_dynamic_context_parallel(SimpleNamespace(dynamic_context_parallel=True)) is True


def test_megatron_context_parallel_size_defaults():
    assert megatron_context_parallel_size(SimpleNamespace()) == 1
    assert megatron_context_parallel_size(SimpleNamespace(context_parallel_size=4)) == 4
    assert megatron_context_parallel_size(SimpleNamespace(context_parallel_size=None)) == 1
