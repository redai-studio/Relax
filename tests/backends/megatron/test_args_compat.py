# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Unit tests for the Megatron compatibility layer (args_compat).

These cases only cover the three compatibility semantics — field renames, value
fallbacks and capability probing — and do not need a GPU; cases that need real
Megatron skip gracefully via ``importorskip`` so CPU CI stays green.
"""

import sys
from argparse import Namespace

import pytest

from relax.backends.megatron import args_compat


def test_resolve_aliased_field_prefers_canonical_name():
    name, value = args_compat.resolve_aliased_field(Namespace(norm_epsilon=1e-6), "norm_epsilon")
    assert (name, value) == ("norm_epsilon", 1e-6)


def test_resolve_aliased_field_falls_back_to_legacy_name():
    name, value = args_compat.resolve_aliased_field(Namespace(layernorm_epsilon=1e-5), "norm_epsilon")
    assert (name, value) == ("layernorm_epsilon", 1e-5)


def test_resolve_aliased_field_reports_missing_field():
    """Neither name exists, meaning the Megatron version is unsupported: it
    must fail explicitly."""
    with pytest.raises(AttributeError, match="neither 'norm_epsilon'"):
        args_compat.resolve_aliased_field(Namespace(), "norm_epsilon")


def test_rope_type_defaults_to_rope_for_dense_attention():
    args = Namespace(rope_type=None, multi_latent_attention=False)
    args_compat.set_rope_type_default(args)
    assert args.rope_type == "rope"


def test_rope_type_defaults_to_yarn_for_mla():
    args = Namespace(rope_type=None, multi_latent_attention=True)
    args_compat.set_rope_type_default(args)
    assert args.rope_type == "yarn"


def test_rope_type_explicit_value_is_preserved():
    args = Namespace(rope_type="rope", multi_latent_attention=True)
    args_compat.set_rope_type_default(args)
    assert args.rope_type == "rope"


def test_rope_type_absent_field_is_left_alone():
    args = Namespace(multi_latent_attention=False)
    args_compat.set_rope_type_default(args)
    assert not hasattr(args, "rope_type")


def _fp16_args(**overrides):
    values = {
        "fp16": True,
        "loss_scale": None,
        "initial_loss_scale": 32768.0,
        "min_loss_scale": 1.0,
        "use_precision_aware_optimizer": True,
        "store_param_remainders": False,
    }
    values.update(overrides)
    return Namespace(**values)


def test_validate_fp16_optimizer_accepts_valid_values():
    args_compat.validate_fp16_optimizer_args(_fp16_args())


@pytest.mark.parametrize("field", ["initial_loss_scale", "min_loss_scale"])
@pytest.mark.parametrize("value", [0, -1.0, float("nan"), float("inf"), True])
def test_validate_fp16_optimizer_rejects_non_positive_loss_scale(field, value):
    with pytest.raises(ValueError, match="must be a finite number greater than 0"):
        args_compat.validate_fp16_optimizer_args(_fp16_args(**{field: value}))


def test_validate_fp16_optimizer_rejects_inverted_loss_scale():
    with pytest.raises(ValueError, match="--min-loss-scale must be less than or equal"):
        args_compat.validate_fp16_optimizer_args(_fp16_args(initial_loss_scale=1.0, min_loss_scale=2.0))


def test_validate_fp16_optimizer_rejects_non_boolean_flag():
    with pytest.raises(ValueError, match="must resolve to a boolean"):
        args_compat.validate_fp16_optimizer_args(_fp16_args(store_param_remainders=1))


def test_alias_table_documents_its_source():
    """New compatibility entries must document their source, so the table never
    becomes unattributed fallbacks."""
    for canonical, (alias, note) in args_compat.FIELD_ALIASES.items():
        assert canonical and alias and note


def test_supports_dynamic_context_parallel_requires_megatron():
    pytest.importorskip("megatron.core")
    assert isinstance(args_compat.supports_dynamic_context_parallel(), bool)


def test_fill_missing_boolean_fields_adds_absent_switch():
    """NPU-side Megatron does not declare the switch: fill False instead of
    letting direct reads raise AttributeError."""
    args = Namespace()

    args_compat.fill_missing_boolean_fields(args)

    assert args.dynamic_context_parallel is False


def test_fill_missing_boolean_fields_preserves_provided_value():
    args = Namespace(dynamic_context_parallel=True)

    args_compat.fill_missing_boolean_fields(args)

    assert args.dynamic_context_parallel is True


def test_fill_missing_boolean_fields_is_idempotent():
    args = Namespace()
    args_compat.fill_missing_boolean_fields(args)
    args_compat.fill_missing_boolean_fields(args)
    assert args.dynamic_context_parallel is False


def test_boolean_fallback_table_documents_its_reason():
    for name, (default, reason) in args_compat.MEGATRON_BOOLEAN_FALLBACKS.items():
        assert isinstance(default, bool), name
        assert reason, name


def test_uses_canonical_norm_epsilon_detects_modern_name():
    assert args_compat.uses_canonical_norm_epsilon(Namespace(norm_epsilon=1e-6)) is True


def test_uses_canonical_norm_epsilon_detects_legacy_name():
    assert args_compat.uses_canonical_norm_epsilon(Namespace(layernorm_epsilon=1e-5)) is False


def test_megatron_test_defaults_match_upstream(monkeypatch):
    """The Megatron fields mirrored on the test side must match the upstream
    defaults."""
    pytest.importorskip("megatron.training.arguments")

    from megatron.training.arguments import parse_args as megatron_parse_args

    from tests.support.megatron_defaults import MEGATRON_TEST_DEFAULTS

    # Upstream parse_args only accepts (extra_args_provider, ignore_unknown_args)
    # and reads sys.argv; isolate it explicitly so pytest's command line is not
    # consumed as Megatron arguments.
    monkeypatch.setattr(sys, "argv", ["megatron"])
    upstream = megatron_parse_args(extra_args_provider=None)
    # Upstream defaults that legitimately differ across the supported parsers:
    # profile_ranks derives [] on the pinned GPU dev parser (MCore 4a4de865),
    # while the NPU image (docker/Dockerfile.npu, core_v0.16.1) declares [0] in
    # ProfilingConfig and its argument factory uses that list as the parse
    # default. The mirror keeps the GPU value; the contract only requires the
    # upstream value to be one of the documented variants — do not widen the
    # mirror to a single value or move production defaults to match a mirror.
    allowed_upstream_variants = {"profile_ranks": ([], [0])}
    for name, value in MEGATRON_TEST_DEFAULTS.items():
        if not hasattr(upstream, name):
            # The field is provided by another parsing path in this Megatron version; the mirror value only builds test objects.
            continue
        upstream_value = getattr(upstream, name)
        if name in allowed_upstream_variants:
            assert upstream_value in allowed_upstream_variants[name], name
            continue
        assert upstream_value == value, name
