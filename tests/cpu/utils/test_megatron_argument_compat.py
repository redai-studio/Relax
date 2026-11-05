# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from argparse import Namespace

import pytest

from relax.utils.megatron_argument_compat import (
    dynamic_context_parallel_enabled,
    normalize_gloo_process_groups,
    resolve_kv_channels,
)


@pytest.mark.parametrize(
    ("options", "expected"),
    [
        ({}, False),
        ({"dynamic_context_parallel": None}, False),
        ({"dynamic_context_parallel": False}, False),
        ({"dynamic_context_parallel": True}, True),
    ],
)
def test_dynamic_context_parallel_uses_native_flag_when_present(
    options: dict[str, bool | None], expected: bool
) -> None:
    assert dynamic_context_parallel_enabled(Namespace(**options)) is expected


@pytest.mark.parametrize(
    ("options", "expected"),
    [
        ({}, False),
        ({"enable_gloo_process_groups": True}, True),
        ({"enable_gloo_process_groups": False}, False),
        ({"enable_gloo_process_groups": None}, None),
        ({"use_gloo_process_groups": False, "enable_gloo_process_groups": True}, False),
        ({"use_gloo_process_groups": None, "enable_gloo_process_groups": True}, None),
    ],
)
def test_normalize_gloo_groups_preserves_native_precedence(
    options: dict[str, bool | None], expected: bool | None
) -> None:
    args = Namespace(**options)
    normalize_gloo_process_groups(args)
    assert vars(args) == {**options, "use_gloo_process_groups": expected}
    normalize_gloo_process_groups(args)
    assert vars(args) == {**options, "use_gloo_process_groups": expected}


@pytest.mark.parametrize(
    ("options", "head_width"),
    [({}, 4), ({"kv_channels": None}, 4), ({"kv_channels": 3}, 3)],
    ids=["historical-missing", "native-unset", "explicit"],
)
def test_weight_conversion_preserves_kv_channels(options: dict[str, int | None], head_width: int) -> None:
    import torch

    from relax.backends.megatron.weight_conversion.llama import convert_llama_to_hf

    args = Namespace(hidden_size=8, num_attention_heads=2, num_query_groups=1, **options)
    combined = torch.arange(32 * head_width).reshape(4 * head_width, 8)

    name = "module.module.decoder.layers.0.self_attention.linear_qkv.weight"
    converted = dict(convert_llama_to_hf(args, name, combined))

    names = [f"model.layers.0.self_attn.{projection}_proj.weight" for projection in ("q", "k", "v")]
    expected = combined.split((2 * head_width, head_width, head_width))
    assert set(converted) == set(names)
    for name, value in zip(names, expected, strict=True):
        torch.testing.assert_close(converted[name], value)


def test_explicit_zero_kv_channels_is_not_replaced() -> None:
    assert resolve_kv_channels(Namespace(kv_channels=0)) == 0
