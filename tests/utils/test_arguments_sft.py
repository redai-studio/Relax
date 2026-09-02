# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from types import SimpleNamespace

import pytest

from relax.utils.arguments import _normalize_sft_max_in_flight_steps


def test_sft_async_prepack_rejects_single_in_flight_step():
    args = SimpleNamespace(sft_max_in_flight_steps=1, sft_async_prepack=True, max_staleness=0)

    with pytest.raises(ValueError, match="requires --sft-max-in-flight-steps >= 2"):
        _normalize_sft_max_in_flight_steps(args, is_sft=True)


def test_sft_async_prepack_rejects_zero_max_staleness_without_alias():
    args = SimpleNamespace(sft_max_in_flight_steps=None, sft_async_prepack=True, max_staleness=0)

    with pytest.raises(ValueError, match="requires --max-staleness >= 1"):
        _normalize_sft_max_in_flight_steps(args, is_sft=True)


def test_sft_async_prepack_maps_two_in_flight_steps_to_one_stale_step():
    args = SimpleNamespace(sft_max_in_flight_steps=2, sft_async_prepack=True, max_staleness=0)

    _normalize_sft_max_in_flight_steps(args, is_sft=True)

    assert args.max_staleness == 1


def test_sft_without_async_prepack_allows_one_in_flight_step():
    args = SimpleNamespace(sft_max_in_flight_steps=1, sft_async_prepack=False, max_staleness=0)

    _normalize_sft_max_in_flight_steps(args, is_sft=True)

    assert args.max_staleness == 0
