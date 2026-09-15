# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from types import SimpleNamespace

import pytest

from tests.utils.test_arguments_opd_teacher_colocate import (
    arguments_module as _arguments_module_fixture,
)


arguments_module = _arguments_module_fixture


def test_sft_async_prepack_rejects_single_in_flight_step(arguments_module):
    args = SimpleNamespace(sft_max_in_flight_steps=1, sft_async_prepack=True, max_staleness=0)

    with pytest.raises(ValueError, match="requires --sft-max-in-flight-steps >= 2"):
        arguments_module._normalize_sft_max_in_flight_steps(args, is_sft=True)


def test_sft_async_prepack_rejects_zero_max_staleness_without_alias(arguments_module):
    args = SimpleNamespace(sft_max_in_flight_steps=None, sft_async_prepack=True, max_staleness=0)

    with pytest.raises(ValueError, match="requires --max-staleness >= 1"):
        arguments_module._normalize_sft_max_in_flight_steps(args, is_sft=True)


def test_sft_async_prepack_maps_two_in_flight_steps_to_one_stale_step(arguments_module):
    args = SimpleNamespace(sft_max_in_flight_steps=2, sft_async_prepack=True, max_staleness=0)

    arguments_module._normalize_sft_max_in_flight_steps(args, is_sft=True)

    assert args.max_staleness == 1


def test_sft_without_async_prepack_allows_one_in_flight_step(arguments_module):
    args = SimpleNamespace(sft_max_in_flight_steps=1, sft_async_prepack=False, max_staleness=0)

    arguments_module._normalize_sft_max_in_flight_steps(args, is_sft=True)

    assert args.max_staleness == 0


def test_sft_train_data_prefetch_rejects_async_prepack(arguments_module):
    args = SimpleNamespace(
        sft_train_data_prefetch=True,
        sft_async_prepack=True,
        per_rank_fetch=True,
        max_staleness=1,
    )

    with pytest.raises(ValueError, match="mutually exclusive"):
        arguments_module._validate_sft_train_data_prefetch(args, is_sft=True)


def test_sft_train_data_prefetch_allows_raw_only_mode(arguments_module):
    args = SimpleNamespace(
        sft_train_data_prefetch=True,
        sft_async_prepack=False,
        per_rank_fetch=True,
        max_staleness=1,
    )

    arguments_module._validate_sft_train_data_prefetch(args, is_sft=True)
