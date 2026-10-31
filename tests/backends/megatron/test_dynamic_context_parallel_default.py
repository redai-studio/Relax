# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Startup path when ``dynamic_context_parallel`` is missing.

The NPU-side Megatron branch does not declare ``--dynamic-context-parallel``,
so the parsed namespace has no such attribute at all, while Relax validation
and business code read it directly. This file pins the fill semantics of the
compatibility layer: only missing fields are filled, provided values are kept,
and enabling still goes through the capability check.
"""

from argparse import Namespace

import pytest


pytest.importorskip("megatron.training.arguments")

from relax.backends.megatron import arguments as megatron_arguments  # noqa: E402


def _npu_shaped_args() -> Namespace:
    """Replicates the NPU parser output: no dynamic_context_parallel,
    everything else present."""
    return Namespace(
        context_parallel_size=1,
        use_dynamic_batch_size=False,
        variable_seq_lengths=None,
        moe_token_dispatcher_type="alltoall",
        pipeline_model_parallel_size=1,
        decoder_first_pipeline_num_layers=None,
        decoder_last_pipeline_num_layers=None,
    )


@pytest.fixture()
def without_upstream_validation(monkeypatch):
    """Skips upstream validate_args and device probing, keeping only Relax's
    own branching."""
    monkeypatch.setattr(megatron_arguments, "_megatron_validate_args", lambda args: None)
    monkeypatch.setattr(megatron_arguments.device_utils, "is_available", lambda: True)


def test_validate_args_fills_missing_dynamic_context_parallel(without_upstream_validation):
    """Regression: with the field missing, the old code raised AttributeError at args.dynamic_context_parallel."""
    args = _npu_shaped_args()
    assert not hasattr(args, "dynamic_context_parallel")

    megatron_arguments.validate_args(args)

    assert args.dynamic_context_parallel is False


def test_validate_args_keeps_explicit_value(without_upstream_validation, monkeypatch):
    """A value provided explicitly is not overwritten; enabling still requires
    Megatron support."""
    monkeypatch.setattr(
        megatron_arguments,
        "supports_dynamic_context_parallel",
        lambda: False,
    )
    args = _npu_shaped_args()
    args.dynamic_context_parallel = True

    with pytest.raises(AssertionError, match="dynamic_context_parallel is not supported"):
        megatron_arguments.validate_args(args)
