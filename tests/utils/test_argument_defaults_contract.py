# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Parameter default contract: declarations, finalization and the test builder
must stay isomorphic.

Covers the acceptance points of task No.13:
- Defaults have a single source (parameter declarations + the finalize stage);
  the test builder no longer carries its own default table.
- finalize is idempotent, filling only missing fields and never overwriting
  existing values.
- Explicit ``None`` / ``False`` / ``0`` / empty-list semantics survive building
  and finalization unchanged.
"""

import argparse

import pytest

from relax.utils.args_finalize import RUNTIME_DERIVED_DEFAULTS, finalize_args, seed_runtime_derived
from tests.support.args import build_relax_args, declared_parameter_defaults, declared_parameter_names


def test_builder_covers_every_declared_parameter():
    names = declared_parameter_names()
    defaults = declared_parameter_defaults()
    assert set(defaults) <= names
    assert "task_type" in names and "seed" in names and "advantage_estimator" in names


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("task_type", "causal_lm"),
        ("seed", 1234),
        ("advantage_estimator", "grpo"),
        ("fully_async", False),
        ("apply_chat_template_kwargs", {}),
        ("lora_rank", 0),
    ],
)
def test_builder_uses_declared_defaults(name, expected):
    assert getattr(build_relax_args(), name) == expected


def test_builder_only_accepts_known_names():
    with pytest.raises(AttributeError, match="unknown Relax argument"):
        build_relax_args(seed_misspelled=1)


def test_builder_keeps_explicit_boundary_values():
    args = build_relax_args(
        seed=0,
        sft_train_data_prefetch=False,
        apply_chat_template_kwargs=[],
        eval_size=None,
        kl_coef=0.0,
    )
    assert args.seed == 0
    assert args.sft_train_data_prefetch is False
    assert args.apply_chat_template_kwargs == []
    assert args.eval_size is None
    assert args.kl_coef == 0.0


def test_overrides_do_not_leak_between_builds():
    first = build_relax_args(seed=7)
    first.apply_chat_template_kwargs["leaked"] = True
    second = build_relax_args()
    assert second.seed == 1234
    assert second.apply_chat_template_kwargs == {}


def test_finalize_is_idempotent():
    args = build_relax_args(seed=11)
    before = dict(vars(args))
    finalize_args(args)
    assert vars(args) == before


def test_finalize_seeds_runtime_derived_attributes():
    args = argparse.Namespace()
    seed_runtime_derived(args)
    for name, (default, _) in RUNTIME_DERIVED_DEFAULTS.items():
        assert getattr(args, name) == default


def test_finalize_does_not_overwrite_runtime_values():
    args = build_relax_args(is_vl_model=True)
    finalize_args(args)
    assert args.is_vl_model is True


def test_finalize_fills_missing_optional_subparser_fields(monkeypatch):
    """Whatever parsing path runs, downstream readers see the same set of
    fields."""
    from relax.utils import args_finalize

    monkeypatch.setattr(
        args_finalize,
        "_sglang_default_attributes",
        lambda: {"sglang_dp_size": 1, "sglang_pp_size": 1},
    )
    args = argparse.Namespace(existing=True)
    finalize_args(args)
    assert args.sglang_dp_size == 1
    assert args.sglang_pp_size == 1
    assert args.existing is True


def test_finalize_keeps_explicit_subparser_values(monkeypatch):
    from relax.utils import args_finalize

    monkeypatch.setattr(args_finalize, "_sglang_default_attributes", lambda: {"sglang_dp_size": 1})
    args = argparse.Namespace(sglang_dp_size=4)
    finalize_args(args)
    assert args.sglang_dp_size == 4


def test_pre_parse_fields_are_part_of_the_builder():
    """Fields declared at the pre-parse stage also belong to the fully parsed
    parameter object."""
    args = build_relax_args(debug_rollout_only=True)
    assert args.debug_rollout_only is True
    assert args.debug_train_only is False
    assert args.train_backend == "megatron"
