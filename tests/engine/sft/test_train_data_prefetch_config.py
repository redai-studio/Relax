# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Exercise automatic SFT pipeline selection without importing GPU
libraries."""

import argparse
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from relax.engine.sft.data_pipeline_config import configure_sft_data_pipeline


_SOURCE = Path(__file__).resolve().parents[3] / "relax/utils/arguments.py"


def make_args(**overrides):
    values = dict(
        loss_type="sft",
        train_backend="megatron",
        max_staleness=1,
        per_rank_fetch=False,
        sft_async_prepack=False,
        sft_train_data_prefetch=False,
        distributed_backend="nccl",
        qkv_format="thd",
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        virtual_pipeline_model_parallel_size=None,
        multimodal_keys=None,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize(
    "overrides,prepack,prefetch,image",
    [
        ({}, True, False, False),
        ({"max_staleness": 0}, False, False, False),
        ({"multimodal_keys": {"image": "images"}}, False, True, True),
        ({"pipeline_model_parallel_size": 2}, False, True, False),
        ({"context_parallel_size": 2}, False, True, False),
        ({"virtual_pipeline_model_parallel_size": 2}, False, True, False),
        ({"dynamic_context_parallel": True}, False, True, False),
        ({"distributed_backend": "hccl"}, False, True, False),
        ({"distributed_backend": "gloo"}, False, True, False),
        ({"qkv_format": "bshd"}, False, True, False),
        ({"use_routing_replay": True}, False, True, False),
        ({"use_rollout_routing_replay": True}, False, True, False),
        ({"use_rollout_indexer_replay": True}, False, True, False),
        ({"train_backend": "fsdp"}, False, False, False),
    ],
)
def test_pipeline_uses_capabilities_and_preserves_budget(overrides, prepack, prefetch, image):
    args = make_args(**overrides)
    configure_sft_data_pipeline(args)
    assert args.per_rank_fetch is (args.train_backend == "megatron")
    assert args.sft_async_prepack is prepack
    assert args.sft_train_data_prefetch is prefetch
    assert args.sft_image_preprocess_on_rank is image
    assert args.max_staleness == overrides.get("max_staleness", 1)


@pytest.mark.parametrize("enabled", [False, True])
def test_legacy_switches_do_not_override_automatic_selection(enabled):
    args = make_args(per_rank_fetch=enabled, sft_async_prepack=enabled, sft_train_data_prefetch=enabled)
    configure_sft_data_pipeline(args)
    assert args.per_rank_fetch and args.sft_async_prepack
    assert not args.sft_train_data_prefetch
    args.max_staleness = 0
    configure_sft_data_pipeline(args)
    assert not args.sft_async_prepack and not args.sft_train_data_prefetch


@pytest.mark.parametrize("per_rank", [False, True])
def test_rl_preserves_explicit_per_rank_fetch(per_rank):
    args = make_args(loss_type="grpo", per_rank_fetch=per_rank)
    configure_sft_data_pipeline(args)
    assert args.per_rank_fetch is per_rank
    assert not args.sft_async_prepack
    assert not args.sft_train_data_prefetch
    assert not args.sft_image_preprocess_on_rank


def test_pipeline_accepts_partial_non_sft_arguments():
    args = SimpleNamespace(train_backend="megatron", per_rank_fetch=True)
    configure_sft_data_pipeline(args)
    assert args.per_rank_fetch is True
    assert args.sft_train_data_prefetch is False
    assert args.sft_image_preprocess_on_rank is False
    assert not hasattr(args, "loss_type")
    assert not hasattr(args, "sft_async_prepack")


def test_pipeline_selection_runs_after_custom_and_backend_validation():
    tree = ast.parse(_SOURCE.read_text())
    parse = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_parse_args_impl")
    start = next(
        i
        for i, n in enumerate(parse.body)
        if isinstance(n, ast.Expr)
        and isinstance(n.value, ast.Call)
        and isinstance(n.value.func, ast.Name)
        and n.value.func.id == "slime_validate_args"
    )
    end = next(
        i
        for i, n in enumerate(parse.body)
        if isinstance(n, ast.Expr)
        and isinstance(n.value, ast.Call)
        and isinstance(n.value.func, ast.Name)
        and n.value.func.id == "configure_sft_data_pipeline"
    )
    args = make_args(debug_rollout_only=False)

    def custom(args):
        args.sft_async_prepack = True
        args.sft_train_data_prefetch = True

    def backend(args):
        args.context_parallel_size = 2
        return args

    namespace = dict(
        args=args,
        slime_validate_args=custom,
        megatron_validate_args=backend,
        configure_sft_data_pipeline=configure_sft_data_pipeline,
    )
    exec(compile(ast.Module(body=parse.body[start : end + 1], type_ignores=[]), str(_SOURCE), "exec"), namespace)
    assert args.sft_train_data_prefetch and not args.sft_async_prepack


def test_legacy_flags_are_accepted_but_hidden():
    tree = ast.parse(_SOURCE.read_text())
    flags = {"--per-rank-fetch", "--sft-async-prepack", "--sft-train-data-prefetch"}
    calls = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "add_argument"
        and n.args
        and isinstance(n.args[0], ast.Constant)
        and n.args[0].value in flags
    ]
    parser = argparse.ArgumentParser()
    module = ast.fix_missing_locations(ast.Module(body=[ast.Expr(n) for n in calls], type_ignores=[]))
    exec(compile(module, str(_SOURCE), "exec"), dict(parser=parser, argparse=argparse))
    assert all(vars(parser.parse_args(sorted(flags))).values())
    assert all(flag not in parser.format_help() for flag in flags)


@pytest.mark.parametrize("steps,expected", [(1, 0), (2, 1), (4, 3)])
def test_late_custom_budget_controls_automatic_pipeline(steps, expected):
    tree = ast.parse(_SOURCE.read_text())
    normalize = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_normalize_sft_max_in_flight_steps"
    )
    validate = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "slime_validate_args")
    names = {"apply_custom_config_overrides", "_normalize_sft_max_in_flight_steps"}
    calls = [
        n
        for n in validate.body
        if isinstance(n, ast.Expr)
        and isinstance(n.value, ast.Call)
        and isinstance(n.value.func, ast.Name)
        and n.value.func.id in names
    ]
    args = make_args(sft_max_in_flight_steps=None)
    namespace = dict(
        args=args,
        is_sft=True,
        apply_custom_config_overrides=lambda args: setattr(args, "sft_max_in_flight_steps", steps),
    )
    exec(compile(ast.Module(body=[normalize, *calls], type_ignores=[]), str(_SOURCE), "exec"), namespace)
    configure_sft_data_pipeline(args)
    assert args.max_staleness == expected
    assert args.sft_async_prepack is (steps >= 2)
