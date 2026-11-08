# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Exercise real sampler wiring without constructing Ray services."""

import argparse
import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


ROOT = Path(__file__).resolve().parents[2]


def make_args(**overrides):
    values = dict(
        balance_by_flops=False,
        loss_type="sft",
        train_backend="megatron",
        use_dynamic_batch_size=True,
        n_samples_per_prompt=1,
        max_staleness=0,
        max_tokens_per_gpu=8192,
        sft_async_prepack=False,
        distributed_backend="nccl",
        qkv_format="thd",
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        virtual_pipeline_model_parallel_size=None,
        multimodal_keys=None,
        hf_checkpoint="unused-config-source",
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def choose_sampler(args, dp_size=2):
    from transfer_queue import SeqlenBalancedSampler

    from relax.engine.sft.bootstrap import resolve_sft_algo_key

    path = ROOT / "relax/core/controller.py"
    tree = ast.parse(path.read_text())
    method = next(
        node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "_initialize_data_system"
    )
    end = next(
        i
        for i, node in enumerate(method.body)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "controller_config" for target in node.targets)
    )
    namespace = dict(
        self=SimpleNamespace(config=args),
        resolve_sft_algo_key=resolve_sft_algo_key,
        compute_dp_size=lambda config: dp_size,
        with_control_plane_affinity=lambda config: {},
        SeqlenBalancedSampler=SeqlenBalancedSampler,
        IdentityWindowSampler=Mock(side_effect=AssertionError("ordinary SFT must use its SFT sampler")),
        logger=Mock(),
    )
    exec(compile(ast.Module(body=method.body[:end], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["sampler"]


def test_real_cli_flag_is_default_off_and_opt_in():
    path = ROOT / "relax/utils/arguments.py"
    calls = [
        node
        for node in ast.walk(ast.parse(path.read_text()))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_argument"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "--balance-by-flops"
    ]
    assert len(calls) == 1
    parser = argparse.ArgumentParser()
    module = ast.fix_missing_locations(ast.Module(body=[ast.Expr(value=calls[0])], type_ignores=[]))
    exec(compile(module, str(path), "exec"), dict(parser=parser))
    assert parser.parse_args([]).balance_by_flops is False
    assert parser.parse_args(["--balance-by-flops"]).balance_by_flops is True


def test_disabled_sft_wiring_keeps_real_baseline_sampler_without_loading_model(monkeypatch):
    from transfer_queue import SeqlenBalancedSampler

    from relax.utils.data import flops_config

    cost = Mock(side_effect=AssertionError("disabled flag must not load model config"))
    monkeypatch.setattr(flops_config, "get_sample_flops_cost", cost)
    args = make_args()
    sampler = choose_sampler(args)
    assert type(sampler) is SeqlenBalancedSampler
    assert sampler.dp_size == 2
    cost.assert_not_called()
    assert not hasattr(args, "_sample_flops_cost")


def test_enabled_sft_wiring_installs_real_adapter_and_loads_config_once(monkeypatch):
    from transfer_queue import SeqlenBalancedSampler

    from relax.utils import misc
    from relax.utils.data.flops_sampler import FlopsBalancedSampler

    hf_config = SimpleNamespace(
        model_type="qwen3",
        hidden_size=128,
        vocab_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=256,
    )
    loader = Mock(return_value=hf_config)
    monkeypatch.setattr(misc, "get_hf_config", loader)
    args = make_args(balance_by_flops=True)
    first = choose_sampler(args)
    second = choose_sampler(args)
    assert isinstance(first, FlopsBalancedSampler)
    assert type(first.base) is SeqlenBalancedSampler
    assert first.base.dp_size == 2
    assert first.cost_model is second.cost_model
    assert first.cost_model is args._sample_flops_cost
    loader.assert_called_once_with(args.hf_checkpoint)


def test_late_pipeline_rejects_automatic_prepack_without_changing_budget():
    from relax.engine.sft.data_pipeline_config import configure_sft_data_pipeline

    args = make_args(balance_by_flops=True, max_staleness=1)
    with pytest.raises(ValueError, match="ordinary SFT data path.*async prepack.*sft-max-in-flight-steps 1"):
        configure_sft_data_pipeline(args)
    assert args.max_staleness == 1
    assert args.sft_async_prepack is True


def test_late_pipeline_accepts_explicit_ordinary_path_without_changing_budget():
    from relax.engine.sft.data_pipeline_config import configure_sft_data_pipeline

    args = make_args(balance_by_flops=True, max_staleness=0)
    configure_sft_data_pipeline(args)
    assert args.max_staleness == 0
    assert args.sft_async_prepack is False
    assert args.sft_train_data_prefetch is False


def test_disabled_flag_preserves_automatic_prepack():
    from relax.engine.sft.data_pipeline_config import configure_sft_data_pipeline

    args = make_args(max_staleness=1)
    configure_sft_data_pipeline(args)
    assert args.max_staleness == 1
    assert args.sft_async_prepack is True
