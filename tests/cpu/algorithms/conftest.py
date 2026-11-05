# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import argparse
import importlib.util
import pathlib
import sys
from types import ModuleType, SimpleNamespace

import pytest


ARGS_PATH = pathlib.Path(__file__).resolve().parents[3] / "relax" / "utils" / "arguments.py"


@pytest.fixture()
def arguments_module(monkeypatch):
    """Load real argument parsing while isolating optional training imports."""
    router_pkg = ModuleType("sglang_router")
    launch_router = ModuleType("sglang_router.launch_router")
    launch_router.RouterArgs = object
    monkeypatch.setitem(sys.modules, "sglang_router", router_pkg)
    monkeypatch.setitem(sys.modules, "sglang_router.launch_router", launch_router)

    sglang_arguments = ModuleType("relax.backends.sglang.arguments")
    sglang_arguments.sglang_parse_args = lambda: None
    sglang_arguments.validate_args = lambda args: args
    monkeypatch.setitem(sys.modules, "relax.backends.sglang.arguments", sglang_arguments)

    device = ModuleType("relax.utils.device")
    device.get_dist_backend = lambda: "gloo"
    monkeypatch.setitem(sys.modules, "relax.utils.device", device)

    eval_config = ModuleType("relax.utils.training.eval_config")
    eval_config.EvalDatasetConfig = dict
    eval_config.build_eval_dataset_configs = lambda args, datasets_config, defaults: []
    eval_config.build_named_prompt_data_configs = lambda values: []
    eval_config.ensure_dataset_list = lambda values: values or []
    monkeypatch.setitem(sys.modules, "relax.utils.training.eval_config", eval_config)

    spec = importlib.util.spec_from_file_location("_algorithm_test_arguments", ARGS_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def algorithm_args_parser(arguments_module) -> argparse.ArgumentParser:
    """Build a fresh parser from the Relax declarations."""
    arguments_module.RouterArgs = SimpleNamespace(add_cli_args=lambda parser, **_kwargs: parser)
    parser = argparse.ArgumentParser()
    arguments_module.get_slime_extra_args_provider()(parser)
    return parser
