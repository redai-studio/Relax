# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Pin the shared rollout-sampling test defaults to the real argument parser.

Task No.13 requires test configs to reuse parameter definitions instead of
copying default tables. ``tests/cpu/agentic_obs_args.py`` is the single source
these factories read from; this test fails CI if the argparse defaults drift.
"""

import argparse
import importlib.util
import pathlib
import sys
from types import ModuleType, SimpleNamespace

import pytest

from tests.cpu.agentic_obs_args import ROLLOUT_SAMPLING_DEFAULTS

# This file lives at <repo>/tests/cpu/, so the repo root is parents[2].
ARGS_PATH = pathlib.Path(__file__).resolve().parents[2] / "relax" / "utils" / "arguments.py"

# The slime parser uses PEP 604 unions (float | None) introduced in 3.10.
_REQUIRES_PY310 = sys.version_info < (3, 10)


def _load_arguments_module(monkeypatch):
    """Load the real argument module with heavy training imports stubbed out.

    Stubs go through ``monkeypatch.setitem`` so they are restored after the
    test, mirroring the fixture in
    tests/cpu/algorithms/test_arguments_spec_driven.py.
    """
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
    eval_config.build_eval_dataset_configs = lambda *a, **k: []
    eval_config.build_named_prompt_data_configs = lambda *a, **k: []
    eval_config.ensure_dataset_list = lambda values: values or []
    monkeypatch.setitem(sys.modules, "relax.utils.training.eval_config", eval_config)

    spec = importlib.util.spec_from_file_location("_agentic_obs_arguments", ARGS_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.RouterArgs = SimpleNamespace(add_cli_args=lambda parser, **_kw: parser)
    return module


@pytest.mark.skipif(_REQUIRES_PY310, reason="arguments.py requires Python 3.10+")
def test_rollout_sampling_defaults_match_parser_declarations(monkeypatch):
    module = _load_arguments_module(monkeypatch)
    parser = argparse.ArgumentParser()
    module.get_slime_extra_args_provider()(parser)
    parsed = vars(parser.parse_args([]))

    for key, expected in ROLLOUT_SAMPLING_DEFAULTS.items():
        assert key in parsed, (
            f"{key!r} is no longer declared by the slime parser; "
            "update tests/cpu/agentic_obs_args.py"
        )
        actual = parsed[key]
        assert actual == expected, (
            f"tests/cpu/agentic_obs_args.py default for {key!r} is {expected!r}, "
            f"but relax/utils/arguments.py declares {actual!r}"
        )
