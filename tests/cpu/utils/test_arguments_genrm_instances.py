# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Normalization of --genrm-instances vs the legacy single-model genRM flags.

``_resolve_genrm_instances`` is the single source of truth downstream code
(``register_genrm``, ``create_genrm_managers``, the GPU split/shared check)
relies on to treat single- and multi-instance genRM configs uniformly. It is
tested standalone here (not through the full ``slime_validate_args`` pipeline)
because it has no dependency on the rest of the argument surface.
"""

import importlib
import sys
from types import ModuleType, SimpleNamespace

import pytest


@pytest.fixture()
def _arguments_module(monkeypatch):
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

    sys.modules.pop("relax.utils.arguments", None)
    module = importlib.import_module("relax.utils.arguments")
    yield module
    sys.modules.pop("relax.utils.arguments", None)


@pytest.fixture()
def _resolve_genrm_instances(_arguments_module):
    return _arguments_module._resolve_genrm_instances


def _args(**overrides):
    defaults = dict(
        genrm_instances=None,
        genrm_model_path=None,
        genrm_num_gpus=1,
        genrm_num_gpus_per_engine=1,
        genrm_engine_config=None,
        genrm_sampling_config=None,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_genrm_disabled_resolves_to_empty(_resolve_genrm_instances):
    assert _resolve_genrm_instances(_args()) == {}


def test_legacy_single_model_path_normalizes_to_default_key(_resolve_genrm_instances):
    args = _args(genrm_model_path="/model", genrm_num_gpus=4, genrm_num_gpus_per_engine=2)

    resolved = _resolve_genrm_instances(args)

    assert list(resolved.keys()) == ["__default__"]
    assert resolved["__default__"] == {
        "model_path": "/model",
        "num_gpus": 4,
        "num_gpus_per_engine": 2,
        "engine_config": {},
        "sampling_config": {},
    }


def test_legacy_single_model_path_keeps_existing_engine_and_sampling_config(_resolve_genrm_instances):
    args = _args(
        genrm_model_path="/model",
        genrm_engine_config={"mem_fraction_static": 0.3},
        genrm_sampling_config={"temperature": 0.5},
    )

    resolved = _resolve_genrm_instances(args)

    assert resolved["__default__"]["engine_config"] == {"mem_fraction_static": 0.3}
    assert resolved["__default__"]["sampling_config"] == {"temperature": 0.5}


def test_genrm_instances_takes_priority_over_legacy_model_path(_resolve_genrm_instances, caplog):
    args = _args(
        genrm_instances={"quality": {"model_path": "/q", "num_gpus": 4}},
        genrm_model_path="/legacy",
    )

    resolved = _resolve_genrm_instances(args)

    assert list(resolved.keys()) == ["quality"]
    assert resolved["quality"]["model_path"] == "/q"


def test_genrm_instances_per_instance_falls_back_to_global_defaults(_resolve_genrm_instances):
    args = _args(
        genrm_instances={
            "quality": {"model_path": "/q", "num_gpus": 4},
            "safety": {"model_path": "/s", "num_gpus": 2, "num_gpus_per_engine": 2},
        },
        genrm_num_gpus_per_engine=1,
        genrm_engine_config={"mem_fraction_static": 0.3},
        genrm_sampling_config={"temperature": 0.5},
    )

    resolved = _resolve_genrm_instances(args)

    assert resolved["quality"]["num_gpus_per_engine"] == 1  # fell back to the global default
    assert resolved["safety"]["num_gpus_per_engine"] == 2  # explicit override kept
    assert resolved["quality"]["engine_config"] == {"mem_fraction_static": 0.3}
    assert resolved["quality"]["sampling_config"] == {"temperature": 0.5}


def test_genrm_instances_requires_model_path(_resolve_genrm_instances):
    args = _args(genrm_instances={"quality": {"num_gpus": 4}})

    with pytest.raises(ValueError, match="model_path"):
        _resolve_genrm_instances(args)


def test_genrm_instances_requires_explicit_num_gpus_no_implicit_split(_resolve_genrm_instances):
    """Each instance must state its own GPU budget -- there is no automatic
    even split across instances, unlike MOPD's teacher budget split."""
    args = _args(genrm_instances={"quality": {"model_path": "/q"}})

    with pytest.raises(ValueError, match="num_gpus"):
        _resolve_genrm_instances(args)


def test_genrm_instances_rejects_reserved_default_key(_resolve_genrm_instances):
    args = _args(genrm_instances={"__default__": {"model_path": "/q", "num_gpus": 1}})

    with pytest.raises(ValueError, match="reserved"):
        _resolve_genrm_instances(args)


def test_genrm_instances_supports_heterogeneous_gpu_budgets(_resolve_genrm_instances):
    args = _args(
        genrm_instances={
            "quality": {"model_path": "/q", "num_gpus": 4},
            "safety": {"model_path": "/s", "num_gpus": 2},
        }
    )

    resolved = _resolve_genrm_instances(args)

    assert resolved["quality"]["num_gpus"] == 4
    assert resolved["safety"]["num_gpus"] == 2


def test_genrm_resource_must_match_instance_gpu_sum(_arguments_module):
    args = _args(
        genrm_instances={"quality": {"model_path": "/q", "num_gpus": 2}},
        resource={"genrm": [1, 1]},
    )
    resolved = _arguments_module._resolve_genrm_instances(args)

    with pytest.raises(ValueError, match="must equal the sum"):
        _arguments_module._validate_genrm_resource_config(args, resolved)
