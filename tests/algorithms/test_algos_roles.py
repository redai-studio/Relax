# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Role registration and critic capabilities without importing Ray services."""

import importlib.util
import sys
from argparse import Namespace
from dataclasses import replace
from pathlib import Path
from types import ModuleType

import pytest

from relax.algorithms import ALGORITHM_SPECS


@pytest.fixture()
def load_registry(monkeypatch):
    """Execute the complete registry with inert component classes, on CPU."""
    components = ModuleType("relax.components")
    components.__path__ = []
    monkeypatch.setitem(sys.modules, components.__name__, components)
    classes = {}
    for module_name, class_name in {
        "actor": "Actor",
        "actor_fwd": "ActorFwd",
        "advantages": "Advantages",
        "critic": "Critic",
        "rollout": "Rollout",
        "sft": "SFT",
    }.items():
        module = ModuleType(f"relax.components.{module_name}")
        classes[class_name] = type(class_name, (), {})
        setattr(module, class_name, classes[class_name])
        monkeypatch.setitem(sys.modules, module.__name__, module)

    def load():
        path = Path(__file__).resolve().parents[2] / "relax/core/registry.py"
        spec = importlib.util.spec_from_file_location("_algorithm_test_registry", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module, classes

    return load


def _args_for(name, **overrides):
    return Namespace(
        **{
            "advantage_estimator": name,
            "multimodal_keys": None,
            "kl_coef": 0.0,
            "fully_async": False,
            "hybrid": False,
            "use_opd": False,
            "use_rollout_logprobs": False,
            "true_on_policy_mode": False,
            "debug_rollout_only": False,
            "debug_train_only": False,
            "loss_type": None,
            **overrides,
        }
    )


def _register_critic_algorithm(monkeypatch):
    name = "new_critic_algorithm"
    monkeypatch.setitem(ALGORITHM_SPECS, name, replace(ALGORITHM_SPECS["ppo"], name=name))
    return name


def test_registered_algorithms_have_the_expected_role_classes(load_registry):
    registry, classes = load_registry()
    base = {
        "actor": classes["Actor"],
        "rollout": classes["Rollout"],
        "advantages": classes["Advantages"],
        "reference": classes["ActorFwd"],
        "actor_fwd": classes["ActorFwd"],
    }
    for name, spec in ALGORITHM_SPECS.items():
        expected = {**base, "critic": classes["Critic"]} if spec.needs_critic else base
        assert registry.ALGOS[name] == expected, name
    assert registry.ALGOS["sft"] == {"sft": classes["SFT"], "actor": classes["Actor"]}


@pytest.mark.parametrize("needs_critic", [False, True])
def test_new_algorithm_is_registered_without_editing_role_tables(monkeypatch, load_registry, needs_critic):
    name = "new_algorithm"
    source = "ppo" if needs_critic else "grpo"
    monkeypatch.setitem(ALGORITHM_SPECS, name, replace(ALGORITHM_SPECS[source], name=name))
    registry, _ = load_registry()

    assert registry.ALGOS[name] == registry.ALGOS[source]
    for fully_async in (False, True):
        assert registry.process_role(_args_for(name, fully_async=fully_async)) is registry.process_role(
            _args_for(source, fully_async=fully_async)
        )


def test_each_algorithm_gets_an_independent_role_dict(load_registry):
    registry, _ = load_registry()
    assert len({id(roles) for roles in registry.ALGOS.values()}) == len(registry.ALGOS)


@pytest.mark.parametrize("consumer", ["actor", "critic", "advantages"])
def test_a_second_critic_algorithm_gets_the_critic_rollout_fields(monkeypatch, consumer):
    from relax.utils.training.data_fields import build_data_fields

    name = _register_critic_algorithm(monkeypatch)
    fields = build_data_fields(_args_for(name), consumer=consumer)
    assert fields == build_data_fields(_args_for("ppo"), consumer=consumer)
    assert ("values" in fields) is (consumer != "critic")


def test_a_second_critic_algorithm_requires_a_critic_resource(monkeypatch):
    from relax.utils.training.ppo_utils import validate_ppo_config

    name = _register_critic_algorithm(monkeypatch)
    with pytest.raises(ValueError, match="requires a 'critic' entry"):
        validate_ppo_config(_args_for(name, resource={"actor": "a"}))
    validate_ppo_config(_args_for("grpo", resource={"actor": "a"}))
