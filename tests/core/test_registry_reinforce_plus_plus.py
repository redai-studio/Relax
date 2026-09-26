# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import importlib
import sys
from types import ModuleType, SimpleNamespace

import pytest


@pytest.fixture()
def registry_module(monkeypatch):
    registry_module_name = "relax.core.registry"
    registry_module_was_loaded = registry_module_name in sys.modules
    original_registry_module = sys.modules.get(registry_module_name)
    core_module = sys.modules.get("relax.core")
    registry_attr_was_set = core_module is not None and hasattr(core_module, "registry")
    original_registry_attr = getattr(core_module, "registry", None) if registry_attr_was_set else None

    component_names = {
        "actor": "Actor",
        "actor_fwd": "ActorFwd",
        "advantages": "Advantages",
        "critic": "Critic",
        "rollout": "Rollout",
        "sft": "SFT",
    }
    classes = {}
    for module_name, class_name in component_names.items():
        module = ModuleType(f"relax.components.{module_name}")
        component_class = type(class_name, (), {})
        setattr(module, class_name, component_class)
        classes[class_name] = component_class
        monkeypatch.setitem(sys.modules, module.__name__, module)

    sys.modules.pop(registry_module_name, None)
    if core_module is not None and registry_attr_was_set:
        delattr(core_module, "registry")
    module = importlib.import_module(registry_module_name)
    try:
        yield module, classes
    finally:
        sys.modules.pop(registry_module_name, None)
        if registry_module_was_loaded:
            sys.modules[registry_module_name] = original_registry_module
        if core_module is not None:
            if registry_attr_was_set:
                setattr(core_module, "registry", original_registry_attr)
            elif hasattr(core_module, "registry"):
                delattr(core_module, "registry")


@pytest.mark.parametrize("estimator", ["reinforce_plus_plus", "reinforce_plus_plus_baseline"])
def test_reinforce_plus_plus_variants_are_registered_without_critic(registry_module, estimator):
    registry, classes = registry_module

    topology = registry.ALGOS[estimator]

    assert topology[registry.ROLES.actor] is classes["Actor"]
    assert topology[registry.ROLES.rollout] is classes["Rollout"]
    assert topology[registry.ROLES.advantages] is classes["Advantages"]
    assert topology[registry.ROLES.reference] is classes["ActorFwd"]
    assert topology[registry.ROLES.actor_fwd] is classes["ActorFwd"]
    assert registry.ROLES.critic not in topology


@pytest.mark.parametrize("estimator", ["reinforce_plus_plus", "reinforce_plus_plus_baseline"])
def test_reinforce_plus_plus_variants_use_sync_colocate_roles(registry_module, estimator):
    registry, _ = registry_module
    config = SimpleNamespace(
        debug_rollout_only=False,
        debug_train_only=False,
        loss_type="policy_loss",
        advantage_estimator=estimator,
        fully_async=False,
        hybrid=False,
    )

    assert registry.process_role(config) is registry.ROLES_COLOCATE
