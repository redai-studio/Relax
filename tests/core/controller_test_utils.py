# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Isolated Controller loader for tests that do not require backend
components."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


_MISSING = object()
_REGISTRY_COMPONENTS = {
    "relax.components.actor": "Actor",
    "relax.components.actor_fwd": "ActorFwd",
    "relax.components.advantages": "Advantages",
    "relax.components.critic": "Critic",
    "relax.components.rollout": "Rollout",
    "relax.components.sft": "SFT",
}
_DCS_SERVICE_MODULES = (
    "relax.distributed.checkpoint_service",
    "relax.distributed.checkpoint_service.coordinator",
    "relax.distributed.checkpoint_service.coordinator.service",
)


def load_controller_with_stubbed_dependencies(module_name: str):
    """Load Controller without optional backend component imports."""
    saved_modules = {
        name: sys.modules.get(name, _MISSING)
        for name in [*_REGISTRY_COMPONENTS, *_DCS_SERVICE_MODULES, "relax.core.registry"]
    }
    core_module = sys.modules.get("relax.core")
    registry_attr_was_set = core_module is not None and hasattr(core_module, "registry")
    original_registry_attr = getattr(core_module, "registry", None) if registry_attr_was_set else None
    distributed_module = sys.modules.get("relax.distributed")
    checkpoint_service_attr_was_set = distributed_module is not None and hasattr(
        distributed_module, "checkpoint_service"
    )
    original_checkpoint_service_attr = (
        getattr(distributed_module, "checkpoint_service", None) if checkpoint_service_attr_was_set else None
    )
    components_module = sys.modules.get("relax.components")
    saved_component_attrs = {}
    if components_module is not None:
        for component_module_name in _REGISTRY_COMPONENTS:
            attr_name = component_module_name.rsplit(".", 1)[1]
            saved_component_attrs[attr_name] = (
                hasattr(components_module, attr_name),
                getattr(components_module, attr_name, None),
            )

    try:
        for component_module_name, class_name in _REGISTRY_COMPONENTS.items():
            module = ModuleType(component_module_name)
            setattr(module, class_name, type(class_name, (), {}))
            sys.modules[component_module_name] = module
        checkpoint_service_module = ModuleType("relax.distributed.checkpoint_service")
        checkpoint_service_module.__path__ = []
        coordinator_module = ModuleType("relax.distributed.checkpoint_service.coordinator")
        coordinator_module.__path__ = []
        dcs_service_module = ModuleType("relax.distributed.checkpoint_service.coordinator.service")
        dcs_service_module.create_dcs_deployment = lambda *_args, **_kwargs: None
        checkpoint_service_module.coordinator = coordinator_module
        coordinator_module.service = dcs_service_module
        sys.modules["relax.distributed.checkpoint_service"] = checkpoint_service_module
        sys.modules["relax.distributed.checkpoint_service.coordinator"] = coordinator_module
        sys.modules["relax.distributed.checkpoint_service.coordinator.service"] = dcs_service_module
        if distributed_module is not None:
            distributed_module.checkpoint_service = checkpoint_service_module
        sys.modules.pop("relax.core.registry", None)
        if core_module is not None and registry_attr_was_set:
            delattr(core_module, "registry")

        controller_path = Path(__file__).resolve().parents[2] / "relax" / "core" / "controller.py"
        spec = importlib.util.spec_from_file_location(module_name, controller_path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.modules.pop(module_name, None)
        for name, original_module in saved_modules.items():
            if original_module is _MISSING:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original_module
        if core_module is not None:
            if registry_attr_was_set:
                setattr(core_module, "registry", original_registry_attr)
            elif hasattr(core_module, "registry"):
                delattr(core_module, "registry")
        if distributed_module is not None:
            if checkpoint_service_attr_was_set:
                setattr(distributed_module, "checkpoint_service", original_checkpoint_service_attr)
            elif hasattr(distributed_module, "checkpoint_service"):
                delattr(distributed_module, "checkpoint_service")
        if components_module is not None:
            for attr_name, (attr_was_set, original_attr) in saved_component_attrs.items():
                if attr_was_set:
                    setattr(components_module, attr_name, original_attr)
                elif hasattr(components_module, attr_name):
                    delattr(components_module, attr_name)
