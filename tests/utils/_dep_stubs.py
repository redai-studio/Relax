# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Test-only dependency stubs for CPU unit tests.

The sandbox running these tests may lack heavy optional dependencies
(fastapi/pydantic/ray/transformers/...). When a module is genuinely
importable it is used as-is (so full CI environments exercise the real
code); otherwise a minimal attribute-compatible stub is installed into
``sys.modules`` so the module under test can be imported and its logic
driven without a Ray/FastAPI runtime. Stubs are import-shims only -- they
never fake behavior of the code under test.

Run: python -m unittest tests.utils.test_genrm_scale_registry -v
"""

import importlib
import sys
import types
from typing import Any, Dict, Optional


def _install(name: str, **attrs: Any) -> types.ModuleType:
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    # Mark as a package so submodule imports (ray.serve, ...) resolve.
    module.__path__ = []  # type: ignore[attr-defined]
    sys.modules[name] = module
    return module


def _ensure(name: str, **attrs: Any) -> bool:
    """Use the real module when importable; otherwise install a stub."""
    try:
        importlib.import_module(name)
        return True
    except Exception:
        _install(name, **attrs)
        return False


# ---------------------------------------------------------------------------
# Minimal pydantic stand-in: subclasses collect class-level Field defaults,
# then kwargs override them on instantiation.
# ---------------------------------------------------------------------------


class _StubBaseModel:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        for klass in reversed(type(self).__mro__):
            for key, value in vars(klass).items():
                if not key.startswith("__"):
                    setattr(self, key, value)
        for key, value in kwargs.items():
            setattr(self, key, value)

    def dict(self) -> Dict[str, Any]:
        return {k: v for k, v in vars(self).items() if not k.startswith("_")}


def _stub_field(default: Any = None, default_factory: Any = None, **kwargs: Any) -> Any:
    if default_factory is not None:
        return default_factory()
    return default


class _StubFastAPI:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    @staticmethod
    def _route(fn):
        return fn

    def get(self, *args: Any, **kwargs: Any):
        return self._route

    def post(self, *args: Any, **kwargs: Any):
        return self._route

    def patch(self, *args: Any, **kwargs: Any):
        return self._route

    def put(self, *args: Any, **kwargs: Any):
        return self._route

    def delete(self, *args: Any, **kwargs: Any):
        return self._route


class _StubHTTPException(Exception):
    def __init__(self, status_code: Optional[int] = None, detail: Any = None) -> None:
        self.status_code = status_code
        self.detail = detail
        super().__init__(detail)


def _stub_deployment(*args: Any, **kwargs: Any):
    def decorate(cls):
        cls.func_or_class = cls
        return cls

    if args and callable(args[0]):
        return decorate(args[0])
    return decorate


class _StubServe:
    deployment = staticmethod(_stub_deployment)

    @staticmethod
    def ingress(app: Any):
        def decorate(cls):
            return cls

        return decorate


def _stub_remote(*args: Any, **kwargs: Any):
    if args and callable(args[0]):
        return args[0]

    def decorate(cls):
        return cls

    return decorate


def install_web_framework_stubs() -> None:
    """Ensure fastapi/pydantic/ray are importable (real or stubbed)."""
    _ensure("fastapi", FastAPI=_StubFastAPI, HTTPException=_StubHTTPException, Request=object, Response=object)
    _ensure("fastapi.responses", StreamingResponse=object)
    _ensure("pydantic", BaseModel=_StubBaseModel, Field=_stub_field)
    _ensure("ray", serve=_StubServe, remote=_stub_remote, get=lambda ref, timeout=None, **kwargs: ref)
    _ensure("ray.serve")
    _ensure("ray.serve.schema", LoggingConfig=lambda **kwargs: types.SimpleNamespace(**kwargs))
    # ray.util submodules referenced by the placement-group import chain.
    _ensure("ray.util")
    _ensure("ray.util.placement_group", PlacementGroup=object)
    _ensure(
        "ray.util.scheduling_strategies",
        NodeAffinitySchedulingStrategy=object,
        PlacementGroupSchedulingStrategy=object,
    )


def import_autoscaler_service():
    """Import relax.utils.autoscaler.autoscaler_service without heavy deps."""
    install_web_framework_stubs()
    return importlib.import_module("relax.utils.autoscaler.autoscaler_service")


def import_genrm_component():
    """Import relax.components.genrm without Ray/SGLang/transformers.

    The deep import chain (placement_group -> actor_group -> utils ->
    tensordict/megatron) is cut at the two bridge modules genrm.py uses:
    ``create_genrm_managers`` and ``load_tokenizer``.
    """
    try:
        return importlib.import_module("relax.components.genrm")
    except Exception:
        pass
    install_web_framework_stubs()
    _install("relax.distributed.ray.placement_group", create_genrm_managers=lambda *a, **k: {})
    _install("relax.utils.data.processing_utils", load_tokenizer=lambda *a, **k: None)
    return importlib.import_module("relax.components.genrm")


def import_genrm_manager():
    """Import relax.distributed.ray.genrm without SGLang/engine deps.

    The manager class needs its base class (multi_engine_manager, which
    evaluates ``ray.exceptions.RayActorError`` at import time) and the
    ``GenRMEngine`` actor class only as a constructor argument, so the deep
    SGLang chain can be cut at that bridge. The real module is used whenever it
    is importable (full CI environments).
    """
    try:
        return importlib.import_module("relax.distributed.ray.genrm")
    except Exception:
        pass
    install_web_framework_stubs()
    import ray

    if not hasattr(ray, "exceptions"):
        exceptions = types.ModuleType("ray.exceptions")
        exceptions.RayActorError = type("RayActorError", (Exception,), {})
        exceptions.RayTaskError = type("RayTaskError", (Exception,), {})
        sys.modules["ray.exceptions"] = exceptions
        ray.exceptions = exceptions
    _install("relax.backends.sglang.sglang_engine", GenRMEngine=type("GenRMEngine", (), {}))
    return importlib.import_module("relax.distributed.ray.genrm")
