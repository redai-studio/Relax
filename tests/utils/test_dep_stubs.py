# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Import bridges must not advertise missing engine dependencies to other
tests."""

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

from tests.utils import _dep_stubs


def test_genrm_import_uses_available_runtime_without_stubs(monkeypatch):
    manager = ModuleType("relax.distributed.ray.genrm")
    importer = Mock(return_value=manager)
    install_web = Mock(side_effect=AssertionError("available runtime must not be stubbed"))
    install_engine = Mock(side_effect=AssertionError("available engine must not be stubbed"))
    monkeypatch.setattr(_dep_stubs.importlib, "import_module", importer)
    monkeypatch.setattr(_dep_stubs, "install_web_framework_stubs", install_web)
    monkeypatch.setattr(_dep_stubs, "_install", install_engine)
    assert _dep_stubs.import_genrm_manager() is manager
    importer.assert_called_once_with(manager.__name__)
    install_web.assert_not_called()
    install_engine.assert_not_called()


@pytest.mark.parametrize("existing", ["absent", "module_only", "parent_only", "both"])
@pytest.mark.parametrize("retry_fails", [False, True])
def test_genrm_import_restores_engine_bindings(monkeypatch, existing, retry_fails):
    engine_name = "relax.backends.sglang.sglang_engine"
    manager_name = "relax.distributed.ray.genrm"
    parent = ModuleType("relax.backends.sglang")
    original_module = ModuleType(engine_name)
    original_attribute = ModuleType(engine_name)
    manager = ModuleType(manager_name)
    if existing in {"module_only", "both"}:
        monkeypatch.setitem(sys.modules, engine_name, original_module)
    else:
        monkeypatch.delitem(sys.modules, engine_name, raising=False)
    if existing in {"parent_only", "both"}:
        parent.sglang_engine = original_attribute
    monkeypatch.setitem(sys.modules, parent.__name__, parent)
    monkeypatch.setitem(sys.modules, "ray", SimpleNamespace(exceptions=SimpleNamespace()))
    monkeypatch.setattr(_dep_stubs, "install_web_framework_stubs", lambda: None)
    attempts = 0

    def import_module(name):
        nonlocal attempts
        if name == parent.__name__:
            return parent
        assert name == manager_name
        attempts += 1
        if attempts == 1:
            raise ImportError("optional engine dependency unavailable")
        bridge = sys.modules[engine_name]
        assert bridge is parent.sglang_engine
        assert bridge is not original_module
        assert hasattr(bridge, "SGLangEngine")
        if retry_fails:
            raise ImportError("another manager dependency unavailable")
        return manager

    monkeypatch.setattr(_dep_stubs.importlib, "import_module", import_module)
    if retry_fails:
        with pytest.raises(ImportError, match="another manager dependency"):
            _dep_stubs.import_genrm_manager()
    else:
        assert _dep_stubs.import_genrm_manager() is manager
    assert attempts == 2
    if existing in {"module_only", "both"}:
        assert sys.modules[engine_name] is original_module
    else:
        assert engine_name not in sys.modules
    if existing in {"parent_only", "both"}:
        assert parent.sglang_engine is original_attribute
    else:
        assert not hasattr(parent, "sglang_engine")
