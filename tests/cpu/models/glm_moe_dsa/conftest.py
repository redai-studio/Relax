# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import importlib
import sys
from pathlib import Path
from types import ModuleType

import pytest


@pytest.fixture
def glm_modules(monkeypatch):
    """Import actual GLM modules without triggering unrelated Bridge
    registrations."""
    root = Path(__file__).resolve().parents[4] / "relax/models"
    for name, path in (("relax.models", root), ("relax.models.glm_moe_dsa", root / "glm_moe_dsa")):
        package = ModuleType(name)
        package.__path__ = [str(path)]
        monkeypatch.setitem(sys.modules, name, package)

    loaded = []

    def load(suffix):
        name = "relax.models.glm_moe_dsa." + suffix
        monkeypatch.delitem(sys.modules, name, raising=False)
        module = importlib.import_module(name)
        loaded.append(name)
        return module

    yield load
    for name in loaded:
        sys.modules.pop(name, None)
