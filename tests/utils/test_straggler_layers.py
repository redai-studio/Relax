# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""The profiler's parts import only the parts before them: ``core`` (the
collection) the standard library and itself; ``analysis`` also numpy, requests
and ``core``; ``megatron_adapter`` Megatron and ``core``; ``collectives``
torch; ``relax_integration`` anything."""

import ast
import subprocess
import sys
from pathlib import Path

import pytest


_PACKAGE = Path(__file__).resolve().parents[2] / "relax" / "utils" / "straggler"
_CORE = "relax.utils.straggler.core"
_ANALYSIS = "relax.utils.straggler.analysis"
# requests: the level poller's HTTP client, imported when a poller is created.
_ANALYSIS_THIRD_PARTY = frozenset({"numpy", "requests"})


def _imports(path):
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, f"{path.name}: relative imports hide which layer they reach"
            yield node.module


def _disallowed(path, packages, third_party=frozenset()):
    return [
        name
        for name in _imports(path)
        if not (
            name.split(".")[0] in sys.stdlib_module_names
            or name.split(".")[0] in third_party
            or any(name == package or name.startswith(package + ".") for package in packages)
        )
    ]


def _files(part):
    return sorted((_PACKAGE / part).glob("*.py"))


@pytest.mark.parametrize("path", _files("core"), ids=lambda path: path.name)
def test_the_collection_imports_only_the_standard_library_and_itself(path):
    assert _disallowed(path, [_CORE]) == []


@pytest.mark.parametrize("path", _files("analysis"), ids=lambda path: path.name)
def test_the_analysis_imports_only_the_standard_library_numpy_requests_and_the_collection(path):
    assert _disallowed(path, [_CORE, _ANALYSIS], _ANALYSIS_THIRD_PARTY) == []


def test_the_megatron_adapter_imports_only_megatron_and_the_collection():
    assert _disallowed(_PACKAGE / "megatron_adapter.py", [_CORE], frozenset({"megatron"})) == []


def test_the_collectives_import_only_torch():
    assert _disallowed(_PACKAGE / "collectives.py", [], frozenset({"torch"})) == []


def test_importing_any_torch_free_part_loads_neither_torch_nor_megatron():
    modules = [
        *(f"{_CORE}.{path.stem}" for path in _files("core") if path.stem != "__init__"),
        *(f"{_ANALYSIS}.{path.stem}" for path in _files("analysis") if path.stem != "__init__"),
        "relax.utils.straggler.megatron_adapter",
        "relax.utils.straggler.control_routes",
    ]
    code = (
        "import importlib, sys\n"
        f"for name in {modules!r}:\n"
        "    importlib.import_module(name)\n"
        "print(sorted({'torch', 'megatron'} & {name.split('.')[0] for name in sys.modules}))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=_PACKAGE.parents[2], capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == "[]"
