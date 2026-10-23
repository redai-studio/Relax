# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Replay the training wheel packaging against pinned HybridEP sources on
CPU."""

import ast
import os
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def hybrid_source(tmp_path):
    source = os.environ.get("BLACKWELL_HYBRIDEP_SOURCE_DIR")
    if not source:
        pytest.skip("Set BLACKWELL_HYBRIDEP_SOURCE_DIR to pinned unpatched HybridEP sources")
    target = tmp_path / "source"
    shutil.copytree(source, target)
    result = subprocess.run(
        [
            "patch",
            "--batch",
            "--forward",
            "--fuzz=0",
            "-p1",
            "-i",
            str(ROOT / "docker/patch/deepep/eb9cee7-hybridep-namespace.patch"),
        ],
        cwd=target,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "offset" not in result.stdout and "fuzz" not in result.stdout
    return target


def test_private_python_payload_retains_backend_files(hybrid_source, tmp_path, monkeypatch):
    from setuptools import Distribution
    from setuptools.command.build_py import build_py

    monkeypatch.chdir(hybrid_source)
    tree = ast.parse((hybrid_source / "setup.py").read_text())
    collect = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "collect_package_files")
    scope = {
        "Path": Path,
        "revision": "+eb9cee7",
        "os": os,
        "shutil": shutil,
        "__file__": str(hybrid_source / "setup.py"),
        "CUDAExtension": lambda *a, **kw: (a, kw),
    }
    monkeypatch.setenv("HYBRID_EP_MULTINODE", "0")
    extension = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "get_extension_hybrid_ep_cpp"
    )
    exec(compile(ast.Module(body=[extension], type_ignores=[]), "setup.py", "exec"), scope)
    positional, _ = scope["get_extension_hybrid_ep_cpp"]()
    assert positional == ("_relax_hybrid_ep.hybrid_ep_cpp",)
    exec(compile(ast.Module(body=[collect], type_ignores=[]), "setup.py", "exec"), scope)
    setup = next(n for n in ast.walk(tree) if isinstance(n, ast.Call) and ast.unparse(n.func) == "setuptools.setup")
    # Exercise the actual package mapping/data through setuptools; only CUDA compilation is excluded.
    kwargs = {
        kw.arg: eval(compile(ast.Expression(kw.value), "setup.py", "eval"), scope)
        for kw in setup.keywords
        if kw.arg not in {"ext_modules", "cmdclass"}
    }
    distribution = Distribution(kwargs)
    distribution.script_name = "setup.py"
    command = build_py(distribution)
    command.ensure_finalized()
    command.build_lib = str(tmp_path / "installed")
    command.run()
    installed = Path(command.build_lib)
    assert kwargs["name"] == "relax_hybrid_ep"
    assert not (installed / "deep_ep").exists()
    assert (installed / "_relax_hybrid_ep/hybrid_ep_buffer.py").is_file()
    backend = sorted((hybrid_source / "deep_ep/backend").rglob("*"))
    assert any(p.is_file() for p in backend)
    for path in backend:
        if path.is_file():
            copy = installed / "_relax_hybrid_ep" / path.relative_to(hybrid_source / "deep_ep")
            assert copy.read_bytes() == path.read_bytes()
    for path in (installed / "_relax_hybrid_ep").glob("*.py"):
        compile(path.read_text(), str(path), "exec")
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                assert all(alias.name not in {"deep_ep_cpp", "hybrid_ep_cpp", "deep_ep"} for alias in node.names)
            if isinstance(node, ast.ImportFrom) and node.module in {"deep_ep_cpp", "hybrid_ep_cpp"}:
                assert node.level == 1
