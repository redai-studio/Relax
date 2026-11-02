# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""CPU coverage of scratch bounds and launch wiring, without CUDA imports."""

import ast
import math
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[3]
BACKEND = ROOT / "relax/models/qwen3_8/ops/qsa_triton_v3"


def definitions(path, names, scope):
    tree = ast.parse(path.read_text())
    body = [
        n
        for n in tree.body
        if getattr(n, "name", None) in names
        or isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id in names for t in n.targets)
    ]
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), scope)
    return scope


@pytest.fixture
def workspace():
    scope = dict(
        dataclass=dataclass,
        torch=SimpleNamespace(Tensor=object),
        triton=SimpleNamespace(cdiv=lambda a, b: (a + b - 1) // b),
    )
    return definitions(
        BACKEND / "shared_backward.py", {"DEFAULT_CAPACITY_BYTES", "Workspace", "workspace_shape"}, scope
    )


@pytest.mark.parametrize("length,heads,budget", [(1, 3, 0), (257, 3, 1024 * 1024), (131072, 3, 24 * 2**30)])
def test_workspace_stays_within_budget(workspace, length, heads, budget):
    rows, cols, cap = workspace["workspace_shape"](length, length, heads, budget)
    assert rows == cols == math.ceil(length / 128)
    assert cap * heads * 128 * 128 * 4 <= budget
    assert cap <= rows * cols


@pytest.mark.parametrize(
    "q,k,h,budget,error",
    [
        (0, 128, 3, 0, ValueError),
        (128, 128, 0, 0, ValueError),
        (131073, 128, 3, 0, ValueError),
        (131072, 131072, 64, 0, ValueError),
        (128, 128, 3, -1, ValueError),
        (128, 128, 3, 25 * 2**30, ValueError),
        (128, 128, 3, 1.5, TypeError),
    ],
)
def test_workspace_rejects_unsupported_bounds(workspace, q, k, h, budget, error):
    with pytest.raises(error):
        workspace["workspace_shape"](q, k, h, budget)


def test_shared_dispatch_is_limited_to_validated_contract():
    bf16 = object()
    torch = SimpleNamespace(
        Tensor=object, dtype=object, bfloat16=bf16, cuda=SimpleNamespace(get_device_capability=lambda device: (10, 3))
    )
    scope = dict(dataclass=dataclass, torch=torch, base=SimpleNamespace(QSAFusedPlan=object))
    definitions(BACKEND.parent / "qsa_triton_v2/__init__.py", {"Config", "DEFAULT"}, scope)
    scope["v2"] = SimpleNamespace(Config=scope["Config"], DEFAULT=scope["DEFAULT"])
    definitions(BACKEND / "__init__.py", {"Config", "DEFAULT", "_SM103_BF16_D256", "_can_share_ds"}, scope)
    q = SimpleNamespace(shape=(257, 3, 256), dtype=bf16, device="cuda:0", is_cuda=True)
    k = SimpleNamespace(shape=(257, 1, 256), dtype=bf16, device="cuda:0")
    plan = SimpleNamespace(ratio=4)
    check = scope["_can_share_ds"]
    config = scope["_SM103_BF16_D256"]
    assert check(q, k, k, plan, None, config)
    assert not check(q, k, k, plan, object(), config)
    assert not check(q, k, k, plan, None, scope["DEFAULT"])
    for target, field, value in [
        (q, "shape", (131073, 3, 256)),
        (q, "dtype", object()),
        (q, "is_cuda", False),
        (k, "device", "cuda:1"),
        (plan, "ratio", 8),
    ]:
        original = getattr(target, field)
        setattr(target, field, value)
        assert not check(q, k, k, plan, None, config)
        setattr(target, field, original)
    torch.cuda.get_device_capability = lambda device: (9, 0)
    assert not check(q, k, k, plan, None, config)


@pytest.mark.parametrize(
    "raw,expected", [(None, False), ("0", False), ("1", True), ("triton_v2", "triton_v2"), ("triton_v3", "triton_v3")]
)
def test_backend_environment_and_dispatch_preserve_existing_defaults(monkeypatch, raw, expected):
    import importlib.util
    import sys
    from types import ModuleType

    spec = importlib.util.spec_from_file_location("v3_test_env", ROOT / "relax/utils/env.py")
    env = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(env)
    name = "RELAX_QWEN38_QSA_FUSED_ATTENTION"
    if raw is None:
        monkeypatch.delenv(name, raising=False)
    else:
        monkeypatch.setenv(name, raw)
    assert env.Envs.RELAX_QWEN38_QSA_FUSED_ATTENTION == expected
    package = ModuleType("relax.models.qwen3_8.ops")
    package.__path__ = []
    monkeypatch.setitem(sys.modules, package.__name__, package)
    versions = {}
    for version in ("triton_v2", "triton_v3"):
        module = ModuleType(package.__name__ + ".qsa_" + version)
        monkeypatch.setitem(sys.modules, module.__name__, module)
        setattr(package, "qsa_" + version, module)
        versions[version] = module
    base = ModuleType("base")
    scope = dict(Envs=env.Envs, qsa_fused_attention=base)
    path = BACKEND.parent.parent / "qsa_fused.py"
    definitions(path, {"backend"}, scope)
    assert scope["backend"]() is (versions.get(expected, base) if expected else None)
