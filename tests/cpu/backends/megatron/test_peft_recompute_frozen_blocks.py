# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""CPU checks of the repository patch against Qwen3.8 Bridge/MCore sources.

Set MEGATRON_BRIDGE and MEGATRON to the source checkout roots when they are not
on sys.path. Only selected definitions are executed; no Megatron/CUDA imports
or distributed collectives are needed.
"""

import ast
import os
import shutil
import subprocess
import sys
from collections.abc import Iterable, MutableSet, Sequence
from functools import wraps
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
from weakref import WeakSet

import pytest


torch = pytest.importorskip("torch")
checkpoint = pytest.importorskip("torch.utils.checkpoint").checkpoint

pytestmark = pytest.mark.filterwarnings("ignore:None of the inputs have requires_grad=True:UserWarning")


def _dependency_source(variable: str, relative: str) -> Path:
    configured = os.environ.get(variable)
    roots = [Path(configured)] if configured else []
    roots.extend(Path(entry) for entry in sys.path if entry)
    for root in roots:
        for candidate in (root / relative, root / "src" / relative):
            if candidate.is_file():
                return candidate
    pytest.skip(f"Requires Qwen3.8 dependency sources; set {variable} to its checkout root")


def _definitions(source: Path, names: set[str], namespace: dict[str, Any]) -> dict[str, Any]:
    tree = ast.parse(source.read_text())
    definitions = [
        node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names
    ]
    assert {node.name for node in definitions} == names
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(source), "exec"), namespace)
    return namespace


@pytest.fixture(scope="module", params=["20260915-4a4de8657.patch"])
def dependency_functions(request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory) -> SimpleNamespace:
    if shutil.which("git") is None:
        pytest.skip("git is required to apply the repository Bridge patch")
    bridge = _dependency_source("MEGATRON_BRIDGE", "megatron/bridge/peft/recompute.py")
    core = _dependency_source("MEGATRON", "megatron/core/utils.py")
    cp = bridge.parents[1] / "models/qwen_vl/modelling_qwen3_vl/utils.py"
    patch = Path(__file__).resolve().parents[4] / "docker/patch/megatron" / request.param
    assert "+++ b/megatron/bridge/peft/recompute.py" in patch.read_text()
    checkout = tmp_path_factory.mktemp("bridge-frozen-recompute")
    target = checkout / "megatron/bridge/peft/recompute.py"
    target.parent.mkdir(parents=True)
    target.write_text(bridge.read_text())

    def apply_patch(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", "apply", "--include=megatron/bridge/peft/recompute.py", *args, str(patch)],
            cwd=checkout,
            capture_output=True,
            text=True,
            check=False,
        )

    # The fixed image may already include the patch; only reverse our temporary copy.
    if apply_patch("--check").returncode:
        result = apply_patch("--reverse", "--check")
        assert result.returncode == 0, result.stderr
        result = apply_patch("--reverse")
        assert result.returncode == 0, result.stderr
    result = apply_patch()
    assert result.returncode == 0, result.stderr
    viewless = _definitions(
        core, {"_kernel_make_viewless_tensor", "MakeViewlessTensor", "make_viewless_tensor"}, {"torch": torch}
    )["make_viewless_tensor"]
    ensure_cp_grad = _definitions(
        cp, {"ensure_requires_grad_for_cp_collective"}, {"torch": torch, "Sequence": Sequence}
    )["ensure_requires_grad_for_cp_collective"]
    return SimpleNamespace(source=target, viewless=viewless, ensure_cp_grad=ensure_cp_grad)


@pytest.fixture
def recompute_case(dependency_functions: SimpleNamespace, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    class CheckpointBlock(torch.nn.Module):
        def __init__(self, adapter: bool = False, recompute: bool = True):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.eye(3), requires_grad=False)
            self.layer = torch.nn.Module()
            self.recompute = recompute
            self.seen_requires_grad = None
            if adapter:
                self.add_adapter()

        def add_adapter(self) -> None:
            self.layer.adapter = torch.nn.Sequential(
                torch.nn.Linear(3, 2, bias=False), torch.nn.Linear(2, 3, bias=False)
            )
            for parameter in self.layer.adapter.parameters():
                torch.nn.init.constant_(parameter, 0.2)

        def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            self.seen_requires_grad = hidden_states.requires_grad
            hidden_states = dependency_functions.viewless(hidden_states, requires_grad=True, keep_graph=True)

            def layer(value: torch.Tensor) -> torch.Tensor:
                result = value @ self.weight.T
                if hasattr(self.layer, "adapter"):
                    result = result + self.layer.adapter(value)
                return result

            if self.recompute:
                return checkpoint(layer, hidden_states, use_reentrant=True)
            return layer(hidden_states)

    # Keep the two recognized types independent, so omission of HybridStack is observable.
    class TransformerBlock(CheckpointBlock):
        pass

    class HybridStack(CheckpointBlock):
        pass

    for name, attribute, cls in (
        ("megatron.core.transformer.transformer_block", "TransformerBlock", TransformerBlock),
        ("megatron.core.models.hybrid.hybrid_block", "HybridStack", HybridStack),
    ):
        module = ModuleType(name)
        setattr(module, attribute, cls)
        monkeypatch.setitem(sys.modules, name, module)
    namespace = _definitions(
        dependency_functions.source,
        {"_iter_unwrapped_models", "maybe_enable_recompute_inputs_grad"},
        {
            "torch": torch,
            "Iterable": Iterable,
            "MutableSet": MutableSet,
            "wraps": wraps,
            "unwrap_model": lambda model: model,
            "print_rank_0": lambda *args: None,
            "PEFT_RECOMPUTE_PATCHED": WeakSet(),
        },
    )

    def root(block_type: type = TransformerBlock, vision_adapter: bool = False) -> torch.nn.Module:
        model = torch.nn.Module()
        model.config = SimpleNamespace(recompute_method="uniform")
        model.vision = block_type(adapter=vision_adapter)
        model.language = block_type(adapter=True)
        return model

    return SimpleNamespace(
        root=root,
        block=TransformerBlock,
        hybrid=HybridStack,
        install=namespace["maybe_enable_recompute_inputs_grad"],
        **vars(dependency_functions),
    )


@pytest.mark.parametrize("kind", ["block", "hybrid"])
@pytest.mark.parametrize("vision_recompute", [False, True])
def test_frozen_vision_has_no_graph_and_lora_gradients_match_reference(
    recompute_case: SimpleNamespace, kind: str, vision_recompute: bool
):
    case = recompute_case
    model = case.root(getattr(case, kind))
    reference = case.root(getattr(case, kind))
    reference.load_state_dict(model.state_dict())
    reference.language.recompute = False
    reference.vision.recompute = False
    model.vision.recompute = vision_recompute
    case.install(model, set())
    inputs = torch.arange(12, dtype=torch.float32).reshape(4, 3) / 10
    vision_output = model.vision(inputs[:, None])
    assert model.vision.seen_requires_grad is False
    assert not vision_output.requires_grad
    assert vision_output.grad_fn is None
    output = model.language(vision_output)
    expected = reference.language(reference.vision(inputs[:, None]))
    output.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(output, expected)
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None
            assert torch.count_nonzero(parameter.grad) > 0
            torch.testing.assert_close(parameter.grad, reference.get_parameter(name).grad)


@pytest.mark.parametrize("kind", ["block", "hybrid"])
def test_frozen_block_preserves_gradient_to_trainable_upstream(recompute_case: SimpleNamespace, kind: str):
    case = recompute_case
    model = case.root(getattr(case, kind))
    model.vision.recompute = True
    case.install(model, set())
    upstream = torch.nn.Linear(3, 3, bias=False)
    inputs = torch.ones(4, 3)
    model.vision(upstream(inputs)[:, None]).sum().backward()
    assert model.vision.seen_requires_grad is True
    torch.testing.assert_close(upstream.weight.grad, torch.full_like(upstream.weight, 4))
    assert model.vision.weight.grad is None


@pytest.mark.parametrize("timing", ["before", "inject_after", "unfreeze_after", "freeze_after"])
def test_checkpointed_vision_adapter_respects_trainability_changes(recompute_case: SimpleNamespace, timing: str):
    case = recompute_case
    model = case.root(vision_adapter=timing != "inject_after")
    model.vision.recompute = True
    if timing == "unfreeze_after":
        model.vision.layer.adapter.requires_grad_(False)
    registry = case.install(model, set())
    original_wrapper = model.vision.forward
    if timing == "inject_after":
        model.vision.add_adapter()
    elif timing == "unfreeze_after":
        model.vision.layer.adapter.requires_grad_(True)
    elif timing == "freeze_after":
        model.vision.layer.adapter.requires_grad_(False)
    case.install(model, registry)
    assert model.vision.forward is original_wrapper
    output = model.vision(torch.ones(4, 1, 3))
    if timing == "freeze_after":
        assert not output.requires_grad and output.grad_fn is None
        return
    output.sum().backward()
    for parameter in model.vision.layer.adapter.parameters():
        assert parameter.grad is not None
        assert torch.count_nonzero(parameter.grad) > 0


def test_viewless_tensor_preserves_frozen_and_upstream_grad_states(dependency_functions: SimpleNamespace):
    for requires_grad in (False, True):
        inputs = torch.ones(4, 3, requires_grad=requires_grad)
        view = inputs[:, None]
        assert view._base is not None
        output = dependency_functions.viewless(view, requires_grad=True, keep_graph=True)
        assert output._base is None
        assert output.requires_grad is requires_grad
        if requires_grad:
            output.sum().backward()
            torch.testing.assert_close(inputs.grad, torch.ones_like(inputs))
        else:
            assert output.grad_fn is None


@pytest.mark.parametrize("empty", [False, True])
def test_cp_output_guard_enables_local_backward_without_vision_graph(recompute_case: SimpleNamespace, empty: bool):
    case = recompute_case
    model = case.root()
    case.install(model, set())
    output = torch.empty(0, 1, 3) if empty else model.vision(torch.ones(4, 1, 3))
    assert output.grad_fn is None and not output.requires_grad
    case.ensure_cp_grad((output,))
    assert output.requires_grad and output.is_leaf
    backward_calls = []

    class CollectiveProbe(torch.autograd.Function):
        @staticmethod
        def forward(ctx: Any, value: torch.Tensor) -> torch.Tensor:
            return value.clone()

        @staticmethod
        def backward(ctx: Any, gradient: torch.Tensor) -> torch.Tensor:
            backward_calls.append(gradient.shape)
            return gradient

    # Local autograd reachability only; this does not simulate a distributed all-reduce.
    CollectiveProbe.apply(output).sum().backward()
    assert backward_calls == [output.shape]
    assert model.vision.weight.grad is None
    with torch.no_grad():
        untouched = torch.ones(1)
        case.ensure_cp_grad((untouched,))
        assert not untouched.requires_grad
