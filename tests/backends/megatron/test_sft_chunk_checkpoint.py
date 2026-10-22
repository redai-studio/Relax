# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""CPU regression tests for SFT per-chunk checkpointing, without a Ray
runtime."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from relax.utils.env import Envs


@pytest.fixture
def loss_fn():
    # Execute the production function with lightweight response/TP adapters.
    source = Path("relax/backends/megatron/loss.py").read_text()
    node = next(
        n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "get_log_probs_and_entropy"
    )
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node],
        type_ignores=[],
    )

    def log_probs(logits, targets, group, with_entropy=False):
        return logits.log_softmax(-1).gather(-1, targets[:, None]), None

    namespace = {
        "torch": torch,
        "Envs": Envs,
        "ce_implementation": lambda branch: "native",
        "checkpoint": checkpoint,
        "mpu": SimpleNamespace(get_tensor_model_parallel_group=lambda: None),
        "validate_opd_topk_gather": lambda *a: None,
        "calculate_log_probs_and_entropy": log_probs,
        "get_responses": lambda hidden, **kw: iter(zip(hidden.split([7, 10]), kw["unconcat_tokens"])),
    }
    exec(compile(ast.fix_missing_locations(module), "relax/backends/megatron/loss.py", "exec"), namespace)
    return namespace["get_log_probs_and_entropy"]


@pytest.mark.parametrize("chunk", [1, 4, 32])
@pytest.mark.parametrize("frozen", [False, True])
@pytest.mark.parametrize("temperature", [1.0, 0.7])
def test_chunk_checkpoint_loss_and_gradients(loss_fn, chunk, frozen, temperature):
    torch.manual_seed(42)
    hidden = torch.randn(17, 8)
    weight = torch.randn(31, 8)
    targets = torch.randint(0, 31, (17,))
    mask = torch.rand(17)
    outputs = []
    for recompute in (False, True):
        h = hidden.clone().requires_grad_(not frozen)
        w = weight.clone().requires_grad_()
        args = SimpleNamespace(
            sft_logits_chunk_size=chunk,
            recompute_loss_function=recompute,
            rollout_temperature=temperature,
            allgather_cp=False,
        )
        _, result = loss_fn(
            h,
            args=args,
            unconcat_tokens=list(targets.split([7, 10])),
            total_lengths=[7, 10],
            response_lengths=[7, 10],
            with_entropy=False,
            non_loss_data=True,
            lm_head_forward=lambda x: (x @ w.T, None),
        )
        loss = -(torch.cat(result["log_probs"]) * mask).sum() / mask.sum()
        loss.backward()
        outputs.append((loss.detach(), h.grad, w.grad))
    for actual, expected in zip(outputs[1], outputs[0]):
        if expected is None:
            assert actual is None
        else:
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_checkpoint_does_not_retain_vocab_tensors(loss_fn):
    h = torch.randn(17, 8, requires_grad=True)
    w = torch.randn(31, 8, requires_grad=True)
    targets = [torch.zeros(7, dtype=torch.long), torch.zeros(10, dtype=torch.long)]
    saved = []

    def pack(x):
        saved.append(tuple(x.shape))
        return x

    args = SimpleNamespace(
        sft_logits_chunk_size=4, recompute_loss_function=True, rollout_temperature=1.0, allgather_cp=False
    )
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda x: x):
        _, result = loss_fn(
            h,
            args=args,
            unconcat_tokens=targets,
            total_lengths=[7, 10],
            response_lengths=[7, 10],
            with_entropy=False,
            non_loss_data=True,
            lm_head_forward=lambda x: (x @ w.T, None),
        )
    assert saved
    assert all(31 not in shape for shape in saved), saved
    torch.cat(result["log_probs"]).sum().backward()
    assert w.grad is not None


@pytest.mark.parametrize("mode", ["sft", "cpt"])
@pytest.mark.parametrize("fail", [False, True])
def test_checkpointed_head_disables_fusion_and_restores_state(mode, fail):
    from contextlib import contextmanager

    tree = ast.parse(Path("relax/backends/megatron/model.py").read_text())
    helper = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_bypass_output_layer")
    call = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "_bypass_output_layer"
    )
    guard = next(k.value for k in call.keywords if k.arg == "unfused_head_wgrad")
    enabled = eval(
        compile(ast.Expression(guard), "guard", "eval"),
        {
            "args": SimpleNamespace(sft_training_mode=mode, recompute_loss_function=True),
            "mtp_enabled": False,
        },
    )
    assert enabled

    class Head(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(3, 2))
            self.sequence_parallel = False
            self.gradient_accumulation_fusion = True

        def forward(self, value, **kwargs):
            assert not self.gradient_accumulation_fusion
            if fail:
                raise ValueError("head failure")
            return value @ self.weight.T, None

    head = Head()
    ns = {
        "contextmanager": contextmanager,
        "torch": torch,
        "Envs": Envs,
        "_find_lm_output_layer": lambda model: model,
        "mpu": SimpleNamespace(get_tensor_model_parallel_group=lambda: None),
        "logger": SimpleNamespace(info=lambda *args: None),
    }
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), helper],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), "head", "exec"), ns)
    value = torch.randn(2, 2, requires_grad=True)
    with ns["_bypass_output_layer"](head, unfused_head_wgrad=enabled) as forward:
        head(value)
        if fail:
            with pytest.raises(ValueError, match="head failure"):
                forward(value)
        else:
            result, _ = forward(value)
            result.sum().backward()
            assert head.weight.grad is not None
        assert head.gradient_accumulation_fusion
    assert "forward" not in head.__dict__
