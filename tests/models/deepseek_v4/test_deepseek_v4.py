# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Run in the training image; select a free CUDA device before enabling GPU
tests.

RUN_DSV4_GPU_TESTS=1 enables CUDA/TE/HDO checks. DSV4_BRIDGE_SOURCE points to
Bridge's native MXFP4 source. Optimizer hooks run in an isolated subprocess.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from typing import Any

import pytest


_GPU = pytest.mark.skipif(
    os.environ.get("RUN_DSV4_GPU_TESTS") != "1", reason="requires RUN_DSV4_GPU_TESTS=1 and a selected free GPU"
)


def _require_cuda() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("requires SM90 or newer in the training image")
    for dependency in ("triton", "megatron", "transformer_engine"):
        if importlib.util.find_spec(dependency) is None:
            pytest.skip(f"requires {dependency} from the training image")


def _bridge_reference() -> dict[str, Any]:
    import torch

    source = os.environ.get("DSV4_BRIDGE_SOURCE")
    if not source:
        pytest.skip("set DSV4_BRIDGE_SOURCE to the installed Bridge native MXFP4 source file")
    functions = {
        "is_float8_e8m0_dtype",
        "scale_from_amax",
        "dequantize_mxfp4_e2m1_packed",
        "quantize_mxfp4_e2m1_like_scale",
    }
    constants = {"FP4_E2M1_MAX", "MXFP4_BLOCK_SIZE", "_FP4_E2M1_TABLE_VALUES"}
    nodes = []
    for node in ast.parse(Path(source).read_text()).body:
        if isinstance(node, ast.FunctionDef) and node.name in functions:
            nodes.append(node)
        elif isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in constants for target in node.targets
        ):
            nodes.append(node)
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), source, "exec"), namespace)
    return namespace


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=_GPU)])
def test_qdq_matches_native_bridge_and_preserves_master(device: str) -> None:
    torch = pytest.importorskip("torch")
    if not hasattr(torch, "float8_e8m0fnu"):
        pytest.skip("requires the training image PyTorch with E8M0 support")
    if device == "cuda":
        _require_cuda()
    bridge = _bridge_reference()
    from relax.models.deepseek_v4.quantization import mxfp4_qdq

    # Exact midpoints, neighboring FP32 values, zero, tiny scales, and random blocks.
    midpoints = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])
    boundary = torch.zeros((1, 32))
    boundary[0, :7], boundary[0, 7:14] = midpoints, -midpoints
    boundary[0, 14:21] = torch.nextafter(midpoints, torch.full_like(midpoints, float("inf")))
    boundary[0, 21:28] = torch.nextafter(midpoints, torch.full_like(midpoints, -float("inf")))
    boundary[0, 28:30] = torch.tensor([6.0, -6.0])
    scales = torch.exp2(torch.tensor([-127, -126, -20, 0, 20, 120], dtype=torch.float32))
    sample = torch.cat(
        [
            boundary * scales[:, None],
            torch.zeros((1, 32)),
            torch.randn((64, 32), generator=torch.Generator().manual_seed(731)),
        ]
    )
    quantize = bridge["quantize_mxfp4_e2m1_like_scale"]
    dequantize = bridge["dequantize_mxfp4_e2m1_packed"]
    template = torch.ones((sample.shape[0], 1)).to(torch.float8_e8m0fnu)
    for dtype in (torch.bfloat16, torch.float32):
        original = sample.to(dtype)
        expected = dequantize(*quantize(original, template), dtype=dtype)
        master = original.to(device).clone().requires_grad_(True)
        effective = mxfp4_qdq(master)
        assert torch.equal(effective.cpu().view(torch.uint8), expected.view(torch.uint8))
        assert torch.equal(master.detach().cpu(), original)
        assert effective.data_ptr() != master.data_ptr() and not effective.requires_grad
        carrier = effective.bfloat16()
        exported = dequantize(*quantize(carrier.cpu(), template), dtype=torch.float32)
        assert torch.equal(exported, effective.float().cpu()), "HF export changed effective Q4 weights"

        invalid = original.clone()
        invalid[:3, 0] = torch.tensor([float("nan"), float("inf"), -float("inf")], dtype=dtype)
        result = mxfp4_qdq(invalid.to(device)).cpu()
        assert torch.equal(result[:3].view(torch.uint8), invalid[:3].view(torch.uint8))
        assert torch.equal(result[3:].view(torch.uint8), expected[3:].view(torch.uint8))


@_GPU
@pytest.mark.parametrize("compute", ["bf16", "fp8", "stock_fp8"])
def test_expert_gradients_and_full_optimizer_resume(tmp_path: Path, compute: str) -> None:
    _require_cuda()
    root = Path(__file__).resolve().parents[3]
    output = tmp_path / compute
    environment = {
        **os.environ,
        "NO_VCS_VERSION": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NVTE_FP8_BLOCK_SCALING_FP32_SCALES": "1",
    }
    environment["PYTHONPATH"] = os.pathsep.join(
        value for value in (str(root), environment.get("PYTHONPATH", "")) if value
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "tests.models.deepseek_v4._optimizer_resume_probe",
            "--output-dir",
            str(output),
            "--compute",
            compute,
        ],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
    report = json.loads((output / "report.json").read_text())
    assert report["status"] == "PASS" and report["expert_compute"] == compute


@pytest.mark.parametrize("mode", ["native", "stock_fp8"])
def test_provider_mode_selects_compute_and_optional_quantizers(monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
    globals_module = pytest.importorskip("megatron.training.global_vars")
    from relax.backends.megatron import model_provider as base_provider
    from relax.models.deepseek_v4 import (
        fp8_weights,
        indexer,
        master_weights,
        provider,
        qat,
    )

    if mode == "native":
        monkeypatch.delenv("RELAX_DSV4_FP4_MODE", raising=False)
    else:
        monkeypatch.setenv("RELAX_DSV4_FP4_MODE", mode)
    monkeypatch.setenv("RELAX_DSV4_FP4_EXPERT_COMPUTE", "bf16")
    monkeypatch.setenv("RELAX_DSV4_FP4_INDEXER", "1")
    monkeypatch.setenv("NVTE_FP8_BLOCK_SCALING_FP32_SCALES", "1")
    model = SimpleNamespace(
        config=SimpleNamespace(experimental_attention_variant="dsv4_hybrid", tensor_model_parallel_size=1),
        _relax_native_fp8_uncovered=(),
    )
    calls: dict[str, Any] = {}
    monkeypatch.setattr(
        globals_module,
        "get_args",
        lambda: SimpleNamespace(megatron_to_hf_mode="bridge", bf16=True, hf_checkpoint="unused"),
    )
    monkeypatch.setattr(base_provider, "get_model_provider_func", lambda *a, **kw: lambda **kwargs: model)
    monkeypatch.setattr(master_weights, "install_master_weight_hooks", lambda: None)
    monkeypatch.setattr(qat, "install_mxfp4_qat", lambda model, **kw: calls.setdefault("routed", kw) or ())
    monkeypatch.setattr(fp8_weights, "install_native_fp8_weights", lambda *a, **kw: calls.setdefault("native", ("w",)))
    monkeypatch.setattr(indexer, "select_process_mode", lambda **kw: calls.setdefault("indexer_mode", kw))
    monkeypatch.setattr(indexer, "install_indexer_qat", lambda *a, **kw: calls.setdefault("indexer", ("q",)))
    assert provider.model_provider() is model
    assert calls["routed"]["compute"] == ("bf16" if mode == "native" else "stock_fp8")
    assert ("native" in calls) == (mode == "native")
    assert ("indexer" in calls) == (mode == "native")
    assert calls["indexer_mode"]["enabled"] == (mode == "native")


@pytest.mark.parametrize("mode,scales", [("invalid", "1"), ("stock_fp8", "0"), ("stock_fp8", None)])
def test_mode_rejects_invalid_name_or_incompatible_fp8_scales(
    monkeypatch: pytest.MonkeyPatch, mode: str, scales: str | None
) -> None:
    pytest.importorskip("megatron")
    from relax.models.deepseek_v4.qat import _fp4_mode

    monkeypatch.setenv("RELAX_DSV4_FP4_MODE", mode)
    if scales is None:
        monkeypatch.delenv("NVTE_FP8_BLOCK_SCALING_FP32_SCALES", raising=False)
    else:
        monkeypatch.setenv("NVTE_FP8_BLOCK_SCALING_FP32_SCALES", scales)
    with pytest.raises(ValueError):
        _fp4_mode()


@pytest.mark.parametrize("bf16_scores", [False, True])
def test_indexer_qat_only_affects_selected_instances(monkeypatch: pytest.MonkeyPatch, bf16_scores: bool) -> None:
    torch = pytest.importorskip("torch")
    csa = pytest.importorskip("megatron.core.transformer.experimental_attention_variant.csa")
    from megatron.core.transformer.experimental_attention_variant import dsa_kernels

    from relax.models.deepseek_v4 import indexer

    monkeypatch.setattr(indexer, "_PROCESS_MODE", None)
    monkeypatch.setattr(indexer, "activation_qdq", lambda value: value + 10)
    monkeypatch.setattr(indexer, "round_scores_", lambda scores: scores + 1)
    monkeypatch.setattr(csa, "rotate_activation", lambda value: value)
    monkeypatch.setattr(dsa_kernels, "_ensure_dsa_namespace", lambda: None)
    monkeypatch.setattr(
        dsa_kernels,
        "_DSA",
        SimpleNamespace(indexer_forward_wrapper=lambda q, k, w, **kwargs: {"scores": torch.tensor(float(q))}),
    )

    class CSA:
        __module__ = csa.__name__

        def __init__(self) -> None:
            self.indexer = SimpleNamespace(compressor=SimpleNamespace(rotate=True))
            self.compressor = None
            self.apply_dsa_kernel_fusion = True
            self.config = SimpleNamespace(dsa_indexer_loss_coeff=0.0)

        def forward(self, *, nested=None, fail=False, entered=None, release=None):
            if entered is not None:
                entered.set()
                assert release.wait(10), "concurrent scope check timed out"
            child = nested.forward() if nested is not None else None
            # Exercise the CP-style direct calls, without invoking indexer.forward.
            q = csa.rotate_activation(1)
            scores = dsa_kernels._DSA.indexer_forward_wrapper(q, None, None)["scores"]
            if fail:
                raise RuntimeError("forward failed")
            return q, scores.item(), child

    selected, other = CSA(), CSA()
    model = SimpleNamespace(named_modules=lambda: iter((("selected", selected),)))
    assert indexer.install_indexer_qat(model, bf16_scores=bf16_scores) == ("selected",)
    assert indexer.install_indexer_qat(model, bf16_scores=bf16_scores) == ("selected",)
    expected = (11, 11 + int(bf16_scores), None)
    assert selected.forward() == expected
    assert selected.forward() == expected  # Full-forward recomputation re-enters the scope.
    assert other.forward() == (1, 1, None)
    assert selected.forward(nested=other) == (*expected[:2], (1, 1, None))
    with pytest.raises(RuntimeError, match="forward failed"):
        selected.forward(fail=True)
    assert indexer._ACTIVE_INDEXER.get() is None
    # Unselected calls must also bypass QAT-only dtype/LSE/precision constraints.
    sentinel = object()
    assert csa.rotate_activation(sentinel) is sentinel
    assert dsa_kernels._DSA.indexer_forward_wrapper(1, None, None, precision="fp32", return_lse=True)["scores"] == 1

    entered, release = Event(), Event()
    with ThreadPoolExecutor(max_workers=1) as pool:
        running = pool.submit(selected.forward, entered=entered, release=release)
        try:
            assert entered.wait(10)
            assert other.forward() == (1, 1, None)
            assert csa.rotate_activation(1) == 1
        finally:
            release.set()
        assert running.result(timeout=10) == expected
    assert indexer._ACTIVE_INDEXER.get() is None


def test_default_bridge_and_qat_imports_do_not_install_hooks(tmp_path: Path) -> None:
    for dependency in ("torch", "megatron", "transformer_engine"):
        if importlib.util.find_spec(dependency) is None:
            pytest.skip(f"requires {dependency} from the training image")
    root = Path(__file__).resolve().parents[3]
    source = """
import importlib, pkgutil, sys
from types import SimpleNamespace
from unittest.mock import patch
import torch
from megatron.bridge import AutoBridge
from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer as DO
from megatron.core.extensions.transformer_engine import TEGroupedLinear
from megatron.core.transformer.experimental_attention_variant import csa, dsa_kernels
from transformer_engine.pytorch.module.base import TransformerEngineBaseModule as TEBase
prefix = 'relax.models.deepseek_v4'
def bindings():
    methods = ('__init__', 'load_state_dict', 'step_with_ready_grads', '_copy_model_params_to_main_params')
    return tuple(getattr(DO, name) for name in methods) + (
        TEBase.get_weight_workspace, TEGroupedLinear.get_weight_workspace,
        csa.CompressedSparseAttention.forward, csa.rotate_activation,
        dsa_kernels._indexer_topk_core, getattr(dsa_kernels, '_DSA', None))
before = bindings()
assert not any(name.startswith(prefix) for name in sys.modules)
from relax.backends.megatron import model_provider
for precision in (None, 'e4m3'):
    provider = SimpleNamespace(fp8=None, fp8_recipe='delayed', provide=lambda **kw: None)
    provider.finalize = lambda: setattr(provider, 'finalized', True)
    bridge = SimpleNamespace(to_megatron_provider=lambda **kw: provider)
    args = SimpleNamespace(custom_model_provider_path=None, megatron_to_hf_mode='bridge',
        hf_checkpoint='unused', fp16=False, bf16=True, fp8=precision,
        fp8_recipe='blockwise' if precision else 'delayed')
    with patch.object(AutoBridge, 'from_hf_pretrained', return_value=bridge), \
         patch.object(model_provider, '_dump_provider_config'):
        selected = model_provider.get_model_provider_func(args)
    assert callable(selected) and provider.finalized
    assert provider.bf16 and provider.params_dtype is torch.bfloat16
    assert provider.fp8 == precision and provider.fp8_recipe == args.fp8_recipe
    assert not any(name.startswith(prefix) for name in sys.modules)
    assert all(left is right for left, right in zip(before, bindings(), strict=True))
package = importlib.import_module(prefix)
for module in pkgutil.iter_modules(package.__path__, package.__name__ + '.'):
    importlib.import_module(module.name)
assert all(left is right for left, right in zip(before, bindings(), strict=True))
assert sys.modules[package.__name__ + '.indexer']._PROCESS_MODE is None
assert not torch.cuda.is_initialized()
"""
    environment = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "",
        "NO_VCS_VERSION": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NVTE_FP8_BLOCK_SCALING_FP32_SCALES": "1",
        "RELAX_DSV4_FP4_MODE": "stock_fp8",
        "RELAX_DSV4_FP4_EXPERT_COMPUTE": "fp8",
        "RELAX_DSV4_FP4_INDEXER": "1",
        "RELAX_DSV4_FP4_BF16_SCORES": "1",
        "RELAX_DSV4_FP4_STATS_INTERVAL": "10",
        "RELAX_DSV4_FP4_STATS_DIR": str(tmp_path),
    }
    environment["PYTHONPATH"] = os.pathsep.join(
        value for value in (str(root), environment.get("PYTHONPATH", "")) if value
    )
    result = subprocess.run(
        [sys.executable, "-c", source],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
