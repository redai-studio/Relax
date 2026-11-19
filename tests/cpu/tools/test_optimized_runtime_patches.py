# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Validate installed runtime behavior and image build helper decisions on
CPU."""

import ast
import subprocess
from contextlib import nullcontext
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize("marker", ["", '; platform_system == "Linux" and platform_machine == "x86_64"'])
def test_runtime_check_accepts_only_documented_nccl_override(marker):
    validator = _load_dependency_validator()
    line = (
        f"torch 2.11.0+cu130 has requirement nvidia-nccl-cu13==2.29.7{marker}, but you have nvidia-nccl-cu13 2.30.7.\n"
    )
    validator(subprocess.CompletedProcess([], 1, line, ""))
    validator(subprocess.CompletedProcess([], 0, "No broken requirements found.\n", ""))


@pytest.mark.parametrize(
    "output",
    [
        "deep-ep 2.1.0 requires pynvml, which is not installed.",
        "torch 2.11.0 has requirement nvidia-nccl-cu13==2.29.7, but you have nvidia-nccl-cu13 2.30.6.",
        "other 1.0 has requirement nvidia-nccl-cu13==2.29.7, but you have nvidia-nccl-cu13 2.30.7.",
        "torch 2.11.0 has requirement numpy==1.0, but you have numpy 2.0.",
        "",
    ],
)
def test_runtime_check_rejects_missing_or_unrelated_conflicts(output):
    with pytest.raises(RuntimeError):
        _load_dependency_validator()(subprocess.CompletedProcess([], 1, output, ""))


def _load_dependency_validator():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "delivery_manifest", ROOT / "docker/scripts/record_qwen35_runtime.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.validate_pip_check


@pytest.mark.parametrize("installed", [None, "0.1.10", "0.1.11"])
def test_copied_jit_source_dependencies_are_checked(tmp_path, monkeypatch, installed):
    scope = _load_dependency_validator().__globals__
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text('[project]\ndependencies = ["apache-tvm-ffi>=0.1.11,<0.2"]\n')

    def version(name):
        if installed is None:
            raise scope["md"].PackageNotFoundError(name)
        return installed

    monkeypatch.setattr(scope["md"], "version", version)
    check = scope["validate_source_dependencies"]
    if installed == "0.1.11":
        check(pyproject)
    else:
        with pytest.raises(RuntimeError, match="apache-tvm-ffi"):
            check(pyproject)


def test_deepep_buffer_initializes_once_on_first_use(monkeypatch):
    import sys
    from types import SimpleNamespace

    from megatron.core.transformer.moe import fused_a2a

    allocations = []
    communicator_initializations = []

    class Buffer:
        @staticmethod
        def get_buffer_size_hint(group, **kwargs):
            return 64

        def __init__(self, group, **kwargs):
            self.group = group
            self.__dict__.update(kwargs)
            allocations.append(self)

    monkeypatch.setitem(sys.modules, "_relax_deep_ep_v2", SimpleNamespace(ElasticBuffer=Buffer))
    monkeypatch.setattr(fused_a2a, "ElasticBuffer", None, raising=False)
    monkeypatch.setattr(fused_a2a, "_elastic_buffer", None)
    # Isolate the NCCL warm-up while exercising the real buffer cache.
    monkeypatch.setattr(
        fused_a2a,
        "torch",
        SimpleNamespace(
            zeros=lambda *args, **kwargs: object(),
            distributed=SimpleNamespace(all_reduce=lambda tensor, group: communicator_initializations.append(group)),
        ),
    )
    group = object()
    buffer = fused_a2a.get_elastic_buffer(group, 256, 2048, 8)
    assert buffer is fused_a2a.get_elastic_buffer(group, 256, 2048, 8)
    assert allocations == [buffer]
    assert communicator_initializations == [group]


def test_fused_expert_dispatch_preserves_counts_and_gradients(monkeypatch):
    """Execute the patched method without CUDA/TE to check dispatch and
    padding."""
    from types import SimpleNamespace

    import torch
    from megatron.core.transformer.moe import experts

    monkeypatch.setattr(experts, "skip_routed_expert_padding", lambda config: False)

    class Offload:
        def __init__(self, value):
            self.value = value

        def __enter__(self):
            return self.value

        def __exit__(self, *args):
            return False

        def group_offload(self, value, **kwargs):
            return value

    monkeypatch.setattr(experts, "off_interface", lambda enabled, value, name: Offload(value))
    config = SimpleNamespace(
        fp8=False,
        fp4=False,
        moe_paged_stash=False,
        delay_offload_until_cuda_graph=False,
        _relax_qwen35_optimized=True,
    )
    counts = torch.tensor([1, 0, 2], dtype=torch.int32)
    hidden = torch.randn(3, 4, requires_grad=True)
    probs = torch.ones(3)
    calls = []

    def fused(value, fc1_counts, probabilities, fc2_counts):
        assert fc1_counts.dtype == torch.int64
        assert fc1_counts is fc2_counts
        assert fc1_counts.tolist() == [1, 0, 2]
        assert value is hidden and probabilities is probs
        calls.append(True)
        return value * 2

    module = SimpleNamespace(
        config=config,
        _use_grouped_tensor=False,
        _fused_ops=(fused,),
        _get_paged_stash_scope=lambda value, _: (value, nullcontext()),
        linear_fc2=SimpleNamespace(use_bias=False),
    )
    result = experts.TEGroupedMLP._fused_forward(module, hidden, counts, probs)
    result.sum().backward()
    torch.testing.assert_close(hidden.grad, torch.full_like(hidden, 2))
    assert calls == [True]
    config._relax_qwen35_optimized = False
    with pytest.raises(RuntimeError, match="moe_use_grouped_tensor"):
        experts.TEGroupedMLP._fused_forward(module, hidden, counts, probs)


def test_conv_backward_passes_strides_to_current_fla_kernel():
    from types import SimpleNamespace

    import torch

    source = ROOT / "relax/models/qwen3_5/ops/causal_conv_backward.py"
    fn = next(n for n in ast.parse(source.read_text()).body if isinstance(n, ast.FunctionDef))
    launches = []

    class Kernel:
        def __getitem__(self, grid):
            def launch(*, stride_dy_n, stride_dy_t, stride_dy_d, **kwargs):
                assert (stride_dy_n, stride_dy_t, stride_dy_d) == kwargs["dy"].stride()
                assert tuple(kwargs[f"stride_dx_{axis}"] for axis in "ntd") == kwargs["dx"].stride()
                kwargs["dx"].copy_(kwargs["dy"])
                kwargs["dw"].zero_()
                launches.append(True)

            return launch

    scope = {
        "torch": torch,
        "triton": SimpleNamespace(
            next_power_of_2=lambda n: 1 << (n - 1).bit_length(), cdiv=lambda n, d: (n + d - 1) // d
        ),
        "causal_conv1d_bwd_kernel": Kernel(),
    }
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(source), "exec"), scope)
    x = torch.randn(1, 7, 10)[..., :4]
    dy = torch.randn(1, 7, 8)[..., ::2]
    dx = torch.empty(1, 7, 12)[..., :4]
    result = scope["causal_conv1d_bwd"](x, dy, None, weight=torch.randn(4, 3), dx_out=dx)
    assert result[0].data_ptr() == dx.data_ptr()
    torch.testing.assert_close(result[0], dy)
    assert launches == [True]
