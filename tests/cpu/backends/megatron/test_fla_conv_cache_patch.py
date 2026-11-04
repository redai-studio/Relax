# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Check static FLA configuration and preserve interpreter numerical tests."""

import ast
import os
from pathlib import Path
from types import SimpleNamespace

import pytest


def _source(relative):
    root = os.environ.get("COMMON_FLA_SOURCE_DIR")
    if not root:
        pytest.skip(
            "Set COMMON_FLA_SOURCE_DIR to FLA 0.5.2 with the static convolution entries and output-config patches"
        )
    return ast.parse((Path(root) / relative).read_text())


@pytest.mark.parametrize("enabled", [False, True])
def test_static_nb_key_preserves_disabled_lengths(monkeypatch, enabled):
    monkeypatch.setenv("RELAX_FLA_CONV_NB_MODE", "original")
    monkeypatch.setenv("RELAX_QWEN38_FLA_CONV_CACHE_REUSE", str(int(enabled)))
    scope = {}
    exec(compile(_source("fla/modules/conv/triton/conv_policy.py"), "policy", "exec"), scope)
    for length in (4096, 16384, 65536, 131072):
        assert scope["conv_nb_kwargs"](1, length) == {"NB": 1 if enabled else length // 1024}


@pytest.mark.parametrize("enabled", [False, True])
def test_static_output_configs_preserve_default_and_selected_profile(enabled):
    tree = _source("fla/ops/common/chunk_o.py")
    kernel = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "chunk_fwd_kernel_o")
    tuner = next(
        n
        for n in kernel.decorator_list
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "fla_cache_autotune"
    )
    configs = next(k.value for k in tuner.keywords if k.arg == "configs")
    scope = {
        "_relax_bool_env": lambda _: enabled,
        "triton": SimpleNamespace(Config=lambda kwargs, **meta: (kwargs, meta)),
    }
    result = eval(compile(ast.Expression(configs), "configs", "eval"), scope)
    assert len(result) == (1 if enabled else 3)
    assert result[0 if enabled else 1] == ({"BK": 64, "BV": 64}, {"num_warps": 4, "num_stages": 3})


@pytest.mark.parametrize("length", [7, 65, 129])
def test_installed_fla_interpreter_nb_preserves_forward_and_gradients(length):
    """Opt-in CPU interpreter test; never launches a CUDA kernel."""
    import os

    if os.environ.get("TRITON_INTERPRET") != "1":
        pytest.skip("Run with TRITON_INTERPRET=1 and installed FLA for CPU kernel verification")
    torch = pytest.importorskip("torch")
    kernels = pytest.importorskip("fla.modules.conv.triton.kernels")
    fwd = kernels.causal_conv1d_fwd_kernel.fn.fn
    bwd = kernels.causal_conv1d_bwd_kernel.fn.fn
    torch.manual_seed(1)
    dim, width, block = 16, 4, 32
    chunks = (length + block - 1) // block
    x = torch.randn(1, length, dim)
    weight, bias, dy = torch.randn(dim, width), torch.randn(dim), torch.randn_like(x)
    results = []
    for nb in (1, 125, 128):
        y = torch.empty_like(x)
        shared = dict(
            x=x,
            weight=weight,
            cu_seqlens=None,
            initial_state=None,
            chunk_indices=None,
            B=1,
            T=length,
            stride_x_n=length * dim,
            stride_x_t=dim,
            stride_x_d=1,
            D=dim,
            W=width,
            BT=block,
            BW=4,
            BD=16,
            NB=nb,
            HAS_WEIGHT=True,
            HAS_BIAS=True,
            USE_INITIAL_STATE=False,
            IS_VARLEN=False,
        )
        fwd[(1, chunks, 1)](y=y, bias=bias, residual=None, ACTIVATION=None, HAS_RESIDUAL=False, **shared)
        dx = torch.empty_like(x)
        dw, db = torch.empty(chunks, dim, width), torch.empty(chunks, dim)
        bwd[(1, chunks, 1)](
            y=y,
            dht=None,
            dy=dy,
            dx=dx,
            dw=dw,
            db=db,
            stride_dx_n=length * dim,
            stride_dx_t=dim,
            stride_dx_d=1,
            ACTIVATION=None,
            USE_FINAL_STATE=False,
            **shared,
        )
        results.append((y, dx, dw, db))
    for result in results[1:]:
        for reference, actual in zip(results[0], result, strict=True):
            torch.testing.assert_close(reference, actual, rtol=0, atol=0)
