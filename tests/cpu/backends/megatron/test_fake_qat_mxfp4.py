# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Tests for the MXFP4 fake-QAT STE used by Kimi K3 RL training.

The training forward must see exactly the values the rollout engine serves,
i.e. ``dequantize(quantize(w))`` with the SAME arithmetic as the weight-push
path (Bridge's ``quantize_mxfp4_e2m1_like_scale`` + E8M0 scale grid). The
vectorized implementation under test is therefore checked for bitwise equality
against a verbatim transcription of that reference arithmetic.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")

pytest.importorskip("megatron.core")

# ---------------------------------------------------------------------------
# Reference arithmetic: verbatim transcription of NVIDIA Megatron-Bridge's
# quantize_mxfp4_e2m1_like_scale + dequantize_mxfp4_e2m1_packed
# (src/megatron/bridge/models/conversion/quantization_utils.py). The weight
# push path calls the real implementation; this copy exists so the equivalence
# test runs without a cluster-side bridge install.
# ---------------------------------------------------------------------------

_FP4_E2M1_TABLE = [
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
]
_FP4_E2M1_MAX = 6.0
_MXFP4_BLOCK_SIZE = 32


def _reference_quantize_mxfp4_e2m1(
    weight: torch.Tensor, *, block_size: int = _MXFP4_BLOCK_SIZE
) -> tuple[torch.Tensor, torch.Tensor]:
    """Transcription of Bridge's quantize_mxfp4_e2m1_like_scale (uint8 scale
    branch)."""
    rows, cols = weight.shape
    weight_f32 = weight.to(torch.float32)
    packed = torch.empty((rows, cols // 2), dtype=torch.uint8, device=weight.device)
    scale_f32 = torch.empty((rows, cols // block_size), dtype=torch.float32, device=weight.device)
    boundaries = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], dtype=torch.float32, device=weight.device)

    max_chunk_elements = 16_000_000
    rows_per_chunk = max(1, min(rows, max_chunk_elements // max(cols, 1)))
    scale_cols = cols // block_size
    for row_start in range(0, rows, rows_per_chunk):
        row_end = min(row_start + rows_per_chunk, rows)
        chunk = weight_f32[row_start:row_end].reshape(-1, scale_cols, block_size)
        chunk_amax = chunk.abs().amax(dim=-1)
        unrounded_scale = torch.where(
            chunk_amax > 0,
            chunk_amax / _FP4_E2M1_MAX,
            torch.ones_like(chunk_amax),
        )
        chunk_scale = torch.exp2(torch.ceil(torch.log2(unrounded_scale)).clamp(min=-127, max=127))
        scale_f32[row_start:row_end] = chunk_scale

        normalized = chunk / chunk_scale[:, :, None]
        codes = torch.bucketize(normalized.abs(), boundaries).to(torch.uint8)
        codes = (codes | ((normalized < 0).to(torch.uint8) * 8)).reshape(row_end - row_start, cols)

        lo = codes[:, 0::2].to(torch.int16)
        hi = codes[:, 1::2].to(torch.int16)
        packed[row_start:row_end] = (lo | (hi << 4)).to(torch.uint8)

    output_scale = (torch.log2(scale_f32).round() + 127).clamp(min=0, max=254).to(torch.uint8)
    return packed.contiguous().view(torch.int8), output_scale


def _reference_dequantize_mxfp4_e2m1_packed(
    weight_packed: torch.Tensor, scale: torch.Tensor, *, dtype: torch.dtype = torch.bfloat16
) -> torch.Tensor:
    """Transcription of Bridge's dequantize_mxfp4_e2m1_packed."""
    w_u8 = weight_packed.view(torch.uint8)
    lo = (w_u8 & 0xF).to(torch.int64)
    hi = (w_u8 >> 4).to(torch.int64)

    table = torch.tensor(_FP4_E2M1_TABLE, dtype=torch.float32, device=weight_packed.device)
    logical = torch.stack([table[lo], table[hi]], dim=-1).reshape(weight_packed.shape[0], -1)

    scale_f32 = torch.ldexp(
        torch.ones_like(scale, dtype=torch.float32),
        scale.to(torch.int32) - 127,
    )
    block_size = logical.shape[1] // scale_f32.shape[1]
    scale_exp = scale_f32.repeat_interleave(block_size, dim=1)

    return (logical * scale_exp).to(dtype)


def _reference_quantize_dequantize(weight: torch.Tensor) -> torch.Tensor:
    packed, scale = _reference_quantize_mxfp4_e2m1(weight)
    return _reference_dequantize_mxfp4_e2m1_packed(packed, scale, dtype=weight.dtype)


# ---------------------------------------------------------------------------
# Quantize-dequantize arithmetic
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape",
    [
        (1, 32),  # single group
        (4, 64),  # multiple groups per row
        (128, 3584),  # K3 expert w1/w3 row geometry (reduced rows)
        (64, 3072),  # K3 expert w2 row geometry (reduced rows)
        (7, 96),  # odd row count
    ],
)
@pytest.mark.parametrize("scale", [1.0, 1e-3, 1e3])
def test_fake_qat_mxfp4_matches_reference_bitwise(shape, scale):
    from relax.backends.megatron.fake_qat_mxfp4 import mxfp4_fake_quantize_tensor

    w = (torch.randn(shape, dtype=torch.float32) * scale).to(torch.bfloat16)
    expected = _reference_quantize_dequantize(w)
    actual = mxfp4_fake_quantize_tensor(w)
    assert actual.dtype == w.dtype
    assert torch.equal(actual, expected)


def test_fake_qat_mxfp4_zero_groups_stay_zero():
    from relax.backends.megatron.fake_qat_mxfp4 import mxfp4_fake_quantize_tensor

    w = torch.zeros(2, 64, dtype=torch.bfloat16)
    w[0, :32] = 1.0
    out = mxfp4_fake_quantize_tensor(w)
    assert torch.equal(out[0, 32:], torch.zeros(32, dtype=torch.bfloat16))
    assert torch.equal(out[1], torch.zeros(64, dtype=torch.bfloat16))
    assert torch.equal(out, _reference_quantize_dequantize(w))


def test_fake_qat_mxfp4_extreme_amax_no_nan():
    from relax.backends.megatron.fake_qat_mxfp4 import mxfp4_fake_quantize_tensor

    w = torch.zeros(4, 64, dtype=torch.bfloat16)
    w[0, 0] = 1e-30  # log2 scale far below -127 -> clamped
    w[1, 0] = 1e30  # large but well inside E8M0 range
    w[2, 0] = 6e37  # amax/6 near fp32 range edge
    w[3, 0] = -1e-38  # subnormal-ish tiny negative
    out = mxfp4_fake_quantize_tensor(w)
    assert torch.isfinite(out.float()).all()
    assert torch.equal(out, _reference_quantize_dequantize(w))


def test_fake_qat_mxfp4_preserves_dtype():
    from relax.backends.megatron.fake_qat_mxfp4 import mxfp4_fake_quantize_tensor

    for dtype in (torch.bfloat16, torch.float32):
        w = torch.randn(8, 64, dtype=dtype)
        assert mxfp4_fake_quantize_tensor(w).dtype == dtype


def test_fake_qat_mxfp4_output_lies_on_e2m1_grid():
    from relax.backends.megatron.fake_qat_mxfp4 import mxfp4_fake_quantize_tensor

    w = torch.randn(16, 128, dtype=torch.bfloat16)
    out = mxfp4_fake_quantize_tensor(w).float()
    groups = out.view(16, 4, 32)
    amax = groups.abs().amax(dim=-1)
    nonzero = amax > 0
    # Per-group scale is a power of two; out / scale must land on the E2M1 grid.
    scale = torch.exp2(torch.ceil(torch.log2(amax[nonzero] / _FP4_E2M1_MAX)))
    codes = groups[nonzero] / scale.unsqueeze(-1)
    grid = torch.tensor(_FP4_E2M1_TABLE)
    assert (codes.unsqueeze(-1) == grid).any(dim=-1).all()


def test_fake_qat_mxfp4_rejects_bad_geometry():
    from relax.backends.megatron.fake_qat_mxfp4 import mxfp4_fake_quantize_tensor

    with pytest.raises(ValueError, match="2-D"):
        mxfp4_fake_quantize_tensor(torch.randn(32, dtype=torch.bfloat16))
    with pytest.raises(ValueError, match="32"):
        mxfp4_fake_quantize_tensor(torch.randn(4, 40, dtype=torch.bfloat16))


# ---------------------------------------------------------------------------
# STE wrapper: autograd + main_grad propagation
# ---------------------------------------------------------------------------


def test_fake_qat_ste_backward_passes_gradient_through():
    from relax.backends.megatron.fake_qat_mxfp4 import mxfp4_fake_quant_ste

    w = torch.randn(8, 64, dtype=torch.float32).to(torch.bfloat16).requires_grad_(True)
    out = mxfp4_fake_quant_ste(w)
    grad = torch.randn_like(out)
    out.backward(grad)
    assert w.grad is not None
    assert torch.equal(w.grad, grad)


def test_fake_qat_ste_propagates_main_grad():
    from relax.backends.megatron.fake_qat_mxfp4 import mxfp4_fake_quant_ste

    w = torch.nn.Parameter(torch.randn(4, 64, dtype=torch.bfloat16))
    main_grad = torch.zeros_like(w)
    w.main_grad = main_grad
    out = mxfp4_fake_quant_ste(w)
    assert out.main_grad is main_grad


# ---------------------------------------------------------------------------
# TEGroupedLinear monkey-patch installer
# ---------------------------------------------------------------------------


class _StubGroupedLinear:
    """Stands in for megatron's TEGroupedLinear in installer tests."""

    def __init__(self, weights):
        self._weights = weights

    def _get_weight_tensors(self):
        return list(self._weights)


def test_install_fake_qat_wraps_get_weight_tensors():
    from relax.backends.megatron.fake_qat_mxfp4 import (
        install_mxfp4_fake_qat,
        mxfp4_fake_quantize_tensor,
    )

    w = torch.randn(8, 64, dtype=torch.bfloat16).requires_grad_(True)
    module = _StubGroupedLinear([w])
    assert install_mxfp4_fake_qat(_StubGroupedLinear) is True
    try:
        (out,) = module._get_weight_tensors()
        assert torch.equal(out.detach(), mxfp4_fake_quantize_tensor(w.detach()))
        out.backward(torch.ones_like(out))
        assert w.grad is not None  # autograd still reaches the master weight
    finally:
        _teardown_stub()


def test_install_fake_qat_idempotent():
    from relax.backends.megatron.fake_qat_mxfp4 import install_mxfp4_fake_qat

    orig = _StubGroupedLinear._get_weight_tensors

    class _Counting(_StubGroupedLinear):
        pass

    assert install_mxfp4_fake_qat(_Counting) is True
    try:
        wrapped = _Counting._get_weight_tensors
        assert install_mxfp4_fake_qat(_Counting) is False
        assert _Counting._get_weight_tensors is wrapped  # not wrapped twice
    finally:
        _Counting._get_weight_tensors = orig
        del _Counting._mxfp4_fake_qat_orig_get_weight_tensors
        del _Counting._mxfp4_fake_qat_installed


def test_install_fake_qat_missing_hook_raises():
    from relax.backends.megatron.fake_qat_mxfp4 import install_mxfp4_fake_qat

    class _NoHook:
        pass

    with pytest.raises(RuntimeError, match="_get_weight_tensors"):
        install_mxfp4_fake_qat(_NoHook)


def _teardown_stub():
    if hasattr(_StubGroupedLinear, "_mxfp4_fake_qat_orig_get_weight_tensors"):
        _StubGroupedLinear._get_weight_tensors = _StubGroupedLinear._mxfp4_fake_qat_orig_get_weight_tensors
        del _StubGroupedLinear._mxfp4_fake_qat_orig_get_weight_tensors
        del _StubGroupedLinear._mxfp4_fake_qat_installed


@pytest.fixture(autouse=True)
def _restore_stub_class():
    yield
    _teardown_stub()


# ---------------------------------------------------------------------------
# Env gate
# ---------------------------------------------------------------------------


def test_maybe_install_env_flag_off_is_noop(monkeypatch):
    from relax.backends.megatron.fake_qat_mxfp4 import maybe_install_mxfp4_fake_qat

    monkeypatch.delenv("OPEN_TRAINING_MXFP4_FAKE_QAT_FLAG", raising=False)
    assert maybe_install_mxfp4_fake_qat() is False


def test_maybe_install_env_flag_on_installs(monkeypatch):
    import relax.backends.megatron.fake_qat_mxfp4 as fake_qat

    monkeypatch.setenv("OPEN_TRAINING_MXFP4_FAKE_QAT_FLAG", "1")
    installed = []
    monkeypatch.setattr(fake_qat, "_import_te_grouped_linear", lambda: _StubGroupedLinear)
    monkeypatch.setattr(fake_qat, "install_mxfp4_fake_qat", lambda cls: installed.append(cls) or True)
    assert fake_qat.maybe_install_mxfp4_fake_qat() is True
    assert installed == [_StubGroupedLinear]


# ---------------------------------------------------------------------------
# Integration: real Bridge arithmetic without importing optional GPU models.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", [(1, 32), (7, 96), (64, 3584), (64, 3072)])
@pytest.mark.parametrize("magnitude", [1e-3, 1.0, 1e3])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_fake_qat_mxfp4_matches_real_bridge_implementation(monkeypatch, shape, magnitude, dtype):
    bridge_spec = importlib.util.find_spec("megatron.bridge")
    if bridge_spec is None:
        pytest.skip("Requires Bridge quantization source")
    path = Path(bridge_spec.origin).parent / "models/conversion/quantization_utils.py"
    if not path.exists():
        pytest.skip("Requires Bridge MXFP4 quantization source")
    functions = {node.name: node for node in ast.parse(path.read_text()).body if isinstance(node, ast.FunctionDef)}
    # Older CI images have these functions but treat uint8 scales as numeric
    # values, rather than E8M0 bytes. Only skip that legacy API; once either
    # E8M0 branch is present, every arithmetic mismatch must fail below.
    e8m0_guards = (
        ("quantize_mxfp4_e2m1_like_scale", "source_scale.dtype == torch.uint8"),
        ("dequantize_mxfp4_e2m1_packed", "scale.dtype == torch.uint8"),
    )
    if not any(
        ast.unparse(node) == guard
        for name, guard in e8m0_guards
        if name in functions
        for node in ast.walk(functions[name])
        if isinstance(node, ast.Compare)
    ):
        pytest.skip(
            "Requires Bridge uint8 E8M0 support (Dockerfile.cu13 pin 6862ee170a2569607e8ebc5dde91f119779093ca)"
        )
    spec = importlib.util.spec_from_file_location("_test_real_bridge_quantization", path)
    bridge_utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bridge_utils)
    if not hasattr(bridge_utils, "quantize_mxfp4_e2m1_like_scale"):
        pytest.skip("This Bridge version has no MXFP4 quantizer")
    monkeypatch.setitem(sys.modules, "megatron.bridge.models.conversion.quantization_utils", bridge_utils)

    from relax.backends.megatron.fake_qat_mxfp4 import mxfp4_fake_quantize_tensor
    from relax.backends.megatron.weight_update.bridge_converter import _Mxfp4OnlineQuantizer

    generator = torch.Generator().manual_seed(123)
    w = (torch.randn(shape, generator=generator) * magnitude).to(dtype)
    template = torch.empty((w.shape[0], w.shape[1] // 32), dtype=torch.uint8)
    packed, scale = bridge_utils.quantize_mxfp4_e2m1_like_scale(w, template, block_size=32)
    expected = bridge_utils.dequantize_mxfp4_e2m1_packed(packed, scale, dtype=dtype)
    assert torch.equal(mxfp4_fake_quantize_tensor(w), expected)
    online_packed, online_scale = _Mxfp4OnlineQuantizer()(w, (1, 32))
    assert torch.equal(online_packed, packed)
    assert torch.equal(online_scale, scale)
