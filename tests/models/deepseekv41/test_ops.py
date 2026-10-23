# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""CPU contracts for the V4.1 ops; GPU kernels are explicitly replaced by
stubs."""

import importlib
import math
import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn
from torch.nn import functional as F


pytest.importorskip("megatron.core.extensions.transformer_engine")

from relax.models.deepseekv41.ops import compressor, indexer, position, quant  # noqa: E402


_KERNEL_PACKAGE = "relax.models.deepseekv41.ops.kernels"


def _stub_kernel(monkeypatch: pytest.MonkeyPatch, name: str, **functions: Any) -> None:
    module = ModuleType(f"{_KERNEL_PACKAGE}.{name}")
    for symbol, function in functions.items():
        setattr(module, symbol, function)
    monkeypatch.setitem(sys.modules, module.__name__, module)


def _yarn_freqs_reference(
    dim: int, seqlen: int, original: int, base: float, factor: float, fast: float, slow: float
) -> torch.Tensor:
    # Independent plain RoPE and YaRN frequency equations.
    exponents = torch.arange(dim // 2, dtype=torch.float32) * (2 / dim)
    frequencies = torch.pow(torch.tensor(base), -exponents)
    if original:
        fast_dim = dim * math.log(original / (fast * 2 * math.pi)) / (2 * math.log(base))
        slow_dim = dim * math.log(original / (slow * 2 * math.pi)) / (2 * math.log(base))
        lo, hi = max(math.floor(fast_dim), 0), min(math.ceil(slow_dim), dim - 1)
        if lo == hi:
            hi += 0.001
        blend = ((torch.arange(dim // 2) - lo) / (hi - lo)).clamp(0, 1)
        frequencies = frequencies * (1 - blend) + frequencies / factor * blend
    angles = torch.arange(seqlen).unsqueeze(1) * frequencies.unsqueeze(0)
    return torch.complex(angles.cos(), angles.sin())


@pytest.mark.parametrize(
    "dim,original,base,factor",
    [(64, 0, 10000, 16), (64, 65536, 160000, 16), (2, 1, 10000, 16)],
)
def test_rope_frequencies_match_plain_yarn_and_collapsed_ramp(
    dim: int, original: int, base: float, factor: float
) -> None:
    actual = position.precompute_freqs_cis(dim, 17, original, base, factor, 32, 1, torch.device("cpu"))
    expected = _yarn_freqs_reference(dim, 17, original, base, factor, 32, 1)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)


@pytest.mark.parametrize("compressed", [False, True])
def test_attention_freqs_selects_compressed_yarn_or_plain_window(compressed: bool) -> None:
    config = SimpleNamespace(
        qk_pos_emb_head_dim=8,
        original_max_position_embeddings=65536,
        csa_compress_rotary_base=160000,
        rotary_base=10000,
        rotary_scaling_factor=16,
        beta_fast=32,
        beta_slow=1,
    )
    actual = position.attention_freqs(config, 12, compressed, torch.device("cpu"))
    expected = _yarn_freqs_reference(8, 12, 65536 if compressed else 0, 160000 if compressed else 10000, 16, 32, 1)
    torch.testing.assert_close(actual, expected)


def test_contiguous_cp_positions_and_frequency_slice_use_explicit_group(monkeypatch: pytest.MonkeyPatch) -> None:
    group = object()
    seen = []

    def rank(process_group: Any) -> int:
        seen.append(process_group)
        return 2

    monkeypatch.setattr(torch.distributed, "get_rank", rank)
    torch.testing.assert_close(position.cp_positions(4, group, torch.device("cpu")), torch.tensor([8, 9, 10, 11]))
    frequencies = torch.arange(16)
    torch.testing.assert_close(position.cp_freqs(frequencies, 4, group, stride=2), torch.tensor([8, 10]))
    torch.testing.assert_close(position.cp_positions(4, None, torch.device("cpu")), torch.arange(4))
    assert seen == [group, group]


def test_window_indices_preserve_causality_and_contiguous_global_positions() -> None:
    actual = position.window_indices(torch.tensor([0, 3, 4, 5]), 4, 8, 2)
    expected = torch.tensor([[0, -1, -1, -1], [0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5]], dtype=torch.int32)
    torch.testing.assert_close(actual, expected.expand(2, -1, -1))
    short = position.window_indices(torch.tensor([0, 1, 2]), 8, 3, 1)
    torch.testing.assert_close(short, torch.tensor([[[0, -1, -1], [0, 1, -1], [0, 1, 2]]], dtype=torch.int32))


def test_all_gather_cp_does_not_cache_tensors_or_microbatch_graphs(monkeypatch: pytest.MonkeyPatch) -> None:
    group = object()
    seen = []

    def gather(tensor: torch.Tensor, *, group: Any) -> tuple[torch.Tensor, torch.Tensor]:
        seen.append((tensor, group))
        return tensor, tensor * 2

    monkeypatch.setattr(position, "all_gather", gather)
    first = torch.ones(2, 3, requires_grad=True)
    second = torch.full((2, 3), 4.0, requires_grad=True)
    first_out = position.all_gather_cp(first, 1, group)
    second_out = position.all_gather_cp(second, 1, group)
    assert len(seen) == 2 and seen[0][0] is first and seen[1][0] is second
    assert all(call[1] is group for call in seen)
    torch.testing.assert_close(second_out, first_out.detach() * 4)
    torch.testing.assert_close(torch.autograd.grad(first_out.sum(), first)[0], torch.full_like(first, 3))
    torch.testing.assert_close(torch.autograd.grad(second_out.sum(), second)[0], torch.full_like(second, 3))


def _compressor_reference(
    x: torch.Tensor, wkv: torch.Tensor, wgate: torch.Tensor | None, norm: torch.Tensor, ratio: int, eps: float
) -> torch.Tensor:
    if ratio == 1:
        values = F.linear(x, wkv)
    else:
        groups = x.shape[0] // ratio
        # One FP32 view per input/weight: sum group gradients in
        # FP32 before a single CastBackward converts them to the BF16 leaf.
        tokens32 = x[: groups * ratio].float()
        wkv32, wgate32 = wkv.float(), wgate.float()
        parts = []
        for group in range(groups):
            tokens = tokens32[group * ratio : (group + 1) * ratio]
            projected = F.linear(tokens, wkv32)
            scores = F.linear(tokens, wgate32)
            parts.append((projected * scores.softmax(dim=0)).sum(dim=0))
        values = torch.stack(parts).to(wkv.dtype)
    values32 = values.float()
    normalized = values32 * torch.rsqrt(values32.square().mean(-1, keepdim=True) + eps)
    return (normalized * norm).to(values.dtype)


@pytest.mark.parametrize("ratio", [1, 2])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_compressor_matches_per_group_reference_and_gradients(ratio: int, dtype: torch.dtype) -> None:
    torch.manual_seed(11)
    config = SimpleNamespace(hidden_size=4, params_dtype=dtype, layernorm_epsilon=1e-6)
    module = compressor.DeepSeekV41Compressor(config, 3, ratio)
    x = torch.randn(5, 2, 4, dtype=dtype, requires_grad=True)
    parameters = [module.linear_wkv.weight]
    if ratio == 2:
        parameters.append(module.linear_wgate.weight)
    parameters.append(module.norm.weight)
    actual = module(x)
    actual_gradients = torch.autograd.grad(actual.float().sum(), [x, *parameters])
    reference_inputs = [tensor.detach().clone().requires_grad_() for tensor in [x, *parameters]]
    reference_x, wkv = reference_inputs[:2]
    wgate = reference_inputs[2] if ratio == 2 else None
    expected = _compressor_reference(reference_x, wkv, wgate, reference_inputs[-1], ratio, config.layernorm_epsilon)
    expected_gradients = torch.autograd.grad(expected.float().sum(), reference_inputs)
    torch.testing.assert_close(actual, expected)
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(
            actual_gradient,
            expected_gradient,
            atol=0.016 if dtype == torch.bfloat16 else 1e-6,
            rtol=0.016 if dtype == torch.bfloat16 else 1e-5,
        )
    assert module.norm.weight.dtype == torch.float32
    if ratio == 2:
        assert torch.count_nonzero(actual_gradients[0][-1]) == 0
    else:
        assert not hasattr(module, "linear_wgate")


@pytest.mark.parametrize("ratio", [0, 4, 128])
def test_compressor_rejects_unsupported_ratios(ratio: int) -> None:
    with pytest.raises(ValueError):
        compressor.DeepSeekV41Compressor(SimpleNamespace(), 32, ratio)


def test_candidate_blocks_force_last_partial_block_and_drop_infinite_padding() -> None:
    logits = torch.tensor([[[100.0, 99.0, 0.0, -torch.inf, -torch.inf, -torch.inf, -torch.inf], [-torch.inf] * 7]])
    lengths = torch.tensor([[3], [0]])
    actual = indexer.select_candidate_blocks(logits, lengths, 1, 2)
    expected = torch.tensor([[[False, False, True, True, False, False, False], [False] * 7]])
    torch.testing.assert_close(actual, expected)
    all_valid = indexer.select_candidate_blocks(logits, lengths, 8, 2)
    torch.testing.assert_close(all_valid[0, 0], torch.tensor([True, True, True, True, False, False, False]))
    assert not all_valid[0, 1].any()


class _CpuTELinear(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, *, config: Any, **kwargs: Any) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(out_dim, in_dim, dtype=config.params_dtype) / in_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
        return F.linear(x, self.weight), None


def _cpu_rotate(x: torch.Tensor, freqs: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    complex_x = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)).contiguous())
    shape = (1, x.shape[1], *([1] * (x.ndim - 3)), freqs.shape[-1])
    rotation = freqs.conj() if inverse else freqs
    rotated = torch.view_as_real(complex_x * rotation.view(shape)).flatten(-2)
    x.copy_(rotated.to(x.dtype))
    return x


def _indexer_fixture(monkeypatch: pytest.MonkeyPatch, layer_id: int, source: int = 0) -> indexer.DeepSeekV41Indexer:
    config = SimpleNamespace(
        dsa_indexer_n_heads=2,
        dsa_indexer_head_dim=4,
        dsa_indexer_topk=3,
        qk_pos_emb_head_dim=2,
        v41_candidate_source_layer_id=source,
        v41_candidate_topk_blocks=1,
        v41_candidate_block_size=2,
        q_lora_rank=4,
        hidden_size=4,
        kv_lora_rank=4,
        params_dtype=torch.float32,
        layernorm_epsilon=1e-6,
        init_method=lambda tensor: tensor,
    )
    monkeypatch.setattr(indexer, "TELinear", _CpuTELinear)
    monkeypatch.setattr(indexer, "apply_rotary_emb", _cpu_rotate)
    monkeypatch.setattr(indexer, "fake_quant_fp4", lambda tensor: tensor)
    return indexer.DeepSeekV41Indexer(config, layer_id, owns_k=True)


def _causal_scores(q: torch.Tensor, keys: torch.Tensor, weights: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    columns = torch.arange(keys.shape[1], dtype=torch.float32)
    logits = columns.expand(q.shape[0], q.shape[1], -1).clone()
    return logits.masked_fill(columns >= lengths.unsqueeze(-1), -torch.inf)


def _indexer_inputs() -> tuple[torch.Tensor, ...]:
    x = torch.randn(2, 5, 4, requires_grad=True)
    qr = torch.randn(2, 5, 4, requires_grad=True)
    keys = torch.randn(2, 5, 4, requires_grad=True)
    frequencies = torch.ones(5, 1, dtype=torch.complex64)
    lengths = torch.tensor([0, 1, 3, 4, 5])
    return x, qr, keys, frequencies, lengths


def test_indexer_query_chunk_matches_full_query_and_preserves_causal_padding(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _indexer_fixture(monkeypatch, 0)
    seen = []

    def logits(q: torch.Tensor, keys: torch.Tensor, weights: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        seen.append(q.shape[1])
        return _causal_scores(q, keys, weights, lengths)

    monkeypatch.setattr(indexer, "indexer_logits", logits)
    inputs = _indexer_inputs()
    module.query_chunk = 2
    actual, candidates = module(*inputs, None)
    assert seen == [2, 2, 1]
    module.query_chunk = 8192
    expected, expected_candidates = module(*inputs, None)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(candidates, expected_candidates)
    torch.testing.assert_close(actual[0], torch.tensor([[-1, -1, -1], [-1, -1, 0], [0, 1, 2], [1, 2, 3], [2, 3, 4]]))
    assert not actual.requires_grad and not candidates.requires_grad
    assert all(not parameter.requires_grad for parameter in module.parameters())


def test_indexer_candidate_consumer_applies_mask_and_requires_source(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _indexer_fixture(monkeypatch, 1)
    monkeypatch.setattr(indexer, "indexer_logits", _causal_scores)
    inputs = _indexer_inputs()
    candidates = torch.zeros(2, 5, 5, dtype=torch.bool)
    candidates[..., 0] = True
    module.query_chunk = 2
    actual, returned = module(*inputs, candidates)
    assert returned is candidates
    assert (actual[..., :2] == -1).all()
    assert (actual[:, 1:, -1] == 0).all()
    with pytest.raises(RuntimeError, match="candidate source"):
        module(*inputs, None)


def test_indexer_pads_when_keys_are_fewer_than_topk(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _indexer_fixture(monkeypatch, 0, source=-1)
    monkeypatch.setattr(indexer, "indexer_logits", _causal_scores)
    x, qr, keys, frequencies, _ = _indexer_inputs()
    indices, candidates = module(x, qr, keys[:, :1], frequencies, torch.ones(5, dtype=torch.long), None)
    assert candidates is None
    torch.testing.assert_close(indices, torch.tensor([-1, -1, 0]).expand(2, 5, -1))


def test_indexer_index_keys_normalizes_rotates_and_freezes_gradients(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _indexer_fixture(monkeypatch, 0)
    latent = torch.randn(2, 5, 4, requires_grad=True)
    frequencies = position.precompute_freqs_cis(2, 5, 0, 10000, 16, 32, 1, torch.device("cpu"))
    expected = module.k_norm(F.linear(latent, module.linear_wk.weight)).detach().clone()
    _cpu_rotate(expected[..., -2:], frequencies)
    actual = module.index_keys(latent, frequencies)
    torch.testing.assert_close(actual, expected)
    assert not actual.requires_grad


def _round_ste(tensor: torch.Tensor) -> torch.Tensor:
    return tensor + (tensor.round() - tensor).detach()


def test_kv_norm_rope_fp8_uses_hard_cache_values_and_soft_normalization_gradients(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(quant, "apply_rotary_emb", _cpu_rotate)
    monkeypatch.setattr(quant, "_fp8_ste", _round_ste)
    calls = []

    def hard(values: torch.Tensor, weight: torch.Tensor, eps: float, freqs: torch.Tensor) -> torch.Tensor:
        calls.append((values, weight, eps, freqs))
        return torch.full_like(values, 0.25)

    monkeypatch.setattr(quant, "_cache_roundtrip", hard)
    values = torch.randn(2, 3, 8, requires_grad=True)
    weight = torch.randn(8, requires_grad=True)
    frequencies = position.precompute_freqs_cis(2, 3, 0, 10000, 16, 32, 1, torch.device("cpu"))
    actual = quant.kv_norm_rope_fp8(values, weight, 1e-6, frequencies, 2)
    torch.testing.assert_close(actual, torch.full_like(actual, 0.25))
    actual_gradients = torch.autograd.grad(actual.sum(), (values, weight))
    reference_values = values.detach().clone().requires_grad_()
    reference_weight = weight.detach().clone().requires_grad_()
    soft = reference_values * torch.rsqrt(reference_values.square().mean(-1, keepdim=True) + 1e-6) * reference_weight
    _cpu_rotate(soft[..., -2:], frequencies)
    soft = soft.clone()
    soft[..., :-2] = _round_ste(soft[..., :-2])
    expected_gradients = torch.autograd.grad(soft.sum(), (reference_values, reference_weight))
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(actual_gradient, expected_gradient)
    assert calls[0][0] is values and calls[0][1] is weight and calls[0][3] is frequencies


def test_compressed_kv_stored_uses_hard_values_but_preserves_identity_gradient(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(quant, "_fp8_ste", _round_ste)
    monkeypatch.setattr(quant, "_cache_roundtrip", lambda latent: torch.full_like(latent, 0.5))
    latent = torch.randn(2, 3, 8, requires_grad=True)
    actual = quant.compressed_kv_stored(latent, 2)
    torch.testing.assert_close(actual, torch.full_like(latent, 0.5))
    torch.testing.assert_close(torch.autograd.grad(actual.sum(), latent)[0], torch.ones_like(latent))


def _sparse_module() -> Any:
    return importlib.import_module("relax.models.deepseekv41.ops.sparse_attention")


@pytest.mark.parametrize("width,multiple,expected", [(63, 64, 64), (640, 128, 640)])
def test_sparse_topk_padding_uses_minus_one_not_a_valid_kv_row(width: int, multiple: int, expected: int) -> None:
    module = _sparse_module()
    indices = torch.arange(width, dtype=torch.int32).view(1, 1, 1, width)
    actual = module._pad_topk_block(indices, multiple)
    assert actual.shape[-1] == expected and actual.is_contiguous()
    torch.testing.assert_close(actual[..., :width], indices)
    assert (actual[..., width:] == -1).all()


@pytest.mark.parametrize(
    "major,heads,dim,groups,available,expected",
    [
        (9, 8, 512, 1, True, "tilelang"),
        (9, 16, 512, 1, True, "tilelang"),
        (9, 32, 512, 1, True, "flash_mla"),
        (9, 64, 512, 1, True, "flash_mla"),
        (10, 8, 512, 1, True, "flash_mla"),
        (10, 8, 576, 1, True, "flash_mla"),
        (8, 64, 512, 1, True, "tilelang"),
        (9, 64, 512, 2, True, "tilelang"),
        (9, 64, 512, 1, False, "tilelang"),
        (9, 64, 128, 1, True, "tilelang"),
    ],
)
def test_sparse_backend_selection_covers_hopper_without_cuda_initialization(
    monkeypatch: pytest.MonkeyPatch, major: int, heads: int, dim: int, groups: int, available: bool, expected: str
) -> None:
    module = _sparse_module()
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (major, 0))
    monkeypatch.setattr(module, "flash_mla_sparse_fwd", (lambda *args, **kwargs: None) if available else None)
    q, kv = torch.empty(1, 2, heads, dim), torch.empty(1, 3, groups, dim)
    config = module._default_config(q, kv, 512)
    assert config.forward_backend == expected


def test_flash_mla_wrapper_pads_heads_topk_batch_offsets_and_sink_lse(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _sparse_module()
    captured = []

    def flash(
        q: torch.Tensor, kv: torch.Tensor, indices: torch.Tensor, scale: float, *, d_v: int, attn_sink: torch.Tensor
    ) -> tuple[torch.Tensor, None, torch.Tensor]:
        captured.append((q, kv, indices, scale, d_v, attn_sink))
        return torch.zeros_like(q), None, torch.full(q.shape[:2], 3.0)

    monkeypatch.setattr(module, "flash_mla_sparse_fwd", flash)
    q, kv = torch.zeros(2, 3, 3, 512), torch.zeros(2, 5, 1, 512)
    indices = torch.tensor([0, 4, -1], dtype=torch.int32).expand(2, 3, 1, -1)
    sink = torch.tensor([0.0, 1.0, 2.0])
    output, lse = module._flash_mla_forward(q, kv, indices, sink, 0.5)
    forwarded_q, forwarded_kv, forwarded_indices, _, d_v, forwarded_sink = captured[0]
    assert forwarded_q.shape == (6, 64, 512) and forwarded_kv.shape == (10, 1, 512)
    assert forwarded_indices.shape == (6, 1, 128) and d_v == 512
    torch.testing.assert_close(forwarded_indices[:3, 0, :3], torch.tensor([0, 4, -1], dtype=torch.int32).expand(3, -1))
    torch.testing.assert_close(forwarded_indices[3:, 0, :3], torch.tensor([5, 9, -1], dtype=torch.int32).expand(3, -1))
    assert (forwarded_indices[..., 3:] == -1).all()
    torch.testing.assert_close(forwarded_sink[:3], sink)
    assert torch.count_nonzero(forwarded_sink[3:]) == 0
    assert output.shape == (2, 3, 3, 512)
    expected_lse = torch.logaddexp2(torch.full((3,), 3 * module._LOG2_E), sink * module._LOG2_E)
    torch.testing.assert_close(lse, expected_lse.expand(2, 3, -1))


def test_sparse_wrapper_sink_backward_and_kernel_settings_use_cpu_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _sparse_module()
    forward_calls, backward_calls = [], []

    def forward(
        q: torch.Tensor,
        kv: torch.Tensor,
        indices: torch.Tensor,
        kv_indices: torch.Tensor,
        sink: torch.Tensor,
        d_v: int,
        *,
        sm_scale: float,
        threads: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        forward_calls.append((indices, kv_indices, threads, sm_scale))
        return torch.zeros(*q.shape[:-1], d_v), torch.full(q.shape[:-1], 2.0)

    def backward(
        q: torch.Tensor,
        kv: torch.Tensor,
        out: torch.Tensor,
        grad: torch.Tensor,
        indices: torch.Tensor,
        kv_indices: torch.Tensor,
        lse: torch.Tensor,
        d_v: int,
        *,
        sm_scale: float,
        num_stages: int,
        split_store: int,
    ) -> tuple[torch.Tensor, ...]:
        backward_calls.append((num_stages, split_store, sm_scale))
        return torch.ones_like(q), torch.ones_like(kv), torch.ones_like(lse)

    _stub_kernel(monkeypatch, "sparse_attention_fwd", sparse_attention_fwd=forward)
    _stub_kernel(monkeypatch, "sparse_attention_bwd", sparse_attention_bwd=backward)
    q, kv = torch.randn(2, 3, 2, 8, requires_grad=True), torch.randn(2, 4, 1, 8, requires_grad=True)
    sink = torch.tensor([0.0, 1.0], requires_grad=True)
    indices = torch.tensor([0, 1, -1], dtype=torch.int32).expand(2, 3, 1, -1)
    config = module.SparseAttentionConfig("tilelang", 128, 2, 4)
    output = module.sparse_attention(q, kv, indices, 0.25, attn_sink=sink, config=config)
    q_grad, kv_grad, sink_grad = torch.autograd.grad(output.sum(), (q, kv, sink))
    torch.testing.assert_close(q_grad, torch.ones_like(q))
    torch.testing.assert_close(kv_grad, torch.ones_like(kv))
    torch.testing.assert_close(sink_grad, -6 * torch.exp2(sink.detach() * module._LOG2_E - 2))
    assert forward_calls[0][0].shape[-1] == 64 and forward_calls[0][2:] == (128, 0.25)
    assert (forward_calls[0][1] >= 0).all()
    assert backward_calls == [(2, 4, 0.25)]
