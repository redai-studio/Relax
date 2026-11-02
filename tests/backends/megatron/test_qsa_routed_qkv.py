# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""CPU source-layout oracle.

No Megatron import or GPU initialization.
"""

import importlib.util
from pathlib import Path

import pytest
import torch


# Load the standalone routing math without importing the Megatron package bootstrap.
_SPEC = importlib.util.spec_from_file_location(
    "qsa_routed_qkv_under_test",
    Path(__file__).resolve().parents[3] / "relax/models/qwen3_8/qsa_routed_qkv.py",
)
module = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(module)
pack, route_table, unpack = module.pack, module.route_table, module.unpack


def expected(full, tp, rank):
    g, j = divmod(rank, tp // 2)
    group = full[:, g * 6656 : (g + 1) * 6656]
    width = 6144 // tp
    return torch.cat(
        (
            group[:, j * width : (j + 1) * width],
            group[:, 3072 + j * width : 3072 + (j + 1) * width],
            group[:, 6144:6400],
            group[:, 6400:6656],
        ),
        dim=-1,
    )


@pytest.mark.parametrize("tp", [4, 8])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("rows", [1, 7])
def test_exact_forward_and_adjoint(tp, dtype, rows):
    torch.manual_seed(14)
    table = route_table(tp)
    full = torch.randn(rows, 13312, dtype=dtype)
    shards = list(full.chunk(tp, dim=-1))
    packed = [pack(shards[r], table[r], inverse=False) for r in range(tp)]
    split = [[rows * sum(piece[2] for piece in route) for route in table[r]] for r in range(tp)]
    messages = [packed[r].split(split[r]) for r in range(tp)]
    gradients = [torch.randint(-3, 4, (rows, 12288 // tp + 512)).to(dtype) for _ in range(tp)]
    backwards = []
    for rank in range(tp):
        incoming = tuple(table[src][rank] for src in range(tp))
        receive = torch.cat([messages[src][rank] for src in range(tp)])
        result = unpack(receive, incoming, rows, 12288 // tp + 512, inverse=False)
        torch.testing.assert_close(result, expected(full, tp, rank), rtol=0, atol=0)
        backward = pack(gradients[rank], incoming, inverse=True)
        backwards.append(backward.split([split[src][rank] for src in range(tp)]))
    ref = full.float().requires_grad_()
    loss = sum((expected(ref, tp, rank) * gradients[rank].float()).sum() for rank in range(tp))
    loss.backward()
    for rank in range(tp):
        receive = torch.cat([backwards[consumer][rank] for consumer in range(tp)])
        result = unpack(receive, table[rank], rows, 13312 // tp, inverse=True)
        torch.testing.assert_close(result.float(), ref.grad.chunk(tp, dim=-1)[rank], rtol=0, atol=0)


@pytest.mark.parametrize("tp", [4, 8])
def test_column_multiplicity_and_empty_peer_routes(tp):
    table = route_table(tp)
    for source in range(tp):
        multiplicity = torch.zeros(13312 // tp, dtype=torch.int32)
        for route in table[source]:
            for begin, _, width in route:
                multiplicity[begin : begin + width] += 1
        global_columns = torch.arange(source * 13312 // tp, (source + 1) * 13312 // tp) % 6656
        reference = torch.where(global_columns >= 6144, tp // 2, 1)
        assert torch.equal(multiplicity, reference)
        assert sum(not route for route in table[source]) >= tp // 2
    for consumer in range(tp):
        coverage = torch.zeros(12288 // tp + 512, dtype=torch.int32)
        for source in range(tp):
            for _, begin, width in table[source][consumer]:
                coverage[begin : begin + width] += 1
        assert torch.equal(coverage, torch.ones_like(coverage))


def test_unsupported_tp_rejected():
    with pytest.raises(ValueError):
        route_table(2)


@pytest.mark.parametrize("tp", [4, 8])
def test_actual_autograd_wrapper_with_mock_transport(tp, monkeypatch):

    rows = 2
    full = torch.randn(rows, 13312)
    grad = [torch.randint(-2, 3, (rows, 12288 // tp + 512)).float() for _ in range(tp)]
    reference = full.clone().requires_grad_()
    sum((expected(reference, tp, r) * grad[r]).sum() for r in range(tp)).backward()
    table = route_table(tp)
    monkeypatch.setattr(module.dist, "get_world_size", lambda group: tp)
    for rank in range(tp):
        monkeypatch.setattr(module.dist, "get_rank", lambda group: rank)
        backward = [False]

        def exchange(output, input, output_split_sizes, input_split_sizes, group):
            if not backward[0]:
                pieces = [pack(full.chunk(tp, -1)[src], (table[src][rank],), inverse=False) for src in range(tp)]
                torch.testing.assert_close(input, pack(full.chunk(tp, -1)[rank], table[rank], inverse=False))
            else:
                pieces = [pack(grad[consumer], (table[rank][consumer],), inverse=True) for consumer in range(tp)]
            assert [piece.numel() for piece in pieces] == output_split_sizes
            output.copy_(torch.cat(pieces))

        monkeypatch.setattr(module.dist, "all_to_all_single", exchange)
        local = full.chunk(tp, -1)[rank].clone().view(rows, 1, -1).requires_grad_()
        output = module.routed_projection(local, object())
        torch.testing.assert_close(output[:, 0], expected(full, tp, rank), rtol=0, atol=0)
        backward[0] = True
        output.backward(grad[rank].view(rows, 1, -1))
        torch.testing.assert_close(local.grad[:, 0], reference.grad.chunk(tp, -1)[rank], rtol=0, atol=0)


@pytest.mark.parametrize("tp", [4, 8])
def test_explicit_projection_preserves_fallback(monkeypatch, tp):
    import sys
    import types
    from types import SimpleNamespace

    calls = []

    class SelfAttention:
        def _relax_native_get_query_key_value_tensors(self, *args):
            calls.append(("fallback", args))
            return "native"

    fake = types.ModuleType("megatron.core.transformer.attention")
    fake.SelfAttention = SelfAttention
    fake.apply_module = lambda fn: fn
    monkeypatch.setitem(sys.modules, fake.__name__, fake)
    attention = SelfAttention()
    attention.world_size = tp
    attention.pg_collection = SimpleNamespace(tp=object())
    attention.core_attention = type("QSACoreAttention", (), {})()
    attention.config = SimpleNamespace(
        num_attention_heads=24,
        num_query_groups=2,
        kv_channels=256,
        attention_output_gate=True,
        context_parallel_size=1,
    )
    hidden = torch.randn(3, 1, 8)
    mixed = torch.randn(3, 1, 13312 // tp)
    local = torch.randn(3, 1, 12288 // tp + 512)

    def project(value):
        assert value is hidden
        calls.append("projection")
        return mixed, None

    def route(value, group):
        assert value is mixed and group is attention.pg_collection.tp
        calls.append("route")
        return local

    attention.linear_qkv = project
    attention.q_layernorm = lambda x: x * 2
    attention.k_layernorm = lambda x: x * 3
    monkeypatch.setattr(module, "routed_projection", route)
    q, k, v, gate = module.forward(attention, hidden, output_gate=True)
    qr, gr, kr, vr = local.split((6144 // tp, 6144 // tp, 256, 256), dim=-1)
    for actual, reference in [(q, qr * 2), (k, kr * 3), (v, vr), (gate, gr)]:
        torch.testing.assert_close(actual.flatten(-2), reference, rtol=0, atol=0)
    assert calls == ["projection", "route"]
    attention.config.context_parallel_size = 2
    assert module.forward(attention, hidden, output_gate=True) == "native"
    assert calls[-1] == ("fallback", (hidden, None, True, False, True))
    attention.config.context_parallel_size = 1
    assert module.forward(attention, hidden, output_gate=True, head_wise_gate=True) == "native"
    assert calls[-1] == ("fallback", (hidden, None, True, True, True))
