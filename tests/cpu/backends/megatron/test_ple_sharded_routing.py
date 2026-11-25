# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Compare packed SP/CP tap routing to an unsharded per-document depthwise
convolution."""

import itertools

import pytest
import torch
import torch.nn.functional as F

from relax.models.qwen3_8.ops.ple_sharded import _build_route, _convolve


@pytest.mark.parametrize("lengths,kernel", [((16, 0, 32), 4), ((16, 32), 1)])
def test_sharded_taps_match_document_convolution_and_gradients(lengths, kernel):
    torch.manual_seed(315)
    dilation = 3
    # Neither cp-major nor tp-major ordering. Transport slots follow the process group.
    coords = ((1, 2), (0, 3), (1, 0), (0, 1), (0, 2), (1, 1), (0, 0), (1, 3))
    bounds = (0, *itertools.accumulate(lengths))
    docs = [torch.arange(begin, end) for begin, end in zip(bounds, bounds[1:], strict=False) if begin != end]
    cp_rows = [torch.cat([doc.chunk(4)[i] for doc in docs for i in (cp, 3 - cp)]) for cp in range(2)]
    local_rows = [cp_rows[cp].chunk(4)[tp] for cp, tp in coords]
    n, channels = bounds[-1], 5
    activation = torch.randn(n, 1, channels, requires_grad=True)
    gated = torch.randn_like(activation, requires_grad=True)
    weight = torch.randn(channels, 1, kernel, requires_grad=True)
    upstream = torch.randn_like(activation)
    # Supervise only selected ranks. Neighbours can still receive nonzero halo adjoints.
    upstream[local_rows[0]] = 0
    expected = []
    for doc in docs:
        x = activation[doc].permute(1, 2, 0)
        x = F.pad(x, (dilation * (kernel - 1), 0))
        conv = F.conv1d(x, weight, dilation=dilation, groups=channels).permute(2, 0, 1)
        expected.append(gated[doc] + F.silu(conv))
    expected = torch.cat(expected)
    expected_grads = torch.autograd.grad(expected, (activation, gated, weight), upstream)

    routes = [_build_route(bounds, coords, rank, kernel, dilation) for rank in range(len(coords))]
    normed = [activation[rows].detach().requires_grad_() for rows in local_rows]
    residual = [gated[rows].detach().requires_grad_() for rows in local_rows]
    weights = [weight.detach().clone().requires_grad_() for _ in coords]
    # Concatenate owner payloads to emulate just the transport in one shared autograd graph.
    outgoing = []
    for r, (cp_ids, _, send, capacity) in enumerate(routes):
        torch.testing.assert_close(torch.from_numpy(cp_ids), cp_rows[coords[r][0]])
        outgoing.append(F.pad(normed[r][send], (0, 0, 0, 0, 0, capacity - len(send))))
    halo = torch.cat(outgoing, dim=0)
    local_outputs = []
    for rank, rows in enumerate(local_rows):
        actual = _convolve(normed[rank], halo, residual[rank], weights[rank], torch.from_numpy(routes[rank][1]))
        torch.testing.assert_close(actual, expected[rows])
        local_outputs.append(actual)
    torch.autograd.backward(local_outputs, [upstream[rows] for rows in local_rows])
    for r, rows in enumerate(local_rows):
        torch.testing.assert_close(normed[r].grad, expected_grads[0][rows])
        torch.testing.assert_close(residual[r].grad, expected_grads[1][rows])
    torch.testing.assert_close(sum(w.grad for w in weights), expected_grads[2])


def test_long_packed_documents_bound_halo_size():
    coords = tuple(itertools.product(range(2), range(4)))
    for rank in range(8):
        _, taps, _, capacity = _build_route((0, 54192, 107952, 119520), coords, rank, kernel=4, dilation=3)
        assert taps.shape == (4, 14940)
        assert capacity == 18


def test_microbatch_layers_share_routes_but_keep_their_own_hashes(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import torch.distributed as dist
    from megatron.core import parallel_state

    from relax.models.qwen3_8.ops.ple_sharded import prepare_sharded_ple

    dist.init_process_group("gloo", init_method=(tmp_path / "group").as_uri(), rank=0, world_size=1)
    try:
        group = dist.group.WORLD
        monkeypatch.setattr(parallel_state, "get_tensor_and_context_parallel_group", lambda: group)
        monkeypatch.setattr(parallel_state, "get_context_parallel_group", lambda: group)
        layer = SimpleNamespace(
            _ngram_ids=torch.arange(16).view(1, 16, 1), tp_group=group, conv_kernel_size=4, conv_dilation=3
        )
        a = prepare_sharded_ple(layer, (0, 16))
        layer._ngram_ids = layer._ngram_ids + 29
        b = prepare_sharded_ple(layer, (0, 16), a.route)
        assert b.route is a.route
        torch.testing.assert_close(b.local_ngram_ids, a.local_ngram_ids + 29)
        layer.conv_dilation = 1
        c = prepare_sharded_ple(layer, (0, 16), b.route)
        assert c.route is not b.route and not torch.equal(c.route.taps, b.route.taps)
    finally:
        dist.destroy_process_group()
