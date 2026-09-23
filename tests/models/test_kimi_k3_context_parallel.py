# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""K3 CP layout/gradient regression using real Core collectives and CPU
kernels.

The CPU convolution/KDA implementations replace FLA kernels. Core's CP all-to-
all, packed-token permutations, and parameter slicing remain real; its CUDA
allocation target is redirected to CPU for Gloo.
"""

import copy
from datetime import timedelta
from itertools import product
from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")
dist = torch.distributed


def _bridge_layers():
    module = pytest.importorskip("megatron.bridge.models.kimi.kimi_k3_layers")
    if not hasattr(module.KimiK3Attention, "_forward_kda_context_parallel"):
        pytest.skip("Installed Bridge does not include the Kimi K3 CP port.")
    return module


def _cpu_core_all_to_all(monkeypatch):
    from megatron.core.tensor_parallel import mappings

    # Core's unequal all-to-all allocates on cuda.current_device() explicitly.
    # Redirect only this module's allocation target; preserve its real
    # collective and autograd implementation, including the backward exchange.
    cpu_torch = SimpleNamespace(**vars(torch))
    cpu_torch.cuda = SimpleNamespace(**vars(torch.cuda))
    cpu_torch.cuda.current_device = lambda: torch.device("cpu")
    monkeypatch.setattr(mappings, "torch", cpu_torch)


class _ReferenceAllGather(torch.autograd.Function):
    """CPU/Gloo FLA reference gather with an explicit SUM-and-slice adjoint.

    Core's CP collectives and their autograd implementations remain unchanged.
    """

    @staticmethod
    def forward(ctx, value, group):
        ctx.group = group
        outputs = [torch.empty_like(value) for _ in range(group.size())]
        dist.all_gather(outputs, value.contiguous(), group=group)
        return torch.cat(outputs, dim=1)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.contiguous().clone()
        dist.all_reduce(grad_output, group=ctx.group)
        return grad_output.chunk(ctx.group.size(), dim=1)[ctx.group.rank()].contiguous(), None


def _gather_context(value, cp_context):
    if cp_context is None:
        return value
    return _ReferenceAllGather.apply(value, cp_context.group)


def _local_context(value, cp_context):
    if cp_context is None:
        return value
    return value.chunk(cp_context.group.size(), dim=1)[cp_context.group.rank()].contiguous()


def _reference_conv(
    *,
    x,
    weight,
    bias=None,
    activation="silu",
    initial_state=None,
    output_final_state=False,
    cu_seqlens=None,
    cp_context=None,
):
    assert activation == "silu"
    assert initial_state is None and not output_final_state
    x = _gather_context(x, cp_context)
    boundaries = cu_seqlens.tolist() if cu_seqlens is not None else [0, x.shape[1]]
    outputs = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        inputs = torch.nn.functional.pad(x[:, start:end].transpose(1, 2), (weight.shape[-1] - 1, 0))
        outputs.append(
            torch.nn.functional.silu(
                torch.nn.functional.conv1d(inputs, weight.unsqueeze(1), bias=bias, groups=weight.shape[0])
            ).transpose(1, 2)
        )
    return _local_context(torch.cat(outputs, dim=1), cp_context), None


def _reference_kda(q, k, v, g, beta, a_log, dt_bias, lower_bound, *, cu_seqlens=None, cp_context=None):
    q, k, v, g, beta = [_gather_context(value, cp_context) for value in (q, k, v, g, beta)]
    q = torch.nn.functional.normalize(q, dim=-1)
    k = torch.nn.functional.normalize(k, dim=-1)
    decay = torch.exp(
        lower_bound * torch.sigmoid(a_log.exp().view(1, 1, -1, 1) * (g + dt_bias.view(1, 1, *g.shape[-2:])))
    )
    boundaries = cu_seqlens.tolist() if cu_seqlens is not None else [0, q.shape[1]]
    outputs = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        state = q.new_zeros(q.shape[0], q.shape[2], v.shape[-1], q.shape[-1])
        for position in range(start, end):
            state = state * decay[:, position].unsqueeze(-2)
            residual = v[:, position] - torch.einsum("bhvk,bhk->bhv", state, k[:, position])
            update = residual * beta[:, position].unsqueeze(-1)
            state = state + update.unsqueeze(-1) * k[:, position].unsqueeze(-2)
            outputs.append(torch.einsum("bhvk,bhk->bhv", state, q[:, position]) / q.shape[-1] ** 0.5)
    return _local_context(torch.stack(outputs, dim=1), cp_context)


class _Linear(torch.nn.Linear):
    def forward(self, value):
        return super().forward(value), None


class _Convolution(torch.nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(hidden_size, 1, 3) * 0.2)
        self.bias = None

    def forward(self, *, x, **kwargs):
        return _reference_conv(x=x, weight=self.weight.squeeze(1), **kwargs)


class _GatedNorm(torch.nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))

    def forward(self, value, gate):
        value = value * torch.rsqrt(value.square().mean(dim=-1, keepdim=True) + 1e-5)
        return value * self.weight * gate.sigmoid()


def _attention(module, cp_group, *, mode="chunkwise", layout="zigzag"):
    attention = module.KimiK3Attention.__new__(module.KimiK3Attention)
    torch.nn.Module.__init__(attention)
    attention.config = SimpleNamespace(
        linear_cp_mode=mode,
        cp_partition_mode=layout,
        kimi_linear_conv_kernel_size=3,
        gdn_conv_pad_alignment=None,
        cuda_graph_impl="none",
    )
    attention.cp_group = cp_group
    attention.cp_size = cp_group.size() if cp_group is not None else 1
    attention.tp_group = None
    attention.local_num_heads = 4
    attention.head_dim = 2
    attention.local_projection_size = 8
    for name in ("q_proj", "k_proj", "v_proj", "g_proj"):
        setattr(attention, name, _Linear(6, 8, bias=False))
    attention.f_a_proj = _Linear(6, 2, bias=False)
    attention.f_b_proj = _Linear(2, 8, bias=False)
    attention.b_proj = _Linear(6, 4, bias=False)
    attention.o_proj = _Linear(8, 6, bias=False)
    attention.q_conv1d = _Convolution(8)
    attention.k_conv1d = _Convolution(8)
    attention.v_conv1d = _Convolution(8)
    attention.A_log = torch.nn.Parameter(torch.randn(4) * 0.1)
    attention.dt_bias = torch.nn.Parameter(torch.randn(8) * 0.1)
    attention.gate_lower_bound = -5.0
    attention.o_norm = _GatedNorm(2)
    return attention.double()


def _partition_indices(boundaries, rank, cp_size, layout):
    if layout == "contiguous":
        length = boundaries[-1] // cp_size
        return torch.arange(rank * length, (rank + 1) * length)
    pieces = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        chunk = (end - start) // (2 * cp_size)
        for index in (rank, 2 * cp_size - rank - 1):
            pieces.append(torch.arange(start + index * chunk, start + (index + 1) * chunk))
    return torch.cat(pieces)


def _run_distributed_cases(rank, world_size, rendezvous):
    import megatron.bridge.models.kimi.kimi_k3_layers as module
    from megatron.core.packed_seq_params import PackedSeqParams
    from torch.utils.checkpoint import checkpoint

    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=rendezvous, rank=rank, world_size=world_size, timeout=timedelta(seconds=90)
    )
    group = dist.group.WORLD
    module.causal_conv1d = _reference_conv
    module.kda = _reference_kda
    module.build_cp_context = lambda **kwargs: SimpleNamespace(**kwargs)
    monkeypatch = pytest.MonkeyPatch()
    _cpu_core_all_to_all(monkeypatch)
    try:
        # Isolate Core's SBHD transpose and its adjoint before the KDA graph.
        route_inputs = torch.arange(120, dtype=torch.float64).view(20, 1, 6)
        route_indices = _partition_indices([0, 20], rank, world_size, "zigzag")
        route_local = route_inputs.index_select(0, route_indices).requires_grad_()
        route_converted, _ = module.convert_module_input_tensors_cp_partition_mode(
            hidden_states=route_local,
            packed_seq_params=None,
            cp_group=group,
            tp_group=None,
            target_partition_mode="contiguous",
            sequence_parallel=False,
            config=SimpleNamespace(cp_partition_mode="zigzag"),
        )
        coefficients = (route_inputs + 1).square()
        (route_converted * coefficients.chunk(world_size)[rank]).sum().backward()
        torch.testing.assert_close(
            route_local.grad, coefficients.index_select(0, route_indices), msg="Core SBHD conversion adjoint"
        )
        for (mode, packed, layout, padded, batch), recompute in product(
            (
                ("headwise", False, "zigzag", False, 1),
                ("headwise", False, "zigzag", False, 2),
                ("headwise", True, "zigzag", False, 1),
                ("headwise", True, "zigzag", True, 1),
                ("chunkwise", False, "zigzag", False, 1),
                ("chunkwise", True, "zigzag", False, 1),
                ("chunkwise", True, "zigzag", True, 1),
                ("chunkwise", True, "contiguous", False, 1),
            ),
            (False, True),
        ):
            case = f"{mode=}, {packed=}, {layout=}, {padded=}, {batch=}, {recompute=}, {rank=}"
            torch.manual_seed(41)
            baseline = _attention(module, None)
            parallel = _attention(module, group, mode=mode, layout=layout)
            parallel.load_state_dict(baseline.state_dict())
            boundaries = [0, 8, 20] if packed else [0, 20]
            full_input = torch.randn(20, batch, 6, dtype=torch.float64, requires_grad=True)
            indices = _partition_indices(boundaries, rank, world_size, layout)
            local_input = full_input.detach().index_select(0, indices).requires_grad_()
            params = None
            if packed:
                cu = torch.tensor(boundaries, dtype=torch.int32)
                actual_cu = torch.tensor([0, 5, 14], dtype=torch.int32) if padded else cu
                params = PackedSeqParams(
                    qkv_format="thd",
                    cu_seqlens_q=actual_cu,
                    cu_seqlens_kv=actual_cu,
                    cu_seqlens_q_padded=cu,
                    cu_seqlens_kv_padded=cu,
                    cp_partition_mode=layout,
                )
                params.cu_seqlens_q_cpu = list(boundaries)
            reference_params = copy.copy(params)
            if reference_params is not None:
                # The oracle includes each packed sequence's physical padding.
                reference_params.cu_seqlens_q = reference_params.cu_seqlens_q_padded
                reference_params.cu_seqlens_kv = reference_params.cu_seqlens_kv_padded
            expected = baseline._forward_kda(full_input, reference_params)
            if recompute:
                actual = checkpoint(parallel._forward_kda, local_input, params, use_reentrant=True)
            else:
                actual = parallel._forward_kda(local_input, params)
            torch.testing.assert_close(actual, expected.index_select(0, indices), atol=2e-7, rtol=2e-6, msg=case)
            assert params is None or params.cp_partition_mode == layout
            actual.square().sum().backward()
            assert params is None or params.cp_partition_mode == layout
            expected.square().sum().backward()
            torch.testing.assert_close(
                local_input.grad,
                full_input.grad.index_select(0, indices),
                atol=3e-6,
                rtol=3e-5,
                msg=lambda details: f"{case}\n{details}",
            )
            baseline_parameters = dict(baseline.named_parameters())
            for name, parameter in parallel.named_parameters():
                assert parameter.grad is not None, (mode, name)
                dist.all_reduce(parameter.grad, group=group)
                torch.testing.assert_close(
                    parameter.grad, baseline_parameters[name].grad, atol=3e-6, rtol=3e-5, msg=f"{case}, {name}"
                )
    finally:
        monkeypatch.undo()
        dist.destroy_process_group()


def test_kimi_k3_fixed_cp2_matches_cp1_outputs_inputs_and_all_parameter_gradients(tmp_path):
    _bridge_layers()
    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("K3 CP numerical CPU test requires PyTorch Gloo support.")
    torch.multiprocessing.spawn(
        _run_distributed_cases,
        args=(2, (tmp_path / "cp_init").as_uri()),
        nprocs=2,
        join=True,
    )


@pytest.mark.parametrize(
    ("mode", "layout", "batch", "message"),
    [
        ("headwise", "contiguous", 1, "requires zigzag"),
        ("chunkwise", "zigzag", 2, "micro_batch_size=1"),
    ],
)
def test_kimi_k3_cp_rejects_unsupported_layouts_before_collectives(mode, layout, batch, message):
    module = _bridge_layers()
    attention = _attention(module, SimpleNamespace(size=lambda: 2), mode=mode, layout=layout)
    with pytest.raises(ValueError, match=message):
        attention._forward_kda(torch.randn(10, batch, 6), None)


@pytest.mark.parametrize(
    ("query", "key_value", "message"),
    [
        ([0, 8, 20], [0, 12, 20], "matching query"),
        ([1, 8, 20], [1, 8, 20], "start at zero"),
        ([0, 12, 8, 20], [0, 12, 8, 20], "strictly increase"),
        ([0, 6, 20], [0, 6, 20], "divisible by 2"),
        ([0, 8, 16], [0, 8, 16], "global CP sequence"),
    ],
)
def test_kimi_k3_cp_rejects_invalid_physical_boundaries_before_collectives(query, key_value, message):
    module = _bridge_layers()
    attention = _attention(module, SimpleNamespace(size=lambda: 2), mode="headwise")
    params = SimpleNamespace(
        qkv_format="thd",
        cp_partition_mode="zigzag",
        cu_seqlens_q_padded=torch.tensor(query, dtype=torch.int32),
        cu_seqlens_kv_padded=torch.tensor(key_value, dtype=torch.int32),
    )
    with pytest.raises(RuntimeError, match=message):
        attention._forward_kda(torch.randn(10, 1, 6), params)


def test_kimi_k3_chunkwise_cp_rejects_short_convolution_halo_before_collectives():
    module = _bridge_layers()
    attention = _attention(module, SimpleNamespace(size=lambda: 2))
    with pytest.raises(ValueError, match="convolution kernel size - 1"):
        attention._forward_kda(torch.randn(1, 1, 6), None)


def test_kimi_k3_cp_rejects_dynamic_groups_before_collectives():
    module = _bridge_layers()
    attention = _attention(module, None)
    with pytest.raises(ValueError, match="fixed context parallel"):
        attention._forward_kda(torch.randn(10, 1, 6), SimpleNamespace(local_cp_size=2))


def test_kimi_k3_kda_kernel_preserves_safe_gate_and_transposed_cp_state(monkeypatch):
    _bridge_layers()
    ops = pytest.importorskip("megatron.bridge.models.kimi.kimi_k3_ops")
    fla_kda = pytest.importorskip("fla.ops.kda")
    captured = {}
    result = torch.empty(1, 8, 4, 2)

    def chunk_kda(**kwargs):
        captured.update(kwargs)
        return result, None

    monkeypatch.setattr(fla_kda, "chunk_kda", chunk_kda)
    monkeypatch.setattr(ops, "_patch_fla_kda_hopper_autotune", lambda: None)
    cp_context = object()
    cu_seqlens = torch.tensor([0, 8], dtype=torch.int32)
    assert (
        ops.kda(
            result,
            result,
            result,
            result,
            result[..., 0],
            result[0, 0, :, 0],
            result[0, 0].flatten(),
            -5.0,
            cu_seqlens=cu_seqlens,
            cp_context=cp_context,
        )
        is result
    )
    assert captured["cp_context"] is cp_context
    assert captured["cu_seqlens"] is cu_seqlens
    assert captured["safe_gate"] and captured["use_gate_in_kernel"] and captured["transpose_state_layout"]
    assert captured["lower_bound"] == -5.0
    assert captured["initial_state"] is None and not captured["output_final_state"]


@pytest.mark.parametrize("as_tensor", [False, True])
def test_kimi_k3_chunkwise_cp_passes_host_physical_boundaries_to_fla(monkeypatch, as_tensor):
    module = _bridge_layers()
    attention = _attention(module, SimpleNamespace(size=lambda: 2), layout="contiguous")
    physical = torch.tensor([0, 8, 20], dtype=torch.int32)
    params = SimpleNamespace(
        qkv_format="thd",
        cp_partition_mode="contiguous",
        cu_seqlens_q=torch.tensor([0, 5, 14], dtype=torch.int32),
        cu_seqlens_kv=torch.tensor([0, 5, 14], dtype=torch.int32),
        cu_seqlens_q_padded=physical,
        cu_seqlens_kv_padded=physical,
        cu_seqlens_q_cpu=physical if as_tensor else [0, 8, 20],
    )

    def inspect_context(**kwargs):
        assert kwargs["cu_seqlens"] is physical
        assert kwargs["cu_seqlens_cpu"].device.type == "cpu"
        torch.testing.assert_close(kwargs["cu_seqlens_cpu"], physical)
        # End this boundary-contract test before any CP collective or FLA kernel.
        raise RuntimeError("captured physical host boundaries")

    monkeypatch.setattr(module, "build_cp_context", inspect_context)
    with pytest.raises(RuntimeError, match="captured physical host boundaries"):
        attention._forward_kda(torch.randn(10, 1, 6), params)

    params.cu_seqlens_q_cpu = [0, 5, 14]
    with pytest.raises(RuntimeError, match="host boundaries must match the physical"):
        attention._forward_kda(torch.randn(10, 1, 6), params)


def test_kimi_k3_chunkwise_cp_reuses_microbatch_route_built_from_host_boundaries(monkeypatch):
    module = _bridge_layers()
    attention = _attention(module, SimpleNamespace(size=lambda: 2, rank=lambda: 0))
    physical = torch.tensor([0, 8, 20], dtype=torch.int32)
    params = SimpleNamespace(
        qkv_format="thd",
        cp_partition_mode="zigzag",
        cu_seqlens_q_padded=physical,
        cu_seqlens_kv_padded=physical,
        cu_seqlens_q_cpu=[0, 8, 20],
    )
    build_route = module.build_thd_cp_partition_route
    routes = []

    def record_route(boundaries, *args, **kwargs):
        assert boundaries.device.type == "cpu"
        route = build_route(boundaries, *args, **kwargs)
        routes.append(route)
        return route

    def inspect_conversion(**kwargs):
        assert kwargs["packed_seq_params"].cp_partition_route is routes[0]
        raise RuntimeError("captured cached host route")

    monkeypatch.setattr(module, "build_thd_cp_partition_route", record_route)
    monkeypatch.setattr(module, "convert_module_input_tensors_cp_partition_mode", inspect_conversion)
    for _ in range(2):
        with pytest.raises(RuntimeError, match="captured cached host route"):
            attention._forward_kda(torch.randn(10, 1, 6), params)
    assert len(routes) == 1
    assert params.cp_partition_mode == "zigzag"
