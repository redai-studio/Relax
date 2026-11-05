# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from relax.backends.megatron.chunkflow import gdn
from relax.backends.megatron.chunkflow.adapters import AttentionAdapter, register_attention_adapter
from relax.backends.megatron.chunkflow.integration import install, uninstall
from relax.backends.megatron.chunkflow.runtime import RECOMPUTE_ATTR, UNIT_ATTR, ChunkflowRuntime
from relax.backends.megatron.chunkflow.scheduler import build_chunks
from relax.backends.megatron.chunkflow.state_store import ChunkState
from relax.backends.megatron.chunkflow.units import build_micro_batches, completed_groups, gen_unit_ops
from tests.cpu.backends.megatron.test_chunkflow_integration import restore_rope_patch as restore_rope_patch


def _bounds(x, cu_seqlens):
    return [0, x.shape[1]] if cu_seqlens is None else cu_seqlens.tolist()


def _conv(
    x,
    weight,
    bias=None,
    initial_state=None,
    output_final_state=False,
    activation=None,
    cu_seqlens=None,
    cp_context=None,
    cu_seqlens_cpu=None,
):
    if cu_seqlens_cpu is not None:
        assert cu_seqlens_cpu.device.type == "cpu"
        torch.testing.assert_close(cu_seqlens_cpu, cu_seqlens.cpu().long())
    outputs, states = [], []
    for index, (lo, hi) in enumerate(zip(_bounds(x, cu_seqlens)[:-1], _bounds(x, cu_seqlens)[1:])):
        state = (
            x.new_zeros(1, x.shape[-1], weight.shape[-1])
            if initial_state is None
            else initial_state[index : index + 1]
        )
        for t in range(lo, hi):
            state = torch.cat((state[:, :, 1:], x[:, t, :, None]), dim=-1)
            y = (state * weight).sum(-1) + bias
            outputs.append(F.silu(y) if activation else y)
        states.append(state)
    return torch.stack(outputs, dim=1), torch.cat(states) if output_final_state else None


def _rule(
    q,
    k,
    v,
    g,
    beta,
    initial_state=None,
    output_final_state=False,
    cu_seqlens=None,
    cp_context=None,
    cu_seqlens_cpu=None,
):
    if cu_seqlens_cpu is not None:
        assert cu_seqlens_cpu.device.type == "cpu"
        torch.testing.assert_close(cu_seqlens_cpu, cu_seqlens.cpu().long())
    outputs, states = [], []
    bounds = _bounds(q, cu_seqlens)
    for index, (lo, hi) in enumerate(zip(bounds[:-1], bounds[1:])):
        state = (
            q.new_zeros(1, q.shape[2], q.shape[3], v.shape[3])
            if initial_state is None
            else initial_state[index : index + 1]
        )
        for t in range(lo, hi):
            state = state * g[:, t].exp()[..., None, None]
            error = (v[:, t] - torch.einsum("bhk,bhkv->bhv", k[:, t], state)) * beta[:, t, :, None]
            state = state + torch.einsum("bhk,bhv->bhkv", k[:, t], error)
            outputs.append(torch.einsum("bhk,bhkv->bhv", q[:, t] * q.shape[-1] ** -0.5, state))
        states.append(state)
    return torch.stack(outputs, dim=1), torch.cat(states) if output_final_state else None


class GatedDeltaNet(torch.nn.Module):
    def __init__(self, kernels, mode="keep", heads=2, dim=4):
        super().__init__()
        self.kernels, self.mode, self.heads, self.dim = kernels, mode, heads, dim
        self.layer_number, self.sp_size = 1, 1
        self.config = SimpleNamespace(context_parallel_size=1)
        self.gdn_pre_gated_delta_rule_fusion = False
        width = heads * dim
        self.in_proj = torch.nn.Linear(width, 4 * width + 2 * heads)
        self.weight = torch.nn.Parameter(torch.randn(3 * width, 4) * 0.1)
        self.bias = torch.nn.Parameter(torch.randn(3 * width) * 0.1)
        self.out_proj = torch.nn.Linear(width, width)
        self.gated_delta_rule = kernels.rule

    def _resolve_cu_seqlens(self, padded, actual, total, name, cp_size=1):
        assert (padded if padded is not None else actual)[-1].item() == total
        return actual

    def forward(self, hidden_states, attention_mask=None, packed_seq_params=None):
        cu = self._resolve_cu_seqlens(
            packed_seq_params.cu_seqlens_q_padded, packed_seq_params.cu_seqlens_q, hidden_states.shape[0], "q"
        )

        def compute(x):
            return self._forward_compute(x, cu, packed_seq_params)

        return (
            checkpoint(compute, hidden_states, use_reentrant=True)
            if self.mode in ("selective", "selective_last")
            else compute(hidden_states)
        )

    def _forward_compute(self, hidden_states, cu_seqlens_q, packed_seq_params):
        width = self.heads * self.dim
        projected = self.in_proj(hidden_states.transpose(0, 1))
        qkv, gate, g, beta = projected.split([3 * width, width, self.heads, self.heads], dim=-1)
        # Exercise native Megatron's extra convolution padding independently of unit padding.
        total = qkv.shape[1]
        qkv = F.pad(qkv, (0, 0, 0, 3))
        cu_conv = cu_seqlens_q.clone()
        cu_conv[-1] += 3
        conv = getattr(self, "pre_gated_delta_rule", self.kernels.causal_conv1d)
        qkv, _ = conv(
            x=qkv,
            weight=self.weight,
            bias=self.bias,
            activation="silu",
            initial_state=None,
            output_final_state=False,
            cu_seqlens=cu_conv,
            cp_context=None,
        )
        q, k, v = [t.reshape(1, total, self.heads, self.dim) for t in qkv[:, :total].chunk(3, dim=-1)]
        out, _ = self.gated_delta_rule(
            F.normalize(q, dim=-1),
            F.normalize(k, dim=-1),
            v,
            g=F.logsigmoid(g.float()) if g.is_cuda else F.logsigmoid(g),
            beta=beta.sigmoid(),
            initial_state=None,
            output_final_state=False,
            cu_seqlens=cu_seqlens_q,
            cp_context=None,
        )
        return self.out_proj(out.flatten(2) * F.silu(gate)).transpose(0, 1)


register_attention_adapter(GatedDeltaNet, None, AttentionAdapter("test-gdn", gdn.validate, gdn.install))


@pytest.fixture
def kernels(monkeypatch):
    ssm = ModuleType("megatron.core.ssm")
    module = ModuleType("megatron.core.ssm.gated_delta_net")
    module.causal_conv1d, module.rule = _conv, _rule
    ssm.gated_delta_net = module
    monkeypatch.setitem(sys.modules, ssm.__name__, ssm)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return module


def _packed(lengths, unit=None):
    cu = [0]
    for length in lengths:
        cu.append(cu[-1] + length)
    packed = SimpleNamespace(
        qkv_format="thd",
        cu_seqlens_q_cpu=cu,
        cu_seqlens_q=torch.tensor(cu, dtype=torch.int32),
        cu_seqlens_q_padded=None,
        cu_seqlens_kv_padded=None,
    )
    if unit is not None:
        setattr(packed, UNIT_ATTR, unit)
        prefix = unit.chunks[0].segments[0].prefix
        packed.cu_seqlens_q_padded = packed.cu_seqlens_q + prefix
    return packed


def _alignment(kernels, mode, chunk_size=4, device="cpu", heads=2, dim=4, padding=3):
    torch.manual_seed(21)
    lengths = [11, 7, 2, 1] if device == "cpu" else [151, 83, 2, 1]
    model = GatedDeltaNet(kernels, mode, heads, dim).to(
        device=device, dtype=torch.float64 if device == "cpu" else torch.bfloat16
    )
    reference = GatedDeltaNet(kernels, mode, heads, dim).to(device=device, dtype=next(model.parameters()).dtype)
    reference.load_state_dict(model.state_dict())
    reference.mode = "keep"
    x = torch.randn(
        sum(lengths), 1, heads * dim, device=device, dtype=next(model.parameters()).dtype, requires_grad=True
    )
    ref_x = x.detach().clone().requires_grad_(True)
    packed = _packed(lengths)
    packed.cu_seqlens_q = packed.cu_seqlens_q.to(device)
    full_packed = packed
    expected = reference(ref_x, packed_seq_params=packed)
    dout = torch.randn_like(expected)
    if mode in ("last", "selective_last"):
        dout.zero_()
        offset = 0
        for length in lengths:
            dout[offset + length - 1] = 1
            offset += length
    expected.backward(dout)
    assert install(model, chunk_size) == 1
    assert install(model, chunk_size) == 0
    runtime = model.chunkflow_runtime
    chunks = build_chunks(lengths, chunk_size)
    units = build_micro_batches(chunks, "staggered")
    ops = gen_unit_ops(units, chunks, 1 if mode == "fr" else 1 << 30)
    pending, actual, identities = {}, {}, {}
    for kind, uid in ops:
        unit = units[uid]
        indices = [
            sum(lengths[: s.seq_id]) + i
            for c in unit.chunks
            for s in c.segments
            for i in range(s.prefix, s.prefix + s.length)
        ]
        if kind == "B":
            pending.pop(uid).backward(dout[indices])
            for group in completed_groups(unit):
                runtime.release_group(group)
            continue
        packed = _packed([s.length for c in unit.chunks for s in c.segments], unit)
        packed.cu_seqlens_q = packed.cu_seqlens_q.to(device)
        setattr(packed, RECOMPUTE_ATTR, kind == "Fr")
        padded = torch.cat((x[indices], torch.randn(padding, 1, heads * dim, device=device, dtype=x.dtype)))

        def forward(t, params=packed):
            return model(t, packed_seq_params=params)

        if kind == "Fp":
            with torch.no_grad():
                forward(padded)
        else:
            out = checkpoint(forward, padded, use_reentrant=True) if mode == "outer" else forward(padded)
            pending[uid] = out[: unit.num_tokens]
            actual.update(zip(indices, pending[uid].detach()))
        if unit.is_dependent and len(unit.chunks) == 1:
            chunk = unit.chunks[0]
            state = runtime.store(1, chunk.group).lookup(chunk.group_idx)[-1].tensors
            for name, leaf in state.items():
                key = (chunk.group, chunk.group_idx, name)
                assert identities.setdefault(key, id(leaf)) == id(leaf)
    runtime.assert_empty("GDN numerical alignment")

    def check(got, want):
        if device == "cpu":
            torch.testing.assert_close(got, want, rtol=1e-9, atol=1e-10)
        else:
            torch.testing.assert_close(got, want, rtol=0.02, atol=0.02 * want.float().abs().max().item() + 1e-6)

    check(torch.stack([actual[i] for i in range(sum(lengths))]), expected)
    check(x.grad, ref_x.grad)
    for got, want in zip(model.parameters(), reference.parameters()):
        check(got.grad, want.grad)
    with torch.no_grad():
        check(model(x, packed_seq_params=full_packed), expected)
    runtime.assert_empty("GDN baseline pass-through")
    assert uninstall(model) == 1
    assert model.state_dict().keys() == reference.state_dict().keys()
    return torch.stack([actual[i] for i in range(sum(lengths))]), x.grad, [p.grad for p in model.parameters()]


@pytest.mark.parametrize("mode", ["keep", "fr", "selective", "selective_last", "outer"])
@pytest.mark.parametrize("padding", [0, 3])
def test_gdn_execution_optimization_matches_original_chunkflow(kernels, monkeypatch, mode, padding):
    results = []
    for enabled in ("0", "1"):
        monkeypatch.setenv("RELAX_CHUNKFLOW_EXECUTION_OPT", enabled)
        # Each complete install/uninstall restores the shared native kernel.
        results.append(_alignment(kernels, mode, padding=padding))
    torch.testing.assert_close(results[1], results[0], rtol=0, atol=0)


@pytest.mark.parametrize("padding", [0, 3])
def test_gdn_execution_optimization_preserves_input_identity_without_padding(monkeypatch, padding):
    monkeypatch.setenv("RELAX_CHUNKFLOW_EXECUTION_OPT", "1")
    received = []

    def kernel(x, initial_state=None, output_final_state=False, cu_seqlens=None):
        received.append(x)
        return x.square(), x[:, -1].square()

    x = torch.randn(1, 4 + padding, 3, requires_grad=True)
    packed = SimpleNamespace(**{UNIT_ATTR: SimpleNamespace(num_tokens=4)})
    context = gdn._Context(SimpleNamespace(), packed, previous={}, current=ChunkState({}))
    token = gdn._ACTIVE.set(context)
    try:
        out, _ = gdn._wrap_kernel(kernel, "conv")(x)
        out.sum().backward()
    finally:
        gdn._ACTIVE.reset(token)
    assert (received[0] is x) == (padding == 0)
    torch.testing.assert_close(x.grad[:, :4], 2 * x[:, :4], rtol=0, atol=0)
    assert torch.count_nonzero(x.grad[:, 4:]) == 0


@pytest.mark.parametrize("mode,chunk_size", [("keep", 2), ("last", 4)])
def test_gdn_state_outputs_and_gradients_match_full_sequence(kernels, mode, chunk_size):
    _alignment(kernels, mode, chunk_size)


def test_gdn_uses_convolution_from_preprocessing_module(kernels, monkeypatch):
    owner = ModuleType(f"{kernels.__name__}.gdn")
    owner.causal_conv1d = _conv
    exec("def pre_gated_delta_rule(self, **kwargs):\n    return causal_conv1d(**kwargs)", vars(owner))
    monkeypatch.setitem(sys.modules, owner.__name__, owner)
    monkeypatch.setattr(GatedDeltaNet, "pre_gated_delta_rule", owner.pre_gated_delta_rule, raising=False)

    _alignment(kernels, "selective")

    assert kernels.causal_conv1d is _conv
    assert owner.causal_conv1d is _conv


def test_gdn_shared_convolution_restored_after_last_model_uninstall(kernels):
    first, second = GatedDeltaNet(kernels).double(), GatedDeltaNet(kernels).double()
    x = torch.randn(5, 1, 8, dtype=torch.float64)
    with torch.no_grad():
        expected = second(x, packed_seq_params=_packed([5]))
    native = kernels.causal_conv1d
    assert install(first, 4) == 1
    wrapped = kernels.causal_conv1d
    assert wrapped is not native
    assert install(second, 4) == 1
    assert kernels.causal_conv1d is wrapped
    assert uninstall(first) == 1
    assert kernels.causal_conv1d is wrapped
    units = build_micro_batches(build_chunks([5], 4))
    with torch.no_grad():
        outputs = []
        for unit in units:
            start = unit.chunks[0].segments[0].prefix
            outputs.append(
                second(x[start : start + unit.num_tokens], packed_seq_params=_packed([unit.num_tokens], unit))
            )
    torch.testing.assert_close(torch.cat(outputs), expected, atol=1e-10, rtol=1e-9)
    second.chunkflow_runtime.release_group(units[0].chunks[0].group)
    assert uninstall(second) == 1
    assert kernels.causal_conv1d is native
    assert "forward" not in first.__dict__ and "forward" not in second.__dict__


def test_unregistered_gdn_subclass_is_rejected_before_patching(kernels):
    class DifferentRecurrence(GatedDeltaNet):
        pass

    model = DifferentRecurrence(kernels)
    native = kernels.causal_conv1d
    with pytest.raises(NotImplementedError, match="DifferentRecurrence"):
        install(model, 4)
    assert kernels.causal_conv1d is native
    assert "forward" not in model.__dict__
    assert not hasattr(model, "chunkflow_runtime")


def test_gdn_rejects_multiple_dependent_chunks_before_state_access():
    chunks = build_chunks([11, 11], 4)
    unit = next(u for u in build_micro_batches(chunks, "lockstep") if u.is_dependent and len(u.chunks) > 1)
    packed = SimpleNamespace(**{UNIT_ATTR: unit})
    runtime = ChunkflowRuntime(4)

    def original(packed_seq_params):
        raise AssertionError("unsupported units must not enter the kernel")

    forward = gdn._wrap_forward(SimpleNamespace(layer_number=1), original, runtime, prepare=False)
    with pytest.raises(NotImplementedError, match="one dependent chunk per unit"):
        forward(packed_seq_params=packed)
    runtime.assert_empty("rejected unit")


@pytest.mark.parametrize("enabled", ["0", "1"])
@pytest.mark.parametrize("padding", [0, 3])
def test_gdn_reuses_unit_lengths_and_passes_cpu_metadata(kernels, monkeypatch, enabled, padding):
    import inspect
    from functools import wraps

    received = []

    @wraps(_rule)
    def rule(*args, **kwargs):
        received.append(inspect.signature(_rule).bind(*args, **kwargs).arguments.get("cu_seqlens_cpu"))
        return _rule(*args, **kwargs)

    monkeypatch.setenv("RELAX_CHUNKFLOW_EXECUTION_OPT", enabled)
    kernels.rule = rule
    model = GatedDeltaNet(kernels)
    install(model, 4)
    unit = build_micro_batches(build_chunks([2, 1], 4), "staggered")[0]
    packed = _packed([2, 1], unit)
    total = 3 + padding
    x = torch.randn(total, 1, 8)
    first = model(x, packed_seq_params=packed)
    metadata = packed._chunkflow_gdn_lengths
    second = model(x, packed_seq_params=packed)
    assert packed._chunkflow_gdn_lengths is metadata
    assert len(received) == 2 and all(value is metadata[2] for value in received)
    assert metadata[2].tolist() == ([0, 2, 3, total] if padding else [0, 2, 3])
    assert (metadata[1] is packed.cu_seqlens_q) == (enabled == "1" and padding == 0)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    token = gdn._ACTIVE.set(gdn._Context(model, SimpleNamespace(_chunkflow_gdn_lengths=metadata)))
    try:
        conv = gdn._wrap_kernel(_conv, "conv")
        # Matching physical boundaries can share CPU metadata; extra conv padding must bypass it.
        for length in (total, total + 3):
            cu = metadata[1].clone()
            cu[-1] = length
            conv(torch.randn(1, length, 24), model.weight, model.bias, cu_seqlens=cu)
    finally:
        gdn._ACTIVE.reset(token)
    with pytest.raises(ValueError, match="physical unit length changed"):
        model(torch.randn(7, 1, 8), packed_seq_params=packed)
    assert uninstall(model) == 1
    assert kernels.causal_conv1d is _conv


@pytest.mark.parametrize(
    "setting,value", [("fusion", True), ("context_parallel_size", 2), ("recompute_granularity", "full")]
)
def test_gdn_install_rejects_unsupported_configuration(kernels, setting, value):
    model = GatedDeltaNet(kernels)
    if setting == "fusion":
        model.gdn_pre_gated_delta_rule_fusion = value
    else:
        setattr(model.config, setting, value)
    with pytest.raises(NotImplementedError):
        install(model, 4)
