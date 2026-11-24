# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""CPU tests for the module brackets of the ``modules`` level: which blocks are
timed, and the forward / backward brackets they record, with and without
activation recompute."""

import pytest


torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402
from torch.utils.checkpoint import checkpoint  # noqa: E402

from relax.utils.straggler.core.modules import module_block  # noqa: E402
from relax.utils.straggler.megatron_adapter import MODULE_NAMES  # noqa: E402
from relax.utils.straggler.megatron_adapter import module_timers as ModuleTimers  # noqa: E402


class _Sink:
    """The collector side of the brackets: events are increasing integers."""

    def __init__(self):
        self.clock = 0
        self.recording = True
        self.pushed = []
        self.discarded = []
        self.errors = []

    def record_event(self):
        if not self.recording:
            return None
        self.clock += 1
        return self.clock

    def push(self, segment, layer, start, end, cpu_ms):
        name, backward = module_block(MODULE_NAMES, segment)
        self.pushed.append((f"{name}_{'bwd' if backward else 'fwd'}", layer, start, end))

    def discard_event(self, event):
        self.discarded.append(event)

    def on_error(self, where, exc):
        self.errors.append((where, exc))

    @property
    def names(self):
        return [(name, layer) for name, layer, _, _ in self.pushed]


class _Block(nn.Module):
    """Returns ``(output, bias)`` like Megatron's attention and MLP."""

    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 4)

    def forward(self, hidden_states, padding_mask=None):
        return self.proj(hidden_states), None


class MoELayer(_Block):
    pass


class _Router(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = nn.Linear(4, 2)

    def forward(self, hidden_states):
        return self.gate(hidden_states).softmax(-1)


class _Dispatcher:
    """A plain object with the token exchange as methods, like Megatron's token
    dispatchers."""

    def token_dispatch(self, hidden_states, probs):
        return hidden_states * 1.0, probs

    def token_combine(self, hidden_states):
        return hidden_states * 1.0


class _Exchange(torch.autograd.Function):
    """Stand-in for an all-to-all: passes the tensor through its own autograd
    node and logs the sink's clock when its backward runs."""

    @staticmethod
    def forward(ctx, tensor, sink, log, name):
        ctx.sink, ctx.log, ctx.name = sink, log, name
        return tensor.clone()

    @staticmethod
    def backward(ctx, grad):
        ctx.log.append((ctx.name, ctx.sink.record_event()))
        return grad, None, None, None


class _ExchangingDispatcher(_Dispatcher):
    """Exchanges the tokens, then the probabilities, each through its own node,
    like Megatron's all-to-all dispatcher; backward runs the probabilities'
    exchange first."""

    def __init__(self, sink, log):
        self._sink, self._log = sink, log

    def token_dispatch(self, hidden_states, probs):
        tokens = _Exchange.apply(hidden_states, self._sink, self._log, "tokens")
        return tokens, _Exchange.apply(probs, self._sink, self._log, "probs")


class _Experts(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 4)

    def forward(self, hidden_states, probs):
        return self.proj(hidden_states) * probs[..., :1], None


class _RoutedMoELayer(nn.Module):
    """Megatron's MoE layer in miniature: route, dispatch, experts, combine,
    shared experts."""

    def __init__(self):
        super().__init__()
        self.router = _Router()
        self.token_dispatcher = _Dispatcher()
        self.experts = _Experts()
        self.shared_experts = nn.Linear(4, 4)

    def forward(self, hidden_states, padding_mask=None):
        probs = self.router(hidden_states)
        tokens, probs = self.token_dispatcher.token_dispatch(hidden_states.clone(), probs)
        output, _ = self.experts(tokens, probs)
        output = self.token_dispatcher.token_combine(output).view_as(hidden_states)
        return output + self.shared_experts(hidden_states), None


class _SortingMoELayer(_RoutedMoELayer):
    """With several experts per rank, Megatron sorts the exchanged tokens and
    probabilities by expert before the experts, and back after them."""

    def forward(self, hidden_states, padding_mask=None):
        probs = self.router(hidden_states)
        tokens, probs = self.token_dispatcher.token_dispatch(hidden_states.clone(), probs)
        output, _ = self.experts(tokens.flip(0), probs.flip(0))
        output = self.token_dispatcher.token_combine(output.flip(0)).view_as(hidden_states)
        return output + self.shared_experts(hidden_states), None


class IdentityOp(nn.Module):
    def forward(self, x):
        return x


class _Layer(nn.Module):
    def __init__(self, layer_number, mlp=None, recompute=False):
        super().__init__()
        self.layer_number = layer_number
        self.self_attention = _Block()
        self.mlp = mlp if mlp is not None else _Block()
        self.recompute = recompute

    def _body(self, x):
        x = x + self.self_attention(x)[0]
        return x + self.mlp(hidden_states=x)[0]

    def forward(self, x):
        return checkpoint(self._body, x, use_reentrant=True) if self.recompute else self._body(x)


class _MtpLayer(_Layer):
    is_mtp_layer = True


def _run(model, x):
    model(x).sum().backward()


def test_attention_and_mlp_of_every_layer_are_found_and_identity_blocks_skipped():
    model = nn.Sequential(_Layer(1), _Layer(2, mlp=MoELayer()), _Layer(3, mlp=IdentityOp()))
    assert ModuleTimers([model], _Sink()).counts() == {"attn": 3, "mlp": 1, "moe": 1}


def test_forward_and_backward_brackets_follow_the_execution_order_and_leave_gradients_unchanged():
    torch.manual_seed(0)
    model = nn.Sequential(_Layer(1), _Layer(2, mlp=MoELayer()))
    x = torch.randn(2, 4, requires_grad=True)
    _run(model, x)
    expected = [p.grad.clone() for p in model.parameters()] + [x.grad.clone()]
    model.zero_grad()
    x.grad = None

    sink = _Sink()
    timers = ModuleTimers([model], sink)
    timers.attach()
    _run(model, x)
    assert sink.names == [
        ("attn_fwd", 1),
        ("mlp_fwd", 1),
        ("attn_fwd", 2),
        ("moe_fwd", 2),
        ("moe_bwd", 2),
        ("attn_bwd", 2),
        ("mlp_bwd", 1),
        ("attn_bwd", 1),
    ]
    spans = sorted((start, end) for _, _, start, end in sink.pushed)
    assert all(start < end <= next_start for (start, end), (next_start, _) in zip(spans, spans[1:])), spans
    assert all(torch.equal(a, b) for a, b in zip([p.grad for p in model.parameters()] + [x.grad], expected))
    assert (sink.discarded, sink.errors) == ([], [])

    timers.detach()
    _run(model, x)
    assert len(sink.pushed) == 8


def test_the_parts_of_a_moe_layer_nest_inside_it_forward_and_backward_and_leave_gradients_unchanged():
    torch.manual_seed(0)
    model = nn.Sequential(_Layer(1, mlp=_RoutedMoELayer()))
    x = torch.randn(2, 4, requires_grad=True)
    _run(model, x)
    expected = [p.grad.clone() for p in model.parameters()]
    model.zero_grad()

    parts = ("router", "dispatch", "experts", "combine", "shared experts")
    sink = _Sink()
    timers = ModuleTimers([model], sink)
    assert timers.counts() == {"attn": 1, "moe": 1, **dict.fromkeys(parts, 1)}
    timers.attach()
    _run(model, x)
    spans = {name: (start, end) for name, _, start, end in sink.pushed}
    for direction in ("fwd", "bwd"):
        outer_start, outer_end = spans[f"moe_{direction}"]
        for part in parts:
            start, end = spans[f"{part}_{direction}"]
            assert outer_start < start < end < outer_end, (part, direction, spans)
    assert all(torch.equal(p.grad, e) for p, e in zip(model.parameters(), expected))
    assert (sink.discarded, sink.errors) == ([], [])

    timers.detach()
    assert "token_dispatch" not in model[0].mlp.token_dispatcher.__dict__


# With one expert per rank the experts take the exchanged tensors themselves.
@pytest.mark.parametrize("layer_class", [_RoutedMoELayer, _SortingMoELayer], ids=["one_expert_per_rank", "sorted"])
def test_the_dispatch_backward_holds_both_exchanges_and_the_experts_backward_neither(layer_class):
    torch.manual_seed(0)
    sink, exchanges = _Sink(), []
    moe = layer_class()
    moe.token_dispatcher = _ExchangingDispatcher(sink, exchanges)
    model = nn.Sequential(_Layer(1, mlp=moe))
    x = torch.randn(2, 4, requires_grad=True)
    _run(model, x)
    expected = [p.grad.clone() for p in model.parameters()]
    model.zero_grad()
    exchanges.clear()

    ModuleTimers([model], sink).attach()
    _run(model, x)
    spans = {name: (start, end) for name, _, start, end in sink.pushed}
    start, end = spans["dispatch_bwd"]
    assert [name for name, _ in exchanges] == ["probs", "tokens"]
    assert all(start < at < end for _, at in exchanges), (spans, exchanges)
    assert spans["experts_bwd"][1] < exchanges[0][1], (spans, exchanges)
    assert all(torch.equal(p.grad, e) for p, e in zip(model.parameters(), expected))
    assert (sink.discarded, sink.errors) == ([], [])


@pytest.mark.parametrize("layer_class", [_RoutedMoELayer, _SortingMoELayer], ids=["one_expert_per_rank", "sorted"])
def test_where_a_part_takes_the_output_of_the_part_before_its_backward_ends_before_that_one_begins(layer_class):
    sink = _Sink()
    moe = layer_class()
    moe.token_dispatcher = _ExchangingDispatcher(sink, [])
    model = nn.Sequential(_Layer(1, mlp=moe))
    ModuleTimers([model], sink).attach()
    _run(model, torch.randn(2, 4, requires_grad=True))
    parts = {"combine_bwd", "experts_bwd", "dispatch_bwd", "router_bwd"}
    spans = sorted((start, end) for name, _, start, end in sink.pushed if name in parts)
    assert len(spans) == len(parts)
    assert all(end < next_start for (_, end), (next_start, _) in zip(spans, spans[1:])), spans


def test_a_method_spec_finds_nothing_on_an_object_without_the_method():
    moe = _RoutedMoELayer()
    moe.token_dispatcher = object()
    counts = ModuleTimers([_Layer(1, mlp=moe)], _Sink()).counts()
    assert counts == {"attn": 1, "moe": 1, "router": 1, "experts": 1, "shared experts": 1}


def test_an_mtp_layer_is_recorded_apart_from_the_decoder_layer_with_the_same_number():
    model = nn.Sequential(_Layer(1), _MtpLayer(1))
    sink = _Sink()
    ModuleTimers([model], sink).attach()
    _run(model, torch.randn(2, 4, requires_grad=True))
    assert sink.names[:4] == [("attn_fwd", 1), ("mlp_fwd", 1), ("attn_fwd", -1), ("mlp_fwd", -1)]


def test_a_block_whose_input_needs_no_gradient_has_no_backward_bracket():
    sink = _Sink()
    model = _Layer(1)
    ModuleTimers([model], sink).attach()
    _run(model, torch.randn(2, 4))
    assert sink.names == [("attn_fwd", 1), ("mlp_fwd", 1), ("mlp_bwd", 1)]


def test_recompute_brackets_the_forward_again_inside_the_backward():
    sink = _Sink()
    model = nn.Sequential(_Layer(1, recompute=True))
    ModuleTimers([model], sink).attach()
    _run(model, torch.randn(2, 4, requires_grad=True))
    # The first forward runs without autograd, so only the recomputed one is followed by backward brackets.
    assert sink.names == [
        ("attn_fwd", 1),
        ("mlp_fwd", 1),
        ("attn_fwd", 1),
        ("mlp_fwd", 1),
        ("mlp_bwd", 1),
        ("attn_bwd", 1),
    ]


def test_nothing_is_recorded_while_the_collector_does_not_record():
    sink = _Sink()
    sink.recording = False
    model = _Layer(1)
    ModuleTimers([model], sink).attach()
    _run(model, torch.randn(2, 4, requires_grad=True))
    assert (sink.pushed, sink.discarded, sink.errors) == ([], [], [])


def test_blocks_get_no_module_hooks_and_detach_restores_the_forward_they_had():
    model = _Layer(1)
    # An instance-level forward, as another wrapper would leave it.
    own_forward = model.mlp.forward
    model.mlp.forward = own_forward
    timers = ModuleTimers([model], _Sink())
    timers.attach()
    assert not (model.mlp._forward_hooks or model.mlp._forward_pre_hooks)
    timers.detach()
    assert "forward" not in model.self_attention.__dict__ and model.mlp.__dict__["forward"] is own_forward


def test_a_block_shared_by_two_layers_is_unwrapped_on_detach():
    shared = _Block()
    timers = ModuleTimers([_Layer(1, mlp=shared), _Layer(2, mlp=shared)], _Sink())
    timers.attach()
    timers.detach()
    assert "forward" not in shared.__dict__


def test_a_block_that_raises_returns_its_start_event_and_the_error_reaches_the_model():
    class _Failing(_Block):
        def forward(self, hidden_states, padding_mask=None):
            raise ValueError("block broke")

    sink = _Sink()
    model = _Layer(1, mlp=_Failing())
    ModuleTimers([model], sink).attach()
    with pytest.raises(ValueError, match="block broke"):
        model(torch.randn(2, 4, requires_grad=True))
    # The last event recorded is the failing block's start.
    assert (sink.names, sink.discarded, sink.errors) == ([("attn_fwd", 1)], [sink.clock], [])


def test_attach_is_idempotent():
    sink = _Sink()
    model = _Layer(1)
    timers = ModuleTimers([model], sink)
    timers.attach()
    timers.attach()
    _run(model, torch.randn(2, 4, requires_grad=True))
    assert len(sink.pushed) == 4
