# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Module brackets for the ``modules`` level: forward and backward of the
blocks that a list of ``ModuleSpec`` names inside every transformer layer
(``megatron_adapter.MODULE_SPECS``: the attention, the MLP / MoE block and the
parts of a MoE layer), recorded on the sampled step only.

The timing is attached when a sampled step at that level starts and removed
when it ends, so no other step pays for it. Each block's forward, or the
method its spec names (such as the token exchange of a MoE layer), is shadowed
by a wrapper that brackets every call. Backward uses gradient hooks on the
block's output (its gradient arriving opens the bracket) and on its input
(closes it), or on all of them for a call with several differentiable inputs
and outputs, which also holds under activation recompute, where the block's
forward runs again inside the backward. Which micro-batch a bracket belongs
to, and whether a forward was a recompute, is read off the trace by
containment (see ``analysis.trace``).

Timing the blocks of another model means adding specs to the adapter's list.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence

from relax.utils.straggler.core.stats import NUM_FIELDS
from relax.utils.straggler.core.timers import EventSink


@dataclass(frozen=True)
class ModuleSpec:
    """One kind of block: the object at attribute ``path`` of a transformer
    layer, named ``name`` in the trace.

    ``method`` times that method of the object instead of its forward, for a
    part that is not a module; ``when`` requires the object's class name to
    contain it. The first spec that matches a (path, method) pair wins.

    The backward is bracketed from the gradient of the first output to that of
    the first input. With ``all_tensors`` every tensor argument and output
    counts, the first on each side to get its gradient, for a call whose
    tensors take separate paths through it: the token exchange of a MoE layer
    sends the tokens and their routing probabilities through two all-to-alls,
    the probabilities' first in the backward, and the experts take both.
    Autograd runs the most recently created of the nodes it can run, so a
    call's own nodes run before those that produced its inputs, and the first
    input to get its gradient ends the call's backward.
    """

    path: str
    name: str
    method: str | None = None
    when: str | None = None
    all_tensors: bool = False


def module_names(specs: Sequence[ModuleSpec]) -> tuple[str, ...]:
    """Block names of ``specs`` in order.

    The forward and backward brackets of block ``index`` are record segments
    ``NUM_FIELDS + 2 * index`` and ``NUM_FIELDS + 2 * index + 1``.
    """
    return tuple(dict.fromkeys(spec.name for spec in specs))


def module_segment(names: Sequence[str], name: str, backward: bool) -> int:
    """The record segment of a block's forward or backward bracket."""
    return NUM_FIELDS + 2 * names.index(name) + backward


def module_block(names: Sequence[str], segment: int) -> tuple[str, bool]:
    """Block name of a module record segment and whether it is a backward."""
    index, backward = divmod(segment - NUM_FIELDS, 2)
    return (names[index] if index < len(names) else f"block {index}"), bool(backward)


def _resolve(obj: Any, path: str) -> Any:
    for part in path.split("."):
        obj = getattr(obj, part, None)
        if obj is None:
            return None
    return obj


class ModuleTimers:
    """The ``specs`` blocks of the transformer layers found in
    ``model_chunks``, bracketed into ``sink`` (the collector) while attached.

    ``layer_of`` gives a module's layer number, ``None`` for a module that is
    not a transformer layer (the adapter negates the layers inside an MTP
    layer, which number from 1 again); blocks whose class is in
    ``placeholders`` are skipped.
    """

    def __init__(
        self,
        model_chunks: Iterable[Any],
        sink: EventSink,
        specs: Sequence[ModuleSpec],
        layer_of: Callable[[Any], int | None],
        placeholders: frozenset[str] = frozenset(),
    ) -> None:
        self._sink = sink
        self.names = module_names(specs)
        # (object, method, block name, layer, all_tensors)
        self._blocks: list[tuple[Any, str, str, int, bool]] = []
        for chunk in model_chunks:
            for module in chunk.modules():
                layer = layer_of(module)
                if layer is None:
                    continue
                taken = set()
                for spec in specs:
                    key = (spec.path, spec.method)
                    block = None if key in taken else _resolve(module, spec.path)
                    if block is None or type(block).__name__ in placeholders:
                        continue
                    if spec.method is not None and not callable(getattr(block, spec.method, None)):
                        continue
                    if spec.when is None or spec.when in type(block).__name__:
                        taken.add(key)
                        self._blocks.append((block, spec.method or "forward", spec.name, layer, spec.all_tensors))
        self._wrappers: list[_Wrapper] = []

    def counts(self) -> Counter:
        """Blocks found, by name."""
        return Counter(name for _, _, name, _, _ in self._blocks)

    def attach(self) -> None:
        if self._wrappers:
            return
        for block, method, name, layer, all_tensors in self._blocks:
            segments = module_segment(self.names, name, False), module_segment(self.names, name, True)
            self._wrappers.append(_Wrapper(block, method, self._sink, segments, layer, all_tensors))

    def detach(self) -> None:
        wrappers, self._wrappers = self._wrappers, []
        # Last in, first out: a block shared by two layers is wrapped once per layer.
        for wrapper in reversed(wrappers):
            wrapper.remove()


class _Wrapper:
    """Brackets the calls of ``owner.<method>`` into ``sink`` by shadowing the
    method with an instance attribute until ``remove``.

    Forward hooks would take every call of a module off PyTorch's fast path,
    and a launch-bound model shows that CPU time in its forwards. The block's
    own exceptions pass through; the profiler's go to ``sink.on_error``.
    """

    __slots__ = ("_owner", "_method", "_saved")

    def __init__(
        self, owner: Any, method: str, sink: EventSink, segments: tuple[int, int], layer: int, all_tensors: bool
    ) -> None:
        self._owner = owner
        self._method = method
        self._saved = owner.__dict__.get(method)
        original = getattr(owner, method)
        fwd, bwd = segments

        def timed(*args: Any, **kwargs: Any) -> Any:
            start = _open(sink)
            try:
                output = original(*args, **kwargs)
            except BaseException:
                if start is not None:
                    sink.discard_event(start)
                raise
            if start is not None and _close(sink, fwd, layer, start):
                if all_tensors:
                    sources, outputs = _grad_tensors((*args, *kwargs.values())), _grad_tensors(output)
                else:
                    source = args[0] if args else kwargs.get("hidden_states")
                    sources, outputs = _grad_tensors(source, first_only=True), _grad_tensors(output, first_only=True)
                _bracket_backward(sink, bwd, layer, sources, outputs)
            return output

        setattr(owner, method, timed)

    def remove(self) -> None:
        if self._saved is None:
            delattr(self._owner, self._method)
        else:
            setattr(self._owner, self._method, self._saved)


def _grad_tensors(value: Any, first_only: bool = False) -> list[Any]:
    """The tensors that take part in the backward among ``value`` or the
    elements of a tuple / list ``value`` (its first only with
    ``first_only``)."""
    if not isinstance(value, (tuple, list)):
        value = (value,)
    elif first_only:
        value = value[:1]
    return [item for item in value if getattr(item, "requires_grad", False)]


def _bracket_backward(sink: EventSink, segment: int, layer: int, sources: list[Any], outputs: list[Any]) -> None:
    """Bracket the backward of one call, from the first gradient to reach one
    of its ``outputs`` to the first to reach one of its ``sources``, when both
    take part in it."""
    if not outputs or not sources:
        return
    try:
        bracket = _BackwardBracket(sink, segment, layer)
        for output in outputs:
            _on_gradient(output, bracket.open)
        # A tensor's hooks run before its grad_fn's prehooks, so where a block's input comes straight from the
        # block before it, its backward ends before that block's begins.
        for source in sources:
            source.register_hook(bracket.close)
    except Exception as exc:
        sink.on_error("modules", exc)


def _on_gradient(tensor: Any, hook: Any) -> None:
    """Call ``hook`` when the gradient of ``tensor`` arrives.

    A non-leaf tensor's hooks run just before its ``grad_fn`` does, so the hook
    goes on that node, where it is cheaper to register.
    """
    if tensor.grad_fn is None:
        tensor.register_hook(hook)
    else:
        tensor.grad_fn.register_prehook(hook)


def _open(sink: EventSink) -> Any:
    """The start event of a bracket; ``None`` when none could be recorded."""
    try:
        return sink.record_event()
    except Exception as exc:
        sink.on_error("modules", exc)
        return None


def _close(sink: EventSink, segment: int, layer: int, start: Any) -> bool:
    """End the bracket opened by ``start`` and push it; ``False`` when it was
    dropped instead."""
    try:
        end = sink.record_event()
        if end is not None:
            sink.push(segment, layer, start, end, 0.0)
            return True
        sink.discard_event(start)
    except Exception as exc:
        sink.discard_event(start)
        sink.on_error("modules", exc)
    return False


class _BackwardBracket:
    """One block's backward for one forward call: opened by the first gradient
    to reach its outputs, closed by the first to reach its inputs; the ones
    after them are ignored.

    The hooks return ``None``, so the gradients pass through unchanged.
    """

    __slots__ = ("_sink", "_segment", "_layer", "_start", "_closed")

    def __init__(self, sink: EventSink, segment: int, layer: int) -> None:
        self._sink = sink
        self._segment = segment
        self._layer = layer
        self._start: Any = None
        self._closed = False

    def open(self, grad: Any) -> None:
        if self._start is None and not self._closed:
            self._start = _open(self._sink)

    def close(self, grad: Any) -> None:
        if self._closed:
            return
        self._closed = True
        start, self._start = self._start, None
        if start is not None:
            _close(self._sink, self._segment, self._layer, start)
