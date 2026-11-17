# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Real CPU checkpoint autograd, tape and replay cursor state machines.

Only the external CUDA selector and unused MCore attention imports are
isolated; the dispatcher, complete tape module, lazy source and manager execute
normally.
"""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from relax.distributed.replay_manager.backend.megatron import IndexerReplay
from relax.distributed.replay_manager.consumer import ConsumerRecord, ConsumerSource


class Source(ConsumerSource):
    def __init__(self, value):
        self.value = value
        self.reads = 0
        self.released = False

    def read(self):
        self.reads += 1
        ids = torch.full((2, 2), self.value, dtype=torch.int32)
        return ids, torch.ones(2, dtype=torch.bool)

    def release(self):
        self.released = True


@pytest.fixture
def modules(monkeypatch):
    root = Path(__file__).resolve().parents[4] / "relax/models/qwen3_8"
    package_name = "relax.models.qwen3_8"
    package = ModuleType(package_name)
    package.__path__ = [str(root)]
    monkeypatch.setitem(sys.modules, package_name, package)
    model_package = ModuleType("relax.models")
    model_package.__path__ = [str(root.parent)]
    monkeypatch.setitem(sys.modules, model_package.__name__, model_package)
    externals = {
        "megatron.core.ssm.mamba_context_parallel": dict(
            _thd_partitioned_indices=None, _undo_attention_load_balancing=None
        ),
        "megatron.core.tensor_parallel": dict(random=SimpleNamespace()),
        "megatron.core.tensor_parallel.mappings": dict(gather_from_sequence_parallel_region=None),
        "megatron.core.transformer.experimental_attention_variant": dict(qsa=SimpleNamespace()),
        package_name + ".qsa_block_mask": {},
    }
    for name, attributes in externals.items():
        module = ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
    calls = []

    def select(owner, hidden, *, replay_data):
        calls.append((owner.layer_number, IndexerReplay.get_stage(), replay_data[0].clone()))
        return SimpleNamespace(selected_ids=replay_data[0], bounds=(0, 1, 2))

    selector = ModuleType(package_name + ".ops.qsa_mixed_ops.selector")
    selector.forward = select
    monkeypatch.setitem(sys.modules, selector.__name__, selector)
    loaded = []
    for name in ("qsa_dispatch", "qsa_cp"):
        spec = importlib.util.spec_from_file_location(package_name + "." + name, root / (name + ".py"))
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, spec.name, module)
        setattr(package, name, module)
        spec.loader.exec_module(module)
        loaded.append(module)
    monkeypatch.setenv("RELAX_QWEN38_QSA_RECOMPUTE_REPLAY", "1")
    IndexerReplay.clear()
    yield *loaded, calls
    IndexerReplay.clear()


def bind(layers, microbatches):
    IndexerReplay.begin_step(list(range(1, layers + 1)))
    owners = [
        SimpleNamespace(layer_number=i, config=SimpleNamespace(relax_use_indexer_replay=True))
        for i in range(1, layers + 1)
    ]
    sources = []
    for owner in owners:
        layer = IndexerReplay._layers[owner.layer_number]
        for microbatch in range(microbatches):
            source = Source(10 * microbatch + owner.layer_number)
            layer.manager.bind(layer.key, ConsumerRecord("indexer", microbatch, source))
            sources.append(source)
    return owners, sources


@pytest.mark.parametrize("cached", [False, True])
def test_qsa_checkpoint_consumes_fifo_microbatches_without_reloading(modules, monkeypatch, cached):
    dispatch, cp, calls = modules
    monkeypatch.setenv("RELAX_QWEN38_QSA_RECOMPUTE_REPLAY", str(int(cached)))
    owners, sources = bind(3, 4)

    def block(value):
        for owner in owners:
            selection = dispatch.indexer_forward(owner, value)
            assert selection.bounds == (0, 1, 2)
            value = value * selection.selected_ids.float().sum()
        return value

    IndexerReplay.set_stage("replay_forward")
    with torch.no_grad():
        for _ in range(4):
            block(torch.tensor(1.0))
    IndexerReplay.reset_forward()
    IndexerReplay.set_stage("replay_backward")
    inputs, outputs = [], []
    for microbatch in range(4):
        value = torch.tensor(float(microbatch + 1), requires_grad=True)
        inputs.append(value)
        with IndexerReplay.forward_stage():
            outputs.append(checkpoint(cp._with_replay(block), value, use_reentrant=True))
    for microbatch, output in enumerate(outputs):
        output.backward()
        scale = 1
        for layer in range(1, 4):
            scale *= 4 * (10 * microbatch + layer)
        torch.testing.assert_close(inputs[microbatch].grad, torch.tensor(float(scale)))
    assert all(source.reads == (2 if cached else 3) for source in sources)
    assert sum(stage == "replay_backward" for _, stage, _ in calls) == (0 if cached else 12)
    assert cp._active_tape.get() is None
    IndexerReplay.finish_step(expect_backward=True)
    assert all(source.released for source in sources)


@pytest.mark.parametrize("failure", ["microbatch", "step", "phase", "overrun"])
def test_qsa_checkpoint_rejects_invalid_cached_consumption(modules, failure):
    dispatch, cp, _ = modules
    owners, sources = bind(1, 2 if failure == "microbatch" else 1)
    owner = owners[0]

    def block(value):
        return value * dispatch.indexer_forward(owner, value).selected_ids.float().sum()

    IndexerReplay.set_stage("replay_backward")
    outputs = []
    for i in range(len(sources)):
        with IndexerReplay.forward_stage():
            outputs.append(
                checkpoint(cp._with_replay(block), torch.tensor(float(i + 1), requires_grad=True), use_reentrant=True)
            )
    if failure == "step":
        bind(1, 1)
        IndexerReplay.set_stage("replay_backward")
        message = "different replay step or layer"
    elif failure == "phase":
        IndexerReplay.set_stage("replay_forward")
        message = "requires backward"
    else:
        message = "microbatch mismatch"
    if failure == "overrun":
        outputs[0].backward(retain_graph=True)
    with pytest.raises(RuntimeError, match=message):
        outputs[-1].backward()
    assert all(source.reads == 1 for source in sources)
    assert cp._active_tape.get() is None


def test_qsa_checkpoint_fallthrough_does_not_consume_rollout(modules, monkeypatch):
    dispatch, cp, calls = modules
    owners, sources = bind(1, 1)
    monkeypatch.setattr(dispatch, "selector_forward", lambda *args, **kwargs: "native")
    monkeypatch.setenv("RELAX_QWEN38_QSA_COMMUNITY_BACKEND", "native")
    run = cp._with_replay(lambda: dispatch.indexer_forward(owners[0], torch.tensor(1.0)))
    assert run() == run() == "native"
    assert not calls and sources[0].reads == 0
