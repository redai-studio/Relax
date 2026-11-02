# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""CPU checks for retention guards and invocation-local checkpoint metadata."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


ROOT = Path(__file__).resolve().parents[3]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def retention(monkeypatch):
    env = load("retention_env", ROOT / "relax/utils/env.py")
    monkeypatch.setitem(sys.modules, "relax.utils.env", env)
    return load("retention_patch", ROOT / "relax/models/qwen3_8/qsa_attention_recompute.py")


def arguments():
    return SimpleNamespace(
        recompute_granularity="full",
        recompute_method="uniform",
        recompute_num_layers=1,
        tensor_model_parallel_size=8,
        pipeline_model_parallel_size=4,
        context_parallel_size=1,
        expert_model_parallel_size=8,
        expert_tensor_parallel_size=1,
        bf16=True,
    )


def test_default_disabled_does_not_import_megatron(retention, monkeypatch):
    monkeypatch.delenv("RELAX_QWEN38_RETAIN_QSA_ATTENTION", raising=False)
    assert retention.configure(None, None) == []


@pytest.mark.parametrize(
    "name,value",
    [
        ("context_parallel_size", 2),
        ("recompute_num_layers", 2),
        ("bf16", False),
        ("fp8", "hybrid"),
        ("cpu_offloading", True),
        ("distribute_saved_activations", True),
        ("virtual_pipeline_model_parallel_size", 2),
        ("cuda_graph_impl", "local"),
    ],
)
def test_rejects_unsupported_recompute_configuration(retention, name, value):
    args = arguments()
    retention.validate_args(args)
    setattr(args, name, value)
    with pytest.raises(RuntimeError):
        retention.validate_args(args)


def test_checkpoint_closures_keep_each_microbatch_metadata(retention, monkeypatch):
    original = Mock(return_value="output")
    layer = SimpleNamespace(
        training=True, config=SimpleNamespace(distribute_saved_activations=False), _relax_native_forward_mlp=original
    )
    pending = []

    def checkpoint(fn, distributed, hidden, padding):
        assert distributed is False
        pending.append((fn, hidden, padding))
        return fn(hidden, padding)

    monkeypatch.setitem(
        sys.modules, "megatron.core", SimpleNamespace(tensor_parallel=SimpleNamespace(checkpoint=checkpoint))
    )
    first, second = object(), object()
    assert (
        retention.retained_mlp(layer, "h1", padding_mask="p1", input_ids="ids1", packed_seq_params=first) == "output"
    )
    retention.retained_mlp(layer, "h2", padding_mask="p2", input_ids="ids2", packed_seq_params=second)
    fn, hidden, padding = pending[0]
    fn(hidden, padding)
    assert original.call_args.args == ("h1",)
    assert original.call_args.kwargs == dict(
        inference_context=None, padding_mask="p1", input_ids="ids1", packed_seq_params=first
    )
    assert not hasattr(layer, "_forward_mlp")


def test_eval_bypasses_checkpoint_and_training_inference_is_rejected(retention, monkeypatch):
    original, checkpoint = Mock(), Mock()
    layer = SimpleNamespace(training=False, _relax_native_forward_mlp=original)
    monkeypatch.setitem(
        sys.modules, "megatron.core", SimpleNamespace(tensor_parallel=SimpleNamespace(checkpoint=checkpoint))
    )
    retention.retained_mlp(layer, "h", "ctx", "mask", "ids", "packed")
    original.assert_called_once_with(
        "h", inference_context="ctx", padding_mask="mask", input_ids="ids", packed_seq_params="packed"
    )
    checkpoint.assert_not_called()
    layer.training = True
    with pytest.raises(RuntimeError, match="training-only"):
        retention.retained_mlp(layer, "h", inference_context="ctx")


def test_subclass_native_super_call_is_checkpointed_once(retention, monkeypatch):
    checkpoints = []
    calls = []

    def checkpoint(fn, distributed, *args):
        checkpoints.append(fn)
        return fn(*args)

    monkeypatch.setitem(
        sys.modules, "megatron.core", SimpleNamespace(tensor_parallel=SimpleNamespace(checkpoint=checkpoint))
    )

    class Base:
        training = True
        config = SimpleNamespace(distribute_saved_activations=False)

        def native(self, hidden, inference_context=None, padding_mask=None, input_ids=None, packed_seq_params=None):
            calls.append((hidden, input_ids, packed_seq_params))
            return hidden

        def forward(self, hidden, **kwargs):
            return retention.retained_mlp(self, hidden, native_forward=Base.native, **kwargs)

    class Child(Base):
        def native(self, hidden, **kwargs):
            return super().forward(hidden, **kwargs)

        def forward(self, hidden, **kwargs):
            return retention.retained_mlp(self, hidden, native_forward=Child.native, **kwargs)

    owner = Child()
    assert owner.forward("hidden", input_ids="tokens", packed_seq_params="packed") == "hidden"
    assert len(checkpoints) == 1
    checkpoints[0]("hidden-replay", None)
    assert calls == [("hidden", "tokens", "packed"), ("hidden-replay", "tokens", "packed")]
    assert len(checkpoints) == 1
    assert retention._active_retained_mlp.get() == frozenset()


def test_split_recompute_preserves_token_metadata_and_hybrid_interfaces():
    import ast
    from contextlib import nullcontext

    path = ROOT / "relax/models/qwen3_8/qsa_split_recompute.py"
    function = next(node for node in ast.parse(path.read_text()).body if isinstance(node, ast.FunctionDef))
    calls = []
    checkpoint_calls = []

    class TransformerLayer:
        _relax_qsa_attention_retained = True

        def __call__(self, **kwargs):
            calls.append(("transformer", kwargs))
            return kwargs["hidden_states"], kwargs["context"]

    class HybridLayer:
        supports_hybrid_recompute_kwargs = True

        def __call__(self, **kwargs):
            calls.append(("hybrid", kwargs))
            return kwargs["hidden_states"], None

    class NarrowLayer:
        def __call__(self, **kwargs):
            calls.append(("narrow", kwargs))
            return kwargs["hidden_states"]

    def checkpoint(fn, distributed, *args):
        checkpoint_calls.append(fn)
        return fn(*args)

    scope = dict(
        TransformerLayer=TransformerLayer,
        nullcontext=nullcontext,
        tensor_parallel=SimpleNamespace(checkpoint=checkpoint),
    )
    module = ast.Module(body=ast.parse("from __future__ import annotations").body + [function], type_ignores=[])
    exec(compile(module, str(path), "exec"), scope)
    stack = SimpleNamespace(
        layers=[TransformerLayer(), HybridLayer(), NarrowLayer()],
        num_layers_per_pipeline_rank=3,
        config=SimpleNamespace(
            fp8=False,
            fp4=False,
            recompute_method="uniform",
            recompute_num_layers=1,
            distribute_saved_activations=False,
        ),
    )
    assert (
        scope[function.name](
            stack,
            "hidden",
            "mask",
            None,
            None,
            None,
            None,
            "packed",
            False,
            padding_mask="padding",
            input_ids="tokens",
        )
        == "hidden"
    )
    assert len(checkpoint_calls) == 2
    assert calls[0][1]["input_ids"] == calls[1][1]["input_ids"] == "tokens"
    assert calls[0][1]["padding_mask"] == calls[1][1]["padding_mask"] == "padding"
    assert "context" not in calls[1][1]
    assert "input_ids" not in calls[2][1] and "padding_mask" not in calls[2][1]
    assert all(values["packed_seq_params"] == "packed" for _, values in calls)
