# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""CPU checks for retention guards and invocation-local checkpoint metadata."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


ROOT = Path(__file__).resolve().parents[4]


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


@pytest.mark.parametrize("enabled", [None, "0"])
def test_disabled_retention_does_not_validate_a_selected_policy(retention, monkeypatch, enabled):
    monkeypatch.setenv("RELAX_QWEN38_QSA_RETENTION_POLICY", "last_stage_l48")
    if enabled is None:
        monkeypatch.delenv("RELAX_QWEN38_RETAIN_QSA_ATTENTION", raising=False)
    else:
        monkeypatch.setenv("RELAX_QWEN38_RETAIN_QSA_ATTENTION", enabled)
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


@pytest.mark.parametrize(("policy", "tp_size"), [("last_stage_l44", 8), ("last_stage_l48", 4)])
def test_last_stage_retention_leaves_earlier_stages_untouched(retention, monkeypatch, policy, tp_size):
    args = arguments()
    args.pipeline_model_parallel_size = 8
    args.tensor_model_parallel_size = tp_size
    monkeypatch.setenv("RELAX_QWEN38_RETAIN_QSA_ATTENTION", "1")
    monkeypatch.setenv("RELAX_QWEN38_QSA_RETENTION_POLICY", policy)
    monkeypatch.setitem(
        sys.modules,
        "megatron.core",
        SimpleNamespace(parallel_state=SimpleNamespace(is_pipeline_last_stage=lambda: False)),
    )
    assert retention.configure(args, None) == []


@pytest.mark.parametrize(
    ("policy", "tp_size", "attention_number"),
    [(None, 8, 44), ("last_stage_l44", 8, 44), ("last_stage_l48", 4, 48)],
)
def test_last_stage_retention_preserves_h800_recipe_layers(retention, monkeypatch, policy, tp_size, attention_number):
    args = arguments()
    args.pipeline_model_parallel_size = 8
    args.tensor_model_parallel_size = tp_size
    args.max_tokens_per_gpu = 131072 if tp_size == 8 else 65536
    monkeypatch.setenv("RELAX_QWEN38_RETAIN_QSA_ATTENTION", "1")
    monkeypatch.delenv("RELAX_QWEN38_QSA_RETENTION_POLICY", raising=False)
    # The 128K launcher provided these settings before the policy was public.
    legacy = {"AIO_TUNE_PP7_RETAIN": "1", "AIO_TUNE_PP7_FULL_LAYER": "47", "AIO_TUNE_PP7_FULL_SCOPE": "mtp"}
    for name, value in legacy.items():
        if policy == "last_stage_l44":
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    if policy is not None:
        monkeypatch.setenv("RELAX_QWEN38_QSA_RETENTION_POLICY", policy)

    class TransformerLayer(SimpleNamespace):
        def _forward_mlp(self):
            raise AssertionError("Model forward is outside the configuration test")

        _relax_native_forward_mlp = _forward_mlp

    class QwenSparseSelfAttention(SimpleNamespace):
        pass

    class BridgeBlock(SimpleNamespace):
        def _checkpointed_forward(self):
            raise AssertionError("Model forward is outside the configuration test")

        _relax_native_checkpointed_forward = _checkpointed_forward

    class HybridStack(SimpleNamespace):
        pass

    def qsa_layer(number):
        return TransformerLayer(
            layer_number=number,
            self_attention=QwenSparseSelfAttention(
                config=SimpleNamespace(qsa_indexer_loss_coeff=0),
                indexer=SimpleNamespace(parameters=lambda: ()),
            ),
            mlp=SimpleNamespace(),
        )

    checkpointed_forward = object()
    megatron_modules = {
        "megatron.core": SimpleNamespace(
            recompute=SimpleNamespace(checkpointed_forward=checkpointed_forward),
            parallel_state=SimpleNamespace(is_pipeline_last_stage=lambda: True),
        ),
        "megatron.core.models.hybrid": SimpleNamespace(
            hybrid_block=SimpleNamespace(HybridStack=HybridStack, checkpointed_forward=checkpointed_forward),
        ),
        "megatron.core.transformer.multi_token_prediction": SimpleNamespace(
            MultiTokenPredictionLayer=type("MTP", (), {})
        ),
        "megatron.core.transformer.transformer_layer": SimpleNamespace(TransformerLayer=TransformerLayer),
        "megatron.bridge.models.qwen_vl.modelling_qwen3_vl.transformer_block": SimpleNamespace(
            Qwen3VLTransformerBlock=BridgeBlock
        ),
    }
    for name, module in megatron_modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    layer_47 = TransformerLayer(layer_number=47)
    decoder_layers = [layer_47, qsa_layer(48)]
    if tp_size == 8:
        decoder_layers.insert(0, qsa_layer(44))
    decoder = BridgeBlock(config=args, layers=decoder_layers)
    mtp = HybridStack(config=args, layers=[qsa_layer(1)], is_mtp_layer=True)
    modules = [("decoder", decoder), ("mtp", mtp)]
    modules.extend((f"decoder.layers.{layer.layer_number}", layer) for layer in decoder_layers)
    modules.append(("mtp.layers.0", mtp.layers[0]))

    manifest = retention.configure(args, [SimpleNamespace(named_modules=lambda: modules)])

    assert {entry["layer_number"]: entry.get("scope", "attention") for entry in manifest} == {
        attention_number: "attention",
        47: "full decoder layer",
        1: "full MTP layer",
    }


def test_full_mtp_retention_uses_native_mlp(retention, monkeypatch):
    original = Mock(return_value="output")
    checkpoint = Mock()
    monkeypatch.setitem(
        sys.modules, "megatron.core", SimpleNamespace(tensor_parallel=SimpleNamespace(checkpoint=checkpoint))
    )
    layer = SimpleNamespace(training=True, _relax_native_forward_mlp=original, _relax_qsa_full_layer_retained=True)
    assert retention.retained_mlp(layer, "h", input_ids="ids") == "output"
    original.assert_called_once_with(
        "h", inference_context=None, padding_mask=None, input_ids="ids", packed_seq_params=None
    )
    checkpoint.assert_not_called()


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
