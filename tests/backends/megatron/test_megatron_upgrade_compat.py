# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""CPU regressions for native GDN and the upgraded GPT MTP boundary.

Set RELAX_MEGATRON_PATCHED_SOURCE to an isolated patched Megatron source root
or its gpt_model.py file to enable GPT cases. No Megatron packages are
imported.
"""

from __future__ import annotations

import ast
import os
import sys
import types
from pathlib import Path

import pytest


_MODEL_SOURCE = Path(__file__).resolve().parents[3] / "relax/backends/megatron/model.py"


def _load_functions(source: Path, names: set[str], namespace: dict, class_name: str | None = None) -> dict:
    tree = ast.parse(source.read_text())
    body = tree.body
    if class_name is not None:
        body = next(node.body for node in body if isinstance(node, ast.ClassDef) and node.name == class_name)
    functions = [node for node in body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in functions} == names
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"), namespace)
    return namespace


def _install_module(monkeypatch, name: str, **attributes) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__dict__.update(attributes)
    monkeypatch.setitem(sys.modules, name, module)
    return module


@pytest.fixture
def gdn_types(monkeypatch):
    calls = []

    class GatedDeltaNet:
        def forward(self, hidden, attention_mask, inference_context, packed_seq_params, *args, **kwargs):
            calls.append((self.cp_size, self.pg_collection.cp))
            if kwargs.get("fail"):
                raise RuntimeError("forward failure")
            return hidden

    class TransformerConfig:
        def __post_init__(self):
            calls.append((self.linear_num_key_heads, self.linear_num_value_heads))
            divisor = self.tensor_model_parallel_size * self.context_parallel_size
            assert self.linear_num_key_heads % divisor == 0
            assert self.linear_num_value_heads % divisor == 0
            if getattr(self, "fail", False):
                raise RuntimeError("validation failure")

    _install_module(
        monkeypatch,
        "megatron.core.ssm.gated_delta_net",
        GatedDeltaNet=GatedDeltaNet,
        causal_conv1d=None,
    )
    _install_module(
        monkeypatch,
        "megatron.core.transformer.transformer_config",
        TransformerConfig=TransformerConfig,
    )
    namespace = _load_functions(
        _MODEL_SOURCE,
        {"_patch_gdn_for_dynamic_cp", "_resolve_gdn_cp"},
        {"__package__": "_relax_upgrade_test"},
    )
    return GatedDeltaNet, TransformerConfig, calls, namespace


@pytest.mark.parametrize("mode", ["headwise", "chunkwise"])
def test_megatron_upgrade_native_gdn_preserves_config_and_forward(gdn_types, mode):
    gdn, config, _, namespace = gdn_types
    gdn._prepare_input_for_gated_delta_rule = lambda self: None
    original_forward, original_post_init = gdn.forward, config.__post_init__

    namespace["_patch_gdn_for_dynamic_cp"](mode)

    assert gdn.forward is original_forward
    assert config.__post_init__ is original_post_init
    assert not getattr(gdn, "_dcp_patched", False)
    assert not getattr(config, "_gdn_cp_relaxed", False)


@pytest.fixture
def gpt_postprocess():
    source_root = os.environ.get("RELAX_MEGATRON_PATCHED_SOURCE")
    if not source_root:
        pytest.skip("RELAX_MEGATRON_PATCHED_SOURCE must point to isolated patched Megatron source")
    source = Path(source_root)
    if source.is_dir():
        source = source / "megatron/core/models/gpt/gpt_model.py"
    assert source.is_file(), f"Patched GPT source does not exist: {source}"
    import torch

    calls = {"roll": [], "prepare": [], "prefetch": [], "mtp": [], "loss": [], "output": []}
    state = {"inference": False}
    cp_group, prefetched_context = object(), object()

    def roll(tensors, *, shifts, dims, **kwargs):
        calls["roll"].append((tensors, kwargs))
        result = [torch.roll(tensor, shifts=shifts, dims=dims) for tensor in tensors]
        for tensor in result:
            tensor.select(dims, shifts).zero_()
        return result

    def prefetch(**kwargs):
        calls["prefetch"].append(kwargs)
        return prefetched_context

    def prepare(**kwargs):
        calls["prepare"].append(kwargs)
        return types.SimpleNamespace(prefetch_halos=prefetch)

    def mtp(**kwargs):
        calls["mtp"].append(kwargs)
        return kwargs["hidden_states"] + 10

    def loss(**kwargs):
        calls["loss"].append(kwargs)
        return kwargs["hidden_states"] + 1

    def output(**kwargs):
        calls["output"].append(kwargs)
        return kwargs["hidden_states"]

    namespace = _load_functions(
        source,
        {"_postprocess"},
        {
            "InferenceMode": types.SimpleNamespace(is_active=lambda: state["inference"]),
            "roll_tensor": roll,
            "resolve_cp_group": lambda static_group, packed: cp_group,
            "prepare_mtp_sequence_roll_context": prepare,
            "process_mtp_loss": loss,
        },
        class_name="GPTModel",
    )
    model = types.SimpleNamespace(
        config=types.SimpleNamespace(mtp_num_layers=2, use_mup=False),
        post_process=True,
        share_embeddings_and_output_weights=False,
        pg_collection=types.SimpleNamespace(cp=object()),
        tp_group=object(),
        training=True,
        output_layer=object(),
        embedding=types.SimpleNamespace(add_position_embedding=True),
        compute_language_model_loss=object(),
        _scale_logits=lambda value: value,
        mtp=mtp,
    )
    kwargs = {
        "hidden_states": torch.arange(8, dtype=torch.float32).reshape(4, 1, 2),
        "input_ids": torch.tensor([[1, 2, 3, 4]]),
        "position_ids": torch.tensor([[0, 1, 2, 3]]),
        "labels": None,
        "loss_mask": torch.tensor([[0, 1, 1, 0]]),
        "rotary_pos_emb": None,
        "rotary_pos_cos": None,
        "rotary_pos_sin": None,
        "mtp_in_postprocess": True,
        "packed_seq_params": object(),
        "output_processor": output,
        "output_processor_context": object(),
    }
    return types.SimpleNamespace(
        fn=namespace["_postprocess"],
        model=model,
        kwargs=kwargs,
        calls=calls,
        state=state,
        cp_group=cp_group,
        prefetched_context=prefetched_context,
        torch=torch,
    )


@pytest.mark.parametrize("preshifted", [False, True])
def test_megatron_upgrade_gpt_mtp_targets_and_mask_stay_aligned(gpt_postprocess, preshifted):
    fixture = gpt_postprocess
    expected_labels = fixture.torch.tensor([[2, 3, 4, 0]])
    mtp_labels = expected_labels if preshifted else fixture.kwargs["input_ids"]
    fixture.kwargs["mtp_kwargs"] = {"mtp_labels": mtp_labels, "labels_are_shifted": preshifted}
    mask = fixture.kwargs["loss_mask"]
    result = fixture.fn(fixture.model, **fixture.kwargs)

    assert len(fixture.calls["roll"]) == (0 if preshifted else 1)
    if not preshifted:
        tensors, roll_kwargs = fixture.calls["roll"][0]
        assert len(tensors) == 1
        assert tensors[0] is mtp_labels
        assert roll_kwargs["cp_group"] is fixture.cp_group
        assert roll_kwargs["packed_seq_params"] is fixture.kwargs["packed_seq_params"]
    assert len(fixture.calls["prepare"]) == len(fixture.calls["prefetch"]) == 1
    assert len(fixture.calls["mtp"]) == len(fixture.calls["loss"]) == len(fixture.calls["output"]) == 1
    prepared = fixture.calls["prepare"][0]
    prefetch = fixture.calls["prefetch"][0]
    processed = fixture.calls["loss"][0]
    output = fixture.calls["output"][0]
    assert fixture.torch.equal(processed["labels"], expected_labels)
    assert prepared["tensor"] is processed["labels"]
    assert prefetch["labels"] is processed["labels"]
    assert prefetch["loss_mask"] is processed["loss_mask"] is output["loss_mask"] is mask
    assert prefetch["width"] == fixture.model.config.mtp_num_layers + 1
    assert prefetch["input_ids"] is fixture.kwargs["input_ids"]
    assert prefetch["position_ids"] is fixture.kwargs["position_ids"]
    assert fixture.calls["mtp"][0]["sequence_roll_context"] is fixture.prefetched_context
    assert processed["sequence_roll_context"] is fixture.prefetched_context
    assert processed["cp_group"] is fixture.cp_group
    assert output["labels"] is None
    assert output["context"] is fixture.kwargs["output_processor_context"]
    assert fixture.torch.equal(result, fixture.kwargs["hidden_states"] + 11)
    assert fixture.torch.equal(mask, fixture.torch.tensor([[0, 1, 1, 0]]))


@pytest.mark.parametrize("inference", [False, True])
def test_megatron_upgrade_gpt_without_mtp_targets_skips_auxiliary_work(gpt_postprocess, inference):
    fixture = gpt_postprocess
    fixture.state["inference"] = inference
    fixture.kwargs["runtime_gather_output"] = inference
    fixture.model.training = not inference
    result = fixture.fn(fixture.model, **fixture.kwargs)

    for operation in ("roll", "prepare", "prefetch", "mtp", "loss"):
        assert fixture.calls[operation] == []
    assert len(fixture.calls["output"]) == 1
    assert result is fixture.kwargs["hidden_states"]
    if inference:
        assert fixture.model._decoder_hidden_states_cache is fixture.kwargs["hidden_states"]
    else:
        assert not hasattr(fixture.model, "_decoder_hidden_states_cache")
