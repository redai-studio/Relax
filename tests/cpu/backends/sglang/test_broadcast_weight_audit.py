# Copyright (c) 2026 Relax Authors. All Rights Reserved.
import json
import sys
from types import SimpleNamespace

import pytest
import torch

from relax.backends.sglang.broadcast_weight_audit import _apply_control


def control(model, weights):
    # Inventory semantics only. Real native dispatch/TP agreement has its own
    # installed-SGLang regression in test_p2p_native_weight_lease.
    _apply_control(model, json.loads(weights[0][0]))


@pytest.mark.parametrize("target_tp,layers", [(2, 40), (4, 3), (1, 2)])
def test_audit_detects_missing_expert_write_and_preserves_other_weights(tmp_path, monkeypatch, target_tp, layers):
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.runtime_context",
        SimpleNamespace(
            get_parallel=lambda: SimpleNamespace(tp_rank=0),
            get_server_args=lambda: SimpleNamespace(tp_size=target_tp, pp_size=1, quantization=None),
        ),
    )
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    model = torch.nn.Module()
    model.layers = torch.nn.ModuleList()
    for _ in range(layers):
        layer = torch.nn.Module()
        layer.experts = torch.nn.Module()
        layer.experts.w13_weight = torch.nn.Parameter(torch.arange(8).reshape(2, 4).float())
        layer.experts.w2_weight = torch.nn.Parameter(torch.arange(8).reshape(4, 2).float())
        model.layers.append(layer)
    model.dense = torch.nn.Parameter(torch.ones(3))
    model.register_buffer("derived", torch.ones(2))
    saved = {name: tensor.clone() for name, tensor in model.named_parameters()}
    tmp_path.chmod(0o700)

    def command(op, version=1):
        control(model, [(json.dumps(dict(op=op, directory=str(tmp_path), engine=0, version=version)), torch.empty(0))])

    command("reference")
    assert torch.equal(model.dense, saved["dense"])
    assert torch.isnan(model.layers[0].experts.w13_weight).all()
    omitted = f"layers.{layers - 1}.experts.w2_weight"
    with torch.no_grad():
        for name, tensor in model.named_parameters():
            if name != omitted:
                tensor.copy_(saved[name])
    with pytest.raises(RuntimeError, match="mismatches"):
        command("compare")
    receipt = json.loads((tmp_path / "engine-0-tp-0-compare.json").read_text())
    assert receipt["differences"] == ["parameter:" + omitted]
    with torch.no_grad():
        dict(model.named_parameters())[omitted].copy_(saved[omitted])
    command("compare")
    assert json.loads((tmp_path / "engine-0-tp-0-compare.json").read_text())["passed"]
    with pytest.raises(RuntimeError, match="Stale"):
        command("compare")
    command("reference", version=2)
    with torch.no_grad():
        for name, tensor in model.named_parameters():
            tensor.copy_(saved[name])
    model.layers[layers - 1].experts.w2_weight = torch.nn.Parameter(saved[omitted].clone())
    with pytest.raises(RuntimeError, match="storage_unchanged=False"):
        command("compare", version=2)
    receipt = json.loads((tmp_path / "engine-0-tp-0-compare.json").read_text())
    assert not receipt["passed"] and not receipt["storage_unchanged"] and not receipt["differences"]


@pytest.mark.parametrize("target_tp,layers,dense", [(2, 40, 826), (4, 3, 5), (1, 2, 7)])
def test_p2p_deferred_reference_requires_prepare_and_detects_dense_or_derived_omission(
    tmp_path, monkeypatch, target_tp, layers, dense
):
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.runtime_context",
        SimpleNamespace(
            get_parallel=lambda: SimpleNamespace(tp_rank=0),
            get_server_args=lambda: SimpleNamespace(tp_size=target_tp, pp_size=1, quantization=None),
        ),
    )
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    model = torch.nn.Module()
    model.layers = torch.nn.ModuleList()
    for _ in range(layers):
        layer = torch.nn.Module()
        layer.experts = torch.nn.Module()
        for name in ("w13_weight", "w2_weight"):
            layer.experts.register_parameter(name, torch.nn.Parameter(torch.ones(2)))
        model.layers.append(layer)
    model.dense = torch.nn.ParameterList([torch.nn.Parameter(torch.ones(2)) for _ in range(dense)])
    model.norm = torch.nn.Module()
    model.norm.register_buffer("gemma_weight", torch.ones(2))
    saved = {name: tensor.clone() for name, tensor in model.named_parameters()}
    tmp_path.chmod(0o700)

    def command(op, version):
        control(
            model,
            [
                (
                    json.dumps(
                        dict(
                            op=op,
                            directory=str(tmp_path),
                            engine=0,
                            version=version,
                            p2p_directory=str(tmp_path / str(version)),
                        )
                    ),
                    torch.empty(0),
                )
            ],
        )

    previous = None
    for version in (2, 3):
        # Simulate the native loader refreshing a genuinely different snapshot.
        with torch.no_grad():
            for name, tensor in model.named_parameters():
                tensor.copy_(saved[name] + version)
            model.norm.gemma_weight.fill_(version)
        reference = {name: tensor.clone() for name, tensor in model.named_parameters()}
        command("reference_deferred", version)
        assert torch.equal(model.dense[0], reference["dense.0"])
        with pytest.raises(RuntimeError, match="Stale"):
            command("compare", version)
        with pytest.raises(RuntimeError, match="prepared matching"):
            command("poison", version)
        model._relax_p2p_state = dict(active=True, version=version, key="0-0", directory=str(tmp_path / str(version)))
        command("poison", version)
        assert all(torch.isnan(tensor).all() for tensor in model.parameters())
        assert torch.isnan(model.norm.gemma_weight).all()
        with pytest.raises(RuntimeError, match="prepared matching"):
            command("poison", version)
        with torch.no_grad():
            for name, tensor in model.named_parameters():
                if name != f"dense.{dense - 1}":
                    tensor.copy_(reference[name])
        model._relax_p2p_state["active"] = False
        with pytest.raises(RuntimeError, match="mismatches"):
            command("compare", version)
        receipt = json.loads((tmp_path / "engine-0-tp-0-compare.json").read_text())
        assert receipt["differences"] == ["buffer:norm.gemma_weight", f"parameter:dense.{dense - 1}"]
        with torch.no_grad():
            model.dense[dense - 1].copy_(reference[f"dense.{dense - 1}"])
            model.norm.gemma_weight.fill_(version)
        command("compare", version)
        receipt = json.loads((tmp_path / "engine-0-tp-0-compare.json").read_text())
        assert receipt["passed"] and len(receipt["poisoned"]) == layers * 2 + dense
        assert receipt["poisoned_buffers"] == ["norm.gemma_weight"]
        assert receipt["actual"] != previous
        previous = receipt["actual"]
