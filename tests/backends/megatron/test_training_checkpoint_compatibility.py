# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""CPU regressions for training compatibility without Megatron imports.

Load the production definitions with their runtime collaborators substituted;
no CUDA initialization or distributed training is performed.
"""

import ast
import json
import string
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


_ROOT = Path(__file__).resolve().parents[3]


def _load(path, names, namespace):
    source = _ROOT / path
    nodes = [node for node in ast.parse(source.read_text()).body if getattr(node, "name", None) in names]
    future = ast.parse("from __future__ import annotations").body[0]
    exec(compile(ast.Module(body=[future, *nodes], type_ignores=[]), str(source), "exec"), namespace)
    return SimpleNamespace(**namespace)


@pytest.mark.parametrize(
    "enabled,selective,preloaded,usable,optimizer,rejected",
    [
        (True, True, True, True, "sgd", True),
        (True, False, False, False, "sgd", True),
        (True, False, True, False, "sgd", True),
        (True, False, True, True, "sgd", False),
        (False, True, False, False, "sgd", False),
        (True, True, False, False, "adam", False),
    ],
)
def test_sgd_selective_offload_rejected_including_tms_fallback(
    monkeypatch, enabled, selective, preloaded, usable, optimizer, rejected
):
    module = _load(
        "relax/backends/megatron/weight_update/train_offload.py",
        {"MegatronTrainStateOffloader"},
        {
            "torch_memory_saver_preloaded": lambda: preloaded,
            "_TmsOffloadStrategy": Mock(),
            "_SelectiveOffloadStrategy": Mock(),
            "logger": Mock(),
        },
    )
    cls = module.MegatronTrainStateOffloader
    monkeypatch.setattr(cls, "_init_torch_memory_saver", staticmethod(lambda args: usable))
    args = SimpleNamespace(offload_train=enabled, selective_offload=selective, optimizer=optimizer)
    if rejected:
        with pytest.raises(ValueError, match="SGD.*selective"):
            cls([], None, args)
    else:
        offloader = cls([], None, args)
        assert offloader.enabled == enabled
        assert offloader.uses_tms == (enabled and not selective and preloaded and usable)


@pytest.mark.parametrize("reset", [False, True])
def test_reset_sgd_momentum_prevents_zero_gradient_update(reset):
    torch = pytest.importorskip("torch")
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.SGD([parameter], lr=0.1, momentum=0.9, weight_decay=0)
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    before = parameter.detach().clone()
    args = SimpleNamespace(
        overlap_grad_reduce=False,
        overlap_param_gather=False,
        empty_unused_memory_level=0,
        reset_optimizer_states=reset,
        manual_gc=False,
    )

    class StopAfterReset(Exception):
        pass

    def stop(args):
        raise StopAfterReset

    module = _load(
        "relax/backends/megatron/model.py",
        {"train"},
        {
            "get_args": lambda: args,
            "DataIterator": type("DataIterator", (), {}),
            "DDP": type("DDP", (), {}),
            "get_model_config": lambda model: SimpleNamespace(),
            "finalize_model_grads": Mock(),
            "resident_sft_param_overlap_enabled": lambda args: False,
            "should_disable_forward_pre_hook": stop,
            "mpu": Mock(),
            "logger": Mock(),
            "torch": torch,
        },
    )
    chain = SimpleNamespace(chained_optimizers=[SimpleNamespace(optimizer=optimizer)], scale_loss=Mock())
    with pytest.raises(StopAfterReset):
        module.train(0, [SimpleNamespace(train=Mock())], chain, None, [[]], [])
    parameter.grad.zero_()
    optimizer.step()
    assert torch.equal(parameter, before) == reset


@pytest.mark.parametrize("reference_template", [None, "different reference template"])
def test_bf16_export_keeps_actor_artifacts_with_weights_only_reference(monkeypatch, tmp_path, reference_template):
    actor, reference = tmp_path / "actor", tmp_path / "reference"
    actor.mkdir()
    reference.mkdir()
    (actor / "config.json").write_text(json.dumps({"quantization_config": {"quant_method": "fp8"}}))
    (reference / "config.json").write_text("{}")
    (actor / "tokenizer.json").write_text("actor template")
    (actor / "processor.json").write_text("actor processor")
    if reference_template is not None:
        (reference / "tokenizer.json").write_text(reference_template)
        (reference / "processor.json").write_text("reference processor")
    recorded = {}

    class Bridge:
        def __init__(self, source):
            self.source = Path(source)
            self.hf_pretrained = SimpleNamespace(save_artifacts=self.save_artifacts)

        def save_artifacts(self, path, **kwargs):
            for name in ("tokenizer.json", "processor.json"):
                (path / name).write_text((self.source / name).read_text())
            config = json.loads((self.source / "config.json").read_text())
            config.pop("quantization_config", None)
            (path / "config.json").write_text(json.dumps(config))

        def save_hf_pretrained(self, model, path, strict):
            self.hf_pretrained.save_artifacts(path, original_source_path=None, additional_files=None)
            recorded.update(weights=self.source, strict=strict)

    monkeypatch.setitem(
        sys.modules,
        "megatron.bridge",
        SimpleNamespace(AutoBridge=SimpleNamespace(from_hf_pretrained=lambda source, **_: Bridge(source))),
    )
    monkeypatch.setitem(
        sys.modules,
        "relax.utils.hf_export",
        SimpleNamespace(
            reference_expects_mtp=lambda _: False,
            reference_expects_vision=lambda _: False,
            reconcile_hf_export_index=Mock(),
        ),
    )
    module = _load(
        "relax/backends/megatron/model.py",
        {"_hf_export_reference", "_resolve_save_hf_path", "save_hf_model"},
        {
            "Path": Path,
            "json": json,
            "string": string,
            "torch": SimpleNamespace(distributed=SimpleNamespace(is_initialized=lambda: False)),
            "patch_megatron_model": lambda model: nullcontext(),
            "is_lora_enabled": lambda args: False,
            "_invoke_save_hf_post_hook": Mock(),
            "logger": Mock(),
        },
    )
    args = SimpleNamespace(
        hf_checkpoint=str(actor),
        ref_load=str(reference),
        save_hf_dtype="bf16",
        save_hf=str(tmp_path / "out"),
        mtp_num_layers=None,
    )
    module.save_hf_model(args, 0, [])
    output = tmp_path / "out/iter_0000000"
    assert recorded == {"weights": reference, "strict": True}
    assert (output / "tokenizer.json").read_text() == "actor template"
    assert (output / "processor.json").read_text() == "actor processor"
    assert "quantization_config" not in json.loads((output / "config.json").read_text())
    assert "quantization_config" in json.loads((actor / "config.json").read_text())
