# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Frozen-reference checksum, optimizer and sidecar tests."""

import ast
import json
import os
import types
from argparse import Namespace
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch

from relax.backends.megatron.reference_integrity import (
    DPOReferenceIdentity,
    canonical_tensor_sha256,
    read_reference_identity,
    reference_identity_path,
    write_reference_identity,
)
from relax.utils.model_source import is_model_source_alias
from relax.utils.training import tensor_backper


def test_megatron_resume_detection_ignores_fresh_output_directory(tmp_path):
    pytest.importorskip("megatron.training.checkpointing")
    from relax.backends.megatron.checkpoint import is_megatron_checkpoint

    output = tmp_path / "run"
    output.mkdir()
    (output / "transformer_config.json").write_text("{}", encoding="utf-8")
    assert not is_megatron_checkpoint(output)
    (output / "latest_checkpointed_iteration.txt").write_text("1", encoding="utf-8")
    assert is_megatron_checkpoint(output)
    assert is_megatron_checkpoint(tmp_path / "iter_0000001")


def test_canonical_tensor_digest_is_order_stable_and_byte_sensitive():
    first = canonical_tensor_sha256([("b", torch.tensor([2.0])), ("a", torch.tensor([1.0]))])
    reordered = canonical_tensor_sha256([("a", torch.tensor([1.0])), ("b", torch.tensor([2.0]))])
    changed = canonical_tensor_sha256([("a", torch.tensor([1.0])), ("b", torch.tensor([3.0]))])
    assert first == reordered
    assert first != changed
    assert first != canonical_tensor_sha256([("a", torch.tensor([1], dtype=torch.int64)), ("b", torch.tensor([2.0]))])


def test_reference_identity_sidecar_is_required_and_rejects_schema_damage(tmp_path):
    path = tmp_path / "relax_dpo_reference.json"
    with pytest.raises(FileNotFoundError):
        read_reference_identity(path)
    identity = DPOReferenceIdentity(2, "a" * 64)
    write_reference_identity(path, identity)
    assert read_reference_identity(path) == identity
    payload = identity.to_dict()
    payload["schema_version"] = 99
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported"):
        read_reference_identity(path)


def test_reference_identity_reads_legacy_source_fields_and_upgrades_on_write(tmp_path):
    path = tmp_path / "relax_dpo_reference.json"
    payload = {
        "schema_version": 1,
        "repository": "repo",
        "revision": "revision",
        "loader_mode": "loader",
        "parameter_sha256": "a" * 64,
        "probe_sha256": "b" * 64,
        "probe_manifest": {"tokens": [[1, 2], [1, 3]], "loss_masks": [[0, 1], [0, 1]]},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    identity = read_reference_identity(path)
    assert identity == DPOReferenceIdentity(1, "a" * 64)
    write_reference_identity(path, identity)
    assert json.loads(path.read_text()) == {"schema_version": 2, "parameter_sha256": "a" * 64}


@pytest.fixture
def reference_actor_methods():
    """Load the real lifecycle methods without importing the GPU actor
    stack."""
    checkpoint_module = pytest.importorskip("relax.backends.megatron.checkpoint")
    source = Path(__file__).resolve().parents[3] / "relax/backends/megatron/actor.py"
    tree = ast.parse(source.read_text())
    actor = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "MegatronTrainRayActor")
    method_names = {
        "_switch_model",
        "_is_standard_dpo",
        "_assert_dpo_reference_identity",
        "_rebuild_dpo_reference",
        "save_model",
    }
    methods = [node for node in actor.body if isinstance(node, ast.FunctionDef) and node.name in method_names]
    for method in methods:
        method.decorator_list = []
    namespace = {
        "DPOReferenceIdentity": DPOReferenceIdentity,
        "os": os,
        "is_model_source_alias": is_model_source_alias,
        "is_megatron_checkpoint": checkpoint_module.is_megatron_checkpoint,
        "_checkpoint_iteration_dir": checkpoint_module._checkpoint_iteration_dir,
        "canonical_tensor_sha256": canonical_tensor_sha256,
        "reference_identity_path": reference_identity_path,
        "write_reference_identity": write_reference_identity,
        "device_utils": types.SimpleNamespace(maybe_backend_process_on_model_switch=lambda: None),
    }
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(source), "exec"), namespace)
    return type("ReferenceActor", (), {name: namespace[name] for name in method_names}), namespace


@pytest.mark.parametrize(
    ("source", "ref_ckpt_step", "outcome", "schema_version"),
    [
        ("hf", None, "success", None),
        ("hf", 7, "success", 1),
        ("hf", None, "success", 2),
        ("hf", 7, "loader_failure", None),
        ("hf", None, "identity_mismatch", 1),
        ("hf", 7, "identity_mismatch", 2),
        ("megatron", None, "success", None),
        ("megatron", 7, "success", 1),
        ("megatron", 0, "success", 2),
        ("iteration", None, "success", None),
        ("megatron", None, "loader_failure", None),
        ("megatron", 7, "identity_mismatch", 1),
    ],
)
def test_reference_rebuild_preserves_actor_and_optimizer(
    monkeypatch, tmp_path, reference_actor_methods, source, ref_ckpt_step, outcome, schema_version
):
    actor_type, namespace = reference_actor_methods
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.Adam([parameter], lr=0.1)
    parameter.square().sum().backward()
    optimizer.step()
    actor_value = parameter.detach().clone()
    monkeypatch.setattr(tensor_backper, "_PIN_MEMORY", False)
    monkeypatch.setattr(tensor_backper, "_NON_BLOCKING", False)
    monkeypatch.setattr(tensor_backper.device_module, "synchronize", lambda: None)

    reference_root = tmp_path / "reference"
    reference_root.mkdir()
    reference_path = reference_root
    expected_step = ref_ckpt_step
    if source == "hf":
        (reference_root / "config.json").write_text("{}", encoding="utf-8")
    else:
        latest = "0" if ref_ckpt_step == 0 else "3"
        (reference_root / "latest_checkpointed_iteration.txt").write_text(latest, encoding="utf-8")
        expected_step = 7 if source == "iteration" else 3 if ref_ckpt_step is None else ref_ckpt_step
        iteration_dir = reference_root / f"iter_{expected_step:07d}"
        iteration_dir.mkdir()
        (iteration_dir / "common.pt").write_bytes(b"reference checkpoint")
        if source == "iteration":
            reference_path = iteration_dir
    instance = actor_type()
    instance.args = Namespace(
        load=str(tmp_path / "actor-checkpoint"),
        hf_checkpoint=str(reference_path),
        ckpt_step=123,
        ref_ckpt_step=ref_ckpt_step,
        non_persistent_ckpt_type="global",
        no_load_optim=False,
        no_load_rng=False,
        finetune=False,
        megatron_to_hf_mode="bridge",
    )
    original_args = vars(instance.args).copy()
    instance.model = [object()]
    instance.optimizer = optimizer
    instance.weights_backuper = tensor_backper.TensorBackuper.create(lambda: [("weight", parameter)], single_tag=None)
    instance.weights_backuper.backup("actor")
    instance._active_model_tag = "actor"
    instance._expected_dpo_reference_identity = None
    if schema_version is not None:
        digest = canonical_tensor_sha256([("weight", torch.tensor([99.0]))])
        payload = {"schema_version": schema_version, "parameter_sha256": digest}
        if outcome == "identity_mismatch":
            payload["parameter_sha256"] = "a" * 64
        if schema_version == 1:
            payload.update(
                repository="old/repo", revision="old-revision", loader_mode="old-loader", probe_sha256="b" * 64
            )
        sidecar = tmp_path / "relax_dpo_reference.json"
        sidecar.write_text(json.dumps(payload), encoding="utf-8")
        instance._expected_dpo_reference_identity = read_reference_identity(sidecar)
    instance._assert_dp_reference_digest_equal = Mock()

    def load_reference(model, loaded_optimizer, scheduler, **kwargs):
        assert model is instance.model
        assert loaded_optimizer is scheduler is None
        assert (
            instance.args.load,
            instance.args.no_load_optim,
            instance.args.no_load_rng,
            instance.args.finetune,
        ) == (str(reference_root), True, True, True)
        assert instance.args.ckpt_step == expected_step
        assert instance.args.non_persistent_ckpt_type is None
        parameter.data.fill_(99)
        if outcome == "loader_failure":
            raise RuntimeError("injected loader failure")

    namespace["load_checkpoint"] = load_reference
    namespace["named_params_and_buffers"] = lambda *args, **kwargs: [("weight", parameter)]
    optimizer_state_before = {name: value.clone() for name, value in optimizer.state[parameter].items()}
    errors = {
        "loader_failure": "injected loader failure",
        "identity_mismatch": "frozen-reference identity mismatch",
    }
    with pytest.raises(RuntimeError, match=errors[outcome]) if outcome in errors else nullcontext():
        instance._rebuild_dpo_reference(str(reference_path))
    torch.testing.assert_close(parameter, actor_value)
    assert instance._active_model_tag == "actor"
    assert vars(instance.args) == original_args
    assert optimizer.state[parameter].keys() == optimizer_state_before.keys()
    for name, expected in optimizer_state_before.items():
        torch.testing.assert_close(optimizer.state[parameter][name], expected, rtol=0, atol=0)
    if outcome in {"loader_failure", "identity_mismatch"}:
        assert "ref" not in instance.weights_backuper.backup_tags
    elif outcome == "success":
        reference = instance.weights_backuper.get("ref")["weight"]
        torch.testing.assert_close(reference, torch.tensor([99.0]))
        parameter.data.add_(1)
        torch.testing.assert_close(reference, torch.tensor([99.0]))
        assert instance._dpo_reference_identity.schema_version == 2
        assert instance._dpo_reference_identity.parameter_sha256 == canonical_tensor_sha256([("weight", reference)])


@pytest.mark.parametrize(
    "source",
    [
        "missing",
        "empty",
        "missing_iteration",
        "empty_iteration",
        "iteration_without_tracker",
        "zero_with_other_tracker",
        "release_with_step",
    ],
)
def test_reference_rebuild_rejects_missing_or_ambiguous_checkpoint(tmp_path, reference_actor_methods, source):
    actor_type, namespace = reference_actor_methods
    reference_path = tmp_path / "reference"
    ref_ckpt_step = None
    error = "existing Hugging Face or Megatron checkpoint"
    if source != "missing":
        reference_path.mkdir()
    if source in {"missing_iteration", "empty_iteration"}:
        (reference_path / "latest_checkpointed_iteration.txt").write_text("7", encoding="utf-8")
        if source == "empty_iteration":
            (reference_path / "iter_0000007").mkdir()
        error = "checkpoint iteration is missing or empty"
    elif source == "iteration_without_tracker":
        reference_path /= "iter_0000007"
        reference_path.mkdir()
        (reference_path / "common.pt").write_bytes(b"reference checkpoint")
        error = "checkpoint root is missing its tracker"
    elif source in {"zero_with_other_tracker", "release_with_step"}:
        ref_ckpt_step = 0 if source == "zero_with_other_tracker" else 7
        latest = "3" if source == "zero_with_other_tracker" else "release"
        (reference_path / "latest_checkpointed_iteration.txt").write_text(latest, encoding="utf-8")
        iteration_dir = reference_path / ("iter_0000000" if ref_ckpt_step == 0 else "release")
        iteration_dir.mkdir()
        (iteration_dir / "common.pt").write_bytes(b"reference checkpoint")
        error = "Megatron cannot select this reference step"
    instance = actor_type()
    instance.args = Namespace(
        load=str(tmp_path / "actor-checkpoint"),
        ckpt_step=123,
        ref_ckpt_step=ref_ckpt_step,
        non_persistent_ckpt_type="global",
        no_load_optim=False,
        no_load_rng=False,
        finetune=False,
    )
    original_args = vars(instance.args).copy()
    instance.model = [object()]
    instance._active_model_tag = "actor"
    instance.weights_backuper = Mock(backup_tags={"actor"})
    namespace["load_checkpoint"] = Mock()

    with pytest.raises(ValueError, match=error):
        instance._rebuild_dpo_reference(str(reference_path))

    namespace["load_checkpoint"].assert_not_called()
    assert vars(instance.args) == original_args
    assert instance._active_model_tag == "actor"


def test_save_model_persists_reference_identity(tmp_path, reference_actor_methods):
    actor_type, namespace = reference_actor_methods
    instance = actor_type()
    instance.args = Namespace(
        loss_type="dpo",
        dpo_reference_free=False,
        debug_rollout_only=False,
        offload_train=False,
        async_save=False,
        save=str(tmp_path),
        save_hf=None,
    )
    instance.role = "actor"
    instance.model = [object()]
    instance.optimizer = instance.opt_param_scheduler = None
    instance._dpo_reference_identity = DPOReferenceIdentity(2, "a" * 64)
    namespace.update(
        dist=types.SimpleNamespace(get_rank=lambda **kwargs: 0, barrier=lambda **kwargs: None),
        get_gloo_group=lambda: None,
        rotate_ckpt=Mock(),
        save=Mock(),
    )
    instance.save_model(7)
    namespace["save"].assert_called_once_with(7, instance.model, None, None, lora_only=False)
    path = reference_identity_path(tmp_path, 7)
    assert read_reference_identity(path) == instance._dpo_reference_identity
    assert json.loads(path.read_text()) == {"schema_version": 2, "parameter_sha256": "a" * 64}
