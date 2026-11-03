# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Frozen replicas keep normal registration and bypass only root sharding."""

import inspect

import pytest
import torch

from relax.backends.fsdp.runtime import fsdp2_wrap, resolve_replicated_modules


class _Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Linear(4, 4)

    def forward(self, value):
        return self.projection(value)


def _model():
    model = torch.nn.Module()
    model._no_split_modules = ("_Block",)
    model.encoder = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 4)).requires_grad_(False)
    model.blocks = torch.nn.ModuleList([_Block(), _Block()])
    model.output = torch.nn.Linear(4, 4)
    return model.to(dtype=torch.bfloat16)


def test_native_fully_shard_exposes_ignored_parameters():
    from torch.distributed.fsdp import fully_shard

    assert "ignored_params" in inspect.signature(fully_shard).parameters


def test_replicated_modules_keep_registration_and_ignore_only_root_parameters(monkeypatch):
    model = _model()
    model._fsdp_replicated_modules = (model.encoder,)
    model.blocks[0].projection._fsdp_fine_grained = True
    model._fsdp_module_groups = ((model.output,),)
    names = list(model.state_dict())
    parameters = dict(model.named_parameters())
    encoder = model.encoder
    calls = []
    monkeypatch.setattr("torch.distributed.fsdp.fully_shard", lambda module, **kwargs: calls.append((module, kwargs)))

    assert resolve_replicated_modules(model) == [encoder]
    assert fsdp2_wrap(model, master_dtype="fp32") is model
    assert [module for module, _ in calls] == [model.blocks[0].projection, [model.output], *model.blocks, model]
    assert all("ignored_params" not in kwargs for _, kwargs in calls[:-1])
    assert calls[-1][1]["ignored_params"] == set(encoder.parameters())
    assert model.encoder is encoder
    assert list(model.state_dict()) == names
    assert all(parameter is parameters[name] for name, parameter in model.named_parameters())
    assert all(parameter.dtype == torch.bfloat16 for parameter in encoder.parameters())
    assert all(parameter.dtype == torch.float32 for parameter in model.parameters() if parameter.requires_grad)


@pytest.mark.parametrize("declaration", [None, (), []])
def test_replicated_modules_default_does_not_change_shard_kwargs(monkeypatch, declaration):
    model = _model()
    if declaration is not None:
        model._fsdp_replicated_modules = declaration
    calls = []
    monkeypatch.setattr("torch.distributed.fsdp.fully_shard", lambda module, **kwargs: calls.append((module, kwargs)))
    assert resolve_replicated_modules(model) == []
    fsdp2_wrap(model)
    assert [module for module, _ in calls] == [*model.blocks, model]
    assert all(set(kwargs) == {"mp_policy", "reshard_after_forward"} for _, kwargs in calls)


@pytest.mark.parametrize("declaration", [None, "encoder", {}, set()])
def test_replicated_modules_reject_nonsequence_declarations(declaration):
    model = _model()
    model._fsdp_replicated_modules = declaration
    with pytest.raises(ValueError, match="sequence of modules"):
        resolve_replicated_modules(model)


@pytest.mark.parametrize(
    "case",
    [
        "foreign",
        "root",
        "nonmodule",
        "trainable",
        "duplicate",
        "ancestor",
        "fine_descendant",
        "fine_ancestor",
        "group_descendant",
        "group_ancestor",
        "block_descendant",
        "block_ancestor",
        "module_alias",
        "child_alias",
        "parameter_alias",
        "storage_alias",
        "two_replicas_share_parameter",
        "two_replicas_share_storage",
        "cycle",
    ],
)
def test_replicated_modules_reject_unsafe_ownership_before_mutation(monkeypatch, case):
    model = _model()
    declared = (model.encoder,)
    if case == "foreign":
        declared = (torch.nn.Linear(4, 4).requires_grad_(False),)
    elif case == "root":
        declared = (model,)
    elif case == "nonmodule":
        declared = ("encoder",)
    elif case == "trainable":
        model.encoder[0].weight.requires_grad_(True)
    elif case == "duplicate":
        declared = (model.encoder, model.encoder)
    elif case == "ancestor":
        declared = (model.encoder, model.encoder[0])
    elif case == "fine_descendant":
        model.encoder[0]._fsdp_fine_grained = True
    elif case == "fine_ancestor":
        model.encoder._fsdp_fine_grained = True
        declared = (model.encoder[0],)
    elif case == "group_descendant":
        model._fsdp_module_groups = ((model.encoder[0],),)
    elif case == "group_ancestor":
        model._fsdp_module_groups = ((model.encoder,),)
        declared = (model.encoder[0],)
    elif case == "block_descendant":
        model.encoder.append(_Block().requires_grad_(False))
    elif case == "block_ancestor":
        model.blocks[0].projection.requires_grad_(False)
        declared = (model.blocks[0].projection,)
    elif case == "module_alias":
        model.encoder_alias = model.encoder
    elif case == "child_alias":
        model.output = model.encoder[0]
    elif case == "parameter_alias":
        model.output.weight = model.encoder[0].weight
    elif case == "storage_alias":
        model.output.weight = torch.nn.Parameter(model.encoder[0].weight.detach(), requires_grad=False)
    elif case in ("two_replicas_share_parameter", "two_replicas_share_storage"):
        model.output.requires_grad_(False)
        weight = model.encoder[0].weight
        model.output.weight = (
            weight if case.endswith("parameter") else torch.nn.Parameter(weight.detach(), requires_grad=False)
        )
        declared = (model.encoder, model.output)
    elif case == "cycle":
        model.encoder.loop = model.encoder
    model._fsdp_replicated_modules = declared
    parameters = [(parameter, parameter.dtype) for parameter in model.parameters()]
    calls = []
    monkeypatch.setattr("torch.distributed.fsdp.fully_shard", lambda module, **kwargs: calls.append(module))
    with pytest.raises(ValueError, match="FSDP"):
        fsdp2_wrap(model, master_dtype="fp32", activation_checkpointing=True)
    assert not calls
    assert all(parameter.dtype == dtype for parameter, dtype in parameters)


def test_replicated_modules_allow_internal_ties_and_disjoint_replicas():
    model = _model()
    model.encoder[1].weight = model.encoder[0].weight
    model.encoder.append(model.encoder[0])
    model.output.requires_grad_(False)
    model._fsdp_replicated_modules = (model.encoder, model.output)
    assert resolve_replicated_modules(model) == [model.encoder, model.output]


def test_replicated_modules_remain_ignored_with_activation_checkpointing(monkeypatch):
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointWrapper

    model = _model()
    model._fsdp_replicated_modules = (model.encoder,)
    encoder = model.encoder
    calls = []
    monkeypatch.setattr("torch.distributed.fsdp.fully_shard", lambda module, **kwargs: calls.append((module, kwargs)))
    fsdp2_wrap(model, activation_checkpointing=True)
    assert len(calls) == 3
    assert all(isinstance(module, CheckpointWrapper) for module, _ in calls[:-1])
    assert all("ignored_params" not in kwargs for _, kwargs in calls[:-1])
    assert calls[-1][1]["ignored_params"] == set(encoder.parameters())
    assert model.encoder is encoder


def test_collective_free_conditioning_is_granted_only_after_successful_wrapping(monkeypatch):
    model = _model()
    model.encoder.eval()
    model._fsdp_replicated_modules = (model.encoder,)
    model._conditioning_collective_free_modules = (model.encoder,)
    calls = []

    def wrap(module, **kwargs):
        assert not hasattr(model, "_conditioning_execution_policy")
        calls.append((module, kwargs))

    monkeypatch.setattr("torch.distributed.fsdp.fully_shard", wrap)
    fsdp2_wrap(model)
    assert model._conditioning_execution_policy == "rank_local"
    assert calls[-1][1]["ignored_params"] == set(model.encoder.parameters())


def test_undeclared_replicated_conditioning_keeps_unanimous_execution(monkeypatch):
    model = _model()
    model._fsdp_replicated_modules = (model.encoder,)
    model._conditioning_execution_policy = "rank_local"  # A stale declaration is not a backend grant.
    monkeypatch.setattr("torch.distributed.fsdp.fully_shard", lambda *_args, **_kwargs: None)
    fsdp2_wrap(model)
    assert model._conditioning_execution_policy == "unanimous"


def test_collective_free_conditioning_accepts_multiple_complete_replica_roots(monkeypatch):
    model = _model()
    model.encoder.eval()
    model.output.eval().requires_grad_(False)
    model._fsdp_replicated_modules = (model.encoder, model.output)
    model._conditioning_collective_free_modules = (model.encoder, model.output)
    calls = []
    monkeypatch.setattr("torch.distributed.fsdp.fully_shard", lambda module, **kwargs: calls.append((module, kwargs)))
    fsdp2_wrap(model)
    assert model._conditioning_execution_policy == "rank_local"
    assert calls[-1][1]["ignored_params"] == set(model.encoder.parameters()) | set(model.output.parameters())


@pytest.mark.parametrize("case", ["nonsequence", "nonmodule", "duplicate", "unreplicated", "descendant", "train"])
def test_collective_free_conditioning_rejects_invalid_declaration_before_mutation(monkeypatch, case):
    model = _model()
    model.encoder.eval()
    model._fsdp_replicated_modules = (model.encoder,)
    declared = (model.encoder,)
    if case == "nonsequence":
        declared = "encoder"
    elif case == "nonmodule":
        declared = ("encoder",)
    elif case == "duplicate":
        declared = (model.encoder, model.encoder)
    elif case == "unreplicated":
        model._fsdp_replicated_modules = ()
    elif case == "descendant":
        declared = (model.encoder[0],)
    elif case == "train":
        model.encoder[0].train()
    model._conditioning_collective_free_modules = declared
    parameters = [(parameter, parameter.dtype) for parameter in model.parameters()]
    calls = []
    monkeypatch.setattr("torch.distributed.fsdp.fully_shard", lambda module, **kwargs: calls.append(module))
    with pytest.raises(ValueError, match="Collective-free conditioning"):
        fsdp2_wrap(model, master_dtype="fp32")
    assert not calls and not hasattr(model, "_conditioning_execution_policy")
    assert all(parameter.dtype == dtype for parameter, dtype in parameters)


def test_collective_free_conditioning_rejects_preexisting_fsdp_boundary(monkeypatch):
    from torch.distributed.fsdp import FSDPModule

    class WrappedEncoder(FSDPModule, torch.nn.Sequential):
        pass

    model = _model()
    model.encoder = torch.nn.Sequential(torch.nn.Linear(4, 4)).eval().requires_grad_(False)
    # Native fully_shard changes the existing module class in place.
    # Constructing WrappedEncoder directly invokes FSDPModule.__new__, which
    # intentionally reconstructs the original Sequential without FSDP.
    model.encoder.__class__ = WrappedEncoder
    assert isinstance(model.encoder, FSDPModule)
    model._fsdp_replicated_modules = (model.encoder,)
    model._conditioning_collective_free_modules = (model.encoder,)
    monkeypatch.setattr(
        "torch.distributed.fsdp.fully_shard", lambda *_args, **_kwargs: pytest.fail("Must reject first")
    )
    with pytest.raises(ValueError, match="nested FSDP"):
        fsdp2_wrap(model)


def test_collective_free_policy_is_not_granted_on_failed_wrap(monkeypatch):
    model = _model()
    model.encoder.eval()
    model._fsdp_replicated_modules = (model.encoder,)
    model._conditioning_collective_free_modules = (model.encoder,)

    def fail(*_args, **_kwargs):
        raise RuntimeError("failed shard")

    monkeypatch.setattr("torch.distributed.fsdp.fully_shard", fail)
    with pytest.raises(RuntimeError, match="failed shard"):
        fsdp2_wrap(model)
    assert not hasattr(model, "_conditioning_execution_policy")
