# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import importlib.util
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from relax.engine.sft.runtime import resident_sft_param_overlap_enabled


_PATH = Path(__file__).resolve().parents[3] / "relax/backends/megatron/resident_shared_expert.py"
_SPEC = importlib.util.spec_from_file_location("resident_shared_expert_under_test", _PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
_verify_gate_modules = _MODULE._verify_gate_modules
verify_resident_shared_readiness = _MODULE.verify_resident_shared_readiness


def config():
    return SimpleNamespace(
        moe_token_dispatcher_type="flex",
        moe_flex_dispatcher_backend="deepep",
        moe_shared_expert_overlap=True,
        cuda_graph_impl="none",
    )


class Shared(torch.nn.Module):
    shared_gate_param_readiness_version = 1

    def __init__(self, *, trainable=True):
        super().__init__()
        self.config = config()
        self.use_shared_expert_gate = True
        self.gate_weight = torch.nn.Parameter(torch.zeros(2), requires_grad=trainable)


class DDP:
    def __init__(self, *modules):
        self.module = torch.nn.ModuleList(modules)
        self.config = config()
        self.ddp_config = SimpleNamespace(
            overlap_param_gather=True, overlap_grad_reduce=True, use_distributed_optimizer=True
        )
        self.param_to_bucket_group = {m.gate_weight: object() for m in modules if m.gate_weight.requires_grad}


def ready_module(*, trainable=True):
    module = Shared(trainable=trainable)

    def forbid_dispatch():
        raise AssertionError("Verification must not dispatch a parameter collective")

    module.gate_weight.ready = forbid_dispatch
    return module


def test_readiness_checks_all_chunks_without_dispatch():
    chunks = [DDP(ready_module(), ready_module(trainable=False)), DDP(), DDP(ready_module())]
    assert _verify_gate_modules(chunks, DDP, Shared, "ready") == (3, 2)


@pytest.mark.parametrize("failure", ["capability", "bucket", "callback", "ddp_overlap", "ddp_type"])
def test_readiness_rejects_incomplete_registration(monkeypatch, failure):
    module = ready_module()
    chunk = DDP(module)
    if failure == "capability":
        monkeypatch.setattr(Shared, "shared_gate_param_readiness_version", 0)
    elif failure == "bucket":
        chunk.param_to_bucket_group.clear()
    elif failure == "callback":
        module.gate_weight.ready = None
    elif failure == "ddp_overlap":
        chunk.ddp_config.overlap_param_gather = False
    elif failure == "ddp_type":
        chunk = object()
    with pytest.raises(ValueError):
        _verify_gate_modules([DDP(ready_module()), chunk], DDP, Shared, "ready")


def test_frozen_gate_needs_no_callback():
    assert _verify_gate_modules([DDP(Shared(trainable=False))], DDP, Shared, "ready") == (1, 0)


@pytest.mark.parametrize("target", ["chunk", "module"])
@pytest.mark.parametrize(
    "field,value",
    [
        ("cuda_graph_impl", "local"),
        ("moe_token_dispatcher_type", "alltoall"),
        ("moe_flex_dispatcher_backend", "other"),
        ("moe_shared_expert_overlap", False),
    ],
)
def test_actual_configuration_overrides_are_rejected(target, field, value):
    module = ready_module()
    chunk = DDP(module)
    setattr((chunk if target == "chunk" else module).config, field, value)
    with pytest.raises(ValueError, match="Actual shared-expert"):
        _verify_gate_modules([chunk], DDP, Shared, "ready")


@pytest.mark.parametrize("flag", ["use_distributed_optimizer", "overlap_grad_reduce"])
def test_actual_ddp_configuration_is_required(flag):
    chunk = DDP(ready_module())
    setattr(chunk.ddp_config, flag, False)
    with pytest.raises(ValueError, match="requires DDP"):
        _verify_gate_modules([chunk], DDP, Shared, "ready")


def test_empty_chunks_are_not_a_gateless_pipeline_stage():
    with pytest.raises(ValueError, match="at least one"):
        _verify_gate_modules([], DDP, Shared, "ready")


def test_shared_args_cannot_bypass_actual_configuration(monkeypatch):
    monkeypatch.setenv("RELAX_SFT_RESIDENT_PARAM_OVERLAP", "1")
    args = Namespace(moe_shared_expert_overlap=False)
    with pytest.raises(ValueError, match="disagree"):
        verify_resident_shared_readiness(args, [DDP(ready_module())])


@pytest.mark.parametrize("resident,shared", [(False, True), (True, False)])
def test_disabled_verification_clears_stale_state_without_importing_megatron(monkeypatch, resident, shared):
    monkeypatch.setenv("RELAX_SFT_RESIDENT_PARAM_OVERLAP", "1" if resident else "0")
    args = Namespace(moe_shared_expert_overlap=shared, _relax_shared_expert_gate_ready_verified=True)
    verify_resident_shared_readiness(args, [object()])
    assert args._relax_shared_expert_gate_ready_verified is False


@pytest.mark.parametrize(
    "field,value",
    [
        ("moe_token_dispatcher_type", "alltoall"),
        ("moe_flex_dispatcher_backend", "other"),
        ("cuda_graph_impl", "local"),
        ("enable_cuda_graph", True),
        ("connections", "1"),
    ],
)
def test_unvalidated_configuration_rejected_before_optional_import(monkeypatch, field, value):
    monkeypatch.setenv("RELAX_SFT_RESIDENT_PARAM_OVERLAP", "1")
    monkeypatch.setenv("CUDA_DEVICE_MAX_CONNECTIONS", "32")
    args = Namespace(
        moe_shared_expert_overlap=True,
        moe_token_dispatcher_type="flex",
        moe_flex_dispatcher_backend="deepep",
        cuda_graph_impl="none",
    )
    if field == "connections":
        monkeypatch.setenv("CUDA_DEVICE_MAX_CONNECTIONS", value)
    else:
        setattr(args, field, value)
    with pytest.raises(ValueError, match="gated to"):
        verify_resident_shared_readiness(args, [])
    assert args._relax_shared_expert_gate_ready_verified is False


def test_runtime_requires_successful_verification(monkeypatch):
    monkeypatch.setenv("RELAX_SFT_RESIDENT_PARAM_OVERLAP", "1")
    args = Namespace(
        loss_type="sft",
        use_distributed_optimizer=True,
        overlap_grad_reduce=True,
        overlap_param_gather=True,
        moe_shared_expert_overlap=True,
    )
    with pytest.raises(ValueError, match="verified gate"):
        resident_sft_param_overlap_enabled(args)
    args._relax_shared_expert_gate_ready_verified = True
    assert resident_sft_param_overlap_enabled(args)
