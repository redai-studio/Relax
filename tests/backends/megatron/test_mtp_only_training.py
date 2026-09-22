# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from argparse import Namespace
from pathlib import Path

import pytest
import torch


pytest.importorskip("megatron.training.arguments", exc_type=ImportError)

from relax.backends.megatron import model_provider as model_provider_mod
from relax.backends.megatron.loss import mtp_only_loss_function
from relax.backends.megatron.model_provider import (
    configure_mtp_detach_paths,
    freeze_model_params,
    validate_mtp_only_trainable_params,
)
from relax.engine.sft.runtime import should_bypass_main_output_layer, should_skip_mtp_only_weight_management


class _MtpOnlyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = torch.nn.Linear(4, 4)
        self.mtp = torch.nn.Linear(4, 4)


class _ConfiguredModule(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config


MTP_PATCH = Path(__file__).resolve().parents[3] / "docker" / "patch" / "latest" / "megatron.patch"


def test_megatron_patch_routes_each_mtp_detach_path_independently():
    patch = MTP_PATCH.read_text()

    assert 'getattr(config, "mtp_detach_lm_head", True)' in patch
    assert 'getattr(self.config, "mtp_detach_embedding", True)' in patch
    assert 'offset == 0 and getattr(self.config, "mtp_detach_backbone", True)' in patch
    assert "hidden_states = hidden_states.detach().requires_grad_(True)" in patch
    assert "mtp_detach_main_model" not in patch


@pytest.mark.parametrize(
    "detach_paths",
    [
        (),
        ("embedding",),
        ("backbone",),
        ("lm-head",),
        ("embedding", "backbone"),
        ("embedding", "lm-head"),
        ("backbone", "lm-head"),
        ("embedding", "backbone", "lm-head"),
    ],
)
def test_mtp_detach_path_gradient_matrix(detach_paths):
    torch.manual_seed(0)
    embedding = torch.nn.Linear(4, 4, bias=False)
    backbone = torch.nn.Linear(4, 4, bias=False)
    mtp = torch.nn.Linear(4, 4, bias=False)
    lm_head = torch.nn.Linear(4, 2, bias=False)
    inputs = torch.randn(3, 4)

    decoder_input = embedding(inputs)
    if "embedding" in detach_paths:
        decoder_input = decoder_input.detach()
    hidden_states = backbone(inputs)
    if "backbone" in detach_paths:
        hidden_states = hidden_states.detach().requires_grad_(True)
    mtp_hidden = mtp(decoder_input + hidden_states)
    if "lm-head" in detach_paths:
        logits = torch.func.functional_call(lm_head, {"weight": lm_head.weight.detach()}, (mtp_hidden,))
    else:
        logits = lm_head(mtp_hidden)

    logits.square().sum().backward()

    assert (embedding.weight.grad is None) is ("embedding" in detach_paths)
    assert (backbone.weight.grad is None) is ("backbone" in detach_paths)
    assert (lm_head.weight.grad is None) is ("lm-head" in detach_paths)
    assert mtp.weight.grad is not None


@pytest.mark.parametrize(
    "detach_paths",
    [(), ("embedding",), ("lm-head",), ("embedding", "lm-head")],
)
def test_tied_embedding_weight_follows_each_active_gradient_path(detach_paths):
    torch.manual_seed(0)
    shared_weight = torch.nn.Parameter(torch.randn(4, 4))
    mtp = torch.nn.Linear(4, 4, bias=False)
    inputs = torch.randn(3, 4)

    decoder_input = inputs @ shared_weight.t()
    if "embedding" in detach_paths:
        decoder_input = decoder_input.detach()
    mtp_hidden = mtp(decoder_input)
    output_weight = shared_weight.detach() if "lm-head" in detach_paths else shared_weight

    (mtp_hidden @ output_weight.t()).square().sum().backward()

    both_shared_paths_detached = "embedding" in detach_paths and "lm-head" in detach_paths
    assert (shared_weight.grad is None) is both_shared_paths_detached
    assert mtp.weight.grad is not None


def test_backbone_detach_preserves_gradients_between_mtp_layers():
    backbone = torch.nn.Linear(4, 4, bias=False)
    first_mtp_layer = torch.nn.Linear(4, 4, bias=False)
    second_mtp_layer = torch.nn.Linear(4, 4, bias=False)

    hidden_states = backbone(torch.randn(3, 4)).detach().requires_grad_(True)
    second_mtp_layer(first_mtp_layer(hidden_states)).square().sum().backward()

    assert backbone.weight.grad is None
    assert first_mtp_layer.weight.grad is not None
    assert second_mtp_layer.weight.grad is not None


@pytest.mark.parametrize(
    ("detach_paths", "expected"),
    [
        ((), (False, False, False)),
        (("embedding",), (True, False, False)),
        (("backbone", "lm-head"), (False, True, True)),
        (("embedding", "backbone", "lm-head"), (True, True, True)),
    ],
)
def test_configure_mtp_detach_paths_updates_all_module_configs(detach_paths, expected):
    shared_config = Namespace()
    model = _ConfiguredModule(shared_config)
    model.child = _ConfiguredModule(Namespace())
    model.shared_child = _ConfiguredModule(shared_config)

    configure_mtp_detach_paths(Namespace(mtp_detach_paths=detach_paths), model)

    for config in (model.config, model.child.config, model.shared_child.config):
        assert (config.mtp_detach_embedding, config.mtp_detach_backbone, config.mtp_detach_lm_head) == expected


def test_configure_mtp_detach_paths_defaults_to_all_detached():
    model = _ConfiguredModule(Namespace())

    configure_mtp_detach_paths(Namespace(), model)

    assert model.config.mtp_detach_embedding is True
    assert model.config.mtp_detach_backbone is True
    assert model.config.mtp_detach_lm_head is True


def test_mtp_only_freeze_exposes_only_mtp_parameters():
    model = _MtpOnlyModel()
    args = Namespace(
        only_train_params_name_list=[r"(^|\.)mtp(\.|$)"],
        freeze_params_name_list=None,
    )

    freeze_model_params(model, args)

    trainable_names = {name for name, param in model.named_parameters() if param.requires_grad}
    assert trainable_names == {"mtp.weight", "mtp.bias"}


def test_mtp_only_trainable_validation_accepts_frozen_backbone():
    model = _MtpOnlyModel()
    freeze_model_params(
        model,
        Namespace(only_train_params_name_list=[r"(^|\.)mtp(\.|$)"], freeze_params_name_list=None),
    )

    validate_mtp_only_trainable_params(Namespace(mtp_only_training=True), [model])


def test_mtp_only_trainable_validation_rejects_non_mtp_parameter():
    model = _MtpOnlyModel()

    with pytest.raises(RuntimeError, match="non-MTP parameters trainable"):
        validate_mtp_only_trainable_params(Namespace(mtp_only_training=True), [model])


def test_mtp_only_trainable_validation_requires_mtp_parameter():
    model = torch.nn.Linear(4, 4)
    model.requires_grad_(False)

    with pytest.raises(RuntimeError, match="found no trainable MTP parameters"):
        validate_mtp_only_trainable_params(Namespace(mtp_only_training=True), [model])


def test_mtp_only_trainable_validation_allows_pp_rank_without_local_mtp(monkeypatch):
    model = torch.nn.Linear(4, 4)
    model.requires_grad_(False)

    monkeypatch.setattr(model_provider_mod.dist, "is_available", lambda: True)
    monkeypatch.setattr(model_provider_mod.dist, "is_initialized", lambda: True)

    def emulate_world_all_reduce(counts, *, group):
        assert group is model_provider_mod.dist.group.WORLD
        counts[0] = 1

    monkeypatch.setattr(model_provider_mod.dist, "all_reduce", emulate_world_all_reduce)

    validate_mtp_only_trainable_params(Namespace(mtp_only_training=True), [model])


@pytest.mark.parametrize(
    ("mtp_only", "loss_type", "sft_chunked", "expected"),
    [
        (True, "sft", False, True),
        (False, "sft", True, True),
        (False, "sft", False, False),
        (False, "policy_loss", True, False),
    ],
)
def test_mtp_only_or_chunked_sft_bypasses_main_output_layer(mtp_only, loss_type, sft_chunked, expected):
    args = Namespace(mtp_only_training=mtp_only, loss_type=loss_type, sft_chunked_logits=sft_chunked)

    assert should_bypass_main_output_layer(args) is expected


class _MtpLossAutoScalerStandIn(torch.autograd.Function):
    """Mimic MCore's MTP loss attachment, whose backward ignores
    grad_output."""

    @staticmethod
    def forward(ctx, hidden_states, mtp_loss):
        return hidden_states

    @staticmethod
    def backward(ctx, grad_output):
        return torch.zeros_like(grad_output), torch.ones((), device=grad_output.device)


def test_mtp_only_zero_anchor_triggers_attached_mtp_loss_backward():
    backbone_hidden = torch.randn(2, 3, requires_grad=True)
    mtp_parameter = torch.nn.Parameter(torch.tensor(2.0))
    hidden_with_mtp_loss = _MtpLossAutoScalerStandIn.apply(backbone_hidden, mtp_parameter.square())

    loss, log = mtp_only_loss_function(Namespace(), {}, hidden_with_mtp_loss, lambda value: value)
    loss.backward()

    assert loss.detach() == 0
    assert log["loss"] == 0
    torch.testing.assert_close(backbone_hidden.grad, torch.zeros_like(backbone_hidden))
    torch.testing.assert_close(mtp_parameter.grad, torch.tensor(4.0))


def _actor_args(**overrides) -> Namespace:
    defaults = {
        "mtp_only_training": True,
        "loss_type": "sft",
        "sft_predict_interval": None,
        "offload_train": False,
        "keep_old_actor": False,
    }
    defaults.update(overrides)
    return Namespace(**defaults)


def test_pure_mtp_only_sft_skips_weight_management():
    assert should_skip_mtp_only_weight_management(_actor_args()) is True


@pytest.mark.parametrize(
    ("overrides", "extra_kwargs"),
    [
        ({"mtp_only_training": False}, {}),
        ({"loss_type": "policy_loss"}, {}),
        ({"sft_predict_interval": 10}, {}),
        ({"offload_train": True}, {}),
        ({"keep_old_actor": True}, {}),
        ({}, {"with_ref": True}),
        ({}, {"with_opd_teacher": True}),
    ],
)
def test_mtp_only_keeps_weight_management_when_another_consumer_needs_it(overrides, extra_kwargs):
    assert should_skip_mtp_only_weight_management(_actor_args(**overrides), **extra_kwargs) is False
