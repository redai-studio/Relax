# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from argparse import Namespace
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

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
from relax.engine.sft.runtime import should_bypass_main_output_layer, should_skip_sft_weight_management


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
        (False, "dpo", True, False),
        (False, "rm", True, False),
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


@pytest.mark.parametrize("prepacked", [False, True])
def test_mtp_only_sft_reaches_train_without_weight_backuper(monkeypatch, prepacked):
    actor_module = pytest.importorskip("relax.backends.megatron.actor", exc_type=ImportError)

    class ReachedTrain(BaseException):
        pass

    actor = object.__new__(actor_module.MegatronTrainRayActor)
    actor.args = _actor_args(
        advantage_estimator="grpo",
        compute_advantages_and_returns=False,
        use_routing_replay=False,
        use_rollout_routing_replay=False,
    )
    actor.model = []
    actor.optimizer = object()
    actor.opt_param_scheduler = object()
    actor.rollout_data_postprocess = None
    assert should_skip_sft_weight_management(actor.args)
    assert not hasattr(actor, "weights_backuper")

    iterator = SimpleNamespace(close=Mock())
    data_iterator, num_microbatches = [iterator], [1]
    make_iterator = Mock(return_value=(data_iterator, num_microbatches))
    train = Mock(side_effect=ReachedTrain)
    monkeypatch.setattr(actor_module, "get_data_iterator", make_iterator)
    monkeypatch.setattr(actor_module, "timer", lambda *_args, **_kwargs: nullcontext())
    monkeypatch.setattr(actor_module, "inverse_timer", lambda *_args, **_kwargs: nullcontext())
    monkeypatch.setattr(actor_module, "log_rollout_data", lambda *_args: None)
    monkeypatch.setattr(actor_module, "train", train)

    kwargs = (
        {"prepared_data_iterator": data_iterator, "prepared_num_microbatches": num_microbatches} if prepacked else {}
    )
    rollout_data = {}
    with pytest.raises(ReachedTrain):
        actor.train_actor(0, rollout_data, **kwargs)

    train.assert_called_once_with(
        0, actor.model, actor.optimizer, actor.opt_param_scheduler, data_iterator, num_microbatches
    )
    iterator.close.assert_called_once()
    if prepacked:
        make_iterator.assert_not_called()
    else:
        make_iterator.assert_called_once_with(actor.args, actor.model, rollout_data)
