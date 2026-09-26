# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Reward-model checkpoint metadata and exact head-schema tests."""

import copy
from types import SimpleNamespace

import pytest
import torch


try:
    from relax.backends.megatron import checkpoint as checkpoint_module
except Exception as exc:
    pytest.skip(f"Megatron checkpoint helpers unavailable: {exc}", allow_module_level=True)


class _TensorMetadata:
    def __init__(self, shape):
        self.global_shape = shape


@pytest.mark.parametrize(
    ("tracker_value", "expected_directory"),
    [("0", "iter_0000000"), ("17", "iter_0000017"), ("release", "release")],
)
def test_checkpoint_iteration_dir_supports_megatron_tracker_formats(tmp_path, tracker_value, expected_directory):
    (tmp_path / "latest_checkpointed_iteration.txt").write_text(tracker_value, encoding="utf-8")
    assert checkpoint_module._checkpoint_iteration_dir(tmp_path) == tmp_path / expected_directory


def test_checkpoint_iteration_dir_rejects_invalid_tracker_metadata(tmp_path):
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("invalid", encoding="utf-8")
    with pytest.raises(RuntimeError, match="cannot resolve Megatron checkpoint iteration"):
        checkpoint_module._checkpoint_iteration_dir(tmp_path)


def test_checkpoint_iteration_dir_honors_explicit_checkpoint_step_for_iteration_tracker(tmp_path):
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("17", encoding="utf-8")
    assert checkpoint_module._checkpoint_iteration_dir(tmp_path, ckpt_step=42) == tmp_path / "iter_0000042"


def test_checkpoint_iteration_dir_honors_explicit_zero_checkpoint_step(tmp_path):
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("17", encoding="utf-8")
    assert checkpoint_module._checkpoint_iteration_dir(tmp_path, ckpt_step=0) == tmp_path / "iter_0000000"


def test_checkpoint_iteration_dir_keeps_release_when_checkpoint_step_is_set(tmp_path):
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("release", encoding="utf-8")
    assert checkpoint_module._checkpoint_iteration_dir(tmp_path, ckpt_step=42) == tmp_path / "release"


def test_checkpoint_iteration_dir_rejects_negative_iterations(tmp_path):
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("-1", encoding="utf-8")
    with pytest.raises(RuntimeError, match="must be non-negative"):
        checkpoint_module._checkpoint_iteration_dir(tmp_path)


def test_reward_model_tensor_metadata_accepts_exact_bias_free_head():
    checkpoint_module._validate_reward_model_tensor_metadata(
        {
            "model.output_layer.weight": _TensorMetadata((1, 1024)),
            "model.decoder.weight": _TensorMetadata((8, 8)),
            "optimizer.state.exp_avg.model.output_layer.weight": _TensorMetadata((1, 1024)),
            "optimizer.state.exp_avg_sq.model.output_layer.weight": _TensorMetadata((1, 1024)),
        },
        1024,
    )


@pytest.mark.parametrize(
    ("metadata", "match"),
    [
        ({}, "exactly one"),
        ({"model.output_layer.weight": _TensorMetadata((2, 1024))}, "shape mismatch"),
        (
            {
                "model.output_layer.weight": _TensorMetadata((1, 1024)),
                "model.output_layer.bias": _TensorMetadata((1,)),
            },
            "unexpected",
        ),
        (
            {
                "model.output_layer.weight": _TensorMetadata((1, 1024)),
                "model.reward_model_head.weight": _TensorMetadata((1, 1024)),
            },
            "unexpected",
        ),
    ],
)
def test_reward_model_tensor_metadata_rejects_missing_extra_and_wrong_shape(metadata, match):
    with pytest.raises(RuntimeError, match=match):
        checkpoint_module._validate_reward_model_tensor_metadata(metadata, 1024)


def test_reward_model_contract_rejects_critic_metadata_before_tensor_load(monkeypatch, tmp_path):
    calls = []
    fake_dist_checkpointing = SimpleNamespace(
        check_is_distributed_checkpoint=lambda path: True,
        load_common_state_dict=lambda path: {
            "args": SimpleNamespace(
                sft_objective="causal_lm", head_type="critic_value_terminal_v1", checkpoint_role="critic"
            )
        },
        load_tensors_metadata=lambda path: calls.append(path),
    )
    import megatron.core

    monkeypatch.setattr(megatron.core, "dist_checkpointing", fake_dist_checkpointing)
    args = SimpleNamespace(
        loss_type="sft",
        sft_objective="reward_model",
        hidden_size=1024,
        no_load_optim=False,
        no_load_rng=False,
        finetune=False,
        reset_optimizer_states=False,
    )
    model = [SimpleNamespace(role="actor")]
    with pytest.raises(RuntimeError, match="RM resume requires checkpoint metadata"):
        checkpoint_module._validate_checkpoint_contract(args, model, tmp_path)
    assert calls == []


def test_reward_model_contract_rejects_release_checkpoint_before_metadata_load(monkeypatch, tmp_path):
    calls = []
    import megatron.core

    monkeypatch.setattr(
        megatron.core,
        "dist_checkpointing",
        SimpleNamespace(load_common_state_dict=lambda path: calls.append(path)),
    )
    args = SimpleNamespace(loss_type="sft", sft_objective="reward_model")
    with pytest.raises(RuntimeError, match="rejects release checkpoints"):
        checkpoint_module._validate_checkpoint_contract(args, [SimpleNamespace(role="actor")], tmp_path / "release")
    assert calls == []


@pytest.mark.parametrize(
    ("args", "role"),
    [
        (SimpleNamespace(loss_type="sft", sft_objective="causal_lm"), "actor"),
        (SimpleNamespace(loss_type="sft", sft_objective="dpo"), "actor"),
        (SimpleNamespace(), "critic"),
    ],
)
def test_non_reward_model_contract_defers_legacy_checkpoint_to_megatron(monkeypatch, tmp_path, args, role):
    common_state_loads = []
    fake_dist_checkpointing = SimpleNamespace(
        check_is_distributed_checkpoint=lambda path: False,
        load_common_state_dict=lambda path: common_state_loads.append(path),
    )
    import megatron.core

    monkeypatch.setattr(megatron.core, "dist_checkpointing", fake_dist_checkpointing)
    checkpoint_module._validate_checkpoint_contract(args, [SimpleNamespace(role=role)], tmp_path)
    assert common_state_loads == []


def test_reward_model_contract_rejects_legacy_checkpoint_before_metadata_load(monkeypatch, tmp_path):
    common_state_loads = []
    fake_dist_checkpointing = SimpleNamespace(
        check_is_distributed_checkpoint=lambda path: False,
        load_common_state_dict=lambda path: common_state_loads.append(path),
    )
    import megatron.core

    monkeypatch.setattr(megatron.core, "dist_checkpointing", fake_dist_checkpointing)
    args = SimpleNamespace(loss_type="sft", sft_objective="reward_model")
    with pytest.raises(RuntimeError, match="RM resume requires a distributed checkpoint"):
        checkpoint_module._validate_checkpoint_contract(args, [SimpleNamespace(role="actor")], tmp_path)
    assert common_state_loads == []


def test_checkpoint_wrapper_delegates_non_reward_model_legacy_checkpoint(monkeypatch, tmp_path):
    import megatron.core

    upstream_loads = []
    fake_dist_checkpointing = SimpleNamespace(
        check_is_distributed_checkpoint=lambda path: False,
        load_common_state_dict=lambda path: pytest.fail("legacy checkpoint must not use distributed common state"),
    )
    args = SimpleNamespace(load=str(tmp_path), loss_type="sft", sft_objective="causal_lm", ckpt_step=None)
    monkeypatch.setattr(megatron.core, "dist_checkpointing", fake_dist_checkpointing)
    monkeypatch.setattr(checkpoint_module, "get_args", lambda: args)
    monkeypatch.setattr(checkpoint_module, "_is_dir_nonempty", lambda _: True)
    monkeypatch.setattr(checkpoint_module, "is_megatron_checkpoint", lambda _: True)
    monkeypatch.setattr(checkpoint_module, "_checkpoint_iteration_dir", lambda *_: tmp_path / "iter_0000001")
    monkeypatch.setattr(
        checkpoint_module,
        "_load_checkpoint_megatron",
        lambda **kwargs: upstream_loads.append(kwargs) or (1, 0),
    )

    assert checkpoint_module.load_checkpoint([SimpleNamespace(role="actor")], None, None, None, False) == (1, 0)
    assert len(upstream_loads) == 1


def test_reward_model_contract_accepts_complete_metadata_and_exact_head(monkeypatch, tmp_path):
    fake_dist_checkpointing = SimpleNamespace(
        check_is_distributed_checkpoint=lambda path: True,
        load_common_state_dict=lambda path: {
            "args": SimpleNamespace(
                sft_objective="reward_model",
                head_type="reward_model_terminal_v1",
                checkpoint_role="actor",
            )
        },
        load_tensors_metadata=lambda path: {"model.output_layer.weight": _TensorMetadata((1, 1024))},
    )
    import megatron.core

    monkeypatch.setattr(megatron.core, "dist_checkpointing", fake_dist_checkpointing)
    args = SimpleNamespace(
        loss_type="sft",
        sft_objective="reward_model",
        hidden_size=1024,
        no_load_optim=False,
        no_load_rng=False,
        finetune=False,
        reset_optimizer_states=False,
    )
    checkpoint_module._validate_checkpoint_contract(args, [SimpleNamespace(role="actor")], tmp_path)


@pytest.mark.parametrize("flag", ["no_load_optim", "no_load_rng", "finetune", "reset_optimizer_states"])
def test_reward_model_contract_rejects_partial_resume_flags(monkeypatch, tmp_path, flag):
    fake_dist_checkpointing = SimpleNamespace(
        check_is_distributed_checkpoint=lambda path: True,
        load_common_state_dict=lambda path: {
            "args": SimpleNamespace(
                sft_objective="reward_model",
                head_type="reward_model_terminal_v1",
                checkpoint_role="actor",
            )
        },
    )
    import megatron.core

    monkeypatch.setattr(megatron.core, "dist_checkpointing", fake_dist_checkpointing)
    values = dict(no_load_optim=False, no_load_rng=False, finetune=False, reset_optimizer_states=False)
    values[flag] = True
    args = SimpleNamespace(loss_type="sft", sft_objective="reward_model", hidden_size=1024, **values)
    with pytest.raises(RuntimeError, match="must restore optimizer, scheduler, and RNG"):
        checkpoint_module._validate_checkpoint_contract(args, [SimpleNamespace(role="actor")], tmp_path)


def test_checkpoint_wrapper_restores_optimizer_scheduler_rng_and_next_step_loss(monkeypatch, tmp_path):
    torch.manual_seed(7)
    source = torch.nn.Linear(3, 1)
    source_optimizer = torch.optim.AdamW(source.parameters(), lr=0.01)
    source_scheduler = torch.optim.lr_scheduler.StepLR(source_optimizer, step_size=1, gamma=0.5)

    def step(model, optimizer, scheduler):
        inputs = torch.randn(4, 3)
        targets = torch.randn(4, 1)
        optimizer.zero_grad()
        loss = torch.nn.functional.mse_loss(model(inputs), targets)
        loss.backward()
        optimizer.step()
        scheduler.step()
        return loss.detach()

    step(source, source_optimizer, source_scheduler)
    saved_model = copy.deepcopy(source.state_dict())
    saved_optimizer = copy.deepcopy(source_optimizer.state_dict())
    saved_scheduler = copy.deepcopy(source_scheduler.state_dict())
    saved_rng = torch.get_rng_state().clone()
    expected_loss = step(source, source_optimizer, source_scheduler)
    expected_parameters = copy.deepcopy(source.state_dict())

    resumed = torch.nn.Linear(3, 1)
    resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=0.01)
    resumed_scheduler = torch.optim.lr_scheduler.StepLR(resumed_optimizer, step_size=1, gamma=0.5)

    def fake_load_checkpoint_megatron(*, ddp_model, optimizer, opt_param_scheduler, **_):
        ddp_model[0].load_state_dict(saved_model)
        optimizer.load_state_dict(saved_optimizer)
        opt_param_scheduler.load_state_dict(saved_scheduler)
        torch.set_rng_state(saved_rng)
        return 1, 0

    monkeypatch.setattr(checkpoint_module, "get_args", lambda: SimpleNamespace(load=str(tmp_path)))
    monkeypatch.setattr(checkpoint_module, "_is_dir_nonempty", lambda _: True)
    monkeypatch.setattr(checkpoint_module, "is_megatron_checkpoint", lambda _: True)
    monkeypatch.setattr(checkpoint_module, "_checkpoint_iteration_dir", lambda *_: tmp_path)
    monkeypatch.setattr(checkpoint_module, "_validate_checkpoint_contract", lambda *_: None)
    monkeypatch.setattr(checkpoint_module, "_load_checkpoint_megatron", fake_load_checkpoint_megatron)

    checkpoint_module.load_checkpoint([resumed], resumed_optimizer, resumed_scheduler, None, False)
    actual_loss = step(resumed, resumed_optimizer, resumed_scheduler)

    torch.testing.assert_close(actual_loss, expected_loss)
    assert resumed_scheduler.state_dict() == source_scheduler.state_dict()
    for name, parameter in resumed.state_dict().items():
        torch.testing.assert_close(parameter, expected_parameters[name])


def test_critic_rejects_reward_model_metadata(monkeypatch, tmp_path):
    fake_dist_checkpointing = SimpleNamespace(
        check_is_distributed_checkpoint=lambda path: True,
        load_common_state_dict=lambda path: {
            "args": SimpleNamespace(
                sft_objective="reward_model",
                head_type="reward_model_terminal_v1",
                checkpoint_role="actor",
            )
        },
    )
    import megatron.core

    monkeypatch.setattr(megatron.core, "dist_checkpointing", fake_dist_checkpointing)
    with pytest.raises(RuntimeError, match="PPO critic load rejects"):
        checkpoint_module._validate_checkpoint_contract(SimpleNamespace(), [SimpleNamespace(role="critic")], tmp_path)


@pytest.mark.parametrize("objective", ["causal_lm", "reward_model"])
def test_checkpoint_missing_args_only_adds_rm_contract_restriction(monkeypatch, tmp_path, objective):
    import megatron.core

    fake_dist_checkpointing = SimpleNamespace(
        check_is_distributed_checkpoint=lambda path: True,
        load_common_state_dict=lambda path: {},
    )
    monkeypatch.setattr(megatron.core, "dist_checkpointing", fake_dist_checkpointing)
    args = SimpleNamespace(loss_type="sft", sft_objective=objective)
    model = [SimpleNamespace(role="actor")]
    if objective == "reward_model":
        with pytest.raises(RuntimeError, match="missing saved args"):
            checkpoint_module._validate_checkpoint_contract(args, model, tmp_path)
    else:
        checkpoint_module._validate_checkpoint_contract(args, model, tmp_path)
