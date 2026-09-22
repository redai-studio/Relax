# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Zero-token no-signal step handling for the shared Megatron trainer."""

from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("megatron.core")


@pytest.fixture()
def model_module(monkeypatch):
    from relax.backends.megatron import model as model_module

    monkeypatch.setattr(torch.cuda, "current_device", lambda: "cpu")
    return model_module


def _two_microbatch_losses(zero: bool) -> list[dict[str, object]]:
    """Two microbatches with explicit CP-local effective token counts."""
    return (
        [
            {"values": torch.tensor([0.0, 1.0, 2.0]), "num_tokens": torch.tensor(0.0)},
            {"values": torch.tensor([0.0, 3.0]), "num_tokens": torch.tensor(0.0)},
        ]
        if zero
        else [
            {"values": torch.tensor([4.0, 1.0, 2.0]), "num_tokens": torch.tensor(4.0)},
            {"values": torch.tensor([6.0, 3.0]), "num_tokens": torch.tensor(6.0)},
        ]
    )


def _patch_mpu(
    monkeypatch,
    model_module,
    *,
    pp_size: int,
    is_last_stage: bool,
    all_reduce_impl,
    broadcast_impl,
):
    mpu = model_module.mpu
    dist = model_module.torch.distributed
    monkeypatch.setattr(mpu, "is_pipeline_last_stage", lambda ignore_virtual=False: is_last_stage)
    monkeypatch.setattr(mpu, "get_data_parallel_group", lambda with_context_parallel=False: object())
    monkeypatch.setattr(mpu, "get_pipeline_model_parallel_world_size", lambda: pp_size)
    monkeypatch.setattr(mpu, "get_pipeline_model_parallel_group", lambda: object())
    monkeypatch.setattr(dist, "all_reduce", all_reduce_impl)
    monkeypatch.setattr(dist, "broadcast", broadcast_impl)


def test_is_global_zero_token_step_true_pp1_no_broadcast(model_module, monkeypatch):
    """PP=1: every rank is the last stage, so the all-reduced count is already
    consistent locally and no pipeline broadcast is required."""
    calls = {"all_reduce": 0, "broadcast": 0}

    def all_reduce(tensor, group=None):
        calls["all_reduce"] += 1
        # Zero token count: leave the tensor as-is.

    def broadcast(tensor, src=0, group=None):
        calls["broadcast"] += 1

    _patch_mpu(
        monkeypatch,
        model_module,
        pp_size=1,
        is_last_stage=True,
        all_reduce_impl=all_reduce,
        broadcast_impl=broadcast,
    )
    assert model_module._is_global_zero_token_step(_two_microbatch_losses(zero=True)) is True
    assert calls == {"all_reduce": 1, "broadcast": 0}


def test_is_global_zero_token_step_false_pp1_no_broadcast(model_module, monkeypatch):
    """PP=1 with a nonzero token count: the local all-reduced count is
    nonzero."""
    calls = {"all_reduce": 0, "broadcast": 0}

    def all_reduce(tensor, group=None):
        calls["all_reduce"] += 1
        tensor.fill_(10)  # nonzero global token count

    def broadcast(tensor, src=0, group=None):
        calls["broadcast"] += 1

    _patch_mpu(
        monkeypatch,
        model_module,
        pp_size=1,
        is_last_stage=True,
        all_reduce_impl=all_reduce,
        broadcast_impl=broadcast,
    )
    assert model_module._is_global_zero_token_step(_two_microbatch_losses(zero=False)) is False
    assert calls == {"all_reduce": 1, "broadcast": 0}


def test_is_global_zero_token_step_last_stage_pp2_broadcasts(model_module, monkeypatch):
    """PP>1: the last stage reduces the count then broadcasts the decision."""
    calls = {"all_reduce": 0, "broadcast": 0}

    def all_reduce(tensor, group=None):
        calls["all_reduce"] += 1

    def broadcast(tensor, src=0, group=None):
        calls["broadcast"] += 1
        assert src == 1  # pp_size - 1

    _patch_mpu(
        monkeypatch,
        model_module,
        pp_size=2,
        is_last_stage=True,
        all_reduce_impl=all_reduce,
        broadcast_impl=broadcast,
    )
    assert model_module._is_global_zero_token_step(_two_microbatch_losses(zero=True)) is True
    assert calls == {"all_reduce": 1, "broadcast": 1}


def test_is_global_zero_token_step_non_last_stage_pp2_enters_broadcast(model_module, monkeypatch):
    """PP>1: a non-last stage skips the all-reduce but must join the broadcast
    so the collective completes."""
    calls = {"all_reduce": 0, "broadcast": 0}

    def all_reduce(tensor, group=None):
        calls["all_reduce"] += 1

    def broadcast(tensor, src=0, group=None):
        calls["broadcast"] += 1
        assert src == 1

    _patch_mpu(
        monkeypatch,
        model_module,
        pp_size=2,
        is_last_stage=False,
        all_reduce_impl=all_reduce,
        broadcast_impl=broadcast,
    )
    assert model_module._is_global_zero_token_step(_two_microbatch_losses(zero=True)) is False
    assert calls == {"all_reduce": 0, "broadcast": 1}


def _make_args() -> SimpleNamespace:
    return SimpleNamespace(
        custom_megatron_before_train_step_hook_path=None,
        ci_test=False,
        enable_mtp_training=False,
        check_for_nan_in_loss_and_grad=True,
        global_batch_size=32,
        dynamic_context_parallel=False,
        use_dynamic_batch_size=False,
        fully_async=False,
        seq_length=512,
        micro_batch_size=1,
        decoder_seq_length=512,
        attention_backend="flash",
    )


def test_train_one_step_skips_optimizer_and_scheduler_on_zero_token_step(model_module, monkeypatch):
    """A global zero-token step must not update parameters or the LR
    scheduler."""
    from megatron.core import mpu

    calls = {"optimizer_step": 0, "scheduler_step": 0}

    optimizer = SimpleNamespace(
        step=lambda: calls.__setitem__("optimizer_step", calls["optimizer_step"] + 1) or (True, 0.0, 0),
        zero_grad=lambda: None,
        get_loss_scale=lambda: SimpleNamespace(item=lambda: 1.0),
        param_groups=[],
    )
    scheduler = SimpleNamespace(
        step=lambda increment=None: calls.__setitem__("scheduler_step", calls["scheduler_step"] + 1)
    )
    model = [SimpleNamespace(zero_grad_buffer=lambda: None)]

    losses_reduced = _two_microbatch_losses(zero=True)

    monkeypatch.setattr(model_module, "_is_global_zero_token_step", lambda losses: True)
    monkeypatch.setattr(model_module, "get_args", lambda: _make_args())
    monkeypatch.setattr(model_module, "get_forward_backward_func", lambda: lambda **_kwargs: losses_reduced)
    monkeypatch.setattr(model_module, "maybe_verify_critic_value_head_movement", lambda *a, **k: None)
    monkeypatch.setattr(mpu, "is_pipeline_last_stage", lambda ignore_virtual=False: False)
    monkeypatch.setattr(mpu, "get_virtual_pipeline_model_parallel_world_size", lambda: None)

    loss_reduced, grad_norm = model_module.train_one_step(
        args=_make_args(),
        rollout_id=0,
        step_id=3,
        data_iterator=[[]],
        model=model,
        optimizer=optimizer,
        opt_param_scheduler=scheduler,
        num_microbatches=1,
        step_global_batch_size=32,
    )

    assert calls["optimizer_step"] == 0
    assert calls["scheduler_step"] == 0
    assert grad_norm == 0.0
    assert loss_reduced == {}


@pytest.mark.parametrize("zero_tokens", [False, True])
@pytest.mark.parametrize("calculate_per_token_loss", [False, True])
def test_train_one_step_preserves_logical_batch_and_capture(
    model_module, monkeypatch, zero_tokens, calculate_per_token_loss
):
    args = _make_args()
    args.calculate_per_token_loss = calculate_per_token_loss
    calls = {"optimizer": 0, "scheduler": [], "capture_begin": 0, "capture_end": 0}
    token_count = 0.0 if zero_tokens else 4.0
    numerator = 0.0 if zero_tokens else 12.0
    losses = [
        {
            "keys": ["loss"],
            "values": torch.tensor([token_count if calculate_per_token_loss else 0.0, numerator]),
            "num_tokens": torch.tensor(token_count),
        }
    ]
    optimizer = SimpleNamespace(
        step=lambda: calls.__setitem__("optimizer", calls["optimizer"] + 1) or (True, 1.0, 0),
        zero_grad=lambda: None,
        param_groups=[],
    )
    scheduler = SimpleNamespace(step=lambda increment: calls["scheduler"].append(increment))
    monkeypatch.setattr(model_module, "get_args", lambda: args)
    monkeypatch.setattr(model_module, "get_forward_backward_func", lambda: lambda **_kwargs: losses)
    monkeypatch.setattr(model_module, "maybe_verify_critic_value_head_movement", lambda *a, **k: None)
    monkeypatch.setattr(model_module.mpu, "get_virtual_pipeline_model_parallel_world_size", lambda: None)
    _patch_mpu(
        monkeypatch,
        model_module,
        pp_size=1,
        is_last_stage=True,
        all_reduce_impl=lambda tensor, group=None: None,
        broadcast_impl=lambda tensor, src, group=None: pytest.fail("PP=1 must not broadcast"),
    )
    monkeypatch.setattr(
        model_module.capture_hooks,
        "begin_step_for",
        lambda *a: calls.__setitem__("capture_begin", calls["capture_begin"] + 1),
    )
    monkeypatch.setattr(
        model_module.capture_hooks,
        "end_step_for",
        lambda: calls.__setitem__("capture_end", calls["capture_end"] + 1),
    )

    metrics, grad_norm = model_module.train_one_step(
        args=args,
        rollout_id=0,
        step_id=3,
        data_iterator=[[]],
        model=[SimpleNamespace(zero_grad_buffer=lambda: None)],
        optimizer=optimizer,
        opt_param_scheduler=scheduler,
        num_microbatches=1,
        step_global_batch_size=2,
    )

    assert calls["optimizer"] == (0 if zero_tokens else 1)
    assert calls["scheduler"] == ([] if zero_tokens else [2])
    assert calls["capture_begin"] == calls["capture_end"] == 1
    assert metrics == {"loss": 0.0 if zero_tokens else (3.0 if calculate_per_token_loss else 6.0)}
    assert grad_norm == (0.0 if zero_tokens else 1.0)
