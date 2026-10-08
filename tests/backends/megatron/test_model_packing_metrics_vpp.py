# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Packing metrics follow the logical final stage under VPP."""

from argparse import Namespace

import pytest
import torch


class _FakeModel:
    role = "actor"

    def __call__(self, **_kwargs) -> torch.Tensor:
        return torch.ones(1)

    def zero_grad_buffer(self) -> None:
        pass


class _FakeOptimizer:
    def zero_grad(self) -> None:
        pass

    def step(self) -> tuple[bool, torch.Tensor, int]:
        return True, torch.tensor(1.0), 0


class _FakeScheduler:
    def step(self, increment: int) -> None:
        assert increment == 3


@pytest.mark.parametrize("num_model_chunks", [1, 2])
def test_train_one_step_counts_each_logical_microbatch_once(monkeypatch, num_model_chunks):
    model_module = pytest.importorskip("relax.backends.megatron.model")

    args = Namespace(
        allgather_cp=False,
        calculate_per_token_loss=False,
        check_for_nan_in_loss_and_grad=True,
        ci_test=False,
        custom_megatron_before_train_step_hook_path=None,
        data_pad_size_multiplier=1,
        decoder_seq_length=None,
        dynamic_context_parallel=False,
        enable_mtp_training=False,
        is_vl_model=False,
        mtp_only_training=False,
        micro_batch_size=1,
        qkv_format="thd",
        seq_length=8,
        task_type="causal_lm",
        use_opd=False,
        use_rollout_indexer_replay=False,
        uses_unsplit_forward=False,
    )
    batch = {
        "full_loss_masks": None,
        "packed_seq_params": None,
        "tokens": torch.ones(1, 8),
    }
    num_microbatches = 3
    models = [_FakeModel() for _ in range(num_model_chunks)]
    data_iterators = [iter(()) for _ in range(num_model_chunks)]

    monkeypatch.setattr(model_module, "get_args", lambda: args)
    monkeypatch.setattr(model_module, "get_batch", lambda *_args, **_kwargs: dict(batch))
    monkeypatch.setattr(model_module, "should_bypass_main_output_layer", lambda _args: False)
    monkeypatch.setattr(model_module.capture_hooks, "begin_step_for", lambda *_args: None)
    monkeypatch.setattr(model_module.capture_hooks, "end_step_for", lambda: None)
    monkeypatch.setattr(model_module, "maybe_verify_critic_value_head_movement", lambda *_args: None)
    monkeypatch.setattr(model_module.mpu, "get_context_parallel_world_size", lambda: 1)
    monkeypatch.setattr(model_module.mpu, "get_pipeline_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(model_module.torch.cuda, "current_device", lambda: "cpu")
    monkeypatch.setattr(model_module.mpu, "get_data_parallel_group", lambda **_kwargs: object())
    monkeypatch.setattr(model_module.mpu, "is_pipeline_last_stage", lambda **_kwargs: True)
    monkeypatch.setattr(model_module.torch.distributed, "all_reduce", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(model_module.torch.distributed, "get_world_size", lambda _group: 1)

    def fake_loss_function(_args, _batch, _num_microbatches, logits, **_kwargs):
        return (
            logits.sum() * 0,
            1,
            {"keys": ["loss"], "values": torch.tensor([0.0, 1.0]), "num_tokens": torch.tensor(1)},
        )

    monkeypatch.setattr(model_module, "loss_function", fake_loss_function)

    def fake_forward_backward_func(**kwargs):
        losses = []
        for _ in range(num_microbatches):
            final_output = None
            final_loss_callback = None
            for chunk_id in range(num_model_chunks):
                output, loss_callback = kwargs["forward_step_func"](data_iterators[chunk_id], models[chunk_id])
                if chunk_id == num_model_chunks - 1:
                    final_output = output
                    final_loss_callback = loss_callback
            assert final_output is not None
            assert final_loss_callback is not None
            _, _, loss = final_loss_callback(final_output)
            losses.append(loss)
        return losses

    monkeypatch.setattr(model_module, "get_forward_backward_func", lambda: fake_forward_backward_func)

    metrics, _ = model_module.train_one_step(
        args,
        rollout_id=0,
        step_id=0,
        data_iterator=data_iterators,
        model=models,
        optimizer=_FakeOptimizer(),
        opt_param_scheduler=_FakeScheduler(),
        num_microbatches=num_microbatches,
        step_global_batch_size=3,
    )

    assert metrics["num_microbatches_mean"] == num_microbatches
    assert metrics["pack_tokens_mean"] == 8
    assert metrics["pack_tokens_max"] == 8
