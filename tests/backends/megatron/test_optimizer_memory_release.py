# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


pytest.importorskip("megatron.core", exc_type=ImportError)


def test_gradient_finalization_releases_cache_before_collectives(monkeypatch):
    from relax.backends.megatron import model as module

    events = []
    model, num_tokens, pg_collection = object(), object(), object()
    monkeypatch.setattr(module.device_module, "empty_cache", lambda: events.append("release"))
    monkeypatch.setattr(module, "available_memory", lambda: {})

    def finalize(actual_model, actual_tokens, *, pg_collection, force_all_reduce):
        assert events == ["release"]
        assert actual_model is model
        assert actual_tokens is num_tokens
        assert pg_collection is expected_group
        assert force_all_reduce is True
        events.append("collective")

    expected_group = pg_collection
    monkeypatch.setattr(module, "finalize_model_grads", finalize)
    module._finalize_model_grads_with_memory_release(
        model, num_tokens, pg_collection=pg_collection, force_all_reduce=True
    )
    assert events == ["release", "collective"]


@pytest.mark.parametrize("level", [0, 1, 2])
def test_train_one_step_releases_cache_before_optimizer_collectives(monkeypatch, level):
    from relax.backends.megatron import model as module

    events = []
    args = Namespace(
        empty_unused_memory_level=level,
        custom_megatron_before_train_step_hook_path=None,
        ci_test=False,
        seq_length=8,
        micro_batch_size=1,
        decoder_seq_length=None,
    )
    monkeypatch.setattr(module, "get_args", lambda: args)
    monkeypatch.setattr(module.capture_hooks, "begin_step_for", Mock())
    monkeypatch.setattr(module.capture_hooks, "end_step_for", Mock())
    monkeypatch.setattr(module, "maybe_verify_critic_value_head_movement", Mock())
    monkeypatch.setattr(module.mpu, "is_pipeline_last_stage", lambda **_: False)
    monkeypatch.setattr(module, "_is_global_zero_token_step", lambda _: False)
    monkeypatch.setattr(module.device_module, "empty_cache", lambda: events.append("release"))
    memory_probe = Mock(return_value={"free_GB": 16})
    monkeypatch.setattr(module, "available_memory", memory_probe)

    def forward_backward(**_):
        events.append("backward")
        return []

    def optimizer_step():
        # NCCL norm reduction happens within this single optimizer step.
        expected = ["clear_model_grads", "clear_optimizer_grads", "backward"]
        assert events == expected + (["release"] if level else [])
        events.append("optimizer")
        return True, 2.0, 0

    monkeypatch.setattr(module, "get_forward_backward_func", lambda: forward_backward)
    optimizer = SimpleNamespace(zero_grad=lambda: events.append("clear_optimizer_grads"), step=optimizer_step)
    scheduler = SimpleNamespace(step=lambda **_: events.append("scheduler"))
    model = SimpleNamespace(zero_grad_buffer=lambda: events.append("clear_model_grads"))
    result = module.train_one_step(args, 1, 0, [], [model], optimizer, scheduler, 1, 8)
    expected = ["clear_model_grads", "clear_optimizer_grads", "backward"]
    expected += (["release"] if level else []) + ["optimizer", "scheduler"]
    expected.extend(["clear_model_grads", "clear_optimizer_grads"])
    if level >= 2:
        expected.append("release")
    assert events == expected
    assert result == ({}, 2.0)
    assert memory_probe.call_count == (2 if level else 0)
