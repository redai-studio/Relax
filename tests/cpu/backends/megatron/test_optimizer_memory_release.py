# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from argparse import Namespace
from datetime import timedelta
from time import monotonic
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest


pytest.importorskip("megatron.core", exc_type=ImportError)


def _expert_adapter_finalize_worker(rank: int, rendezvous: str) -> None:
    import torch
    import torch.distributed as dist
    from megatron.bridge.peft.utils import mark_expert_parallel_replicated

    from relax.backends.megatron import model as module

    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=60))
    try:
        singletons = [dist.new_group([index], backend="gloo") for index in range(2)]
        groups = SimpleNamespace(
            ep=dist.group.WORLD,
            dp_cp=dist.group.WORLD,
            tp=singletons[rank],
            pp=singletons[rank],
            embd=None,
            pos_embd=None,
        )

        class Chunk(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.fused = torch.nn.Parameter(torch.ones(1))
                self.eager = torch.nn.Parameter(torch.ones(1))
                self.unshared = torch.nn.Parameter(torch.ones(1))
                self.config = SimpleNamespace(
                    timers=None,
                    moe_router_enable_expert_bias=False,
                    moe_router_load_balancing_type="none",
                    sequence_parallel=False,
                    qk_layernorm=False,
                    mtp_num_layers=None,
                )

            def finish_grad_sync(self, **kwargs) -> None:
                # The DDP boundary is already complete; exercise real MCore
                # token normalization and Bridge EP reduction below.
                pass

            def scale_gradients(self, scaling: torch.Tensor) -> None:
                for parameter in self.parameters():
                    grad = getattr(parameter, "main_grad", parameter.grad)
                    if grad is not None:
                        grad.mul_(scaling)

        class Configured(Exception):
            pass

        for level in (0, 1):
            chunk = Chunk()
            mark_expert_parallel_replicated(chunk.fused, ep_group=groups.ep)
            mark_expert_parallel_replicated(chunk.eager, ep_group=groups.ep)
            args = Namespace(
                lora_rank=16, empty_unused_memory_level=level, overlap_grad_reduce=False, overlap_param_gather=False
            )
            with (
                patch.object(module, "get_args", return_value=args),
                patch.object(module.mpu, "get_expert_model_parallel_group", return_value=groups.ep),
                patch.object(module, "resident_sft_param_overlap_enabled", side_effect=Configured),
                patch.object(module.device_module, "empty_cache"),
                patch.object(module, "available_memory", return_value={}),
            ):
                # Repeated rollout setup must not stack reducers or retain
                # per-layer fallback hooks (which double-count eager grads).
                for _ in range(2):
                    try:
                        module.train(0, [chunk], SimpleNamespace(scale_loss=lambda x: x), None, [[]], [])
                    except Configured:
                        pass
                chunk.fused.main_grad = torch.tensor([float(rank + 1)])
                chunk.fused.grad_added_to_main_grad = True
                (chunk.eager * (rank + 3) + chunk.unshared * (rank + 5)).sum().backward()
                chunk.config.finalize_model_grads_func([chunk], torch.tensor(rank + 2), pg_collection=groups)
            torch.testing.assert_close(chunk.fused.main_grad, torch.tensor([3 / 5]))
            torch.testing.assert_close(chunk.eager.grad, torch.tensor([7 / 5]))
            torch.testing.assert_close(chunk.unshared.grad, torch.tensor([(rank + 5) / 5]))
            chunk.fused.grad = chunk.fused.main_grad
            torch.optim.SGD([chunk.fused, chunk.eager], lr=0.1).step()
            values = torch.cat([chunk.fused.detach(), chunk.eager.detach()])
            copies = [torch.empty_like(values) for _ in range(2)]
            dist.all_gather(copies, values, group=groups.ep)
            torch.testing.assert_close(copies[0], copies[1], rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


def test_lora_finalize_sums_fused_and_eager_expert_grads_once(tmp_path):
    """Two CPU ranks cover EP synchronization absent from metadata-only
    tests."""
    import torch.multiprocessing as mp

    context = mp.spawn(
        _expert_adapter_finalize_worker,
        args=((tmp_path / "gloo-rendezvous").as_uri(),),
        nprocs=2,
        join=False,
    )
    deadline = monotonic() + 120
    try:
        while not context.join(timeout=max(0, deadline - monotonic())):
            if monotonic() >= deadline:
                raise TimeoutError("expert adapter gradient regression exceeded 120 seconds")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)


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


@pytest.mark.parametrize(
    ("mode", "offload_master"),
    [
        ("legacy", None),
        ("disabled", None),
        ("zero_fraction", None),
        ("enabled", True),
        ("pre_forward_sync", False),
        ("mxfp8_overlap", False),
        ("mxfp8_no_overlap", True),
    ],
)
def test_train_one_step_offloads_between_updates_before_forward(monkeypatch, mode, offload_master):
    from relax.backends.megatron import model as module

    args = Namespace(
        empty_unused_memory_level=0,
        custom_megatron_before_train_step_hook_path=None,
        ci_test=False,
        seq_length=8,
        micro_batch_size=1,
        decoder_seq_length=None,
        reuse_grad_buf_for_mxfp8_param_ag=mode.startswith("mxfp8"),
        overlap_param_gather=mode == "mxfp8_overlap",
    )
    monkeypatch.setattr(module, "get_args", lambda: args)
    monkeypatch.setattr(module.capture_hooks, "begin_step_for", Mock())
    monkeypatch.setattr(module.capture_hooks, "end_step_for", Mock())
    monkeypatch.setattr(module, "maybe_verify_critic_value_head_movement", Mock())
    monkeypatch.setattr(module.mpu, "is_pipeline_last_stage", lambda **_: False)
    monkeypatch.setattr(module, "_is_global_zero_token_step", lambda _: False)

    events = []
    master_location = "cuda"

    def offload(*, offload_master):
        nonlocal master_location
        events.append(("offload", offload_master))
        if offload_master:
            master_location = "cpu"

    def forward_backward(**_):
        expected = [("offload", offload_master)] if offload_master is not None else []
        assert events == expected + ["clear_model_grads", "clear_optimizer_grads"]
        assert master_location == ("cpu" if offload_master else "cuda")
        events.append("backward")
        return []

    def step():
        nonlocal master_location
        # The real optimizer reloads masters for the update. The next training
        # iteration must offload them again, not only on initial setup.
        master_location = "cuda"
        events.append("optimizer")
        return True, 2.0, 0

    monkeypatch.setattr(module, "get_forward_backward_func", lambda: forward_backward)
    offload_call = Mock(side_effect=offload)
    optimizer = SimpleNamespace(
        zero_grad=lambda: events.append("clear_optimizer_grads"),
        step=step,
        offload_optimizer_state_for_forward=offload_call,
        optimizer_state_offload_requires_pre_forward_param_sync=lambda: mode == "pre_forward_sync",
    )
    if mode != "legacy":
        optimizer.config = SimpleNamespace(
            chunked_optimizer_state_offload=mode != "disabled",
            optimizer_state_offload_fraction=0.0 if mode == "zero_fraction" else 1.0,
        )
    scheduler = SimpleNamespace(step=Mock())
    model = SimpleNamespace(zero_grad_buffer=lambda: events.append("clear_model_grads"))
    for step_id in range(2):
        events.clear()
        assert module.train_one_step(args, 1, step_id, [], [model], optimizer, scheduler, 1, 8) == ({}, 2.0)
    assert offload_call.call_count == (0 if offload_master is None else 2)
    assert scheduler.step.call_count == 2
