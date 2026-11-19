# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Complete-group transfer and CPU pixel ownership across the queue
boundary."""

import asyncio
import copy
import weakref
from argparse import Namespace
from types import SimpleNamespace

import pytest
import torch

from relax.agentic.pipeline import GroupExport, SampleExport, SessionExport, transfer
from relax.utils import env
from relax.utils.types import Sample


def _sample():
    return Sample(
        index=0,
        session_id="session",
        tokens=[1, 2],
        response_length=1,
        rollout_log_probs=[-0.1],
        custom_advantage=0.5,
        multimodal_train_inputs={
            "pixel_values": torch.ones(4, 4, dtype=torch.bfloat16),
            "image_grid_thw": torch.tensor([[1, 2, 2]]),
        },
    )


def test_optimization_flags_pass_strict_env_validation(monkeypatch):
    monkeypatch.setattr(env, "os", SimpleNamespace(environ={}))
    flags = {
        "RELAX_STRICT_ENV": "1",
        "RELAX_VCS": "1",
        "RELAX_VCS_KEEP_IMAGES": "1",
        "RELAX_VCS_PRUNE_IMAGES": "2",
        "RELAX_TRAIN_VISION_CACHE": "1",
        "RELAX_TRAIN_VISION_CACHE_MAX_BYTES": "268435456",
        "RELAX_TRANSFER_BATCH_GROUPS": "1",
        "RELAX_MM_RELEASE_AFTER_TRANSFER": "1",
        "RELAX_MM_RELEASE_TRIM": "1",
        "RELAX_MM_RELEASE_LOG": "1",
        "RELAX_MM_PIXEL_DTYPE": "bf16",
    }
    env.validate_env(flags)
    assert env.resolve_in("RELAX_TRANSFER_BATCH_GROUPS", flags) == 1
    assert env.resolve_in("RELAX_MM_RELEASE_AFTER_TRANSFER", flags) is True
    assert env.resolve_in("RELAX_TRAIN_VISION_CACHE", {}) is False
    assert env.resolve_in("RELAX_VCS", {}) is False
    assert env.resolve_in("RELAX_TRANSFER_BATCH_GROUPS", {}) == 0
    assert env.resolve_in("RELAX_MM_RELEASE_AFTER_TRANSFER", {}) is False


@pytest.mark.parametrize(
    "fully_async,threshold,ready", [(False, 0, False), (False, 1, True), (False, 2, False), (True, 1, False)]
)
def test_complete_group_can_transfer_before_rollout_is_full(monkeypatch, fully_async, threshold, ready):
    monkeypatch.setattr(transfer, "_TRANSFER_BATCH_GROUPS", threshold)
    domain = transfer.TransferDomain(
        args=Namespace(
            fully_async=fully_async,
            use_dynamic_global_batch_size=False,
            rollout_batch_size=8,
            global_batch_size=64,
            num_iters_per_train_update=1,
            n_samples_per_prompt=8,
        )
    )
    domain.open_partition(3, target_groups=8)
    group = GroupExport(
        group_id="task", sessions=(SessionExport(exports=(SampleExport(name="context", sample=_sample()),)),)
    )
    domain.accept(group)
    batches = domain.detach_ready_batches()
    assert bool(batches) is ready
    assert domain.total_debt == 7
    if ready:
        partition_id, groups, is_last = batches[0]
        assert partition_id == 3 and groups == (group,) and is_last is False


@pytest.mark.parametrize("outcome", ["success", "failure", "cancel"])
def test_pixels_release_only_after_successful_queue_write(monkeypatch, outcome):
    monkeypatch.setenv("RELAX_TRAIN_VISION_CACHE", "1")
    monkeypatch.setattr(transfer, "_MM_RELEASE_AFTER_TRANSFER", True)
    monkeypatch.setattr(transfer, "_MM_RELEASE_TRIM", False)
    sample = _sample()
    grid = sample.multimodal_train_inputs["image_grid_thw"]
    args = Namespace(
        custom_reward_post_process_path=None,
        agentic_custom_advantage_path="unused:advantage",
        advantage_estimator="grpo",
        debug_train_only=False,
        multimodal_keys=None,
        use_opd=False,
    )

    async def run():
        entered, allow_finish = asyncio.Event(), asyncio.Event()
        stored = {}

        async def put(**kwargs):
            # Model a successful transport serialization without retaining the source tensors.
            stored.update(copy.deepcopy(kwargs))
            entered.set()
            await allow_finish.wait()
            if outcome == "failure":
                raise RuntimeError("queue unavailable")

        task = asyncio.create_task(
            transfer._transfer_batch_to_data_system(
                args=args,
                batch_samples=[[sample]],
                partition_id=3,
                data_system_client=SimpleNamespace(async_put=put),
            )
        )
        try:
            await asyncio.wait_for(entered.wait(), 5)
            assert sample.multimodal_train_inputs["pixel_values"].shape == (4, 4)
            if outcome == "cancel":
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            else:
                allow_finish.set()
                if outcome == "failure":
                    with pytest.raises(RuntimeError, match="queue unavailable"):
                        await task
                else:
                    await task
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
        return stored

    stored = asyncio.run(run())
    mm = sample.multimodal_train_inputs
    assert mm["pixel_values"].shape == ((0, 4) if outcome == "success" else (4, 4))
    assert mm["pixel_values"].dtype == torch.bfloat16
    assert mm["image_grid_thw"] is grid
    assert sample.tokens == [1, 2] and sample.rollout_log_probs == [-0.1]
    transferred = stored["data"]["multimodal_train_inputs"][0]
    assert transferred["pixel_values"].shape == (4, 4)
    assert transferred["_relax_vision_cache"][0]["key"][0] == "session"
    assert stored["partition_id"] == "train_3"


def test_pixel_reference_is_gone_before_malloc_trim(monkeypatch):
    sample = _sample()
    old_pixels = weakref.ref(sample.multimodal_train_inputs["pixel_values"])
    calls = []

    def trim(unused):
        assert old_pixels() is None
        calls.append("trim")

    monkeypatch.setattr(transfer, "_MM_RELEASE_AFTER_TRANSFER", True)
    monkeypatch.setattr(transfer, "_MM_RELEASE_TRIM", True)
    monkeypatch.setattr(transfer, "_MM_RELEASE_LOG", False)
    monkeypatch.setattr(transfer, "_MM_LIBC", SimpleNamespace(malloc_trim=trim))
    assert transfer._release_transferred_pixels([sample]) == 32
    assert calls == ["trim"]


def test_pixel_release_is_disabled_by_default(monkeypatch):
    sample = _sample()
    monkeypatch.setattr(transfer, "_MM_RELEASE_AFTER_TRANSFER", False)
    assert transfer._release_transferred_pixels([sample]) == 0
    assert sample.multimodal_train_inputs["pixel_values"].shape == (4, 4)
