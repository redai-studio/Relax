# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Token-budget polling must leave the shared controller available to
writers."""

import pytest


@pytest.mark.parametrize("state", ["missing", "pending", "dummy", "drained", "window_drained"])
def test_token_budget_empty_fetch_returns_without_blocking_request_thread(monkeypatch, state):
    module = pytest.importorskip("transfer_queue.controller")
    controller_type = module.TransferQueueController.__ray_metadata__.modified_class
    controller = object.__new__(controller_type)
    partition = module.DataPartitionStatus("train_0")
    partition.production_completed = state == "drained"
    if state == "drained":
        import torch

        partition.actual_sample_count = 1
        partition.global_indexes.add(0)
        partition.consumption_status["train"] = torch.ones(1, dtype=torch.int8)
    controller.partitions = {} if state == "missing" else {"train_0": partition}

    class Sampler:
        def __call__(self, *_args, **_kwargs):
            return [], []

        def is_dummy_round(self, *_args):
            return state == "dummy"

        def is_window_drained(self, *_args, **_kwargs):
            return state == "window_drained"

    controller.sampler = Sampler()
    controller.scan_data_status = lambda *_args: []

    def blocked_request_thread(_seconds):
        pytest.fail("An empty fetch blocked the controller's shared request thread")

    monkeypatch.setattr(module.time, "sleep", blocked_request_thread)
    meta = controller.get_metadata(
        data_fields=["tokens"],
        partition_id="train_0",
        mode="fetch",
        task_name="train",
        token_budget=128,
        sampling_config={"dp_rank": 0, "batch_index": 0, "rollout_mini_index": 0, "window_quota": 128},
    )
    assert meta.size == 0
    expected = (
        {"dummy_round": True}
        if state == "dummy"
        else ({"stream_end": True} if state in {"drained", "window_drained"} else {})
    )
    assert (meta.extra_info or {}) == expected
