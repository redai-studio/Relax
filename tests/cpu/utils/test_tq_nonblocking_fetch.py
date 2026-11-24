# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Fully async replay polling must leave the TQ controller available to
writers."""

from types import SimpleNamespace

import pytest

from relax.distributed.replay_manager import core
from relax.distributed.replay_manager.config import ReplayConfig


def _configure_timeout(monkeypatch, *, references=True, replay=True, fully_async=True, hybrid=False, polling=True):
    monkeypatch.setenv("RELAX_REPLAY_REFERENCES", "1" if references else "0")
    monkeypatch.setenv("RELAX_REPLAY_RUN_ID", "abcd")
    manager = core.ReplayManager(config=ReplayConfig.from_env())
    monkeypatch.setattr(core, "replay_data_manager", manager)
    args = SimpleNamespace(
        train_backend="megatron",
        use_rollout_routing_replay=replay,
        colocate=not fully_async,
        fully_async=fully_async,
        hybrid=hybrid,
        polling_mode=polling,
    )
    runtime_env = {"env_vars": {"TQ_CONTROLLER_GET_METADATA_TIMEOUT": "3"}}
    core.configure_training_replay(args, runtime_env)
    return int(runtime_env["env_vars"]["TQ_CONTROLLER_GET_METADATA_TIMEOUT"])


@pytest.mark.parametrize(
    "mode", [{"fully_async": False}, {"hybrid": True}, {"replay": False}, {"references": False}, {"polling": False}]
)
def test_other_tq_consumers_keep_their_configured_timeout(monkeypatch, mode):
    assert _configure_timeout(monkeypatch, **mode) == 3


@pytest.mark.parametrize("state", ["missing", "pending", "dummy", "drained", "window_drained"])
def test_token_budget_empty_fetch_returns_without_blocking_request_thread(monkeypatch, state):
    module = pytest.importorskip("transfer_queue.controller")
    # Ray exports the already-imported driver's controller class to the worker.
    monkeypatch.setattr(module, "TQ_CONTROLLER_GET_METADATA_TIMEOUT", 3)
    worker_timeout = _configure_timeout(monkeypatch)
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

    monkeypatch.setattr(module, "time", SimpleNamespace(time=lambda: 100.0, sleep=blocked_request_thread))
    meta = controller.get_metadata(
        data_fields=["tokens"],
        partition_id="train_0",
        mode="fetch",
        task_name="train",
        token_budget=128,
        sampling_config={"dp_rank": 0, "batch_index": 0, "rollout_mini_index": 0, "window_quota": 128},
    )
    assert meta.size == 0
    assert worker_timeout == -1
    expected = (
        {"dummy_round": True}
        if state == "dummy"
        else ({"stream_end": True} if state in {"drained", "window_drained"} else {})
    )
    assert (meta.extra_info or {}) == expected
