# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Exercise rank agreement with two concurrent CPU ranks and real artifacts."""

import copy
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import ray
import torch.distributed as dist

from relax.engine.lora_versions import training
from relax.engine.lora_versions.artifact import read_adapter_artifact
from relax.engine.lora_versions.training import LoRATrainingConfig, TrainingLoRAPublisher
from relax.utils import distributed_utils
from tests.engine.lora_versions.test_artifact import frozen


class ThreadRanks:
    """Check collective order as well as results, without sockets or GPUs."""

    def __init__(self, monkeypatch):
        self.local = threading.local()
        self.fence = threading.Barrier(2, timeout=5)
        self.rounds = {}
        monkeypatch.setattr(distributed_utils, "get_gloo_group", lambda: self)
        monkeypatch.setattr(dist, "get_rank", lambda group=None: self.local.rank)
        monkeypatch.setattr(dist, "get_world_size", lambda group=None: 2)
        monkeypatch.setattr(dist, "broadcast_object_list", self.broadcast)
        monkeypatch.setattr(dist, "all_gather_object", self.gather)
        monkeypatch.setattr(dist, "barrier", lambda group: self.exchange("barrier", None))

    def exchange(self, kind, value):
        step = self.local.step
        self.local.step += 1
        values = self.rounds.setdefault(step, {})
        values[self.local.rank] = (kind, copy.deepcopy(value))
        self.fence.wait()
        assert {entry[0] for entry in values.values()} == {kind}, "mismatched rank collectives"
        result = [values[rank][1] for rank in range(2)]
        self.fence.wait()
        return result

    def broadcast(self, values, *, src, group):
        assert src == 0 and group is self
        values[:] = self.exchange("broadcast", values)[src]

    def gather(self, values, value, *, group):
        assert group is self
        values[:] = self.exchange("gather", value)

    def run(self, function):
        self.rounds.clear()
        results = [None, None]

        def worker(rank):
            self.local.rank, self.local.step = rank, 0
            try:
                results[rank] = function(rank)
            except Exception as error:
                results[rank] = error

        threads = [threading.Thread(target=worker, args=(rank,), daemon=True) for rank in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(7)
        assert not any(thread.is_alive() for thread in threads), "rank did not leave the collective"
        assert not any(isinstance(result, threading.BrokenBarrierError) for result in results), results
        return results


@pytest.fixture
def ranks(monkeypatch):
    return ThreadRanks(monkeypatch)


@pytest.fixture
def publishers(tmp_path, monkeypatch, ranks):
    monkeypatch.setattr(ray, "get", lambda value, **kwargs: value)
    calls = []

    def publish(path):
        artifact = read_adapter_artifact(path)
        calls.append(artifact)
        return {"status": "published", "version_id": artifact.identity.version_id}

    manager = SimpleNamespace(publish_training_lora=SimpleNamespace(remote=Mock(side_effect=publish)))
    config = LoRATrainingConfig(str(tmp_path), "frozen-base-content", (1,))
    workers = [
        TrainingLoRAPublisher(
            SimpleNamespace(
                export_local_adapter=Mock(side_effect=lambda params: dict(params)),
                gather_full_adapter=Mock(side_effect=lambda params, **kwargs: params),
                config_dict=lambda: frozen().config,
            ),
            config,
            manager,
            10,
        )
        for _ in range(2)
    ]
    return workers, manager.publish_training_lora.remote, calls


def assert_rank_errors(results, message):
    assert all(isinstance(result, RuntimeError) and message in str(result) for result in results), results
    assert str(results[0]) == str(results[1])


def test_training_exports_once_per_rank_and_only_rank_zero_writes(monkeypatch, ranks, publishers):
    workers, rpc, calls = publishers
    writer = Mock(wraps=training.write_adapter_artifact)
    monkeypatch.setattr(training, "write_adapter_artifact", writer)
    params = frozen().copy_tensors()
    result = ranks.run(lambda rank: workers[rank].publish(params))
    assert result == [{"status": "published", "version_id": 0}] * 2
    assert rpc.call_count == writer.call_count == 1
    for worker in workers:
        worker.sync.export_local_adapter.assert_called_once()
        worker.sync.gather_full_adapter.assert_called_once_with(params, all_gather=False)
        assert worker.published_version == 0 and worker.pending_version is None
    identity = calls[0].identity
    for tensor in params.values():
        tensor.zero_()
    assert read_adapter_artifact(calls[0].path).identity == identity


def test_lost_publication_reply_retries_same_artifact_before_new_weights(ranks, publishers):
    workers, rpc, calls = publishers
    publish = rpc.side_effect

    def lost_reply(path):
        publish(path)
        raise TimeoutError("commit reply lost")

    rpc.side_effect = lost_reply
    assert_rank_errors(ranks.run(lambda rank: workers[rank].publish(frozen().copy_tensors())), "unconfirmed")
    assert [worker.pending_version for worker in workers] == [0, 0]
    rpc.side_effect = publish
    assert (
        ranks.run(lambda rank: workers[rank].publish(frozen(value=9).copy_tensors()))
        == [{"status": "published", "version_id": 0}] * 2
    )
    assert calls[0] == calls[1]
    assert all(worker.sync.export_local_adapter.call_count == 1 for worker in workers)
    result = ranks.run(lambda rank: workers[rank].publish(frozen(value=9).copy_tensors()))
    assert result == [{"status": "published", "version_id": 2}] * 2
    assert calls[2].identity.digest != calls[0].identity.digest


def test_capacity_pressure_keeps_old_version_and_next_export_is_fresh(ranks, publishers):
    workers, rpc, calls = publishers
    publish = rpc.side_effect
    ranks.run(lambda rank: workers[rank].publish(frozen().copy_tensors()))
    rpc.side_effect = lambda path: {"status": "capacity_skipped", "default_version": 0}
    results = ranks.run(lambda rank: workers[rank].publish(frozen(value=2).copy_tensors()))
    assert all(result["status"] == "capacity_skipped" for result in results)
    assert all(worker.published_version == 0 and worker.skipped_total == 1 for worker in workers)
    rpc.side_effect = publish
    assert (
        ranks.run(lambda rank: workers[rank].publish(frozen(value=3).copy_tensors()))
        == [{"status": "published", "version_id": 2}] * 2
    )
    assert calls[-1].identity.version_id == 2


@pytest.mark.parametrize("stage", ["export", "gather", "write", "rpc", "initial_capacity", "wrong_version"])
def test_training_failures_are_shared_before_ranks_leave(monkeypatch, ranks, publishers, stage):
    workers, rpc, _calls = publishers
    if stage == "export":
        workers[1].sync.export_local_adapter.side_effect = ValueError("bad export")
    elif stage == "gather":
        workers[1].sync.gather_full_adapter.side_effect = ValueError("bad gather")
    elif stage == "write":
        monkeypatch.setattr(training, "write_adapter_artifact", Mock(side_effect=OSError("disk full")))
    elif stage == "rpc":
        rpc.side_effect = lambda path: {"status": "failed", "error": "load rejected"}
    elif stage == "initial_capacity":
        rpc.side_effect = lambda path: {"status": "capacity_skipped", "default_version": None}
    else:
        rpc.side_effect = lambda path: {"status": "published", "version_id": 99}
    results = ranks.run(lambda rank: workers[rank].publish(frozen().copy_tensors()))
    assert all(isinstance(result, RuntimeError) for result in results), results
    assert str(results[0]) == str(results[1])
    if stage in {"export", "gather", "write"}:
        rpc.assert_not_called()
    if stage == "export":
        assert all(worker.sync.gather_full_adapter.call_count == 0 for worker in workers)
    assert all(worker.published_version is None for worker in workers)


@pytest.mark.parametrize(
    "overrides",
    [
        {"artifact_dir": "relative"},
        {"base_fingerprint": ""},
        {"probe_input_ids": ()},
        {"probe_input_ids": (-1,)},
        {"probe_input_ids": (True,)},
        {"max_versions": 1},
        {"reconcile_interval_s": float("inf")},
        {"reconcile_interval_s": 0},
    ],
)
def test_training_configuration_rejects_invalid_inputs(tmp_path, overrides):
    config = {"artifact_dir": str(tmp_path), "base_fingerprint": "base", "probe_input_ids": (1,)}
    with pytest.raises(ValueError):
        LoRATrainingConfig(**(config | overrides))
