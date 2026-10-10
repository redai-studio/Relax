# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples/models/kimi-k3/tools"))
import convert_kimi_k3_torch_dist_to_hf as baseline
import convert_kimi_k3_torch_dist_to_hf_parallel as parallel


def _node(name: str, gpus: int, cpus: int = 16, alive: bool = True) -> dict:
    return {"NodeID": name, "Alive": alive, "Resources": {"GPU": gpus, "CPU": cpus}}


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("gpu_driver", [False, True])
def test_export_placement_uses_eligible_nodes(packed: bool, gpu_driver: bool) -> None:
    ray = SimpleNamespace(nodes=lambda: [_node("worker", 8), _node("head", 8 if gpu_driver else 0)])
    args = SimpleNamespace(world_size=8, gpus_per_node=8 if packed else None, cpus_per_worker=2)
    nodes = parallel._pin_nodes(ray, "head", args)
    expected = "head" if gpu_driver else "worker"
    assert nodes == ([expected] * 8 if packed else [expected] + [None] * 7)


def test_export_placement_excludes_dead_or_cpu_starved_nodes() -> None:
    ray = SimpleNamespace(nodes=lambda: [_node("dead", 8, alive=False), _node("head", 8, cpus=1)])
    args = SimpleNamespace(world_size=8, gpus_per_node=None, cpus_per_worker=2)
    with pytest.raises(RuntimeError, match="No alive export node"):
        parallel._pin_nodes(ray, "head", args)


def test_packed_export_checks_whole_node_capacity() -> None:
    ray = SimpleNamespace(nodes=lambda: [_node("head", 0), _node("a", 8), _node("b", 8, cpus=8)])
    args = SimpleNamespace(world_size=16, gpus_per_node=8, cpus_per_worker=2)
    with pytest.raises(RuntimeError, match="found 1 eligible"):
        parallel._pin_nodes(ray, "head", args)


@pytest.mark.parametrize("parallel_export", [False, True])
def test_export_main_places_rank_zero_and_rendezvous_on_gpu_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, parallel_export: bool
) -> None:
    import ray

    head_id, worker_id = "a" * 56, "b" * 56
    args = SimpleNamespace(
        world_size=2,
        cpus_per_worker=2,
        gpus_per_node=None,
        pp=1,
        expert_tp=1,
        ep=2,
        repo_root=str(tmp_path),
        output_dir=str(tmp_path / "output"),
        replace_output=False,
        after_job_id=None,
        ray_address="unused",
        merge_lora=False,
    )
    monkeypatch.setattr(baseline, "_parse_args", lambda: args)

    def publish_output(args, summaries):
        output = Path(args.output_dir)
        Path(args.staging_dir).rename(output)
        (output / "relax_export_report.json").write_text("{}")

    publish = Mock(side_effect=publish_output)
    monkeypatch.setattr(baseline, "_publish", publish)
    calls = []

    class Task:
        def __init__(self, function, resources, strategy=None):
            self.function, self.resources, self.strategy = function, resources, strategy

        def options(self, *, scheduling_strategy):
            assert scheduling_strategy.soft is False
            return Task(self.function, self.resources, scheduling_strategy)

        def remote(self, *values):
            if self.function is baseline._rendezvous_address:
                assert self.strategy.node_id == worker_id
                assert self.resources["num_cpus"] == 0
                return ("worker-address", 23456)
            rank, master, port, worker_args = values
            calls.append((rank, master, port, self.strategy, self.resources))
            staging = Path(worker_args.staging_dir)
            staging.mkdir(exist_ok=True)
            (staging / f"parallel_export_rank_{rank:03d}.json").write_text(json.dumps({"rank": rank}))
            return {"tensors": 1}

    monkeypatch.setattr(ray, "init", Mock())
    monkeypatch.setattr(ray, "shutdown", Mock())
    monkeypatch.setattr(ray, "cluster_resources", lambda: {"GPU": 2})
    monkeypatch.setattr(ray, "nodes", lambda: [_node(head_id, 0), _node(worker_id, 2)])
    monkeypatch.setattr(ray, "get_runtime_context", lambda: SimpleNamespace(get_node_id=lambda: head_id))
    monkeypatch.setattr(ray, "remote", lambda **resources: lambda function: Task(function, resources))
    monkeypatch.setattr(ray, "get", lambda value: value)
    cancel = Mock()
    monkeypatch.setattr(ray, "cancel", cancel)
    (parallel if parallel_export else baseline).main()
    assert len(calls) == 2
    assert calls[0][3].node_id == worker_id
    for rank, master, port, strategy, resources in calls:
        assert (master, port) == ("worker-address", 23456)
        assert resources["num_gpus"] == 1
        assert resources["num_cpus"] == 2
    publish.assert_called_once()
    cancel.assert_not_called()
    ray.shutdown.assert_called_once()
