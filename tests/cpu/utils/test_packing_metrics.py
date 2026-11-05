# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Packing statistics count physical packs once across CP replicas."""

import json
import multiprocessing
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist

from relax.utils.training.packing_metrics import PackingMetrics


def _packing_worker(rank, rendezvous, output):
    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2, timeout=timedelta(seconds=15)
    )
    results = []
    try:
        for mode in ("static", "dynamic", "vision", "empty"):
            metrics = PackingMetrics()
            batch = {"tokens": torch.empty(1, 8)}
            if mode == "dynamic":
                batch.update(tokens=torch.empty(1, 6 if rank == 0 else 10), dynamic_cp_size=1)
            elif mode == "vision":
                batch["vlm_packed_seq_params"] = SimpleNamespace(cu_seqlens_q_cpu=[0, 16, 24])
            if mode != "empty":
                metrics.add(batch, static_cp_size=2)
            results.append(metrics.reduce(dist.group.WORLD, torch.device("cpu")))
        Path(output, f"metrics-{rank}.json").write_text(json.dumps(results))
    finally:
        dist.destroy_process_group()


def test_packing_metrics_reduce_static_dynamic_and_vision_packs(tmp_path):
    ctx = multiprocessing.get_context("spawn")
    workers = [
        ctx.Process(target=_packing_worker, args=(rank, str(tmp_path / "rendezvous"), str(tmp_path)))
        for rank in range(2)
    ]
    try:
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join(timeout=45)
        assert all(not worker.is_alive() and worker.exitcode == 0 for worker in workers)
        results = [json.loads((tmp_path / f"metrics-{rank}.json").read_text()) for rank in range(2)]
        assert results[0] == results[1]
        assert results[0] == [
            {"num_microbatches_mean": 1.0, "pack_tokens_mean": 16.0, "pack_tokens_max": 16.0},
            {"num_microbatches_mean": 1.0, "pack_tokens_mean": 8.0, "pack_tokens_max": 10.0},
            {"num_microbatches_mean": 1.0, "pack_tokens_mean": 24.0, "pack_tokens_max": 24.0},
            {"num_microbatches_mean": 0.0, "pack_tokens_mean": 0.0, "pack_tokens_max": 0.0},
        ]
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
            worker.join(timeout=5)
