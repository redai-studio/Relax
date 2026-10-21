# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Two real CPU/Gloo ranks closing straggler windows: whatever fails locally,
every rank has to issue the same collectives in the same order."""

from __future__ import annotations

import json
from datetime import timedelta

import pytest


torch = pytest.importorskip("torch")


class _Unpicklable(str):
    def __reduce__(self):
        raise MemoryError("injected while pickling, before any collective")


def _rank_worker(rank, world_size, init_file, out_dir, failure):
    import torch.distributed as dist

    from relax.utils.straggler import runtime
    from relax.utils.straggler.collector import StragglerCollector
    from relax.utils.straggler.detector import RankMeta
    from tests.utils.straggler_helpers import FakeEvent, bracket

    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=60),
    )
    try:
        runtime.init_straggler_gather_group(timeout_s=10)
        calls = []

        def gather(values):
            calls.append("row")
            return runtime.gloo_all_gather(values)

        def gather_objects(obj):
            calls.append("objects")
            gathered = runtime.gloo_all_gather_objects(obj)
            if failure == "decode" and rank == 0:
                # Like a MemoryError while decoding: the collectives are done on every rank, only rank 0 fails.
                raise MemoryError("injected after the metadata collectives")
            return gathered

        host = {"pickle": _Unpicklable("host"), "oversized": "h" * 2000}.get(failure, "") if rank == 0 else ""
        collector = StragglerCollector(
            rank_meta=RankMeta(rank=rank, dp=rank, tp=0, pp=0, host=host),
            is_primary=rank == 0,
            report_interval=1,
            event_factory=FakeEvent,
            gather=gather,
            gather_objects=gather_objects,
            is_capturing=lambda: False,
            register_gc_callback=False,
            background=False,
        )
        for rollout_id in range(3):
            bracket(collector)
            collector.end_step(rollout_id=rollout_id)
        with open(f"{out_dir}/rank{rank}.json", "w") as f:
            json.dump({"calls": calls, "disabled": collector.disabled, "reason": collector.health.disabled_reason}, f)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not torch.distributed.is_available() or not torch.distributed.is_gloo_available(),
    reason="torch.distributed with Gloo is required",
)
@pytest.mark.parametrize("failure", ["pickle", "oversized", "decode"])
def test_a_metadata_failure_on_one_rank_keeps_every_rank_on_the_same_collectives(tmp_path, failure):
    import torch.multiprocessing as mp

    mp.spawn(_rank_worker, args=(2, str(tmp_path / "gloo"), str(tmp_path), failure), nprocs=2, join=True)
    ranks = [json.loads((tmp_path / f"rank{rank}.json").read_text()) for rank in range(2)]
    assert ranks[0]["calls"] == ranks[1]["calls"] == ["objects", "row", "row", "row"]
    # Without the metadata table rank 0 cannot be analysed, so it asks everyone to switch off in that same window.
    assert ranks[0]["disabled"] and ranks[1]["disabled"]
    assert "rank 0 asked to switch the profiler off" in ranks[1]["reason"]
