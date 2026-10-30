# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Two real CPU/Gloo ranks closing straggler windows: whatever fails locally,
every rank has to issue the same collectives in the same order."""

from __future__ import annotations

import itertools
import json
from datetime import timedelta

import pytest


torch = pytest.importorskip("torch")


class _Unpicklable(str):
    def __reduce__(self):
        raise MemoryError("injected while pickling, before any collective")


def _rank_worker(rank, world_size, init_file, out_dir, failure):
    import torch.distributed as dist

    from relax.utils.straggler import collectives
    from relax.utils.straggler.analysis.report import WindowAnalyzer
    from relax.utils.straggler.core.collector import StragglerCollector
    from relax.utils.straggler.core.stats import RankMeta
    from tests.utils.straggler_helpers import FakeEvent, bracket

    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=60),
    )
    try:
        collectives.init_straggler_gather_group(timeout_s=10)
        calls = []

        def gather(values):
            calls.append("row")
            return collectives.gloo_all_gather(values)

        def gather_objects(obj):
            calls.append("objects")
            gathered = collectives.gloo_all_gather_objects(obj)
            if failure == "decode" and rank == 0:
                # Like a MemoryError while decoding: the collectives are done on every rank, only rank 0 fails.
                raise MemoryError("injected after the metadata collectives")
            return gathered

        host = _Unpicklable("host") if failure == "pickle" and rank == 0 else ""
        collector = StragglerCollector(
            rank_meta=RankMeta(rank=rank, dp=rank, tp=0, pp=0, host=host),
            is_primary=rank == 0,
            report_interval=1,
            event_factory=FakeEvent,
            gather=gather,
            gather_objects=gather_objects,
            is_capturing=lambda: False,
            analyze=WindowAnalyzer(),
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


def _trace_worker(rank, world_size, init_file, out_dir, failure):
    import torch.distributed as dist

    from relax.utils.straggler import collectives
    from relax.utils.straggler.analysis.report import WindowAnalyzer
    from relax.utils.straggler.core.collector import StragglerCollector
    from relax.utils.straggler.core.stats import RankMeta
    from tests.utils.straggler_helpers import FakeEvent, bracket

    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=60),
    )
    try:
        collectives.init_straggler_gather_group(timeout_s=10)
        calls, sunk = [], []
        window = itertools.count(1)

        def gather(values):
            failing = failure if rank == 0 and next(window) == 2 else None
            if failing == "pack":
                # torch.tensor cannot take the row: rank 0 sends NaN and raises after the collective.
                return collectives.gloo_all_gather([*values[:-1], object()])
            gathered = collectives.gloo_all_gather(values)
            if failing == "readout":
                # Like Tensor.tolist() failing: the collective is done on every rank, only rank 0 has no table.
                raise MemoryError("injected after the row collective")
            return gathered

        def gather_to(sample, dst):
            calls.append(dst)
            return collectives.gloo_gather_objects_to(sample, dst)

        def sink(traces, meta, first_rollout):
            sunk.append([None if trace is None else len(trace.records) for trace in traces])
            return {}

        collector = StragglerCollector(
            rank_meta=RankMeta(rank=rank, dp=rank, tp=0, pp=0, primary=rank == 1),
            is_primary=rank == 1,
            report_interval=1,
            event_factory=FakeEvent,
            gather=gather,
            gather_objects=collectives.gloo_all_gather_objects,
            is_capturing=lambda: False,
            analyze=WindowAnalyzer(trace_sink=sink),
            register_gc_callback=False,
            background=False,
            trace=True,
            gather_to=gather_to,
            primary_rank=collectives.gloo_primary_rank(rank == 1),
            level=2,
        )
        for rollout_id in range(4):
            bracket(collector)
            collector.end_step(rollout_id=rollout_id)
        with open(f"{out_dir}/rank{rank}.json", "w") as f:
            json.dump({"calls": calls, "sunk": sunk, "errors": collector.health.errors_total}, f)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not torch.distributed.is_available() or not torch.distributed.is_gloo_available(),
    reason="torch.distributed with Gloo is required",
)
@pytest.mark.parametrize(
    ("failure", "sunk", "errors"),
    [
        (None, [[None, None], [6, 6], [6, 6], [6, 6]], [0, 0]),
        # Rank 0 loses the second window's table after the collective; rank 1 still analyses that window.
        ("readout", [[None, None], [6, 6], [6, 6], [6, 6]], [1, 0]),
        # Rank 0 cannot send its second row; both ranks skip that window.
        ("pack", [[None, None], [6, 6], [6, 6]], [1, 0]),
    ],
)
def test_every_rank_hands_its_sample_to_the_primary_rank_in_the_same_window(tmp_path, failure, sunk, errors):
    import torch.multiprocessing as mp

    mp.spawn(_trace_worker, args=(2, str(tmp_path / "gloo"), str(tmp_path), failure), nprocs=2, join=True)
    ranks = [json.loads((tmp_path / f"rank{rank}.json").read_text()) for rank in range(2)]
    assert ranks[0]["calls"] == ranks[1]["calls"] == [1, 1, 1, 1]
    # The first window has no sample; each later one brings one bracket per rank to rank 1 only.
    assert ranks[1]["sunk"] == sunk
    assert ranks[0]["sunk"] == []
    assert [r["errors"] for r in ranks] == errors


@pytest.mark.skipif(
    not torch.distributed.is_available() or not torch.distributed.is_gloo_available(),
    reason="torch.distributed with Gloo is required",
)
@pytest.mark.parametrize("failure", ["pickle", "decode"])
def test_a_metadata_failure_on_one_rank_keeps_every_rank_on_the_same_collectives(tmp_path, failure):
    import torch.multiprocessing as mp

    mp.spawn(_rank_worker, args=(2, str(tmp_path / "gloo"), str(tmp_path), failure), nprocs=2, join=True)
    ranks = [json.loads((tmp_path / f"rank{rank}.json").read_text()) for rank in range(2)]
    assert ranks[0]["calls"] == ranks[1]["calls"] == ["objects", "row", "row", "row"]
    # Without the metadata table rank 0 cannot be analysed, so it asks everyone to switch off in that same window.
    assert ranks[0]["disabled"] and ranks[1]["disabled"]
    assert "rank 0 asked to switch the profiler off" in ranks[1]["reason"]
