# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Two real Gloo groups expose abandonment hidden by a single-rank mock."""

import json
import multiprocessing
from datetime import timedelta
from pathlib import Path
from time import monotonic, sleep

import pytest
import torch.distributed as dist

from task_delta_weight_sync.native_call import guarded_native_call


def _native_rank(rank: int, directory: str) -> None:
    root = Path(directory)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{root / 'rendezvous'}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=15),
    )
    control = dist.new_group(backend="gloo", timeout=timedelta(seconds=5))

    def normal():
        sleep(0.03 * rank)
        return {"rank": rank, "unchanged": [1, 2, 3]}

    result = guarded_native_call(normal, process_group=control, timeout_s=5, failure_path=root / f"normal-{rank}.json")
    assert result == {"rank": rank, "unchanged": [1, 2, 3]}
    assert not (root / f"normal-{rank}.json").exists()
    dist.barrier()

    def returned_error():
        if rank == 0:
            raise ValueError("native returned error")
        return "peer returned normally"

    with pytest.raises(RuntimeError, match="native returned error"):
        guarded_native_call(
            returned_error, process_group=control, timeout_s=5, failure_path=root / f"returned-{rank}.json"
        )
    # Reusing both groups proves a returned error did not abandon either rank.
    dist.barrier()
    (root / f"completed-{rank}.json").write_text(json.dumps({"normal": result, "returned_error_converged": True}))

    def abandoned_collective():
        if rank == 0:
            raise ValueError("peer abandoned data barrier")
        (root / "data-barrier-entered.json").write_text(json.dumps({"rank": rank}))
        dist.barrier()  # Rank zero deliberately never enters this data collective.
        raise AssertionError("Abandoned data barrier unexpectedly returned")

    def quarantine():
        (root / f"quarantine-{rank}.json").write_text(json.dumps({"called": True}))
        return []

    guarded_native_call(
        abandoned_collective,
        process_group=control,
        timeout_s=5,
        failure_path=root / f"stuck-{rank}.json",
        quarantine=quarantine,
    )
    raise AssertionError("An in-flight abandoned collective must terminate this source process")


def test_native_calls_converge_or_exit_both_ranks_with_failure_receipts(tmp_path, monkeypatch):
    # No Ray, Megatron, GPU allocation, or mock collective: a second Gloo group
    # must remain usable while the native thread is stuck in the first group.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    context = multiprocessing.get_context("spawn")
    processes = [context.Process(target=_native_rank, args=(rank, str(tmp_path))) for rank in range(2)]
    deadline = monotonic() + 35
    try:
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=max(0, deadline - monotonic()))
        assert not any(process.is_alive() for process in processes), "Native supervision exceeded the parent deadline"
        assert [process.exitcode for process in processes] == [74, 74]
        assert json.loads((tmp_path / "data-barrier-entered.json").read_text()) == {"rank": 1}
        for rank in range(2):
            completed = json.loads((tmp_path / f"completed-{rank}.json").read_text())
            assert completed == {
                "normal": {"rank": rank, "unchanged": [1, 2, 3]},
                "returned_error_converged": True,
            }
            returned = json.loads((tmp_path / f"returned-{rank}.json").read_text())
            assert not returned["in_flight"] and not returned["deadline_exceeded"]
            assert all(item["done"] for item in returned["statuses"])
            assert any("native returned error" in (item["error"] or "") for item in returned["statuses"])
            stuck = json.loads((tmp_path / f"stuck-{rank}.json").read_text())
            assert stuck["in_flight"] and not stuck["deadline_exceeded"]
            assert len(stuck["statuses"]) == 2
            assert stuck["statuses"][0]["done"] and "peer abandoned" in stuck["statuses"][0]["error"]
            assert not stuck["statuses"][1]["done"]
            assert stuck["quarantine_errors"] == []
            assert json.loads((tmp_path / f"quarantine-{rank}.json").read_text()) == {"called": True}
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
            if process.pid is not None:
                process.join(timeout=3)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=3)
