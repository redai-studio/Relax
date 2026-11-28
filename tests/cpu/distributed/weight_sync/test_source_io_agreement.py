"""Real Gloo ranks agree on errors and drain file work before raising."""

import json
import multiprocessing
import time
from datetime import timedelta
from pathlib import Path


def _agreement_worker(rank, rendezvous, output):
    import torch.distributed as dist

    from relax.distributed.weight_sync.source_io import agree_errors, agree_io_state, coordinated_io

    dist.init_process_group("gloo", rank=rank, world_size=2, init_method=rendezvous, timeout=timedelta(seconds=20))
    group = dist.group.WORLD
    try:
        assert agree_errors(None, group) == [None, None]
        assert agree_errors("peer failure" if rank == 1 else None, group) == [None, "peer failure"]
        states = agree_io_state({"done": rank == 0, "error": None, "expired": rank == 1}, group)
        assert not all(state["done"] for state in states)
        assert all(state["expired"] for state in states)
        assert all(state["error"] is None for state in states)
        states = agree_io_state({"done": True, "error": "bad read" if rank == 1 else None, "expired": False}, group)
        assert all(state["done"] for state in states)
        assert [state["error"] for state in states] == [None, "bad read"]

        def work():
            if rank == 1:
                raise OSError("rank-one read failed")
            time.sleep(0.15)
            Path(output, "writer-drained").write_text("finished")

        try:
            coordinated_io(work, lambda state: agree_io_state(state, group), stage="test read", timeout=5)
        except RuntimeError as error:
            assert "rank-one read failed" in str(error)
        else:
            raise AssertionError("The peer failure was lost")
        assert Path(output, "writer-drained").read_text() == "finished"
        assert agree_errors(None, group) == [None, None]
        Path(output, f"rank-{rank}.json").write_text(json.dumps({"passed": True}))
        # Both ranks finish their assertions before either tears down Gloo.
        dist.barrier(group=group)
    finally:
        dist.destroy_process_group(group)


def test_two_real_ranks_preserve_failure_details_and_wait_for_file_workers(tmp_path):
    context = multiprocessing.get_context("spawn")
    processes = [
        context.Process(target=_agreement_worker, args=(rank, f"file://{tmp_path / 'rendezvous'}", str(tmp_path)))
        for rank in range(2)
    ]
    try:
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=40)
        assert [process.exitcode for process in processes] == [0, 0]
        assert all(json.loads((tmp_path / f"rank-{rank}.json").read_bytes())["passed"] for rank in range(2))
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
