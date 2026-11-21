import json
import time
from pathlib import Path

import pytest

from relax.distributed.weight_sync.deadline import time_budget
from relax.distributed.weight_sync.sglang_loader import EmbeddedSGLangGroup
from relax.distributed.weight_sync.sglang_plugin import WorkerBusyError
from relax.distributed.weight_sync.store import atomic_write, canonical


def group(tmp_path):
    adapter = object.__new__(EmbeddedSGLangGroup)
    adapter.tp_size, adapter.pp_size = 2, 1
    adapter.audit_directory = tmp_path
    adapter._rpc_failed = False
    adapter._serving = False
    adapter._workers = {str(rank): {"instance": str(rank)} for rank in range(2)}
    return adapter


@pytest.mark.parametrize("other", ["busy", "error", "success", "wrong-request"])
def test_only_complete_correlated_busy_receipts_are_retryable(tmp_path, other):
    adapter = group(tmp_path)

    def send(payload):
        request = json.loads(payload)
        for rank in range(2):
            row = {
                "rank": rank,
                "request_id": request["request_id"],
                "success": False,
                "error_code": "WORKER_BUSY",
                "identity": adapter._workers[str(rank)],
            }
            if rank == 1:
                if other == "error":
                    row["error_code"] = "WORKER_ERROR"
                elif other == "success":
                    row["success"] = True
                elif other == "wrong-request":
                    row["request_id"] = "different-request"
            atomic_write(Path(request["receipts"]) / f"rank-{rank}.json", canonical(row))

    adapter._send_rpc = send
    with pytest.raises(RuntimeError) as caught:
        adapter._rpc("flush")
    assert isinstance(caught.value, WorkerBusyError) == (other == "busy")
    assert not adapter._serving


def test_flush_waits_for_scheduler_cleanup_without_reopening_admission(tmp_path):
    adapter = group(tmp_path)
    attempts = []

    def rpc(phase):
        attempts.append(phase)
        assert not adapter._serving
        if len(attempts) <= 2:
            raise WorkerBusyError("probe batch is still retiring")

    adapter._rpc = rpc
    adapter.flush()
    assert attempts == ["flush", "flush", "flush"]
    observation = json.loads(next(tmp_path.glob("flush-drain-*.json")).read_bytes())
    assert observation["passed"] and observation["busy_attempts"] == 2
    assert not adapter._serving


def test_persistent_busy_obeys_the_shared_operation_deadline(tmp_path):
    adapter = group(tmp_path)

    def rpc(phase):
        raise WorkerBusyError("still busy")

    adapter._rpc = rpc
    started = time.monotonic()
    with time_budget(0.05), pytest.raises(TimeoutError, match="deadline"):
        adapter.flush()
    assert time.monotonic() - started < 1
    observation = json.loads(next(tmp_path.glob("flush-drain-*.json")).read_bytes())
    assert not observation["passed"] and observation["busy_attempts"] > 0
    assert not adapter._serving


def test_real_flush_error_is_not_retried(tmp_path):
    adapter = group(tmp_path)
    attempts = []

    def rpc(phase):
        attempts.append(phase)
        raise RuntimeError("stale worker identity")

    adapter._rpc = rpc
    with pytest.raises(RuntimeError, match="stale worker identity"):
        adapter.flush()
    assert attempts == ["flush"]
    assert not adapter._serving
