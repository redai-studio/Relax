# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import multiprocessing
import time

import psutil
import pytest

from relax.backends.sglang.process_cleanup import terminate_server


def _kill_tree(pid):
    parent = psutil.Process(pid)
    for child in parent.children(recursive=True):
        child.kill()
    parent.kill()


def test_server_cleanup_confirms_exit_before_returning():
    process = multiprocessing.get_context("spawn").Process(target=time.sleep, args=(60,))
    process.start()
    try:
        terminate_server(process, _kill_tree, timeout=2)
        assert not process.is_alive()
        assert process.exitcode is not None
    finally:
        if process.is_alive():
            process.kill()
        process.join()


def test_server_cleanup_failure_keeps_process_handle_for_retry():
    process = multiprocessing.get_context("spawn").Process(target=time.sleep, args=(60,))
    process.start()
    try:
        with pytest.raises(RuntimeError, match="exit was not confirmed"):
            terminate_server(process, lambda pid: None, timeout=0.01)
        assert process.is_alive()
        terminate_server(process, _kill_tree, timeout=2)
    finally:
        if process.is_alive():
            process.kill()
        process.join()


def test_missing_parent_is_not_a_worker_exit_receipt():
    process = multiprocessing.get_context("spawn").Process(target=time.sleep, args=(0,))
    process.start()
    process.join()
    with pytest.raises(RuntimeError, match="worker tree"):
        terminate_server(process, _kill_tree, timeout=0.01)


def test_node_cleanup_confirms_recorded_server_exit():
    from relax.backends.sglang.process_cleanup import capture_server_identity, cleanup_server_identity

    process = multiprocessing.get_context("spawn").Process(target=time.sleep, args=(60,))
    process.start()
    try:
        identity = capture_server_identity(process, node_id="node-a", port=1234)
        assert identity["server_pid"] == process.pid
        cleanup = multiprocessing.get_context("spawn").Process(target=cleanup_server_identity, args=(identity, 2))
        cleanup.start()
        cleanup.join(timeout=5)
        assert cleanup.exitcode == 0
        process.join(timeout=2)
        assert not process.is_alive()
        cleanup_server_identity(identity, timeout=2)  # Retry after parent disappeared.
    finally:
        if process.is_alive():
            process.kill()
        process.join()


def test_node_cleanup_never_kills_reused_pid(monkeypatch):
    from relax.backends.sglang.process_cleanup import cleanup_server_identity

    current = psutil.Process()
    identity = {
        "server_pid": current.pid,
        "processes": [{"pid": current.pid, "create_time": current.create_time() - 100}],
    }
    monkeypatch.setattr(psutil.Process, "kill", lambda self: pytest.fail("killed a reused PID"))
    cleanup_server_identity(identity)


def test_node_cleanup_requires_a_complete_tree_record():
    from relax.backends.sglang.process_cleanup import cleanup_server_identity

    with pytest.raises(RuntimeError, match="identity"):
        cleanup_server_identity({"server_pid": 123})
