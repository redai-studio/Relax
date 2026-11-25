# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Exercise scheduler startup and SGLang's real post-load Gloo barrier.

Model loading and HTTP serving are external, expensive boundaries. The
scheduler subprocess runs the production Relax entrypoint, then the installed
SGLang barrier with two real ranks. A deliberately short upstream default keeps
the straggler regression bounded without loading model weights.
"""

import multiprocessing
import os
import sys
from datetime import timedelta
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest
import torch.distributed as dist


def _scheduler_barrier_child(entrypoint, rank, rendezvous, entered, release, results, references):
    phase = "imports"
    try:
        import ray
        from sglang.srt.managers import scheduler
        from sglang.srt.model_executor.model_runner_components import load_model_utils

        from relax.distributed.replay_manager.backend import sglang as replay_backend

        replay_initialized = []

        def run_barrier(*_args, **_kwargs):
            nonlocal phase
            phase = "barrier"
            dist.init_process_group(
                "gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=15)
            )
            try:
                if references:
                    assert replay_initialized == [True], "reference scheduler initialization was bypassed"
                if rank == 0:
                    entered.set()
                elif not release.wait(15):
                    raise TimeoutError("test did not release the delayed TP rank")
                load_model_utils.dist_barrier_after_load(elastic_ep_backend=None, tp_rank=rank)
            finally:
                dist.destroy_process_group()

        with (
            # Scale down only SGLang's original 480-second default. The Relax
            # override must reach the function that actually consumes it.
            patch.object(load_model_utils, "UNBALANCED_MODEL_LOADING_TIMEOUT_S", 0.1),
            patch.object(
                load_model_utils,
                "get_parallel",
                lambda: SimpleNamespace(tp_group=SimpleNamespace(cpu_group=dist.group.WORLD)),
            ),
            patch.object(scheduler, "run_scheduler_process", run_barrier),
            patch.object(ray, "is_initialized", lambda: True),
            patch.object(replay_backend, "_install_tq_topology_cache", lambda: None),
            patch.object(
                replay_backend.replay_data_manager,
                "initialize_worker",
                lambda *_args, **_kwargs: replay_initialized.append(True),
            ),
            patch.dict(os.environ, {"RELAX_REPLAY_REGISTRATION": "{}", "RELAX_REPLAY_ENGINE_PUBLISHER": "[]"}),
        ):
            (entrypoint or scheduler.run_scheduler_process)()
        results.put({"rank": rank, "phase": phase, "ok": True})
    except Exception as error:
        if phase == "imports":
            entered.set()
        results.put({"rank": rank, "phase": phase, "ok": False, "error": f"{type(error).__name__}: {error}"})


@pytest.mark.parametrize("references", [False, True], ids=["ordinary", "reference-publication"])
def test_scheduler_loading_timeout_reaches_straggler_barrier(monkeypatch, tmp_path, references):
    pytest.importorskip("sglang.srt.model_executor.model_runner_components.load_model_utils", exc_type=ImportError)
    from relax.backends.sglang import sglang_engine

    monkeypatch.setenv("RELAX_MODEL_LOADING_TIMEOUT_S", "5")
    monkeypatch.setenv("RELAX_OPTIMIZE_ROUTING_REPLAY", "0")
    context = multiprocessing.get_context("spawn")
    entered, release, results = context.Event(), context.Event(), context.Queue()
    processes = []

    def launch(_server_args, *, run_scheduler_process_func=None, run_detokenizer_process_func=None):
        if references:
            from relax.distributed.replay_manager.backend.sglang import run_reference_detokenizer

            assert run_detokenizer_process_func is run_reference_detokenizer
        for rank in range(2):
            process = context.Process(
                target=_scheduler_barrier_child,
                args=(
                    run_scheduler_process_func,
                    rank,
                    (tmp_path / "barrier").as_uri(),
                    entered,
                    release,
                    results,
                    references,
                ),
            )
            process.start()
            processes.append(process)

    # Replace only the HTTP server boundary; let Relax select and serialize
    # the real scheduler entrypoint into fresh Python processes.
    http_server = ModuleType("sglang.srt.entrypoints.http_server")
    http_server.launch_server = launch
    monkeypatch.setitem(sys.modules, http_server.__name__, http_server)
    monkeypatch.setattr("sglang.srt.utils.kill_itself_when_parent_died", lambda: None)
    try:
        sglang_engine._launch_server_with_patches(SimpleNamespace(), os.getppid(), references)
        assert entered.wait(60), "scheduler did not reach the real post-load barrier"
        # The second rank is deliberately held past the scaled upstream
        # timeout but well inside the configured five-second loading budget.
        release.wait(0.4)
        release.set()
        observed = [results.get(timeout=30) for _ in processes]
        for process in processes:
            process.join(timeout=10)
            assert process.exitcode == 0
        assert all(row["phase"] == "barrier" for row in observed), observed
        assert all(row["ok"] for row in observed), observed
    finally:
        release.set()
        for process in processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join(timeout=5)
        results.close()
        results.join_thread()
