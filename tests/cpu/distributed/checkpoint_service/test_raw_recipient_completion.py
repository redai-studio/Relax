# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Four CPU/Gloo ranks: two PP sources, AF and delayed Ref consume one stop.

Mocks cannot prove cross-role collective completion. Four ranks are the minimum
for this PP/role interleaving; groups have a 15-second timeout and each test
phase a 45-second process deadline. HTTP serves the actual coordinator bodies.
"""

import ast
import asyncio
import json
import logging
import multiprocessing
import queue
import threading
import time
import traceback
from datetime import timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

import httpx
import torch
import torch.distributed as dist


ROOT = Path(__file__).resolve().parents[4]
BACKEND = ROOT / "relax/distributed/checkpoint_service/backends/device_direct.py"
COORDINATOR = ROOT / "relax/distributed/checkpoint_service/coordinator/service.py"
CASES = ("slow_reference", "consecutive", "ack_failure", "receiver_ready_failure", "clear_failure", "post_failure")


def _actual_functions(path, names, namespace):
    nodes = [
        node
        for node in ast.walk(ast.parse(path.read_text()))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names
    ]
    assert {node.name for node in nodes} == set(names)
    for node in nodes:
        node.decorator_list = []
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *nodes],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace


def _worker(rank, store, url, phase, af_ack, release_ref, source_done, entries, output):
    torch.set_num_threads(1)
    try:
        dist.init_process_group(
            "gloo", init_method="file://" + store, rank=rank, world_size=4, timeout=timedelta(seconds=15)
        )
        sources = dist.new_group([0, 1], timeout=timedelta(seconds=15))
        raw = [
            dist.new_group([0, 2, 3], timeout=timedelta(seconds=15)),
            dist.new_group([1, 2, 3], timeout=timedelta(seconds=15)),
        ]
        af = dist.new_group([2], timeout=timedelta(seconds=15))
        ref = dist.new_group([3], timeout=timedelta(seconds=15))
        local = sources if rank < 2 else af if rank == 2 else ref
        namespace = _actual_functions(
            BACKEND,
            {"_raw_group_status", "_complete_raw_group", "_finish_actor_fwd_ref_sync", "recv_weight"},
            dict(
                torch=torch, dist=dist, httpx=httpx, get_gloo_group=lambda: local, logger=logging.getLogger(__name__)
            ),
        )
        complete = namespace["_complete_raw_group"]
        for case in CASES:
            phase.wait(timeout=45)

            def acknowledge(group, device, **kwargs):
                if rank == 2:
                    af_ack.set()
                if case == "receiver_ready_failure" and rank == 3 and group is raw[1]:
                    namespace["_raw_group_status"](group, device, True, **kwargs)
                    return "injected receiver device fence failure"
                result = complete(group, device, **kwargs)
                if case == "ack_failure" and rank == 1:
                    raise RuntimeError("injected source fence failure")
                return result

            namespace["_complete_raw_group"] = acknowledge
            backend = SimpleNamespace(
                device=torch.device("cpu"),
                coordinator_url=url,
                http_client=httpx.Client(timeout=10, trust_env=False),
                args=SimpleNamespace(dcs_recv_weight_meta_wait_timeout_s=0.1),
                model=None,
                _is_pp_src_rank=True,
                _megatron=SimpleNamespace(load_weight=lambda *args: None),
                _model_update_groups_for_actor_fwd_ref=raw[rank] if rank < 2 else {"pp1": raw[1], "pp0": raw[0]},
            )
            error = None
            try:
                for _publication in range(2 if case == "consecutive" else 1):
                    if rank < 2:
                        namespace["_finish_actor_fwd_ref_sync"](backend)
                    elif case != "post_failure":
                        if case == "slow_reference" and rank == 3:
                            assert release_ref.wait(timeout=20)
                        entries[rank] += 1
                        namespace["recv_weight"](backend)
            except Exception as exc:
                error = str(exc)
            finally:
                if rank < 2:
                    source_done[rank].set()
                backend.http_client.close()
            output.put((case, rank, error))
            phase.wait(timeout=45)
        dist.destroy_process_group()
    except BaseException:
        output.put(("WORKER_FATAL", rank, traceback.format_exc()))
        raise


def test_all_roles_consume_stop_before_source_clear_and_share_errors(tmp_path):
    namespace = _actual_functions(
        COORDINATOR,
        {"send_weight_meta", "recv_weight_meta", "clear_weight_meta"},
        dict(asyncio=asyncio, time=time, Response=lambda **values: values),
    )
    coordinator = SimpleNamespace(weight_meta_buffer=[], _lock=threading.Lock())
    state = dict(case=None, clear_calls=0, post_calls=0)
    context = multiprocessing.get_context("spawn")
    clear_waiting, allow_clear = context.Event(), context.Event()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            state["post_calls"] += 1
            if state["case"] == "post_failure":
                self.send_error(500, "injected stop post failure")
                return
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            result = asyncio.run(namespace["send_weight_meta"](coordinator, payload))
            self.respond(result)

        def do_GET(self):
            parsed = urlsplit(self.path)
            if parsed.path == "/clear_weight_meta":
                state["clear_calls"] += 1
                if state["case"] == "clear_failure":
                    self.send_error(500, "injected metadata clear failure")
                    return
                if state["case"] == "consecutive" and state["clear_calls"] == 1:
                    clear_waiting.set()
                    assert allow_clear.wait(timeout=20)
                result = asyncio.run(namespace["clear_weight_meta"](coordinator))
            else:
                query = parse_qs(parsed.query)
                result = asyncio.run(
                    namespace["recv_weight_meta"](
                        coordinator, int(query["index"][0]), float(query.get("wait_timeout_s", [0])[0])
                    )
                )
            self.respond(result)

        def respond(self, value):
            data = json.dumps(value).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    phase = context.Barrier(5)
    entries = context.Array("i", 4)
    af_ack, release_ref = context.Event(), context.Event()
    source_done, output = [context.Event(), context.Event()], context.Queue()
    workers = [
        context.Process(
            target=_worker,
            args=(
                rank,
                str(tmp_path / "gloo"),
                f"http://127.0.0.1:{server.server_port}",
                phase,
                af_ack,
                release_ref,
                source_done,
                entries,
                output,
            ),
        )
        for rank in range(4)
    ]
    try:
        for worker in workers:
            worker.start()
        for case in CASES:
            state.update(case=case, clear_calls=0, post_calls=0)
            coordinator.weight_meta_buffer.clear()
            af_ack.clear()
            release_ref.clear()
            clear_waiting.clear()
            allow_clear.clear()
            for rank in range(4):
                entries[rank] = 0
            for done in source_done:
                done.clear()
            phase.wait(timeout=45)
            if case == "slow_reference":
                if not af_ack.wait(timeout=20):
                    details = []
                    while True:
                        try:
                            details.append(output.get_nowait())
                        except queue.Empty:
                            break
                    raise AssertionError(f"AF did not reach stop completion: {details}")
                assert state["clear_calls"] == 0 and coordinator.weight_meta_buffer
                assert not any(done.is_set() for done in source_done)
                release_ref.set()
            if case == "consecutive":
                assert clear_waiting.wait(timeout=20)
                assert [entries[2], entries[3]] == [1, 1], "Receiver reentered before source clear"
                assert coordinator.weight_meta_buffer
                allow_clear.set()
            phase.wait(timeout=45)
            rows = [output.get(timeout=3) for _ in range(4)]
            assert {row[:2] for row in rows} == {(case, rank) for rank in range(4)}, rows
            source_errors = [row[2] for row in rows if row[1] < 2]
            receiver_errors = [row[2] for row in rows if row[1] >= 2]
            assert state["post_calls"] == (2 if case == "consecutive" else 1)
            if case in ("slow_reference", "consecutive"):
                assert receiver_errors == [None, None], rows
                assert source_errors == [None, None]
                assert state["clear_calls"] == (2 if case == "consecutive" else 1)
                assert not coordinator.weight_meta_buffer
                if case == "consecutive":
                    assert [entries[2], entries[3]] == [2, 2]
            else:
                phase_name = {
                    "ack_failure": "completion",
                    "receiver_ready_failure": "completion",
                    "clear_failure": "clear",
                    "post_failure": "stop",
                }[case]
                assert all(error is not None and phase_name in error for error in source_errors), rows
                assert source_errors[0] == source_errors[1]
                if case == "post_failure":
                    assert receiver_errors == [None, None]
                else:
                    assert all(error is not None and "completion failed" in error for error in receiver_errors), rows
                assert state["clear_calls"] == int(case == "clear_failure")
                assert bool(coordinator.weight_meta_buffer) == (case != "post_failure")
        for worker in workers:
            worker.join(timeout=10)
            assert worker.exitcode == 0
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
            worker.join(timeout=5)
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_prior_device_fence_failure_is_encoded_before_successful_ready():
    statuses = []

    def fence(device):
        raise RuntimeError("injected prior device work failure")

    def status(group, device, failed, **kwargs):
        statuses.append(failed)
        return failed

    namespace = _actual_functions(
        BACKEND,
        {"_complete_raw_group"},
        dict(torch=torch, device_module=SimpleNamespace(synchronize=fence), _raw_group_status=status),
    )
    # CUDA is only a device descriptor: no allocation, probe or CUDA call.
    error = namespace["_complete_raw_group"](None, torch.device("cuda", 0), wait=False)
    assert statuses == [True]
    assert error == "injected prior device work failure"
