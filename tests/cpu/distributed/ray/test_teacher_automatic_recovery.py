# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Dedicated teacher recovery through the production owner, on real CPU Ray."""

import asyncio
import time
from types import SimpleNamespace
from urllib.request import urlopen

import pytest
import ray

from relax.distributed.ray.inference_manager import InferenceManager, ModelManager
from relax.distributed.ray.placement_planner import Phase, PlacementClaim, PlacementPlan
from relax.engine.inference.ownership import LifecycleAuthority
from relax.engine.inference.routing import ModelUnavailableError


def _owner_class():
    class Owner(InferenceManager.__ray_metadata__.modified_class):
        def install_engine_standin(self, events):
            import os
            import socket
            import sys
            import threading
            from http.server import BaseHTTPRequestHandler, HTTPServer
            from types import ModuleType

            import psutil

            from relax.distributed.ray.multi_engine_manager import MultiEngineManager

            os.environ["RELAX_INFERENCE_HEALTH_INTERVAL_S"] = "0.05"

            class Engine:
                def __init__(self, args, **kwargs):
                    self.server = None

                def init(self, host, port):
                    # A deterministic barrier holds the replacement in init.
                    ray.get(events.initializing.remote(), timeout=30)

                    class Handler(BaseHTTPRequestHandler):
                        def do_GET(self):
                            self.send_response(200)
                            self.end_headers()
                            self.wfile.write(str(port).encode())

                        def log_message(self, *args):
                            pass

                    self.server = HTTPServer((host, port), Handler)
                    threading.Thread(target=self.server.serve_forever, daemon=True).start()
                    return {
                        "url": f"http://{host}:{port}",
                        "pid": os.getpid(),
                        "node_id": ray.get_runtime_context().get_node_id(),
                    }

                def get_release_evidence(self):
                    process = psutil.Process()
                    return {
                        "node_id": ray.get_runtime_context().get_node_id(),
                        "server_pid": process.pid,
                        "processes": [{"pid": process.pid, "create_time": process.create_time()}],
                    }

                def health_generate(self, **kwargs):
                    return self.server is not None

                def shutdown(self):
                    if self.server is not None:
                        self.server.shutdown()
                        self.server.server_close()

                def resume_memory_occupation(self, **kwargs):
                    return True

            class Runtime(MultiEngineManager):
                def __init__(self, args, **kwargs):
                    super().__init__(args, num_slots=1, engine_actor_cls=Engine)

                def _resolve_placement(self, rank):
                    return (None, [0], [0]), False, 0

                def _engine_actor_options(self, *args):
                    return {"num_cpus": 0}

                def _allocate_engine_addr_and_ports(self, *, new_engines):
                    with socket.socket() as sock:
                        sock.bind(("127.0.0.1", 0))
                        port = sock.getsockname()[1]
                    return {slot: {"host": "127.0.0.1", "port": port} for slot, _ in new_engines}

            module = ModuleType("relax.distributed.ray.teacher_manager")
            module.TeacherEngineRuntime = Runtime
            sys.modules[module.__name__] = module

        def engine(self):
            return self._runtimes["teacher"].engines[0]

    return ray.remote(num_cpus=0)(Owner)


def test_dedicated_teacher_is_rebuilt_automatically_and_requests_follow_new_endpoint():
    started = not ray.is_initialized()
    if started:
        ray.init(address="local", num_cpus=2, include_dashboard=False)

    @ray.remote(num_cpus=0)
    class Events:
        def __init__(self):
            self.count = 0
            self.entered, self.finish = asyncio.Event(), asyncio.Event()

        async def initializing(self):
            self.count += 1
            if self.count > 1:
                self.entered.set()
                await self.finish.wait()

        async def wait_replacement(self):
            await self.entered.wait()

        def allow_replacement(self):
            self.finish.set()

        def starts(self):
            return self.count

    claim = PlacementClaim("teacher", "teacher", "teacher", 0, 1, frozenset({Phase.GENERATE}))
    authority = LifecycleAuthority.remote(PlacementPlan((), (claim,)))
    args = SimpleNamespace(_inference_authority=authority, offload_rollout=False)
    owner = _owner_class().remote(args, "teacher")
    events = Events.remote()
    try:
        ray.get(owner.install_engine_standin.remote(events), timeout=15)
        ray.get(owner.claim_placement.remote(), timeout=15)
        ray.get(owner.add_model.remote("teacher", args, {}), timeout=30)
        model = ModelManager(owner, "teacher")
        before = ray.get(model.get_inference_snapshot.remote(), timeout=5)
        old_url = before["engines"][0]["base_url"]
        with urlopen(old_url, timeout=5) as response:
            assert response.status == 200
        ray.kill(ray.get(owner.engine.remote(), timeout=5), no_restart=True)
        # No recover() call and no failure hint: the production health loop must act.
        ray.get(events.wait_replacement.remote(), timeout=15)
        with pytest.raises(ModelUnavailableError):
            ray.get(model.acquire_inference.remote("during-recovery"), timeout=5)
        # Bursts of hints cannot start concurrent recovery.
        ray.get([owner.report_failure.remote("teacher", old_url) for _ in range(5)], timeout=5)
        assert ray.get(events.starts.remote(), timeout=5) == 2
        ray.get(events.allow_replacement.remote(), timeout=5)
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            after = ray.get(model.get_inference_snapshot.remote(), timeout=5)
            if after["state"] == "ready":
                break
        assert after["state"] == "ready"
        assert after["topology_revision"] > before["topology_revision"]
        permit = ray.get(model.acquire_inference.remote("after-recovery"), timeout=5)
        assert permit.target.base_url != old_url
        with urlopen(permit.target.base_url, timeout=5) as response:
            assert response.read().decode() == permit.target.base_url.rsplit(":", 1)[1]
        ray.get(model.release_inference.remote(permit), timeout=5)
    finally:
        ray.get(events.allow_replacement.remote(), timeout=5)
        ray.get(owner.shutdown.remote(), timeout=30)
        for actor in (events, owner, authority):
            ray.kill(actor, no_restart=True)
        if started:
            ray.shutdown()
