# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Isolated real Serve wiring check; inference/model/workload boundaries are
CPU stubs."""

import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import ray
from ray import serve

from relax.components.inference_gateway import InferenceGatewayService
from relax.engine.inference.types import (
    LifecycleState,
    ModelSnapshot,
    ReplicaSnapshot,
    Role,
    RoleSnapshot,
    RoutingSpec,
)


def setup():
    from relax.components import rollout
    from relax.utils.data import processing_utils

    class Workload:
        def __init__(self, *args, **kwargs):
            self.step = 0
            self._healthy = True

        async def stop(self):
            pass

        async def run(self):
            pass

        def get_step(self):
            return self.step

        def set_step(self, step):
            self.step = step

    rollout.RolloutWorkload = Workload
    processing_utils.load_tokenizer = lambda *a, **k: SimpleNamespace(apply_chat_template=lambda *a, **k: [1, 2])


class HTTP(BaseHTTPRequestHandler):
    def do_POST(self):
        data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        assert data["input_ids"] == [1, 2]
        output = json.dumps({"text": " accepted "}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(output)))
        self.end_headers()
        self.wfile.write(output)

    def log_message(self, *args):
        pass


@ray.remote
class Owner:
    def __init__(self, upstream):
        self.upstream = upstream
        self.pending = set()
        self.roles = []

    def snapshot(self, role):
        role = Role(role)
        return RoleSnapshot(
            role,
            "smoke",
            models=(
                ModelSnapshot(
                    "judge",
                    (ReplicaSnapshot("judge/0", LifecycleState.READY, self.upstream, direct_eligible=True),),
                    state=LifecycleState.READY,
                    admission=True,
                ),
            ),
            routing=RoutingSpec(default_model="judge"),
        )

    def admit_request(self, role, model, trace):
        self.roles.append(str(role))
        self.pending.add(trace)
        return trace

    def complete_request(self, permit):
        self.pending.discard(permit)

    def counts(self):
        return len(self.roles), len(self.pending)


if __name__ == "__main__":
    import requests

    server = ThreadingHTTPServer(("127.0.0.1", 0), HTTP)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    ray.init(
        address="local",
        num_cpus=4,
        include_dashboard=False,
        log_to_driver=False,
        runtime_env={"worker_process_setup_hook": setup},
    )
    try:
        serve.start(http_options={"host": "127.0.0.1", "port": port})
        owner = Owner.remote(f"http://127.0.0.1:{server.server_address[1]}")
        config = SimpleNamespace(
            _rollout_engine_manager=owner,
            _genrm_instances_resolved={"judge": {"model_path": "stub", "sampling_config": {}}},
        )
        for role in ("rollout", "genrm", "teacher"):
            serve.run(InferenceGatewayService.bind(None, None, 0, config, role), name=role, route_prefix=f"/{role}")
            url = f"http://127.0.0.1:{port}/{role}"
            discovery = requests.get(url + "/engines", timeout=30)
            assert discovery.status_code == 200, discovery.text
            assert discovery.json()["schema_version"] == 2
            assert discovery.json()["role"] == role
            body = {"messages": []} if role == "genrm" else {"input_ids": [1, 2]}
            response = requests.post(url + "/generate", json=body, timeout=30)
            assert response.status_code == 200, response.text
            assert response.json() == ({"response": "accepted"} if role == "genrm" else {"text": " accepted "})
        assert ray.get(owner.counts.remote()) == (3, 0)
        print("REAL_SERVE_THREE_ROLES_OK", flush=True)
    finally:
        serve.shutdown()
        ray.shutdown()
        server.shutdown()
