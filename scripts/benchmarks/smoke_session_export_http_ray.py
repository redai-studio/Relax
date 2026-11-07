# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Opt-in CPU integration assertions, not a performance benchmark.

The worker owns a private local Ray/Serve runtime. Only compiler, generation,
and Agent launcher dependencies are fixtures; Session and ingress methods run
unchanged. The outer ManagedProcess always reaps the entire owned tree.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import inspect
import json
import os
import selectors
import socket
import sys
import tempfile
import threading
import time
from argparse import Namespace
from collections import Counter
from pathlib import Path
from typing import Any
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
PREFIX = "SESSION_HTTP_RAY_SMOKE "


def fixture_args() -> Namespace:
    return Namespace(
        hf_checkpoint="cpu-integration-fixture",
        agentic_session_lifecycle=True,
        agentic_program_admission=False,
        agentic_admission_scope="all",
        agentic_admission_max_wait_s=3,
        agentic_admission_headroom=0.1,
        agentic_admission_pressure_threshold=0.9,
        sglang_server_concurrency=8,
        rollout_num_gpus=1,
        rollout_num_gpus_per_engine=1,
        sglang_router_ip=None,
        sglang_router_port=None,
        use_rollout_routing_replay=False,
        fully_async=False,
        partial_rollout=False,
        use_slime_router=False,
        rollout_max_context_len=256,
        eval_max_context_len=256,
        rollout_temperature=1.0,
        rollout_top_p=1.0,
        rollout_top_k=-1,
        rollout_max_response_len=2,
        rollout_stop=None,
        rollout_stop_token_ids=None,
        rollout_skip_special_tokens=False,
        sglang_enable_deterministic_inference=False,
        agentic_reasoning_parser=None,
        agentic_tool_call_parser=None,
    )


async def poll(call: Any, description: str) -> Any:
    async def wait() -> Any:
        while True:
            value = await call()
            if value:
                return value
            await asyncio.sleep(0.01)

    return await asyncio.wait_for(wait(), timeout=10)


async def exercise(ray_dir: str, basic: bool) -> dict[str, Any]:
    import httpx
    import numpy as np
    import ray
    from ray import serve

    from relax.agentic.pipeline import RuntimeGroupError, SessionSpec, TrainingFieldArtifact
    from relax.agentic.pipeline.runtime import BackendGenerateResult, EncodedMessages
    from relax.agentic.runner import SessionOutput
    from relax.agentic.session import service
    from relax.agentic.session.admission_coordinator import AdmissionCoordinator

    class Codec:
        def decode(self, values: list[int], **_kwargs: Any) -> str:
            return ":".join(map(str, values))

    class Compiler:
        apply_chat_template_kwargs: dict[str, Any] = {}

        async def encode_messages(self, *_args: Any, **_kwargs: Any) -> Any:
            return EncodedMessages(train_prompt_ids=[11, 17], backend_prompt_ids=[11, 17])

        encode_observation_delta = encode_messages

    class Backend:
        tokenizer = Codec()
        compiler = Compiler()

        def __init__(self) -> None:
            self.closed: list[str] = []
            self.stopped = False

        async def generate(self, **_kwargs: Any) -> Any:
            await asyncio.sleep(0)
            return BackendGenerateResult(
                new_tokens=[23, 29],
                new_log_probs=[-0.125, -0.25],
                finish_type="stop",
                meta_info={"prompt_tokens": 2, "completion_tokens": 2, "weight_version": "fixture"},
                elapsed=0.0,
                new_topk_token_ids=np.array([[23, 31], [29, 37]], np.int32),
                new_topk_log_probs=np.array([[-0.125, -1.0], [-0.25, -2.0]], np.float32),
            )

        async def abort_request(self, _request_id: str) -> None:
            return None

        async def close_session(self, session_id: str, **_kwargs: Any) -> bool:
            self.closed.append(session_id)
            return True

        async def shutdown(self) -> None:
            self.stopped = True

    class Process:
        def __init__(self) -> None:
            self.output = SessionOutput(reward=1.0, metadata={"fixture": "cpu-http"})
            self.finished = asyncio.Event()
            self.terminations = 0

        async def wait(self) -> None:
            await self.finished.wait()

        async def set_timeout_active(self, _active: bool) -> None:
            return None

        async def terminate_and_join(self) -> None:
            self.terminations += 1
            self.finished.set()

    class Launcher:
        def __init__(self) -> None:
            self.processes: dict[str, Any] = {}

        def start_agent(self, session_input: Any) -> Any:
            process = Process()
            self.processes[session_input.session_id] = process
            return process

    class Ledger(AdmissionCoordinator.__ray_metadata__.modified_class):
        async def acquire_request_permit(self, permit_id: str) -> bool:
            granted = await super().acquire_request_permit(permit_id)
            if granted:
                self.grants = getattr(self, "grants", 0) + 1
            return granted

        async def release_request_permit(self, permit_id: str) -> None:
            active = permit_id in self._active_request_permits
            await super().release_request_permit(permit_id)
            if active:
                self.releases = getattr(self, "releases", 0) + 1

        async def snapshot(self) -> dict[str, Any]:
            return {
                "active_permits": len(self._active_request_permits),
                "grants": getattr(self, "grants", 0),
                "releases": getattr(self, "releases", 0),
            }

    class Shard(service.AgenticSessionShard.__ray_metadata__.modified_class):
        def __init__(self, args: Any, coordinator: Any, route: str) -> None:
            backend, launcher = Backend(), Launcher()
            initialize = next(
                base.__dict__["__init__"]
                for base in type(self).__mro__
                if inspect.isfunction(base.__dict__.get("__init__"))
                and "SGLangBackendAdapter" in base.__dict__["__init__"].__globals__
            )
            # Invoke the production constructor beneath Ray's tracing wrapper.
            # Only dependency factories are replaced during this call.
            with patch.dict(
                initialize.__globals__,
                {
                    "SGLangBackendAdapter": lambda _args: backend,
                    "ManagedAgentLauncher": lambda *_args: launcher,
                    "load_agent_app_spec_from_args": lambda _args: None,
                    "resolve_chat_api_base_url": lambda: "http://unused.invalid",
                },
            ):
                initialize(self, args, coordinator)
            assert self._generation_backend is backend and self._agent_launcher is launcher
            self.route = route
            self.audit: dict[str, Any] = {}
            self.calls: dict[str, Any] = {}
            self.exited: set[str] = set()
            self.cancelled: set[str] = set()
            self.gate: Any = None
            self.fail_next = False
            self.builds: list[str] = []

        async def create(self, label: str, count: int = 1) -> dict[str, Any]:
            group_id = f"{self.route}-{label}"
            ids = tuple(f"{self.route}.session-{label}-{i}" for i in range(count))
            await self.start_group(
                "train",
                group_id,
                tuple(
                    SessionSpec(
                        session_id=sid, index=i, metadata={"label": label}, sampling_params={"max_new_tokens": 2}
                    )
                    for i, sid in enumerate(ids)
                ),
            )
            self.audit.update((sid, self._session_records[sid]) for sid in ids)
            return {"group": group_id, "sessions": ids}

        async def inspect_group(self, gid: str) -> dict[str, Any]:
            group = self._groups[gid]
            return {"ready": group.first_request_barrier_crossed, "terminal": group.terminal}

        async def finish(self, ids: Any) -> None:
            for sid in ids:
                self._agent_launcher.processes[sid].finished.set()

        async def arm(self) -> None:
            self.gate = (threading.Event(), threading.Event())

        async def unblock(self) -> None:
            if self.gate is not None:
                self.gate[1].set()

        async def inject_error(self) -> None:
            self.fail_next = True

        def _build_session_export_transport(self, **kwargs: Any) -> Any:
            self.builds.append(kwargs["session"].session_id)
            if self.gate is not None:
                self.gate[0].set()
                if not self.gate[1].wait(15):
                    raise TimeoutError("test export gate was not released")
            if self.fail_next:
                self.fail_next = False
                raise ValueError("injected smoke assembly failure")
            return super()._build_session_export_transport(**kwargs)

        async def finalize_rpc(self, sid: str) -> None:
            self.calls["finalize"] = asyncio.current_task()
            try:
                await self._finalize_terminal_session(self.audit[sid])
            finally:
                self.exited.add("finalize")

        async def shutdown(self) -> None:
            self.calls["shutdown"] = asyncio.current_task()
            self.exited.discard("shutdown")
            try:
                await super().shutdown()
            finally:
                self.exited.add("shutdown")

        async def observe_cancellation(self, label: str) -> bool:
            # CPython 3.10+ test observation only: do not replace shield/join.
            task = self.calls.get(label)
            waiter = getattr(task, "_fut_waiter", None)
            if waiter is None or waiter.done():
                return False

            def delivered(future: Any) -> None:
                if future.cancelled():
                    self.cancelled.add(label)

            waiter.add_done_callback(delivered)
            return True

        async def snapshot(self) -> dict[str, Any]:
            await asyncio.sleep(0)
            state = await self.debug_state()
            state.update(
                gate_started=self.gate is not None and self.gate[0].is_set(),
                gate_blocked=self.gate is not None and self.gate[0].is_set() and not self.gate[1].is_set(),
                retained=[
                    sid
                    for sid, s in self.audit.items()
                    if s.lock.locked()
                    and s.forest is not None
                    and s.resources is not None
                    and not s.result_cell.terminal
                ],
                audit_refs={
                    sid: {
                        "forest": session.forest is not None,
                        "resources": session.resources is not None,
                        "response_index": bool(session.resp_state_hash_by_request_id),
                        "live_requests": bool(session.live_irs),
                        "queued_requests": bool(session.queued_irs),
                    }
                    for sid, session in self.audit.items()
                },
                builds=list(self.builds),
                exited=sorted(self.exited),
                cancel_delivered=sorted(self.cancelled),
                closed=dict(Counter(self._generation_backend.closed)),
                terminations={sid: p.terminations for sid, p in self._agent_launcher.processes.items()},
                export_threads=[t.name for t in threading.enumerate() if t.name.startswith("session-export")],
                backend_stopped=self._generation_backend.stopped,
                owned_tasks=[
                    task.get_name()
                    for task in asyncio.all_tasks()
                    if not task.done()
                    and task.get_name().startswith(
                        ("agent-process-", "session-finalize:", "session-finish:", "drop-group:", "sglang-permit-")
                    )
                ],
            )
            return state

    if ray.is_initialized():
        raise RuntimeError("Smoke requires an uninitialized private Ray driver")
    import uuid

    identity = "session-smoke-" + uuid.uuid4().hex
    ray.init(
        address="local",
        namespace=identity,
        _temp_dir=ray_dir,
        num_cpus=2,
        num_gpus=0,
        include_dashboard=False,
        object_store_memory=128 * 1024**2,
        _node_ip_address="127.0.0.1",
        log_to_driver=False,
    )
    coordinator = ray.remote(num_cpus=0.1)(Ledger).remote(fixture_args())
    actor = (
        ray.remote(num_cpus=0.25, max_concurrency=64)(Shard)
        .options(name=identity)
        .remote(fixture_args(), coordinator, identity)
    )
    application = identity + "-ingress"
    scenarios: list[str] = []
    training_checked = 0

    def record(name: str) -> None:
        scenarios.append(name)
        print(PREFIX + json.dumps({"progress": name}), flush=True)

    try:
        with socket.socket() as socket_probe:
            socket_probe.bind(("127.0.0.1", 0))
            port = socket_probe.getsockname()[1]
        serve.start(http_options={"host": "127.0.0.1", "port": port})
        serve.run(
            service.AgenticChatAPIService.options(num_replicas=1, ray_actor_options={"num_cpus": 0.25}).bind(
                fixture_args(), ((identity, actor),), coordinator
            ),
            name=application,
            route_prefix="/session-smoke",
        )
        await actor.resume_generation.remote(0)
        async with httpx.AsyncClient(
            base_url=f"http://127.0.0.1:{port}/session-smoke", timeout=10, trust_env=False
        ) as client:

            async def prepare(label: str, count: int = 1) -> Any:
                group = await actor.create.remote(label, count)
                payload = json.loads(
                    (ROOT / "tests/agentic/fixtures/chat_completions/text_list.request.json").read_text()
                )
                payload.update(max_tokens=2, logprobs=True)
                requests = [
                    asyncio.create_task(
                        client.post("/v1/chat/completions", headers={"Authorization": f"Bearer {sid}"}, json=payload)
                    )
                    for sid in group["sessions"]
                ]

                async def ready() -> bool:
                    return (await actor.inspect_group.remote(group["group"]))["ready"]

                await poll(ready, "first request barrier")
                await actor.lease_group.remote(group["group"], 0)
                for response in await asyncio.gather(*requests):
                    assert response.status_code == 200, response.text
                    assert response.json()["usage"]["completion_tokens"] == 2
                return group

            async def terminal(group: Any) -> Any:
                async def done() -> bool:
                    return (await actor.inspect_group.remote(group["group"]))["terminal"]

                await poll(done, "terminal group")
                return await actor.take_session_results.remote(group["group"], tuple(group["sessions"]))

            async def clean(stopped: bool = False) -> dict[str, Any]:
                state = await actor.snapshot.remote()
                assert all(
                    state[key] == 0
                    for key in (
                        "group_count",
                        "session_count",
                        "retained_request_count",
                        "backend_request_count",
                        "live_process_count",
                        "permit_cleanup_task_count",
                    )
                ), state
                assert all(n == 1 for n in state["closed"].values()), state
                assert state["closed"].keys() == state["terminations"].keys(), state
                assert all(n == 1 for n in state["terminations"].values()), state
                ledger = await coordinator.snapshot.remote()
                assert ledger["active_permits"] == 0 and ledger["grants"] >= 1
                assert ledger["grants"] == ledger["releases"], ledger
                assert not state["owned_tasks"], state
                assert all(not any(refs.values()) for refs in state["audit_refs"].values()), state
                if stopped:
                    assert state["backend_stopped"] and not state["export_threads"], state
                return state

            async def successful(label: str) -> None:
                nonlocal training_checked
                group = await prepare(label)
                await actor.finish.remote(group["sessions"])
                (transport,) = await terminal(group)
                payload = transport.exports[0]["sample_payload"]
                assert payload["tokens"].dtype == np.int32
                assert payload["loss_mask"].dtype == np.uint8
                assert payload["rollout_log_probs"].dtype == np.float64
                sample = TrainingFieldArtifact(payload).to_sample()
                assert sample.tokens == sample.rollout_tokens == [11, 17, 23, 29]
                assert sample.loss_mask == [1, 1] and sample.rollout_log_probs == [-0.125, -0.25]
                np.testing.assert_array_equal(sample.rollout_topk_token_ids, [[23, 31], [29, 37]])
                assert sample.metadata["agentic_trace"]["turns"][0]["events"]["chat_service_http_return_at"] > 0
                before = copy.deepcopy(payload)
                sample.tokens[0] = -1
                sample.rollout_topk_token_ids[0, 0] = -1
                np.testing.assert_array_equal(payload["tokens"], before["tokens"])
                np.testing.assert_array_equal(payload["rollout_topk_token_ids"], before["rollout_topk_token_ids"])
                await actor.release_group.remote(group["group"])
                await clean()
                training_checked += 1

            await successful("normal")
            record("http_chat_and_training_roundtrip")
            print(PREFIX + json.dumps({"progress": "basic passed"}), flush=True)
            if not basic:
                pending = await actor.create.remote("sse-disconnect")
                request = json.loads(
                    (ROOT / "tests/agentic/fixtures/chat_completions/text_list.request.json").read_text()
                )
                request.update(max_tokens=2, stream=True)
                async with client.stream(
                    "POST",
                    "/v1/chat/completions",
                    headers={"Authorization": f"Bearer {pending['sessions'][0]}"},
                    json=request,
                ) as stream:
                    assert stream.status_code == 200
                    frames = stream.aiter_lines()
                    assert await asyncio.wait_for(frames.__anext__(), 2) == ": connected"

                    async def waiting_lease() -> bool:
                        return (await actor.inspect_group.remote(pending["group"]))["ready"]

                    await poll(waiting_lease, "SSE pending first request")
                # Socket close is not treated as proof of backend cancellation.
                await actor.drop_group.remote(pending["group"])
                await clean()
                record("sse_disconnect_then_explicit_drop")
                group = await prepare("rpc-cancel")
                await actor.arm.remote()
                ref = actor.finalize_rpc.remote(group["sessions"][0])

                async def blocked() -> bool:
                    return (await actor.snapshot.remote())["gate_started"]

                await poll(blocked, "active export gate")
                await poll(lambda: actor.observe_cancellation.remote("finalize"), "shield cancellation observation")
                ray.cancel(ref, force=False)

                async def delivered() -> Any:
                    state = await actor.snapshot.remote()
                    return state if "finalize" in state["cancel_delivered"] else None

                state = await poll(delivered, "RPC cancel delivery")
                assert (
                    state["gate_blocked"]
                    and state["retained"] == list(group["sessions"])
                    and "finalize" not in state["exited"]
                ), state
                assert all(sid not in state["closed"] and state["terminations"][sid] == 0 for sid in group["sessions"])
                assert not ray.wait([ref], timeout=0)[0]
                await actor.unblock.remote()
                try:
                    await ref
                except ray.exceptions.TaskCancelledError:
                    pass
                else:
                    raise AssertionError("cancelled export RPC unexpectedly succeeded")
                assert (await terminal(group)) == (None,)
                await actor.release_group.remote(group["group"])
                state = await clean()
                assert state["builds"].count(group["sessions"][0]) == 1, state
                record("delivered_ray_cancel_drains_export")
                await actor.inject_error.remote()
                group = await prepare("assembly-error")
                await actor.finish.remote(group["sessions"])
                (error,) = await terminal(group)
                assert isinstance(error, RuntimeGroupError) and "injected smoke" in str(error)
                await actor.drop_group.remote(group["group"])
                await successful("recovery")
                record("assembly_error_then_recovery")
                for operation in ("drop", "shutdown"):
                    group = await prepare(operation, 2)
                    await actor.arm.remote()
                    await actor.finish.remote(group["sessions"])

                    async def queued() -> Any:
                        state = await actor.snapshot.remote()
                        return (
                            state
                            if state["gate_blocked"] and set(state["retained"]) == set(group["sessions"])
                            else None
                        )

                    await poll(queued, "active and queued finalizers")
                    ref = actor.drop_group.remote(group["group"]) if operation == "drop" else actor.shutdown.remote()
                    if operation == "shutdown":
                        await poll(
                            lambda: actor.observe_cancellation.remote("shutdown"), "shutdown shield observation"
                        )
                        ray.cancel(ref, force=False)

                        async def shutdown_delivered() -> Any:
                            state = await actor.snapshot.remote()
                            return state if "shutdown" in state["cancel_delivered"] else None

                        await poll(shutdown_delivered, "shutdown cancel delivery")
                    response = await asyncio.wait_for(client.get("/healthz"), 2)
                    assert response.status_code == 200 and response.json()["ok"]
                    state = await actor.snapshot.remote()
                    assert state["gate_blocked"] and set(state["retained"]) == set(group["sessions"]), state
                    assert all(
                        sid not in state["closed"] and state["terminations"][sid] == 0 for sid in group["sessions"]
                    )
                    if operation == "shutdown":
                        assert "shutdown" not in state["exited"], state
                    assert not ray.wait([ref], timeout=0)[0]
                    await actor.unblock.remote()
                    try:
                        await ref
                    except ray.exceptions.TaskCancelledError:
                        assert operation == "shutdown"
                    else:
                        assert operation == "drop"
                    state = await clean(stopped=operation == "shutdown")
                    if operation == "shutdown":
                        built = [sid for sid in state["builds"] if sid in group["sessions"]]
                        assert len(built) == 1, state
                    record(operation + "_active_and_queued_http_responsive")
            await actor.shutdown.remote()
            final_state = await clean(stopped=True)
            debug = (await client.get("/debug_state")).json()
            assert not any(debug["totals"].values()), debug
        return {
            "gpu_count": ray.cluster_resources().get("GPU", 0),
            "scenarios": scenarios,
            "training_exports_checked": training_checked,
            "request_permits": await coordinator.snapshot.remote(),
            "final_state": final_state,
            "driver_pid": os.getpid(),
            "namespace": identity,
        }
    finally:
        failing = sys.exc_info()[0] is not None
        cleanup_errors = []
        for call in (actor.unblock.remote, actor.shutdown.remote, coordinator.shutdown.remote):
            try:
                await asyncio.wait_for(call(), 10)
            except BaseException as error:
                cleanup_errors.append(repr(error))
        for close in (
            lambda: serve.delete(application),
            lambda: ray.kill(actor, no_restart=True),
            lambda: ray.kill(coordinator, no_restart=True),
            serve.shutdown,
            ray.shutdown,
        ):
            try:
                close()
            except BaseException as error:
                cleanup_errors.append(repr(error))
        if cleanup_errors and not failing:
            raise RuntimeError(f"Private runtime cleanup failed: {cleanup_errors}")


def run_smoke(*, basic: bool = False, timeout: float = 120) -> dict[str, Any]:
    sys.path.insert(0, str(ROOT))
    from scripts.benchmarks.process_supervisor import ManagedProcess

    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="relax-http-smoke-") as ray_dir:
        command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--ray-dir", ray_dir]
        if basic:
            command.append("--basic")
        env = {
            **os.environ,
            "PYTHONPATH": str(ROOT),
            "CUDA_VISIBLE_DEVICES": "",
            "RAY_ADDRESS": "local",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        }
        result = None
        lines = []
        with ManagedProcess(command, cwd=ROOT, env=env) as process:
            with selectors.DefaultSelector() as selector:
                for pipe in (process.stdout, process.stderr):
                    selector.register(pipe, selectors.EVENT_READ)
                buffers = {pipe: b"" for pipe in (process.stdout, process.stderr)}
                while selector.get_map():
                    if time.monotonic() - started > timeout:
                        raise TimeoutError("private HTTP/Ray smoke exceeded total deadline")
                    for key, _mask in selector.select(0.05):
                        chunk = os.read(key.fileobj.fileno(), 65536)
                        if not chunk:
                            selector.unregister(key.fileobj)
                            continue
                        buffers[key.fileobj] += chunk
                        while b"\n" in buffers[key.fileobj]:
                            raw, buffers[key.fileobj] = buffers[key.fileobj].split(b"\n", 1)
                            line = raw.decode(errors="replace")
                            lines.append(line)
                            if line.startswith(PREFIX):
                                message = json.loads(line[len(PREFIX) :])
                                if "result" in message:
                                    result = message["result"]
                                else:
                                    print(line, flush=True)
                code = process.wait()
            if code != 0 or result is None:
                raise AssertionError("HTTP/Ray smoke failed:\n" + "\n".join(lines[-60:]))
            result["supervisor_pid"] = process.supervisor_pid
            result["reaped_descendants"] = process.reaped_descendants
        result["owned_processes_reaped"] = (
            not Path(f"/proc/{process.pid}").exists() and not Path(f"/proc/{process.supervisor_pid}").exists()
        )
        assert result["owned_processes_reaped"]
    result["private_directory_removed"] = not Path(ray_dir).exists()
    result["elapsed_s"] = time.monotonic() - started
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--ray-dir", help=argparse.SUPPRESS)
    parser.add_argument("--basic", action="store_true")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    if args.worker:
        sys.path.insert(0, str(ROOT))
        result = asyncio.run(exercise(args.ray_dir, args.basic))
        print(PREFIX + json.dumps({"result": result}), flush=True)
    else:
        result = run_smoke(basic=args.basic)
        rendered = json.dumps(result, indent=2) + "\n"
        if args.json_out:
            args.json_out.parent.mkdir(parents=True, exist_ok=True)
            args.json_out.write_text(rendered)
        print(rendered, end="")


if __name__ == "__main__":
    main()
