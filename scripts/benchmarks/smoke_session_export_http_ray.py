# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Exercise production Session routing/export through local HTTP and Ray.

The backend/compiler and Agent process are deterministic CPU fixtures. The
Serve ingress, Ray actors/ObjectRefs, Session lifecycle, assembly and cleanup
are real. Run as a module from the repository root; no existing cluster is
used.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import socket
import tempfile
import threading
import time
from argparse import Namespace
from collections import Counter
from pathlib import Path
from typing import Any, Awaitable, Callable
from uuid import uuid4

import httpx
import numpy as np
import psutil
import ray
from ray import serve

from relax.agentic.pipeline import RuntimeGroupError, SessionExportTransport, SessionSpec, TrainingFieldArtifact
from relax.agentic.pipeline.runtime import BackendGenerateResult, EncodedMessages
from relax.agentic.runner import SessionOutput
from relax.agentic.session.admission_coordinator import AdmissionCoordinator
from relax.agentic.session.service import AgenticChatAPIService, AgenticSessionShard, SessionPhase
from scripts.benchmarks.benchmark_session_export import ReplayTokenizer, content_digest, distribution, make_shard


SHARD_NAME = "agentic_session_shard_0"
OWNED_TASK_PREFIXES = ("agent-process-", "session-finalize:", "session-finish:", "drop-group:", "chat:")


def smoke_args() -> Namespace:
    return Namespace(
        hf_checkpoint="smoke-model",
        agentic_session_lifecycle=True,
        use_rollout_routing_replay=False,
        num_layers=2,
        moe_router_topk=2,
        fully_async=False,
        partial_rollout=False,
        use_slime_router=False,
        agentic_admission_scope="all",
        agentic_program_admission=False,
        agentic_reasoning_parser=None,
        agentic_tool_call_parser=None,
        rollout_max_context_len=262144,
        eval_max_context_len=262144,
        rollout_max_response_len=2,
        rollout_temperature=1.0,
        rollout_top_p=1.0,
        rollout_top_k=-1,
        rollout_stop=None,
        rollout_stop_token_ids=None,
        rollout_skip_special_tokens=False,
        sglang_enable_deterministic_inference=False,
        sglang_router_ip=None,
        sglang_router_port=None,
        agentic_admission_max_wait_s=5.0,
        agentic_admission_headroom=0.1,
        agentic_admission_pressure_threshold=0.9,
        sglang_server_concurrency=32,
        rollout_num_gpus=1,
        rollout_num_gpus_per_engine=1,
    )


def response_tokens(length: int) -> list[int]:
    return [79, 75] if length == 2 else [i % 1024 for i in range(length)]


def response_logprobs(length: int) -> list[float]:
    return [-0.125 - (i % 7) / 16 for i in range(length)]


class MockCompiler:
    apply_chat_template_kwargs: dict[str, Any] = {}

    async def encode_messages(self, messages: Any, **kwargs: Any) -> EncodedMessages:
        return EncodedMessages(train_prompt_ids=[31] * 32, backend_prompt_ids=[31] * 32)

    encode_observation_delta = encode_messages


class MockBackend:
    def __init__(self) -> None:
        self.tokenizer = ReplayTokenizer()
        self.compiler = MockCompiler()
        self.closed: list[str] = []
        self.stopped = False

    async def generate(self, *, input_ids: list[int], sampling_params: dict[str, Any], **kwargs: Any) -> Any:
        await asyncio.sleep(0.002)
        length = sampling_params["max_new_tokens"]
        return BackendGenerateResult(
            new_tokens=response_tokens(length),
            new_log_probs=response_logprobs(length),
            finish_type="stop",
            meta_info={
                "prompt_tokens": len(input_ids),
                "completion_tokens": length,
                "weight_version": "v1",
                "finish_reason": {"type": "stop"},
            },
            elapsed=0.002,
            new_topk_token_ids=np.tile(np.arange(4, dtype=np.int32), (length, 1)),
            new_topk_log_probs=np.full((length, 4), -0.5, np.float32),
        )

    async def abort_request(self, request_id: str) -> None:
        pass

    async def close_session(self, session_id: str, **kwargs: Any) -> bool:
        self.closed.append(session_id)
        return True

    async def shutdown(self) -> None:
        self.stopped = True


class ControlledProcess:
    def __init__(self) -> None:
        self.output = SessionOutput(reward=1.0, metadata={"source": "http-ray-smoke"})
        self.exited = asyncio.Event()
        self.terminations = 0

    async def wait(self) -> None:
        await self.exited.wait()

    async def set_timeout_active(self, active: bool) -> None:
        pass

    async def terminate_and_join(self) -> None:
        self.terminations += 1
        self.exited.set()


class ControlledLauncher:
    def __init__(self) -> None:
        self.processes: dict[str, ControlledProcess] = {}

    def start_agent(self, session_input: Any) -> ControlledProcess:
        process = ControlledProcess()
        self.processes[session_input.session_id] = process
        return process


class ObservedCoordinator(AdmissionCoordinator.__ray_metadata__.modified_class):
    async def smoke_permits(self) -> int:
        return len(self._active_request_permits)


class SmokeShard(AgenticSessionShard.__ray_metadata__.modified_class):
    """Inject constructor fixtures; keep production lifecycle methods
    intact."""

    def __init__(self, mode: str, coordinator: Any) -> None:
        self.__dict__.update(make_shard(mode).__dict__)
        self.args = smoke_args()
        self._generation_backend = MockBackend()
        self._agent_launcher = ControlledLauncher()
        self._sglang_request_limiter = coordinator
        self._admission_client = None
        self._gate: tuple[threading.Event, threading.Event] | None = None
        self._fail_next = False
        self._materializations = 0
        self._rpc_tasks: dict[str, asyncio.Task[Any]] = {}
        self._exited_rpcs: set[str] = set()

    async def create(self, number: int, length: int) -> tuple[str, str]:
        session_id = f"{SHARD_NAME}.session-{number}"
        group_id = f"group-{number}"
        await self.start_group(
            "train",
            group_id,
            (
                SessionSpec(
                    session_id=session_id,
                    index=number,
                    group_index=number,
                    label="smoke",
                    metadata={"fixture": "http-ray"},
                    train_metadata={"loss": "grpo"},
                    sampling_params={"max_new_tokens": length},
                ),
            ),
        )
        return group_id, session_id

    async def ready(self, group_id: str) -> bool:
        return self._groups[group_id].first_request_barrier_crossed

    async def finish_agents(self, session_ids: list[str]) -> None:
        for session_id in session_ids:
            self._agent_launcher.processes[session_id].exited.set()

    async def collect(self, groups: list[tuple[str, str]]) -> tuple[Any, ...]:
        while not all(self._groups[group_id].terminal for group_id, _ in groups):
            await self.wait_for_state_change(self._state_revision)
        outcomes = []
        for group_id, session_id in groups:
            outcomes.extend(await self.take_session_results(group_id, (session_id,)))
        return tuple(outcomes)

    async def direct_export(self, session_id: str) -> None:
        await self._observe_rpc("export", self._finalize_terminal_session(self._session_records[session_id]))

    async def shutdown(self) -> None:
        await self._observe_rpc("shutdown", super().shutdown())

    async def _observe_rpc(self, name: str, work: Awaitable[None]) -> None:
        task = asyncio.current_task()
        assert task is not None
        self._rpc_tasks[name] = task
        self._exited_rpcs.discard(name)
        try:
            await work
        finally:
            self._exited_rpcs.add(name)

    async def cancellation_checkpoint(self, name: str) -> dict[str, Any] | None:
        task = self._rpc_tasks.get(name)
        if task is None or not task.cancelling():
            return None
        # Task.cancel() queues its wakeup before this checkpoint yields. Let
        # that wakeup run so an ordinary shield can expose premature return.
        await asyncio.sleep(0)
        return {
            **await self.diagnostics(),
            "rpc_done": task.done() or name in self._exited_rpcs,
            "retained_sessions": [
                session.session_id
                for session in self._session_records.values()
                if session.phase is SessionPhase.FINALIZING
                and session.lock.locked()
                and session.forest is not None
                and session.resources is not None
                and not session.result_cell.terminal
                and session.session_id not in self._generation_backend.closed
                and self._agent_launcher.processes[session.session_id].terminations == 0
            ],
        }

    async def arm_gate(self) -> None:
        self._gate = threading.Event(), threading.Event()

    async def gate_started(self) -> bool:
        return self._gate is not None and self._gate[0].is_set()

    async def release_gate(self) -> None:
        if self._gate is not None:
            self._gate[1].set()

    async def fail_next_export(self) -> None:
        self._fail_next = True

    def _build_session_export_transport(self, **kwargs: Any) -> SessionExportTransport:
        self._materializations += 1
        gate = self._gate
        if gate is not None:
            gate[0].set()
            if not gate[1].wait(30):
                raise TimeoutError("Smoke materialization gate was not released")
        if self._fail_next:
            self._fail_next = False
            raise ValueError("smoke materialization failure")
        return super()._build_session_export_transport(**kwargs)

    async def diagnostics(self) -> dict[str, Any]:
        snapshot = await self.debug_state()
        return {
            **snapshot,
            "pid": os.getpid(),
            "materializations": self._materializations,
            "gate_blocked": self._gate is not None and self._gate[0].is_set() and not self._gate[1].is_set(),
            "finalizing_sessions": sum(
                session.phase is SessionPhase.FINALIZING for session in self._session_records.values()
            ),
            "export_threads": sum(t.name.startswith("session-export") for t in threading.enumerate()),
            "owned_tasks": [
                task.get_name()
                for task in asyncio.all_tasks()
                if not task.done() and task.get_name().startswith(OWNED_TASK_PREFIXES)
            ],
            "closed": dict(Counter(self._generation_backend.closed)),
            "terminated": {key: process.terminations for key, process in self._agent_launcher.processes.items()},
            "backend_stopped": self._generation_backend.stopped,
        }


async def until(predicate: Callable[[], Awaitable[Any]], description: str) -> Any:
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        value = await predicate()
        if value:
            return value
        await asyncio.sleep(0.01)
    raise TimeoutError(f"Timed out waiting for {description}")


async def post_chat(client: httpx.AsyncClient, session_id: str, messages: list[dict[str, str]], **kwargs: Any) -> Any:
    response = await client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {session_id}"},
        json={"model": "smoke-model", "messages": messages, **kwargs},
    )
    assert response.status_code == 200, response.text
    return response.json()


async def prepare(client: httpx.AsyncClient, shard: Any, number: int, length: int, turns: int = 1) -> tuple[str, str]:
    group_id, session_id = await shard.create.remote(number, length)
    messages = [{"role": "user", "content": "turn-0"}]
    first = asyncio.create_task(post_chat(client, session_id, messages))
    await until(lambda: shard.ready.remote(group_id), "first-request barrier")
    assert not first.done()
    await shard.lease_group.remote(group_id, 0)
    result = await first
    for turn in range(1, turns):
        messages.extend([result["choices"][0]["message"], {"role": "user", "content": f"turn-{turn}"}])
        result = await post_chat(client, session_id, messages)
    assert result["usage"]["completion_tokens"] == length
    return group_id, session_id


def check_export(outcome: Any, length: int, turns: int) -> str:
    assert isinstance(outcome, SessionExportTransport), repr(outcome)
    assert len(outcome.exports) == 1
    payload = outcome.exports[0]["sample_payload"]
    assert payload["tokens"].dtype == np.int32
    assert payload["loss_mask"].dtype == np.uint8
    assert payload["rollout_log_probs"].dtype == np.float64
    sample = TrainingFieldArtifact(payload).to_sample()
    tokens, logprobs = response_tokens(length), response_logprobs(length)
    assert sample.tokens == ([31] * 32 + tokens) * turns
    assert sample.loss_mask == [1] * length + ([0] * 32 + [1] * length) * (turns - 1)
    assert sample.rollout_log_probs == logprobs + ([0.0] * 32 + logprobs) * (turns - 1)
    assert sample.reward == 1.0 and sample.status.value == "completed"
    assert sample.rollout_tokens == sample.tokens
    topk = np.tile(np.arange(4, dtype=np.int32), (length, 1))
    expected_topk = np.concatenate([topk] + [np.concatenate([np.full((32, 4), -1, np.int32), topk])] * (turns - 1))
    np.testing.assert_array_equal(sample.rollout_topk_token_ids, expected_topk)
    expected_probs = np.full(expected_topk.shape, -0.5, np.float32)
    expected_probs[expected_topk == -1] = -np.inf
    np.testing.assert_array_equal(sample.rollout_topk_log_probs, expected_probs)
    trace = sample.metadata["agentic_trace"]
    assert trace["turn_count"] == turns
    assert all("chat_service_http_return_at" in turn["events"] for turn in trace["turns"])
    # Timings vary across real HTTP/Ray requests; every other Sample field is compared.
    training = sample.to_dict()
    training.pop("non_generation_time")
    training["metadata"] = {key: value for key, value in sample.metadata.items() if key != "agentic_trace"}
    return content_digest(training)


async def release_groups(shard: Any, groups: list[tuple[str, str]]) -> None:
    for group_id, _ in groups:
        await shard.release_group.remote(group_id)


async def assert_empty(shard: Any, coordinator: Any, *, stopped: bool = False) -> dict[str, Any]:
    state = await shard.diagnostics.remote()
    for key in (
        "group_count",
        "session_count",
        "retained_request_count",
        "backend_request_count",
        "live_process_count",
        "permit_cleanup_task_count",
    ):
        assert state[key] == 0, state
    assert not state["owned_tasks"], state
    assert all(count == 1 for count in state["closed"].values()), state
    assert all(count == 1 for count in state["terminated"].values()), state
    assert state["closed"].keys() == state["terminated"].keys(), state
    assert await coordinator.smoke_permits.remote() == 0
    if stopped:
        assert state["backend_stopped"] and state["export_threads"] == 0, state
    return state


async def assert_cancellation_pending(shard: Any, ref: Any, name: str, session_ids: list[str]) -> None:
    state = await until(lambda: shard.cancellation_checkpoint.remote(name), f"{name} cancellation delivery")
    assert not state["rpc_done"], f"Cancelled {name} RPC returned before materialization finished: {state}"
    assert state["gate_blocked"], state
    assert set(state["retained_sessions"]) == set(session_ids), state
    assert not ray.wait([ref], timeout=0)[0], f"Cancelled {name} RPC became ready before materialization finished"


async def exercise(client: httpx.AsyncClient, shard: Any, coordinator: Any, mode: str) -> dict[str, Any]:
    await shard.resume_generation.remote(0)
    warmup = await prepare(client, shard, 1000, 2)
    await shard.finish_agents.remote([warmup[1]])
    check_export((await shard.collect.remote([warmup]))[0], 2, 1)
    await release_groups(shard, [warmup])
    short = await prepare(client, shard, 900, 2)
    groups = [await prepare(client, shard, number, 16352, turns=8) for number in range(4)]
    short_latencies, health_latencies = [], []
    export_triggered = asyncio.Event()
    http_while_finalizing = asyncio.Event()
    start = time.monotonic()

    async def probe(number: int, health: bool) -> None:
        due = start + number * 0.025
        await asyncio.sleep(max(0, due - time.monotonic()))
        if health:
            response = await client.get("/healthz")
            assert response.status_code == 200 and response.json()["shards"][0]["ok"]
        else:
            result = await post_chat(client, short[1], [{"role": "user", "content": f"probe-{number}"}], logprobs=True)
            assert result["choices"][0]["message"]["content"] == "ok"
            assert len(result["choices"][0]["logprobs"]["content"]) == 2
        (health_latencies if health else short_latencies).append(time.monotonic() - due)
        if health and export_triggered.is_set() and not http_while_finalizing.is_set():
            # Optional observation: a later diagnostics RPC can miss a fast
            # export. Mandatory overlap checks use lifecycle gates below.
            if (await shard.diagnostics.remote())["finalizing_sessions"] > 0:
                http_while_finalizing.set()

    probes = [asyncio.create_task(probe(number, health)) for number in range(40) for health in (False, True)]
    await asyncio.sleep(0.05)
    export_triggered.set()
    await shard.finish_agents.remote([session_id for _, session_id in groups])
    outcomes = await shard.collect.remote(groups)
    export_wall = time.monotonic() - start - 0.05
    await asyncio.gather(*probes)
    assert len(short_latencies) == len(health_latencies) == 40
    digests = [check_export(outcome, 16352, 8) for outcome in outcomes]
    await release_groups(shard, groups)
    await shard.drop_group.remote(short[0])
    await assert_empty(shard, coordinator)
    result = {
        "mode": mode,
        "completed_exports": len(outcomes),
        "tokens_per_export": 131072,
        "short_requests": distribution(short_latencies),
        "health_requests": distribution(health_latencies),
        "export_wall_s": export_wall,
        "training_digests": digests,
        "http_while_finalizing": http_while_finalizing.is_set(),
    }
    if mode == "thread":
        result["lifecycle"] = await lifecycle_checks(client, shard, coordinator)
    else:
        await shard.shutdown.remote()
    state = await assert_empty(shard, coordinator, stopped=True)
    assert state["pid"] != os.getpid()
    result["cleanup"] = {
        key: state[key]
        for key in (
            "group_count",
            "session_count",
            "retained_request_count",
            "backend_request_count",
            "live_process_count",
            "permit_cleanup_task_count",
            "export_threads",
        )
    }
    return result


async def lifecycle_checks(client: httpx.AsyncClient, shard: Any, coordinator: Any) -> list[str]:
    # Disconnect a real pending SSE socket, then let Group cleanup own cancellation.
    group_id, session_id = await shard.create.remote(2000, 2)
    async with client.stream(
        "POST",
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {session_id}"},
        json={"messages": [{"role": "user", "content": "disconnect"}], "stream": True},
    ) as stream:
        assert stream.status_code == 200
        assert await anext(stream.aiter_lines()) == ": connected"
        await until(lambda: shard.ready.remote(group_id), "pending SSE request")
    await shard.drop_group.remote(group_id)
    await assert_empty(shard, coordinator)

    # A real Ray RPC cancellation must drain the active thread before cleanup.
    group = await prepare(client, shard, 2001, 2)
    await shard.arm_gate.remote()
    export_ref = shard.direct_export.remote(group[1])
    await until(lambda: shard.gate_started.remote(), "running export")
    ray.cancel(export_ref, force=False)
    await assert_cancellation_pending(shard, export_ref, "export", [group[1]])
    await shard.release_gate.remote()
    try:
        await export_ref
    except ray.exceptions.TaskCancelledError:
        pass
    else:
        raise AssertionError("Ray export cancellation was not propagated")
    check_export((await shard.collect.remote([group]))[0], 2, 1)
    await release_groups(shard, [group])
    await assert_empty(shard, coordinator)

    await shard.fail_next_export.remote()
    failed = await prepare(client, shard, 2002, 2)
    await shard.finish_agents.remote([failed[1]])
    outcome = (await shard.collect.remote([failed]))[0]
    assert isinstance(outcome, RuntimeGroupError) and "smoke materialization failure" in str(outcome)
    await shard.drop_group.remote(failed[0])
    good = await prepare(client, shard, 2003, 2)
    await shard.finish_agents.remote([good[1]])
    check_export((await shard.collect.remote([good]))[0], 2, 1)
    await release_groups(shard, [good])
    await assert_empty(shard, coordinator)

    for operation, number in (("drop", 2010), ("shutdown", 2020)):
        groups = [await prepare(client, shard, number + i, 2) for i in range(2)]
        await shard.arm_gate.remote()
        await shard.finish_agents.remote([session_id for _, session_id in groups])
        await until(lambda: shard.gate_started.remote(), "active and queued finalizers")

        async def both_finalizing() -> bool:
            return (await shard.diagnostics.remote())["finalizing_sessions"] == 2

        await until(both_finalizing, "queued finalizer admission")
        state = await shard.diagnostics.remote()
        assert state["session_count"] == 2
        if operation == "drop":
            refs = [shard.drop_group.remote(group_id) for group_id, _ in groups]
        else:
            refs = [shard.shutdown.remote()]

            async def shutdown_started() -> bool:
                return not (await shard.diagnostics.remote())["train_generation_open"]

            await until(shutdown_started, "shutdown ownership")
            ray.cancel(refs[0], force=False)
            await assert_cancellation_pending(shard, refs[0], "shutdown", [session_id for _, session_id in groups])
        response = await asyncio.wait_for(client.get("/healthz"), timeout=2)
        assert response.status_code == 200 and response.json()["ok"]
        # The gate stays closed across the real HTTP round trip, making this
        # overlap proof independent of export speed and diagnostics scheduling.
        state = await shard.diagnostics.remote()
        assert state["gate_blocked"] and state["finalizing_sessions"] == 2, state
        assert not ray.wait(refs, timeout=0)[0], "Cleanup returned while materialization was still blocked"
        await shard.release_gate.remote()
        cancellation_received = False
        for ref in refs:
            try:
                await ref
            except ray.exceptions.TaskCancelledError:
                assert operation == "shutdown"
                cancellation_received = True
        if operation == "shutdown":
            assert cancellation_received, "Ray shutdown cancellation was not propagated"
        await assert_empty(shard, coordinator, stopped=operation == "shutdown")
    response = await client.get("/debug_state")
    assert all(value == 0 for value in response.json()["totals"].values())
    discarded = await client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {groups[0][1]}"},
        json={"messages": [{"role": "user", "content": "after shutdown"}]},
    )
    assert discarded.status_code == 404
    return [
        "sse_disconnect_then_drop",
        "ray_export_cancel",
        "materialization_error_then_success",
        "drop_active_and_queued",
        "cancelled_shutdown_active_and_queued",
        "http_during_gated_export",
    ]


def run_smoke() -> dict[str, Any]:
    if ray.is_initialized():
        raise RuntimeError("Refusing to reuse or shut down an existing Ray runtime")
    with tempfile.TemporaryDirectory(prefix="t14-http-ray-") as root:
        result = None
        try:
            context = ray.init(
                address="local",
                namespace=f"task14-smoke-{uuid4().hex}",
                _temp_dir=root,
                num_cpus=2,
                num_gpus=0,
                include_dashboard=False,
                object_store_memory=128 * 1024 * 1024,
                log_to_driver=False,
            )
            assert Path(context.address_info["session_dir"]).is_relative_to(root)
            assert ray.cluster_resources().get("GPU", 0) == 0
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
            serve.start(http_options={"host": "127.0.0.1", "port": port})
            results = []
            for mode in ("inline", "thread"):
                coordinator = ray.remote(num_cpus=0.1)(ObservedCoordinator).remote(smoke_args())
                shard = (
                    ray.remote(num_cpus=0.5, max_concurrency=128)(SmokeShard)
                    .options(name=SHARD_NAME)
                    .remote(mode, coordinator)
                )
                application = f"task14-{mode}"
                try:
                    serve.run(
                        AgenticChatAPIService.options(
                            num_replicas=1, max_ongoing_requests=128, ray_actor_options={"num_cpus": 0.5}
                        ).bind(smoke_args(), ((SHARD_NAME, shard),), coordinator),
                        name=application,
                        route_prefix="/agentic",
                    )

                    async def run() -> Any:
                        async with httpx.AsyncClient(
                            base_url=f"http://127.0.0.1:{port}/agentic",
                            trust_env=False,
                            timeout=30,
                            limits=httpx.Limits(max_connections=128),
                        ) as client:
                            return await exercise(client, shard, coordinator, mode)

                    results.append(asyncio.run(run()))
                finally:
                    ray.get(shard.release_gate.remote(), timeout=10)
                    ray.get(shard.shutdown.remote(), timeout=30)
                    serve.delete(application)
                    ray.get(coordinator.shutdown.remote(), timeout=10)
                    ray.kill(shard, no_restart=True)
                    ray.kill(coordinator, no_restart=True)
            assert results[0]["training_digests"] == results[1]["training_digests"]
            result = {
                "ray_version": ray.__version__,
                "http_transport": "loopback TCP / production Ray Serve ingress",
                "num_gpus": 0,
                "training_content_equal": True,
                "runs": results,
            }
        finally:
            # In this fresh driver, descendants belong to this local runtime.
            # Capture before shutdown so reparented workers remain observable.
            owned_processes = psutil.Process().children(recursive=True)
            if ray.is_initialized():
                serve.shutdown()
            ray.shutdown()
            _, alive = psutil.wait_procs(owned_processes, timeout=10)
            assert not alive, f"Owned Ray processes survived shutdown: {[process.pid for process in alive]}"
        assert result is not None
        result["runtime_processes_remaining"] = 0
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    result = run_smoke()
    rendered = json.dumps(result, indent=2) + "\n"
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
