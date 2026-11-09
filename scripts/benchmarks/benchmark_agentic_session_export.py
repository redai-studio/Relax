# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""CPU comparison of real AgenticSessionShard exports under long and short
load.

The Shard, Forest, Sample and export artifacts are real; only inference, the
agent process and request permits are deterministic fakes. Each measured run
uses one Shard event loop and a separate client loop, with long sessions exiting
in waves while short chat and health requests are sent open-loop. Request
latency starts at its scheduled time, so queueing is included. Export fields are
hashed after protocol-5 serialization and compared across fresh baseline and
candidate processes.

Example::

    python scripts/benchmarks/benchmark_agentic_session_export.py compare \
      --baseline-src /path/to/base --candidate-src . --tokenizer /path/to/tokenizer \
      --preset long --set groups=2 --set turns=96 --set sessions_per_group=8 \
      --set window_s=12 --set wave_interval_s=4 --repeats 3 --out-dir /tmp/unique-run
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import hashlib
import json
import math
import os
import pickle
import statistics
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import Any

import numpy as np
import psutil


_SHARD_ALLOCATOR_ENV = {"MALLOC_ARENA_MAX": "2", "MALLOC_TRIM_THRESHOLD_": "0"}
_CHARS_PER_TOKEN = 4
_CORPUS_CHARS = 1 << 21
_CORPUS = (
    np.frombuffer(b"abcdefghijklmnopqrstuvwxyz   ", dtype=np.uint8)[
        np.random.default_rng(7).integers(0, 29, size=_CORPUS_CHARS)
    ]
    .tobytes()
    .decode("ascii")
)
_ARRAY_FIELDS = (
    ("tokens", np.int64),
    ("rollout_tokens", np.int64),
    ("loss_mask", np.int64),
    ("rollout_log_probs", np.float64),
    ("rollout_topk_token_ids", np.int64),
    ("rollout_topk_log_probs", np.float64),
    ("rollout_routed_experts", np.int64),
)
_REQUIRED_FIELDS = {"tokens", "loss_mask", "rollout_log_probs"}
_SCALARS = ("prompt", "response", "response_length", "custom_advantage", "status", "reward", "abort_count")


def _logger():
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if root not in sys.path:
        sys.path.insert(0, root)
    from relax.utils.logging_utils import get_logger

    return get_logger(__name__)


def _bootstrap_src(src: str) -> str:
    root = os.path.realpath(src)
    if not os.path.isdir(os.path.join(root, "relax")):
        raise SystemExit(f"--src {src!r} does not contain a relax/ package")
    sys.path.insert(0, root)
    import relax

    loaded = os.path.realpath(os.path.dirname(os.path.dirname(relax.__file__)))
    if os.path.normcase(loaded) != os.path.normcase(root):
        raise SystemExit(f"relax was imported from {loaded}, expected {root}")
    return root


def _percentiles(samples_s: list[float]) -> dict[str, float]:
    if not samples_s:
        return {"count": 0}
    values = np.asarray(samples_s, dtype=np.float64) * 1e3
    return {
        "count": int(values.size),
        "p50": float(np.percentile(values, 50)),
        "p99": float(np.percentile(values, 99)),
        "max": float(values.max()),
    }


class LagProbe:
    """Count each 2 ms deadline in the formal window, including missed
    slots."""

    def __init__(self, tick_s: float) -> None:
        self.tick_s, self.slots = tick_s, []
        self._start = self._end = 0.0
        self._task: asyncio.Task[None] | None = None

    async def start(self, *, window_s: float) -> float:
        loop = asyncio.get_running_loop()
        self._start = loop.time()
        self._end = self._start + window_s
        self._task = loop.create_task(self._run(), name="bench-lag-probe")
        return self._start

    async def _run(self) -> None:
        loop = asyncio.get_running_loop()
        due = self._start + self.tick_s
        while due < self._end:
            await asyncio.sleep(max(0.0, due - loop.time()))
            now = loop.time()
            if now < due:
                continue
            while due <= now and due < self._end:
                self.slots.append(now - due)
                due += self.tick_s

    async def stop(self) -> list[float]:
        if self._task is not None:
            await self._task
        return self.slots


class RssSampler(threading.Thread):
    def __init__(self) -> None:
        super().__init__(name="bench-rss-sampler", daemon=True)
        self._halt = threading.Event()
        self._proc = psutil.Process()
        self.start_rss = self.peak_rss = self._proc.memory_info().rss

    def run(self) -> None:
        while not self._halt.wait(0.02):
            self.peak_rss = max(self.peak_rss, self._proc.memory_info().rss)

    def finish(self) -> dict[str, float]:
        self._halt.set()
        self.join()
        self.peak_rss = max(self.peak_rss, self._proc.memory_info().rss)
        return {"start_mib": self.start_rss / (1 << 20), "peak_mib": self.peak_rss / (1 << 20)}


def _seed(*parts: Any) -> int:
    return int.from_bytes(hashlib.sha256("/".join(map(str, parts)).encode()).digest()[:8], "little")


def _text(seed: int, chars: int) -> str:
    if chars >= _CORPUS_CHARS:
        raise ValueError(f"message of {chars} chars exceeds {_CORPUS_CHARS}-char corpus")
    start = seed % (_CORPUS_CHARS - chars)
    return "x" + _CORPUS[start : start + chars - 1]


def _content_digest(payload: dict[str, Any]) -> str:
    """Hash benchmark-listed training arrays/scalars, excluding timestamps."""
    resolved = dict(payload)
    if resolved.get("_rollout_tokens_shared"):
        resolved["rollout_tokens"] = resolved.get("tokens")
    digest = hashlib.sha256()
    for name, dtype in _ARRAY_FIELDS:
        value = resolved.get(name)
        digest.update(name.encode() + (b"\1" if name in resolved else b"\0"))
        if name in _REQUIRED_FIELDS and value is None:
            raise ValueError(f"required export field {name!r} is missing or None")
        if value is not None:
            try:
                array = np.ascontiguousarray(np.asarray(value, dtype=dtype))
            except (TypeError, ValueError) as error:
                raise ValueError(f"export field {name!r} is not a comparable array") from error
            digest.update(repr(array.shape).encode() + array.tobytes())
    metadata = resolved.get("metadata") or {}
    scalars = {key: resolved.get(key) for key in _SCALARS}
    scalars.update(
        weight_versions=resolved.get("weight_versions"),
        tools=metadata.get("tools"),
        chat_template_kwargs=metadata.get("chat_template_kwargs"),
        rollout_turns=metadata.get("rollout_turns"),
    )
    digest.update(json.dumps(scalars, sort_keys=True, default=str).encode())
    return digest.hexdigest()


def _expected_probe_count(rate: float, window_s: float) -> int:
    if not math.isfinite(rate) or not math.isfinite(window_s) or rate <= 0 or window_s <= 0:
        return -1
    return math.ceil(rate * window_s)


def _run_validation_errors(result: dict[str, Any]) -> list[str]:
    errors = []
    if result.get("valid") is not True or result.get("consumer_completed") is not True:
        errors.append("export count/window/content checks failed")
    exports = result.get("exports") or {}
    expected, completed = exports.get("expected", -1), exports.get("completed", -1)
    if expected <= 0 or completed != expected or result.get("exports_in_window") is not True:
        errors.append(f"exports incomplete or late: expected={expected}, completed={completed}")
    digests = result.get("digests") or {}
    if len(digests) != completed or any(not values for values in digests.values()):
        errors.append("export digest count mismatch")
    if any(d.startswith(("not-exported:", "invalid-content:")) for values in digests.values() for d in values):
        errors.append("an export lacked required training fields")
    if (result.get("loop_lag_ms") or {}).get("count", 0) <= 0:
        errors.append("event-loop lag window produced no samples")
    scenario, probes = result.get("scenario") or {}, result.get("probe_counts") or {}
    for kind in ("short", "health"):
        expected = _expected_probe_count(float(scenario.get(f"{kind}_rps", 0)), float(result.get("window_s", 0)))
        counts = probes.get(kind) or {}
        if expected <= 0 or counts.get("expected") != expected:
            errors.append(f"invalid expected {kind} request count")
        for field in ("scheduled", "attempted", "completed", "succeeded"):
            if counts.get(field) != expected:
                errors.append(f"{kind} requests {field}={counts.get(field)}, expected={expected}")
        if counts.get("failed") != 0 or result.get(f"{kind}_errors", 0) != 0:
            errors.append(f"{kind} requests failed")
        if (result.get(f"{kind}_ms") or {}).get("count") != expected:
            errors.append(f"{kind} latency sample count differs from scheduled requests")
    teardown = result.get("teardown") or {}
    for field in ("pending_tasks", "groups", "session_records", "inflight_exports", "export_worker_threads"):
        if teardown.get(field) != 0:
            errors.append(f"teardown {field}={teardown.get(field)}")
    if teardown.get("leaked_threads") != []:
        errors.append(f"teardown leaked_threads={teardown.get('leaked_threads')}")
    return errors


@dataclasses.dataclass
class Scenario:
    groups: int = 4
    sessions_per_group: int = 8
    turns: int = 64
    init_tokens: int = 2048
    obs_tokens: int = 1024
    resp_tokens: int = 256
    wave_interval_s: float = 4.0
    window_s: float = 0.0
    short_sessions: int = 8
    short_rps: float = 50.0
    short_resp_tokens: int = 16
    health_rps: float = 20.0
    decode_delay_s: float = 0.002
    lag_tick_s: float = 0.002
    export_workers: int = 0

    @property
    def window(self) -> float:
        return self.window_s or self.groups * self.wave_interval_s + 8.0


PRESETS = {
    "smoke": {"groups": 2, "sessions_per_group": 2, "turns": 8, "obs_tokens": 256, "window_s": 6.0},
    "default": {},
    "long": {"groups": 2, "turns": 96, "obs_tokens": 1536, "resp_tokens": 128},
}


def build_scenario(preset: str, overrides: list[str]) -> Scenario:
    scenario = Scenario(**PRESETS[preset])
    defaults = {field.name: getattr(scenario, field.name) for field in dataclasses.fields(scenario)}
    for value in overrides:
        key, sep, raw = value.partition("=")
        if not sep or key not in defaults:
            raise SystemExit(f"invalid scenario override {value!r}; choose from {sorted(defaults)}")
        setattr(scenario, key, type(defaults[key])(raw))
    return scenario


def load_tokenizer(path: str) -> tuple[Any, np.ndarray]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(path)
    excluded = set(tokenizer.get_added_vocab().values()) | set(tokenizer.all_special_ids)
    pool = np.fromiter((i for i in tokenizer.get_vocab().values() if i not in excluded), dtype=np.int64)
    return tokenizer, np.sort(pool)


class FakeCompiler:
    def __init__(self, pool: np.ndarray) -> None:
        self.apply_chat_template_kwargs, self.processor = {}, None
        self.cpu_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="bench-compiler")
        self.pool = pool

    def _encode(self, messages: list[dict[str, Any]]) -> list[int]:
        text = json.dumps(messages, ensure_ascii=False, sort_keys=True)
        rng = np.random.default_rng(_seed("encode", hashlib.sha256(text.encode()).hexdigest()))
        return rng.choice(self.pool, size=max(1, len(text) // _CHARS_PER_TOKEN)).tolist()

    async def encode_messages(self, messages: list[dict[str, Any]], **_: Any) -> Any:
        from relax.agentic.pipeline.runtime import EncodedMessages

        ids = await asyncio.get_running_loop().run_in_executor(self.cpu_executor, self._encode, messages)
        return EncodedMessages(train_prompt_ids=ids, backend_prompt_ids=list(ids))

    encode_observation_delta = encode_messages


class FakeBackend:
    def __init__(self, tokenizer: Any, pool: np.ndarray, scenario: Scenario) -> None:
        self.tokenizer, self.pool, self.delay = tokenizer, pool, scenario.decode_delay_s
        self.compiler = FakeCompiler(pool)

    async def generate(self, *, input_ids: list[int], sampling_params: dict[str, Any], request_id: str, **_: Any):
        from relax.agentic.pipeline.runtime import BackendGenerateResult

        await asyncio.sleep(self.delay)
        count = int(sampling_params["max_new_tokens"])
        rng = np.random.default_rng(_seed("generate", request_id))
        return BackendGenerateResult(
            new_tokens=rng.choice(self.pool, size=count).tolist(),
            new_log_probs=(-rng.random(count)).tolist(),
            finish_type="stop",
            meta_info={
                "finish_reason": {"type": "stop"},
                "weight_version": "1",
                "prompt_tokens": len(input_ids),
                "completion_tokens": count,
            },
            elapsed=self.delay,
        )

    async def abort_request(self, request_id: str) -> None:
        del request_id

    async def close_session(self, session_id: str, *, timeout_s: float) -> bool:
        del session_id, timeout_s
        return True

    async def shutdown(self) -> None:
        self.compiler.cpu_executor.shutdown(wait=True)


class FakeAgentProcess:
    def __init__(self) -> None:
        self.done, self.raw, self._output = asyncio.Event(), "", None

    def finish(self, raw: str) -> None:
        self.raw = raw
        self.done.set()

    async def wait(self) -> None:
        await self.done.wait()
        if self._output is None:
            from relax.agentic.runner.ipc import _session_output_from_text

            self._output = _session_output_from_text(self.raw)

    @property
    def output(self) -> Any:
        return self._output

    async def set_timeout_active(self, active: bool) -> None:
        del active

    async def terminate_and_join(self) -> None:
        self.done.set()


class FakeLauncher:
    def __init__(self, *_: Any) -> None:
        self.processes: dict[str, FakeAgentProcess] = {}

    def start_agent(self, session_input: Any) -> FakeAgentProcess:
        process = FakeAgentProcess()
        self.processes[session_input.session_id] = process
        return process


class _Remote:
    def __init__(self, value: Any) -> None:
        self.value = value

    def remote(self, *_: Any) -> asyncio.Future[Any]:
        future = asyncio.get_running_loop().create_future()
        future.set_result(self.value)
        return future


class FakeLimiter:
    acquire_request_permit, release_request_permit, cancel_request_permit = _Remote(True), _Remote(None), _Remote(None)


def build_args() -> SimpleNamespace:
    return SimpleNamespace(
        sglang_router_ip="127.0.0.1",
        sglang_router_port=1,
        rollout_temperature=1.0,
        rollout_top_p=1.0,
        rollout_top_k=-1,
        rollout_max_response_len=1024,
        rollout_stop=None,
        rollout_stop_token_ids=None,
        rollout_skip_special_tokens=False,
        sglang_enable_deterministic_inference=False,
        rollout_seed=1234,
        rollout_max_context_len=None,
        eval_max_context_len=None,
        agentic_session_lifecycle=False,
        agentic_program_admission=False,
        agentic_admission_scope="train",
        agentic_admission_expected_decode_cap=None,
        agentic_reasoning_parser=None,
        agentic_tool_call_parser=None,
        partial_rollout=False,
        partial_rollout_max_aborted_count=None,
        fully_async=False,
        use_slime_router=False,
        use_rollout_routing_replay=False,
        num_layers=None,
        moe_router_topk=None,
        hf_checkpoint="bench",
    )


def make_shard(backend: FakeBackend, launcher: FakeLauncher, args: SimpleNamespace, workers: int) -> Any:
    from relax.agentic.session import service

    if workers and hasattr(service, "_SESSION_EXPORT_WORKERS"):
        service._SESSION_EXPORT_WORKERS = workers
    service.SGLangBackendAdapter = lambda _args: backend
    service.ManagedAgentLauncher = lambda *_: launcher
    service.load_agent_app_spec_from_args = lambda _args: None
    service.resolve_chat_api_base_url = lambda: "http://127.0.0.1:1/"
    return service.AgenticSessionShard.__ray_metadata__.modified_class(args, FakeLimiter())


async def _shard_leaks(shard: Any, keep: set[Any]) -> dict[str, Any]:
    tasks = [
        task
        for task in asyncio.all_tasks()
        if task not in keep and task is not asyncio.current_task() and not task.done()
    ]
    executor = getattr(shard, "_export_executor", None)
    return {
        "pending_tasks": len(tasks),
        "groups": len(shard._groups),
        "session_records": len(shard._session_records),
        "inflight_exports": len(getattr(shard, "_inflight_exports", ())),
        "export_worker_threads": sum(thread.is_alive() for thread in getattr(executor, "_threads", ())),
    }


def _transport(results: tuple[Any, ...]) -> tuple[float, list[list[str]], list[int]]:
    buffers: list[pickle.PickleBuffer] = []
    wire = pickle.dumps(results, protocol=5, buffer_callback=buffers.append)
    received = pickle.loads(wire, buffers=[bytes(buffer.raw()) for buffer in buffers])
    taken = time.monotonic()
    digests, token_counts = [], []
    for result in received:
        exports = getattr(result, "exports", None)
        if exports is None:
            digests.append([f"not-exported:{type(result).__name__}"])
            token_counts.append(0)
            continue
        current, count = [], 0
        for export in exports:
            payload = export.get("sample_payload") or {}
            try:
                current.append(_content_digest(payload))
            except ValueError as error:
                current.append(f"invalid-content:{error}")
            token_values = payload.get("tokens")
            count += len(token_values) if token_values is not None else 0
        digests.append(current)
        token_counts.append(count)
    return taken, digests, token_counts


def _batch_throughput(
    exit_at: dict[str, float], taken_at: dict[str, float], groups: list[list[str]]
) -> dict[str, float]:
    completed = [sid for group in groups for sid in group if sid in taken_at]
    elapsed = max((taken_at[sid] for sid in completed), default=0.0) - min(
        (exit_at[sid] for sid in completed), default=0.0
    )
    return {
        "completed": len(completed),
        "elapsed_s": elapsed,
        "per_s": len(completed) / elapsed if elapsed > 0 else 0.0,
    }


class Bench:
    """One process; Shard loop is here, simulated Ray callers use a second
    loop."""

    def __init__(self, scenario: Scenario, tokenizer: Any, pool: np.ndarray) -> None:
        self.sc = scenario
        self.shard_loop = asyncio.get_running_loop()
        self.backend, self.launcher = FakeBackend(tokenizer, pool, scenario), FakeLauncher()
        self.shard = make_shard(self.backend, self.launcher, build_args(), scenario.export_workers)
        self.client_loop = asyncio.new_event_loop()
        self.client_thread = threading.Thread(target=self.client_loop.run_forever, name="bench-client", daemon=True)
        self.transport = ThreadPoolExecutor(max_workers=1, thread_name_prefix="bench-transport")
        self.threads_before = {thread.ident for thread in threading.enumerate()}
        self.exit_at: dict[str, float] = {}
        self.taken_at: dict[str, float] = {}
        self.digests: dict[str, list[str]] = {}
        self.tokens: dict[str, int] = {}
        self.latency = {kind: [] for kind in ("short", "health")}
        self.errors = {kind: [] for kind in ("short", "health")}
        self.probe_counts = {
            kind: {key: 0 for key in ("scheduled", "attempted", "completed", "succeeded", "failed")}
            for kind in ("short", "health")
        }
        self.window_start, self.main_task = 0.0, None

    async def on_shard(self, coro: Any) -> Any:
        return await asyncio.wrap_future(asyncio.run_coroutine_threadsafe(coro, self.shard_loop))

    def _initial_messages(self, sid: str, tokens: int) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": "You are a benchmark agent."},
            {"role": "user", "content": _text(_seed(sid, "init"), tokens * _CHARS_PER_TOKEN)},
        ]

    async def _chat(self, sid: str, messages: list[dict[str, str]], tokens: int) -> dict[str, Any]:
        return await self.on_shard(
            self.shard.chat(
                session_id=sid,
                messages=messages,
                tools=[],
                chat_template_kwargs={},
                max_completion_tokens=tokens,
                stop=None,
                seed=None,
            )
        )

    async def _agent(self, sid: str, turns: int) -> None:
        history = self._initial_messages(sid, self.sc.init_tokens)
        for turn in range(turns):
            result = await self._chat(sid, list(history), self.sc.resp_tokens)
            if result.get("error"):
                raise RuntimeError(f"{sid} turn {turn}: {result['error']}")
            history.append(result["message"])
            if turn + 1 < turns:
                history.append({"role": "user", "content": _text(_seed(sid, "obs", turn), self.sc.obs_tokens * 4)})

    async def _build_group(self, kind: str, index: int, count: int, turns: int = 0) -> list[str]:
        from relax.agentic.pipeline import SessionSpec

        group_id = f"{kind}-g{index}"
        ids = [f"{group_id}-s{i}" for i in range(count)]
        specs = tuple(
            SessionSpec(session_id=sid, group_index=index, index=i, input_payload={"bench": sid})
            for i, sid in enumerate(ids)
        )
        await self.on_shard(self.shard.start_group("train", group_id, specs))
        if kind == "short":
            agents = [
                asyncio.create_task(self._chat(sid, self._initial_messages(sid, 32), self.sc.short_resp_tokens))
                for sid in ids
            ]
        else:
            agents = [asyncio.create_task(self._agent(sid, turns or 1)) for sid in ids]
        revision = -1
        while True:
            progress = await self.on_shard(self.shard.wait_for_state_change(revision))
            revision = progress.revision
            if any(group.group_id == group_id and group.ready_for_lease for group in progress.groups):
                break
        await self.on_shard(self.shard.lease_group(group_id, 1))
        await asyncio.gather(*agents)
        return ids

    def _exit(self, sid: str) -> None:
        self.exit_at[sid] = time.monotonic()
        self.shard_loop.call_soon_threadsafe(
            self.launcher.processes[sid].finish, json.dumps({"reward": 1.0, "metadata": {"bench_session": sid}})
        )

    async def _consume(self, expected: set[str]) -> None:
        pending, revision = set(expected), -1
        loop = asyncio.get_running_loop()
        while pending:
            progress = await self.on_shard(self.shard.wait_for_state_change(revision))
            revision = progress.revision
            for group in progress.groups:
                ready = tuple(sid for sid in group.takeable_session_ids if sid in pending)
                if ready:
                    pending.difference_update(ready)
                    results = await self.on_shard(self.shard.take_session_results(group.group_id, ready))
                    taken, digests, tokens = await loop.run_in_executor(self.transport, _transport, results)
                    for sid, digest, count in zip(ready, digests, tokens, strict=True):
                        self.taken_at[sid], self.digests[sid], self.tokens[sid] = taken, digest, count

    async def _probe(self, kind: str, due: float, sid: str) -> None:
        counts = self.probe_counts[kind]
        counts["attempted"] += 1
        try:
            if kind == "short":
                result = await self._chat(sid, self._initial_messages(sid, 32), self.sc.short_resp_tokens)
                ok = not result.get("error")
            else:
                result = await self.on_shard(self.shard.health())
                ok = isinstance(result, dict) and result.get("ok") is True
            if not ok:
                self.errors[kind].append("request returned not-ok")
                counts["failed"] += 1
            else:
                counts["succeeded"] += 1
        except Exception as error:
            self.errors[kind].append(f"{type(error).__name__}: {error}")
            counts["failed"] += 1
        finally:
            counts["completed"] += 1
            self.latency[kind].append(time.monotonic() - due)

    async def _schedule(self, kind: str, rate: float, until: float, ids: list[str]) -> int:
        tasks, index, period = [], 0, 1.0 / rate
        while self.window_start + index * period < until:
            due = self.window_start + index * period
            await asyncio.sleep(max(0.0, due - time.monotonic()))
            tasks.append(asyncio.create_task(self._probe(kind, due, ids[index % len(ids)])))
            index += 1
        self.probe_counts[kind]["scheduled"] = index
        await asyncio.gather(*tasks)
        return index

    async def _waves(self, groups: list[list[str]]) -> None:
        for index, ids in enumerate(groups):
            due = self.window_start + 1.0 + index * self.sc.wave_interval_s
            await asyncio.sleep(max(0.0, due - time.monotonic()))
            for sid in ids:
                self._exit(sid)

    async def _teardown(self, group_ids: list[str]) -> dict[str, Any]:
        for group_id in group_ids:
            await self.on_shard(self.shard.drop_group(group_id))
        await self.on_shard(self.shard.shutdown())
        return await self.on_shard(_shard_leaks(self.shard, {self.main_task}))

    async def client_main(self) -> dict[str, Any]:
        sc = self.sc
        await self.on_shard(self.shard.resume_generation(1))
        started = time.monotonic()
        long_groups = await asyncio.gather(
            *(self._build_group("long", i, sc.sessions_per_group, sc.turns) for i in range(sc.groups))
        )
        build_s = time.monotonic() - started
        group_ids = [f"long-g{i}" for i in range(sc.groups)]
        long_ids = {sid for group in long_groups for sid in group}
        warm = await self._build_group("warm", 0, 1, max(2, sc.turns // 4))
        short = await self._build_group("short", 0, sc.short_sessions)
        group_ids += ["warm-g0", "short-g0"]
        consumer = asyncio.create_task(self._consume(long_ids | set(warm)))
        for sid in warm:
            self._exit(sid)
        while any(sid not in self.taken_at for sid in warm):
            if consumer.done():
                await consumer
            await asyncio.sleep(0.02)
        await asyncio.sleep(1.0)

        lag = LagProbe(sc.lag_tick_s)
        rss = RssSampler()
        cpu0 = time.process_time()
        rss.start()
        self.window_start = await self.on_shard(lag.start(window_s=sc.window))
        until = self.window_start + sc.window
        await asyncio.gather(
            self._schedule("short", sc.short_rps, until, short),
            self._schedule("health", sc.health_rps, until, short),
            self._waves(long_groups),
        )
        lag_samples = await self.on_shard(lag.stop())
        try:
            await asyncio.wait_for(asyncio.shield(consumer), timeout=120 + 20 * sc.window)
            consumer_ok = True
        except asyncio.TimeoutError:
            consumer.cancel()
            await asyncio.gather(consumer, return_exceptions=True)
            consumer_ok = False
        memory = rss.finish()
        cpu_s, sampled_s = time.process_time() - cpu0, time.monotonic() - self.window_start
        teardown = await self._teardown(group_ids)
        completed = [sid for sid in long_ids if sid in self.taken_at]
        throughput = _batch_throughput(self.exit_at, self.taken_at, long_groups)
        in_window = bool(completed) and max(self.taken_at[sid] for sid in completed) <= until
        return {
            "scenario": dataclasses.asdict(sc),
            "window_s": sc.window,
            "build_s": build_s,
            "valid": len(completed) == len(long_ids) and in_window and consumer_ok,
            "exports_in_window": in_window,
            "consumer_completed": consumer_ok,
            "loop_lag_ms": _percentiles(lag_samples),
            "short_ms": _percentiles(self.latency["short"]),
            "health_ms": _percentiles(self.latency["health"]),
            "short_errors": len(self.errors["short"]),
            "health_errors": len(self.errors["health"]),
            "probe_counts": {
                kind: {"expected": _expected_probe_count(getattr(sc, f"{kind}_rps"), sc.window), **counts}
                for kind, counts in self.probe_counts.items()
            },
            "exports": {
                "expected": len(long_ids),
                "completed": len(completed),
                "latency_ms": _percentiles([self.taken_at[sid] - self.exit_at[sid] for sid in completed]),
                "batch_elapsed_s": throughput["elapsed_s"],
                "batch_wallclock_per_s": throughput["per_s"],
                "mean_tokens": statistics.fmean(self.tokens[sid] for sid in completed) if completed else 0,
            },
            "measurement_scope": {"configured_window_s": sc.window, "cpu_rss_elapsed_s": sampled_s},
            "cpu": {"process_s": cpu_s},
            "memory": memory,
            "teardown": teardown,
            "digests": {sid: self.digests[sid] for sid in sorted(completed)},
        }


async def _shard_main(scenario: Scenario, tokenizer: Any, pool: np.ndarray) -> dict[str, Any]:
    bench = Bench(scenario, tokenizer, pool)
    bench.main_task = asyncio.current_task()
    bench.client_thread.start()
    try:
        future = asyncio.run_coroutine_threadsafe(bench.client_main(), bench.client_loop)
        result = await asyncio.wrap_future(future)
    finally:
        bench.client_loop.call_soon_threadsafe(bench.client_loop.stop)
        bench.client_thread.join()
        bench.client_loop.close()
        bench.transport.shutdown(wait=True)
    result["teardown"]["leaked_threads"] = sorted(
        thread.name for thread in threading.enumerate() if thread.ident not in bench.threads_before
    )
    return result


def _write_json(path: str | None, result: dict[str, Any]) -> None:
    text = json.dumps(result, indent=2, sort_keys=True)
    if path:
        with open(path, "x", encoding="utf-8") as stream:
            stream.write(text)
    else:
        sys.stdout.write(text + "\n")


def run_main(ns: argparse.Namespace) -> None:
    if ns.out and os.path.exists(ns.out):
        raise SystemExit(f"refusing to overwrite result file {ns.out!r}")
    root = _bootstrap_src(ns.src)
    logger = _logger()
    scenario = build_scenario(ns.preset, ns.set or [])
    missing = {key: value for key, value in _SHARD_ALLOCATOR_ENV.items() if os.environ.get(key) != value}
    if missing:
        logger.warning("benchmark allocator environment differs from production: %s", missing)
    tokenizer, pool = load_tokenizer(ns.tokenizer)
    try:
        result = asyncio.run(_shard_main(scenario, tokenizer, pool))
    except Exception as error:
        result = {
            "src": root,
            "python": sys.version.split()[0],
            "valid": False,
            "validation_errors": [f"benchmark execution failed: {type(error).__name__}: {error}"],
        }
        _write_json(ns.out, result)
        logger.error("benchmark failed; diagnostics saved to %s", ns.out or "stdout")
        raise SystemExit(2) from error
    result.update(src=root, python=sys.version.split()[0])
    result["validation_errors"] = _run_validation_errors(result)
    result["valid"] = not result["validation_errors"]
    _write_json(ns.out, result)
    if not result["valid"]:
        logger.error("invalid benchmark; diagnostics saved to %s", ns.out or "stdout")
        raise SystemExit(2)


_METRICS = (
    ("event-loop lag P99 (ms)", ("loop_lag_ms", "p99")),
    ("short chat P99 (ms)", ("short_ms", "p99")),
    ("health P99 (ms)", ("health_ms", "p99")),
    ("completed exports", ("exports", "completed")),
    ("export latency P99 (ms)", ("exports", "latency_ms", "p99")),
    ("batch export time (s)", ("exports", "batch_elapsed_s")),
    ("batch throughput (exports/s)", ("exports", "batch_wallclock_per_s")),
    ("process CPU (s)", ("cpu", "process_s")),
    ("peak RSS (MiB)", ("memory", "peak_mib")),
)


def _get(result: dict[str, Any], path: tuple[str, ...]) -> float:
    value: Any = result
    for key in path:
        value = value.get(key, 0.0) if isinstance(value, dict) else 0.0
    return float(value)


def _workload_scenario(scenario: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in scenario.items() if key != "export_workers"}


def _write_comparison_invalid(out_dir: str, errors: list[str], runs: dict[str, list[dict[str, Any]]]) -> None:
    diagnostic = {
        "valid": False,
        "errors": errors,
        "run_files": sorted(name for name in os.listdir(out_dir) if name.endswith(".json")),
        "runs": {
            variant: [
                {"valid": run.get("valid"), "validation_errors": run.get("validation_errors", [])}
                for run in variant_runs
            ]
            for variant, variant_runs in runs.items()
        },
    }
    with open(os.path.join(out_dir, "comparison-invalid.json"), "x", encoding="utf-8") as stream:
        json.dump(diagnostic, stream, indent=2, sort_keys=True)


def _comparison_errors(runs: dict[str, list[dict[str, Any]]]) -> list[str]:
    errors, reference_scenario, reference_digests = [], None, None
    for variant, variant_runs in runs.items():
        for index, result in enumerate(variant_runs):
            errors.extend(f"{variant}[{index}]: {error}" for error in _run_validation_errors(result))
            scenario = _workload_scenario(result.get("scenario") or {})
            if reference_scenario is None:
                reference_scenario = scenario
            elif scenario != reference_scenario:
                errors.append(f"{variant}[{index}]: workload differs from other runs")
            digests = json.dumps(result.get("digests") or {}, sort_keys=True)
            if reference_digests is None:
                reference_digests = digests
            elif digests != reference_digests:
                errors.append(f"{variant}[{index}]: exported training fields differ")
    return errors


def compare_main(ns: argparse.Namespace) -> None:
    logger = _logger()
    if os.path.exists(ns.out_dir):
        raise SystemExit(f"refusing to reuse result directory {ns.out_dir!r}; choose a new path")
    os.makedirs(ns.out_dir)
    runs: dict[str, list[dict[str, Any]]] = {"baseline": [], "candidate": []}
    errors = []
    if ns.repeats <= 0:
        errors.append(f"repeats must be positive, got {ns.repeats}")
    invalid = [
        item
        for item in (ns.candidate_set or [])
        if item.partition("=")[0] != "export_workers" or not item.partition("=")[1]
    ]
    if invalid:
        errors.append("candidate-only overrides may change export_workers only: " + ", ".join(invalid))
    if errors:
        _write_comparison_invalid(ns.out_dir, errors, runs)
        raise SystemExit(2)

    env = {**os.environ, **_SHARD_ALLOCATOR_ENV}
    sources = {"baseline": ns.baseline_src, "candidate": ns.candidate_src}
    for repeat in range(ns.repeats):
        order = ("baseline", "candidate") if repeat % 2 == 0 else ("candidate", "baseline")
        for variant in order:
            output = os.path.join(ns.out_dir, f"{variant}-{repeat}.json")
            command = [
                sys.executable,
                os.path.abspath(__file__),
                "run",
                "--src",
                sources[variant],
                "--tokenizer",
                ns.tokenizer,
                "--preset",
                ns.preset,
                "--out",
                output,
            ]
            for value in [*(ns.set or []), *((ns.candidate_set or []) if variant == "candidate" else [])]:
                command += ["--set", value]
            logger.info("benchmark run %s/%s: %s", repeat + 1, ns.repeats, variant)
            try:
                completed = subprocess.run(command, env=env, check=False, timeout=ns.timeout)
            except subprocess.TimeoutExpired:
                _write_comparison_invalid(ns.out_dir, [f"{variant}[{repeat}] exceeded timeout {ns.timeout}s"], runs)
                raise SystemExit(2)
            if os.path.exists(output):
                with open(output, encoding="utf-8") as stream:
                    runs[variant].append(json.load(stream))
            if completed.returncode != 0:
                _write_comparison_invalid(ns.out_dir, [f"{variant}[{repeat}] exited {completed.returncode}"], runs)
                raise SystemExit(2)

    errors = _comparison_errors(runs)
    if errors:
        _write_comparison_invalid(ns.out_dir, errors, runs)
        raise SystemExit(2)
    lines = [
        f"preset={ns.preset}; shared overrides={ns.set or []}; candidate-only={ns.candidate_set or []}; repeats={ns.repeats}",
        "",
        "| metric | baseline median [min, max] | candidate median [min, max] | change |",
        "| --- | --- | --- | --- |",
    ]
    for label, path in _METRICS:
        values = {variant: [_get(run, path) for run in runs[variant]] for variant in runs}
        medians = {variant: statistics.median(items) for variant, items in values.items()}
        cells = {
            variant: f"{medians[variant]:.2f} [{min(items):.2f}, {max(items):.2f}]"
            for variant, items in values.items()
        }
        change = (medians["candidate"] - medians["baseline"]) / medians["baseline"] * 100 if medians["baseline"] else 0
        lines.append(f"| {label} | {cells['baseline']} | {cells['candidate']} | {change:+.1f}% |")
    lines += ["", "all runs valid: True", "benchmark-listed export fields identical across all runs: True"]
    report = "\n".join(lines) + "\n"
    with open(os.path.join(ns.out_dir, "summary.md"), "x", encoding="utf-8") as stream:
        stream.write(report)
    logger.info("comparison report saved to %s", os.path.join(ns.out_dir, "summary.md"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "compare"):
        command = commands.add_parser(name)
        command.add_argument("--tokenizer", required=True)
        command.add_argument("--preset", default="default", choices=sorted(PRESETS))
        command.add_argument("--set", action="append", metavar="FIELD=VALUE")
        if name == "run":
            command.add_argument("--src", required=True)
            command.add_argument("--out", help="write this run's JSON; existing files are never overwritten")
        else:
            command.add_argument("--baseline-src", required=True)
            command.add_argument("--candidate-src", required=True)
            command.add_argument("--candidate-set", action="append", metavar="FIELD=VALUE")
            command.add_argument("--repeats", type=int, default=5)
            command.add_argument("--out-dir", required=True, help="a new, unused result directory")
            command.add_argument("--timeout", type=float, default=1800.0)
    args = parser.parse_args()
    (run_main if args.command == "run" else compare_main)(args)


if __name__ == "__main__":
    main()
