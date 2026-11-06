# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""CPU benchmark: how long-trajectory exports affect one SessionShard.

The benchmark drives the real ``AgenticSessionShard`` (real Forest commits, real
sample assembly, real ``SessionOutput`` parsing) against a mock SGLang backend
and mock agent processes. Threading mirrors a Ray async actor: the Shard runs on
the main thread's event loop, while a client thread plays Ray's core worker and
remote callers (agents, short-request probes, the Runtime result consumer) and
submits coroutines with ``run_coroutine_threadsafe``.

Each ``run`` is one process: warmup (build long trajectories, open short probe
sessions, one warmup export), then a measurement window in which long sessions
exit in waves while short chats and health checks are sent at a fixed rate.

* Request latency is measured from the scheduled send time, so queueing counts.
* Event-loop lag comes from a ticker inside the Shard loop. Every tick slot,
  including slots missed during a stall, contributes one lateness sample, so
  P99 is the delay seen by a callback arriving at a random time.
* Export latency is agent exit signal -> session result taken by the consumer.
* Exported training fields are digested, so ``compare`` can check that the
  baseline and candidate trees export identical content.

Usage::

    git archive origin/main relax | tar -x -C /tmp/relax-base
    python scripts/benchmarks/benchmark_agentic_session_export.py compare \\
        --baseline-src /tmp/relax-base --candidate-src . \\
        --tokenizer /path/to/hf_tokenizer_dir --preset default --repeats 5
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import cProfile
import dataclasses
import hashlib
import io
import json
import math
import os
import pickle
import pstats
import statistics
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import Any, Optional

import numpy as np
import psutil


# Mirrors _AGENTIC_SHARD_ALLOCATOR_ENV in relax/agentic/session/service.py; the
# allocator arena count changes how a second Python thread allocates memory.
_SHARD_ALLOCATOR_ENV = {"MALLOC_ARENA_MAX": "2", "MALLOC_TRIM_THRESHOLD_": "0"}
_CHARS_PER_TOKEN = 4
_CORPUS_CHARS = 1 << 21


def _bootstrap_src(src: str) -> str:
    """Import ``relax`` only from ``src`` (baseline or candidate tree)."""

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
    """Summarize seconds as milliseconds (linear-interpolated percentiles)."""

    if not samples_s:
        return {"count": 0}
    arr = np.asarray(samples_s, dtype=np.float64) * 1e3
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p99": float(np.percentile(arr, 99)),
        "p999": float(np.percentile(arr, 99.9)),
        "max": float(arr.max()),
    }


class LagProbe:
    """Event-loop lateness ticker that runs on the Shard loop."""

    def __init__(self, tick_s: float) -> None:
        self.tick_s = tick_s
        self.raw: list[float] = []
        self.slots: list[float] = []
        self._start_at = 0.0
        self._until = 0.0
        self._task: Optional[asyncio.Task[None]] = None

    async def start(self, *, window_s: float) -> float:
        """Start a fresh probe and return the formal window start timestamp."""

        loop = asyncio.get_running_loop()
        self._start_at = loop.time()
        self._until = self._start_at + window_s
        self._task = asyncio.get_running_loop().create_task(self._run(), name="bench-lag-probe")
        return self._start_at

    async def _run(self) -> None:
        loop = asyncio.get_running_loop()
        deadline = self._start_at + self.tick_s
        while deadline < self._until:
            await asyncio.sleep(max(0.0, deadline - loop.time()))
            now = loop.time()
            if now < deadline:
                continue
            self.raw.append(now - deadline)
            # Charge every slot that elapsed while the loop was stalled.
            while deadline <= now and deadline < self._until:
                self.slots.append(now - deadline)
                deadline += self.tick_s

    async def stop(self) -> tuple[list[float], list[float]]:
        if self._task is not None:
            await self._task
        return self.raw, self.slots


class RssSampler(threading.Thread):
    """Peak resident set size inside the measurement window."""

    def __init__(self, interval_s: float = 0.02) -> None:
        super().__init__(name="bench-rss-sampler", daemon=True)
        self._interval_s = interval_s
        self._halt = threading.Event()
        self._proc = psutil.Process()
        self.start_rss = self._proc.memory_info().rss
        self.peak_rss = self.start_rss

    def run(self) -> None:
        while not self._halt.wait(self._interval_s):
            self.peak_rss = max(self.peak_rss, self._proc.memory_info().rss)

    def finish(self) -> dict[str, float]:
        self._halt.set()
        self.join()
        self.peak_rss = max(self.peak_rss, self._proc.memory_info().rss)
        mib = 1 << 20
        return {"start_mib": self.start_rss / mib, "peak_mib": self.peak_rss / mib}


def _thread_cpu() -> dict[int, float]:
    """CPU seconds per native thread id (user + system)."""

    return {t.id: t.user_time + t.system_time for t in psutil.Process().threads()}


_CORPUS = (
    np.frombuffer(b"abcdefghijklmnopqrstuvwxyz   ", dtype=np.uint8)[
        np.random.default_rng(7).integers(0, 29, size=_CORPUS_CHARS)
    ]
    .tobytes()
    .decode("ascii")
)


def _seed(*parts: Any) -> int:
    return int.from_bytes(hashlib.sha256("/".join(map(str, parts)).encode()).digest()[:8], "little")


def _text(seed: int, chars: int) -> str:
    """Deterministic message text; length drives Forest hash and JSON cost."""

    if chars >= _CORPUS_CHARS:
        raise ValueError(f"message of {chars} chars exceeds the {_CORPUS_CHARS}-char corpus")
    start = seed % (_CORPUS_CHARS - chars)
    return "x" + _CORPUS[start : start + chars - 1]


# Export fields included in this benchmark's content identity check.
_DIGEST_ARRAYS = (
    ("tokens", np.int64),
    ("rollout_tokens", np.int64),
    ("loss_mask", np.int64),
    ("rollout_log_probs", np.float64),
    ("rollout_topk_token_ids", np.int64),
    ("rollout_topk_log_probs", np.float64),
    ("rollout_routed_experts", np.int64),
)
_DIGEST_SCALARS = ("prompt", "response", "response_length", "custom_advantage", "status", "reward", "abort_count")
_REQUIRED_DIGEST_ARRAYS = {"tokens", "loss_mask", "rollout_log_probs"}


def _content_digest(payload: dict[str, Any]) -> str:
    """Hash the listed benchmark fields, minus nondeterministic timestamps."""

    h = hashlib.sha256()
    resolved = dict(payload)
    if resolved.get("_rollout_tokens_shared"):
        resolved["rollout_tokens"] = resolved.get("tokens")
    for name, dtype in _DIGEST_ARRAYS:
        value = resolved.get(name)
        h.update(name.encode())
        h.update(b"\x01" if name in resolved else b"\x00")
        if name in _REQUIRED_DIGEST_ARRAYS and value is None:
            raise ValueError(f"required export field {name!r} is missing or None")
        if value is not None:
            try:
                arr = np.ascontiguousarray(np.asarray(value, dtype=dtype))
            except (TypeError, ValueError) as error:
                raise ValueError(f"export field {name!r} is not a comparable array") from error
            h.update(repr(arr.shape).encode())
            h.update(arr.tobytes())
    metadata = resolved.get("metadata") or {}
    scalars = {name: resolved.get(name) for name in _DIGEST_SCALARS}
    scalars["weight_versions"] = resolved.get("weight_versions")
    scalars["tools"] = metadata.get("tools")
    scalars["chat_template_kwargs"] = metadata.get("chat_template_kwargs")
    scalars["rollout_turns"] = metadata.get("rollout_turns")
    h.update(json.dumps(scalars, sort_keys=True, default=str).encode())
    return h.hexdigest()


def _expected_probe_count(rate: float, window_s: float) -> int:
    """Count fixed-rate sends with due times in ``[0, window_s)``."""

    if not math.isfinite(rate) or not math.isfinite(window_s) or rate <= 0 or window_s <= 0:
        return -1
    return int(math.ceil(rate * window_s))


def _run_validation_errors(result: dict[str, Any]) -> list[str]:
    """Fail closed on incomplete work, probe failures, bad payloads, or
    leaks."""

    errors: list[str] = []
    if result.get("valid") is not True:
        errors.append("export count/window/content checks failed")
    if result.get("consumer_completed") is False:
        errors.append("result consumer did not finish")

    exports = result.get("exports") or {}
    expected_exports = int(exports.get("expected", -1))
    completed_exports = int(exports.get("completed", -1))
    if expected_exports <= 0 or completed_exports != expected_exports:
        errors.append(f"exports incomplete: expected={expected_exports}, completed={completed_exports}")
    if result.get("exports_in_window") is not True:
        errors.append("not all exports were taken inside the configured window")
    digests = result.get("digests") or {}
    if len(digests) != completed_exports:
        errors.append(f"export digest count mismatch: completed={completed_exports}, digested={len(digests)}")
    flat_digests = [digest for values in digests.values() for digest in values]
    if any(not values for values in digests.values()):
        errors.append("an exported Session had no Sample payloads to compare")
    if any(digest.startswith(("not-exported:", "invalid-content:")) for digest in flat_digests):
        errors.append("an export was missing or lacked required benchmark content fields")
    if (result.get("loop_lag_ms") or {}).get("count", 0) <= 0:
        errors.append("event-loop lag window produced no deadline samples")

    scenario = result.get("scenario") or {}
    probe_counts = result.get("probe_counts") or {}
    for kind, rate_field, error_field in (
        ("short", "short_rps", "short_errors"),
        ("health", "health_rps", "health_errors"),
    ):
        expected = _expected_probe_count(float(scenario.get(rate_field, 0.0)), float(result.get("window_s", 0.0)))
        counts = probe_counts.get(kind) or {}
        if expected <= 0:
            errors.append(f"invalid expected {kind} probe count: {expected}")
            continue
        if counts.get("expected") != expected:
            errors.append(f"{kind} probe expectation mismatch: expected={expected}, reported={counts.get('expected')}")
        for field in ("scheduled", "attempted", "completed", "succeeded"):
            if counts.get(field) != expected:
                errors.append(f"{kind} probes {field}={counts.get(field)}, expected={expected}")
        if counts.get("failed") != 0 or result.get(error_field, 0) != 0:
            errors.append(f"{kind} requests failed: {result.get(error_field, counts.get('failed'))}")
        latency_count = (result.get(f"{kind}_ms") or {}).get("count", 0)
        if latency_count != expected:
            errors.append(f"{kind} latency sample count={latency_count}, expected={expected}")

    teardown = result.get("teardown") or {}
    for field in ("pending_tasks", "groups", "session_records", "inflight_exports", "export_worker_threads"):
        if teardown.get(field) != 0:
            errors.append(f"teardown {field}={teardown.get(field)}")
    if teardown.get("leaked_threads") != []:
        errors.append(f"teardown leaked_threads={teardown.get('leaked_threads')}")
    return errors


@dataclasses.dataclass
class Scenario:
    """Load shape; each wave releases one group of long sessions at once."""

    groups: int = 4
    sessions_per_group: int = 8
    turns: int = 64
    init_tokens: int = 2048
    obs_tokens: int = 1024
    resp_tokens: int = 256
    export_mode: str = "implicit"  # "implicit" leaf export or "explicit" JSONL records
    wave_interval_s: float = 4.0
    window_s: float = 0.0  # 0 = waves * interval + 8 s
    short_sessions: int = 8
    short_rps: float = 50.0
    short_resp_tokens: int = 16
    health_rps: float = 20.0
    decode_delay_s: float = 0.002
    r3_layers: int = 0  # > 0 enables rollout routing replay payloads
    r3_topk: int = 3
    lag_tick_s: float = 0.002
    export_workers: int = 0  # ablation: > 0 overrides _SESSION_EXPORT_WORKERS where it exists

    @property
    def window(self) -> float:
        return self.window_s or self.groups * self.wave_interval_s + 8.0

    @property
    def long_tokens(self) -> int:
        return self.init_tokens + self.turns * self.resp_tokens + (self.turns - 1) * self.obs_tokens


PRESETS: dict[str, dict[str, Any]] = {
    "smoke": {"groups": 2, "sessions_per_group": 4, "turns": 12, "obs_tokens": 512, "window_s": 10.0},
    "default": {},
    "long": {"groups": 2, "turns": 96, "obs_tokens": 1536, "resp_tokens": 128},
    "wide": {"groups": 6, "turns": 48, "obs_tokens": 768},
    "explicit": {"export_mode": "explicit"},
    "r3": {"turns": 32, "obs_tokens": 1024, "r3_layers": 6, "r3_topk": 4},
}


def build_scenario(preset: str, overrides: list[str]) -> Scenario:
    scenario = Scenario(**PRESETS[preset])
    defaults = {f.name: getattr(scenario, f.name) for f in dataclasses.fields(Scenario)}
    for item in overrides:
        key, _, raw = item.partition("=")
        if key not in defaults:
            raise SystemExit(f"unknown scenario field {key!r}; choose from {sorted(defaults)}")
        setattr(scenario, key, type(defaults[key])(raw))
    if scenario.r3_layers and (scenario.r3_layers * scenario.r3_topk * 4) % 3:
        raise SystemExit("r3_layers * r3_topk must be a multiple of 3 (base64 row alignment)")
    return scenario


def load_tokenizer(path: str) -> tuple[Any, np.ndarray]:
    """Real HF tokenizer plus the ids of its ordinary (non-added) tokens."""

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(path)
    added = set(tokenizer.get_added_vocab().values()) | set(tokenizer.all_special_ids)
    pool = np.fromiter((i for i in tokenizer.get_vocab().values() if i not in added), dtype=np.int64)
    return tokenizer, np.sort(pool)


class FakeCompiler:
    """SGLangMessageCompiler stand-in: deterministic ids, executor hop."""

    def __init__(self, token_pool: np.ndarray) -> None:
        self.apply_chat_template_kwargs: dict[str, Any] = {}
        self.processor = None
        self.cpu_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="bench-compiler")
        self._pool = token_pool

    def _encode_sync(self, messages: list[dict[str, Any]]) -> list[int]:
        text = json.dumps(messages, ensure_ascii=False, sort_keys=True)
        rng = np.random.default_rng(_seed("encode", hashlib.sha256(text.encode()).hexdigest()))
        count = max(1, len(text) // _CHARS_PER_TOKEN)
        return rng.choice(self._pool, size=count).tolist()

    async def encode_messages(self, messages: list[dict[str, Any]], **_: Any) -> Any:
        from relax.agentic.pipeline.runtime import EncodedMessages

        ids = await asyncio.get_running_loop().run_in_executor(self.cpu_executor, self._encode_sync, messages)
        return EncodedMessages(train_prompt_ids=ids, backend_prompt_ids=list(ids))

    encode_observation_delta = encode_messages


class FakeBackend:
    """SGLangBackendAdapter subset used by the Shard; SGLang-shaped results."""

    def __init__(self, tokenizer: Any, token_pool: np.ndarray, scenario: Scenario) -> None:
        self.tokenizer = tokenizer
        self.compiler = FakeCompiler(token_pool)
        self._pool = token_pool
        self._delay_s = scenario.decode_delay_s
        self._r3_row_chars = 0
        if scenario.r3_layers:
            import pybase64

            rows = scenario.long_tokens + 64 * scenario.turns + 1024
            raw = np.random.default_rng(11).integers(0, 64, size=(rows, scenario.r3_layers, scenario.r3_topk))
            self._r3_b64 = pybase64.b64encode(raw.astype(np.int32).tobytes()).decode("ascii")
            self._r3_rows = rows
            self._r3_row_chars = scenario.r3_layers * scenario.r3_topk * 4 * 4 // 3

    async def generate(self, *, input_ids: list[int], sampling_params: dict[str, Any], request_id: str, **_: Any):
        from relax.agentic.pipeline.runtime import BackendGenerateResult

        await asyncio.sleep(self._delay_s)
        count = int(sampling_params["max_new_tokens"])
        rng = np.random.default_rng(_seed("generate", request_id))
        tokens = rng.choice(self._pool, size=count).tolist()
        meta: dict[str, Any] = {
            "finish_reason": {"type": "stop"},
            "weight_version": "1",
            "prompt_tokens": len(input_ids),
            "completion_tokens": count,
        }
        if self._r3_row_chars:
            rows = len(input_ids) + count - 1
            if rows > self._r3_rows:
                raise RuntimeError(f"R3 buffer has {self._r3_rows} rows, request needs {rows}")
            meta["routed_experts"] = self._r3_b64[: rows * self._r3_row_chars]
        return BackendGenerateResult(
            new_tokens=tokens,
            new_log_probs=(-rng.random(count)).tolist(),
            finish_type="stop",
            meta_info=meta,
            elapsed=self._delay_s,
        )

    async def abort_request(self, request_id: str) -> None:
        del request_id

    async def close_session(self, session_id: str, *, timeout_s: float) -> bool:
        del session_id, timeout_s
        return True

    async def shutdown(self) -> None:
        self.compiler.cpu_executor.shutdown(wait=True)


class FakeAgentProcess:
    """ManagedAgentProcess stand-in; a bench coroutine plays the agent."""

    def __init__(self) -> None:
        self._done = asyncio.Event()
        self._raw_output = ""
        self._output: Any = None

    def finish(self, raw_output: str) -> None:
        """Called on the Shard loop when the simulated agent process exits."""

        self._raw_output = raw_output
        self._done.set()

    async def wait(self) -> None:
        await self._done.wait()
        if self._output is None:
            from relax.agentic.runner.ipc import _session_output_from_text

            # ManagedAgentProcess._run parses the output file on the Shard loop.
            self._output = _session_output_from_text(self._raw_output)

    @property
    def output(self) -> Any:
        return self._output

    async def set_timeout_active(self, active: bool) -> None:
        del active

    async def terminate_and_join(self) -> None:
        self._done.set()


class FakeLauncher:
    """ManagedAgentLauncher stand-in that records one process per Session."""

    def __init__(self, *_: Any) -> None:
        self.processes: dict[str, FakeAgentProcess] = {}

    def start_agent(self, session_input: Any) -> FakeAgentProcess:
        process = FakeAgentProcess()
        self.processes[session_input.session_id] = process
        return process


class _Remote:
    """``actor.method.remote(...)`` that returns a resolved awaitable."""

    def __init__(self, value: Any) -> None:
        self._value = value

    def remote(self, *_: Any) -> "asyncio.Future[Any]":
        future = asyncio.get_running_loop().create_future()
        future.set_result(self._value)
        return future


class FakeLimiter:
    """SGLang request-permit coordinator that grants every permit."""

    acquire_request_permit = _Remote(True)
    release_request_permit = _Remote(None)
    cancel_request_permit = _Remote(None)


def build_args() -> SimpleNamespace:
    """Every Shard argument the Session path reads, with train defaults."""

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


def make_shard(backend: FakeBackend, launcher: FakeLauncher, args: SimpleNamespace, export_workers: int) -> Any:
    """Build the real Shard class (unwrapped from Ray) with fake backends."""

    from relax.agentic.session import service

    if export_workers and hasattr(service, "_SESSION_EXPORT_WORKERS"):
        service._SESSION_EXPORT_WORKERS = export_workers
    service.SGLangBackendAdapter = lambda _args: backend
    service.ManagedAgentLauncher = lambda *_: launcher
    service.load_agent_app_spec_from_args = lambda _args: None
    service.resolve_chat_api_base_url = lambda: "http://127.0.0.1:1/"
    shard_cls = service.AgenticSessionShard.__ray_metadata__.modified_class
    return shard_cls(args, FakeLimiter())


async def _loop_thread_time() -> tuple[float, int]:
    return time.thread_time(), threading.get_native_id()


async def _shard_leaks(shard: Any, keep: set[Any]) -> dict[str, Any]:
    """Shard-loop state left after shutdown; every count should be zero."""

    tasks = [t for t in asyncio.all_tasks() if t not in keep and t is not asyncio.current_task() and not t.done()]
    export_executor = getattr(shard, "_export_executor", None)
    return {
        "pending_tasks": len(tasks),
        "pending_task_names": sorted({t.get_name() for t in tasks})[:20],
        "groups": len(shard._groups),
        "session_records": len(shard._session_records),
        "inflight_exports": len(getattr(shard, "_inflight_exports", ())),
        "export_worker_threads": sum(thread.is_alive() for thread in getattr(export_executor, "_threads", ())),
    }


def _export_throughput(
    exit_at: dict[str, float], taken_at: dict[str, float], groups: list[list[str]]
) -> dict[str, Any]:
    """Separate whole-batch wall throughput from summed active-wave
    throughput."""

    completed = [session_id for group in groups for session_id in group if session_id in taken_at]
    completed_groups = [group for group in groups if all(session_id in taken_at for session_id in group)]
    drains = [
        max(taken_at[session_id] for session_id in group) - min(exit_at[session_id] for session_id in group)
        for group in completed_groups
    ]
    batch_elapsed = (
        max((taken_at[session_id] for session_id in completed), default=0.0)
        - min((exit_at[session_id] for session_id in completed), default=0.0)
        if completed
        else 0.0
    )
    active_completed = sum(len(group) for group in completed_groups)
    active_elapsed = sum(drains)
    return {
        "drain_durations_s": drains,
        "batch_completed": len(completed),
        "batch_elapsed_s": batch_elapsed,
        "batch_wallclock_per_s": len(completed) / batch_elapsed if batch_elapsed > 0 else 0.0,
        "active_drain_completed": active_completed,
        "active_drain_elapsed_s": active_elapsed,
        "active_drain_per_s": active_completed / active_elapsed if active_elapsed > 0 else 0.0,
    }


def _transport(results: tuple[Any, ...]) -> tuple[float, list[list[str]], list[int]]:
    """Pickle like Ray returns do (protocol 5, out-of-band), then digest."""

    buffers: list[pickle.PickleBuffer] = []
    pickle.dumps(results, protocol=5, buffer_callback=buffers.append)
    for buffer in buffers:
        bytes(buffer.raw())
    pickled_at = time.monotonic()
    digests: list[list[str]] = []
    tokens: list[int] = []
    for result in results:
        exports = getattr(result, "exports", None)
        if exports is None:
            digests.append([f"not-exported:{type(result).__name__}:{result}"])
            tokens.append(0)
            continue
        export_digests = []
        token_count = 0
        for export in exports:
            payload = export.get("sample_payload") or {}
            try:
                export_digests.append(_content_digest(payload))
            except ValueError as error:
                export_digests.append(f"invalid-content:{error}")
            token_values = payload.get("tokens")
            token_count += len(token_values) if token_values is not None else 0
        digests.append(export_digests)
        tokens.append(token_count)
    return pickled_at, digests, tokens


class Bench:
    """One run: the Shard on the main-thread loop, callers on another."""

    def __init__(self, scenario: Scenario, tokenizer: Any, token_pool: np.ndarray) -> None:
        self.sc = scenario
        self.threads_before = {t.ident for t in threading.enumerate()}
        self.args = build_args()
        if scenario.r3_layers:
            self.args.use_rollout_routing_replay = True
            self.args.num_layers = scenario.r3_layers
            self.args.moe_router_topk = scenario.r3_topk
        self.backend = FakeBackend(tokenizer, token_pool, scenario)
        self.launcher = FakeLauncher()
        self.shard_loop = asyncio.get_running_loop()
        self.shard = make_shard(self.backend, self.launcher, self.args, scenario.export_workers)
        self.client_loop = asyncio.new_event_loop()
        self.client_thread = threading.Thread(target=self.client_loop.run_forever, name="bench-client", daemon=True)
        self.transport = ThreadPoolExecutor(max_workers=1, thread_name_prefix="bench-transport")
        self.histories: dict[str, list[dict[str, Any]]] = {}
        self.outputs: dict[str, str] = {}
        self.exit_at: dict[str, float] = {}
        self.taken_at: dict[str, float] = {}
        self.digests: dict[str, list[str]] = {}
        self.tokens: dict[str, int] = {}
        self.main_task: Optional[asyncio.Task[Any]] = None
        self.latency: dict[str, list[tuple[float, float]]] = {"short": [], "health": []}
        self.errors: dict[str, list[str]] = {"short": [], "health": []}
        self.probe_counts = {
            kind: {"scheduled": 0, "attempted": 0, "completed": 0, "succeeded": 0, "failed": 0}
            for kind in ("short", "health")
        }
        self.inflight = {"now": 0, "peak": 0}
        self.window_start = 0.0

    async def on_shard(self, coro: Any) -> Any:
        """Ray-style call: run ``coro`` on the Shard loop, await here."""

        return await asyncio.wrap_future(asyncio.run_coroutine_threadsafe(coro, self.shard_loop))

    def _initial_messages(self, session_id: str, init_tokens: int) -> list[dict[str, Any]]:
        return [
            {"role": "system", "content": "You are a benchmark agent."},
            {"role": "user", "content": _text(_seed(session_id, "init"), init_tokens * _CHARS_PER_TOKEN)},
        ]

    async def _chat(self, session_id: str, messages: list[dict[str, Any]], max_tokens: int) -> dict[str, Any]:
        return await self.on_shard(
            self.shard.chat(
                session_id=session_id,
                messages=messages,
                tools=[],
                chat_template_kwargs={},
                max_completion_tokens=max_tokens,
                stop=None,
                seed=None,
            )
        )

    async def long_agent(self, session_id: str, turns: int) -> None:
        """Multi-turn agent: send history, append reply and observation."""

        sc = self.sc
        history = self._initial_messages(session_id, sc.init_tokens)
        for turn in range(turns):
            response = await self._chat(session_id, list(history), sc.resp_tokens)
            if response.get("error"):
                raise RuntimeError(f"{session_id} turn {turn}: {response['error']}")
            history.append(response["message"])
            if turn + 1 < turns:
                obs = _text(_seed(session_id, "obs", turn), sc.obs_tokens * _CHARS_PER_TOKEN)
                history.append({"role": "user", "content": obs})
        self.histories[session_id] = history

    async def short_agent(self, session_id: str) -> None:
        response = await self._chat(session_id, self._initial_messages(session_id, 32), self.sc.short_resp_tokens)
        if response.get("error"):
            raise RuntimeError(f"{session_id}: {response['error']}")

    def output_text(self, session_id: str) -> str:
        """The agent's output file: reward/metadata or one JSONL record."""

        meta = {"bench_session": session_id}
        if self.sc.export_mode == "explicit":
            record = {"name": "main", "messages": self.histories[session_id], "reward": 1.0, "metadata": meta}
            return json.dumps(record, ensure_ascii=False)
        return json.dumps({"reward": 1.0, "metadata": meta})

    def exit_session(self, session_id: str) -> None:
        """Simulated agent process exit (client loop -> Shard loop)."""

        self.exit_at[session_id] = time.monotonic()
        process = self.launcher.processes[session_id]
        self.shard_loop.call_soon_threadsafe(process.finish, self.outputs[session_id])

    async def build_group(self, kind: str, group_index: int, sessions: int, turns: int = 0) -> list[str]:
        """Start a group, cross its barrier, lease it, run its agents."""

        from relax.agentic.pipeline import SessionSpec

        group_id = f"{kind}-g{group_index}"
        session_ids = [f"{group_id}-s{i}" for i in range(sessions)]
        specs = tuple(
            SessionSpec(session_id=sid, group_index=group_index, index=i, input_payload={"bench": sid})
            for i, sid in enumerate(session_ids)
        )
        await self.on_shard(self.shard.start_group("train", group_id, specs))
        if kind == "short":
            agents = [asyncio.ensure_future(self.short_agent(sid)) for sid in session_ids]
        else:
            agents = [asyncio.ensure_future(self.long_agent(sid, turns)) for sid in session_ids]
        revision = -1
        while True:
            progress = await self.on_shard(self.shard.wait_for_state_change(revision))
            revision = progress.revision
            if any(g.group_id == group_id and g.ready_for_lease for g in progress.groups):
                break
        await self.on_shard(self.shard.lease_group(group_id, 1))
        await asyncio.gather(*agents)
        return session_ids

    async def consume(self, expected: set[str]) -> None:
        """Runtime-side observer: take ready Sessions and transport them."""

        pending = set(expected)
        revision = -1
        loop = asyncio.get_running_loop()
        while pending:
            progress = await self.on_shard(self.shard.wait_for_state_change(revision))
            revision = progress.revision
            for group in progress.groups:
                ready = tuple(sid for sid in group.takeable_session_ids if sid in pending)
                if not ready:
                    continue
                pending.difference_update(ready)
                results = await self.on_shard(self.shard.take_session_results(group.group_id, ready))
                pickled_at, digests, tokens = await loop.run_in_executor(self.transport, _transport, results)
                for sid, digest, count in zip(ready, digests, tokens, strict=True):
                    self.taken_at[sid] = pickled_at
                    self.digests[sid] = digest
                    self.tokens[sid] = count

    async def probe(self, kind: str, scheduled: float, session_id: str) -> None:
        """One open-loop request; latency counts from its scheduled time."""

        counts = self.probe_counts[kind]
        counts["attempted"] += 1
        self.inflight["now"] += 1
        self.inflight["peak"] = max(self.inflight["peak"], self.inflight["now"])
        succeeded = False
        failure: Optional[str] = None
        try:
            if kind == "short":
                messages = self._initial_messages(session_id, 32)
                result = await self._chat(session_id, messages, self.sc.short_resp_tokens)
                if result.get("error"):
                    failure = str(result["error"])[:200]
                else:
                    succeeded = True
            else:
                result = await self.on_shard(self.shard.health())
                if not isinstance(result, dict) or not result.get("ok"):
                    failure = f"health returned not ok: {result!r}"[:200]
                else:
                    succeeded = True
        except asyncio.CancelledError:
            failure = "CancelledError"
            raise
        except Exception as error:
            failure = f"{type(error).__name__}: {error}"[:200]
        finally:
            self.inflight["now"] -= 1
            if failure is not None:
                self.errors[kind].append(failure)
                counts["failed"] += 1
            elif succeeded:
                counts["succeeded"] += 1
            counts["completed"] += 1
            done = time.monotonic()
            self.latency[kind].append((scheduled - self.window_start, done - scheduled))

    async def schedule(self, kind: str, rate: float, until: float, session_ids: list[str]) -> int:
        """Fire requests at fixed times, however fast the Shard answers."""

        tasks: list[asyncio.Task[None]] = []
        period = 1.0 / rate
        index = 0
        while True:
            due = self.window_start + index * period
            if due >= until:
                break
            await asyncio.sleep(max(0.0, due - time.monotonic()))
            sid = session_ids[index % len(session_ids)]
            tasks.append(asyncio.ensure_future(self.probe(kind, due, sid)))
            index += 1
        self.probe_counts[kind]["scheduled"] += index
        await asyncio.gather(*tasks)
        return index

    async def waves(self, groups: list[list[str]]) -> list[float]:
        """Release one long group per wave, after 1 s of quiet baseline."""

        offsets = []
        for index, session_ids in enumerate(groups):
            offset = 1.0 + index * self.sc.wave_interval_s
            await asyncio.sleep(max(0.0, self.window_start + offset - time.monotonic()))
            offsets.append(time.monotonic() - self.window_start)
            for sid in session_ids:
                self.exit_session(sid)
        return offsets

    async def teardown(self, group_ids: list[str]) -> dict[str, Any]:
        for group_id in group_ids:
            await self.on_shard(self.shard.drop_group(group_id))
        started = time.monotonic()
        await self.on_shard(self.shard.shutdown())
        shutdown_s = time.monotonic() - started
        leaks = await self.on_shard(_shard_leaks(self.shard, {self.main_task}))
        return {"shutdown_s": shutdown_s, **leaks}

    async def client_main(self, profile: int) -> dict[str, Any]:
        sc = self.sc
        await self.on_shard(self.shard.resume_generation(1))
        started = time.monotonic()
        long_groups = list(
            await asyncio.gather(
                *(self.build_group("long", g, sc.sessions_per_group, sc.turns) for g in range(sc.groups))
            )
        )
        build_s = time.monotonic() - started
        group_ids = [f"long-g{g}" for g in range(sc.groups)]
        if profile:
            result = await self.on_shard(_profile_export(self.shard, long_groups[0][0], profile))
            return {**result, "build_s": build_s, "teardown": await self.teardown(group_ids)}

        # Warmup: one export outside the window (lazy pools, first-call paths), short sessions open.
        warm_ids = await self.build_group("warm", 0, 1, max(2, sc.turns // 4))
        short_ids = await self.build_group("short", 0, sc.short_sessions)
        group_ids += ["warm-g0", "short-g0"]
        long_ids = {sid for group in long_groups for sid in group}
        for sid in [*warm_ids, *long_ids]:
            self.outputs[sid] = self.output_text(sid)
        consumer = asyncio.ensure_future(self.consume({*warm_ids, *long_ids}))
        for sid in warm_ids:
            self.exit_session(sid)
        while any(sid not in self.taken_at for sid in warm_ids):
            await asyncio.sleep(0.02)
        lag = LagProbe(sc.lag_tick_s)
        await asyncio.sleep(1.0)

        # Measurement window: probes at fixed rates, long groups exit in waves.
        sampling_started = time.monotonic()
        loop_cpu0, loop_tid = await self.on_shard(_loop_thread_time())
        threads0, proc0 = _thread_cpu(), time.process_time()
        rss = RssSampler()
        rss.start()
        self.window_start = await self.on_shard(lag.start(window_s=sc.window))
        until = self.window_start + sc.window
        short_scheduled, health_scheduled, wave_offsets = await asyncio.gather(
            self.schedule("short", sc.short_rps, until, short_ids),
            self.schedule("health", sc.health_rps, until, short_ids),
            self.waves(long_groups),
        )
        raw, slots = await self.on_shard(lag.stop())
        consumer_completed = True
        try:
            await asyncio.wait_for(asyncio.shield(consumer), timeout=120 + 20 * sc.window)
        except asyncio.TimeoutError:
            consumer_completed = False
            consumer.cancel()
            await asyncio.gather(consumer, return_exceptions=True)
        loop_cpu1, _ = await self.on_shard(_loop_thread_time())
        threads1, proc1 = _thread_cpu(), time.process_time()
        # Resolve names now: teardown stops the export and compiler worker threads.
        names = {t.native_id: t.name for t in threading.enumerate()}
        memory = rss.finish()
        sampling_elapsed_s = time.monotonic() - sampling_started
        teardown = await self.teardown(group_ids)
        done = [sid for sid in long_ids if sid in self.taken_at]
        export_s = [self.taken_at[sid] - self.exit_at[sid] for sid in done]
        throughput = _export_throughput(self.exit_at, self.taken_at, long_groups)
        in_window = bool(done) and max(self.taken_at[sid] for sid in done) <= until
        names[loop_tid] = "shard-loop"
        thread_cpu = {
            names.get(tid, f"tid-{tid}"): round(cpu - threads0.get(tid, 0.0), 4)
            for tid, cpu in threads1.items()
            if cpu - threads0.get(tid, 0.0) > 1e-3
        }
        timeline = [0.0] * (int(sc.window) + 1)
        for offset, latency in self.latency["short"]:
            bucket = min(len(timeline) - 1, max(0, int(offset)))
            timeline[bucket] = max(timeline[bucket], latency * 1e3)
        return {
            "scenario": dataclasses.asdict(sc),
            "window_s": sc.window,
            "build_s": build_s,
            # Valid only if every export finished inside the sampled window, so its stalls were measured.
            "valid": len(done) == len(long_ids)
            and in_window
            and consumer_completed
            and not any(d.startswith(("not-exported", "invalid-content")) for d in self._flat()),
            "exports_in_window": in_window,
            "consumer_completed": consumer_completed,
            "loop_lag_ms": _percentiles(slots),
            "loop_lag_raw_ms": _percentiles(raw),
            "short_ms": _percentiles([lat for _, lat in self.latency["short"]]),
            "health_ms": _percentiles([lat for _, lat in self.latency["health"]]),
            "short_errors": len(self.errors["short"]),
            "health_errors": len(self.errors["health"]),
            "probe_counts": {
                "short": {
                    "expected": _expected_probe_count(sc.short_rps, sc.window),
                    "scheduled": short_scheduled,
                    **self.probe_counts["short"],
                },
                "health": {
                    "expected": _expected_probe_count(sc.health_rps, sc.window),
                    "scheduled": health_scheduled,
                    **self.probe_counts["health"],
                },
            },
            "error_samples": (self.errors["short"] + self.errors["health"])[:5],
            "probe_inflight_peak": self.inflight["peak"],
            "exports": {
                "expected": len(long_ids),
                "completed": len(done),
                "latency_ms": _percentiles(export_s),
                "wave_drain_ms": _percentiles(throughput["drain_durations_s"]),
                "batch_elapsed_s": throughput["batch_elapsed_s"],
                "batch_wallclock_per_s": throughput["batch_wallclock_per_s"],
                "active_drain_completed": throughput["active_drain_completed"],
                "active_drain_elapsed_s": throughput["active_drain_elapsed_s"],
                "active_drain_per_s": throughput["active_drain_per_s"],
                "mean_tokens": statistics.fmean(self.tokens[sid] for sid in done) if done else 0.0,
            },
            "measurement_scope": {
                "configured_window_s": sc.window,
                "cpu_rss_elapsed_s": sampling_elapsed_s,
                "cpu_rss_scope": (
                    "sampling starts immediately before the formal window and ends after probe/consumer drain; "
                    "includes transport and content hashing"
                ),
            },
            "cpu": {
                "process_s": proc1 - proc0,
                "shard_loop_s": loop_cpu1 - loop_cpu0,
                "threads_s": thread_cpu,
            },
            "memory": memory,
            "teardown": teardown,
            "timeline_short_max_ms": [round(v, 1) for v in timeline],
            "wave_offsets_s": wave_offsets,
            "digests": {sid: self.digests[sid] for sid in sorted(done)},
        }

    def _flat(self) -> list[str]:
        return [d for digests in self.digests.values() for d in digests]


async def _profile_export(shard: Any, session_id: str, repeats: int) -> dict[str, Any]:
    """Stage timings and a cProfile of one long Session's assembly."""

    from relax.agentic.pipeline import TrainingFieldArtifact
    from relax.agentic.session.state import _messages_tools_template_state_hash

    forest = shard._session_records[session_id].forest
    tokenizer = shard._generation_backend.tokenizer
    (leaf,) = forest.export_leaf_hashes()
    messages = forest.full_messages(leaf)
    stages: dict[str, list[float]] = {}

    def timed(name: str, fn: Any) -> Any:
        started = time.perf_counter()
        value = fn()
        stages.setdefault(name, []).append((time.perf_counter() - started) * 1e3)
        return value

    profiler = cProfile.Profile()
    for _ in range(repeats):
        sample = timed("build_sample", lambda: forest.build_sample(leaf_state_hash=leaf, tokenizer=tokenizer))
        timed("decode_all_tokens", lambda: tokenizer.decode(sample.tokens, skip_special_tokens=False))
        timed("deepcopy_to_dict", lambda: copy.deepcopy(sample.to_dict()))
        timed("explicit_state_hash", lambda: _messages_tools_template_state_hash(messages, [], {}))
        timed("from_sample", lambda: TrainingFieldArtifact.from_sample(sample))
        rebuilt = forest.build_sample(leaf_state_hash=leaf, tokenizer=tokenizer)
        profiler.runcall(TrainingFieldArtifact.from_sample, rebuilt)
        profiler.runcall(forest.build_sample, leaf_state_hash=leaf, tokenizer=tokenizer)
    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.sort_stats("tottime").print_stats(18)
    stats.sort_stats("cumulative").print_stats(12)
    return {
        "profile_session": session_id,
        "tokens": len(sample.tokens),
        "turns": sum(node.kind == "resp" for node in forest.lineage(leaf)),
        "stage_median_ms": {name: statistics.median(values) for name, values in stages.items()},
        "cprofile": stream.getvalue(),
    }


async def _shard_main(scenario: Scenario, tokenizer: Any, pool: np.ndarray, profile: int) -> dict[str, Any]:
    bench = Bench(scenario, tokenizer, pool)
    bench.main_task = asyncio.current_task()
    bench.client_thread.start()
    try:
        future = asyncio.run_coroutine_threadsafe(bench.client_main(profile), bench.client_loop)
        result = await asyncio.wrap_future(future)
    finally:
        bench.client_loop.call_soon_threadsafe(bench.client_loop.stop)
        bench.client_thread.join()
        bench.client_loop.close()
        bench.transport.shutdown(wait=True)
    result["teardown"]["leaked_threads"] = sorted(
        t.name for t in threading.enumerate() if t.ident not in bench.threads_before
    )
    return result


def run_main(ns: argparse.Namespace) -> None:
    root = _bootstrap_src(ns.src)
    scenario = build_scenario(ns.preset, ns.set or [])
    missing = {k: v for k, v in _SHARD_ALLOCATOR_ENV.items() if os.environ.get(k) != v}
    if missing:
        print(f"warning: production Shard allocator env is not set: {missing}", file=sys.stderr)
    tokenizer, pool = load_tokenizer(ns.tokenizer)
    try:
        result = asyncio.run(_shard_main(scenario, tokenizer, pool, ns.profile))
    except Exception as error:
        result = {
            "src": root,
            "python": sys.version.split()[0],
            "valid": False,
            "validation_errors": [f"benchmark execution failed: {type(error).__name__}: {error}"],
        }
        if ns.out:
            with open(ns.out, "w", encoding="utf-8") as fh:
                json.dump(result, fh, indent=2, sort_keys=True)
        else:
            print(json.dumps(result, indent=2, sort_keys=True))
        destination = os.path.abspath(ns.out) if ns.out else "stdout"
        print(f"invalid benchmark run; diagnostics preserved in {destination}", file=sys.stderr)
        raise SystemExit(2) from error
    result["src"] = root
    result["python"] = sys.version.split()[0]
    if not ns.profile:
        result["validation_errors"] = _run_validation_errors(result)
        result["valid"] = not result["validation_errors"]
    if ns.out:
        with open(ns.out, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2, sort_keys=True)
    if ns.profile:
        print(result["cprofile"])
        print(json.dumps({k: v for k, v in result.items() if k != "cprofile"}, indent=2, sort_keys=True))
    elif not ns.out:
        print(json.dumps(result, indent=2, sort_keys=True))
    if not ns.profile and not result["valid"]:
        destination = os.path.abspath(ns.out) if ns.out else "stdout"
        print(f"invalid benchmark run; diagnostics preserved in {destination}", file=sys.stderr)
        raise SystemExit(2)


_METRICS = (
    ("event-loop lag P99 (ms)", ("loop_lag_ms", "p99")),
    ("event-loop lag P99.9 (ms)", ("loop_lag_ms", "p999")),
    ("event-loop lag max (ms)", ("loop_lag_ms", "max")),
    ("short chat P50 (ms)", ("short_ms", "p50")),
    ("short chat P99 (ms)", ("short_ms", "p99")),
    ("short chat max (ms)", ("short_ms", "max")),
    ("health P99 (ms)", ("health_ms", "p99")),
    ("short requests sent", ("short_ms", "count")),
    ("exports completed", ("exports", "completed")),
    ("export latency P50 (ms)", ("exports", "latency_ms", "p50")),
    ("export latency P99 (ms)", ("exports", "latency_ms", "p99")),
    ("wave drain median (ms)", ("exports", "wave_drain_ms", "p50")),
    ("batch wall-clock drain (s)", ("exports", "batch_elapsed_s")),
    ("batch wall-clock throughput (/s)", ("exports", "batch_wallclock_per_s")),
    ("active-drain throughput (/s)", ("exports", "active_drain_per_s")),
    ("process CPU (s)", ("cpu", "process_s")),
    ("Shard loop thread CPU (s)", ("cpu", "shard_loop_s")),
    ("peak RSS (MiB)", ("memory", "peak_mib")),
    ("shutdown (s)", ("teardown", "shutdown_s")),
)


def _get(result: dict[str, Any], path: tuple[str, ...]) -> float:
    value: Any = result
    for key in path:
        value = value.get(key, 0.0) if isinstance(value, dict) else 0.0
    return float(value)


def _workload_scenario(scenario: dict[str, Any]) -> dict[str, Any]:
    """Exclude the candidate's export-worker ablation from workload
    identity."""

    return {key: value for key, value in scenario.items() if key != "export_workers"}


def _write_comparison_invalid(out_dir: str, errors: list[str], runs: dict[str, list[dict[str, Any]]]) -> None:
    diagnostic = {
        "valid": False,
        "errors": errors,
        "run_files": sorted(
            os.path.basename(path)
            for path in (
                os.path.join(out_dir, name)
                for name in os.listdir(out_dir)
                if name.endswith(".json") and name != "comparison-invalid.json"
            )
        ),
        "runs": {
            variant: [
                {
                    "valid": result.get("valid"),
                    "validation_errors": result.get("validation_errors", []),
                    "scenario": result.get("scenario"),
                }
                for result in variant_runs
            ]
            for variant, variant_runs in runs.items()
        },
    }
    path = os.path.join(out_dir, "comparison-invalid.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(diagnostic, fh, indent=2, sort_keys=True)
    print(f"invalid comparison; diagnostics preserved in {path}", file=sys.stderr)


def _comparison_validation_errors(runs: dict[str, list[dict[str, Any]]]) -> list[str]:
    errors: list[str] = []
    reference_scenario: Optional[dict[str, Any]] = None
    reference_digests: Optional[str] = None
    for variant, variant_runs in runs.items():
        for repeat, result in enumerate(variant_runs):
            run_errors = _run_validation_errors(result)
            if run_errors:
                errors.extend(f"{variant}[{repeat}]: {error}" for error in run_errors)
            scenario = _workload_scenario(result.get("scenario") or {})
            if reference_scenario is None:
                reference_scenario = scenario
            elif scenario != reference_scenario:
                errors.append(f"{variant}[{repeat}]: workload scenario differs from the other runs")
            digests = json.dumps(result.get("digests") or {}, sort_keys=True)
            if reference_digests is None:
                reference_digests = digests
            elif digests != reference_digests:
                errors.append(f"{variant}[{repeat}]: benchmark-listed export fields differ across runs")
    return errors


def compare_main(ns: argparse.Namespace) -> None:
    """Alternate which variant runs first; each run is a fresh process."""

    os.makedirs(ns.out_dir, exist_ok=True)
    if ns.repeats <= 0:
        _write_comparison_invalid(
            ns.out_dir, [f"repeats must be positive, got {ns.repeats}"], {"baseline": [], "candidate": []}
        )
        raise SystemExit(2)
    invalid_overrides = [
        item
        for item in (ns.candidate_set or [])
        if item.partition("=")[0] != "export_workers" or not item.partition("=")[1]
    ]
    if invalid_overrides:
        _write_comparison_invalid(
            ns.out_dir,
            [
                "candidate-only overrides may change export_workers only; use --set for shared workload values: "
                + ", ".join(invalid_overrides)
            ],
            {"baseline": [], "candidate": []},
        )
        raise SystemExit(2)

    srcs = {"baseline": ns.baseline_src, "candidate": ns.candidate_src}
    runs: dict[str, list[dict[str, Any]]] = {"baseline": [], "candidate": []}
    env = {**os.environ, **_SHARD_ALLOCATOR_ENV}
    for rep in range(ns.repeats):
        order = ("baseline", "candidate") if rep % 2 == 0 else ("candidate", "baseline")
        for variant in order:
            out = os.path.join(ns.out_dir, f"{variant}-{rep}.json")
            cmd = [sys.executable, os.path.abspath(__file__), "run", "--src", srcs[variant]]
            cmd += ["--tokenizer", ns.tokenizer, "--preset", ns.preset, "--out", out]
            for item in [*(ns.set or []), *((ns.candidate_set or []) if variant == "candidate" else [])]:
                cmd += ["--set", item]
            print(f"[{rep + 1}/{ns.repeats}] {variant} ...", flush=True)
            try:
                completed = subprocess.run(cmd, env=env, check=False, timeout=ns.timeout)
            except subprocess.TimeoutExpired as error:
                _write_comparison_invalid(ns.out_dir, [f"{variant}[{rep}] exceeded timeout {error.timeout}s"], runs)
                raise SystemExit(2) from error
            if completed.returncode != 0:
                if os.path.exists(out):
                    with open(out, encoding="utf-8") as fh:
                        runs[variant].append(json.load(fh))
                _write_comparison_invalid(
                    ns.out_dir, [f"{variant}[{rep}] exited with status {completed.returncode}"], runs
                )
                raise SystemExit(2)
            with open(out, encoding="utf-8") as fh:
                runs[variant].append(json.load(fh))

    validation_errors = _comparison_validation_errors(runs)
    if validation_errors:
        _write_comparison_invalid(ns.out_dir, validation_errors, runs)
        raise SystemExit(2)

    lines = [
        f"preset={ns.preset} overrides={ns.set or []} candidate-only={ns.candidate_set or []} repeats={ns.repeats}",
        "",
    ]
    lines.append("| metric | baseline median [min, max] | candidate median [min, max] | change |")
    lines.append("| --- | --- | --- | --- |")
    for label, path in _METRICS:
        cells = []
        medians = []
        for variant in ("baseline", "candidate"):
            values = [_get(r, path) for r in runs[variant]]
            medians.append(statistics.median(values))
            cells.append(f"{medians[-1]:.2f} [{min(values):.2f}, {max(values):.2f}]")
        change = (medians[1] - medians[0]) / medians[0] * 100 if medians[0] else 0.0
        lines.append(f"| {label} | {cells[0]} | {cells[1]} | {change:+.1f}% |")

    lines += ["", "all runs valid: True"]
    lines.append("benchmark-listed export fields identical across all runs: True")
    lines.append(f"hashed arrays: {[name for name, _dtype in _DIGEST_ARRAYS]}")
    lines.append(
        f"hashed scalars: {list(_DIGEST_SCALARS)} plus weight_versions/tools/chat_template_kwargs/rollout_turns"
    )
    for variant in runs:
        leaks = [(r["teardown"]["pending_tasks"], r["teardown"]["leaked_threads"]) for r in runs[variant]]
        lines.append(f"{variant} teardown (pending tasks, leaked threads): {leaks}")
    lines += ["", "short chat max latency per second of window (repeat 0; waves at "]
    lines[-1] += ", ".join(f"{t:.0f}s" for t in runs["baseline"][0]["wave_offsets_s"]) + "):"
    lines.append("| t (s) | baseline (ms) | candidate (ms) |")
    lines.append("| --- | --- | --- |")
    base_tl, cand_tl = runs["baseline"][0]["timeline_short_max_ms"], runs["candidate"][0]["timeline_short_max_ms"]
    for second, (b, c) in enumerate(zip(base_tl, cand_tl)):
        lines.append(f"| {second} | {b:.0f} | {c:.0f} |")
    for variant in runs:
        lines.append(f"{variant} CPU by thread (repeat 0, s): {runs[variant][0]['cpu']['threads_s']}")
    report = "\n".join(lines)
    print(report)
    with open(os.path.join(ns.out_dir, "summary.md"), "w", encoding="utf-8") as fh:
        fh.write(report + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "compare"):
        p = sub.add_parser(name)
        p.add_argument("--tokenizer", required=True, help="local HF tokenizer directory")
        p.add_argument("--preset", default="default", choices=sorted(PRESETS))
        p.add_argument("--set", action="append", metavar="FIELD=VALUE", help="override a Scenario field")
        if name == "run":
            p.add_argument("--src", required=True, help="repo root whose relax/ package is measured")
            p.add_argument("--out", help="write the result JSON here")
            p.add_argument("--profile", type=int, default=0, help="profile N assemblies instead of a window")
        else:
            p.add_argument("--baseline-src", required=True)
            p.add_argument("--candidate-src", required=True)
            p.add_argument("--candidate-set", action="append", metavar="FIELD=VALUE", help="candidate-only override")
            p.add_argument("--repeats", type=int, default=5)
            p.add_argument("--out-dir", default="bench-agentic-export")
            p.add_argument("--timeout", type=float, default=1800.0, help="seconds per run")
    ns = parser.parse_args()
    if ns.command == "run":
        run_main(ns)
    else:
        compare_main(ns)


if __name__ == "__main__":
    main()
