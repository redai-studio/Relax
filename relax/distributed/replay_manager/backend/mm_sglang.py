# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Opt-in frozen vision capture at SGLang's complete-media boundary."""

from __future__ import annotations

import hashlib
import inspect
import sys
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
from threading import Condition
from time import perf_counter
from typing import Any


REQUEST_PREFIX = "relax-mm-"
META_FIELD = "mm_embedding_refs"
_CAPTURE: ContextVar[MMCapture | None] = ContextVar("relax_mm_capture", default=None)
_FULL_ITEMS: ContextVar[Sequence[Any]] = ContextVar("relax_mm_full_items", default=())


def media_description(item: Any) -> tuple[str, tuple[int, int, int], str]:
    """Use the engine's image hash as the authoritative deduplication key."""
    import torch

    value = item.hash
    if value is None:
        raise ValueError("MM replay requires a media hash")
    digest = (
        value.lower()
        if isinstance(value, str) and len(value) == 64
        else hashlib.sha256(str(value).encode()).hexdigest()
    )
    if any(char not in "0123456789abcdef" for char in digest):
        raise ValueError("MM replay requires a hexadecimal media hash")
    modality = item.modality.name.lower()
    if modality not in ("image", "video"):
        raise ValueError("MM replay supports images and videos only")
    grid = getattr(item, modality + "_grid_thw", None)
    if isinstance(grid, torch.Tensor):
        if grid.device.type != "cpu":
            raise ValueError("MM replay requires producer CPU grid metadata")
        grid = grid.numpy().reshape(-1, 3).tolist()
    if grid is None or len(grid) != 1 or len(grid[0]) != 3:
        raise ValueError("MM replay requires one media grid per item")
    shape = tuple(int(value) for value in grid[0])
    if any(value <= 0 for value in shape) or shape[1] * shape[2] % 4:
        raise ValueError("Invalid R4 vision grid")
    return digest, shape, modality


@dataclass
class _Media:
    digest: str
    grid: tuple[int, int, int]
    modality: str
    future: Future | None = None


class MMCapture:
    """Bounded host snapshots; cache leases live until the matching KV
    flush."""

    def __init__(self, store: Any, *, publisher: Any = None, release: Callable | None = None) -> None:
        from ..mm_embedding import MMPublisher, release_media_entries

        self.store = store
        self.publisher = MMPublisher(store) if publisher is None else publisher
        self.release = release_media_entries if release is None else release
        self.requests: dict[str, list[_Media]] = {}
        self.cache: dict[str, tuple[tuple[int, int, int], str, Future]] = {}
        self.pending_bytes = self.cache_bytes = 0
        self.metrics = dict.fromkeys(
            ("published_bytes", "publish_seconds", "queue_wait_seconds", "pending_peak_bytes"), 0
        )
        self._condition = Condition()
        self._worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mm-replay-publish")

    def bind(self, rid: str, mm_inputs: Any) -> None:
        if not rid.startswith(REQUEST_PREFIX):
            return
        previous = self.requests.get(rid)
        if previous is not None and mm_inputs is None:
            return
        items = [] if mm_inputs is None else list(mm_inputs.mm_items)
        ordered = []
        for item in items:
            description = media_description(item)
            grid = description[1]
            rows = grid[0] * grid[1] * grid[2] // 4
            accumulated, first = 0, None
            for start, end in item.offsets or []:
                if first is None:
                    first = start
                accumulated += end - start + 1
                if end < start or accumulated > rows:
                    raise ValueError("MM placeholder spans exceed one complete media")
                if accumulated == rows:
                    ordered.append((first, description))
                    accumulated, first = 0, None
            if accumulated or first is not None or not item.offsets:
                raise ValueError("MM placeholder spans do not cover complete media")
        metadata = [description for _, description in sorted(ordered, key=lambda pair: pair[0])]
        if previous is not None:
            if [(m.digest, m.grid, m.modality) for m in previous] != metadata:
                raise ValueError("MM replay request changed its media layout")
            return
        self.requests[rid] = [_Media(*description) for description in metadata]

    def capture(self, item: Any, embedding: Any) -> None:
        import torch

        digest, grid, modality = media_description(item)
        wanted = [m for media in self.requests.values() for m in media if m.digest == digest and m.future is None]
        if not wanted:
            return
        expected = grid[0] * grid[1] * grid[2] // 4
        if (
            not isinstance(embedding, torch.Tensor)
            or embedding.dtype != torch.bfloat16
            or tuple(embedding.shape) != (expected, 2560)
        ):
            raise ValueError("MM replay requires complete BF16 [visual tokens,2560] embeddings")
        cached = self.cache.get(digest)
        if cached is not None:
            if cached[:2] != (grid, modality):
                raise ValueError("One media hash has incompatible MM geometry")
            ready = cached[2]
        else:
            size = embedding.numel() * embedding.element_size()
            if self.cache_bytes + size > self.store.config.capacity_bytes:
                raise BufferError("MM replay cache exceeds its bounded budget")
            with self._condition:
                # One indivisible large source is allowed; host copies are still
                # chunked below. No second source joins it before it drains.
                started = perf_counter()
                self._condition.wait_for(
                    lambda: not self.pending_bytes or self.pending_bytes + size <= self.store.config.publication_bytes
                )
                self.metrics["queue_wait_seconds"] += perf_counter() - started
                self.pending_bytes += size
                self.metrics["pending_peak_bytes"] = max(self.metrics["pending_peak_bytes"], self.pending_bytes)
            source = embedding.detach()
            ready_event = None
            if source.is_cuda:
                ready_event = torch.cuda.Event()
                ready_event.record(torch.cuda.current_stream(source.device))
            copy_stream = None
            snapshot_bytes = 0

            def snapshot_rows(start: int, stop: int) -> Any:
                nonlocal copy_stream, snapshot_bytes
                snapshot_bytes += (stop - start) * 5120
                if not source.is_cuda:
                    return source[start:stop].contiguous().view(torch.uint8).numpy().copy()
                # The worker's default device can differ from this TP rank.
                with torch.cuda.device(source.device):
                    if copy_stream is None:
                        copy_stream = torch.cuda.Stream(device=source.device)
                        copy_stream.wait_event(ready_event)
                    host = torch.empty((stop - start, 5120), dtype=torch.uint8, pin_memory=True)
                    with torch.cuda.stream(copy_stream):
                        host.copy_(source[start:stop].contiguous().view(torch.uint8), non_blocking=True)
                        source.record_stream(copy_stream)
                        copied = torch.cuda.Event()
                        copied.record(copy_stream)
                    copied.synchronize()  # background worker, never the model thread
                    return host.numpy()

            def publish() -> dict:
                started = perf_counter()
                try:
                    result = self.publisher.publish_chunks(digest, snapshot_rows, grid, modality)
                    self.metrics["published_bytes"] += snapshot_bytes
                    return result
                finally:
                    with self._condition:
                        self.metrics["publish_seconds"] += perf_counter() - started
                        self.pending_bytes -= size
                        self._condition.notify_all()

            try:
                ready = self._worker.submit(publish)
            except BaseException:
                with self._condition:
                    self.pending_bytes -= size
                    self._condition.notify_all()
                raise
            self.cache[digest] = (grid, modality, ready)
            self.cache_bytes += size
        for media in wanted:
            if (media.grid, media.modality) != (grid, modality):
                raise ValueError("One media hash has incompatible request geometry")
            media.future = self._worker.submit(self._acquire, media, ready)

    def _acquire(self, media: _Media, ready: Future | None = None) -> dict:
        if ready is not None:
            ready.result()  # cache owns a separate lease; never copy it to a request
        entry = self.publisher.lookup(media.digest, media.grid, media.modality)
        if entry is None:
            raise RuntimeError(
                "MM replay is missing the complete media embedding, including its prefix-cache reference"
            )
        return entry

    def ready(self, rid: str) -> bool:
        media = self.requests.get(rid)
        if media is None:
            raise RuntimeError("MM replay request was not registered before generation")
        for item in media:
            if item.future is None:
                cached = self.cache.get(item.digest)
                item.future = self._worker.submit(self._acquire, item, None if cached is None else cached[2])
        return all(item.future.done() for item in media)

    def finish(self, rid: str) -> list[dict]:
        if not self.ready(rid):
            raise RuntimeError("MM replay body publication has not completed")
        media = self.requests[rid]
        try:
            result = [item.future.result() for item in media]
        except BaseException:
            self.abort(rid)
            raise
        self.requests.pop(rid)
        return result

    def _release_future(self, future: Future) -> None:
        if not future.cancelled() and future.exception() is None:
            self.release(self.store, [future.result()])

    def abort(self, rid: str) -> None:
        for item in self.requests.pop(rid, []):
            if item.future is not None:
                # Queue directly behind its acquisition, instead of a callback
                # that could enqueue cleanup after a concurrent flush barrier.
                self._worker.submit(self._release_future, item.future)

    def flush(self) -> None:
        for rid in tuple(self.requests):
            self.abort(rid)
        futures = [entry[2] for entry in self.cache.values()]
        self.cache.clear()
        self.cache_bytes = 0

        def release_cache() -> None:
            for future in futures:
                self._release_future(future)

        self._worker.submit(release_cache).result()
        from relax.utils.logging_utils import get_logger

        get_logger(__name__).info("mm_replay_producer cumulative: %s", self.metrics)

    def close(self) -> None:
        # ThreadPoolExecutor's interpreter-exit hook may already have joined
        # workers. Never submit new work from this final process cleanup.
        self._worker.shutdown(wait=True)
        futures = [item.future for media in self.requests.values() for item in media if item.future is not None]
        futures.extend(entry[2] for entry in self.cache.values())
        self.requests.clear()
        self.cache.clear()
        self.cache_bytes = 0
        for future in futures:
            self._release_future(future)


def accept_mm_response(sample: Any, meta: dict) -> dict:
    """Validate CPU layout before removing raw pixels from the sample queue."""
    from ..ref_data import validate_sample

    if (meta.get("finish_reason") or {}).get("type") == "abort":
        return {"version": 1, "media": []}
    entries = meta.get(META_FIELD)
    if not isinstance(entries, list):
        raise RuntimeError("SGLang did not return MM embedding references")
    envelope = {"version": 1, "media": entries}
    validate_sample({"mm_embedding": envelope}, sample.tokens)
    inputs = sample.multimodal_train_inputs or {}
    for modality, key in (("image", "image_grid_thw"), ("video", "video_grid_thw")):
        grid = inputs.get(key)
        expected = [entry["grid"] for entry in entries if entry["modality"] == modality]
        if grid is None:
            if expected:
                raise ValueError("MM replay is missing training media grid metadata")
            continue
        if hasattr(grid, "device") and grid.device.type != "cpu":
            raise ValueError("MM replay ingress requires CPU grid metadata")
        actual = grid.numpy().reshape(-1, 3).tolist() if hasattr(grid, "numpy") else [list(row) for row in grid]
        if actual != expected:
            raise ValueError("Rollout MM replay and training media grids differ")
    if any(key in inputs for key in ("pixel_values", "pixel_values_videos")) and not entries:
        raise ValueError("MM replay response omitted visual embeddings")
    sample.multimodal_train_inputs = {
        key: value for key, value in inputs.items() if key not in ("pixel_values", "pixel_values_videos")
    }
    return envelope


def _aborted(req: Any) -> bool:
    return any(type(getattr(req, name, None)).__name__ == "FINISH_ABORT" for name in ("finished_reason", "to_finish"))


def validate_capture_api() -> None:
    """Reject incompatible installed engines without importing their CUDA
    code."""
    import ast
    import importlib.util
    from pathlib import Path

    spec = importlib.util.find_spec("sglang")
    if spec is None or spec.origin is None:
        raise RuntimeError("MM replay requires an installed SGLang package")
    root = Path(spec.origin).parent / "srt" / "managers"
    required = {
        "mm_schedule.py": {
            "_batch_encode_per_image_misses",
            "_get_precomputed_embedding",
            "_get_chunked_embedding_full",
            "get_embedding_chunk",
        },
        "mm_utils.py": {"general_mm_embed_routine"},
    }
    for filename, names in required.items():
        path = root / filename
        functions = (
            set()
            if not path.is_file()
            else {node.name for node in ast.parse(path.read_text()).body if isinstance(node, ast.FunctionDef)}
        )
        if not names <= functions:
            raise RuntimeError(f"Installed SGLang lacks MM replay capture hooks in {filename}")
    contracts = (
        (
            "scheduler_components/output_streamer.py",
            "SchedulerOutputStreamer",
            {"build_additional_customized_info"},
            set(),
        ),
        ("io_struct.py", "GenerateReqInput", set(), {"rid", "cache_salt"}),
    )
    for filename, name, methods, fields in contracts:
        path = root / filename
        classes = (
            []
            if not path.is_file()
            else [
                node
                for node in ast.parse(path.read_text()).body
                if isinstance(node, ast.ClassDef) and node.name == name
            ]
        )
        if len(classes) != 1:
            raise RuntimeError(f"Installed SGLang lacks the MM replay {name} contract")
        body = classes[0].body
        available_methods = {node.name for node in body if isinstance(node, ast.FunctionDef)}
        available_fields = {
            node.target.id for node in body if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
        }
        if not methods <= available_methods or not fields <= available_fields:
            raise RuntimeError(f"Installed SGLang lacks the MM replay {name} contract")


def install_capture_patch(manager: Any) -> None:
    """Install only in opt-in scheduler workers, without editing SGLang
    files."""
    import sglang.srt.managers.mm_schedule as schedule
    import sglang.srt.managers.mm_utils as utilities
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.managers.scheduler import Scheduler
    from sglang.srt.runtime_context import get_parallel

    if getattr(Scheduler, "_relax_mm_replay", False):
        return

    def owner() -> bool:
        parallel = get_parallel()
        return (
            parallel.attn_tp_rank == 0
            and getattr(parallel, "attn_cp_rank", 0) == 0
            and getattr(parallel, "pp_rank", 0) == 0
        )

    runtime = None

    def get_runtime() -> MMCapture | None:
        nonlocal runtime
        # Installation precedes SGLang runtime initialization. Resolve ownership
        # only after the first scheduler/model callback enters the initialized runtime.
        if runtime is None and owner():
            import atexit

            runtime = MMCapture(manager.store())
            atexit.register(runtime.close)
        return runtime

    original_general = utilities.general_mm_embed_routine
    signature = inspect.signature(original_general)

    @wraps(original_general)
    def general(*args: Any, **kwargs: Any) -> Any:
        active = get_runtime()
        if active is None:
            return original_general(*args, **kwargs)
        values = signature.bind_partial(*args, **kwargs).arguments
        batch = values["forward_batch"]
        if batch.mm_inputs is not None:
            for rid, mm_inputs in zip(batch.rids or [], batch.mm_inputs, strict=True):
                active.bind(rid, mm_inputs)
        if not any(rid.startswith(REQUEST_PREFIX) for rid in batch.rids or []):
            return original_general(*args, **kwargs)
        token = _CAPTURE.set(active)
        try:
            return original_general(*args, **kwargs)
        finally:
            _CAPTURE.reset(token)

    utilities.general_mm_embed_routine = general
    for name in ("sglang.srt.models.qwen3_vl", "sglang.srt.models.qwen4_exp"):
        module = sys.modules.get(name)
        if module is not None and getattr(module, "general_mm_embed_routine", None) is original_general:
            module.general_mm_embed_routine = general

    original_batch = schedule._batch_encode_per_image_misses

    @wraps(original_batch)
    def batch_misses(data_embedding_func: Any, requests: Any, device: Any) -> Any:
        result = original_batch(data_embedding_func, requests, device)
        capture = _CAPTURE.get()
        if capture is not None:
            for request in requests:
                for _, item, start, end in request.overlapping:
                    capture.capture(item, result[(item.hash, end - start + 1)])
        return result

    schedule._batch_encode_per_image_misses = batch_misses

    original_precomputed = schedule._get_precomputed_embedding

    @wraps(original_precomputed)
    def precomputed(items: Any, *args: Any, **kwargs: Any) -> Any:
        result = original_precomputed(items, *args, **kwargs)
        capture = _CAPTURE.get()
        if capture is not None and result is not None:
            for item in items:
                capture.capture(item, item.precomputed_embeddings)
        return result

    schedule._get_precomputed_embedding = precomputed

    original_full = schedule._get_chunked_embedding_full

    @wraps(original_full)
    def full(data_embedding_func: Any, items: Any, *args: Any, **kwargs: Any) -> Any:
        token = _FULL_ITEMS.set(items)
        try:
            return original_full(data_embedding_func, items, *args, **kwargs)
        finally:
            _FULL_ITEMS.reset(token)

    schedule._get_chunked_embedding_full = full
    original_chunk = schedule.get_embedding_chunk

    @wraps(original_chunk)
    def chunk(embedding: Any, *args: Any, **kwargs: Any) -> Any:
        capture, items = _CAPTURE.get(), _FULL_ITEMS.get()
        if capture is not None and items:
            offset = 0
            for item in items:
                _, grid, _ = media_description(item)
                rows = grid[0] * grid[1] * grid[2] // 4
                capture.capture(item, embedding[offset : offset + rows])
                offset += rows
            if offset != embedding.shape[0]:
                raise ValueError("MM replay fallback embedding differs from media layout")
        return original_chunk(embedding, *args, **kwargs)

    schedule.get_embedding_chunk = chunk

    original_run = Scheduler.run_batch

    @wraps(original_run)
    def run(self: Any, batch: Any, *args: Any, **kwargs: Any) -> Any:
        active = get_runtime()
        if active is not None:
            for req in batch.reqs:
                if not getattr(req, "finished_output", False) and not _aborted(req):
                    active.bind(req.rid, req.multimodal_inputs)
        return original_run(self, batch, *args, **kwargs)

    Scheduler.run_batch = run

    def cleanup_request(req: Any) -> None:
        from .sglang import _abort_request_publication

        if runtime is not None:
            runtime.abort(req.rid)
        # A combined MM+R3 request can be parked before the ordinary replay
        # output wrapper has registered it in its own pending dictionary.
        _abort_request_publication(req)

    def request_ready(req: Any) -> bool:
        try:
            return runtime.ready(req.rid)
        except BaseException as primary:
            try:
                cleanup_request(req)
            except Exception as cleanup:
                primary.add_note(f"MM request cleanup failed: {cleanup!r}")
            raise

    def finish_request(req: Any) -> list[dict]:
        try:
            return runtime.finish(req.rid)
        except BaseException as primary:
            try:
                cleanup_request(req)
            except Exception as cleanup:
                primary.add_note(f"MM request cleanup failed: {cleanup!r}")
            raise

    original_streamer = Scheduler.get_output_streamer_class

    def streamer(scheduler: Any) -> type:
        base = original_streamer(scheduler)

        class MMOutputStreamer(base):
            has_additional_customized_info = True

            def stream_output(self, reqs: Any, return_logprob: Any, skip_req: Any = None) -> Any:
                if get_runtime() is None:
                    return super().stream_output(reqs, return_logprob, skip_req)
                pending = getattr(scheduler, "_relax_mm_pending", {})
                scheduler._relax_mm_pending = pending
                immediate = []
                for req in reqs:
                    # The native scheduler may revisit a completed request while
                    # its asynchronous publication is parked here.
                    if req.finished_output:
                        pending.pop(req.rid, None)
                        continue
                    if _aborted(req):
                        cleanup_request(req)
                        immediate.append(req)
                    elif (
                        req is not skip_req
                        and not req.finished_output
                        and req.rid.startswith(REQUEST_PREFIX)
                        and req.finished()
                        and not request_ready(req)
                    ):
                        if len(pending) >= scheduler.max_running_requests:
                            oldest = next(iter(pending))
                            for media in runtime.requests[oldest]:
                                if media.future is not None:
                                    media.future.result()
                            drain(scheduler)
                        pending[req.rid] = (req, return_logprob)
                    else:
                        pending.pop(req.rid, None)
                        immediate.append(req)
                if immediate:
                    return super().stream_output(immediate, return_logprob, skip_req)
                return None

            def build_additional_customized_info(self, reqs: Any) -> dict:
                result = super().build_additional_customized_info(reqs)
                result[META_FIELD] = [
                    finish_request(req)
                    if runtime is not None
                    and req.rid.startswith(REQUEST_PREFIX)
                    and req.finished()
                    and not _aborted(req)
                    else []
                    for req in reqs
                ]
                return result

        return MMOutputStreamer

    Scheduler.get_output_streamer_class = streamer

    def drain(scheduler: Any) -> None:
        if runtime is None:
            return
        pending = getattr(scheduler, "_relax_mm_pending", {})
        for rid, (req, return_logprob) in tuple(pending.items()):
            if req.finished_output:
                pending.pop(rid, None)
                continue
            if _aborted(req):
                pending.pop(rid)
                cleanup_request(req)
                scheduler.output_streamer.stream_output([req], return_logprob)
            elif request_ready(req):
                pending.pop(rid)
                scheduler.output_streamer.stream_output([req], return_logprob)

    original_input = Scheduler.process_input_requests

    @wraps(original_input)
    def process_input(self: Any, *args: Any, **kwargs: Any) -> Any:
        result = original_input(self, *args, **kwargs)
        drain(self)
        return result

    Scheduler.process_input_requests = process_input
    original_idle = Scheduler.maybe_sleep_on_idle

    @wraps(original_idle)
    def idle(self: Any, *args: Any, **kwargs: Any) -> Any:
        if getattr(self, "_relax_mm_pending", None):
            import time

            time.sleep(0.001)
            return None
        return original_idle(self, *args, **kwargs)

    Scheduler.maybe_sleep_on_idle = idle
    original_cancel = Scheduler.abort_request

    @wraps(original_cancel)
    def cancel(self: Any, recv_req: Any, *args: Any, **kwargs: Any) -> Any:
        if runtime is not None:
            for rid in tuple(runtime.requests):
                if recv_req.abort_all or rid.startswith(recv_req.rid):
                    runtime.abort(rid)
            pending = getattr(self, "_relax_mm_pending", {})
            for rid, (req, _) in tuple(pending.items()):
                if recv_req.abort_all or rid.startswith(recv_req.rid):
                    cleanup_request(req)
                    pending.pop(rid)
                    req.finished_output = True
        return original_cancel(self, recv_req, *args, **kwargs)

    Scheduler.abort_request = cancel
    original_abort = Req.set_finish_with_abort

    @wraps(original_abort)
    def abort(req: Any, *args: Any, **kwargs: Any) -> Any:
        if runtime is not None:
            runtime.abort(req.rid)
        return original_abort(req, *args, **kwargs)

    Req.set_finish_with_abort = abort
    original_flush = Scheduler.flush_cache

    @wraps(original_flush)
    def flush(self: Any, *args: Any, **kwargs: Any) -> Any:
        # Finished requests awaiting publication are outside native queues.
        # A successful native KV flush must not invalidate their MM holds.
        if getattr(self, "_relax_mm_pending", None):
            drain(self)
        if getattr(self, "_relax_mm_pending", None) or getattr(self, "_relax_replay_pending", None):
            return False
        result = original_flush(self, *args, **kwargs)
        if result and runtime is not None:
            runtime.flush()
        return result

    Scheduler.flush_cache = flush
    Scheduler._relax_mm_replay = True
