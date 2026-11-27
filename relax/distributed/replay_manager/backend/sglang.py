# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""SGLang/rollout integration: request, capture, response and publication."""

import ast
import asyncio
import importlib
import importlib.util
import json
import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from copy import copy
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pybase64

from ..codecs import moe_route_codec
from ..core import ReplayManager, replay_data_manager
from ..producer import Producer
from ..ref_data import (
    MoERouteSpan,
    ReplayRef,
    decode_refs,
    encode_refs,
    iter_refs,
    moe_route_spans,
    validate_mm_envelope,
    validate_sample,
)


_CAPTURE_FIELDS = {"moe": "routed_experts"}
ROUTE_ATTEMPT_PREFIX = "relax-r3-attempt-"
AGENTIC_R3_REQUEST_PREFIX = "relax-agentic-r3-"
AGENTIC_R3_MM_PREFIX = AGENTIC_R3_REQUEST_PREFIX + "mm-"


@dataclass
class _CustomRouteCapture:
    # Child tasks inherit the capture object. Closing it also fences late responses in those tasks.
    refs: dict[str, ReplayRef] | None
    previous: dict[str, set[str]]
    parent: "_CustomRouteCapture | None"

    @property
    def active_refs(self) -> dict[str, ReplayRef] | None:
        if self.parent is not None and self.parent.active_refs is None:
            return None
        return self.refs


_custom_routes: ContextVar[_CustomRouteCapture | None] = ContextVar("custom_routes", default=None)


def track_custom_replay_outputs(generate: Callable) -> Callable:
    """Transfer newly adopted routes and the input's existing holds to the
    returned rows."""

    @wraps(generate)
    async def tracked(args, sample, sampling_params, evaluation=False):
        from relax.utils.async_utils import finish_before_cancellation

        from .. import replay_data_manager as manager

        custom = getattr(sample, "generate_function_path", None) or getattr(
            args, "custom_generate_function_path", None
        )
        parent = _custom_routes.get()
        if evaluation or not custom or not manager.uses_references:
            return await generate(args, sample, sampling_params, evaluation=evaluation)

        previous = {record: {lease} for record, lease in getattr(sample, "_replay_route_holds", {}).items()}
        # Generation can replace the input manifest in place. Snapshot the owned input refs first.
        refs = (
            {ref.record: ref for ref in iter_refs(sample.rollout_replay_refs or {}) if ref.record in previous}
            if previous
            else {}
        )
        capture = _CustomRouteCapture(refs, previous, parent)
        token = _custom_routes.set(capture)
        holders: dict[str, set[str]] = {}
        try:
            try:
                result = await generate(args, sample, sampling_params, evaluation=evaluation)
            finally:
                capture.refs = None
                _custom_routes.reset(token)
            if refs:
                rows = result if isinstance(result, list) else [result]
                seen = set()
                for index, row in enumerate(rows):
                    visible = {ref.record for ref in iter_refs(row.rollout_replay_refs or {})}
                    records = visible & refs.keys()
                    existing = getattr(row, "_replay_route_holds", {})
                    if not records and not existing:
                        continue
                    if id(row) in seen:
                        row = rows[index] = copy(row)
                    seen.add(id(row))
                    row._replay_route_holds = {key: lease for key, lease in existing.items() if key in visible}
                    for record in records:
                        lease = uuid4().hex
                        row._replay_route_holds[record] = lease
                        holders.setdefault(record, set()).add(lease)
                await finish_before_cancellation(
                    manager.store().invoke_async("assign_route_outputs", list(refs.values()), holders, previous),
                    "replay-assign-outputs",
                )
                if parent is not None:
                    parent_refs = parent.active_refs
                    if parent_refs is None:
                        raise asyncio.CancelledError
                    # Only finished child outputs pass to a still-running parent.
                    for record, leases in holders.items():
                        parent_refs[record] = refs[record]
                        parent.previous.setdefault(record, set()).update(leases)
            return result
        except BaseException as error:
            if refs:
                try:
                    await finish_before_cancellation(
                        manager.store().invoke_async(
                            "discard",
                            list(refs.values()),
                            keep_queued=True,
                            output_ids=set().union(*previous.values(), *holders.values()),
                        ),
                        "replay-discard-custom-generation",
                    )
                except Exception as cleanup:
                    error.add_note(f"Custom generation replay cleanup failed: {cleanup!r}")
            raise

    return tracked


def _response_references(value: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Decode either a binary wire envelope or an already decoded manifest.

    Kind validation belongs to the receiver. Cleanup uses all decoded kinds,
    including unexpected ones, to retain ownership after a parsing failure.
    """
    if "replay_manifest" in value:
        return decode_refs(pybase64.b64decode(value["replay_manifest"], validate=True))
    ref = ReplayRef.from_dict(value)
    return {ref.kind: value}


async def discard_failed_generation(
    manager: ReplayManager, tasks: list[asyncio.Task], group: list, error: BaseException
) -> None:
    """Drain sibling requests before releasing samples that will never enter
    the training queue."""
    for task in tasks:
        task.cancel()
    completed = await asyncio.gather(*tasks, return_exceptions=True)
    samples = list(group)
    for result in completed:
        if not isinstance(result, BaseException):
            samples.extend(result if isinstance(result, list) else [result])
    for sample in samples:
        try:
            await manager.discard_async([sample])
        except Exception as cleanup:
            error.add_note(f"Replay group cleanup failed: {cleanup!r}")


@dataclass(frozen=True)
class SGLangProducer(Producer):
    """Rollout-facing adapter: request capture, accept references, build TQ
    fields.

    In reference mode SGLang publishes layer chunks to the TQ body partition;
    the sample queue carries manifests. Full mode retains its tensor wire
    format.
    """

    def request_options(self, *, evaluation: bool = False, request_id: str | None = None) -> dict[str, Any]:
        from ..config import references_enabled
        from .mm_sglang import AGENTIC_MM_ATTEMPT_PREFIX, ATTEMPT_PREFIX, REQUEST_PREFIX

        self.manager._ensure_open()
        options = {}
        reference = self.manager.uses_references if request_id is None else references_enabled()
        media = "mm_embedding" in self.kinds and not evaluation
        if "moe" in self.kinds and not (evaluation and reference):
            options["return_routed_experts"] = True
        if request_id is not None:
            if "moe" in self.kinds and reference:
                # A retry of the same logical request can overlap an unacknowledged registry close.
                if not evaluation:
                    request_id = f"{request_id}-{uuid4().hex}"
                request_id = (AGENTIC_R3_MM_PREFIX if media else AGENTIC_R3_REQUEST_PREFIX) + request_id
            elif media:
                request_id = AGENTIC_MM_ATTEMPT_PREFIX + uuid4().hex if reference else REQUEST_PREFIX + request_id
            options["rid"] = request_id
        elif media:
            options["rid"] = ATTEMPT_PREFIX + uuid4().hex
        elif reference and "moe" in self.kinds and not evaluation:
            options["rid"] = ROUTE_ATTEMPT_PREFIX + uuid4().hex
        if media:
            options["cache_salt"] = "relax-mm-embedding-replay"
        return options

    def capture_tokens_before_normalization(
        self, prefix: list[int], generated: list[int], tokenizer: Any, processor: Any, *, evaluation: bool
    ) -> list[int] | None:
        """Keep original route row IDs when rollout can replace media
        tokens."""
        if not self.manager.uses_references or evaluation or not self.kinds:
            return None
        if processor is not None or any(
            hasattr(tokenizer, key) for key in ("image_token_id", "audio_token_id", "video_token_id")
        ):
            return prefix + list(generated)
        return None

    def accept_response(
        self, sample: Any, meta: dict, args: Any, *, evaluation: bool = False, captured_tokens: list[int] | None = None
    ) -> None:
        self.manager._ensure_open()
        if self.manager.uses_references:
            if evaluation or not self.kinds:
                return
            refs = self.accept_route_refs(meta, args, sample.tokens, captured_tokens)
            if "mm_embedding" in self.kinds:
                from .mm_sglang import accept_training_mm_response

                refs["mm_embedding"] = accept_training_mm_response(sample, meta, args)
            sample.rollout_replay_refs = refs
        else:
            if "routed_experts" in meta:
                sample.rollout_routed_experts = decode_routed_experts(
                    meta, len(sample.tokens) - 1, args.num_layers, args.moe_router_topk
                )

    def accept_route_refs(
        self, meta: dict, args: Any, tokens: list[int], captured_tokens: list[int] | None = None
    ) -> dict[str, dict]:
        """Accept SGLang route references against the capture and training
        token sequences."""
        capture = tokens if captured_tokens is None else captured_tokens
        refs = {}
        if len(capture) != len(tokens):
            raise ValueError("Replay token normalization must preserve row positions")
        for kind in self.kinds - {"mm_embedding"}:
            field = _CAPTURE_FIELDS[kind]
            value = meta[field]
            if isinstance(value, dict) and "replay_manifest" in value:
                decoded = _response_references(value)
                if set(decoded) != {kind}:
                    raise ValueError("SGLang response contains unexpected replay kinds")
                value = meta[field] = decoded[kind]
            if not isinstance(value, dict):
                # External/unpatched SGLang keeps its original HTTP payload.
                # Publish it at ingress; TQ still carries only a reference.
                values = decode_routed_experts(meta, len(tokens) - 1, args.num_layers, args.moe_router_topk)
                if hasattr(args, "tq_config"):
                    self.manager.store().register_local_owner(args.tq_config)
                value = self._publish_response(
                    values, kind=kind, tokens=capture, codec=moe_route_codec(getattr(args, "num_experts", None))
                )
                # The enclosing response guard now owns cleanup on failure.
                meta[field] = value
            if capture != tokens:
                validate_sample({kind: value}, capture)
                from ..ref_data import token_digest

                value = {**value, "training_tokens": token_digest(tokens)}
            refs[kind] = value
        if refs:
            validate_sample(refs, tokens)
        return refs

    def response_media(self, meta: dict, *, required: bool = False) -> dict | None:
        """Decode the media envelope, preserving a distinct lease for each
        occurrence."""
        from .mm_sglang import response_media

        if "mm_embedding" not in self.kinds:
            return None
        envelope = response_media(meta)
        if required and envelope is None:
            raise ValueError("Backend omitted multimodal embedding references")
        return envelope

    @contextmanager
    def response_holds(self, meta: dict) -> Iterator[tuple[list[ReplayRef], list[dict]]]:
        """Keep wire ownership guarded while the receiver adopts decoded
        records and leases."""
        with self.response(meta):
            envelope = self.response_media(meta)
            if envelope is not None:
                validate_mm_envelope(envelope)
            yield self.response_refs(meta), envelope["media"] if envelope is not None else []

    @contextmanager
    def response_route_span(
        self, meta: dict, args: Any, tokens: list[int], captured_tokens: list[int], start: int, *, required: bool
    ) -> Iterator[MoERouteSpan | None]:
        """Guard the route publication through validation and merge into the
        accepted lineage."""
        if meta.get("routed_experts") is None:
            if required:
                raise ValueError("Backend omitted routed experts for newly accepted tokens")
            yield None
            return

        # The receiver already holds the media leases. A failed route merge
        # must clean up only the route (including inline ingress publication).
        routes = self if self.kinds == {"moe"} else self.manager.register_producer({"moe"})
        with routes.response(meta):
            refs = routes.accept_route_refs(meta, args, tokens[start:], captured_tokens[start:])
            (span,) = moe_route_spans(refs["moe"])
            if (span.ref.layers, span.ref.topk) != (args.num_layers, args.moe_router_topk):
                raise ValueError("Agentic MoE replay shape does not match the model")
            yield MoERouteSpan(span.ref, start, len(tokens) - 1, span.training_tokens)

    @staticmethod
    def response_refs(meta: dict) -> list[ReplayRef]:
        return [
            ReplayRef.from_dict(ref)
            for field in _CAPTURE_FIELDS.values()
            if isinstance(value := meta.get(field), dict)
            for ref in _response_references(value).values()
        ]

    async def adopt_native_response(self, sample: Any, args: Any, meta: dict) -> None:
        # Custom generators may fan one response out to independently queued samples.
        custom = getattr(sample, "generate_function_path", None) or getattr(
            args, "custom_generate_function_path", None
        )
        refs = self.response_refs(meta)
        owner = uuid4().hex
        await self.adopt_response(refs, owner)
        scope = _custom_routes.get()
        if refs and scope is not None:
            active_refs = scope.active_refs
            if active_refs is None:
                from relax.utils.async_utils import finish_before_cancellation

                await finish_before_cancellation(
                    self.manager.store().invoke_async("discard", refs, owner=owner), "replay-discard-late-response"
                )
                raise asyncio.CancelledError
            active_refs.update((ref.record, ref) for ref in refs)
        elif not custom:
            for ref in refs:
                self.manager.native_route_owners[ref.record] = sample

    async def adopt_response(self, refs: list[ReplayRef], owner: str) -> None:
        """Settle any in-flight adoption before compensating a cancelled
        receiver."""
        import asyncio

        if not refs:
            return
        store = self.manager.store()
        adoption = asyncio.create_task(store.invoke_async("adopt_response", refs, owner))
        try:
            await asyncio.shield(adoption)
        except BaseException as error:
            try:
                await asyncio.gather(adoption, return_exceptions=True)
                await store.invoke_async("discard", refs, owner=owner)
            except Exception as cleanup:
                error.add_note(f"Replay response handoff cleanup failed: {cleanup!r}")
            raise

    def discard_response(self, meta: dict, *, evaluation: bool = False) -> None:
        """Release each kind's ownership in a discarded response."""
        if not self.manager.uses_references or evaluation:
            return
        errors = []
        if "mm_embedding" in self.kinds:
            from ..mm_embedding import release_media_entries
            from .mm_sglang import META_FIELD

            entries = meta.get(META_FIELD, [])
            if entries:
                try:
                    release_media_entries(self.manager.store(), entries)
                except Exception as cleanup:
                    errors.append(cleanup)
        for field in _CAPTURE_FIELDS.values():
            value = meta.get(field)
            if isinstance(value, dict):
                try:
                    refs = _response_references(value)
                    self.manager.store().invoke(
                        "discard", [ReplayRef.from_dict(ref) for ref in refs.values()], unadopted=True
                    )
                except Exception as cleanup:
                    errors.append(cleanup)
        if errors:
            error, *others = errors
            for other in others:
                error.add_note(f"Additional replay response cleanup failure: {other!r}")
            raise error

    @contextmanager
    def response(self, meta: dict, *, evaluation: bool = False):
        """Hold response ownership until parsing succeeds; failed responses
        never reach TQ."""
        try:
            yield
        except BaseException as error:
            try:
                self.discard_response(meta, evaluation=evaluation)
            except Exception as cleanup:
                error.add_note(f"Replay response cleanup failed: {cleanup!r}")
            raise


async def post_generation_request(
    manager: ReplayManager,
    post: Callable,
    url: str,
    payload: dict,
    *,
    headers: dict | None,
    fail_fast_no_workers: bool,
    retry_after_no_workers: bool = False,
) -> dict:
    """Keep response leases with the attempt until the receiver has its
    manifest."""
    import httpx

    from relax.utils.async_utils import finish_before_cancellation
    from relax.utils.http_utils import MAX_RETRIES

    from .mm_sglang import ATTEMPT_PREFIX, META_FIELD

    rid = payload.get("rid", "")
    agentic_routes = rid.startswith(AGENTIC_R3_REQUEST_PREFIX) and payload.get("return_routed_experts")
    if not agentic_routes and not rid.startswith((ATTEMPT_PREFIX, ROUTE_ATTEMPT_PREFIX)):
        return await post(url, payload, headers=headers, fail_fast_no_workers=fail_fast_no_workers)

    if retry_after_no_workers:
        # The previous registry close may still be in flight after the router's 503.
        prefix = ATTEMPT_PREFIX if rid.startswith(ATTEMPT_PREFIX) else ROUTE_ATTEMPT_PREFIX
        rid = prefix + uuid4().hex
        payload = {**payload, "rid": rid}
    store = manager.store()
    meta = {}
    try:
        await finish_before_cancellation(store.open_response_attempt(rid), "open-replay-attempt")
        attempts = MAX_RETRIES if rid.startswith(ROUTE_ATTEMPT_PREFIX) else 1
        for index in range(attempts):
            try:
                output = await post(
                    url, payload, headers=headers, fail_fast_no_workers=fail_fast_no_workers, max_retries=1
                )
                break
            except (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout):
                # These failures precede dispatch, so the same wire ID is safe to retry.
                if index + 1 == attempts:
                    raise
                await asyncio.sleep(1)
        meta = output["meta_info"]
        accepted = [entry["lease"] for entry in meta.get(META_FIELD, ())]
        accepted.extend(ref.record for ref in SGLangProducer.response_refs(meta))
        await finish_before_cancellation(
            store.invoke_async("finish_response_attempt", rid, accepted), "accept-replay-response"
        )
        return output
    except BaseException as error:
        try:
            await finish_before_cancellation(store.abandon_response_attempt(rid), "close-replay-attempt")
        except (Exception, asyncio.CancelledError) as cleanup:
            error.add_note(f"Replay attempt {rid} cleanup failed: {cleanup!r}")
        # Acceptance may have committed just before its waiter was cancelled.
        if meta:
            try:
                manager.register_producer({"mm_embedding"}).discard_response(meta)
            except Exception as cleanup:
                error.add_note(f"Replay response cleanup failed: {cleanup!r}")
        raise


def decode_routed_experts(meta_info: dict[str, Any], num_rows: int, num_layers: int, topk: int) -> np.ndarray:
    values = np.frombuffer(pybase64.b64decode(meta_info["routed_experts"], validate=True), dtype=np.int32)
    expected_shape = (num_rows, num_layers, topk)
    if values.size != num_rows * num_layers * topk:
        raise ValueError(
            f"Routed-expert alignment mismatch: {values.size=} {expected_shape=}, "
            f"finish_reason={meta_info.get('finish_reason')}, prompt_tokens={meta_info.get('prompt_tokens')}, "
            f"completion_tokens={meta_info.get('completion_tokens')}, "
            f"output_logprobs={len(meta_info.get('output_token_logprobs') or [])}"
        )
    return values.reshape(expected_shape)


def _scheduler_contract() -> tuple[str, str, str]:
    root = Path(importlib.util.find_spec("sglang").origin).parent
    if (root / "srt/managers/scheduler_output_processor_mixin.py").is_file():
        return "sglang.srt.managers.scheduler_output_processor_mixin", "SchedulerOutputProcessorMixin", ""
    return (
        "sglang.srt.managers.scheduler_components.batch_result_processor",
        "SchedulerBatchResultProcessor",
        "_",
    )


def _validate_capture_api(kinds: frozenset[str]) -> None:
    """Check installed source without importing SGLang's CUDA runtime."""
    if "mm_embedding" in kinds:
        from .mm_sglang import validate_capture_api

        validate_capture_api()
    root = Path(importlib.util.find_spec("sglang").origin).parent
    module, classname, prefix = _scheduler_contract()

    def methods(path, name):
        if not path.is_file():
            return set()
        tree = ast.parse(path.read_text())
        cls = next((node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name), None)
        return {node.name for node in cls.body if isinstance(node, ast.FunctionDef)} if cls else set()

    collectors = methods(root.joinpath(*module.split(".")[1:]).with_suffix(".py"), classname)
    encoders = methods(root / "srt/managers/detokenizer_manager.py", "DetokenizerManager")
    for kind in kinds - {"mm_embedding"}:
        field = _CAPTURE_FIELDS[kind]
        if prefix + "maybe_collect_" + field not in collectors or not (
            "_b64_encode_per_request" in encoders or "_extract_" + field in encoders
        ):
            raise RuntimeError(f"Installed SGLang lacks the {kind} replay capture/publication API")


def _abort_request_publication(req: Any, *, keep_response: bool = False) -> None:
    """Drop only this request's unpublished bodies, after all writes drain."""
    error = None
    media = getattr(req, "_relax_replay_media", None)
    # A resumable pause still returns accepted tokens and both replay kinds.
    # Keep its independent media response leases until terminal output drains.
    if media is not None and not keep_response:
        try:
            media.abort(req.rid)
            req._relax_replay_media = None
        except Exception as exc:
            error = exc
    streams = getattr(req, "_relax_replay_streams", {})
    for kind, publication in streams.items():
        if keep_response and kind == "moe":
            continue
        try:
            publication.abort()
        except Exception as exc:
            error = error or exc
    if error is not None:
        raise error
    req._relax_replay_streams = {"moe": streams["moe"]} if keep_response and "moe" in streams else {}
    req._relax_replay_finish = None


def _request_aborted(req: Any) -> bool:
    return any(type(getattr(req, name, None)).__name__ == "FINISH_ABORT" for name in ("finished_reason", "to_finish"))


def _keep_aborted_response(req: Any) -> bool:
    return getattr(req, "_relax_replay_keep_aborted_response", False)


def _install_abort_patch() -> None:
    """Waiting/chunked requests can disappear without a finished collector."""
    from sglang.srt.managers.scheduler import Scheduler

    from .mm_sglang import AGENTIC_MM_ATTEMPT_PREFIX

    original = Scheduler.abort_request
    if getattr(original, "_relax_replay_cleanup", False):
        return

    @wraps(original)
    def abort(self, recv_req, *args, **kwargs):
        pending = getattr(self, "_relax_replay_pending", {})
        # Generation has already finished for these requests. Match native
        # abort semantics: let their terminal response wait for publication.
        # A bare AbortReq would retain tokens already sent to the tokenizer
        # while dropping the replay manifests needed to accept those tokens.
        completed = {key for key, (req, _) in pending.items() if req.finished()}
        requests = list(self.waiting_queue)
        if getattr(self, "chunked_req", None) is not None:
            requests.append(self.chunked_req)
        inactive = {id(req) for req in requests}
        for name in ("running_batch", "last_batch"):
            batch = getattr(self, name, None)
            if batch is not None:
                requests.extend(batch.reqs)
        targets = {
            id(req): req
            for req in requests
            if id(req) not in completed and (recv_req.abort_all or req.rid.startswith(recv_req.rid))
        }

        def cleanup(*, resumable: bool = False):
            error = None
            for req in targets.values():
                try:
                    finish = getattr(req, "to_finish", None)
                    # Keep only scheduler pauses that can still return tokens
                    # over HTTP. Error-status aborts never deliver a manifest.
                    keep_response = (
                        resumable
                        and id(req) not in inactive
                        and not getattr(req, "finished_output", False)
                        and not getattr(req, "is_retracted", False)
                        and (
                            req.rid.startswith(AGENTIC_MM_ATTEMPT_PREFIX)
                            or (
                                req.rid.startswith(AGENTIC_R3_REQUEST_PREFIX)
                                and getattr(req, "return_routed_experts", False)
                            )
                        )
                        and type(finish).__name__ == "FINISH_ABORT"
                        and getattr(finish, "status_code", None) is None
                    )
                    req._relax_replay_keep_aborted_response = keep_response
                    _abort_request_publication(req, keep_response=keep_response)
                except Exception as exc:
                    error = error or exc
            if error is not None:
                raise error

        try:
            result = original(self, recv_req, *args, **kwargs)
        except BaseException as primary:
            try:
                cleanup()
            except Exception as secondary:
                primary.add_note(f"Replay abort cleanup failed: {secondary!r}")
            raise
        cleanup(resumable=True)
        return result

    abort._relax_replay_cleanup = True
    Scheduler.abort_request = abort


def _snapshot_layer(buffer: Any, slots: Any, layer: int) -> np.ndarray:
    """Gather bounded stable rows into one owned chunk before yielding
    scheduler.

    Flattening a contiguous cache is a view: gather only this layer's rows
    directly into owned output. Strided caches retain the basic-slice path.
    Background I/O never dereferences a live/recyclable SGLang KV slot.
    """
    output = np.empty((len(slots), buffer.shape[2]), dtype=np.int32)
    if buffer.flags.c_contiguous:
        # _advance_publication already validates slots. clip avoids take's
        # extra output buffer in raise mode; no invalid slot is clamped here.
        indices = np.asarray(slots, dtype=np.intp) * buffer.shape[1] + layer
        np.take(buffer.reshape(-1, buffer.shape[2]), indices, axis=0, out=output, mode="clip")
        return output
    boundaries = np.flatnonzero(np.diff(slots) != 1) + 1
    for start, end in zip(np.r_[0, boundaries], np.r_[boundaries, len(slots)], strict=True):
        src = int(slots[start])
        np.copyto(output[start:end], buffer[src : src + int(end - start), layer], casting="no")
    return output


def _advance_publication(
    processor: Any,
    req: Any,
    target: int,
    capturers: dict[str, Any],
    *,
    final: bool,
    asynchronous: bool = False,
    engine_publisher: tuple[int, str] | None = None,
) -> dict:
    """Publish newly stable whole chunks; seal the remaining tail only at
    finish."""
    generation = getattr(req, "retraction_count", 0)
    if getattr(req, "_relax_replay_generation", generation) != generation:
        _abort_request_publication(req)
    req._relax_replay_generation = generation
    if not hasattr(req, "_relax_replay_streams"):
        req._relax_replay_streams = {}
    streams = req._relax_replay_streams
    agentic_r3 = getattr(req, "rid", "").startswith(AGENTIC_R3_REQUEST_PREFIX)
    moe_start = getattr(req, "routed_experts_start_len", 0) if agentic_r3 else 0
    try:
        for kind, getter in capturers.items():
            field = _CAPTURE_FIELDS[kind]
            if not getattr(req, "return_" + field, False):
                continue
            start_row = moe_start if kind == "moe" else 0
            captured_rows = max(0, target - start_row)
            if not captured_rows and kind not in streams:
                continue
            capturer = getter()
            if capturer is None:
                raise RuntimeError(f"SGLang {kind} capturer is unavailable")
            host = capturer.host_cache if hasattr(capturer, "host_cache") else capturer.get_host_cache()
            buffer = host.buffer
            if buffer.device.type != "cpu" or str(buffer.dtype) != "torch.int32" or buffer.ndim != 3:
                raise RuntimeError("Replay streaming requires completed int32 CPU capture storage")
            codec = getattr(capturer, "publication_codec", None)
            if kind not in streams:
                model_config = getattr(getattr(processor, "model_config", None), "hf_text_config", None)
                num_experts = getattr(model_config, "num_experts", None) or getattr(
                    model_config, "n_routed_experts", None
                )
                streams[kind] = replay_data_manager.register_producer([kind]).begin(
                    kind=kind,
                    layers=buffer.shape[1],
                    topk=capturer.topk_size if codec is not None else buffer.shape[2],
                    codec=codec.name if codec is not None else moe_route_codec(num_experts),
                    engine_publisher=engine_publisher,
                )
            publication = streams[kind]
            if captured_rows < publication.rows:
                raise RuntimeError("Finalized token range overlaps already published replay chunks")
            end = captured_rows if final else captured_rows // publication.chunk_rows * publication.chunk_rows
            if end > publication.rows:
                # Relax's older async D2H patch defers the CPU slot scatter until
                # the legacy getter. Streaming bypasses that getter, so drain it here.
                flush = getattr(capturer, "_flush_pending_scatter", None)
                if flush is not None:
                    flush()
                first = publication.rows
                # Transfer the lightweight stable slot range once. Body copies
                # remain chunked and admitted by the writer before KV reuse.
                kv = getattr(req, "kv", req)
                slots = (
                    processor.req_to_token_pool.req_to_token[kv.req_pool_idx][first + start_row : end + start_row]
                    .cpu()
                    .numpy()
                )
                if len(slots) != end - first or np.any(slots < 0) or np.any(slots >= len(buffer)):
                    raise ValueError("Replay capture slot range is invalid")
                cache = buffer.numpy()
            while publication.rows < end:
                start = publication.rows
                stop = min(start + publication.chunk_rows, end)
                chunk_slots = slots[start - first : stop - first]
                publication.append(
                    stop - start,
                    (lambda layer: capturer.snapshot_layer(chunk_slots, layer, stop))
                    if codec is not None
                    else (lambda layer: _snapshot_layer(cache, chunk_slots, layer)),
                    encoded_width=capturer.encoded_width(stop) if codec is not None else None,
                )
        if not final:
            return {}
        output_ids = req.output_ids_through_stop if hasattr(req, "output_ids_through_stop") else req.output_ids
        prompt_ids = req.origin_input_ids
        unpadded = getattr(req, "origin_input_ids_unpadded", prompt_ids)
        if len(unpadded) == len(prompt_ids):
            prompt_ids = unpadded
        tokens = list(prompt_ids) + list(output_ids)
        from .mm_sglang import ATTEMPT_PREFIX

        rid = getattr(req, "rid", "")
        attempt = rid if agentic_r3 or rid.startswith((ATTEMPT_PREFIX, ROUTE_ATTEMPT_PREFIX)) else None
        pending = {
            kind: publication.finish_async(tokens[moe_start:] if kind == "moe" else tokens, attempt=attempt)
            for kind, publication in streams.items()
        }
        return pending if asynchronous else {kind: future.result().to_dict() for kind, future in pending.items()}
    except BaseException as primary:
        try:
            _abort_request_publication(req)
        except Exception as cleanup:
            raise primary from cleanup
        raise


def install_scheduler_patch(
    manager: ReplayManager | None = None,
    *,
    kinds: frozenset[str] | None = None,
) -> None:
    """Intercept completed batch copies and finished responses in SGLang only.

    The intermediate layer owns publication. SGLang still owns capture, token
    acceptance, cache slots and scheduling; TQ receives only sealed references.
    """
    global replay_data_manager
    selected = replay_data_manager if manager is None else manager
    selected_kinds = frozenset(_CAPTURE_FIELDS) if kinds is None else kinds
    if not selected_kinds.issubset(_CAPTURE_FIELDS):
        raise ValueError("The SGLang capture adapter does not implement the requested replay kind")
    module, classname, prefix = _scheduler_contract()
    upstream = importlib.import_module(module)
    cls = getattr(upstream, classname)
    if getattr(cls, "_relax_replay_references", False):
        if getattr(cls, "_relax_replay_manager", None) is not selected:
            raise RuntimeError("SGLang replay hooks are already bound to another manager")
        if getattr(cls, "_relax_replay_kinds", selected_kinds) != selected_kinds:
            raise RuntimeError("SGLang replay hooks are already bound to another kind selection")
        return
    replay_data_manager = selected
    cls._relax_replay_manager = selected
    cls._relax_replay_kinds = selected_kinds
    raw_publisher = os.environ.get("RELAX_REPLAY_ENGINE_PUBLISHER")
    engine_publisher = tuple(json.loads(raw_publisher)) if raw_publisher is not None else None
    capturers = {
        kind: getattr(upstream, name)
        for kind, name in (("moe", "get_global_experts_capturer"),)
        if kind in selected_kinds and hasattr(upstream, name)
    }
    collectors = {kind: getattr(cls, prefix + "maybe_collect_" + _CAPTURE_FIELDS[kind]) for kind in capturers}

    def native_request(req):
        # Keep upstream's partial-range and old speculative-stop semantics.
        # These requests use its collector and are published at rollout ingress.
        return (
            bool(getattr(req, "routed_experts_start_len", 0))
            and not getattr(req, "rid", "").startswith(AGENTIC_R3_REQUEST_PREFIX)
        ) or (
            not prefix
            and hasattr(req, "output_ids_through_stop")
            and len(req.output_ids_through_stop) != len(req.output_ids)
        )

    def owner(self):
        if prefix:
            from sglang.srt.runtime_context import get_parallel

            parallel = get_parallel()
        else:
            parallel = self
        # Match the response-producing rank within each attention-DP replica.
        # Global TP0 would omit every other DP replica's requests.
        rank = getattr(parallel, "attn_tp_rank", None)
        if rank is None:
            rank = parallel.tp_rank
        return rank == 0 and getattr(parallel, "attn_cp_rank", 0) == 0 and getattr(parallel, "pp_rank", 0) == 0

    if prefix and "moe" in capturers:
        _install_owner_capture_patch(lambda: owner(None))

    asynchronous = _install_response_patch(owner)

    def collect(self, req):
        if getattr(req, "_relax_replay_finish", None) is not None:
            return
        if getattr(req, "_relax_replay_published", False) or getattr(req, "_relax_replay_native_collected", False):
            return
        kinds = {kind for kind, field in _CAPTURE_FIELDS.items() if getattr(req, "return_" + field, False)}
        if not kinds:
            return
        if not kinds <= capturers.keys():
            raise RuntimeError("SGLang lacks a requested replay capturer")
        output_ids = req.output_ids_through_stop if hasattr(req, "output_ids_through_stop") else req.output_ids
        if _request_aborted(req):
            if not (_keep_aborted_response(req) and output_ids and "moe" in kinds):
                _abort_request_publication(req)
                return
            _abort_request_publication(req, keep_response=True)
            kinds = {"moe"}
        if not owner(self):
            for kind in kinds:
                setattr(req, _CAPTURE_FIELDS[kind], None)
            return
        if native_request(req):
            _abort_request_publication(req)
            for kind in kinds:
                collectors[kind](self, req)
            req._relax_replay_native_collected = True
            return
        refs = _advance_publication(
            self,
            req,
            len(req.origin_input_ids) + len(output_ids) - 1,
            {kind: capturers[kind] for kind in kinds},
            final=True,
            asynchronous=asynchronous,
            engine_publisher=engine_publisher,
        )
        if asynchronous and refs:
            req._relax_replay_finish = refs
            return
        for kind, ref in refs.items():
            setattr(req, _CAPTURE_FIELDS[kind], ref)
        # Ownership now belongs to the response/rollout handoff, not the live Req.
        req._relax_replay_streams = {}
        req._relax_replay_published = True

    def wrap_result(original):
        @wraps(original)
        def process(self, batch, result, *args, **kwargs):
            if not owner(self):
                return original(self, batch, result, *args, **kwargs)
            try:
                # These are the upstream method's existing copy/finalize actions,
                # moved before publication. Clear holders so upstream won't redo.
                if getattr(result, "copy_done", None) is not None:
                    result.copy_done.synchronize()
                for field in ("routed_experts_output",):
                    capture = getattr(result, field, None)
                    if capture is not None:
                        capture.finalize()
                        setattr(result, field, None)
                for i, req in enumerate(batch.reqs):
                    if _request_aborted(req):
                        if not _keep_aborted_response(req):
                            _abort_request_publication(req)
                    elif not req.finished() and not req.is_retracted and not native_request(req):
                        # Only the already accepted prefix is stable. A verify
                        # batch may also contain speculative tokens not accepted yet.
                        accepted = len(req.origin_input_ids) + len(req.output_ids)
                        target = min(int(batch.seq_lens_cpu[i]), accepted) - 1
                        _advance_publication(
                            self, req, max(0, target), capturers, final=False, engine_publisher=engine_publisher
                        )
                return original(self, batch, result, *args, **kwargs)
            except BaseException as primary:
                for req in batch.reqs:
                    try:
                        _abort_request_publication(req)
                    except Exception as cleanup:
                        primary.add_note(f"Replay publication cleanup failed: {cleanup!r}")
                raise

        return process

    for kind in capturers:
        setattr(cls, prefix + "maybe_collect_" + _CAPTURE_FIELDS[kind], collect)
    if capturers:
        for method in ("process_batch_result_prefill", "process_batch_result_decode"):
            setattr(cls, method, wrap_result(getattr(cls, method)))
    from sglang.srt.managers.schedule_batch import Req

    _install_abort_patch()
    for method in ("reset_for_retract", "set_finish_with_abort"):
        original = getattr(Req, method)

        def reset(req, *args, _original=original, **kwargs):
            req._relax_replay_keep_aborted_response = False
            _abort_request_publication(req)
            result = _original(req, *args, **kwargs)
            req._relax_replay_generation = getattr(req, "retraction_count", 0)
            return result

        if not getattr(original, "_relax_replay_cleanup", False):
            reset._relax_replay_cleanup = True
            setattr(Req, method, reset)
    cls._relax_replay_references = True


def _install_owner_capture_patch(owner: Callable[[], bool]) -> None:
    """Only response owners export MoE routing snapshots in reference mode.

    Keep the capturer and capture() intact on every rank: scattered MoE
    backends may run collectives there, and cache policy uses its presence.
    Only the result snapshot/D2H is redundant on non-response ranks.
    """
    from sglang.srt.state_capturer.routed_experts import RoutedExpertsCapturer

    from relax.utils.logging_utils import get_logger

    cls = RoutedExpertsCapturer
    if getattr(cls, "_relax_replay_owner_capture", False):
        return
    original = cls.on_forward_end

    @wraps(original)
    def on_forward_end(self, *args: Any, **kwargs: Any) -> Any:
        export = owner()
        if not getattr(self, "_relax_replay_snapshot_role_logged", False):
            import os

            get_logger(__name__).info(
                "MoE reference snapshot: pid=%s export=%s; capture collectives remain enabled", os.getpid(), export
            )
            self._relax_replay_snapshot_role_logged = True
        if not export:
            return None
        return original(self, *args, **kwargs)

    cls.on_forward_end = on_forward_end
    cls._relax_replay_owner_capture = True


def _install_response_patch(owner: Callable) -> bool:
    """Send completed references from the scheduler thread, never the
    writer."""
    from sglang.srt.managers.scheduler import Scheduler

    if getattr(Scheduler, "_relax_replay_async_output", False):
        return True
    if hasattr(Scheduler, "stream_output"):
        original_output = Scheduler.stream_output
    else:
        original_streamer_class = Scheduler.get_output_streamer_class

        def original_output(self, reqs, return_logprob, skip_req=None):
            return original_streamer_class(self).stream_output(self.output_streamer, reqs, return_logprob, skip_req)

        def replay_streamer_class(scheduler):
            base = original_streamer_class(scheduler)

            class ReplayOutputStreamer(base):
                def stream_output(self, reqs, return_logprob, skip_req=None):
                    return output(scheduler, reqs, return_logprob, skip_req)

            return ReplayOutputStreamer

        Scheduler.get_output_streamer_class = replay_streamer_class
    original_input = Scheduler.process_input_requests
    original_idle = Scheduler.maybe_sleep_on_idle

    def drain(self, *, wait: bool = False):
        pending = getattr(self, "_relax_replay_pending", {})
        ready = {False: [], True: []}
        for key, (req, return_logprob) in tuple(pending.items()):
            if getattr(req, "finished_output", False):
                pending.pop(key)
                continue
            ordinary = getattr(req, "_relax_replay_finish", None) or {}
            media = getattr(req, "_relax_replay_media", None)
            try:
                if _request_aborted(req) and not _keep_aborted_response(req):
                    _abort_request_publication(req)
                    ordinary, media = {}, None
                else:
                    # Check every producer before releasing either kind's hold.
                    futures = list(ordinary.values())
                    if media is not None:
                        futures.extend(media.response_futures(req.rid))
                    if not wait and not all(future.done() for future in futures):
                        continue
                    for future in futures:
                        future.result()
                wait = False  # Backpressure waits for at most one request.
                ready[bool(return_logprob)].append((key, req, ordinary, media))
            except BaseException as primary:
                pending.pop(key)
                try:
                    _abort_request_publication(req)
                except Exception as cleanup:
                    primary.add_note(f"Replay publication cleanup failed: {cleanup!r}")
                raise

        for return_logprob, batch in ready.items():
            if not batch:
                continue
            for key, req, ordinary, media in batch:
                for kind, future in ordinary.items():
                    setattr(req, _CAPTURE_FIELDS[kind], future.result().to_dict())
                if media is not None:
                    req._relax_mm_output = media.finish(req.rid)
                pending.pop(key)
            # A failed native send has uncertain delivery. Preserve the bodies
            # for receiver adoption or teardown of this run.
            original_output(self, [req for _, req, _, _ in batch], return_logprob)
            for _, req, ordinary, _ in batch:
                if ordinary:
                    req._relax_replay_streams = {}
                    req._relax_replay_finish = None
                    req._relax_replay_published = True

    @wraps(original_output)
    def output(self, reqs, return_logprob, skip_req=None):
        if not owner(self):
            return original_output(self, reqs, return_logprob, skip_req)
        pending = getattr(self, "_relax_replay_pending", None)
        if pending is None:
            pending = self._relax_replay_pending = {}
        immediate = []
        for req in reqs:
            key = id(req)
            if getattr(req, "finished_output", False):
                pending.pop(key, None)
                continue
            if req is skip_req:
                immediate.append(req)
            elif _request_aborted(req) and not _keep_aborted_response(req):
                pending.pop(key, None)
                _abort_request_publication(req)
                immediate.append(req)
            elif getattr(req, "_relax_replay_finish", None) is not None or (
                getattr(req, "_relax_replay_media", None) is not None and req.finished()
            ):
                if key not in pending:
                    if len(pending) >= self.max_running_requests:
                        drain(self, wait=True)
                    pending[key] = (req, return_logprob)
            else:
                immediate.append(req)
        if immediate:
            original_output(self, immediate, return_logprob, skip_req)
        drain(self)

    @wraps(original_input)
    def process_input(self, *args, **kwargs):
        # This runs even for empty input / recv_skipper iterations. Process
        # cancellations before releasing any newly completed response.
        result = original_input(self, *args, **kwargs)
        drain(self)
        return result

    @wraps(original_idle)
    def idle(self, *args, **kwargs):
        if getattr(self, "_relax_replay_pending", None):
            # The fixed SGLang idle poll otherwise waits up to one second for
            # external input; writer completion does not wake that socket.
            import time

            time.sleep(0.001)
            return None
        return original_idle(self, *args, **kwargs)

    if hasattr(Scheduler, "flush_cache"):
        original_flush = Scheduler.flush_cache

        @wraps(original_flush)
        def flush(self, *args, **kwargs):
            # Pending terminal requests are outside the native scheduler queues.
            if getattr(self, "_relax_replay_pending", None):
                return False
            return original_flush(self, *args, **kwargs)

        Scheduler.flush_cache = flush

    if hasattr(Scheduler, "stream_output"):
        Scheduler.stream_output = output
    Scheduler.process_input_requests = process_input
    Scheduler.maybe_sleep_on_idle = idle
    Scheduler._relax_replay_async_output = True
    return True


def _wire_ref(value: dict) -> dict[str, str]:
    ref = ReplayRef.from_dict(value)
    return {"replay_manifest": pybase64.b64encode(encode_refs({ref.kind: value})).decode("ascii")}


def _encode_responses(values: Any, encode: Any) -> Any:
    """Keep reference metadata and encode native payloads in their original
    order."""
    if values is None or not any(isinstance(value, dict) for value in values):
        return encode(values)
    for value in values:
        if isinstance(value, dict):
            ReplayRef.from_dict(value)
    native = [value for value in values if not isinstance(value, dict)]
    if not native:
        return [_wire_ref(value) for value in values]
    encoded = iter(encode(native))
    return [_wire_ref(value) if isinstance(value, dict) else next(encoded) for value in values]


def install_detokenizer_patch() -> None:
    from sglang.srt.managers.detokenizer_manager import DetokenizerManager

    cls = DetokenizerManager
    if getattr(cls, "_relax_replay_references", False):
        return
    if hasattr(cls, "_b64_encode_per_request"):
        original = cls._b64_encode_per_request

        def encode(values):
            return _encode_responses(values, original)

        cls._b64_encode_per_request = staticmethod(encode)
    else:

        def wrap(original, field):
            def extract(self, recv_obj):
                values = getattr(recv_obj, field)

                def encode(native):
                    response = copy(recv_obj)
                    setattr(response, field, native)
                    return original(self, response)

                return _encode_responses(values, encode)

            return extract

        for field in _CAPTURE_FIELDS.values():
            name = "_extract_" + field
            if hasattr(cls, name):
                setattr(cls, name, wrap(getattr(cls, name), field))
    cls._relax_replay_references = True


def _install_tq_topology_cache() -> None:
    from .._tq_compat import install_tq_topology_cache

    install_tq_topology_cache()


def configure_engine_publication(args: Any, server_args: dict, rank: int, *, external: bool) -> bool:
    from ..config import rollout_kinds

    kinds = rollout_kinds(args)
    if not kinds:
        return False
    local = not external and server_args.get("disaggregation_mode") in (None, "null")
    validate_configuration(args, server_args, local_publication=local)
    if not local:
        from relax.utils.logging_utils import get_logger

        get_logger(__name__).info(
            "SGLang retains its native HTTP payload; Replay publishes references at rollout ingress"
        )
        return False

    import ray

    # Spawned scheduler processes construct their own clients from this registration.
    replay_data_manager.initialize(kinds=kinds, connection=args.tq_config)
    context = ray.get_runtime_context()
    os.environ["RELAX_REPLAY_REGISTRATION"] = json.dumps(replay_data_manager.export_registration())
    os.environ["RELAX_REPLAY_RAY_ADDRESS"] = context.gcs_address
    publisher = (rank, context.get_actor_id())
    replay_data_manager.store().invoke("register_engine_publisher", *publisher)
    os.environ["RELAX_REPLAY_ENGINE_PUBLISHER"] = json.dumps(publisher)
    return True


def run_reference_scheduler(*args, **kwargs):
    from relax.utils.env import Envs

    if Envs.RELAX_OPTIMIZE_ROUTING_REPLAY:
        from relax.backends.sglang.routing_replay_patch import apply_patch

        apply_patch()
    import ray

    if not ray.is_initialized():
        if not Envs.RELAX_REPLAY_RAY_ADDRESS:
            raise RuntimeError("Scheduler needs its parent replay Ray address; refusing automatic cluster discovery")
        ray.init(address=Envs.RELAX_REPLAY_RAY_ADDRESS, namespace="relax-replay", log_to_driver=False)
    registration = os.environ.get("RELAX_REPLAY_REGISTRATION")
    if registration is None:
        raise RuntimeError("Scheduler requires its parent's replay registration")
    if "RELAX_REPLAY_ENGINE_PUBLISHER" not in os.environ:
        raise RuntimeError("Scheduler requires its replay publisher generation")
    definitions = json.loads(registration)
    _install_tq_topology_cache()
    replay_data_manager.initialize_worker(definitions, engine="sglang")
    from sglang.srt.managers.scheduler import run_scheduler_process

    return run_scheduler_process(*args, **kwargs)


def run_reference_detokenizer(*args, **kwargs):
    install_detokenizer_patch()
    from sglang.srt.managers.detokenizer_manager import run_detokenizer_process

    return run_detokenizer_process(*args, **kwargs)


def validate_configuration(args: Any, server_args: dict, *, local_publication: bool = True) -> None:
    from ..config import rollout_kinds

    kinds = rollout_kinds(args)
    if not local_publication:
        if "mm_embedding" in kinds:
            raise ValueError("MM embedding replay requires local SGLang capture without Prefill/Decode disaggregation")
        return
    if server_args.get("pp_size", 1) > 1:
        raise RuntimeError(
            "SGLang reference publication needs a PP stage-aggregation adapter; "
            "the current fixed-version hook cannot assume PP0 owns every layer"
        )
    # Algorithm compatibility stays in Relax's existing replay-flag validation.
    # This adapter checks only the installed publication API, never training modes.
    _validate_capture_api(kinds)


def apply_sglang_patch(manager: ReplayManager | None = None, *, kinds: frozenset[str] | None = None) -> None:
    """Install the supported scheduler hooks once in the scheduler process."""
    selected = replay_data_manager if manager is None else manager
    if selected.uses_references:
        ordinary = None if kinds is None else kinds - {"mm_embedding"}
        if ordinary is None or ordinary:
            install_scheduler_patch(selected, kinds=ordinary)
        if kinds is not None and "mm_embedding" in kinds:
            from .mm_sglang import install_capture_patch

            install_capture_patch(selected)
