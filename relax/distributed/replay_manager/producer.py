# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Engine-independent publication, admission and immutable reference
handoff."""

import asyncio
from collections.abc import Iterable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from threading import Condition
from typing import TYPE_CHECKING, Any, Callable, Sequence

from .codecs import replay_codec
from .config import rollout_kinds
from .ref_data import FIELDS, REFERENCE_FIELD, ReplayRef, publisher_holds, token_digest


if TYPE_CHECKING:
    from .core import ReplayManager


def prepare_replay_export(sample: Any, args: Any) -> None:
    """Apply the training input policy to a selected export in canonical sample
    format."""
    envelope = (sample.rollout_replay_refs or {}).get("mm_embedding")
    if envelope is not None:
        from .mm_embedding import prepare_mm_sample, training_needs_pixels

        prepare_mm_sample(sample, envelope, keep_pixels=training_needs_pixels(args))


@dataclass
class PublisherHoldings:
    """Published route records and per-occurrence media leases held by one
    owner."""

    refs: dict[str, ReplayRef] = field(default_factory=dict)
    media: dict[str, dict] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return bool(self.refs or self.media)

    def transfer_to(self, target: "PublisherHoldings", manifests: Iterable[dict]) -> None:
        refs, media = publisher_holds(manifests)
        for ref in refs:
            target.refs[ref.record] = ref
            self.refs.pop(ref.record, None)
        for entry in media:
            target.media[entry["lease"]] = entry
            self.media.pop(entry["lease"], None)

    async def discard(self) -> None:
        if self:
            await self._release(list(self.refs.values()), list(self.media.values()))

    async def _release(self, refs: list[ReplayRef], media: list[dict]) -> None:
        from . import replay_data_manager

        await replay_data_manager.store().discard_async(refs, media=media)
        for ref in refs:
            self.refs.pop(ref.record, None)
        for entry in media:
            self.media.pop(entry["lease"], None)

    async def adopt_response(
        self, args: Any, session_id: str, meta: dict, is_current: Callable[[], bool], *, cleanup_timeout: float
    ) -> bool:
        from . import replay_data_manager as manager

        kinds = rollout_kinds(args)
        if not manager.uses_references or not kinds:
            return True
        producer = manager.register_producer(kinds)
        with producer.response_holds(meta) as (refs, media):
            if not is_current():
                await manager.store().invoke_async("discard", refs, unadopted=True, media=media)
                return False
            # Media holds are distinct even for repeated occurrences of one record.
            self.media.update((entry["lease"], entry) for entry in media)
            if refs:
                await producer.adopt_response(refs, session_id)
                self.refs.update((ref.record, ref) for ref in refs)
            if not is_current():
                await asyncio.wait_for(self._release(refs, media), cleanup_timeout)
                return False
        return True


@dataclass
class PendingSampleWrite:
    """Track whether TQ may already own the samples when a transfer fails."""

    manager: "ReplayManager"
    samples: Sequence[Any]
    started: bool = False

    def mark_started(self) -> None:
        self.started = True

    async def discard_if_unsent(self, error: BaseException, *, native: bool = False) -> None:
        if self.started:
            return
        try:
            await self.manager.discard_async(self.samples, unsent=True, native=native)
        except Exception as cleanup:
            error.add_note(f"Replay cleanup before TQ dispatch failed: {cleanup!r}")


@dataclass(frozen=True)
class Producer:
    manager: "ReplayManager"
    kinds: frozenset[str]

    def begin(
        self,
        *,
        kind: str,
        layers: int,
        topk: int,
        codec: str = "int32",
        engine_publisher: tuple[int, str] | None = None,
    ) -> "ProducerPublication":
        if kind not in self.kinds:
            raise ValueError("Replay kind is not registered with this producer")
        return self.manager.store().begin_publication(
            kind=kind, layers=layers, topk=topk, codec=codec, engine_publisher=engine_publisher
        )

    def _publish_response(self, values: Any, *, kind: str, tokens: list[int], codec: str = "int32") -> dict:
        import numpy as np

        publication = self.begin(kind=kind, layers=values.shape[1], topk=values.shape[2], codec=codec)
        try:
            for start in range(0, len(values), publication.chunk_rows):
                stop = min(start + publication.chunk_rows, len(values))
                publication.append(
                    stop - start, lambda layer: np.array(values[start:stop, layer], copy=True, order="C")
                )
            return publication.finish(tokens).to_dict()
        except BaseException as error:
            try:
                publication.abort()
            except Exception as cleanup:
                error.add_note(f"Replay ingress cleanup failed: {cleanup!r}")
            raise

    def build_tq_fields(self, samples: Sequence[Any]) -> dict[str, Any]:
        self.manager._ensure_open()
        if not samples or (self.manager.uses_references and not self.kinds):
            return {}
        result = self.manager.build_tq_fields(samples, required_fields=frozenset(FIELDS[kind] for kind in self.kinds))
        if self.manager.uses_references:
            for refs in result[REFERENCE_FIELD]:
                if not self.kinds.issubset(refs):
                    raise ValueError("Registered producer is missing replay kinds")
        return result


class ProducerChunkWriter:
    """One bounded queue per store; reserve bytes before taking a slot
    snapshot.

    Workers receive independent CPU arrays, never a KV-slot view. Backpressure
    blocks the scheduler only when the in-flight byte budget is exhausted. The
    TQ writers may use another capacity_bytes for packing. Each worker has its
    own transport client, while all share the snapshot byte budget. This budget
    does not include TQ serialization or stored bodies.
    """

    def __init__(self, capacity_bytes: int, *, max_workers: int = 1) -> None:
        if capacity_bytes <= 0:
            raise ValueError("Publication byte budget must be positive")
        self.capacity_bytes = capacity_bytes
        self.pending_bytes = self.peak_bytes = 0
        self._error = None
        self._closed = False
        self._condition = Condition()
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="replay-publish")
        # Waiting for write acknowledgements must not consume body workers.
        self._sealer = ThreadPoolExecutor(max_workers=1, thread_name_prefix="replay-seal")

    def submit(self, size: int, snapshot: Callable[[], Any], write_block: Callable[[Any], None]):
        if size > self.capacity_bytes:
            raise BufferError("Replay layer chunk exceeds publication budget")
        with self._condition:
            self._condition.wait_for(
                lambda: self._closed or self._error is not None or self.pending_bytes + size <= self.capacity_bytes
            )
            if self._closed:
                raise RuntimeError("Replay writer is closing")
            if self._error is not None:
                raise self._error
            self.pending_bytes += size
            self.peak_bytes = max(self.peak_bytes, self.pending_bytes)

        def done() -> None:
            with self._condition:
                self.pending_bytes -= size
                self._condition.notify_all()

        try:
            values = snapshot()  # scheduler thread: copy before KV slots can be reused
            arrays = values if isinstance(values, tuple) else (values,)
            actual = sum(value.nbytes for value in arrays)
            if actual != size or any(not value.flags.c_contiguous or not value.flags.owndata for value in arrays):
                raise ValueError("Publication requires an owned contiguous CPU chunk of the reserved size")

            def write():
                try:
                    write_block(values)
                except BaseException as exc:
                    with self._condition:
                        # Keep a lightweight sticky failure, not its traceback
                        # retaining the failed chunk. The Future keeps the
                        # original exception for the request's finish/abort.
                        self._error = RuntimeError(f"Replay writer failed: {type(exc).__name__}: {exc}")
                        self._condition.notify_all()
                    raise
                finally:
                    done()

            with self._condition:
                if self._closed:
                    raise RuntimeError("Replay writer is closing")
                return self._executor.submit(write)
        except BaseException:
            done()
            raise

    def submit_completion(self, complete: Callable[[], Any]) -> Future:
        """Enqueue a callback that seals after its publication's writes."""
        with self._condition:
            if self._closed:
                raise RuntimeError("Replay writer is closing")
            return self._sealer.submit(complete)

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()
        self._executor.shutdown(wait=True)
        self._sealer.shutdown(wait=True)


class ProducerPublication:
    """One request/kind's write-once body, invisible to TQ until sealed.

    Full stable chunks can be submitted during generation. Only the last chunk
    may be short. Retraction/abort drains the writer before removing this body.
    """

    def __init__(
        self,
        store: Any,
        writer: ProducerChunkWriter,
        *,
        kind: str,
        layers: int,
        topk: int,
        codec: str = "int32",
        record: str | None = None,
        media_hash: str | None = None,
        engine_publisher: tuple[int, str] | None = None,
    ):
        from .core import new_data_id

        if kind not in FIELDS or min(layers, topk) <= 0:
            raise ValueError("Invalid streaming replay shape/kind")
        replay_codec(codec).validate(kind, topk)
        if kind == "mm_embedding" and (codec != "bf16-raw-v1" or layers != 1):
            raise ValueError("Media replay requires a single BF16 embedding tensor")
        self.store, self.writer = store, writer
        self.kind, self.layers, self.topk = kind, layers, topk
        self.codec = codec
        self.chunk_rows = min(store.chunk_rows, writer.capacity_bytes // (topk * replay_codec(codec).element_bytes))
        if not self.chunk_rows:
            raise BufferError("One replay row exceeds publication budget")
        self.record = new_data_id() if record is None else record
        self.rows = 0
        self._futures = []
        self._sealed = self._aborted = False
        self._finish_future = None
        store.invoke(
            "begin",
            self.record,
            kind,
            layers,
            topk,
            self.chunk_rows,
            codec,
            media_hash=media_hash,
            engine_publisher=engine_publisher,
        )

    def append(self, count: int, snapshot_layer: Callable[[int], Any], *, encoded_width: int | None = None) -> None:
        import numpy as np

        if (
            self._sealed
            or self._aborted
            or self._finish_future is not None
            or self.rows % self.chunk_rows
            or not 0 < count <= self.chunk_rows
        ):
            raise ValueError("Replay publication is closed or chunk is not appendable")
        pending = []
        for future in self._futures:
            if future.done():
                future.result()
            else:
                pending.append(future)
        self._futures = pending
        byte_payload = self.codec in {"bf16-raw-v1", "uint8"}
        width = (
            self.topk * replay_codec(self.codec).element_bytes
            if byte_payload
            else (self.topk if self.codec == "int32" else encoded_width)
        )
        if not byte_payload and self.codec != "int32":
            replay_codec(self.codec).validate_width(width)
        size = count * width * (1 if byte_payload else 4)

        def snapshot(layer: int) -> Any:
            values = snapshot_layer(layer)
            if self.codec == "uint8":
                if (
                    not isinstance(values, np.ndarray)
                    or values.dtype not in (np.uint8, np.int32, np.int64)
                    or values.shape != (count, self.topk)
                ):
                    raise ValueError("Uint8 MoE replay requires an integer routing snapshot")
                # These are bounded, completed CPU snapshots. Never narrow GPU
                # tensors or silently wrap negative / >255 expert indices.
                if values.min() < 0 or values.max() > 255:
                    raise ValueError("Uint8 MoE replay expert indices must be in [0, 255]")
                return values.astype(np.uint8, order="C", copy=True)
            dtype = np.uint8 if byte_payload else np.int32
            if not isinstance(values, np.ndarray) or values.dtype != dtype or values.shape != (count, width):
                raise ValueError("Replay layer snapshot differs from its codec shape/dtype")
            return values

        # Amortize RPCs across layers while preserving one key per layer.
        # Admission still happens before snapshots; packing runs on the writer.
        batch_budget = (
            min(self.writer.capacity_bytes, 1 << 20)
            if self.codec in {"int32", "uint8"}
            else self.writer.capacity_bytes
        )
        layers_per_batch = max(1, batch_budget // size)
        for first in range(0, self.layers, layers_per_batch):
            layers = tuple(range(first, min(first + layers_per_batch, self.layers)))
            self._futures.append(
                self.writer.submit(
                    size * len(layers),
                    lambda layers=layers: tuple(snapshot(layer) for layer in layers),
                    lambda values, layers=layers, chunk=self.rows // self.chunk_rows: self.store.write_blocks(
                        self.record, chunk, layers, values
                    ),
                )
            )
        self.rows += count

    def finish(self, tokens: list[int]) -> ReplayRef:
        if self._sealed or self._aborted:
            raise ValueError("Replay publication is closed")
        return self.finish_async(tokens).result()

    def finish_async(self, tokens: list[int], *, attempt: str | None = None) -> Future:
        """Seal on the writer; callers publish the reference only once
        ready."""
        if self._aborted:
            raise ValueError("Replay publication was aborted")
        if self._finish_future is not None:
            if token_digest(tokens) != self._finish_digest:
                raise ValueError("Replay finalization token identity changed")
            return self._finish_future
        if self._sealed or self.rows != len(tokens) - 1:
            raise ValueError("Replay publication does not cover the finalized request")
        return self._finish_with_digest(token_digest(tokens), attempt)

    def finish_media(self, image_hash: str) -> ReplayRef:
        """Seal media coordinates with their hash, not generated token
        identity."""
        if self.kind != "mm_embedding" or self.codec != "bf16-raw-v1" or self._aborted or self._sealed:
            raise ValueError("Invalid media replay finalization")
        if len(image_hash) != 64 or any(c not in "0123456789abcdef" for c in image_hash) or self.rows <= 0:
            raise ValueError("Invalid media hash or empty publication")
        return self._finish_with_digest(image_hash).result()

    def _finish_with_digest(self, digest: str, attempt: str | None = None) -> Future:
        self._finish_digest = digest
        writes = tuple(self._futures)

        def complete():
            # Writes were enqueued first, but may still run on another worker.
            for future in writes:
                future.result()
            ref = self.store.invoke("finish_publication", self.record, self.rows, digest, attempt)
            self._futures.clear()
            self._sealed = True
            return ref

        self._finish_future = self.writer.submit_completion(complete)
        return self._finish_future

    def abort(self) -> None:
        if self._aborted:
            return
        # Drain every worker before deletion: no late writer may recreate deleted body keys.
        for future in self._futures:
            try:
                future.result()
            except Exception:
                pass  # The primary failure propagates through submit/finish.
        if self._finish_future is not None:
            try:
                self._finish_future.result()
            except Exception:
                pass  # Drain sealing before deleting any of its bodies.
        self.store.invoke("abort", self.record)
        self._futures.clear()
        self._aborted = True
