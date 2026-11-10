# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Engine-independent publication, admission and immutable reference
handoff."""

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from threading import Condition
from typing import TYPE_CHECKING, Any, Callable, Sequence

from .ref_data import FIELDS, REFERENCE_FIELD, ReplayRef, token_digest


if TYPE_CHECKING:
    from .core import ReplayManager


@dataclass(frozen=True)
class Producer:
    manager: "ReplayManager"
    kinds: frozenset[str]

    def begin(self, *, kind: str, layers: int, topk: int) -> "ProducerPublication":
        if kind not in self.kinds:
            raise ValueError("Replay kind is not registered with this producer")
        return self.manager.store().begin_publication(kind=kind, layers=layers, topk=topk)

    def _publish_response(self, values: Any, *, kind: str, tokens: list[int]) -> dict:
        import numpy as np

        publication = self.begin(kind=kind, layers=values.shape[1], topk=values.shape[2])
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
    TQ writer may additionally pack up to 1 MiB; single-layer writes use a
    view. This budget does not include TQ serialization or stored bodies.
    """

    def __init__(self, capacity_bytes: int) -> None:
        if capacity_bytes <= 0:
            raise ValueError("Publication byte budget must be positive")
        self.capacity_bytes = capacity_bytes
        self.pending_bytes = self.peak_bytes = 0
        self._error = None
        self._closed = False
        self._condition = Condition()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="replay-publish")

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
            if sum(value.nbytes for value in arrays) != size or any(
                not value.flags.c_contiguous or not value.flags.owndata for value in arrays
            ):
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
        """Run publication sealing after already queued body writes."""
        with self._condition:
            if self._closed:
                raise RuntimeError("Replay writer is closing")
            return self._executor.submit(complete)

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()
        self._executor.shutdown(wait=True)


class ProducerPublication:
    """One request/kind's write-once body, invisible to TQ until sealed.

    Full stable chunks can be submitted during generation. Only the last chunk
    may be short. Retraction/abort drains the writer before removing this body.
    """

    def __init__(self, store: Any, writer: ProducerChunkWriter, *, kind: str, layers: int, topk: int):
        from .core import new_data_id

        if kind not in FIELDS or min(layers, topk) <= 0:
            raise ValueError("Invalid streaming replay shape/kind")
        self.store, self.writer = store, writer
        self.kind, self.layers, self.topk = kind, layers, topk
        self.chunk_rows = min(store.chunk_rows, writer.capacity_bytes // (topk * 4))
        if not self.chunk_rows:
            raise BufferError("One replay row exceeds publication budget")
        self.record = new_data_id()
        self.rows = 0
        self._futures = []
        self._sealed = self._aborted = False
        self._finish_future = None
        store.invoke("begin", self.record, kind, layers, topk, self.chunk_rows)

    def append(self, count: int, snapshot_layer: Callable[[int], Any]) -> None:
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
        size = count * self.topk * 4

        def snapshot(layer: int) -> Any:
            values = snapshot_layer(layer)
            if not isinstance(values, np.ndarray) or values.dtype != np.int32 or values.shape != (count, self.topk):
                raise ValueError("Replay layer snapshot must be int32 [chunk rows, topk]")
            return values

        # Amortize RPCs across layers while preserving one key per layer.
        # Admission still happens before snapshots; packing runs on the writer.
        layers_per_batch = max(1, min(self.writer.capacity_bytes, 1 << 20) // size)
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

    def finish_async(self, tokens: list[int]) -> Future:
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
        digest = self._finish_digest = token_digest(tokens)
        writes = tuple(self._futures)

        def complete():
            # These writes were enqueued before this task on the same worker.
            for future in writes:
                future.result()
            ref = self.store.invoke("finish_publication", self.record, self.rows, digest)
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
