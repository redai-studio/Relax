# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""BF16 media bodies with image-hash deduplication and independent response
holds."""

from collections.abc import Callable, Sequence
from time import monotonic, sleep
from typing import Any

from .codecs import BF16_RAW
from .ref_data import ReplayRef, validate_mm_envelope


def _geometry(image_hash: str, grid: Sequence[int], modality: str) -> list[int]:
    result = list(grid)
    if (
        not isinstance(image_hash, str)
        or len(image_hash) != 64
        or any(c not in "0123456789abcdef" for c in image_hash)
        or modality not in ("image", "video")
        or len(result) != 3
        or any(type(n) is not int or n <= 0 for n in result)
        or result[1] % 2
        or result[2] % 2
    ):
        raise ValueError("Invalid media hash, modality or merged-token grid")
    return result


class MMPublisher:
    """Publish once per image hash; every lookup acquires a distinct hold.

    Methods run on the caller's background publication worker. They never
    initialize CUDA or copy tensors from the model's execution stream.
    """

    def __init__(self, store: Any, *, pending_timeout: float = 60.0) -> None:
        if pending_timeout <= 0:
            raise ValueError("Media publication timeout must be positive")
        self.store = store
        self.pending_timeout = pending_timeout

    @staticmethod
    def _entry(image_hash: str, grid: list[int], modality: str, result: dict) -> dict:
        entry = {
            "hash": image_hash,
            "modality": modality,
            "grid": grid,
            "ref": result["ref"],
            "lease": result["lease"],
        }
        validate_mm_envelope({"version": 1, "media": [entry]})
        return entry

    def _acquire(self, method: str, *args: Any) -> dict | None:
        from .core import new_data_id

        # Allocate at the caller so an acknowledgement lost after registration
        # can roll back the exact hold, as in TQReplayStorage.acquire_many.
        lease = new_data_id()
        try:
            return self.store.invoke(method, *args, lease=lease)
        except BaseException as error:
            try:
                self.store.invoke("rollback_acquire", [lease])
            except Exception as cleanup:
                self.store._pending_releases.append([lease])
                error.add_note(f"Media lease rollback pending: {cleanup!r}")
            if method == "claim_media":
                try:
                    self.store.invoke("abort_media_claim", args[0], args[2])
                except Exception as cleanup:
                    error.add_note(f"Media claim cleanup failed: {cleanup!r}")
            raise

    def lookup(self, image_hash: str, grid: Sequence[int], modality: str = "image") -> dict | None:
        grid = _geometry(image_hash, grid, modality)
        result = self._acquire("lookup_media", image_hash, grid, modality)
        return None if result is None else self._entry(image_hash, grid, modality, result)

    def publish(
        self,
        image_hash: str,
        bits: Any | Callable[[], Any],
        grid: Sequence[int],
        modality: str = "image",
    ) -> dict:
        return self._publish(image_hash, bits, grid, modality, chunked=False)

    def publish_chunks(
        self,
        image_hash: str,
        snapshot_rows: Callable[[int, int], Any],
        grid: Sequence[int],
        modality: str = "image",
    ) -> dict:
        """Snapshot bounded row ranges only after publication byte
        admission."""
        return self._publish(image_hash, snapshot_rows, grid, modality, chunked=True)

    def _publish(self, image_hash: str, bits: Any, grid: Sequence[int], modality: str, *, chunked: bool) -> dict:
        import numpy as np

        from .core import new_data_id

        grid = _geometry(image_hash, grid, modality)
        record = new_data_id()
        deadline = monotonic() + self.pending_timeout
        while True:
            claim = self._acquire("claim_media", image_hash, grid, record, modality)
            if claim["status"] == "ready":
                return self._entry(image_hash, grid, modality, claim)
            if claim["status"] == "claimed":
                break
            if monotonic() >= deadline:
                raise TimeoutError("Another producer did not complete its media publication")
            sleep(0.001)
        first_lease = claim["lease"]
        publication = None
        try:
            rows = grid[0] * grid[1] * grid[2] // 4
            width = BF16_RAW.topk * BF16_RAW.element_bytes
            values = None if chunked else (bits() if callable(bits) else bits)
            if not chunked and (
                not isinstance(values, np.ndarray) or values.dtype != np.uint8 or values.shape != (rows, width)
            ):
                raise ValueError("Media snapshot must contain BF16 raw bytes [visual tokens, 5120]")
            publication = self.store.begin_publication(
                kind="mm_embedding", layers=1, topk=BF16_RAW.topk, codec=BF16_RAW.name, record=record
            )
            for start in range(0, rows, publication.chunk_rows):
                stop = min(start + publication.chunk_rows, rows)

                def snapshot(layer: int, start: int = start, stop: int = stop) -> Any:
                    data = bits(start, stop) if chunked else values[start:stop]
                    if (
                        not isinstance(data, np.ndarray)
                        or data.dtype != np.uint8
                        or data.shape != (stop - start, width)
                    ):
                        raise ValueError("Media chunk differs from its BF16 raw byte shape")
                    # The caller may reuse its pinned snapshot buffer for the
                    # next chunk while TQ still writes this independent array.
                    return data.copy()

                publication.append(stop - start, snapshot)
            ref = publication.finish_media(image_hash)
            return self._entry(image_hash, grid, modality, {"ref": ref.to_dict(), "lease": first_lease})
        except BaseException as error:
            try:
                self.store.invoke("rollback_acquire", [first_lease])
            except Exception as cleanup:
                self.store._pending_releases.append([first_lease])
                error.add_note(f"Media initial lease cleanup pending: {cleanup!r}")
            if publication is not None:
                try:
                    publication.abort()
                except Exception as cleanup:
                    error.add_note(f"Media publication cleanup failed: {cleanup!r}")
            try:
                self.store.invoke("abort_media_claim", image_hash, record)
            except Exception as cleanup:
                error.add_note(f"Media claim cleanup failed: {cleanup!r}")
            raise


def release_media_entries(store: Any, entries: Sequence[dict]) -> None:
    """Release response/cache holds idempotently, without touching queue
    holds."""
    if not entries:
        return
    validate_mm_envelope({"version": 1, "media": list(entries)})
    store.invoke(
        "release_media", [ReplayRef.from_dict(item["ref"]) for item in entries], [item["lease"] for item in entries]
    )


def read_mm_embedding(store: Any, entry: dict, *, device: Any = "cpu", lease: str | None = None) -> Any:
    """Read a whole media under an active lease, preserving every BF16 bit."""
    import numpy as np
    import torch

    validate_mm_envelope({"version": 1, "media": [entry]})
    ref = ReplayRef.from_dict(entry["ref"])
    store.invoke("authorize_read", ref, entry["lease"] if lease is None else lease)
    device = torch.device(device)
    width = ref.topk * 2
    output = torch.empty((ref.rows, width), dtype=torch.uint8, device=device)
    pending, pending_bytes = [], 0
    limit = store.encoded_batch_limit(device=device)

    def flush() -> None:
        bodies = store.read_encoded([key for key, _, _ in pending], device=device)
        if len(bodies) != len(pending):
            raise RuntimeError("TQ is missing a media replay chunk")
        for (_, start, count), body in zip(pending, bodies, strict=True):
            if isinstance(body, np.ndarray):
                if body.dtype != np.uint8 or body.shape != (count, width):
                    raise RuntimeError("Media replay body differs from its BF16 byte shape")
                tensor = torch.from_numpy(body)
            elif isinstance(body, torch.Tensor) and body.dtype == torch.uint8 and body.shape == (count, width):
                tensor = body
            else:
                raise RuntimeError("Media replay body differs from its BF16 byte shape")
            if tensor.device.type == "cpu" and device.type == "cuda":
                pinned = torch.empty(tensor.shape, dtype=torch.uint8, pin_memory=True)
                pinned.copy_(tensor)
                tensor = pinned
            output[start : start + count].copy_(tensor, non_blocking=True)
        pending.clear()

    for start in range(0, ref.rows, ref.chunk_rows):
        count = min(ref.chunk_rows, ref.rows - start)
        size = count * width
        if pending and pending_bytes + size > limit:
            flush()
            pending_bytes = 0
        pending.append((f"{ref.record}:{start // ref.chunk_rows}:0", start, count))
        pending_bytes += size
    if pending:
        flush()
    return output.view(torch.bfloat16)
