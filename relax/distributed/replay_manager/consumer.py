# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Consumer tasks, replay sources, reader leases and bounded
materialization."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import OrderedDict
from contextlib import nullcontext
from dataclasses import dataclass
from threading import RLock
from typing import TYPE_CHECKING, Any, Literal, Sequence

from ._encoded_prefetch import _EncodedReadAhead
from ._prefetch import _ConsumerReadAhead
from .codecs import QSA_BITMAP, _QSABitmapReplay, replay_codec
from .ref_data import FIELDS, REFERENCE_FIELD, ReplayRef, validate_sample


if TYPE_CHECKING:
    import numpy as np
    import torch
    from numpy.typing import NDArray

    from .core import ReplayManager, ReplayRegion


@dataclass(frozen=True)
class ConsumerChannelKey:
    """One replay kind and layer; coordinates preserve the backend's layer
    numbering."""

    kind: str
    layer: int
    coordinate: Literal["global_layer", "local_registration"] = "global_layer"


@dataclass(frozen=True)
class ConsumerRecord:
    """Bind one layer/microbatch to a tensor, (tensor, mask), or lazy reader.

    The payload keeps its existing storage; this record adds no tensor copy.
    """

    kind: str
    microbatch_id: int
    payload: Any


class ConsumerSource(ABC):
    """A readable lease bound to one layer/microbatch's final layout.

    read must remain repeatable through forward and recomputation. release
    drops this consumer's lease, never unilaterally deletes shared data. It
    must be idempotent so a failed release can be retried safely. A reference
    implementation must provide both operations; the manager does not guess how
    a locator is resolved or when other ranks finish.
    """

    @abstractmethod
    def read(self) -> Any: ...

    @abstractmethod
    def release(self) -> None: ...


@dataclass(frozen=True)
class ConsumerSampleRowMap:
    """Map one sample's captured rows to this rank's microbatch rows."""

    sample_index: int
    output_rows: "NDArray[np.int64]"
    source_rows: "NDArray[np.int64]"


@dataclass(frozen=True)
class ConsumerBatchLayout:
    """Selected microbatch's final row positions, shared by all local layers.

    This describes existing packing/padding and rank slices, not TQ sampling.
    """

    size: int
    samples: tuple[ConsumerSampleRowMap, ...]
    valid: "NDArray[np.bool_]"
    padding_offsets: "NDArray[np.int64]"

    def __len__(self) -> int:
        return self.size

    @classmethod
    def from_coordinates(cls, coordinates: "NDArray[np.int64]") -> "ConsumerBatchLayout":
        import numpy as np

        if coordinates.ndim != 2 or coordinates.shape[1] != 2 or np.any(coordinates[:, 0] < -1):
            raise ValueError("Replay coordinates must be [sample index, source row], with -1 for padding")
        sample_ids, source_rows = coordinates[:, 0], coordinates[:, 1]
        valid = sample_ids >= 0
        samples = []
        for sample_index in np.unique(sample_ids[valid]):
            positions = np.flatnonzero(sample_ids == sample_index)
            samples.append(ConsumerSampleRowMap(int(sample_index), positions, source_rows[positions]))
        return cls(len(coordinates), tuple(samples), valid, source_rows[~valid, None])


@dataclass(frozen=True)
class ConsumerLayerBinding:
    """Map a local model channel to its captured layer; no body reads occur
    here."""

    capture_layer: int
    manager: "ReplayManager"
    channel: ConsumerChannelKey


@dataclass(frozen=True)
class ConsumerModelLayout:
    """Captured shape and local model-layer destinations for one replay
    kind."""

    kind: str
    topk: int
    capture_layers: int
    bindings: tuple[ConsumerLayerBinding, ...]


@dataclass(frozen=True)
class Consumer:
    """Adapt an already selected TQ batch and bind lazy model-side readers.

    Full payloads retain their existing path. References stay lightweight until
    the model consumes a laid-out layer; TrainActor never opens body storage.
    """

    manager: "ReplayManager"
    kinds: frozenset[str]

    def fields(self, fields: Sequence[str]) -> list[str]:
        self.manager._ensure_open()
        requested = list(fields) + [FIELDS[kind] for kind in sorted(self.kinds)]
        return self.manager.config.fields(requested)

    def allows_per_rank_fetch(self, fields: Sequence[str]) -> bool:
        return not set(FIELDS.values()).intersection(self.fields(fields))

    def accept_batch(self, batch: dict[str, Any] | None, *, data_fields: Sequence[str], region: ReplayRegion) -> None:
        """Validate a postprocessed batch; queue selection stays with the
        caller.

        Reference decoding happens at the existing CPU postprocess boundary.
        This check neither reads tensor bodies nor acquires reader holds.
        """
        self.manager._ensure_open()
        if region.manager is not self.manager:
            raise ValueError("Replay region belongs to another manager")
        fields = self.fields(data_fields)
        if batch is not None and REFERENCE_FIELD in fields and REFERENCE_FIELD not in batch:
            raise ValueError("TQ response is missing requested replay references")
        region.validate(batch)

    def receive_references(self, batch: dict[str, Any]) -> None:
        self.manager._ensure_open()
        if REFERENCE_FIELD not in batch:
            return
        refs = self.manager.receive(REFERENCE_FIELD, batch[REFERENCE_FIELD], device="cpu")
        for ref, tokens in zip(refs, batch["tokens"], strict=True):
            # Called before the existing token H2D boundary, never on GPU.
            if hasattr(tokens, "device") and tokens.device.type != "cpu":
                raise ValueError("Replay identity validation requires CPU tokens")
            validate_sample(ref, tokens.tolist() if hasattr(tokens, "tolist") else list(tokens))
        batch[REFERENCE_FIELD] = refs

    def receive_tensors(self, batch: dict[str, Any], *, device: Any) -> None:
        self.manager._ensure_open()
        for kind in ("moe", "indexer"):
            field = FIELDS[kind]
            if field in batch and (kind == "moe" or kind in self.kinds):
                batch[field] = self.manager.receive(field, batch[field], device=device)


class ConsumerReadSession:
    """One microbatch/rank's reader holds, shared by its layer sources.

    The builder owns the first reference. Each bound source retains one; only
    the final release updates the store. A local lock excludes close during
    reads. Storage still authorizes every participating reference and lease.
    """

    def __init__(self, store: Any, refs: list[ReplayRef], *, partition: str | None = None) -> None:
        self.store = store
        self.refs = tuple(refs)
        self._leases = store.acquire_many(refs, partition=partition)
        self._users = 1
        self._closing = False
        self._bitmap_authorized: set[int] = set()
        self._lock = RLock()

    @property
    def read_scratch_bytes(self) -> int:
        """Conservative body plus row-selection scratch, including a large
        chunk.

        Storage uses its publication byte limit for explicit-key read batches.
        One indivisible chunk can exceed that limit.
        """
        largest_chunk = max(min(ref.rows, ref.chunk_rows) * ref.topk * 4 for ref in self.refs)
        return 2 * max(self.store.publication_bytes, largest_chunk)

    def retain(self) -> None:
        with self._lock:
            if self._closing:
                raise RuntimeError("Replay reader is closing or released")
            self._users += 1

    def read_into(self, sample: int, layer: int, rows: Any, output: Any, destinations: Any) -> None:
        with self._lock:
            if self._closing:
                raise RuntimeError("Replay reader is closing or released")
            self.store._read_into(self.refs[sample], layer, rows, output, destinations, self._leases[sample])

    def _read_layer_into(self, samples: tuple[ConsumerSampleRowMap, ...], layer: int, output: Any) -> None:
        with self._lock:
            if self._closing:
                raise RuntimeError("Replay reader is closing or released")
            if len(samples) == 1:
                rows = samples[0]
                self.read_into(rows.sample_index, layer, rows.source_rows, output, rows.output_rows)
            elif samples:
                self.store._read_many_into(
                    [
                        (
                            self.refs[rows.sample_index],
                            rows.source_rows,
                            rows.output_rows,
                            self._leases[rows.sample_index],
                        )
                        for rows in samples
                    ],
                    layer,
                    output,
                )

    def _read_layers_into(self, samples: tuple[ConsumerSampleRowMap, ...], outputs: list[tuple[int, Any]]) -> None:
        with self._lock:
            if self._closing:
                raise RuntimeError("Replay reader is closing or released")
            requests = [
                (self.refs[row.sample_index], row.source_rows, row.output_rows, self._leases[row.sample_index])
                for row in samples
            ]
            self.store._read_layers_into([(requests, layer, output) for layer, output in outputs])

    def release(self) -> None:
        with self._lock:
            if self._users == 0:
                return
            if self._users > 1:
                self._users -= 1
                return
            # Collection may partially succeed before raising. Forbid reads
            # from then on, but retain the final hold locally for a retry.
            self._closing = True
            self.store.release_many(list(self.refs), self._leases)
            self._users = 0


class ConsumerTensorCache:
    """Bounded CPU materializations and compressed GPU bodies for one batch.

    Shared by kinds, layers and microbatches; eviction never drops reader
    leases. Raw replay keeps its CPU policy. Both bitmap transports share the
    same small GPU cache; CPU transport copies only on a miss, GDR receives on
    the device.
    """

    def __init__(self, capacity_bytes: int) -> None:
        if capacity_bytes < 0:
            raise ValueError("Materialization budget must be nonnegative")
        self.capacity_bytes = capacity_bytes
        self.entries = OrderedDict()
        self.bytes = self.peak_bytes = self.hits = self.evictions = 0
        self._source_bytes: dict[Any, int] = {}
        self._working_set_bytes = 0
        self.encoded_read_bytes = self.encoded_h2d_bytes = self.decoded_rows = 0
        self.logical_decoded_bytes = 0
        self.encoded_rdma_bytes = self.gdr_read_calls = 0
        self.encoded_read_batches = self.encoded_read_bodies = self.encoded_batch_peak_bytes = 0
        self.encoded_fetch_seconds = 0.0
        self.gpu_bytes = self.gpu_peak_bytes = 0
        self.gpu_capacity_bytes = capacity_bytes
        self._read_ahead = _ConsumerReadAhead(self)
        self._encoded_read_ahead: _EncodedReadAhead | None = None
        self._encoded_groups: dict[Any, dict[Any, ConsumerLayerReader]] = {}
        self.encoded_prefetch_scheduled = self.encoded_prefetch_hits = self.encoded_prefetch_ready_hits = 0
        self.encoded_prefetch_wait_seconds = 0.0
        self.encoded_prefetch_peak_bytes = 0

    @property
    def prefetch_bytes(self) -> int:
        return self._read_ahead.reserved_bytes

    @property
    def prefetch_peak_bytes(self) -> int:
        return self._read_ahead.peak_managed_bytes

    def _register_source(self, source: ConsumerLayerReader, size: int) -> None:
        self._source_bytes[source._cache_key] = size
        self._working_set_bytes += size
        if source.reader.refs[0].codec == "int32":
            self._read_ahead.register(source)
        elif source.reader.refs[0].codec == QSA_BITMAP.name:
            if self._encoded_read_ahead is None:
                self._encoded_read_ahead = _EncodedReadAhead(min(self.capacity_bytes, 32 << 20))
            self._encoded_groups.setdefault(source._cache_group, {})[source._cache_key] = source

    def _unregister(self, key: Any) -> None:
        self._working_set_bytes -= self._source_bytes.pop(key, 0)
        self._read_ahead.unregister(key)
        for group_key, group in tuple(self._encoded_groups.items()):
            if key in group:
                del group[key]
                if not group:
                    del self._encoded_groups[group_key]
                break
        if not self._encoded_groups and self._encoded_read_ahead is not None:
            self._encoded_read_ahead.close()
            self._encoded_read_ahead = None
        if not self._source_bytes:
            # Encoded entries use immutable body keys, not source keys. Drop
            # their tensors explicitly rather than retaining them through the
            # cache/read-ahead reference cycle until a later cyclic GC pass.
            for remaining in tuple(self.entries):
                self.discard(remaining)

    def get(self, key: Any) -> Any:
        value = self.entries.get(key)
        if value is None:
            return None
        self.hits += 1
        self.entries.move_to_end(key)
        return value[0]

    def discard(self, key: Any) -> None:
        value = self.entries.pop(key, None)
        if value is not None:
            self.bytes -= value[1]
            if any(t.is_cuda for t in value[0] if t is not None):
                self.gpu_bytes -= value[1]

    def put(self, key: Any, value: tuple[torch.Tensor, torch.Tensor | None]) -> None:
        size = sum(t.numel() * t.element_size() for t in value if t is not None)
        on_gpu = any(t.is_cuda for t in value if t is not None)
        if size > self.capacity_bytes or (on_gpu and size > self.gpu_capacity_bytes):
            return
        while self.entries and (
            self.bytes + size > self.capacity_bytes or (on_gpu and self.gpu_bytes + size > self.gpu_capacity_bytes)
        ):
            self.discard(next(iter(self.entries)))
            self.evictions += 1
        self.entries[key] = (value, size)
        self.bytes += size
        if on_gpu:
            self.gpu_bytes += size
            self.gpu_peak_bytes = max(self.gpu_peak_bytes, self.gpu_bytes)
        self.peak_bytes = max(self.peak_bytes, self.bytes)


@dataclass(frozen=True)
class _BitmapSelection:
    key: str
    count: int
    source_rows: np.ndarray
    output_rows: np.ndarray
    width_bound: int


class ConsumerLayerReader(ConsumerSource):
    """Resolve one layer's final microbatch layout at the model's consumption
    boundary."""

    def __init__(
        self,
        reader: "ConsumerReadSession",
        batch_layout: ConsumerBatchLayout,
        *,
        layer: int,
        kind: str,
        device: Any,
        num_experts: int = 0,
        cache: ConsumerTensorCache | None = None,
        _qsa_bitmap_mode: bool = False,
    ) -> None:
        refs = reader.refs
        if not refs or any(m.kind != kind or m.topk != refs[0].topk or not 0 <= layer < m.layers for m in refs):
            raise ValueError("Incompatible replay manifests for a layer")
        if any(rows.sample_index >= len(refs) for rows in batch_layout.samples):
            raise ValueError("Replay row plan refers to a missing sample")
        if kind == "moe" and num_experts <= 0:
            raise ValueError("MoE replay requires a positive expert count")
        self.reader = reader
        self.batch_layout = batch_layout
        self.layer = layer
        self.kind = kind
        self.device = device
        self.num_experts = num_experts
        self.released = False
        self.cache = cache
        self._qsa_bitmap_mode = _qsa_bitmap_mode
        if _qsa_bitmap_mode and refs[0].codec != QSA_BITMAP.name:
            raise ValueError("Direct QSA bits require the QSA bitmap codec")
        self._cache_key = object()
        self._cache_group = (id(reader), id(batch_layout))
        reader.retain()
        if cache is not None:
            # Layouts are bound before model consumption. Estimate the original
            # host working set without fetching bodies or allocating tensors.
            row_bytes = refs[0].topk * (8 if kind == "moe" else 4) + (kind != "moe")
            cache._register_source(self, len(batch_layout) * row_bytes)

    def read(self) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | _QSABitmapReplay:
        with self.cache._read_ahead.consumer_lock if self.cache is not None else nullcontext():
            return self._read()

    def _read(self) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | _QSABitmapReplay:
        import torch

        if self.released:
            raise RuntimeError("Replay source has been released")
        if replay_codec(self.reader.refs[0].codec).encoded:
            return self._read_bitmap()
        cached = self.cache._read_ahead.read(self) if self.cache is not None else self._materialize()
        staging, mask = cached
        device = torch.device(self.device)
        # Callers own their returned tensors, while cached CPU storage stays
        # immutable. CUDA gets an independent allocation on the current stream.
        dtype = torch.long if self.kind == "moe" else torch.int32
        if staging.dtype == dtype:
            values = staging.clone() if device.type == "cpu" else staging.to(device, non_blocking=True)
        else:
            values = staging.to(device=device, dtype=dtype, non_blocking=True, copy=True)
        if mask is None:
            return values
        return values, mask.clone() if device.type == "cpu" else mask.to(device, non_blocking=True)

    def _read_bitmap(self) -> tuple[torch.Tensor, torch.Tensor] | _QSABitmapReplay:
        with self.reader._lock:
            if self.reader._closing:
                raise RuntimeError("Replay reader is closing or released")
            return self._materialize_bitmap()

    def _materialize_bitmap(self) -> tuple[torch.Tensor, torch.Tensor] | _QSABitmapReplay:
        import numpy as np
        import torch

        codec = replay_codec(self.reader.refs[0].codec)
        topk = self.reader.refs[0].topk
        words = ((max(ref.rows + 1 for ref in self.reader.refs) // 4 + 1) + 31) // 32
        output = torch.full(
            (len(self.batch_layout), words if self._qsa_bitmap_mode else topk),
            0 if self._qsa_bitmap_mode else -1,
            dtype=torch.int32,
            device=self.device,
        )
        selections = []
        pending_authorizations = {}
        for sample in self.batch_layout.samples:
            if not 0 <= sample.sample_index < len(self.reader.refs):
                raise ValueError("Replay row plan refers to a missing sample")
            ref = self.reader.refs[sample.sample_index]
            if ref.codec != codec.name or ref.topk != topk:
                raise ValueError("Cannot mix indexer codecs within a microbatch")
            if (
                sample.source_rows.ndim != 1
                or sample.output_rows.shape != sample.source_rows.shape
                or np.any(sample.source_rows < 0)
                or np.any(sample.source_rows >= ref.rows)
                or np.any(sample.output_rows < 0)
                or np.any(sample.output_rows >= len(self.batch_layout))
            ):
                raise ValueError("Invalid encoded replay row layout")
            lease = self.reader._leases[sample.sample_index]
            # A reader owns immutable refs/leases until its locked release.
            # Authorize once per sample rather than per model layer/recompute.
            if sample.sample_index not in self.reader._bitmap_authorized:
                pending_authorizations[sample.sample_index] = (ref, lease)
            chunks = sample.source_rows // ref.chunk_rows
            width_bound = codec.max_encoded_width
            for chunk in np.unique(chunks):
                if codec.name == QSA_BITMAP.name:
                    # Each immutable body is snapshotted at its own append
                    # stop, not the eventual document length. Remaps keep rows.
                    stop = min((int(chunk) + 1) * ref.chunk_rows, ref.rows)
                    blocks = stop // 4
                    width_bound = min(codec.max_encoded_width, 1 if blocks <= 512 else 1 + (blocks + 31) // 32)
                key = f"{ref.record}:{chunk}:{self.layer}"
                select = chunks == chunk
                selections.append(
                    _BitmapSelection(
                        key,
                        min(ref.chunk_rows, ref.rows - int(chunk) * ref.chunk_rows),
                        sample.source_rows[select] % ref.chunk_rows,
                        sample.output_rows[select],
                        width_bound,
                    )
                )
        if pending_authorizations:
            identities = list(pending_authorizations.values())
            if len(identities) == 1:
                self.reader.store.invoke("authorize_read", *identities[0])
            else:
                self.reader.store.invoke(
                    "authorize_reads", [ref for ref, _ in identities], [lease for _, lease in identities]
                )
            # A failed authorization must not leave even its successful prefix
            # marked. Duplicate references retain their distinct reader leases.
            self.reader._bitmap_authorized.update(pending_authorizations)
        capability = getattr(self.reader.store, "encoded_batch_limit", None)
        limit = capability(device=self.device) if capability is not None and selections else 0
        cursor = 0
        while cursor < len(selections):
            group = []
            bodies = {}
            misses = []
            reserved = 0
            while cursor < len(selections):
                selection = selections[cursor]
                if selection.key not in bodies:
                    cached = self.cache.get(("encoded", selection.key)) if self.cache is not None else None
                    # Reserve a proven document bound for QSA snapshots; other
                    # codecs retain their conservative maximum domain.
                    size = (
                        cached[0].nbytes
                        if cached is not None
                        else selection.count * selection.width_bound * codec.element_bytes
                    )
                    if group and (not limit or reserved + size > limit or len(bodies) >= 128):
                        cached = None
                        break
                    bodies[selection.key] = cached[0] if cached is not None else None
                    if cached is None:
                        misses.append(selection)
                    cached = None
                    reserved += size
                group.append(selection)
                cursor += 1
                # Preserve unbatched custom transports, and allow an
                # unavoidable single body larger than the batch budget.
                if not limit or reserved >= limit:
                    break
            if misses:
                bodies.update(self._fetch_bitmap_bodies(misses, output.device))
            if self.cache is not None:
                self.cache.encoded_batch_peak_bytes = max(
                    self.cache.encoded_batch_peak_bytes, sum(body.nbytes for body in bodies.values())
                )
            for selection in group:
                data = bodies[selection.key]
                # Hits can be evicted while later misses are inserted. Record
                # every consuming stream before dropping the group's hold.
                data.record_stream(torch.cuda.current_stream())
                # Both transports share the same bounded GPU encoded cache.
                # Decode into the final microbatch, without another payload copy.
                if self.cache is not None:
                    self.cache.decoded_rows += len(selection.source_rows)
                    self.cache.logical_decoded_bytes += len(selection.source_rows) * topk * codec.element_bytes
                if np.all(np.diff(selection.source_rows) == 1) and np.all(np.diff(selection.output_rows) == 1):
                    if self._qsa_bitmap_mode:
                        from ._bitmap import copy_dense_bits_into

                        copy_dense_bits_into(
                            data,
                            None,
                            None,
                            output,
                            source_start=int(selection.source_rows[0]),
                            destination_start=int(selection.output_rows[0]),
                            count=len(selection.source_rows),
                        )
                    else:
                        codec.decode_contiguous_into(
                            data,
                            int(selection.source_rows[0]),
                            int(selection.output_rows[0]),
                            len(selection.source_rows),
                            output,
                        )
                else:
                    # Pageable CUDA coordinate copies synchronize the model
                    # stream. Keep arbitrary row maps asynchronous too.
                    host_coordinates = torch.empty((2, len(selection.source_rows)), dtype=torch.long, pin_memory=True)
                    host_coordinates[0].numpy()[:] = selection.source_rows
                    host_coordinates[1].numpy()[:] = selection.output_rows
                    rows, destinations = host_coordinates.to(self.device, non_blocking=True).unbind()
                    if self._qsa_bitmap_mode:
                        from ._bitmap import copy_dense_bits_into

                        copy_dense_bits_into(data, rows, destinations, output)
                    else:
                        codec.decode_into(data, rows, destinations, output)
                del data
            bodies.clear()
        # CPU layout is authoritative. The model can compute exact fallback
        # only for missing rows without a GPU nonzero()/item() synchronization.
        output._relax_indexer_missing_rows = tuple(int(row) for row in np.flatnonzero(~self.batch_layout.valid))
        host_mask = torch.empty(self.batch_layout.valid.shape, dtype=torch.bool, pin_memory=True)
        host_mask.numpy()[:] = self.batch_layout.valid
        self._schedule_bitmap_prefetch(selections)
        mask = host_mask.to(self.device, non_blocking=True)
        if self._qsa_bitmap_mode:
            return _QSABitmapReplay(output, mask, output._relax_indexer_missing_rows)
        return output, mask

    def _schedule_bitmap_prefetch(self, selections: list[_BitmapSelection]) -> None:
        # Only a foreground read may select the transport or initialize CUDA.
        # A CPU worker reads immutable bodies and never takes this reader lock.
        if (
            self.cache is None
            or self.reader.refs[0].codec != QSA_BITMAP.name
            or getattr(self.reader.store, "_device_reads", None) is not False
        ):
            return
        ahead = self.cache._encoded_read_ahead
        if ahead is None or ahead.pending is not None or not ahead.capacity_bytes:
            return
        limit = min(ahead.capacity_bytes, self.reader.store.encoded_batch_limit(device=self.device))
        if not limit:
            return
        following = False
        for candidate in self.cache._encoded_groups[self._cache_group].values():
            if candidate is self:
                following = True
                continue
            if not following or candidate.released:
                continue
            # This group shares the exact reader, row layout and authorization;
            # only the captured layer in each immutable key changes.
            keys = []
            seen = set()
            reserved = 0
            for selection in selections:
                key = selection.key.rsplit(":", 1)[0] + f":{candidate.layer}"
                if key in seen:
                    continue
                seen.add(key)
                cached = self.cache.entries.get(("encoded", key))
                size = cached[1] if cached is not None else selection.count * selection.width_bound * 4
                if reserved + size > limit or len(seen) > 128:
                    break
                reserved += size
                if cached is None:
                    keys.append(key)
            if not keys:
                continue
            frozen_keys = tuple(keys)
            store = candidate.reader.store
            if ahead.schedule(
                candidate._cache_key,
                frozen_keys,
                reserved,
                lambda: store.read_encoded(list(frozen_keys), device="cpu"),
            ):
                self.cache.encoded_prefetch_scheduled += 1
                self.cache.encoded_prefetch_peak_bytes = max(self.cache.encoded_prefetch_peak_bytes, reserved)
            break

    def _fetch_bitmap_bodies(self, selections: list[_BitmapSelection], device: Any) -> dict[str, torch.Tensor]:
        from time import perf_counter

        import numpy as np
        import torch

        started = perf_counter()
        keys = tuple(selection.key for selection in selections)
        blocks = None
        ahead = self.cache._encoded_read_ahead if self.cache is not None else None
        if ahead is not None:
            window = ahead.pending
            ready = window is not None and window.future.done()
            blocks = ahead.take(self._cache_key, keys)
            if blocks is not None:
                self.cache.encoded_prefetch_hits += 1
                self.cache.encoded_prefetch_ready_hits += ready
                self.cache.encoded_prefetch_wait_seconds += perf_counter() - started
        if blocks is None:
            blocks = self.reader.store.read_encoded(list(keys), device=self.device)
        if self.cache is not None:
            self.cache.encoded_fetch_seconds += perf_counter() - started
        if len(blocks) != len(selections):
            raise ValueError("Missing encoded replay bodies")
        # Validate the complete batch before putting any member in the cache.
        for selection, block in zip(selections, blocks, strict=True):
            if (
                block.dtype not in (np.int32, torch.int32)
                or block.ndim != 2
                or block.shape[0] != selection.count
                or not 1 <= block.shape[1] <= selection.width_bound
            ):
                raise ValueError("Invalid encoded replay body shape")
            if isinstance(block, torch.Tensor) and (not block.is_cuda or block.device != device):
                raise ValueError("GDR body arrived on an unexpected device")
        result = {}
        host_elements = sum(block.size for block in blocks if isinstance(block, np.ndarray))
        # One pinned allocation per bounded transport batch avoids per-body
        # cudaHostAlloc. CUDA bodies remain independent allocations so cache
        # eviction cannot retain an unaccounted shared GPU slab.
        host_batch = torch.empty(host_elements, dtype=torch.int32, pin_memory=True) if host_elements else None
        host_offset = 0
        for selection, block in zip(selections, blocks, strict=True):
            if isinstance(block, torch.Tensor):
                staging = block
                if self.cache is not None:
                    self.cache.encoded_rdma_bytes += block.nbytes
            else:
                host_staging = host_batch[host_offset : host_offset + block.size].view(block.shape)
                host_offset += block.size
                host_staging.numpy()[:] = block
                staging = host_staging.to(self.device, non_blocking=True)
                if self.cache is not None:
                    self.cache.encoded_h2d_bytes += host_staging.nbytes
            result[selection.key] = staging
            if self.cache is not None:
                self.cache.encoded_read_bytes += block.nbytes
                self.cache.put(("encoded", selection.key), (staging, None))
        if self.cache is not None:
            self.cache.encoded_read_batches += 1
            self.cache.encoded_read_bodies += len(blocks)
            if all(isinstance(block, torch.Tensor) for block in blocks):
                self.cache.gdr_read_calls += 1
        return result

    def _materialize(self) -> tuple[torch.Tensor, torch.Tensor | None]:
        staging = self._allocate_staging()
        self.reader._read_layer_into(self.batch_layout.samples, self.layer, staging.numpy())
        return self._finish_materialize(staging)

    def _allocate_staging(self) -> torch.Tensor:
        import torch

        first = self.reader.refs[0]
        device = torch.device(self.device)
        # Narrow only under cache pressure. Small working sets keep their
        # original representation to avoid an unnecessary per-read CUDA cast.
        compact_cache = self.cache is not None and 0 < self.cache.capacity_bytes < self.cache._working_set_bytes
        staging_dtype = torch.int32
        if self.kind == "moe" and (not compact_cache or self.num_experts > 2**31):
            staging_dtype = torch.long
        # A pageable as_tensor(..., device="cuda") serializes each replay read
        # with the current CUDA stream. Pinned staging lets the copy enqueue
        # on that stream without waiting for preceding model kernels.
        staging = torch.empty(
            (len(self.batch_layout), first.topk),
            dtype=staging_dtype,
            device="cpu",
            pin_memory=device.type == "cuda",
        )
        output = staging.numpy()
        output.fill(-1)
        return staging

    def _finish_materialize(self, staging: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        import numpy as np
        import torch

        first = self.reader.refs[0]
        device = torch.device(self.device)
        compact_cache = self.cache is not None and 0 < self.cache.capacity_bytes < self.cache._working_set_bytes
        output = staging.numpy()
        if self.kind == "moe":
            output[~self.batch_layout.valid] = (
                (self.batch_layout.padding_offsets * first.layers + self.layer) * first.topk + np.arange(first.topk)
            ) % self.num_experts
            # Keep the bounded host cache compact when every index is exactly
            # representable. Storage remains int32; consumers still get int64.
            if (
                compact_cache
                and 0 < output.size <= self.cache.capacity_bytes
                and output.min() >= 0
                and output.max() <= 255
            ):
                compact = torch.empty(staging.shape, dtype=torch.uint8, device="cpu", pin_memory=device.type == "cuda")
                compact.numpy()[:] = output
                staging = compact
            return staging, None
        mask = torch.empty(
            self.batch_layout.valid.shape, dtype=torch.bool, device="cpu", pin_memory=device.type == "cuda"
        )
        mask.numpy()[:] = self.batch_layout.valid
        # Never mutate/reuse a cached buffer; PyTorch also tracks outstanding
        # asynchronous copies if an eviction drops the final Python reference.
        return staging, mask

    def release(self) -> None:
        with self.cache._read_ahead.consumer_lock if self.cache is not None else nullcontext():
            self._release()

    def _release(self) -> None:
        if not self.released:
            if self.cache is not None:
                self.cache._read_ahead.drain()
                if self.reader.refs[0].codec == QSA_BITMAP.name and self.cache._encoded_read_ahead is not None:
                    self.cache._encoded_read_ahead.drain()
                self.cache.discard(self._cache_key)
            self.reader.release()
            if self.cache is not None:
                self.cache._unregister(self._cache_key)
            self.released = True


@dataclass(frozen=True)
class ConsumerRequest:
    """A layer request after the engine has selected samples and arranged rows.

    refs identify recordings; rows describe this rank's selection. payload is
    the already arranged full-mode value or a previously bound lazy source.
    """

    kind: str
    layer: int
    refs: tuple[ReplayRef, ...] = ()
    rows: ConsumerBatchLayout | None = None
    device: Any = "cpu"
    num_experts: int = 0
    microbatch_id: int = 0
    payload: Any = None

    def __post_init__(self) -> None:
        if bool(self.refs) == (self.payload is not None):
            raise ValueError("Consumer request requires exactly one of references or a full payload")
        if self.refs and (self.rows is None or any(ref.kind != self.kind for ref in self.refs)):
            raise ValueError("Reference consumer request requires a matching kind and row layout")


class ConsumerExecutor:
    """Resolve requests with a shared session; bound reads remain lazy and
    repeatable."""

    @staticmethod
    def bind(
        request: ConsumerRequest,
        *,
        reader: ConsumerReadSession | None = None,
        cache: ConsumerTensorCache | None = None,
    ) -> Any:
        if request.payload is not None:
            return request.payload
        if reader is None or tuple(reader.refs) != request.refs:
            raise ValueError("Consumer request does not match its acquired reader session")
        return ConsumerLayerReader(
            reader,
            request.rows,
            layer=request.layer,
            kind=request.kind,
            device=request.device,
            num_experts=request.num_experts,
            cache=cache,
            _qsa_bitmap_mode=request.refs[0].codec == QSA_BITMAP.name,
        )

    @staticmethod
    def execute(request: ConsumerRequest) -> Any:
        if request.payload is None:
            raise ValueError("Bind a reference request before executing it")
        return request.payload.read() if isinstance(request.payload, ConsumerSource) else request.payload


@dataclass(frozen=True)
class ConsumerReadTask:
    """Framework adapter supplies the final row plan and local layer
    destinations."""

    model: ConsumerModelLayout
    rows: ConsumerBatchLayout
    microbatch: int
    device: Any
    num_experts: int = 0


def bind_read_task(
    task: ConsumerReadTask, reader: ConsumerReadSession, cache: ConsumerTensorCache | None = None
) -> None:
    for binding in task.model.bindings:
        request = ConsumerRequest(
            kind=task.model.kind,
            layer=binding.capture_layer,
            refs=tuple(reader.refs),
            rows=task.rows,
            device=task.device,
            num_experts=task.num_experts,
            microbatch_id=task.microbatch,
        )
        source = ConsumerExecutor.bind(request, reader=reader, cache=cache)
        try:
            binding.manager.bind(binding.channel, ConsumerRecord(task.model.kind, task.microbatch, source))
        except BaseException as primary:
            try:
                source.release()
            except Exception as cleanup:
                primary.add_note(f"Replay source release failed: {cleanup!r}")
            raise
