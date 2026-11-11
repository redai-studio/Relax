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

from ._prefetch import _ConsumerReadAhead
from .codecs import replay_codec
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
        self.gpu_capacity_bytes = min(capacity_bytes, 64 << 20)
        self._read_ahead = _ConsumerReadAhead(self)

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

    def _unregister(self, key: Any) -> None:
        self._working_set_bytes -= self._source_bytes.pop(key, 0)
        self._read_ahead.unregister(key)

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
        self._cache_key = object()
        self._cache_group = (id(reader), id(batch_layout))
        reader.retain()
        if cache is not None:
            # Layouts are bound before model consumption. Estimate the original
            # host working set without fetching bodies or allocating tensors.
            row_bytes = refs[0].topk * (8 if kind == "moe" else 4) + (kind != "moe")
            cache._register_source(self, len(batch_layout) * row_bytes)

    def read(self) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        with self.cache._read_ahead.consumer_lock if self.cache is not None else nullcontext():
            return self._read()

    def _read(self) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
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

    def _read_bitmap(self) -> tuple[torch.Tensor, torch.Tensor]:
        with self.reader._lock:
            if self.reader._closing:
                raise RuntimeError("Replay reader is closing or released")
            return self._materialize_bitmap()

    def _materialize_bitmap(self) -> tuple[torch.Tensor, torch.Tensor]:
        import numpy as np
        import torch

        codec = replay_codec(self.reader.refs[0].codec)
        topk = self.reader.refs[0].topk
        output = torch.full((len(self.batch_layout), topk), -1, dtype=torch.int32, device=self.device)
        selections = []
        for sample in self.batch_layout.samples:
            ref = self.reader.refs[sample.sample_index]
            if ref.codec != codec.name or ref.topk != topk:
                raise ValueError("Cannot mix indexer codecs within a microbatch")
            lease = self.reader._leases[sample.sample_index]
            # A reader owns immutable refs/leases until its locked release.
            # Authorize once per sample rather than per model layer/recompute.
            if sample.sample_index not in self.reader._bitmap_authorized:
                self.reader.store.invoke("authorize_read", ref, lease)
                self.reader._bitmap_authorized.add(sample.sample_index)
            chunks = sample.source_rows // ref.chunk_rows
            for chunk in np.unique(chunks):
                key = f"{ref.record}:{chunk}:{self.layer}"
                select = chunks == chunk
                selections.append(
                    _BitmapSelection(
                        key,
                        min(ref.chunk_rows, ref.rows - int(chunk) * ref.chunk_rows),
                        sample.source_rows[select] % ref.chunk_rows,
                        sample.output_rows[select],
                    )
                )
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
                    # Missing widths come from TQ metadata. Reserve the codec's
                    # maximum, not an estimate based on this recipe's lengths.
                    size = (
                        cached[0].nbytes
                        if cached is not None
                        else selection.count * codec.max_encoded_width * codec.element_bytes
                    )
                    if group and (not limit or reserved + size > limit or len(bodies) >= 32):
                        cached = None
                        break
                    bodies[selection.key] = cached[0] if cached is not None else None
                    if cached is None:
                        misses.append(selection)
                    cached = None
                    reserved += size
                group.append(selection)
                cursor += 1
                # Preserve singleton CPU reads, and the previous behavior for
                # an unavoidable single body larger than the batch budget.
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
                # Pageable as_tensor(..., device="cuda") synchronizes the model
                # stream. Keep these small coordinate copies asynchronous too.
                host_coordinates = torch.empty((2, len(selection.source_rows)), dtype=torch.long, pin_memory=True)
                host_coordinates[0].numpy()[:] = selection.source_rows
                host_coordinates[1].numpy()[:] = selection.output_rows
                rows, destinations = host_coordinates.to(self.device, non_blocking=True).unbind()
                # Both transports share the same bounded GPU encoded cache.
                # Decode into the final microbatch, without another payload copy.
                if self.cache is not None:
                    self.cache.decoded_rows += len(selection.source_rows)
                    self.cache.logical_decoded_bytes += len(selection.source_rows) * topk * codec.element_bytes
                codec.decode_into(data, rows, destinations, output)
                del data
            bodies.clear()
        # CPU layout is authoritative. The model can compute exact fallback
        # only for missing rows without a GPU nonzero()/item() synchronization.
        output._relax_indexer_missing_rows = tuple(int(row) for row in np.flatnonzero(~self.batch_layout.valid))
        host_mask = torch.empty(self.batch_layout.valid.shape, dtype=torch.bool, pin_memory=True)
        host_mask.numpy()[:] = self.batch_layout.valid
        return output, host_mask.to(self.device, non_blocking=True)

    def _fetch_bitmap_bodies(self, selections: list[_BitmapSelection], device: Any) -> dict[str, torch.Tensor]:
        from time import perf_counter

        import numpy as np
        import torch

        codec = replay_codec(self.reader.refs[0].codec)
        started = perf_counter()
        blocks = self.reader.store.read_encoded([selection.key for selection in selections], device=self.device)
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
                or not 1 <= block.shape[1] <= codec.max_encoded_width
            ):
                raise ValueError("Invalid encoded replay body shape")
            if isinstance(block, torch.Tensor) and (not block.is_cuda or block.device != device):
                raise ValueError("GDR body arrived on an unexpected device")
            if len(blocks) > 1 and not isinstance(block, torch.Tensor):
                raise ValueError("Batched bitmap reads require GPU bodies")
        result = {}
        for selection, block in zip(selections, blocks, strict=True):
            if isinstance(block, torch.Tensor):
                staging = block
                if self.cache is not None:
                    self.cache.encoded_rdma_bytes += block.nbytes
            else:
                host_staging = torch.empty(block.shape, dtype=torch.int32, pin_memory=True)
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
