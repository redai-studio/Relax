# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Replay manager: wire data, train-region ownership and local consumer
state."""

import importlib
import json
import uuid
from collections.abc import Awaitable, Callable, Collection, Iterable, MutableMapping, Sequence
from contextlib import AbstractContextManager, contextmanager
from dataclasses import asdict, dataclass, field, replace
from typing import TYPE_CHECKING, Any
from weakref import WeakValueDictionary

from .config import ReplayConfig, default_reference_r3, get_store, rollout_kinds, validate_reference_training
from .consumer import ConsumerRecord, ConsumerSource
from .ref_data import FIELDS, REFERENCE_FIELD, ReplayRef, iter_refs, publisher_holds, register_kind


if TYPE_CHECKING:
    from .consumer import Consumer, ConsumerReadSession
    from .producer import Producer
    from .storage import TQReplayStorage


def _output_ids(samples: Sequence[Any]) -> set[str]:
    return {token for sample in samples for token in getattr(sample, "_replay_route_holds", {}).values()}


def _sample_media(samples: Sequence[Any]) -> list[dict]:
    return [
        entry
        for sample in samples
        for entry in (sample.rollout_replay_refs or {}).get("mm_embedding", {}).get("media", [])
    ]


def refs_from_samples(samples: Sequence[Any]) -> list[ReplayRef]:
    return [ref for sample in samples for ref in iter_refs(sample.rollout_replay_refs or {})]


def new_data_id() -> str:
    """Allocate locally so the publisher knows the ID even if registration
    loses its reply."""
    return uuid.uuid4().hex


class ConsumerPhase:
    def __init__(self, *, environment: MutableMapping[str, str] | None = None, key: str | None = None) -> None:
        if (environment is None) != (key is None):
            raise ValueError("A phase environment and key must be supplied together")
        self._environment = environment
        self._key = key
        self._stage = "fallthrough"

    def get(self) -> str:
        # Do not silently initialize a missing legacy environment variable.
        return self._stage if self._environment is None else self._environment[self._key]

    def set(self, stage: str) -> None:
        if stage not in {"fallthrough", "record", "replay_forward", "replay_backward"}:
            raise ValueError(f"Invalid replay stage {stage!r}")
        if self._environment is None:
            self._stage = stage
        else:
            self._environment[self._key] = stage

    def enter_forward(self) -> str:
        previous = self.get()
        self.set("replay_forward")
        return previous


class ConsumerConsumptionError(RuntimeError):
    def __init__(self, phase: str, consumed: int, recorded: int) -> None:
        self.phase = phase
        self.consumed = consumed
        self.recorded = recorded
        super().__init__(f"Replay {phase} consumed {consumed}/{recorded} records")


@dataclass
class ConsumerChannel:
    kind: str
    entries: list[Any] = field(default_factory=list)
    microbatch_ids: dict[int, int] = field(default_factory=dict)
    forward_index: int = 0
    backward_index: int = 0

    def bind(self, record: ConsumerRecord) -> None:
        """Append one source in schedule order."""
        if record.kind != self.kind or record.microbatch_id != len(self.entries):
            raise ValueError("Replay kind or microbatch order does not match the channel")
        self.microbatch_ids[len(self.entries)] = record.microbatch_id
        self.entries.append(record.payload)

    def consume(
        self,
        phase: str,
        *,
        microbatch_id: int | None = None,
        advance_before_lookup: bool = False,
        materialize: bool = True,
    ) -> Any:
        if phase not in ("replay_forward", "replay_backward"):
            raise ValueError(f"Cannot consume replay in stage {phase!r}")
        cursor = "forward_index" if phase == "replay_forward" else "backward_index"
        index = getattr(self, cursor)
        if microbatch_id is not None:
            recorded_id = self.microbatch_ids.get(index)
            if recorded_id is None or recorded_id != microbatch_id:
                raise RuntimeError(f"Replay microbatch mismatch: expected={microbatch_id}, cursor={index}")
        # Legacy Indexer advances even on overrun; MoE advances only on success.
        if advance_before_lookup:
            setattr(self, cursor, index + 1)
        payload = self.entries[index]
        if payload is None:
            raise ValueError("Bind a reference request before executing it")
        if materialize and isinstance(payload, ConsumerSource):
            payload = payload.read()
        if not advance_before_lookup:
            setattr(self, cursor, index + 1)
        return payload

    def reset_forward(self, *, require_consumed: bool = False) -> None:
        if require_consumed and self.forward_index != len(self.entries):
            raise ConsumerConsumptionError("replay_forward", self.forward_index, len(self.entries))
        self.forward_index = 0

    def check_consumed(self, *, expect_backward: bool = False) -> None:
        cursors = ("forward_index", "backward_index") if expect_backward else ("forward_index",)
        for cursor in cursors:
            count = getattr(self, cursor)
            if count != len(self.entries):
                phase = "replay_forward" if cursor == "forward_index" else "replay_backward"
                raise ConsumerConsumptionError(phase, count, len(self.entries))

    def release_sources(self) -> None:
        # Inline tensors retain Python's original ownership semantics. A bound
        # source owns a consumer lease, which is released only at this boundary.
        error = None
        for payload in self.entries:
            if isinstance(payload, ConsumerSource):
                try:
                    payload.release()
                except Exception as exc:
                    error = error or exc
        if error is not None:
            # Keep failed leases reachable for retry, while still attempting
            # every release. A failed clear must not be reported as success.
            raise error

    def clear(self) -> None:
        self.release_sources()
        # Rebinding preserves the original MoE API's externally held lists.
        self.entries = []
        self.microbatch_ids = {}
        self.forward_index = self.backward_index = 0


@dataclass(frozen=True)
class ReplayKind:
    """Importable producer/consumer contract for built-ins and future adapters.

    New indexer kinds register their own implementation through this contract;
    the manager only provides MoE routes and visual embeddings by default.
    """

    name: str
    field: str
    producer: str
    consumer: str
    patches: tuple[tuple[str, str], ...] = ()


def _load_adapter(path: str) -> Callable:
    module, name = path.rsplit(":", 1)
    return getattr(importlib.import_module(module), name)


def _builtin_kinds() -> dict[str, ReplayKind]:
    prefix = "relax.distributed.replay_manager.backend"
    return {
        kind: ReplayKind(
            kind,
            FIELDS[kind],
            prefix + ".sglang:SGLangProducer",
            prefix + ".megatron:MegatronConsumer",
            (("sglang", prefix + ".sglang:apply_sglang_patch"),),
        )
        for kind in ("moe", "mm_embedding")
    }


class ReplayManager:
    """A model adapter owns this manager and decides when its records expire.

    Full/reference delivery follows one immutable configuration. Local channels
    own binding, repeated reads and consumer leases. Adapters retain model
    semantics, layout, capture placement and legacy error policy.
    """

    def __init__(
        self,
        *,
        config: ReplayConfig | None = None,
        store_factory: Callable[[], "TQReplayStorage"] | None = None,
    ) -> None:
        self.config = config if config is not None else ReplayConfig()
        self._closed = False
        self._store_factory = store_factory
        self._store_instance: TQReplayStorage | None = None
        # A copied native sample can borrow the same manifest, even with the same index.
        self.native_route_owners: WeakValueDictionary[str, Any] = WeakValueDictionary()
        self._owns_run = False
        self._run_stopped = False
        self._finalization_error: Exception | None = None
        self._kinds = _builtin_kinds()
        self._enabled_kinds = None
        self._installed_engines: dict[str, frozenset[str]] = {}

    def register_kind(self, definition: ReplayKind) -> None:
        """Register the same importable definition in each participating
        process."""
        self._ensure_open()
        previous = self._kinds.get(definition.name)
        if previous is not None and previous != definition:
            raise ValueError("Replay kind already has another definition")
        if self._installed_engines and previous is None:
            raise RuntimeError("Register replay kinds before installing engine hooks")
        register_kind(definition.name, definition.field)
        self._kinds[definition.name] = definition

    def initialize(
        self,
        *,
        kinds: Iterable[str],
        engine: str | None = None,
        connection: Any = None,
        owner: bool = False,
    ) -> None:
        """Select kinds, connect existing storage and install process-local
        hooks."""
        self._ensure_open()
        selected = frozenset(kinds)
        if not selected.issubset(self._kinds):
            raise ValueError(f"Unregistered replay kinds: {selected - self._kinds.keys()}")
        if self._enabled_kinds is not None and self._enabled_kinds != selected:
            raise RuntimeError("Replay configuration cannot change within a process")
        self._enabled_kinds = selected
        if owner and self.uses_references and not self._owns_run:
            self.start_run()
        if connection is not None and self.uses_references:
            self.store().register_local_owner(connection)
        if engine is not None and engine not in self._installed_engines:
            patches: dict[str, set[str]] = {}
            for kind in sorted(selected):
                for target, path in self._kinds[kind].patches:
                    if target == engine:
                        patches.setdefault(path, set()).add(kind)
            # Kinds sharing an engine function share one patch installation.
            for path, patch_kinds in patches.items():
                _load_adapter(path)(manager=self, kinds=frozenset(patch_kinds))
            self._installed_engines[engine] = selected

    def register_storage(self, factory: Callable[[], "TQReplayStorage"]) -> None:
        """Register the process-local body storage before it is first used."""
        self._ensure_open()
        if self._store_instance is not None or self._owns_run:
            raise RuntimeError("Cannot replace an active replay storage")
        self._store_factory = factory

    def export_registration(self) -> dict[str, Any]:
        """Transfer definitions, never live clients or monkey-patched
        functions."""
        if self._enabled_kinds is None:
            raise RuntimeError("Initialize replay kinds before exporting worker registration")
        return {"kinds": [asdict(self._kinds[name]) for name in sorted(self._enabled_kinds)]}

    def initialize_worker(self, registration: dict[str, Any], *, engine: str) -> None:
        definitions = []
        for item in registration["kinds"]:
            definition = ReplayKind(**{**item, "patches": tuple(tuple(pair) for pair in item["patches"])})
            self.register_kind(definition)
            definitions.append(definition.name)
        self.initialize(kinds=definitions, engine=engine)

    def start_run(self) -> None:
        """Open the driver-owned registry before constructing workers."""
        if self.uses_references:
            self.store().start_run()
            self._owns_run = True
            self._run_stopped = False

    def finish_run(self) -> None:
        """Called by standard normal/error shutdown, never by an in-job
        restart."""
        if self._finalization_error is not None:
            raise RuntimeError("Replay finalization previously failed") from self._finalization_error
        if self._owns_run:
            try:
                self.store().close_run()
            except Exception as error:
                self._finalization_error = error
                try:
                    self.store().abandon_run()
                except Exception as cleanup:
                    error.add_note(f"Replay registry cleanup failed: {cleanup!r}")
                raise
            finally:
                self._owns_run = False
                self._run_stopped = True

    def prepare_restart(self, runtime_env: dict[str, Any]) -> None:
        """Discard the failed generation with TQ and isolate replayed rollout.

        Global recovery rolls training back to a checkpoint and destroys TQ;
        its references and dead-worker leases must not survive into the new
        run. Call start_run only after reconnecting Ray.
        """
        if not self.uses_references:
            return
        import os

        self._run_stopped = True
        if self._owns_run:
            self.store().abandon_run()
        self._owns_run = False
        self._store_instance = None
        self._finalization_error = None
        self.config = replace(self.config, run_id=new_data_id())
        os.environ["RELAX_REPLAY_RUN_ID"] = self.config.run_id
        runtime_env.setdefault("env_vars", {})["RELAX_REPLAY_RUN_ID"] = self.config.run_id

    @property
    def uses_references(self) -> bool:
        return self.config.delivery == "reference"

    def store(self) -> "TQReplayStorage":
        self._ensure_open()
        if not self.uses_references:
            raise RuntimeError("Full replay delivery does not use a body store")
        if self._store_instance is None:
            self._store_instance = self._store_factory() if self._store_factory is not None else get_store(self.config)
        return self._store_instance

    def reader(
        self,
        refs: list["ReplayRef"],
        *,
        partition: str | None = None,
    ) -> AbstractContextManager["ConsumerReadSession"]:
        return self.store().reader(refs, partition=partition)

    def register_producer(self, kinds: Iterable[str], *, adapter: Callable | None = None) -> "Producer":
        self._ensure_open()
        kinds = frozenset(kinds)
        if not kinds.issubset(FIELDS):
            raise ValueError(f"Unsupported replay kinds: {kinds - FIELDS.keys()}")
        if adapter is None:
            paths = {self._kinds[kind].producer for kind in kinds if kind in self._kinds}
            if not kinds.issubset(self._kinds) or len(paths) > 1:
                raise ValueError("Register producers with a common adapter together")
            adapter = _load_adapter(next(iter(paths)) if paths else self._kinds["moe"].producer)
        return adapter(self, kinds)

    def register_consumer(self, kinds: Iterable[str] = (), *, adapter: Callable | None = None) -> "Consumer":
        self._ensure_open()
        kinds = frozenset(kinds)
        if not kinds.issubset(FIELDS):
            raise ValueError(f"Unsupported replay kinds: {kinds - FIELDS.keys()}")
        if adapter is None:
            paths = {self._kinds[kind].consumer for kind in kinds if kind in self._kinds}
            if not kinds.issubset(self._kinds) or len(paths) > 1:
                raise ValueError("Register consumers with a common adapter together")
            adapter = _load_adapter(next(iter(paths)) if paths else self._kinds["moe"].consumer)
        return adapter(self, kinds)

    def region(self, partition: str) -> "ReplayRegion":
        self._ensure_open()
        if not isinstance(partition, str) or not partition:
            raise ValueError("Replay region requires an explicit TQ partition identity")
        return ReplayRegion(self, partition)

    async def on_partition_cleared(self, partition: str) -> None:
        if self.uses_references and not self._run_stopped:
            await self.store().invoke_async("retire_partition", partition)

    def sample_refs(self, samples: Sequence[Any]) -> list["ReplayRef"]:
        return refs_from_samples(samples) if self.uses_references else []

    def handoff_buffered_routes(self, groups: Sequence[Sequence[Any]]) -> None:
        """Carry native publisher ownership across the Ray buffer's
        serialization boundary."""
        if not self.native_route_owners:
            return
        for group in groups:
            for sample in group:
                owned = {
                    ref.record
                    for ref in iter_refs(sample.rollout_replay_refs or {})
                    if self.native_route_owners.get(ref.record) is sample
                }
                if owned:
                    sample._buffered_route_records = owned
                    for record in owned:
                        del self.native_route_owners[record]

    def receive_buffered_routes(self, groups: Sequence[Sequence[Any]]) -> None:
        if self.uses_references:
            for group in groups:
                for sample in group:
                    for record in vars(sample).pop("_buffered_route_records", ()):
                        self.native_route_owners[record] = sample

    def discard(self, samples: Sequence[Any]) -> None:
        if self.uses_references:
            refs, media = publisher_holds(sample.rollout_replay_refs or {} for sample in samples)
            if refs or media:
                self.store().invoke("discard", refs, media=media, output_ids=_output_ids(samples))

    async def discard_async(self, samples: Sequence[Any], *, unsent: bool = False, native: bool = False) -> None:
        from relax.utils.async_utils import finish_before_cancellation

        if self.uses_references:
            refs, media = [], []
            for sample in samples:
                routes, leases = publisher_holds((sample.rollout_replay_refs or {},))
                if native:
                    owned = getattr(sample, "_replay_route_holds", {})
                    routes = [
                        ref
                        for ref in routes
                        if ref.record in owned or self.native_route_owners.get(ref.record) is sample
                    ]
                refs.extend(routes)
                media.extend(leases)
            if refs or media:
                await finish_before_cancellation(
                    self.store().invoke_async(
                        "discard", refs, media=media, keep_queued=unsent, output_ids=_output_ids(samples)
                    ),
                    "replay-discard",
                )

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("Replay manager is closed")

    def build_tq_fields(
        self, samples: Sequence[Any], *, required_fields: frozenset[str] = frozenset()
    ) -> dict[str, Any]:
        self._ensure_open()
        if self.uses_references:
            from .ref_data import REFERENCE_FIELD, validate_sample

            values = []
            for sample in samples:
                refs = sample.rollout_replay_refs
                if not refs:
                    raise RuntimeError("Reference mode requires SGLang-published replay manifests")
                validate_sample(refs, sample.tokens)
                for kind, field in FIELDS.items():
                    if field in required_fields and kind not in refs:
                        raise RuntimeError(f"Reference mode is missing {kind} replay")
                values.append(refs)
            return {REFERENCE_FIELD: values}

        unknown = required_fields - frozenset((FIELDS["moe"],))
        if unknown:
            raise KeyError(f"Unsupported replay fields: {sorted(unknown)}")
        fields = {}
        if samples[0].rollout_routed_experts is not None:
            fields["rollout_routed_experts"] = [sample.rollout_routed_experts for sample in samples]
        return fields

    def handles(self, field: str) -> bool:
        return field in ({REFERENCE_FIELD} if self.uses_references else FIELDS.values())

    def pack(self, field: str, values: list[Any]) -> Any:
        self._ensure_open()
        if self.uses_references:
            import torch

            from .ref_data import encode_refs

            if field not in {REFERENCE_FIELD}:
                raise KeyError(field)
            return torch.nested.as_nested_tensor(
                [torch.frombuffer(bytearray(encode_refs(value)), dtype=torch.uint8) for value in values],
                layout=torch.jagged,
            )
        import numpy as np
        import torch

        if field not in frozenset(FIELDS.values()):
            raise KeyError(field)
        # Preserve the original CPU/int32 jagged wire format, including when
        # other TensorDict fields are being packed onto a requested device.
        tensors = [
            torch.from_numpy(np.ascontiguousarray(value.reshape(value.shape[0], -1))).to(torch.int32)
            for value in values
        ]
        return torch.nested.as_nested_tensor(tensors, layout=torch.jagged)

    def receive(self, field: str, values: Iterable[Any], *, device: Any) -> list[Any]:
        self._ensure_open()
        if self.uses_references:
            from .ref_data import decode_refs

            if field not in {REFERENCE_FIELD}:
                raise KeyError(field)
            result = []
            for value in values:
                # Descriptors remain CPU metadata: never copy replay contents to GPU here.
                decoded = decode_refs(value.cpu().numpy().tobytes())
                list(iter_refs(decoded))
                result.append(decoded)
            return result
        import torch
        from tensordict.tensorclass import NonTensorData

        if field not in frozenset(FIELDS.values()):
            raise KeyError(field)
        if field == "rollout_routed_experts":
            device = "cpu"
        dtype = torch.int32
        return [
            torch.as_tensor(value.data if isinstance(value, NonTensorData) else value, dtype=dtype, device=device)
            for value in values
        ]

    def close(self) -> None:
        self._closed = True


@dataclass(frozen=True)
class ReplayRegion:
    manager: "ReplayManager"
    partition: str

    async def put(self, samples: Sequence[Any], write: Callable[[], Awaitable[Any]]) -> None:
        """Hold reference bodies around the caller's queue write.

        Transfer publisher ownership only on success. An exception or
        cancellation keeps the holds because queue delivery may be uncertain.
        """
        self.manager._ensure_open()
        refs = self.manager.sample_refs(samples)
        if not refs:
            await write()
            return
        import asyncio

        from relax.utils.async_utils import finish_before_cancellation

        store = self.manager.store()
        reservation = new_data_id()
        attach = asyncio.create_task(store.invoke_async("attach", refs, self.partition, reservation=reservation))
        try:
            await asyncio.shield(attach)
        except BaseException as error:
            # No sample put was issued. Settle the metadata RPC before removing
            # just this writer's hold, including a lost acknowledgement.
            async def undo() -> None:
                await asyncio.gather(attach, return_exceptions=True)
                await store.invoke_async("cancel_attach", refs, self.partition, reservation)

            try:
                await finish_before_cancellation(undo(), "replay-cancel-attach")
            except Exception as cleanup:
                error.add_note(f"Replay attach rollback failed: {cleanup!r}")
            raise
        await write()
        await finish_before_cancellation(
            store.invoke_async(
                "commit",
                refs,
                self.partition,
                reservation,
                media=_sample_media(samples),
                output_ids=_output_ids(samples),
            ),
            "replay-commit",
        )

    def retire(self) -> None:
        self.manager._ensure_open()
        if self.manager.uses_references:
            self.manager.store().retire_partition(self.partition)

    def reader(self, refs: list[ReplayRef]) -> AbstractContextManager["ConsumerReadSession"]:
        self.manager._ensure_open()
        return self.manager.reader(refs, partition=self.partition)

    def validate(self, batch: dict[str, Any] | None) -> None:
        self.manager._ensure_open()
        if batch is None or REFERENCE_FIELD not in batch:
            return
        refs = [ref for sample_refs in batch[REFERENCE_FIELD] for ref in iter_refs(sample_refs)]
        self.manager.store().validate_partition(refs, self.partition)


class ReplayInventory:
    """Serialized run metadata; tensor bodies are owned by the registered
    transport.

    The deployment adapter hosts this object in one Ray actor. It also works
    in-process for component tests without starting Ray or a second TQ cluster.
    """

    def __init__(self, config: ReplayConfig, delete: Callable) -> None:
        from threading import RLock

        self.run = self.owner = config.run_id
        self.capacity_bytes = config.capacity_bytes
        self._delete = delete
        self._used_bytes = 0
        self._lock = RLock()
        self.state = {"objects": {}, "closed_partitions": [], "closed": False}
        self._media: dict[str, dict] = {}
        self._response_attempts: dict[str, dict[str, ReplayRef]] = {}
        self._engine_publishers: dict[int, str] = {}
        self._retired_publishers: set[tuple[int, str]] = set()

    @contextmanager
    def _state(self, *, write: bool = True):
        with self._lock:
            yield self.state

    def finish_publication(self, record: str, rows: int, token_identity: str, attempt: str | None = None) -> ReplayRef:
        if self.state["closed"] or self.state.get("closing"):
            raise RuntimeError("Replay run is closing")
        item = self.state["objects"].get(record)
        if item is None or item["engine_publisher"] in self._retired_publishers:
            raise RuntimeError("Replay publication no longer exists")
        ref = ReplayRef(
            2,
            self.owner,
            self.run,
            record,
            item["kind"],
            rows,
            item["layers"],
            item["topk"],
            item["chunk_rows"],
            token_identity,
            item.get("codec", "int32"),
        )
        if ref.kind == "mm_embedding":
            media = self._media.get(ref.tokens)
            if media is None or media["record"] != record:
                raise RuntimeError("Media publication no longer owns its claim")
        self.seal(ref)
        if attempt is not None:
            # The HTTP receiver can disappear while the writer is sealing this body.
            holds = self._response_attempts.get(attempt)
            if holds is None:
                self.discard([ref], unadopted=True)
            else:
                holds[record] = ref
        if ref.kind == "mm_embedding":
            # Seal and install the first publisher/cache reader hold in this
            # same inventory transaction, before any other producer can drop it.
            lease = media["initial_lease"]
            if lease not in item["readers"]:
                item["readers"].append(lease)
            item["media_producer_readers"].setdefault(lease, media["publisher"])
            item["publisher"] = False  # The cache lease now protects the sealed body.
        return ref

    def begin(
        self,
        record: str,
        kind: str,
        layers: int,
        topk: int,
        chunk_rows: int,
        codec: str = "int32",
        *,
        media_hash: str | None = None,
        engine_publisher: tuple[int, str] | None = None,
    ) -> None:
        if self.state["closed"] or self.state.get("closing"):
            raise RuntimeError("Replay run is closed")
        if engine_publisher is not None and self._engine_publishers.get(engine_publisher[0]) != engine_publisher[1]:
            raise RuntimeError("Replay publisher generation has been retired")
        if kind == "mm_embedding" and self._media.get(media_hash, {}).get("record") != record:
            raise RuntimeError("Media publication no longer owns its claim")
        if record in self.state["objects"]:
            item = self.state["objects"][record]
            if any(
                item[name] != value
                for name, value in (
                    ("kind", kind),
                    ("layers", layers),
                    ("topk", topk),
                    ("chunk_rows", chunk_rows),
                    ("engine_publisher", engine_publisher),
                )
            ):
                raise ValueError("Replay record already exists with another shape")
            return  # Idempotent metadata retry; never rewrite a body.
        self.state["objects"][record] = dict(
            manifest=None,
            bytes=0,
            ready=False,
            writing=True,
            publisher=True,
            engine_publisher=engine_publisher,
            media_hash=media_hash,
            partitions=[],
            readers=[],
            media_producer_readers={},
            blocks={},
            kind=kind,
            codec=codec,
            layers=layers,
            topk=topk,
            chunk_rows=chunk_rows,
        )

    def reserve_layers(
        self, record: str, chunk: int, layers: tuple[int, ...], count: int, encoded_bytes: int | None = None
    ) -> None:
        item = self.state["objects"][record]
        if not item["writing"] or not item["publisher"] or self.state.get("closing"):
            raise RuntimeError("Replay publication is not writable")
        if (
            not layers
            or len(set(layers)) != len(layers)
            or any(not 0 <= layer < item["layers"] for layer in layers)
            or not 0 < count <= item["chunk_rows"]
            or chunk < 0
        ):
            raise ValueError("Invalid replay block")
        keys = [f"{record}:{chunk}:{layer}" for layer in layers]
        if any(key in item["blocks"] for key in keys):
            raise ValueError("Replay blocks are write-once")
        size = count * item["topk"] * 4 * len(keys)
        if item.get("codec") == "bf16-raw-v1":
            expected = count * item["topk"] * 2 * len(keys)
            if encoded_bytes != expected:
                raise ValueError("Invalid BF16 byte reservation")
            size = expected
        elif item.get("codec", "int32") != "int32":
            from .codecs import replay_codec

            codec = replay_codec(item["codec"])
            maximum = count * codec.max_encoded_width * codec.element_bytes * len(keys)
            if encoded_bytes is None or not 0 < encoded_bytes <= maximum:
                raise ValueError("Invalid encoded replay reservation")
            size = encoded_bytes
        if self._used_bytes + size > self.capacity_bytes:
            raise BufferError("Replay store capacity exhausted during publication")
        # Validate the entire batch before changing inventory.
        item["blocks"].update({key: {"count": count, "ready": False} for key in keys})
        item["bytes"] += size
        self._used_bytes += size

    def written_layers(self, record: str, chunk: int, layers: tuple[int, ...]) -> None:
        item = self.state["objects"][record]
        blocks = item["blocks"]
        selected = [blocks[f"{record}:{chunk}:{layer}"] for layer in layers]
        for block in selected:
            block["ready"] = True
        if not item["publisher"] and all(block["ready"] for block in blocks.values()):
            item["writing"] = False
            self._collect(self.state, (record,))

    def seal(self, ref: ReplayRef) -> None:
        if self.state["closed"] or self.state.get("closing"):
            raise RuntimeError("Replay run is closing")
        item = self.state["objects"][ref.record]
        expected = {
            f"{ref.record}:{chunk}:{layer}": min(ref.chunk_rows, ref.rows - chunk * ref.chunk_rows)
            for chunk in range((ref.rows + ref.chunk_rows - 1) // ref.chunk_rows)
            for layer in range(ref.layers)
        }
        if (
            ref.run != self.run
            or ref.owner != self.owner
            or any(getattr(ref, name) != item[name] for name in ("kind", "layers", "topk", "chunk_rows"))
            or set(item["blocks"]) != set(expected)
            or any(not block["ready"] or block["count"] != expected[key] for key, block in item["blocks"].items())
        ):
            raise RuntimeError("Cannot seal incomplete replay publication")
        item.update(manifest=ref.to_dict(), ready=True, writing=False)

    def abort(self, record: str) -> None:
        item = self.state["objects"].get(record)
        if item is None:
            return
        if item.get("response_owner") or item["readers"] or item["partitions"]:
            raise RuntimeError("Cannot abort adopted replay")
        if any(not block["ready"] for block in item["blocks"].values()):
            raise RuntimeError("Replay write acknowledgement is uncertain; retain publication for recovery")
        item.update(publisher=False, writing=False)
        self._collect(self.state, (record,))

    def authorize_read(self, ref: ReplayRef, lease: str) -> None:
        if lease not in self._object(self.state, ref)["readers"]:
            raise RuntimeError("Replay read requires an active reader lease")

    def authorize_reads(self, refs: list[ReplayRef], leases: list[str]) -> None:
        for ref, lease in zip(refs, leases, strict=True):
            self.authorize_read(ref, lease)

    def _collect(self, state: dict, records: Iterable[str] | None = None) -> None:
        objects = state["objects"]
        collected = {
            record: objects[record]
            for record in (tuple(objects) if records is None else records)
            if record in objects
            and not objects[record].get("output_holders")
            and not any(objects[record][name] for name in ("writing", "publisher", "partitions", "readers"))
        }
        if collected:
            self._delete_collected(state, collected)

    def _delete_collected(self, state: dict, collected: dict[str, dict]) -> None:
        # One transport operation per collection, not one RPC pair per record.
        self._delete([key for item in collected.values() for key in item["blocks"]])
        for record, item in collected.items():
            self._used_bytes -= item["bytes"]
            del state["objects"][record]
            media_hash = item.get("media_hash")
            if media_hash is not None and self._media.get(media_hash, {}).get("record") == record:
                del self._media[media_hash]

    def close(self) -> None:
        self.state["closing"] = True
        # Closing fences reserve/begin first. Acknowledged blocks have no
        # in-flight puts, so an unsealed, abandoned publication can be reclaimed.
        for item in self.state["objects"].values():
            if item["writing"] and all(block["ready"] for block in item["blocks"].values()):
                item["writing"] = False
        if any(
            item["writing"] or item["partitions"] or set(item["readers"]) - item["media_producer_readers"].keys()
            for item in self.state["objects"].values()
        ):
            raise RuntimeError("Replay run still has writers, readers or queue holds")
        for item in self.state["objects"].values():
            # Owning-driver shutdown runs after producer engines are disposed.
            # MM cache/response leases are producer holds, not training readers;
            # SIGKILL cannot run their process-exit cleanup. Never drop leases
            # acquired by generic acquire_many for active training consumers.
            producer_readers = item["media_producer_readers"]
            item["readers"] = [lease for lease in item["readers"] if lease not in producer_readers]
            producer_readers.clear()
            item.pop("output_holders", None)
            item["publisher"] = False
        self._collect(self.state)
        self._response_attempts.clear()
        self.state["closed"] = True

    def snapshot(self) -> dict:
        from copy import deepcopy

        return deepcopy(self.state)

    def _object(self, state: dict, ref: ReplayRef) -> dict:
        if ref.owner != self.owner or ref.run != self.run:
            raise RuntimeError("Replay reference belongs to another replay run; wrong replay registry")
        item = state["objects"].get(ref.record)
        if item is None or not item["ready"] or ref.to_dict() != item["manifest"]:
            raise RuntimeError("Replay reference is missing, not ready, or has different metadata")
        return item

    def adopt_response(self, refs: list[ReplayRef], owner: str) -> None:
        with self._state() as state:
            items = [self._object(state, ref) for ref in refs]
            if any(not item["publisher"] or item.get("response_owner") not in (None, owner) for item in items):
                raise RuntimeError("Replay response already transferred to another owner")
            for item in items:
                item["response_owner"] = owner

    def assign_route_outputs(
        self, refs: list[ReplayRef], holders: dict[str, set[str]], previous: dict[str, set[str]]
    ) -> None:
        """Hand off each publisher or input hold, preserving other outputs
        sharing the record."""
        with self._state() as state:
            items = []
            for ref in refs:
                selected = holders.get(ref.record, set())
                if not selected and ref.record not in state["objects"]:
                    continue  # The generator already discarded this rejected candidate.
                item = self._object(state, ref)
                old = previous.get(ref.record)
                owned = not old.isdisjoint(item.get("output_holders", ())) if old else item["publisher"]
                if selected and not owned:
                    raise RuntimeError("Custom generation no longer owns a selected replay route")
                items.append((item, old, selected))
            for item, old, selected in items:
                owners = item.setdefault("output_holders", set())
                if old:
                    owners.difference_update(old)
                else:
                    item["publisher"] = False
                owners.update(selected)
            self._collect(state, (ref.record for ref in refs))

    def register_engine_publisher(self, slot: int, generation: str) -> None:
        """Fence the replaced engine before reusing its capacity."""
        with self._state() as state:
            if (slot, generation) in self._retired_publishers:
                raise RuntimeError("Replay publisher generation has been retired")
            previous = self._engine_publishers.get(slot)
            self._engine_publishers[slot] = generation
            if previous is not None and previous != generation:
                retired = (slot, previous)
                self._retired_publishers.add(retired)
                for item in state["objects"].values():
                    # A sealed manifest may already belong to a response.
                    if not item["ready"] and item["engine_publisher"] == retired:
                        item["publisher"] = False
                        item["writing"] = not all(block["ready"] for block in item["blocks"].values())
                for digest, media in tuple(self._media.items()):
                    item = state["objects"].get(media["record"])
                    if item is not None and item["ready"]:
                        # Responses may already be in transit. Only leases tied
                        # to the retired engine's local cache can be removed here.
                        for lease, owner in tuple(item["media_producer_readers"].items()):
                            if owner == retired:
                                del item["media_producer_readers"][lease]
                                item["readers"].remove(lease)
                        continue
                    if media["publisher"] != retired:
                        continue
                    del self._media[digest]
                    if item is not None:
                        item["publisher"] = False
                        # Reserved keys may still receive writes. Hold them
                        # until acknowledgement or teardown of the old TQ run.
                        if all(block["ready"] for block in item["blocks"].values()):
                            item["writing"] = False
            self._collect(state)
            if any(item["writing"] and not item["publisher"] for item in state["objects"].values()):
                raise RuntimeError("Retired replay publisher has unacknowledged writes; restart the replay run")
            if any(
                item["ready"]
                and item["publisher"]
                and not item["partitions"]
                and not item["readers"]
                and not item.get("response_owner")
                and item["engine_publisher"] in self._retired_publishers
                for item in state["objects"].values()
            ):
                raise RuntimeError("Retired replay publisher has sealed references without a confirmed handoff")

    def claim_media(
        self,
        image_hash: str,
        grid: list[int],
        record: str,
        modality: str,
        *,
        lease: str | None = None,
        publisher: tuple[int, str] | None = None,
        cache: bool = True,
    ) -> dict:
        """Choose one publisher by image hash, and retain every ready lookup
        atomically."""
        with self._state() as state:
            if state["closed"] or state.get("closing"):
                raise RuntimeError("Replay run is closing")
            if publisher is not None and self._engine_publishers.get(publisher[0]) != publisher[1]:
                raise RuntimeError("Media publisher generation has been retired")
            lease = uuid.uuid4().hex if lease is None else lease
            if not isinstance(lease, str) or len(lease) != 32 or any(c not in "0123456789abcdef" for c in lease):
                raise ValueError("Invalid media reader lease")
            previous = self._media.get(image_hash)
            if previous is None:
                self._media[image_hash] = {
                    "record": record,
                    "grid": grid,
                    "modality": modality,
                    "initial_lease": lease,
                    "publisher": publisher,
                }
                return {"status": "claimed", "lease": lease}
            if previous["grid"] != grid or previous["modality"] != modality:
                raise ValueError("One media hash has incompatible geometry or modality")
            item = state["objects"].get(previous["record"])
            if item is None or not item["ready"]:
                return {"status": "pending"}
            ref = ReplayRef.from_dict(item["manifest"])
            if lease not in item["readers"]:
                item["readers"].append(lease)
            item["media_producer_readers"].setdefault(lease, publisher if cache else None)
            return {"status": "ready", "ref": ref.to_dict(), "lease": lease}

    def open_response_attempt(self, rid: str, abandoned: Sequence[str] = ()) -> None:
        with self._state():
            if rid in abandoned:
                raise ValueError("Cannot reopen an abandoned replay request ID")
            for previous in abandoned:
                self.finish_response_attempt(previous)
            self._response_attempts.setdefault(rid, {})

    def finish_response_attempt(self, rid: str, accepted: Sequence[str] = ()) -> None:
        """Transfer delivered leases and reclaim those whose response was
        lost."""
        with self._state():
            holds = self._response_attempts.get(rid, {})
            unused = holds.keys() - set(accepted)
            routes = [holds[key] for key in unused if holds[key].kind == "moe"]
            media = [key for key in unused if holds[key].kind == "mm_embedding"]
            if routes:
                self.discard(routes, unadopted=True)
            if media:
                self.release_media([holds[key] for key in media], media)
            self._response_attempts.pop(rid, None)

    def lookup_media(
        self,
        image_hash: str,
        grid: list[int],
        modality: str,
        *,
        lease: str | None = None,
        publisher: tuple[int, str] | None = None,
        attempt: str | None = None,
    ) -> dict | None:
        with self._state():
            if attempt is not None and attempt not in self._response_attempts:
                return {"status": "closed"}
            previous = self._media.get(image_hash)
            if previous is None:
                return None
            result = self.claim_media(
                image_hash, grid, previous["record"], modality, lease=lease, publisher=publisher, cache=False
            )
            if result["status"] != "ready":
                return None
            if attempt is not None:
                self._response_attempts[attempt][result["lease"]] = ReplayRef.from_dict(result["ref"])
            return result

    def abort_media_claim(self, image_hash: str, record: str) -> None:
        with self._state() as state:
            previous = self._media.get(image_hash)
            if previous is not None and previous["record"] == record and record not in state["objects"]:
                del self._media[image_hash]

    def release_media(self, refs: list[ReplayRef], leases: list[str]) -> None:
        """Release independent response/cache holds even when another response
        queued the body."""
        with self._state() as state:
            for ref, lease in zip(refs, leases, strict=True):
                if ref.kind != "mm_embedding" or ref.owner != self.owner or ref.run != self.run:
                    raise ValueError("Invalid media publisher release")
                item = state["objects"].get(ref.record)
                if item is not None:
                    item = self._object(state, ref)
                    if lease in item["readers"]:
                        item["readers"].remove(lease)
                    item["media_producer_readers"].pop(lease, None)
                    item["publisher"] = False
            self._collect(state, (ref.record for ref in refs))

    def acquire_many(
        self, refs: list[ReplayRef], *, partition: str | None = None, leases: list[str] | None = None
    ) -> list[str]:
        """Validate first, then register one reader per sample in one
        transaction."""
        leases = [uuid.uuid4().hex for _ in refs] if leases is None else leases
        if len(leases) != len(refs) or any(not lease for lease in leases):
            raise ValueError("Every replay reference requires a nonempty reader identity")
        with self._state() as state:
            items = self._partition_items(state, refs, partition)
            for item, lease in zip(items, leases, strict=True):
                if lease not in item["readers"]:
                    item["readers"].append(lease)
        return leases

    def rollback_acquire(self, leases: list[str]) -> None:
        """Undo this request's generated lease IDs even when input refs were
        invalid.

        An acquire response can be lost after commit. Conversely, validation
        may reject a foreign ref before any lease exists; rollback must not
        validate those refs again or manufacture a permanently pending cleanup.
        """
        identities = set(leases)
        for item in self.state["objects"].values():
            item["readers"] = [lease for lease in item["readers"] if lease not in identities]
            item["media_producer_readers"] = {
                lease: owner for lease, owner in item["media_producer_readers"].items() if lease not in identities
            }
        self._collect(self.state)

    def _partition_items(self, state: dict, refs: list[ReplayRef], partition: str | None) -> list[dict]:
        if (
            state["closed"]
            or state.get("closing")
            or (partition is not None and partition in state["closed_partitions"])
        ):
            raise RuntimeError("Replay consumer requested a closed partition/run")
        items = [self._object(state, ref) for ref in refs]
        if partition is not None and any(partition not in item["partitions"] for item in items):
            raise RuntimeError("Replay reference does not belong to the requested train partition")
        return items

    def validate_partition(self, refs: list[ReplayRef], partition: str) -> None:
        """Early queue check; scoped acquire repeats it atomically with lease
        creation."""
        with self._state(write=False) as state:
            self._partition_items(state, refs, partition)

    def attach(self, refs: list[ReplayRef], partition: str, *, reservation: str | None = None) -> None:
        """Reserve a queue hold before put; retain it on an uncertain put."""
        from .storage import BODY_PARTITION_PREFIX

        if partition.startswith(BODY_PARTITION_PREFIX):
            raise ValueError("Replay sample partition cannot use the body namespace")
        with self._state() as state:
            if partition in state["closed_partitions"] or state["closed"] or state.get("closing"):
                raise RuntimeError("Cannot attach replay to a closed partition/run")
            items = [self._object(state, m) for m in refs]
            for item in items:
                if partition not in item["partitions"]:
                    item["partitions"].append(partition)
                item.setdefault("partition_writers", {}).setdefault(partition, set()).add(reservation)

    def cancel_attach(self, refs: list[ReplayRef], partition: str, reservation: str) -> None:
        for ref in refs:
            item = self.state["objects"].get(ref.record)
            if item is None:
                continue
            writers = item.get("partition_writers", {}).get(partition)
            if writers and reservation in writers:
                writers.remove(reservation)
                if not writers:
                    del item["partition_writers"][partition]
                    item["partitions"].remove(partition)
        self._collect(self.state, (ref.record for ref in refs))

    def commit(
        self,
        refs: list[ReplayRef],
        partition: str | None = None,
        reservation: str | None = None,
        *,
        media: Sequence[dict] = (),
        output_ids: Collection[str] = (),
    ) -> None:
        with self._state() as state:
            items = [self._object(state, m) for m in refs]
            if any(not item["partitions"] and not item.get("retired_partitions") for item in items):
                raise RuntimeError("Cannot commit replay without a queue hold")
            for item in items:
                if partition in item.get("partition_writers", {}):
                    writers = item["partition_writers"][partition]
                    writers.discard(reservation)
                    writers.add(None)  # A confirmed sample keeps the partition hold until retirement.
                if "output_holders" in item:
                    item["output_holders"].difference_update(output_ids)
                item["publisher"] = False
            if media:
                self.release_media(
                    [ReplayRef.from_dict(item["ref"]) for item in media], [item["lease"] for item in media]
                )
            # A successful put can be acknowledged after consumers retire its
            # partition. Reader holds still prevent collection through recompute.
            self._collect(state, (ref.record for ref in refs))

    def discard(
        self,
        refs: list[ReplayRef],
        *,
        unadopted: bool = False,
        owner: str | None = None,
        media: Sequence[dict] = (),
        output_ids: Collection[str] = (),
        keep_queued: bool = False,
    ) -> None:
        """Release known discarded samples before queue ownership transfers."""
        with self._state() as state:
            items = []
            for ref in refs:
                if ref.owner != self.owner or ref.run != self.run:
                    raise RuntimeError("Cannot discard replay from another replay run")
                if ref.record in state["objects"]:
                    item = self._object(state, ref)
                    if unadopted and (item.get("response_owner") or item["partitions"]):
                        continue
                    if owner is not None and item.get("response_owner") not in (None, owner):
                        continue
                    if item["partitions"] and not (output_ids and "output_holders" in item):
                        if keep_queued:
                            continue
                        raise RuntimeError("Cannot discard queued replay; retire its partition instead")
                    items.append(item)
            for item in items:
                if "output_holders" in item:
                    item["output_holders"].difference_update(output_ids)
                item["publisher"] = False
            if media:
                self.release_media(
                    [ReplayRef.from_dict(item["ref"]) for item in media], [item["lease"] for item in media]
                )
            self._collect(state, (ref.record for ref in refs))

    def release_many(self, refs: list[ReplayRef], leases: list[str]) -> None:
        """Idempotent batch release, including retries after partial
        collection."""
        with self._state() as state:
            items = []
            for ref, lease in zip(refs, leases, strict=True):
                if ref.owner != self.owner or ref.run != self.run:
                    raise RuntimeError("Cannot release replay from another replay run")
                item = state["objects"].get(ref.record)
                if item is not None and lease in item["readers"]:
                    items.append((self._object(state, ref), lease))
            for item, lease in items:
                if lease in item["readers"]:
                    item["readers"].remove(lease)
                item["media_producer_readers"].pop(lease, None)
            self._collect(state, (ref.record for ref in refs))

    def retire_partition(self, partition: str) -> None:
        with self._state() as state:
            if partition not in state["closed_partitions"]:
                state["closed_partitions"].append(partition)
            for item in state["objects"].values():
                if partition in item["partitions"]:
                    item["partitions"].remove(partition)
                    item.get("partition_writers", {}).pop(partition, None)
                    item.setdefault("retired_partitions", []).append(partition)
            self._collect(state)


replay_data_manager = ReplayManager(config=ReplayConfig.from_env())


def enable_reference_replay() -> None:
    """Resolve CLI-selected delivery before clients or workers are created."""
    import os

    manager = replay_data_manager
    manager._ensure_open()
    if not manager.uses_references or not manager.config.run_id:
        if (
            manager._enabled_kinds is not None
            or manager._installed_engines
            or manager._store_instance is not None
            or manager._owns_run
        ):
            raise RuntimeError("Cannot enable reference replay after replay manager initialization")
        # Entry points may already hold this singleton; preserve its identity.
        manager.config = replace(manager.config, delivery="reference", run_id=manager.config.run_id or new_data_id())
    os.environ["RELAX_REPLAY_REFERENCES"] = "1"
    os.environ["RELAX_REPLAY_RUN_ID"] = manager.config.run_id


def configure_training_replay(args: Any, runtime_env: dict) -> None:
    """Resolve delivery on the driver and register worker hooks before Ray
    starts."""
    if replay_data_manager.uses_references or default_reference_r3(args):
        enable_reference_replay()
    validate_reference_training(args)
    kinds = rollout_kinds(args)
    replay_data_manager.initialize(kinds=kinds)
    if not isinstance(runtime_env.get("env_vars"), dict):
        runtime_env["env_vars"] = {}
    runtime_env["env_vars"]["RELAX_REPLAY_REGISTRATION"] = json.dumps(replay_data_manager.export_registration())
    # The TQ controller also serves replay publishers. Only shorten empty reads
    # in polling mode, since blocking fixed-size reads share this timeout.
    if kinds and replay_data_manager.uses_references and args.fully_async and not args.hybrid and args.polling_mode:
        from transfer_queue import controller as tq_controller

        # TQ 9784ad0 checks elapsed > timeout. With equal clock readings, zero
        # sleeps for another poll interval; -1 returns the empty batch immediately.
        # Ray serializes the driver's timeout, so set it on both sides.
        tq_controller.TQ_CONTROLLER_GET_METADATA_TIMEOUT = -1
        runtime_env["env_vars"]["TQ_CONTROLLER_GET_METADATA_TIMEOUT"] = "-1"
