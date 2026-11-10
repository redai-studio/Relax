# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Replay manager: wire data, train-region ownership and local consumer
state."""

import importlib
import uuid
from collections.abc import Callable, Iterable, Iterator, MutableMapping, Sequence
from contextlib import AbstractContextManager, contextmanager
from dataclasses import asdict, dataclass, field, replace
from functools import wraps
from typing import TYPE_CHECKING, Any

from .config import ReplayConfig, get_store
from .consumer import (
    ConsumerChannelKey,
    ConsumerExecutor,
    ConsumerRecord,
    ConsumerRequest,
    ConsumerSource,
)
from .ref_data import FIELDS, REFERENCE_FIELD, ReplayRef, register_kind, token_digest


if TYPE_CHECKING:
    from .consumer import Consumer, ConsumerReadSession
    from .producer import Producer
    from .storage import TQReplayStorage


def refs_from_samples(samples: list[Any]) -> list[ReplayRef]:
    return [ReplayRef.from_dict(item) for sample in samples for item in (sample.rollout_replay_refs or {}).values()]


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
    entries: list[Any] = field(default_factory=list)
    microbatch_ids: dict[int, int] = field(default_factory=dict)
    forward_index: int = 0
    backward_index: int = 0

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
    """Importable adapters and engine patches for one registered replay
    kind."""

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
        for kind in ("moe", "indexer")
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
        phase: ConsumerPhase | None = None,
        store_factory: Callable[[], "TQReplayStorage"] | None = None,
    ) -> None:
        self.config = config if config is not None else ReplayConfig()
        self.phase = phase if phase is not None else ConsumerPhase()
        self._channels: dict[ConsumerChannelKey, ConsumerChannel] = {}
        self._closed = False
        self._store_factory = store_factory
        self._store_instance: TQReplayStorage | None = None
        self._owns_run = False
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
        """Called by Relax's training entry point, before constructing
        workers."""
        if self.uses_references:
            self.store().start_run()
            self._owns_run = True

    def finish_run(self) -> None:
        """Called by standard normal/error shutdown, never by an in-job
        restart."""
        if self._owns_run:
            self.store().close_run()
            self._owns_run = False

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

    def register_transport(self, factory: Callable[[], "TQReplayStorage"]) -> None:
        """Compatibility name for register_storage."""
        self.register_storage(factory)

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

    def sample_refs(self, samples: Sequence[Any]) -> list["ReplayRef"]:
        return refs_from_samples(samples) if self.uses_references else []

    def discard(self, samples: Sequence[Any]) -> None:
        refs = self.sample_refs(samples)
        if refs:
            self.store().discard(refs)

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
        import numpy as np

        unknown = required_fields - frozenset((FIELDS["moe"], FIELDS["indexer"]))
        if unknown:
            raise KeyError(f"Unsupported replay fields: {sorted(unknown)}")
        fields = {}
        if samples[0].rollout_routed_experts is not None:
            fields["rollout_routed_experts"] = [sample.rollout_routed_experts for sample in samples]
        if "rollout_indexer_topk" in required_fields:
            values = [getattr(sample, "rollout_indexer_topk", None) for sample in samples]
            if any(value is None for value in values):
                raise RuntimeError("Every training sample must contain rollout_indexer_topk when replay is enabled")
            widths = set()
            for sample, value in zip(samples, values, strict=False):
                if not isinstance(value, np.ndarray):
                    raise TypeError(f"rollout_indexer_topk must be a numpy array, got {type(value)}")
                if value.ndim != 2 or value.shape[0] != len(sample.tokens) - 1:
                    raise RuntimeError(
                        "rollout_indexer_topk must be a 2-D array with one row per input token except the last, "
                        f"got replay={value.shape}, tokens={len(sample.tokens)}"
                    )
                if value.dtype != np.int32:
                    raise RuntimeError(f"rollout_indexer_topk must use int32, got {value.dtype}")
                widths.add(value.shape[1])
            if len(widths) != 1:
                raise RuntimeError(f"Inconsistent rollout_indexer_topk widths in one batch: {sorted(widths)}")
            fields["rollout_indexer_topk"] = values
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
                for item in decoded.values():
                    ReplayRef.from_dict(item)
                result.append(decoded)
            return result
        import torch
        from tensordict.tensorclass import NonTensorData

        if field not in frozenset(FIELDS.values()):
            raise KeyError(field)
        dtype = torch.long if field == "rollout_routed_experts" else torch.int32
        return [
            torch.as_tensor(value.data if isinstance(value, NonTensorData) else value, dtype=dtype, device=device)
            for value in values
        ]

    def register_channel(self, key: ConsumerChannelKey) -> ConsumerChannel:
        self._ensure_open()
        if key in self._channels:
            raise ValueError(f"Replay channel already registered: {key}")
        channel = ConsumerChannel()
        self._channels[key] = channel
        return channel

    def _channel(self, key: ConsumerChannelKey) -> ConsumerChannel:
        self._ensure_open()
        return self._channels[key]

    def bind(self, key: ConsumerChannelKey, record: ConsumerRecord) -> None:
        """Bind an already laid-out layer/microbatch to its consumer
        channel."""
        channel = self._channel(key)
        if record.kind != key.kind or record.microbatch_id != len(channel.entries):
            raise ValueError("Replay kind or microbatch order does not match the channel")
        channel.microbatch_ids[len(channel.entries)] = record.microbatch_id
        channel.entries.append(record.payload)

    def consume(
        self,
        key: ConsumerChannelKey,
        phase: str | None = None,
        *,
        microbatch_id: int | None = None,
        advance_before_lookup: bool = False,
    ) -> Any:
        channel = self._channel(key)
        if phase is None:
            phase = self.phase.get()
        if phase not in ("replay_forward", "replay_backward"):
            raise ValueError(f"Cannot consume replay in stage {phase!r}")
        cursor = "forward_index" if phase == "replay_forward" else "backward_index"
        index = getattr(channel, cursor)
        if microbatch_id is not None:
            recorded_id = channel.microbatch_ids.get(index)
            if recorded_id is None or recorded_id != microbatch_id:
                raise RuntimeError(f"Replay microbatch mismatch: expected={microbatch_id}, cursor={index}")
        # Legacy Indexer advances even on overrun; MoE advances only on success.
        if advance_before_lookup:
            setattr(channel, cursor, index + 1)
        payload = ConsumerExecutor.execute(
            ConsumerRequest(kind=key.kind, layer=key.layer, microbatch_id=index, payload=channel.entries[index])
        )
        if not advance_before_lookup:
            setattr(channel, cursor, index + 1)
        return payload

    def reset_forward(self, key: ConsumerChannelKey, *, require_consumed: bool = False) -> None:
        channel = self._channel(key)
        if require_consumed and channel.forward_index != len(channel.entries):
            raise ConsumerConsumptionError("replay_forward", channel.forward_index, len(channel.entries))
        channel.forward_index = 0

    def check_consumed(self, key: ConsumerChannelKey, *, expect_backward: bool = False) -> None:
        channel = self._channel(key)
        cursors = ("forward_index", "backward_index") if expect_backward else ("forward_index",)
        for cursor in cursors:
            count = getattr(channel, cursor)
            if count != len(channel.entries):
                phase = "replay_forward" if cursor == "forward_index" else "replay_backward"
                raise ConsumerConsumptionError(phase, count, len(channel.entries))

    def clear(self) -> None:
        error = None
        for channel in self._channels.values():
            try:
                channel.clear()
            except Exception as exc:
                error = error or exc
        if error is not None:
            raise error

    def release_sources(self) -> None:
        """Release bound leases without clearing externally held inline
        views."""
        error = None
        for channel in self._channels.values():
            try:
                channel.release_sources()
            except Exception as exc:
                error = error or exc
        if error is not None:
            raise error

    def close(self) -> None:
        self.clear()
        self._channels.clear()
        self._closed = True


@dataclass(frozen=True)
class ReplayRegion:
    manager: "ReplayManager"
    partition: str

    @contextmanager
    def queue_ownership(self, samples: Sequence[Any]) -> Iterator[None]:
        """Hold reference bodies around the caller's queue write.

        Transfer publisher ownership only on success. An exception or
        cancellation keeps the holds because queue delivery may be uncertain.
        """
        self.manager._ensure_open()
        refs = self.manager.sample_refs(samples)
        if refs:
            self.manager.store().attach(refs, self.partition)
        yield
        if refs:
            self.manager.store().commit(refs)

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
        refs = [ReplayRef.from_dict(item) for sample_refs in batch[REFERENCE_FIELD] for item in sample_refs.values()]
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
        self._lock = RLock()
        self.state = {"objects": {}, "closed_partitions": [], "closed": False}

    @contextmanager
    def _state(self, *, write: bool = True):
        with self._lock:
            yield self.state

    def finish_publication(self, record: str, rows: int, token_identity: str) -> ReplayRef:
        if self.state["closed"] or self.state.get("closing"):
            raise RuntimeError("Replay run is closing")
        item = self.state["objects"][record]
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
        )
        self.seal(ref)
        return ref

    def begin(self, record: str, kind: str, layers: int, topk: int, chunk_rows: int) -> None:
        if self.state["closed"] or self.state.get("closing"):
            raise RuntimeError("Replay run is closed")
        if record in self.state["objects"]:
            item = self.state["objects"][record]
            if any(
                item[name] != value
                for name, value in (("kind", kind), ("layers", layers), ("topk", topk), ("chunk_rows", chunk_rows))
            ):
                raise ValueError("Replay record already exists with another shape")
            return  # Idempotent metadata retry; never rewrite a body.
        self.state["objects"][record] = dict(
            manifest=None,
            bytes=0,
            ready=False,
            writing=True,
            publisher=True,
            partitions=[],
            readers=[],
            blocks={},
            kind=kind,
            layers=layers,
            topk=topk,
            chunk_rows=chunk_rows,
        )

    def reserve_layers(self, record: str, chunk: int, layers: tuple[int, ...], count: int) -> None:
        item = self.state["objects"][record]
        if not item["writing"] or self.state.get("closing"):
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
        used = sum(obj["bytes"] for obj in self.state["objects"].values())
        if used + size > self.capacity_bytes:
            raise BufferError("Replay store capacity exhausted during publication")
        # Validate the entire batch before changing inventory.
        item["blocks"].update({key: {"count": count, "ready": False} for key in keys})
        item["bytes"] += size

    def written_layers(self, record: str, chunk: int, layers: tuple[int, ...]) -> None:
        blocks = self.state["objects"][record]["blocks"]
        selected = [blocks[f"{record}:{chunk}:{layer}"] for layer in layers]
        for block in selected:
            block["ready"] = True

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
        if item["readers"] or item["partitions"]:
            raise RuntimeError("Cannot abort adopted replay")
        if any(not block["ready"] for block in item["blocks"].values()):
            raise RuntimeError("Replay write acknowledgement is uncertain; retain publication for recovery")
        item.update(publisher=False, writing=False)
        self._collect(self.state)

    def authorize_read(self, ref: ReplayRef, lease: str) -> None:
        if lease not in self._object(self.state, ref)["readers"]:
            raise RuntimeError("Replay read requires an active reader lease")

    def authorize_reads(self, refs: list[ReplayRef], leases: list[str]) -> None:
        for ref, lease in zip(refs, leases, strict=True):
            self.authorize_read(ref, lease)

    def _collect(self, state: dict) -> None:
        for record, item in list(state["objects"].items()):
            if not item["writing"] and not item["publisher"] and not item["partitions"] and not item["readers"]:
                self._delete(list(item["blocks"]))
                del state["objects"][record]

    def close(self) -> None:
        self.state["closing"] = True
        # Closing fences reserve/begin first. Acknowledged blocks have no
        # in-flight puts, so an unsealed, abandoned publication can be reclaimed.
        for item in self.state["objects"].values():
            if item["writing"] and all(block["ready"] for block in item["blocks"].values()):
                item["writing"] = False
        if any(item["writing"] or item["readers"] or item["partitions"] for item in self.state["objects"].values()):
            raise RuntimeError("Replay run still has writers, readers or queue holds")
        for item in self.state["objects"].values():
            item["publisher"] = False
        self._collect(self.state)
        self.state["closed"] = True

    def snapshot(self) -> dict:
        from copy import deepcopy

        return deepcopy(self.state)

    def _object(self, state: dict, ref: ReplayRef) -> dict:
        if ref.owner != self.owner or ref.run != self.run:
            raise RuntimeError("Replay reference belongs to another replay run; wrong replay registry")
        item = state["objects"].get(ref.record)
        if item is None or not item["ready"] or ref.to_dict() not in (item["manifest"], item.get("capture_manifest")):
            raise RuntimeError("Replay reference is missing, not ready, or has different metadata")
        return item

    def remap_tokens(self, ref: ReplayRef, captured: list[int], training: list[int]) -> ReplayRef:
        """Bind a validated, length-preserving token normalization before TQ.

        Body rows stay unchanged. Preserve the capture identity so failed/lost
        acknowledgements can still discard or retry using the original ref. A
        record accepts only one training identity, never arbitrary rebinding.
        """
        if len(captured) != ref.rows + 1 or len(training) != len(captured) or token_digest(captured) != ref.tokens:
            raise ValueError("Replay token mapping does not match captured identity or row count")
        mapped = replace(ref, tokens=token_digest(training))
        with self._state() as state:
            item = self._object(state, ref)
            if not item["publisher"] or item["partitions"] or item["readers"] or item.get("retired_partitions"):
                raise RuntimeError("Replay token mapping must precede queue and reader ownership")
            if item.get("capture_manifest") is not None:
                if item["capture_manifest"] != ref.to_dict() or item["manifest"] != mapped.to_dict():
                    raise RuntimeError("Replay training token identity is already bound")
            elif mapped != ref:
                item["capture_manifest"] = ref.to_dict()
                item["manifest"] = mapped.to_dict()
        return mapped

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

    def attach(self, refs: list[ReplayRef], partition: str) -> None:
        """Reserve a queue hold before put; retain it on an uncertain put."""
        with self._state() as state:
            if partition in state["closed_partitions"] or state["closed"] or state.get("closing"):
                raise RuntimeError("Cannot attach replay to a closed partition/run")
            items = [self._object(state, m) for m in refs]
            for item in items:
                if partition not in item["partitions"]:
                    item["partitions"].append(partition)

    def commit(self, refs: list[ReplayRef]) -> None:
        with self._state() as state:
            items = [self._object(state, m) for m in refs]
            if any(not item["partitions"] and not item.get("retired_partitions") for item in items):
                raise RuntimeError("Cannot commit replay without a queue hold")
            for item in items:
                item["publisher"] = False
            # A successful put can be acknowledged after consumers retire its
            # partition. Reader holds still prevent collection through recompute.
            self._collect(state)

    def discard(self, refs: list[ReplayRef]) -> None:
        """Release known discarded samples before queue ownership transfers."""
        with self._state() as state:
            items = []
            for ref in refs:
                if ref.owner != self.owner or ref.run != self.run:
                    raise RuntimeError("Cannot discard replay from another replay run")
                if ref.record in state["objects"]:
                    item = self._object(state, ref)
                    if item["partitions"]:
                        raise RuntimeError("Cannot discard queued replay; retire its partition instead")
                    items.append(item)
            for item in items:
                item["publisher"] = False
            self._collect(state)

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
            self._collect(state)

    def retire_partition(self, partition: str) -> None:
        with self._state() as state:
            if partition not in state["closed_partitions"]:
                state["closed_partitions"].append(partition)
            for item in state["objects"].values():
                if partition in item["partitions"]:
                    item["partitions"].remove(partition)
                    item.setdefault("retired_partitions", []).append(partition)
            self._collect(state)


replay_data_manager = ReplayManager(config=ReplayConfig.from_env())


def install_tq_cleanup_hook() -> None:
    """Link TQ partition cleanup to Replay once, before clients are created.

    TQ's synchronous client binds this same async method at construction.
    Partial sample deletion is not a partition boundary and stays unchanged.
    """
    if not replay_data_manager.uses_references:
        return

    from transfer_queue.client import AsyncTransferQueueClient

    original = AsyncTransferQueueClient.async_clear_partition
    if getattr(original, "_relax_replay_cleanup", False):
        return

    @wraps(original)
    async def clear_partition(client: Any, partition_id: str) -> Any:
        # A failed/cancelled TQ clear must keep its reference bodies live.
        result = await original(client, partition_id=partition_id)
        # Retirement removes the partition hold, never an active reader lease.
        from .storage import BODY_PARTITION_PREFIX

        if not partition_id.startswith(BODY_PARTITION_PREFIX):
            replay_data_manager.region(partition_id).retire()
        return result

    clear_partition._relax_replay_cleanup = True
    AsyncTransferQueueClient.async_clear_partition = clear_partition
