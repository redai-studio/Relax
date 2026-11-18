# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""TQ body transport; a metadata-only coordinator protects publication and
readers.

Bodies use TQ's explicit-key API, never the training sampler. Each layer/chunk
is a separate value. SimpleStorage returns a whole block; row selection happens
here, not in the storage service. No files or mmap are used.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from threading import RLock, Thread, current_thread
from typing import TYPE_CHECKING, Any, Callable, Iterator

from .ref_data import ReplayRef


if TYPE_CHECKING:
    import numpy as np


_NAMESPACE = "relax-replay"
_BASE_READ_BATCH_BYTES = 64 << 20
BODY_PARTITION_PREFIX = "__replay_body__:"


@contextmanager
def _body_thread_policy(target: int, default: int) -> Iterator[None]:
    """Match TQ's policy on each client's event-loop thread."""
    import torch

    torch.set_num_threads(target)
    try:
        yield
    finally:
        torch.set_num_threads(default)


@dataclass(frozen=True)
class _ReadSelection:
    """Validated source rows and their final output positions for one layer."""

    ref: ReplayRef
    layer: int
    source_rows: np.ndarray
    output_rows: np.ndarray
    output: np.ndarray


@dataclass(frozen=True)
class _RowCopy:
    output: np.ndarray
    output_rows: np.ndarray
    block_rows: np.ndarray


@dataclass
class _ReadBlock:
    """One immutable body key, potentially serving several output
    selections."""

    rows: int
    topk: int
    kind: str
    copies: list[_RowCopy] = field(default_factory=list)


def _validate_read_layers(
    layers: list[tuple[list[tuple[ReplayRef, Any, Any, str | None]], int, Any]],
) -> tuple[list[_ReadSelection], list[tuple[ReplayRef, str | None]]]:
    """Normalize the complete window before authorization or body I/O."""
    import numpy as np

    normalized = []
    authorized = {}
    for requests, layer, output in layers:
        for ref, rows, destinations, lease in requests:
            rows = np.asarray(rows, dtype=np.int64)
            destinations = np.asarray(destinations, dtype=np.int64)
            if (
                rows.ndim != 1
                or destinations.shape != rows.shape
                or not 0 <= layer < ref.layers
                or np.any(rows < 0)
                or np.any(rows >= ref.rows)
                or output.ndim != 2
                or output.shape[1] != ref.topk
                or output.dtype not in (np.int32, np.int64)
                or not output.flags.writeable
                or np.any(destinations < 0)
                or np.any(destinations >= len(output))
            ):
                raise ValueError("Invalid replay read layout")
            normalized.append(_ReadSelection(ref, layer, rows, destinations, output))
            authorized[(ref, lease)] = None
    return normalized, list(authorized)


def _ray() -> Any:
    import ray

    if not ray.is_initialized():
        raise RuntimeError("Replay requires an explicitly connected existing Ray runtime")
    return ray


def _call(actor: Any, method: str, *args: Any, **kwargs: Any) -> Any:
    return _ray().get(getattr(actor, method).remote(*args, **kwargs))


async def _clear_empty_body_partition(client: Any, partition: str) -> None:
    """Remove drained replay metadata, including TQ's falsey empty
    BatchMeta."""
    if not partition.startswith(BODY_PARTITION_PREFIX):
        raise ValueError("Replay cleanup requires a private body partition")
    # The owning registry fences writers and drains all holds BEFORE calling
    # this adapter. Never use metadata deletion to discard remaining bodies.
    metadata = await client._get_partition_meta(partition)
    if metadata is not None:
        if metadata.size:
            raise RuntimeError("Replay body partition is not empty after inventory close")
        # Pinned TQ's public clear_partition returns early for an empty BatchMeta.
        # Reuse its existing acknowledged RPC, on the client's own event loop.
        await client._clear_partition_in_controller(partition)
    if partition in await client.async_get_partition_list():
        raise RuntimeError("Replay body partition metadata remains after close")


def _select_body_rows(field_data: Any, positions: list[int], fallback: Callable) -> Any:
    """Copy strided CPU body slices without scheduling PyTorch's thread
    pool."""
    import numpy as np
    import torch

    if (
        isinstance(field_data, torch.Tensor)
        and not field_data.is_nested
        and field_data.device.type == "cpu"
        and field_data.dtype == torch.int32
        and field_data.ndim == 3
        and len(positions) > 1
        and min(positions) >= 0
        and max(positions) < len(field_data)
    ):
        stride = positions[1] - positions[0]
        if stride > 1 and all(positions[i] - positions[i - 1] == stride for i in range(2, len(positions))):
            return torch.from_numpy(
                np.ascontiguousarray(field_data.numpy()[positions[0] : positions[-1] + 1 : stride])
            )
    return fallback(field_data, positions)


def _pack_body_views(values: list[Any], fallback: Callable) -> Any:
    """Keep received body buffers alive without stacking and immediately
    unbinding."""
    import torch
    from tensordict import NonTensorData, NonTensorStack

    if values and all(
        isinstance(value, torch.Tensor)
        and not value.is_nested
        and value.device.type == "cpu"
        and value.dtype == torch.int32
        and value.ndim == 2
        for value in values
    ):
        # Dynamo may compile model kernels concurrently with this I/O thread.
        # Tensorclass constructors require keyword data in that context.
        return NonTensorStack(*(NonTensorData(data=value) for value in values))
    return fallback(values)


def tq_client_factory(connection: Any, *, body_partition: str) -> Any:
    """Connect sockets to resolved endpoints, never bootstrap a TQ cluster."""
    import psutil
    import torch
    from transfer_queue.client import TQ_NUM_THREADS, TransferQueueClient
    from transfer_queue.storage.managers import simple_storage_manager
    from transfer_queue.utils.common import DEFAULT_TORCH_NUM_THREADS

    if connection is None or not connection.controller.zmq_info:
        raise RuntimeError("Replay needs the resolved TQ connection from Relax initialization")

    # The pinned TQ context queries physical cores on every get/put. Replay clients
    # live within one worker/run, so resolve this CPU topology once per client.
    physical_cores = psutil.cpu_count(logical=False)
    body_threads = (
        min(TQ_NUM_THREADS, physical_cores)
        if isinstance(physical_cores, int) and physical_cores > 0 and TQ_NUM_THREADS > 0
        else None
    )

    class ReplayTQClient(TransferQueueClient):
        async def async_put(
            self,
            data: Any,
            metadata: Any = None,
            partition_id: str | None = None,
            data_parser: Callable | None = None,
            custom_meta: list[dict[str, Any]] | None = None,
            is_last: bool = False,
        ) -> Any:
            # Explicit-key replay bodies need neither insert nor parser/custom
            # metadata handling. Keep all other paths on TQ's implementation.
            body = data.get("replay_body")
            if (
                body_threads is None
                or getattr(self, "storage_manager", None) is None
                or metadata is None
                or not metadata
                or metadata.size == 0
                or data_parser is not None
                or custom_meta is not None
                or is_last
                or set(data.keys()) != {"replay_body"}
                or not isinstance(body, torch.Tensor)
                or body.is_nested
                or body.device.type != "cpu"
                or body.dtype != torch.int32
                or body.ndim != 3
            ):
                return await super().async_put(
                    data,
                    metadata=metadata,
                    partition_id=partition_id,
                    data_parser=data_parser,
                    custom_meta=custom_meta,
                    is_last=is_last,
                )
            with _body_thread_policy(body_threads, DEFAULT_TORCH_NUM_THREADS):
                await self.storage_manager.put_data(data, metadata, data_parser=None)
            await self.async_set_custom_meta(metadata)
            return metadata.add_fields(data)

        async def async_get_data(self, metadata: Any) -> Any:
            # Keep TQ's validation/empty-result behavior and unusual topology
            # fallback. The override is bound by TQ's synchronous constructor.
            if (
                body_threads is None
                or getattr(self, "storage_manager", None) is None
                or not metadata
                or metadata.size == 0
                or len(metadata.field_names) == 0
            ):
                return await super().async_get_data(metadata)
            with _body_thread_policy(body_threads, DEFAULT_TORCH_NUM_THREADS):
                return await self.storage_manager.get_data(metadata)

        async def async_clear_partition(self, partition_id: str) -> None:
            if partition_id.startswith(BODY_PARTITION_PREFIX):
                if partition_id != body_partition:
                    raise ValueError("Replay client cannot finalize another run's body partition")
                await _clear_empty_body_partition(self, partition_id)
            else:
                await super().async_clear_partition(partition_id)

    client = ReplayTQClient(
        client_id=f"replay-{os.getpid()}-{uuid.uuid4().hex}", controller_info=connection.controller.zmq_info
    )
    backend = connection.backend.storage_backend
    try:
        client.initialize_storage_manager(backend, connection.backend[backend])
        if isinstance(client.storage_manager, simple_storage_manager.AsyncSimpleStorageManager):
            select = client.storage_manager._select_by_positions

            def select_body_rows(field_data: Any, positions: list[int]) -> Any:
                return _select_body_rows(field_data, positions, select)

            client.storage_manager._select_by_positions = select_body_rows
            pack = client.storage_manager._pack_field_values

            def pack_body_views(values: list[Any]) -> Any:
                return _pack_body_views(values, pack)

            client.storage_manager._pack_field_values = pack_body_views
    except BaseException:
        client.close()
        raise
    return client


class TQBodyBackend:
    """Process-local client with explicit keys and a dedicated synchronous
    loop."""

    def __init__(self, run: str, client_factory: Callable[[], Any]) -> None:
        self.partition = BODY_PARTITION_PREFIX + run
        self._factory = client_factory
        self._client = None
        self._write_clients: dict[Thread, Any] = {}
        self._pid = os.getpid()
        self._lock = RLock()

    def client(self) -> Any:
        if os.getpid() != self._pid:
            raise RuntimeError("Create replay transport after spawn; do not inherit TQ sockets across fork")
        with self._lock:
            if self._client is None:
                self._client = self._factory()
            return self._client

    def write_many(self, keys: list[str], values: tuple[Any, ...] | np.ndarray) -> None:
        import numpy as np
        import torch
        from tensordict import TensorDict

        if os.getpid() != self._pid:
            raise RuntimeError("Create replay transport after spawn; do not inherit TQ sockets across fork")
        # Sync TQ clients own event loops and socket pools. Parallel writers
        # never share them with each other or with the foreground reader.
        with self._lock:
            thread = current_thread()
            if thread not in self._write_clients:
                self._write_clients[thread] = self._factory()
            client = self._write_clients[thread]
        # Deterministic key is reserved in the inventory BEFORE this call, so a
        # lost allocation/write response cannot lose the only address of a body.
        meta = client.kv_retrieve_meta(keys=keys, partition_id=self.partition, create=True)
        if meta.size != len(keys):
            raise RuntimeError("TQ returned an unexpected replay allocation")
        if (
            isinstance(values, np.ndarray)
            and values.dtype == np.int32
            and values.ndim == 3
            and values.flags.c_contiguous
        ):
            body = torch.from_numpy(values)
        elif len(values) == 1:
            body = torch.from_numpy(values[0]).unsqueeze(0)
        elif all(value.dtype == np.int32 and value.shape == values[0].shape for value in values):
            # Pack homogeneous replay bodies without scheduling PyTorch's CPU
            # thread pool for each small layer batch. The tensor owns a strong
            # reference to the new NumPy allocation until put completes.
            body = torch.from_numpy(np.stack(values))
        else:
            # Preserve the existing conversion/error behavior for other dtypes
            # and heterogeneous shapes.
            body = torch.stack([torch.from_numpy(value) for value in values])
        client.put(TensorDict({"replay_body": body}, batch_size=[len(keys)]), metadata=meta)

    def read(self, keys: list[str]) -> list[Any]:
        import torch

        # Serialize foreground and speculative reads on the same sync client.
        # Client creation alone does not establish caller-thread safety.
        with self._lock:
            client = self.client()
            meta = client.kv_retrieve_meta(keys=keys, partition_id=self.partition, create=False)
            if meta.size != len(keys):
                raise RuntimeError("TQ is missing a sealed replay block")
            meta = meta.select_fields(["replay_body"])
            values = client.get_data(meta)["replay_body"]
            if isinstance(values, torch.Tensor):
                blocks = values.unbind()
            elif isinstance(values, Sequence) and all(
                isinstance(value, torch.Tensor)
                and not value.is_nested
                and value.device.type == "cpu"
                and value.dtype == torch.int32
                and value.ndim == 2
                for value in values
            ):
                blocks = values
            else:
                raise RuntimeError("TQ returned invalid replay body buffers")
            if len(blocks) != len(keys):
                raise RuntimeError("TQ returned an unexpected replay block count")
            # NumPy retains each Tensor, which owns its received transport buffer.
            return [value.cpu().numpy() for value in blocks]

    def delete(self, keys: list[str]) -> None:
        if not keys:
            return
        client = self.client()
        meta = client.kv_retrieve_meta(keys=keys, partition_id=self.partition, create=False)
        if meta.size:
            client.clear_samples(meta)

    def close(self) -> None:
        with self._lock:
            # The owning store drains its writer before closing the transport.
            errors = []
            for thread, client in tuple(self._write_clients.items()):
                try:
                    client.close()
                except Exception as exc:
                    errors.append(exc)
                else:
                    del self._write_clients[thread]
            if self._client is not None:
                try:
                    self._client.close()  # Local sockets only. Never tq.close().
                except Exception as exc:
                    errors.append(exc)
                else:
                    self._client = None
            if errors:
                for extra in errors[1:]:
                    errors[0].add_note(f"Another replay client failed to close: {extra!r}")
                raise errors[0]


def _connection_identity(connection: Any) -> Any:
    """Compare resolved endpoints by value after Ray serializes opaque ZMQ
    info."""
    from collections.abc import Mapping
    from enum import Enum

    from omegaconf import OmegaConf

    if OmegaConf.is_config(connection):
        connection = OmegaConf.to_container(connection, resolve=True)
    if isinstance(connection, Enum):
        return _connection_identity(connection.value)
    if isinstance(connection, Mapping):
        return {key: _connection_identity(value) for key, value in connection.items()}
    if isinstance(connection, (list, tuple)):
        return tuple(_connection_identity(value) for value in connection)
    if hasattr(connection, "to_dict"):
        return _connection_identity(connection.to_dict())
    if connection is None or isinstance(connection, (str, bool, int, float)):
        return connection
    raise TypeError(f"Unsupported replay TQ connection value: {type(connection).__name__}")


def _new_registry(config: Any, driver: str) -> Any:
    # Keep the deployment-specific class out of the core state machine.
    from .core import ReplayInventory

    class Registry(ReplayInventory):
        def __init__(self) -> None:
            self.connection = None
            self.connection_identity = None
            self.driver = driver
            self.backend = TQBodyBackend(
                config.run_id,
                lambda: tq_client_factory(self.connection, body_partition=BODY_PARTITION_PREFIX + config.run_id),
            )
            super().__init__(config, self.backend.delete)

        def configure(self, connection: Any) -> None:
            identity = _connection_identity(connection)
            if self.connection is not None and self.connection_identity != identity:
                raise RuntimeError("Replay TQ endpoints changed within a run; restart recovery is not implemented")
            self.connection = connection
            self.connection_identity = identity

        def get_connection(self) -> Any:
            if self.connection is None:
                raise RuntimeError("Replay TQ connection has not been registered")
            return self.connection

        def finish(self, token: str) -> None:
            if token != self.driver:
                raise RuntimeError("Only the owning driver may close replay")
            self.close()
            if self.connection is not None:
                self.backend.client().clear_partition(self.backend.partition)
            self.backend.close()

    return _ray().remote(num_cpus=0)(Registry).options(name=f"replay-{config.run_id}", namespace=_NAMESPACE).remote()


class TQReplayStorage:
    """Publication/reader facade; payload bytes go directly through TQ clients.

    A caller may inject an in-process coordinator and client factory for CPU
    contract tests. Production always resolves the named run coordinator.
    """

    def __init__(self, config: Any, *, inventory: Any = None, client_factory: Callable | None = None) -> None:
        if not config.run_id or any(c not in "0123456789abcdef" for c in config.run_id):
            raise ValueError("Replay requires a shared hexadecimal run identity")
        self.config = config
        self.run = self.owner = config.run_id
        self.chunk_rows = config.chunk_rows
        self.publication_bytes = config.publication_bytes
        self._inventory = inventory
        self._registry_handle = None
        self._driver = None
        self._writer = None
        self._pid = os.getpid()
        self._backend = TQBodyBackend(self.run, client_factory or self._make_client)
        self._pending_releases = []

    @property
    def read_batch_bytes(self) -> int:
        """Amortize reads when the configured cache can cover their scratch.

        Keep the existing cap for default/smaller caches. Above that, ordinary
        batch scratch (twice the batch size) takes at most half the cache.
        """
        return min(self.publication_bytes, max(_BASE_READ_BATCH_BYTES, self.config.materialization_bytes // 4))

    def _registry(self) -> Any:
        if self._registry_handle is None:
            self._registry_handle = _ray().get_actor(f"replay-{self.run}", namespace=_NAMESPACE)
        return self._registry_handle

    def invoke(self, method: str, *args: Any, **kwargs: Any) -> Any:
        if os.getpid() != self._pid:
            raise RuntimeError("Replay transport must be constructed in the current process")
        if self._inventory is not None:
            with self._inventory._lock:
                return getattr(self._inventory, method)(*args, **kwargs)
        return _call(self._registry(), method, *args, **kwargs)

    def _make_client(self) -> Any:
        return tq_client_factory(self.invoke("get_connection"), body_partition=self._backend.partition)

    def start_run(self) -> None:
        if self._driver is not None:
            raise RuntimeError("Replay run has previous state; only its owning driver may close it")
        token = uuid.uuid4().hex
        if self._inventory is not None:
            self._inventory.state["driver"] = token
        else:
            self._registry_handle = _new_registry(self.config, token)
        self._driver = token

    def register_local_owner(self, connection: Any = None) -> str:
        # Retained integration entry; TQ no longer needs node-local body owners.
        if connection is not None:
            from omegaconf import OmegaConf

            backend = connection.backend.storage_backend
            resolved = OmegaConf.create(
                {
                    "controller": {"zmq_info": connection.controller.zmq_info},
                    "backend": {"storage_backend": backend, backend: connection.backend[backend]},
                },
                flags={"allow_objects": True},
            )
            self.invoke("configure", resolved)
        else:
            self.invoke("get_connection")
        return self.owner

    def begin_publication(self, **kwargs: Any) -> Any:
        from .producer import ProducerChunkWriter, ProducerPublication

        if self._writer is None:
            self._writer = ProducerChunkWriter(self.publication_bytes)
        return ProducerPublication(self, self._writer, **kwargs)

    def write_blocks(
        self, record: str, chunk: int, layers: tuple[int, ...], values: tuple[Any, ...] | np.ndarray
    ) -> None:
        if not layers or len(layers) != len(values) or any(value.shape != values[0].shape for value in values):
            raise ValueError("Replay batch requires equally shaped layer blocks")
        self.invoke("reserve_layers", record, chunk, layers, len(values[0]))
        self._backend.write_many([f"{record}:{chunk}:{layer}" for layer in layers], values)
        try:
            self.invoke("written_layers", record, chunk, layers)
        except Exception:
            # Only metadata acknowledgement is idempotent; never retry an
            # ambiguous payload put. Unacknowledged blocks remain inventoried.
            self.invoke("written_layers", record, chunk, layers)

    def remap_tokens(self, ref: ReplayRef, captured: list[int], training: list[int]) -> ReplayRef:
        return self.invoke("remap_tokens", ref, captured, training)

    def attach(self, refs: list[ReplayRef], partition: str) -> None:
        if partition.startswith(BODY_PARTITION_PREFIX):
            raise ValueError("Replay sample partition cannot use the body namespace")
        self.invoke("attach", refs, partition)

    def commit(self, refs: list[ReplayRef]) -> None:
        self.invoke("commit", refs)

    def discard(self, refs: list[ReplayRef]) -> None:
        self.invoke("discard", refs)

    def validate_partition(self, refs: list[ReplayRef], partition: str) -> None:
        self.invoke("validate_partition", refs, partition)

    def acquire_many(self, refs: list[ReplayRef], *, partition: str | None = None) -> list[str]:
        leases = [uuid.uuid4().hex for _ in refs]
        try:
            self.invoke("acquire_many", refs, partition=partition, leases=leases)
        except BaseException as primary:
            try:
                self.invoke("rollback_acquire", leases)
            except Exception as cleanup:
                self._pending_releases.append(leases)
                primary.add_note(f"Replay lease rollback pending: {cleanup!r}")
            raise
        return leases

    def release_many(self, refs: list[ReplayRef], leases: list[str]) -> None:
        self.invoke("release_many", refs, leases)

    @contextmanager
    def reader(self, refs: list[ReplayRef], *, partition: str | None = None) -> Iterator[Any]:
        from .consumer import ConsumerReadSession

        session = ConsumerReadSession(self, refs, partition=partition)
        try:
            yield session
        except BaseException as primary:
            try:
                session.release()
            except Exception as cleanup:
                raise primary from cleanup
            raise
        else:
            session.release()

    def _read_into(
        self, ref: ReplayRef, layer: int, rows: Any, output: Any, destinations: Any, lease: str | None = None
    ) -> None:
        self._read_many_into([(ref, rows, destinations, lease)], layer, output)

    def _read_many_into(self, requests: list[tuple[ReplayRef, Any, Any, str | None]], layer: int, output: Any) -> None:
        self._read_layers_into([(requests, layer, output)])

    def _read_layers_into(
        self,
        layers: list[tuple[list[tuple[ReplayRef, Any, Any, str | None]], int, Any]],
        *,
        on_layer: Callable[[int], None] | None = None,
    ) -> None:
        """Read local layers with bounded, explicit-key TQ batches.

        Requests come from disjoint sample positions in ConsumerBatchLayout.
        Validate the entire window before I/O. Each distinct reference/lease is
        authorized once, including different leases for duplicate references.
        on_layer receives an output's position after all its rows are written.
        The caller must still observe completion/failure of the entire read.
        """
        import numpy as np

        normalized, identities = _validate_read_layers(layers)
        positions = {id(output): index for index, (_, _, output) in enumerate(layers)}
        if on_layer is not None and (
            len(positions) != len(layers)
            or any(
                np.shares_memory(output, other)
                for index, (_, _, output) in enumerate(layers)
                for _, _, other in layers[:index]
            )
        ):
            raise ValueError("Progressive replay reads require non-overlapping layer outputs")
        if not normalized:
            if on_layer is not None:
                for index in range(len(layers)):
                    on_layer(index)
            return
        if len(identities) == 1:
            self.invoke("authorize_read", *identities[0])
        else:
            self.invoke("authorize_reads", [ref for ref, _ in identities], [lease for _, lease in identities])

        pending: dict[str, _ReadBlock] = {}
        pending_bytes = 0
        remaining = dict.fromkeys(positions, 0)
        enumerated = {id(output) for requests, _, output in layers if not requests}
        published: set[int] = set()

        def publish_ready() -> None:
            if on_layer is not None:
                for identity in positions:
                    if identity in enumerated and not remaining[identity] and identity not in published:
                        on_layer(positions[identity])
                        published.add(identity)

        publish_ready()

        def flush() -> None:
            blocks = self._backend.read(list(pending))
            for plan, block in zip(pending.values(), blocks, strict=True):
                if block.dtype != np.int32:
                    raise RuntimeError("TQ replay body shape/dtype differs from manifest")
                if block.shape != (plan.rows, plan.topk):
                    if plan.kind != "indexer" or plan.topk <= 1 or block.shape != (plan.rows, (plan.topk + 1) // 2):
                        raise RuntimeError("TQ replay body shape/dtype differs from manifest")
                    block = np.ascontiguousarray(block, dtype="<i4").view("<i2")[:, : plan.topk]
                for copy in plan.copies:
                    source, destination = copy.block_rows, copy.output_rows
                    if len(source) and np.all(np.diff(source) == 1) and np.all(np.diff(destination) == 1):
                        # Basic slices are views: avoid the full temporary
                        # produced by advanced indexing for contiguous CP rows.
                        np.copyto(
                            copy.output[int(destination[0]) : int(destination[-1]) + 1],
                            block[int(source[0]) : int(source[-1]) + 1],
                        )
                    else:
                        copy.output[destination] = block[source]
                    if on_layer is not None:
                        remaining[id(copy.output)] -= 1
            pending.clear()
            publish_ready()

        for index, selection in enumerate(normalized):
            ref = selection.ref
            chunks = selection.source_rows // ref.chunk_rows
            offsets = selection.source_rows % ref.chunk_rows
            for chunk in np.unique(chunks):
                key = f"{ref.record}:{chunk}:{selection.layer}"
                count = min(ref.chunk_rows, ref.rows - int(chunk) * ref.chunk_rows)
                if key not in pending:
                    size = count * ref.topk * 4
                    if pending and pending_bytes + size > self.read_batch_bytes:
                        flush()
                        pending_bytes = 0
                    # As in single-sample reads, one indivisible chunk may
                    # exceed the budget; no additional chunk joins that call.
                    pending[key] = _ReadBlock(count, ref.topk, ref.kind)
                    pending_bytes += size
                indexes = np.flatnonzero(chunks == chunk)
                pending[key].copies.append(
                    _RowCopy(selection.output, selection.output_rows[indexes], offsets[indexes])
                )
                if on_layer is not None:
                    remaining[id(selection.output)] += 1
            if on_layer is not None and (
                index + 1 == len(normalized) or normalized[index + 1].output is not selection.output
            ):
                enumerated.add(id(selection.output))
                publish_ready()
        if pending:
            flush()

    def retire_partition(self, partition: str) -> None:
        self.invoke("retire_partition", partition)

    def reclaim_stopped_publishers(self) -> int:
        # Never infer remote process death from a local PID. Uncertain writes
        # remain visible for recovery instead of being deleted by a timeout.
        return len(self.invoke("snapshot")["objects"])

    def close_run(self) -> None:
        if self._writer is not None:
            self._writer.close()
        pending, self._pending_releases = self._pending_releases, []
        for leases in pending:
            try:
                self.invoke("rollback_acquire", leases)
            except Exception:
                self._pending_releases.append(leases)
        if self._pending_releases:
            raise RuntimeError("Replay lease rollback remains incomplete")
        if self._inventory is not None:
            self.invoke("close")
        else:
            if self._driver is None:
                raise RuntimeError("This process does not own replay run shutdown")
            self.invoke("finish", self._driver)
            _ray().kill(self._registry(), no_restart=True)
            self._driver = None
        self._backend.close()
