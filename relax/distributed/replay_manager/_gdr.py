# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""GPU receive policy for compressed replay bodies, separate from sample TQ.

CPU producers retain host RDMA. Only initialized GPU consumers opt into GDR;
the controller's resolved connection and its CPU/sample clients stay unchanged.
"""

from collections.abc import Callable
from typing import Any


# Bound persistent registered HBM independently of codec and chunk shape.
# TQ splits larger batches to fit, instead of using its 1 GiB default.
_GDR_STAGING_MB = 8


def reader_connection(connection: Any) -> tuple[Any, bool]:
    """Copy the resolved connection before selecting GPU body reception."""
    from copy import deepcopy

    import torch

    backend = connection.backend.storage_backend
    if backend != "MooncakeStore":
        return connection, False
    if connection.backend[backend].protocol != "rdma":
        raise ValueError("GPU replay reception requires the RDMA backend")
    if not torch.cuda.is_initialized():
        raise RuntimeError("Initialize the consuming GPU before constructing a GDR replay client")
    result = deepcopy(connection)
    result.backend[backend].use_gdr = True
    result.backend[backend].gdr_staging_buffer_mb = _GDR_STAGING_MB
    return result, True


def require_gdr_client(client: Any) -> None:
    """Fail closed instead of quietly returning CPU tensors and doing H2D."""
    storage = client.storage_manager.storage_client
    if not getattr(storage, "use_gdr", False) or getattr(storage, "_gdr_staging", None) is None:
        raise RuntimeError("Compressed replay requested GDR but the TQ client did not enable GPU reception")


class GdrBodyReader:
    """Read registered TQ bodies to independent GPU tensors.

    Reuse the pinned manager's key/schema mapping and storage client's checked
    GET implementation. Avoid the generic TensorDict stack, which would copy
    every received body again on the GPU. The existing Mooncake staging unpack
    still makes one bounded compressed D2D copy, never an int32 top-k
    expansion.
    """

    def __init__(self, client_factory: Callable[[], Any], *, partition: str) -> None:
        from threading import RLock

        self._factory = client_factory
        self.partition = partition
        self._client = None
        self._lock = RLock()
        self.read_bytes = self.read_calls = 0
        self.read_seconds = 0.0

    def read(self, keys: list[str]) -> list[Any]:
        from time import perf_counter

        import torch

        with self._lock:
            if self._client is None:
                candidate = self._factory()
                try:
                    require_gdr_client(candidate)
                except BaseException:
                    candidate.close()
                    raise
                self._client = candidate
                import atexit

                atexit.register(self.close)
            client = self._client
            started = perf_counter()
            meta = client.kv_retrieve_meta(keys=keys, partition_id=self.partition, create=False)
            if meta.size != len(keys):
                raise RuntimeError("TQ is missing a sealed GPU replay body")
            meta = meta.select_fields(["replay_body"])
            manager = client.storage_manager
            body_keys = manager._generate_keys(meta.field_names, meta.global_indexes)
            shapes, dtypes, custom = manager._get_shape_type_custom_backend_meta_list(meta)
            if len(body_keys) != len(keys):
                raise RuntimeError("Unexpected TQ GPU replay field mapping")
            values = self._receive(manager.storage_client, body_keys, shapes, dtypes, custom)
            if len(values) != len(keys) or any(
                not isinstance(value, torch.Tensor) or not value.is_cuda or value.dtype != torch.int32
                for value in values
            ):
                raise RuntimeError("TQ GDR replay GET did not return int32 GPU bodies")
            self.read_bytes += sum(value.nbytes for value in values)
            self.read_calls += 1
            self.read_seconds += perf_counter() - started
            return values

    @staticmethod
    def _receive(storage: Any, keys: Any, shapes: Any, dtypes: Any, custom: Any) -> list[Any]:
        import os
        from contextlib import nullcontext

        import torch

        staging = storage._gdr_staging
        if not staging._initialized:
            allocation_context = nullcontext()
            if "torch_memory_saver" in os.environ.get("LD_PRELOAD", ""):
                import ctypes

                active = ctypes.CDLL(None).tms_get_interesting_region
                active.restype = ctypes.c_bool
                if active():
                    from torch_memory_saver import torch_memory_saver

                    allocation_context = torch_memory_saver.disable()
            # Registered physical pages must survive colocate pause/resume.
            # Only the persistent MR is excluded; received tensors stay managed.
            with allocation_context:
                staging.lazy_init(storage._store)
        consumer_stream = torch.cuda.current_stream()
        # Allocate on the stream that performs TQ's raw CUDA D2D unpack.
        # Allocating on the model stream could reuse a block still read there.
        with torch.cuda.stream(staging._stream):
            values = storage.get(keys=keys, shapes=shapes, dtypes=dtypes, custom_backend_meta=custom)
        for value in values:
            if isinstance(value, torch.Tensor) and value.is_cuda:
                value.record_stream(consumer_stream)
        return values

    def close(self) -> None:
        with self._lock:
            if self._client is not None:
                self._client.close()
                self._client = None
