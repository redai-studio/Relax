# Copyright (c) 2026 Relax Authors. All Rights Reserved.


from __future__ import annotations

from examples.chunkflow.state_store import ChunkState, StateStore, StateStoreError


RUNTIME_ATTR = "chunkflow_runtime"
UNIT_ATTR = "chunkflow_unit"
RECOMPUTE_ATTR = "chunkflow_recompute"


class ChunkflowRuntime:
    def __init__(self, chunk_size: int) -> None:
        self.chunk_size = chunk_size
        self._stores: dict[tuple[int, int], StateStore] = {}

    def store(self, layer_number: int, group_id: int) -> StateStore:
        key = (layer_number, group_id)
        store = self._stores.get(key)
        if store is None:
            store = StateStore()
            self._stores[key] = store
        return store

    def release_group(self, group_id: int) -> int:
        keys = [k for k in self._stores if k[1] == group_id]
        for key in keys:
            del self._stores[key]
        return len(keys)

    def history(
        self,
        layer_number: int,
        group_id: int,
        chunk_id: int,
        *,
        recompute: bool = False,
        allow_replay: bool = False,
    ) -> list[ChunkState]:
        """Return earlier states and this chunk's stable cache, in order.

        Only adapters handling native selective checkpointing may allow replay
        without ChunkFlow's explicit recompute marker. Replays never replace
        leaves.
        """
        store = self.store(layer_number, group_id)
        if not recompute:
            if chunk_id == len(store):
                store.put(chunk_id, ChunkState({}))
            elif not allow_replay:
                raise StateStoreError(f"ChunkFlow expected new chunk {len(store)}, got {chunk_id}")
        return store.lookup(chunk_id)

    def live_groups(self) -> list[int]:
        return sorted({group_id for _, group_id in self._stores})

    def assert_empty(self, where: str) -> None:
        if self._stores:
            groups = self.live_groups()
            raise RuntimeError(
                f"chunkflow: {len(self._stores)} store(s) still live after {where} "
                f"(groups {groups}). A dependency group's last backward must call release_group()."
            )


def attach_runtime(model, chunk_size: int) -> ChunkflowRuntime:
    runtime = getattr(model, RUNTIME_ATTR, None)
    if runtime is None:
        runtime = ChunkflowRuntime(chunk_size)
        setattr(model, RUNTIME_ATTR, runtime)
    return runtime


def find_runtime(model) -> ChunkflowRuntime | None:
    seen = 0
    while model is not None and seen < 8:
        runtime = getattr(model, RUNTIME_ATTR, None)
        if runtime is not None:
            return runtime
        model = getattr(model, "module", None)
        seen += 1
    return None


def require_runtime(model) -> ChunkflowRuntime:
    runtime = find_runtime(model)
    if runtime is None:
        raise RuntimeError("chunkflow runtime is missing; install ChunkFlow before selecting its schedule")
    return runtime
