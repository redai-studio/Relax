# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Opt-in native same-snapshot audit, excluded from performance runs."""

from pathlib import Path
from typing import Any

import ray
import torch

from .common import atomic_json, digest


class NativeSnapshotAudit:
    def __init__(self, sender: Any, root: str) -> None:
        self.sender = sender
        self.root = Path(root)
        self.version = None
        self.directory = None
        self.before = self.storage = None

    def _sources(self) -> list:
        backend = self.sender.backend
        return [
            (name, tensor)
            for name, tensor in backend._megatron.named_params_and_buffers(
                backend.args, backend.model, include_persistent_buffers=True
            )
            if backend._should_sync_tensor(name, tensor)
        ]

    @staticmethod
    def _inventory(values: list) -> dict:
        return {name: dict(shape=list(t.shape), dtype=str(t.dtype), sha256=digest(t)) for name, t in values}

    @staticmethod
    def _storage(values: list) -> dict:
        return {name: (t.data_ptr(), t.storage_offset(), t.stride()) for name, t in values}

    def prepare_reference(self) -> None:
        sender, backend = self.sender, self.sender.backend
        self.version = backend.weight_version + 1
        self.directory = self.root / f"version-{self.version}"

        def create_directory() -> None:
            if not self.root.is_dir() or self.root.stat().st_mode & 0o077:
                raise RuntimeError("P2P audit requires an existing private root")
            if sender.rank == 0:
                self.directory.mkdir(mode=0o700, exist_ok=False)

        sender._phase(create_directory)
        values = sender._phase(self._sources)
        self.before = sender._phase(lambda: self._inventory(values))
        self.storage = self._storage(values)

        def pause() -> None:
            if sender.rank == 0:
                sender._fanout("/pause_generation")
                ray.get([engine.flush_cache.remote() for engine in backend.rollout_engines.values()], timeout=300)

        sender._phase(pause)
        # Reuse only the original data path: no version advance or generation
        # resume. All Actor ranks participate in the same native collectives.
        converted, original, size = [], [], 0
        for name, tensor in values:
            if ".experts." not in name:
                size = backend._update_weight_from_distributed(
                    name, tensor, converted, original, size, True, False, pbar=None
                )
        if converted:
            if backend._rollout_bucket_splitter is not None:
                backend._rollout_bucket_splitter(converted, require_complete=True)
            backend._update_bucket_weights_from_distributed(converted, pbar=None)
        converted.clear()
        original.clear()
        backend._run_expert_pass(rollout_only=True, actor_fwd_only=False, pbar=None)
        sender._phase(lambda: self.control("reference_deferred"))

    def control(self, op: str) -> None:
        if self.sender.rank != 0:
            return
        from .common import serialize_control

        empty = torch.empty(0)
        futures = []
        for engine in self.sender.engines:
            command = dict(
                op=op,
                directory=str(self.directory),
                engine=engine,
                version=self.version,
                p2p_directory=str(self.sender.root / str(self.version)),
            )
            blob = serialize_control(command, empty)
            payload = dict(
                serialized_named_tensors=[blob] * self.sender.backend.args.rollout_num_gpus_per_engine,
                load_format="relax.backends.sglang.broadcast_weight_audit.control",
                selector="target",
                flush_cache=False,
                abort_all_requests=False,
            )
            futures.append(
                self.sender.pool.submit(self.sender._request, engine, "/update_weights_from_tensor", payload)
            )
        responses = [future.result() for future in futures]
        if (
            not responses
            or len(responses) != len(self.sender.engines)
            or any(not isinstance(r, dict) or r.get("success") is not True for r in responses)
        ):
            raise RuntimeError(f"P2P snapshot receiver rejected {op}: {responses}")

    def verify_source(self, version: int) -> None:
        if version != self.version:
            raise RuntimeError("P2P audit version changed during transfer")
        values = self._sources()
        after = self._inventory(values)
        storage_unchanged = self.storage == self._storage(values)
        passed = self.before == after and storage_unchanged
        atomic_json(
            self.directory / f"actor-{self.sender.rank}.json",
            dict(
                version=version,
                mode="expert-shard-p2p",
                stage="after-writer-finish-before-commit",
                before=self.before,
                after=after,
                storage_unchanged=storage_unchanged,
                passed=passed,
            ),
        )
        if not passed:
            raise RuntimeError("Native reference or P2P mutated Actor source tensors")
