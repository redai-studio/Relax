# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Async I/O around the serial version manager, without changing rollout
routing."""

import asyncio
from typing import TYPE_CHECKING

from .artifact import AdapterArtifact
from .identity import AdapterSnapshot
from .manager import CleanupObservation, LoRAVersionManager, Publication, PublicationError, Replica, VersionState


if TYPE_CHECKING:
    from relax.backends.sglang.lora_client import SGLangLoRAClient


class LoRAPublisher:
    def __init__(
        self, manager: LoRAVersionManager, engines: dict[str, "SGLangLoRAClient"], probe_input_ids: list[int]
    ) -> None:
        if not probe_input_ids:
            raise ValueError("a nonempty model-valid generation probe is required")
        self.manager = manager
        self.engines = dict(engines)
        self.probe_input_ids = list(probe_input_ids)
        self.snapshots: dict[int, AdapterSnapshot | AdapterArtifact] = {}
        self._publish_lock = asyncio.Lock()
        self._reconcile_lock = asyncio.Lock()

    @classmethod
    async def connect(
        cls, engines: dict[str, "SGLangLoRAClient"], probe_input_ids: list[int], max_versions: int = 2
    ) -> "LoRAPublisher":
        inventories = await asyncio.gather(*(client.inventory() for client in engines.values()))
        if any(row["max_loras_per_batch"] < max_versions + 1 for row in inventories):
            raise ValueError("managed LoRA requires max_loras_per_batch >= max_versions + 1")
        if any(row["adapters"] or row["fences"] for row in inventories):
            raise ValueError("connect requires fresh engines; an existing publisher must retain its metadata")
        manager = LoRAVersionManager(
            {name: row["instance_id"] for name, row in zip(engines, inventories)}, max_versions
        )
        return cls(manager, engines, probe_input_ids)

    async def _verify(self, publication: Publication, engine_id: str, instance_id: str) -> None:
        snapshot = self.snapshots[publication.identity.version_id]
        client = self.engines[engine_id]
        receipt = await client.load(snapshot, instance_id, publication.attempt_id)
        if (
            receipt["instance_id"] != instance_id
            or receipt["attempt_id"] != publication.attempt_id
            or receipt["lora_name"] != snapshot.identity.lora_name
            or receipt["digest"] != snapshot.identity.digest
            or receipt["phase"] != "loaded"
            or receipt["admission_closed"]
            or receipt["attempt_fenced"]
            or not receipt["pinned"]
            or receipt["fingerprints"] != snapshot.fingerprints
        ):
            raise PublicationError("REPLICA_INVALID", f"invalid load receipt from {engine_id}")
        await client.probe(snapshot.identity.lora_name, self.probe_input_ids)
        # Recheck process and ID after the probe, including a restart at the same URL.
        inventory = await client.inventory()
        if inventory["instance_id"] != instance_id or not any(
            row["lora_id"] == receipt["lora_id"]
            and row["attempt_id"] == publication.attempt_id
            and row["phase"] == "loaded"
            and not row["admission_closed"]
            and not row["attempt_fenced"]
            for row in inventory["adapters"]
        ):
            raise PublicationError("REPLICA_INVALID", f"replica changed during probe on {engine_id}")
        if not self.manager.record_verified(
            snapshot.identity.version_id,
            publication.attempt_id,
            Replica(engine_id, instance_id, receipt["lora_id"]),
            receipt["fingerprints"],
            pinned=receipt["pinned"],
        ):
            raise PublicationError("STALE_ATTEMPT", "publication changed while verifying")

    async def publish(self, snapshot: AdapterSnapshot | AdapterArtifact) -> Publication:
        """Reserve -> load every engine -> checksum/probe -> single commit.

        Cancellation fences via subsequent reconciliation. The manager retains
        occupied capacity and this object retains snapshot storage meanwhile.
        The runtime should call reconcile periodically, including after errors.
        File artifacts must remain available through recovery and cleanup; this
        publisher drops metadata after reclamation but never deletes shared
        files.
        """
        async with self._publish_lock:
            reservation = self.manager.reserve(snapshot.identity, snapshot.fingerprints)
            publication = reservation.publication
            if not reservation.should_load:
                return publication
            self.snapshots[snapshot.identity.version_id] = snapshot
            try:
                results = await asyncio.gather(
                    *(self._verify(publication, engine, instance) for engine, instance in publication.targets),
                    return_exceptions=True,
                )
                for result in results:
                    if isinstance(result, BaseException):
                        raise result
                return self.manager.commit(
                    snapshot.identity.version_id, publication.attempt_id, publication.expected_default
                )
            except BaseException:
                self.manager.cancel(snapshot.identity.version_id, publication.attempt_id)
                await self.reconcile()
                raise

    async def reconcile(self) -> dict[str, Exception]:
        """One bounded pass; unreachable/busy engines retain capacity for
        retry.

        Fence is also an inventory observation. Never infer absence from a
        missing load ACK, timeout, or an unregistered name.
        """
        async with self._reconcile_lock:
            errors = {}
            for version_id in tuple(self.snapshots):
                publication = self.manager.publication(version_id)
                if publication.state not in (VersionState.DRAINING, VersionState.ABORTING):
                    continue
                for engine_id, instance_id in publication.targets:
                    client = self.engines[engine_id]
                    try:
                        expected_id = next((r.lora_id for r in publication.verified if r.engine_id == engine_id), None)
                        row = await client.control(
                            "fence", instance_id, publication.identity.lora_name, publication.attempt_id, expected_id
                        )
                        self.manager.observe_cleanup(
                            version_id,
                            publication.attempt_id,
                            CleanupObservation(
                                engine_id,
                                row["instance_id"],
                                row["lora_id"],
                                row["requests"],
                                row["admission_closed"],
                                row["attempt_fenced"],
                            ),
                        )
                        candidate = next(
                            (r for r in self.manager.unload_candidates(version_id) if r.engine_id == engine_id), None
                        )
                        if candidate is not None and row["phase"] not in ("loading", "unloading"):
                            row = await client.control(
                                "unload",
                                instance_id,
                                publication.identity.lora_name,
                                publication.attempt_id,
                                candidate.lora_id,
                            )
                            self.manager.observe_cleanup(
                                version_id,
                                publication.attempt_id,
                                CleanupObservation(
                                    engine_id,
                                    row["instance_id"],
                                    row["lora_id"],
                                    row["requests"],
                                    row["admission_closed"],
                                    row["attempt_fenced"],
                                ),
                            )
                    except Exception as exc:
                        errors[f"{version_id}/{engine_id}"] = exc
                if self.manager.publication(version_id).state in (VersionState.ABORTED, VersionState.RECLAIMED):
                    del self.snapshots[version_id]
            return errors

    async def replace_engine(
        self,
        engine_id: str,
        old_instance: str,
        new_instance: str,
        *,
        client: "SGLangLoRAClient | None" = None,
    ) -> None:
        """Host must first confirm the old process exited, not just timed out.

        Reload all bindable/bound snapshots before the replacement can serve.
        Retry after a lost ACK or probe failure. A definitive backend load
        failure requires another process replacement before loading again. Pass
        a changed endpoint as ``client`` instead of mutating ``engines``.
        """
        # Keep the same lock order as publish's failure reconciliation.
        async with self._publish_lock, self._reconcile_lock:
            replaced = self.manager.replace_engine(engine_id, old_instance, new_instance)
            if not replaced and self.manager.engine_instance(engine_id) != new_instance:
                raise PublicationError("ENGINE_INSTANCE_MISMATCH", "replacement lost the instance CAS")
            if not replaced and client is not None and client is not self.engines[engine_id]:
                raise PublicationError("ENGINE_INSTANCE_MISMATCH", "replacement retry must retain its client")
            if replaced:
                self.engines[engine_id].forget_replaced_instance(old_instance)
            if client is not None:
                self.engines[engine_id] = client
            for version_id in tuple(self.snapshots):
                publication = self.manager.publication(version_id)
                if publication.state not in (VersionState.PUBLISHED, VersionState.RETIRED):
                    continue
                if dict(publication.targets)[engine_id] != new_instance:
                    raise PublicationError("ENGINE_INSTANCE_MISMATCH", "replacement lost the instance CAS")
                try:
                    await self._verify(publication, engine_id, new_instance)
                except PublicationError as exc:
                    current = self.manager.publication(version_id)
                    # Session release is synchronous and does not take these
                    # locks. A fully checked replica may become unnecessary
                    # while I/O yields; leave it for the next cleanup pass.
                    if not (
                        exc.code == "STALE_ATTEMPT"
                        and publication.state is VersionState.RETIRED
                        and current.state is VersionState.DRAINING
                        and current.attempt_id == publication.attempt_id
                        and dict(current.targets)[engine_id] == new_instance
                    ):
                        raise
