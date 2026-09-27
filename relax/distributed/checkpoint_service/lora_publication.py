# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Immutable LoRA publication: frozen snapshot identity plus staged transport.

The *identity* of a published adapter lives in ``LoRAVersionRegistry``; this module owns
the two things that must hold no matter who moves the bytes:

* the exact digest of one frozen ``AdapterSnapshot`` (RFC §9.4 / §9.6), and
* the staged ``Begin -> Bucket* -> End -> fleet commit`` driver with its clean-vs-fatal
  failure semantics (RFC §12 / §13), the §13.3 cleanup fan-out and the §17.2 reclaim
  fan-out.

Hardware is injected: the caller passes the engine fan-out, the rank-0 NCCL broadcast, and
a blocking Registry facade, so the whole protocol is exercisable without engines, NCCL or
Megatron.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import ray
import torch

from relax.agentic.session.lora_version import (
    LORA_VERSION_REGISTRY_ACTOR_NAME,
    LoRAVersionError,
    Publication,
    VersionBinding,
)
from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)

#: The one control endpoint every staged operation rides on (no new HTTP routes).
UPDATE_LORA_ENDPOINT = "/update_lora_from_distributed"


class LoRAPublicationError(RuntimeError):
    """Publication, cleanup or reclaim failure.

    ``kind`` is ``RETRYABLE`` when every engine confirmed the candidate ABSENT
    (the caller may explicitly retry the same version), and ``FATAL`` when
    engine or transport state is dirty/ambiguous (the run fails closed and
    capacity is NOT released).
    """

    def __init__(self, kind: str, message: str = "") -> None:
        super().__init__(message or kind)
        self.kind = kind


@dataclass(frozen=True)
class AdapterSnapshot:
    """One immutable adapter: the only source of bytes for a version's
    transport.

    ``tensors`` must own their storage — hashing then transmitting must both
    see the bytes the trainer had at the instant the snapshot was taken,
    whatever later steps do to the live parameters.
    """

    config: Dict[str, Any]
    tensors: Dict[str, torch.Tensor]
    manifest: Dict[str, str]
    digest: str


@dataclass(frozen=True)
class EngineReply:
    """One engine's answer to one control request."""

    success: bool
    message: str = ""
    #: Transport/ACK unknown (unreachable engine, timeout, 5xx) as opposed to a clean
    #: engine refusal (4xx): only the latter leaves the engine in a state the publisher
    #: can act on, so only the latter may end in FAILED_RETRYABLE.
    ambiguous: bool = False
    receipt: Optional[Dict[str, Any]] = None


def classify_engine_response(status: int, body: Any) -> EngineReply:
    """Map one HTTP answer from an engine onto a publisher verdict.

    Only an explicit ``success: true`` on a 2xx counts as success. A 4xx is a
    *clean* refusal: the engine's control layer answered and rejected the
    request, so what it holds is knowable and the candidate can be cleaned up.
    Anything else — 5xx, a 2xx without ``success``, a malformed body — leaves
    the engine's state unknown and must fail the publication closed.
    """

    payload = body if isinstance(body, dict) else {}
    detail = payload.get("error_message") or payload.get("message")
    if 200 <= status < 300 and payload.get("success") is True:
        return EngineReply(True, str(detail or ""), receipt=payload.get("publication_receipt"))
    if 400 <= status < 500:
        return EngineReply(False, f"HTTP {status}: {detail or body}")
    return EngineReply(False, f"HTTP {status}: {detail or body}", ambiguous=True)


@dataclass
class PublicationOutcome:
    """What ``publish()`` did, for logging and acceptance reporting."""

    status: str  # "PUBLISHED" | "NO_OP"
    version_id: Optional[int] = None
    lora_name: Optional[str] = None
    bucket_count: int = 0
    max_bucket_bytes: int = 0
    bucket_cap_bytes: int = 0


def tensor_dtype_name(tensor: torch.Tensor) -> str:
    """Wire-format dtype name (``bfloat16``, not ``torch.bfloat16``)."""

    return str(tensor.dtype).replace("torch.", "")


def tensor_manifest_hash(tensor: torch.Tensor) -> str:
    """``SHA256(dtype + shape + contiguous bytes)`` (RFC §9.4).

    Hashes raw bytes through a ``uint8`` view: bfloat16 has no numpy dtype, and
    going through numpy would silently round-trip half the dtypes we train
    with.
    """

    frozen = tensor.detach().to("cpu").contiguous()
    hasher = hashlib.sha256()
    hasher.update(str(tensor_dtype_name(frozen)).encode())
    hasher.update(str(list(frozen.shape)).encode())
    hasher.update(frozen.view(torch.uint8).numpy().tobytes())
    return hasher.hexdigest()


def adapter_digest(config: Mapping[str, Any], manifest: Mapping[str, str]) -> str:
    """``SHA256(canonical config + sorted(name, hash))`` (RFC §9.4).

    Bucket boundaries deliberately do not participate: the digest identifies
    content, not transport, so re-bucketing the same adapter still maps to the
    same version.
    """

    hasher = hashlib.sha256()
    hasher.update(json.dumps(dict(config), sort_keys=True, separators=(",", ":"), default=str).encode())
    for name in sorted(manifest):
        hasher.update(name.encode())
        hasher.update(manifest[name].encode())
    return hasher.hexdigest()


def materialize_adapter_snapshot(config: Mapping[str, Any], tensors: Mapping[str, torch.Tensor]) -> AdapterSnapshot:
    """Freeze one owned copy of an exported adapter (RFC §9.6).

    The bridge exports views over the live trainer parameters, so the snapshot
    clones: without that, the next training step could change the bytes we hash
    or send.
    """

    frozen = {name: tensor.detach().to("cpu").contiguous().clone() for name, tensor in tensors.items()}
    manifest = {name: tensor_manifest_hash(tensor) for name, tensor in frozen.items()}
    return AdapterSnapshot(
        config=dict(config),
        tensors=frozen,
        manifest=manifest,
        digest=adapter_digest(config, manifest),
    )


class RayLoRAVersionRegistryClient:
    """Blocking facade over the fleet Registry actor.

    The publisher runs inside a Ray actor and is synchronous, so ``ray.get`` is
    the right call shape here — the Registry API itself stays non-blocking.
    """

    def __init__(self, handle: Any) -> None:
        self._handle = handle

    @classmethod
    def from_deployment(cls) -> "RayLoRAVersionRegistryClient":
        """Fetch the live Registry.

        The handle is re-fetched per publication: a missing actor (the whole
        deployment restarted under us) must fail this run closed instead of
        publishing against a stale epoch.
        """

        return cls(ray.get_actor(LORA_VERSION_REGISTRY_ACTOR_NAME))

    def allocate(self, digest: str, *, version_id: int) -> Publication:
        return ray.get(self._handle.allocate.remote(digest, version_id=version_id))

    def claim_publication(self, version_id: int, attempt_id: int) -> None:
        ray.get(self._handle.claim_publication.remote(version_id, attempt_id))

    def record_prepared(self, version_id: int, attempt_id: int, receipts: Dict[str, dict]) -> None:
        ray.get(self._handle.record_prepared.remote(version_id, attempt_id, receipts))

    def record_ready(self, version_id: int, attempt_id: int, receipts: Dict[str, dict]) -> None:
        ray.get(self._handle.record_ready.remote(version_id, attempt_id, receipts))

    def mark_published(self, version_id: int, attempt_id: int) -> Publication:
        return ray.get(self._handle.mark_published.remote(version_id, attempt_id))

    def mark_retryable_failure(self, version_id: int, attempt_id: int) -> None:
        ray.get(self._handle.mark_retryable_failure.remote(version_id, attempt_id))

    def mark_fatal_failure(self, version_id: int, attempt_id: int) -> None:
        ray.get(self._handle.mark_fatal_failure.remote(version_id, attempt_id))

    def claim_reclaimable(self) -> Optional[VersionBinding]:
        return ray.get(self._handle.claim_reclaimable.remote())

    def mark_reclaimed(self, version_id: int) -> None:
        ray.get(self._handle.mark_reclaimed.remote(version_id))

    def mark_reclaim_fatal(self, version_id: int) -> None:
        ray.get(self._handle.mark_reclaim_fatal.remote(version_id))


def adapter_tensor_bytes(snapshot: AdapterSnapshot) -> List[int]:
    """Byte size per tensor, in the snapshot's own (== transport) order."""

    return [tensor.numel() * tensor.element_size() for tensor in snapshot.tensors.values()]


def _bucket_byte_sizes(tensor_bytes: Sequence[int], bucket_sizes: Sequence[int]) -> List[int]:
    """Bytes per bucket, for the telemetry the staged protocol must report (§12
    Phase 2)."""

    sizes = []
    offset = 0
    for count in bucket_sizes:
        sizes.append(sum(tensor_bytes[offset : offset + count]))
        offset += count
    return sizes


class LoRAPublisher:
    """Drives one immutable adapter version from frozen snapshot to fleet
    ``PUBLISHED``.

    Only rank 0 (the NCCL source) owns a publisher. ``fire`` must not raise
    (per-engine delivery problems surface in ``collect``), because a bucket the
    source cannot account for is a wedged collective, not a retryable error.
    """

    def __init__(
        self,
        *,
        fire: Callable[[str, Dict[str, Any]], Dict[str, Any]],
        collect: Callable[[Dict[str, Any]], Dict[str, EngineReply]],
        broadcast: Callable[[Sequence[str], int], None],
        registry: RayLoRAVersionRegistryClient,
        bucket_cap_bytes: int = 0,
        group_name: str = "",
    ) -> None:
        self._fire = fire
        self._collect = collect
        self._broadcast = broadcast
        self._registry = registry
        self._bucket_cap_bytes = bucket_cap_bytes
        self._group_name = group_name
        self._engine_incarnations: List[str] = []

    # ------------------------------------------------------------------
    # Entry points
    # ------------------------------------------------------------------

    def publish(
        self, snapshot: AdapterSnapshot, bucket_sizes: Sequence[int], *, version_id: int
    ) -> PublicationOutcome:
        """Publish ``snapshot`` as a new immutable version (§12).

        ``version_id`` identifies the publication event, independently of its
        content. The caller must reuse it when retrying that event.
        ``bucket_sizes`` are tensor counts per bucket; the caller builds them
        with the publication-specific soft cap so the adapter travels in
        several buckets.
        """

        # Resolve identity before any engine RPC: conflicting content and late
        # completed replays must not unload unrelated resources. Only capacity
        # refusal may trigger reclamation followed by a fresh atomic admission.
        try:
            publication = self._registry.allocate(snapshot.digest, version_id=version_id)
        except LoRAVersionError as exc:
            if exc.code != "CAPACITY_ERROR" or not self.reclaim_once():
                raise
            publication = self._registry.allocate(snapshot.digest, version_id=version_id)
        if publication.no_op:
            logger.info(
                "[lora-version] digest=%s version %d (%s) already completed publication; exact no-op",
                snapshot.digest[:16],
                publication.version_id,
                publication.lora_name,
            )
            return PublicationOutcome(
                status="NO_OP",
                version_id=publication.version_id,
                lora_name=publication.lora_name,
                bucket_cap_bytes=self._bucket_cap_bytes,
            )

        # allocate() is an idempotent lookup, not permission to replay NCCL.
        # Exactly one caller may drive each attempt, even after a lost reply.
        self._registry.claim_publication(publication.version_id, publication.attempt_id)
        tensor_bytes = adapter_tensor_bytes(snapshot)
        total_bytes = sum(tensor_bytes)
        bucket_bytes = _bucket_byte_sizes(tensor_bytes, bucket_sizes)
        bucket_count = len(bucket_sizes)
        max_bucket_bytes = max(bucket_bytes, default=0)
        logger.info(
            "[lora-version] publishing version %d (%s) digest=%s: %d tensors, %.1f MiB, "
            "bucket_cap=%.1f MiB, buckets=%d, max_bucket=%.1f MiB, attempt=%d",
            publication.version_id,
            publication.lora_name,
            snapshot.digest[:16],
            len(tensor_bytes),
            total_bytes / 1024**2,
            self._bucket_cap_bytes / 1024**2,
            bucket_count,
            max_bucket_bytes / 1024**2,
            publication.attempt_id,
        )
        if bucket_count == 1:
            logger.warning(
                "LoRA staged publication has only one data-bearing bucket (cap %.1f MiB for %.1f MiB of "
                "adapter): generation overlap is not structurally guaranteed. Lower "
                "--lora-publication-bucket-size to publish in several buckets.",
                self._bucket_cap_bytes / 1024**2,
                total_bytes / 1024**2,
            )

        self._begin(snapshot, publication, bucket_sizes)
        self._send_buckets(snapshot, publication, bucket_sizes)
        self._end(snapshot, publication)

        self._registry.mark_published(publication.version_id, publication.attempt_id)
        logger.info(
            "[lora-version] version %d (%s) is fleet-published; new Sessions bind it",
            publication.version_id,
            publication.lora_name,
        )
        # A predecessor whose Sessions already released can go right away.
        self.reclaim_once()
        return PublicationOutcome(
            status="PUBLISHED",
            version_id=publication.version_id,
            lora_name=publication.lora_name,
            bucket_count=bucket_count,
            max_bucket_bytes=max_bucket_bytes,
            bucket_cap_bytes=self._bucket_cap_bytes,
        )

    def reclaim_once(self) -> bool:
        """Unload one retired version no Session can request any more (§17.2).

        Exactly-once: the Registry hands out a version at most once, so at most one unload
        RPC per engine is ever sent for it.
        """

        claim = self._registry.claim_reclaimable()
        if claim is None:
            return False
        logger.info("[lora-version] reclaiming retired version %d (%s)", claim.version_id, claim.lora_name)
        replies = self._fan_out({"op": "unload", "lora_name": claim.lora_name, "reason": "reclaim"}, "unload")
        bad = {engine_id: reply for engine_id, reply in replies.items() if not reply.success}
        if bad:
            # §17.3: an unload we cannot confirm is never retried (not executed vs already
            # executed is unknowable) and never releases capacity.
            self._registry.mark_reclaim_fatal(claim.version_id)
            raise LoRAPublicationError(
                "FATAL",
                f"reclaim of version {claim.version_id} ({claim.lora_name}) was not confirmed on "
                f"{sorted(bad)}: "
                + "; ".join(f"{engine_id}: {reply.message}" for engine_id, reply in sorted(bad.items())),
            )
        self._registry.mark_reclaimed(claim.version_id)
        logger.info("[lora-version] version %d (%s) reclaimed; its slot is free", claim.version_id, claim.lora_name)
        return True

    # ------------------------------------------------------------------
    # Phases
    # ------------------------------------------------------------------

    def _begin(self, snapshot: AdapterSnapshot, publication: Publication, bucket_sizes: Sequence[int]) -> None:
        """§12 Phase 1: build the candidate on every engine before any
        collective."""

        replies = self._fan_out(
            {
                "op": "begin",
                "protocol_version": 2,
                "version_id": publication.version_id,
                "digest": publication.digest,
                "expected_checksums": snapshot.manifest,
                "names": list(snapshot.tensors),
                "dtypes": [tensor_dtype_name(tensor) for tensor in snapshot.tensors.values()],
                "shapes": [list(tensor.shape) for tensor in snapshot.tensors.values()],
                "bucket_sizes": list(bucket_sizes),
                "lora_name": publication.lora_name,
                "attempt_id": publication.attempt_id,
                "config_dict": snapshot.config,
                "pinned": True,
                "group_name": self._group_name,
            },
            "begin",
        )
        bad = {engine_id: reply for engine_id, reply in replies.items() if not reply.success}
        if not bad:
            try:
                receipts = {engine: reply.receipt or {} for engine, reply in replies.items()}
                self._registry.record_prepared(publication.version_id, publication.attempt_id, receipts)
                self._engine_incarnations = [receipt["engine_incarnation"] for receipt in receipts.values()]
            except Exception as exc:
                raise self._fatal(publication, f"invalid PREPARED receipts: {exc}") from exc
            return
        logger.error(
            "[lora-version] Begin failed on %s for version %d attempt %d",
            sorted(bad),
            publication.version_id,
            publication.attempt_id,
        )
        # No NCCL bucket ever started, so accepted engines only need their staging discarded.
        clean = self._cleanup_unpublished(publication)
        self._fail(
            publication,
            clean,
            f"Begin failed on {sorted(bad)}: "
            + "; ".join(f"{engine_id}: {reply.message}" for engine_id, reply in sorted(bad.items())),
        )

    def _send_buckets(
        self,
        snapshot: AdapterSnapshot,
        publication: Publication,
        bucket_sizes: Sequence[int],
    ) -> None:
        """§12 Phase 2: fire, broadcast, then await each bucket in turn.

        The order matters and is the whole point of staging: the engines must
        already be blocked in the collective when the source enters it, and the
        next bucket never starts before this one's ACKs are in.
        """

        names = list(snapshot.tensors)
        offset = 0
        for index, count in enumerate(bucket_sizes):
            chunk = list(names[offset : offset + count])
            payload = {
                "op": "bucket",
                "engine_incarnations": self._engine_incarnations,
                "lora_name": publication.lora_name,
                "attempt_id": publication.attempt_id,
                "bucket_index": index,
                "names": chunk,
                "dtypes": [tensor_dtype_name(snapshot.tensors[name]) for name in chunk],
                "shapes": [list(snapshot.tensors[name].shape) for name in chunk],
                "bucket_sizes": [len(chunk)],
                "group_name": self._group_name,
            }
            try:
                pending = self._fire(UPDATE_LORA_ENDPOINT, payload)
                self._require_fleet(pending, f"bucket {index}")
                self._broadcast(chunk, index)
            except BaseException as exc:  # noqa: BLE001 - collective state is unknown: fatal
                raise self._fatal(
                    publication,
                    f"bucket {index} collective did not complete ({exc!r}); refusing to retransmit",
                ) from exc
            replies = self._collect_fleet(pending, f"bucket {index}")
            unknown = {engine for engine, reply in replies.items() if reply.ambiguous}
            if unknown:
                raise self._fatal(publication, f"bucket {index} completion is unknown on {sorted(unknown)}")
            bad = {engine_id: reply for engine_id, reply in replies.items() if not reply.success}
            if bad:
                logger.error(
                    "[lora-version] bucket %d rejected by %s for version %d attempt %d",
                    index,
                    sorted(bad),
                    publication.version_id,
                    publication.attempt_id,
                )
                clean = self._cleanup_unpublished(publication)
                self._fail(
                    publication,
                    clean,
                    f"bucket {index} rejected by {sorted(bad)}: "
                    + "; ".join(f"{engine_id}: {reply.message}" for engine_id, reply in sorted(bad.items())),
                )
            offset += count
        if offset != len(names):
            # Publisher-side protocol bug: the engines are holding a partial stash.
            clean = self._cleanup_unpublished(publication)
            self._fail(publication, clean, f"bucket plan covered {offset} of {len(names)} tensors")

    def _end(self, snapshot: AdapterSnapshot, publication: Publication) -> None:
        """§12 Phase 3: every engine verifies its stash and loads the adapter
        locally."""

        replies = self._fan_out(
            {
                "op": "end",
                "engine_incarnations": self._engine_incarnations,
                "lora_name": publication.lora_name,
                "attempt_id": publication.attempt_id,
                "expected_checksums": snapshot.manifest,
            },
            "end",
        )
        unknown = {engine_id: reply for engine_id, reply in replies.items() if not reply.success and reply.ambiguous}
        if unknown:
            # §13.5: the engine's local state after a failed load is UNKNOWN (the load may have
            # written backend state), so this version can never be called ABSENT. Best-effort
            # cleanup of the engines we can still talk to avoids extra pinned orphans; the run
            # fails closed and capacity stays occupied.
            self._best_effort_cleanup(publication)
            raise self._fatal(
                publication,
                f"End could not be determined on {sorted(unknown)}: "
                + "; ".join(f"{engine_id}: {reply.message}" for engine_id, reply in sorted(unknown.items())),
            )
        bad = {engine_id: reply for engine_id, reply in replies.items() if not reply.success}
        if bad:
            # §13.3: a partial success does NOT keep its READY_LOCAL candidate for a retry —
            # cleanup every unpublished resource, then settle on the cleanup verdict.
            logger.error(
                "[lora-version] End failed on %s for version %d attempt %d; cleaning up all engines",
                sorted(bad),
                publication.version_id,
                publication.attempt_id,
            )
            clean = self._cleanup_unpublished(publication)
            self._fail(
                publication,
                clean,
                f"End failed on {sorted(bad)}: "
                + "; ".join(f"{engine_id}: {reply.message}" for engine_id, reply in sorted(bad.items())),
            )

        # Re-check exact local identity after End. A restarted/missing engine must
        # not inherit its predecessor's READY. The Registry also validates the CAS
        # revision and both receipts before changing the default.
        try:
            self._registry.record_ready(
                publication.version_id,
                publication.attempt_id,
                {engine: reply.receipt or {} for engine, reply in replies.items()},
            )
            confirmed = self._fan_out(
                {
                    "op": "status",
                    "lora_name": publication.lora_name,
                    "attempt_id": publication.attempt_id,
                    "engine_incarnations": self._engine_incarnations,
                },
                "ready confirmation",
            )
            if any(not reply.success for reply in confirmed.values()):
                raise LoRAPublicationError("FATAL", "READY was invalidated before commit")
            self._registry.record_ready(
                publication.version_id,
                publication.attempt_id,
                {engine: reply.receipt or {} for engine, reply in confirmed.items()},
            )
        except Exception as exc:
            raise self._fatal(publication, f"READY confirmation failed: {exc}") from exc

    # ------------------------------------------------------------------
    # Failure paths
    # ------------------------------------------------------------------

    def _cleanup_unpublished(self, publication: Publication) -> bool:
        """§13.3: make the candidate ABSENT everywhere, including never-fleet-published
        ``READY_LOCAL`` engines.

        ``op=unload`` is name-scoped and idempotent on every local state (staged candidate,
        registered-but-unpublished, or never existed), so one fan-out covers the whole
        cleanup and its verdict is exactly "all engines ABSENT". The attempt_id is sent so
        a retry of the same version (same name, new attempt) is never stripped by a late
        or duplicated cleanup of this attempt.
        """

        replies = self._fan_out(
            {
                "op": "unload",
                "lora_name": publication.lora_name,
                "attempt_id": publication.attempt_id,
                "reason": "cleanup_unpublished",
            },
            "cleanup",
        )
        bad = {engine_id: reply for engine_id, reply in replies.items() if not reply.success}
        if bad:
            logger.error(
                "[lora-version] cleanup of version %d (%s) was not confirmed on %s",
                publication.version_id,
                publication.lora_name,
                sorted(bad),
            )
            return False
        logger.info(
            "[lora-version] version %d (%s) confirmed ABSENT on every engine",
            publication.version_id,
            publication.lora_name,
        )
        return True

    def _best_effort_cleanup(self, publication: Publication) -> None:
        replies = self._fan_out(
            {
                "op": "unload",
                "lora_name": publication.lora_name,
                "attempt_id": publication.attempt_id,
                "reason": "best_effort_cleanup",
            },
            "cleanup",
        )
        bad = sorted(engine_id for engine_id, reply in replies.items() if not reply.success)
        if bad:
            logger.error("[lora-version] best-effort cleanup left %s unconfirmed", bad)

    def _require_fleet(self, pending: Mapping[str, Any], phase: str) -> None:
        """§11.4: versioned publication needs the complete target fleet.

        An empty fan-out means every engine was missing or unreachable: publishing to nobody
        would make the version look fleet-ready without a single engine holding it, so fail hard.
        """

        if set(pending) != {"engine0", "engine1"}:
            raise LoRAPublicationError(
                "FATAL", f"{phase}: expected fixed engines engine0/engine1, got {sorted(pending)}"
            )

    def _fan_out(self, payload: Dict[str, Any], phase: str) -> Dict[str, EngineReply]:
        pending = self._fire(UPDATE_LORA_ENDPOINT, payload)
        self._require_fleet(pending, phase)
        return self._collect_fleet(pending, phase)

    def _collect_fleet(self, pending: Dict[str, Any], phase: str) -> Dict[str, EngineReply]:
        replies = self._collect(pending)
        missing = set(pending) - set(replies)
        if missing:
            # A fired engine with no reply is not "fine": treat it like an unreachable engine.
            replies.update(
                {engine_id: EngineReply(False, f"{phase}: no reply", ambiguous=True) for engine_id in missing}
            )
        return replies

    def _fail(self, publication: Publication, clean: bool, message: str) -> None:
        """Settle a failed attempt: retryable only when every engine is
        confirmed ABSENT."""

        if clean:
            self._registry.mark_retryable_failure(publication.version_id, publication.attempt_id)
            raise LoRAPublicationError(
                "RETRYABLE",
                f"{message} | all engines confirmed the candidate ABSENT; version "
                f"{publication.version_id} may be explicitly retried with the same version_id and digest",
            )
        self._registry.mark_fatal_failure(publication.version_id, publication.attempt_id)
        raise LoRAPublicationError(
            "FATAL",
            f"{message} | cleanup was not confirmed, so engine state is unknown and the run fails closed",
        )

    def _fatal(self, publication: Publication, message: str) -> LoRAPublicationError:
        try:
            self._registry.mark_fatal_failure(publication.version_id, publication.attempt_id)
        except LoRAVersionError as exc:  # a newer attempt already owns the version: report both
            message = f"{message} | {exc}"
        return LoRAPublicationError("FATAL", message)
