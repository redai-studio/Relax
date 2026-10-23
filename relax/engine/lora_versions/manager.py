# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Serial publication decisions, independent of Ray, SGLang and transport.

All methods are synchronous. The integration can host this class in a Ray actor
with max_concurrency=1; RPCs, probes and inventory polling run outside it. Only
this owner changes the default. Engine cleanup reports must fence the attempt
before an absence releases capacity, including on engines that never
acknowledged a load. Engine-side compare-and-unload remains required.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum

from .identity import VersionIdentity


class PublicationError(ValueError):
    """Stable code plus explanation, also picklable across actor boundaries."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(code, message)
        self.code = code

    def __str__(self) -> str:
        return f"{self.code}: {self.args[1]}"


class VersionState(str, Enum):
    LOADING = "loading"
    PUBLISHED = "published"
    RETIRED = "retired"
    DRAINING = "draining"
    ABORTING = "aborting"
    ABORTED = "aborted"
    RECLAIMED = "reclaimed"


@dataclass(frozen=True)
class Replica:
    engine_id: str
    instance_id: str
    lora_id: str


@dataclass(frozen=True)
class SessionKey:
    shard_id: str
    instance_id: str
    sequence: int

    def __post_init__(self) -> None:
        if self.sequence < 0:
            raise ValueError("session sequence must be nonnegative")


@dataclass(frozen=True)
class CleanupObservation:
    """An engine's state after fencing/closing this publication attempt.

    ``lora_id=None`` means physically absent, not merely unregistered. A fenced
    attempt rejects delayed loads; closing admission alone does not. Positive
    request counts continue to protect draining adapters.
    """

    engine_id: str
    instance_id: str
    lora_id: str | None
    requests: int
    admission_closed: bool
    attempt_fenced: bool

    def __post_init__(self) -> None:
        if self.requests < 0 or (self.lora_id is None and self.requests):
            raise ValueError("request counts must be nonnegative and absent replicas must have zero requests")


@dataclass(frozen=True)
class Publication:
    identity: VersionIdentity
    attempt_id: int
    state: VersionState
    expected_default: int | None
    targets: tuple[tuple[str, str], ...]
    verified: tuple[Replica, ...]
    session_count: int
    absent_engines: tuple[str, ...]


@dataclass(frozen=True)
class Reservation:
    publication: Publication
    should_load: bool


@dataclass
class _Version:
    identity: VersionIdentity
    fingerprints: tuple[tuple[str, int], ...]
    attempt_id: int
    expected_default: int | None
    targets: dict[str, str]
    state: VersionState = VersionState.LOADING
    verified: dict[str, Replica] = field(default_factory=dict)
    cleanup: dict[str, CleanupObservation] = field(default_factory=dict)
    session_count: int = 0


@dataclass
class _Shard:
    instance_id: str
    active: bool = True
    watermark: int = -1
    bindings: dict[int, int] = field(default_factory=dict)
    closed: set[int] = field(default_factory=set)


class LoRAVersionManager:
    """One serial owner for a fixed engine pool; replacement uses instance CAS.

    Retain identities after reclamation for conflict detection and idempotent
    retries. Snapshots are owned by the publisher, outside this metadata-only
    state machine. Waiting queues and engine pool resizing belong to the
    publication/runtime integration, not this core.
    """

    def __init__(self, engines: Mapping[str, str], max_versions: int = 2) -> None:
        if not engines or any(not name or not instance for name, instance in engines.items()):
            raise ValueError("at least one named engine with an instance ID is required")
        if max_versions < 2:
            raise ValueError("max_versions must be at least two")
        self._engines = dict(engines)
        self._max_versions = max_versions
        self._versions: dict[int, _Version] = {}
        self._shards: dict[str, _Shard] = {}
        self._default: int | None = None
        self._next_attempt = 0

    @property
    def default_version(self) -> int | None:
        return self._default

    def engine_instance(self, engine_id: str) -> str:
        """Return the current process identity for replacement CAS retries."""
        return self._engines[engine_id]

    @property
    def occupied_versions(self) -> int:
        return sum(v.state not in (VersionState.ABORTED, VersionState.RECLAIMED) for v in self._versions.values())

    @property
    def serving_engines(self) -> tuple[str, ...]:
        """Only engines holding all currently bindable or bound versions
        serve."""
        live = [v for v in self._versions.values() if v.state in (VersionState.PUBLISHED, VersionState.RETIRED)]
        if not live:
            return ()
        return tuple(
            engine
            for engine, instance in sorted(self._engines.items())
            if all(engine in v.verified and v.verified[engine].instance_id == instance for v in live)
        )

    def publication(self, version_id: int) -> Publication:
        """Return detached immutable metadata, safe to pass to a caller."""
        v = self._versions[version_id]
        return Publication(
            v.identity,
            v.attempt_id,
            v.state,
            v.expected_default,
            tuple(sorted(v.targets.items())),
            tuple(v.verified[e] for e in sorted(v.verified)),
            v.session_count,
            tuple(sorted(e for e, o in v.cleanup.items() if o.lora_id is None and o.attempt_fenced)),
        )

    def reserve(self, identity: VersionIdentity, fingerprints: Mapping[str, int]) -> Reservation:
        """Decide idempotency before capacity; only a new reservation may
        load."""
        existing = self._versions.get(identity.version_id)
        if existing is not None:
            if existing.identity.digest != identity.digest:
                raise PublicationError("VERSION_CONFLICT", f"{existing.identity.digest} != {identity.digest}")
            if existing.state is not VersionState.ABORTED:
                return Reservation(self.publication(identity.version_id), should_load=False)
            if tuple(sorted(fingerprints.items())) != existing.fingerprints:
                raise PublicationError("REPLICA_INVALID", "retry fingerprints must match the original snapshot")
        if self._default is not None and identity.version_id <= self._default:
            raise PublicationError("VERSION_SUPERSEDED", "a retry cannot replace a newer published version")
        if not fingerprints:
            raise ValueError("expected transport fingerprints are required")
        if self.occupied_versions >= self._max_versions:
            raise PublicationError("CAPACITY_EXHAUSTED", "live, loading and unconfirmed replicas occupy capacity")
        if any(v.state is VersionState.LOADING for v in self._versions.values()):
            raise PublicationError("PUBLICATION_IN_PROGRESS", "only one publication may load at a time")
        self._next_attempt += 1
        self._versions[identity.version_id] = _Version(
            identity,
            tuple(sorted(fingerprints.items())),
            self._next_attempt,
            self._default,
            dict(self._engines),
        )
        return Reservation(self.publication(identity.version_id), should_load=True)

    def record_verified(
        self,
        version_id: int,
        attempt_id: int,
        replica: Replica,
        fingerprints: Mapping[str, int],
        *,
        pinned: bool,
    ) -> bool:
        """Record a checksum-checked load after a successful generation probe.

        Also accepts recovery of a published/retired version after engine
        replacement. Old attempts/instances are ignored, never resurrected.
        """
        v = self._versions[version_id]
        if (
            v.attempt_id != attempt_id
            or v.state not in (VersionState.LOADING, VersionState.PUBLISHED, VersionState.RETIRED)
            or v.targets.get(replica.engine_id) != replica.instance_id
        ):
            return False
        previous = v.verified.get(replica.engine_id)
        if not replica.lora_id or not pinned or tuple(sorted(fingerprints.items())) != v.fingerprints:
            v.verified.pop(replica.engine_id, None)
            self.cancel(version_id, attempt_id)
            raise PublicationError("REPLICA_INVALID", "probe receipt must identify pinned, matching adapter contents")
        if previous is not None and previous != replica:
            v.verified.pop(replica.engine_id, None)
            self.cancel(version_id, attempt_id)
            raise PublicationError("LORA_ID_MISMATCH", "one attempt cannot overwrite its verified replica")
        v.verified[replica.engine_id] = replica
        return True

    def commit(self, version_id: int, attempt_id: int, expected_default: int | None) -> Publication:
        """The only default-switch point; all target instances must be
        verified."""
        v = self._versions[version_id]
        if v.attempt_id != attempt_id:
            raise PublicationError("STALE_ATTEMPT", "publication attempt has been replaced")
        if v.state in (VersionState.PUBLISHED, VersionState.RETIRED, VersionState.DRAINING, VersionState.RECLAIMED):
            return self.publication(version_id)
        if v.state is not VersionState.LOADING:
            raise PublicationError("PUBLICATION_CANCELLED", "cancel won before commit")
        if expected_default != self._default or expected_default != v.expected_default:
            raise PublicationError("DEFAULT_CHANGED", "default version does not match the reservation")
        if any(
            engine not in v.verified or v.verified[engine].instance_id != instance or self._engines[engine] != instance
            for engine, instance in v.targets.items()
        ):
            raise PublicationError("VERSION_NOT_READY", "every target engine must be verified before commit")
        if self._default is not None:
            old = self._versions[self._default]
            old.state = VersionState.RETIRED if old.session_count else VersionState.DRAINING
        v.state = VersionState.PUBLISHED
        self._default = version_id
        return self.publication(version_id)

    def cancel(self, version_id: int, attempt_id: int) -> bool:
        v = self._versions[version_id]
        if v.attempt_id != attempt_id:
            return False
        if v.state is VersionState.LOADING:
            v.state = VersionState.ABORTING
        return v.state in (VersionState.ABORTING, VersionState.ABORTED)

    def observe_cleanup(self, version_id: int, attempt_id: int, observation: CleanupObservation) -> bool:
        """Apply fenced inventory evidence, not a mere unload RPC success.

        Callers fence *every* target, even when a load reply was lost. Engine
        fencing and stop-admission are monotonic within an attempt; once both
        hold, request counts can only decrease and no new instance can load.
        Reject an entire regressive observation rather than combining fields
        from different snapshots. Engine observations must still be truthful
        and internally consistent; this is not a substitute for engine fencing.
        """
        v = self._versions[version_id]
        o = observation
        if (
            v.attempt_id != attempt_id
            or v.state not in (VersionState.DRAINING, VersionState.ABORTING)
            or v.targets.get(o.engine_id) != o.instance_id
        ):
            return False
        verified = v.verified.get(o.engine_id)
        previous = v.cleanup.get(o.engine_id)
        known_id = verified.lora_id if verified is not None else (previous.lora_id if previous is not None else None)
        if o.lora_id is not None and known_id is not None and o.lora_id != known_id:
            return False
        if previous is not None:
            # Keep confirmed absence and monotonic drain progress despite
            # delayed inventory responses. Counts may grow before admission
            # and loading are both closed, so do not take a running minimum.
            if (
                (previous.lora_id is None and previous.attempt_fenced)
                or (previous.attempt_fenced and not o.attempt_fenced)
                or (previous.admission_closed and not o.admission_closed)
                or (previous.attempt_fenced and previous.admission_closed and o.requests > previous.requests)
            ):
                return False
        v.cleanup[o.engine_id] = o
        if all(
            engine in v.cleanup and v.cleanup[engine].lora_id is None and v.cleanup[engine].attempt_fenced
            for engine in v.targets
        ):
            v.state = VersionState.ABORTED if v.state is VersionState.ABORTING else VersionState.RECLAIMED
            v.verified.clear()
        return True

    def unload_candidates(self, version_id: int) -> tuple[Replica, ...]:
        """Return safe compare-and-unload targets; execution belongs to the
        engine.

        Repeated candidates are intentional: lost replies require retrying an
        idempotent engine operation, always with the same instance/lora IDs.
        """
        v = self._versions[version_id]
        if v.state not in (VersionState.DRAINING, VersionState.ABORTING):
            return ()
        return tuple(
            Replica(o.engine_id, o.instance_id, o.lora_id)
            for _, o in sorted(v.cleanup.items())
            if o.lora_id is not None and o.attempt_fenced and o.admission_closed and o.requests == 0
        )

    def replace_engine(self, engine_id: str, expected_instance: str, new_instance: str) -> bool:
        """Invalidate receipts on a confirmed process replacement, using CAS.

        The transport must reject control requests addressed to old instances.
        Recovery receipts are required before this engine can serve again.
        """
        if not new_instance:
            raise ValueError("an engine instance ID is required")
        if self._engines[engine_id] != expected_instance or new_instance == expected_instance:
            return False
        self._engines[engine_id] = new_instance
        for v in self._versions.values():
            if v.state in (VersionState.ABORTED, VersionState.RECLAIMED):
                continue
            v.targets[engine_id] = new_instance
            v.verified.pop(engine_id, None)
            v.cleanup.pop(engine_id, None)
            if v.state is VersionState.LOADING:
                v.state = VersionState.ABORTING
        return True

    def register_shard(self, shard_id: str, instance_id: str, *, expected_instance: str | None = None) -> None:
        """Register/restart a shard; an old registration cannot displace a new
        one."""
        if not shard_id or not instance_id:
            raise ValueError("a shard name and instance ID are required")
        previous = self._shards.get(shard_id)
        if previous is not None and previous.instance_id == instance_id:
            if not previous.active:
                raise PublicationError("SESSION_CLOSED", "a dead shard instance cannot restart")
            return
        if (previous.instance_id if previous is not None else None) != expected_instance:
            raise PublicationError("SHARD_INSTANCE_CHANGED", "shard replacement must match its previous instance")
        if previous is not None:
            self.close_shard(shard_id, previous.instance_id)
        self._shards[shard_id] = _Shard(instance_id)

    def bind(self, key: SessionKey) -> VersionIdentity:
        shard = self._shards.get(key.shard_id)
        if (
            shard is None
            or not shard.active
            or shard.instance_id != key.instance_id
            or key.sequence <= shard.watermark
            or key.sequence in shard.closed
        ):
            raise PublicationError("SESSION_CLOSED", "session or shard instance is closed")
        if key.sequence in shard.bindings:
            return self._versions[shard.bindings[key.sequence]].identity
        if self._default is None:
            raise PublicationError("NO_PUBLISHED_VERSION", "first publication has not committed")
        if not self.serving_engines:
            raise PublicationError("LORA_VERSION_UNAVAILABLE", "no engine is ready to serve the published versions")
        shard.bindings[key.sequence] = self._default
        v = self._versions[self._default]
        v.session_count += 1
        return v.identity

    def release(self, key: SessionKey) -> bool:
        shard = self._shards.get(key.shard_id)
        if (
            shard is None
            or shard.instance_id != key.instance_id
            or not shard.active
            or key.sequence <= shard.watermark
            or key.sequence in shard.closed
        ):
            return False
        shard.closed.add(key.sequence)
        version_id = shard.bindings.pop(key.sequence, None)
        if version_id is not None:
            v = self._versions[version_id]
            v.session_count -= 1
            if v.session_count == 0 and v.state is VersionState.RETIRED:
                v.state = VersionState.DRAINING
        return True

    def close_shard(self, shard_id: str, instance_id: str) -> bool:
        shard = self._shards.get(shard_id)
        if shard is None or shard.instance_id != instance_id or not shard.active:
            return False
        for sequence in tuple(shard.bindings):
            self.release(SessionKey(shard_id, instance_id, sequence))
        shard.active = False
        shard.closed.clear()
        return True

    def advance_watermark(self, shard_id: str, instance_id: str, sequence: int) -> bool:
        shard = self._shards.get(shard_id)
        if shard is None or shard.instance_id != instance_id or not shard.active or sequence <= shard.watermark:
            return False
        if any(active <= sequence for active in shard.bindings):
            raise PublicationError("SESSION_STILL_ACTIVE", "watermark would close a bound session")
        shard.watermark = sequence
        shard.closed = {closed for closed in shard.closed if closed > sequence}
        return True
