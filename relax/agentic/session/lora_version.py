# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Immutable LoRA version registry: the Task-7 fleet control plane.

One shared writer (a named Ray actor in deployment, a plain object in tests) owns
every piece of fleet-visible LoRA state:

* which immutable `lora_name` versions exist, and their publication state;
* which one is the fleet default (what a *new* Session binds);
* which logical versions still own runtime resources (capacity);
* the authoritative `session_id -> version_id` map, i.e. who may still generate
  with a retired version.

DCS publishers and Agentic SessionShards all funnel their state transitions
through this object. The actor only *decides* (atomically, in one queue);
snapshotting tensors, NCCL transport, engine loads, unloads and probes all happen
outside it in the publisher, so no Registry call ever waits on the network.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional, Set


#: Adapter-name prefix for managed versions; the epoch keeps names fresh across
#: redeploys so a stale engine-local registration can never collide with a new one.
LORA_VERSION_NAME_PREFIX = "relax_policy_lora"

#: Named Ray actor holding the one Registry every participant talks to.
LORA_VERSION_REGISTRY_ACTOR_NAME = "relax_lora_version_registry"


def versioned_lora_publication_enabled(args) -> bool:
    """Is this run on the immutable versioned path (Task 7)?

    Everything else — non-Agentic rollouts, merge mode, non-fully-async — keeps
    the legacy fixed-name adapter update path unchanged.
    """

    return bool(
        getattr(args, "enable_versioned_lora_publication", False)
        and getattr(args, "use_agentic_rollout", False)
        and getattr(args, "fully_async", False)
        and getattr(args, "lora_adapter_mode", False)
    )


class VersionState(str, Enum):
    """Fleet-visible state of one logical version."""

    LOADING = "LOADING"
    PUBLISHED = "PUBLISHED"
    RETIRED = "RETIRED"
    RECLAIMING = "RECLAIMING"
    RECLAIMED = "RECLAIMED"
    FAILED_RETRYABLE = "FAILED_RETRYABLE"
    FAILED_FATAL = "FAILED_FATAL"


#: States that still own runtime resources (or a publication reservation) and
#: therefore count against the logical capacity.
CAPACITY_OWNING_STATES = frozenset(
    {
        VersionState.LOADING,
        VersionState.PUBLISHED,
        VersionState.RETIRED,
        VersionState.RECLAIMING,
        VersionState.FAILED_FATAL,
    }
)


class LoRAVersionError(RuntimeError):
    """Registry refusal.

    ``code`` is the machine-checkable reason.
    """

    def __init__(self, code: str, message: str = "") -> None:
        super().__init__(message or code)
        self.code = code


@dataclass
class VersionEntry:
    """One immutable version's identity plus its current fleet state."""

    version_id: int
    digest: str
    lora_name: str
    state: VersionState = VersionState.LOADING
    current_attempt_id: int = 1
    #: A reclaim whose outcome could not be confirmed; the version stays RECLAIMING
    #: (capacity NOT released) and the run fails closed.
    reclaim_fatal: bool = False
    driven: bool = False
    expected_default_revision: int = 0
    prepared_incarnations: Dict[str, str] = field(default_factory=dict)
    ready_incarnations: Dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class VersionBinding:
    """What a Session is bound to: an immutable, already-published version."""

    version_id: int
    digest: str
    lora_name: str


@dataclass(frozen=True)
class Publication:
    """Result of ``allocate``/``retry_publication``.

    ``no_op`` marks a completed publication of this exact version identity. It
    may no longer be the default; replay must not re-transport or re-commit.
    """

    version_id: int
    digest: str
    lora_name: str
    attempt_id: int
    no_op: bool = False


@dataclass
class RegistryStatus:
    """Read-only snapshot for logging and acceptance reporting."""

    deployment_epoch: str
    default_version: Optional[int]
    default_revision: int
    versions: Dict[int, dict] = field(default_factory=dict)
    session_bindings: Dict[str, int] = field(default_factory=dict)
    capacity_owning: int = 0


class LoRAVersionRegistry:
    """Single-writer state machine over immutable LoRA versions.

    Every mutating entry point is callable from a Ray actor method (or directly
    in tests). Methods never block on IO and never call back into publisher
    code.
    """

    def __init__(self, logical_capacity: int = 2, deployment_epoch: Optional[str] = None) -> None:
        self.logical_capacity = logical_capacity
        self.deployment_epoch = deployment_epoch or uuid.uuid4().hex[:12]
        self.versions: Dict[int, VersionEntry] = {}
        self.session_bindings: Dict[str, int] = {}
        #: Sessions that released before they ever bound. Kept as tombstones so a late
        #: first-bind of an already-closed Session cannot resurrect a Session ref.
        self.closed_sessions: Set[str] = set()
        self.default_version: Optional[int] = None
        self.default_revision = 0
        self.target_engines = frozenset({"engine0", "engine1"})

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _key(self, session_id: str) -> str:
        """Scope Session identity to this deployment epoch.

        Client-supplied session ids are reused across runs; the epoch prefix
        keeps a stale binding from a previous Registry lifetime out of this
        one.
        """

        return f"{self.deployment_epoch}:{session_id}"

    def _entry(self, version_id: int) -> VersionEntry:
        entry = self.versions.get(version_id)
        if entry is None:
            raise LoRAVersionError("UNKNOWN_VERSION", f"version {version_id} is not known in this epoch")
        return entry

    def _capacity_owning(self) -> int:
        return sum(1 for entry in self.versions.values() if entry.state in CAPACITY_OWNING_STATES)

    def _require_known_fleet(self) -> None:
        if any(entry.state is VersionState.FAILED_FATAL or entry.reclaim_fatal for entry in self.versions.values()):
            raise LoRAVersionError("PUBLICATION_BLOCKED", "unconfirmed engine state requires deployment recovery")

    def _admit_capacity(self) -> None:
        if self._capacity_owning() >= self.logical_capacity:
            raise LoRAVersionError(
                "CAPACITY_ERROR",
                f"{self.logical_capacity} managed LoRA versions already own resources; "
                "reclaim a retired version before publishing another",
            )

    def _require_attempt(self, entry: VersionEntry, attempt_id: int) -> None:
        """Current-attempt CAS: late replies from an old attempt must not
        mutate state."""

        if entry.current_attempt_id != attempt_id:
            raise LoRAVersionError(
                "ATTEMPT_CONFLICT",
                f"attempt {attempt_id} is stale for version {entry.version_id} (current {entry.current_attempt_id})",
            )

    def _make_name(self, version_id: int, digest: str) -> str:
        return f"{LORA_VERSION_NAME_PREFIX}@{self.deployment_epoch}-{version_id}-{digest[:16]}"

    def _publication(self, entry: VersionEntry, *, no_op: bool = False) -> Publication:
        return Publication(
            version_id=entry.version_id,
            digest=entry.digest,
            lora_name=entry.lora_name,
            attempt_id=entry.current_attempt_id,
            no_op=no_op,
        )

    def _published_default(self) -> VersionEntry:
        if self.default_version is None:
            raise LoRAVersionError(
                "NO_PUBLISHED_VERSION",
                "no managed LoRA version is published yet; refusing to fall back to the base model",
            )
        return self._entry(self.default_version)

    # ------------------------------------------------------------------
    # Publication
    # ------------------------------------------------------------------

    def allocate(self, digest: str, *, version_id: int) -> Publication:
        """Reserve a caller-selected version identity within this epoch.

        The caller fixes the ID before its first reservation and reuses it on
        retries. Digest validates that identity's immutable content; a new ID
        may carry the same digest as any historical version. Capacity is
        checked before admitting a new or retryable publication.
        """

        if type(version_id) is not int or version_id <= 0:
            raise LoRAVersionError("INVALID_VERSION", "version_id must be a positive integer")
        live = self.versions.get(version_id)
        if live is not None and live.digest != digest:
            raise LoRAVersionError("VERSION_CONFLICT", f"version {version_id} already holds a different digest")
        self._require_known_fleet()

        if live is not None and live.state in {
            VersionState.PUBLISHED,
            VersionState.RETIRED,
            VersionState.RECLAIMING,
            VersionState.RECLAIMED,
        }:
            # Completed publications remain idempotency records after retirement.
            # A replay must neither reload resources nor roll back the default.
            return self._publication(live, no_op=True)

        if live is not None and live.state is VersionState.LOADING:
            # Duplicate request for the same version: report the state we
            # already have instead of loading, transferring or committing twice.
            return self._publication(live)

        if live is not None and live.state is VersionState.FAILED_RETRYABLE:
            # Same content, previous attempt cleaned itself back to ABSENT: retry in
            # place (same version id, same name, next attempt) under fresh admission.
            return self.retry_publication(live.version_id, digest)

        if live is not None and live.state is VersionState.FAILED_FATAL:
            raise LoRAVersionError("VERSION_FATAL", f"version {live.version_id} failed fatally for this content")

        self._admit_capacity()
        if any(entry.state is VersionState.LOADING for entry in self.versions.values()):
            raise LoRAVersionError("PUBLICATION_IN_PROGRESS", "another publication attempt is unresolved")
        entry = VersionEntry(
            version_id=version_id,
            digest=digest,
            lora_name=self._make_name(version_id, digest),
            expected_default_revision=self.default_revision,
        )
        self.versions[entry.version_id] = entry
        return self._publication(entry)

    def retry_publication(self, version_id: int, digest: str) -> Publication:
        """Start a clean transport attempt for an already-reserved version.

        Only a ``FAILED_RETRYABLE`` entry may retry, and only after its predecessors
        confirmed every engine back to ABSENT. Capacity is re-admitted in the same
        turn: once other versions took the slots, the retry is refused unchanged.
        """

        entry = self._entry(version_id)
        if entry.digest != digest:
            raise LoRAVersionError(
                "VERSION_CONFLICT",
                f"version {version_id} already holds a different digest",
            )
        if entry.state in {
            VersionState.PUBLISHED,
            VersionState.RETIRED,
            VersionState.RECLAIMING,
            VersionState.RECLAIMED,
        }:
            return self._publication(entry, no_op=True)
        if entry.state is not VersionState.FAILED_RETRYABLE:
            raise LoRAVersionError(
                "INVALID_STATE",
                f"version {version_id} is {entry.state.value}; only FAILED_RETRYABLE may retry",
            )
        self._require_known_fleet()
        self._admit_capacity()
        if any(other.state is VersionState.LOADING for other in self.versions.values()):
            raise LoRAVersionError("PUBLICATION_IN_PROGRESS", "another publication attempt is unresolved")
        entry.current_attempt_id += 1
        entry.state = VersionState.LOADING
        entry.driven = False
        entry.expected_default_revision = self.default_revision
        entry.prepared_incarnations.clear()
        entry.ready_incarnations.clear()
        return self._publication(entry)

    def claim_publication(self, version_id: int, attempt_id: int) -> None:
        """Grant exactly one caller permission to send this attempt's
        collectives.

        Losing the reply or the publisher never grants another driver
        permission to replay it. Recovery requires a confirmed cleanup and a
        fresh attempt.
        """
        entry = self._entry(version_id)
        self._require_attempt(entry, attempt_id)
        self._require_known_fleet()
        if entry.state is not VersionState.LOADING or entry.driven:
            raise LoRAVersionError("PUBLICATION_IN_PROGRESS", "this attempt already has a transport owner")
        entry.driven = True

    def _receipt_incarnations(self, entry: VersionEntry, receipts: Dict[str, dict], state: str) -> Dict[str, str]:
        if set(receipts) != self.target_engines:
            raise LoRAVersionError("FLEET_MISMATCH", "both fixed target engines must acknowledge publication")
        identity = {
            "lora_name": entry.lora_name,
            "version_id": entry.version_id,
            "digest": entry.digest,
            "attempt_id": entry.current_attempt_id,
        }
        incarnations = {}
        for engine, receipt in receipts.items():
            if any(receipt.get(key) != value for key, value in identity.items()) or receipt.get("state") != state:
                raise LoRAVersionError("READY_MISMATCH", f"{engine} acknowledged a different publication")
            incarnation = receipt.get("engine_incarnation")
            if not isinstance(incarnation, str) or not incarnation:
                raise LoRAVersionError("READY_MISMATCH", f"{engine} omitted its incarnation")
            incarnations[engine] = incarnation
        return incarnations

    def record_prepared(self, version_id: int, attempt_id: int, receipts: Dict[str, dict]) -> None:
        entry = self._entry(version_id)
        self._require_attempt(entry, attempt_id)
        if entry.state is not VersionState.LOADING or not entry.driven:
            raise LoRAVersionError("INVALID_STATE", "publication transport has not been claimed")
        incarnations = self._receipt_incarnations(entry, receipts, "PREPARED")
        if entry.prepared_incarnations and entry.prepared_incarnations != incarnations:
            raise LoRAVersionError("ENGINE_RESTARTED", "prepared engine incarnation changed")
        entry.prepared_incarnations = incarnations

    def record_ready(self, version_id: int, attempt_id: int, receipts: Dict[str, dict]) -> None:
        entry = self._entry(version_id)
        self._require_attempt(entry, attempt_id)
        if entry.state is not VersionState.LOADING:
            raise LoRAVersionError("INVALID_STATE", "only a loading publication can become ready")
        incarnations = self._receipt_incarnations(entry, receipts, "READY_LOCAL")
        if incarnations != entry.prepared_incarnations:
            raise LoRAVersionError("ENGINE_RESTARTED", "READY belongs to a different engine incarnation")
        entry.ready_incarnations = incarnations

    def mark_published(self, version_id: int, attempt_id: int) -> Publication:
        """Fleet commit: the linearization point for "new Sessions use B".

        The predecessor goes RETIRED (not unloaded) so its bound Sessions keep
        generating; only this call moves the default.
        """

        entry = self._entry(version_id)
        if entry.state is VersionState.PUBLISHED:
            # Duplicate commit of the same attempt (or a replay after a lost reply).
            return self._publication(entry)
        if entry.state in (VersionState.RETIRED, VersionState.RECLAIMING, VersionState.RECLAIMED):
            # An already-committed-then-retired version must never become default again
            # because an old commit was retried.
            return self._publication(entry)
        if entry.state is not VersionState.LOADING:
            raise LoRAVersionError("INVALID_STATE", f"version {version_id} is {entry.state.value}")
        self._require_attempt(entry, attempt_id)

        self._require_known_fleet()
        if set(entry.ready_incarnations) != self.target_engines:
            raise LoRAVersionError("NOT_READY", "both engines must be ready before commit")
        if entry.expected_default_revision != self.default_revision:
            raise LoRAVersionError("REVISION_CONFLICT", "default changed since publication reservation")
        previous = self.default_version
        entry.state = VersionState.PUBLISHED
        self.default_version = version_id
        self.default_revision += 1
        if previous is not None and previous != version_id:
            predecessor = self.versions.get(previous)
            if predecessor is not None and predecessor.state is VersionState.PUBLISHED:
                predecessor.state = VersionState.RETIRED
        return self._publication(entry)

    def mark_retryable_failure(self, version_id: int, attempt_id: int) -> None:
        """Terminal clean failure: every engine confirmed the candidate
        ABSENT."""

        entry = self._entry(version_id)
        self._require_attempt(entry, attempt_id)
        if entry.state is not VersionState.LOADING:
            raise LoRAVersionError("INVALID_STATE", f"version {version_id} is {entry.state.value}")
        entry.state = VersionState.FAILED_RETRYABLE

    def mark_fatal_failure(self, version_id: int, attempt_id: int) -> None:
        """Dirty or ambiguous backend/transport state: fail the run closed."""

        entry = self._entry(version_id)
        self._require_attempt(entry, attempt_id)
        entry.state = VersionState.FAILED_FATAL

    def mark_default_failure(self, version_id: int, reason: str) -> None:
        """The live default itself is unusable; the run cannot continue.

        Recorded (not silently ignored) so the publisher's fail-closed path and
        the acceptance report can both point at the version that broke.
        """

        entry = self._entry(version_id)
        entry.state = VersionState.FAILED_FATAL
        if self.default_version == version_id:
            self.default_version = None

    # ------------------------------------------------------------------
    # Session binding
    # ------------------------------------------------------------------

    def bind_latest(self, session_id: str) -> VersionBinding:
        """Bind a Session to the current default, once.

        Reads the published default and records the Session reference in the
        same actor turn, so the ordering against ``mark_published`` is well
        defined: a Session either binds A (before commit) or B (after), never a
        mix.
        """

        key = self._key(session_id)
        bound = self.session_bindings.get(key)
        if bound is not None:
            entry = self._entry(bound)
            return VersionBinding(entry.version_id, entry.digest, entry.lora_name)
        if key in self.closed_sessions:
            raise LoRAVersionError(
                "SESSION_CLOSED",
                f"session {session_id} is already closed; a late bind must not hold a new reference",
            )

        entry = self._published_default()
        self.session_bindings[key] = entry.version_id
        return VersionBinding(entry.version_id, entry.digest, entry.lora_name)

    def release(self, session_id: str) -> bool:
        """Drop one Session reference; idempotent (repeat closes must not
        double-release).

        A release that arrives before the Session's first bind leaves a
        tombstone: the reference count must not go negative, and a late bind
        must not re-open it.
        """

        key = self._key(session_id)
        self.closed_sessions.add(key)
        return self.session_bindings.pop(key, None) is not None

    # ------------------------------------------------------------------
    # Reclaim
    # ------------------------------------------------------------------

    def claim_reclaimable(self) -> Optional[VersionBinding]:
        """Atomically claim a retired version nobody can request any more.

        Exactly one caller can win a version, so the reclaim fan-out is sent at
        most once per version. ``RECLAIMED`` releases the logical slot;
        ``FAILED_RETRYABLE`` already released it earlier.
        """

        for version_id in sorted(self.versions):
            entry = self.versions[version_id]
            if entry.state is not VersionState.RETIRED:
                continue
            if version_id in self.session_bindings.values():
                continue
            entry.state = VersionState.RECLAIMING
            return VersionBinding(entry.version_id, entry.digest, entry.lora_name)
        return None

    def mark_reclaimed(self, version_id: int) -> None:
        entry = self._entry(version_id)
        if entry.state is VersionState.RECLAIMED:
            return
        if entry.reclaim_fatal:
            # A late ACK after an ambiguous reclaim must not resurrect the run.
            raise LoRAVersionError("RECLAIM_AMBIGUOUS", f"version {version_id} reclaim already marked fatal")
        if entry.state is not VersionState.RECLAIMING:
            raise LoRAVersionError("INVALID_STATE", f"version {version_id} is {entry.state.value}")
        entry.state = VersionState.RECLAIMED

    def mark_reclaim_fatal(self, version_id: int) -> None:
        """Ambiguous unload completion: keep RECLAIMING (capacity stays
        occupied).

        The publisher must not retry the unload, so the version stays out of
        ``claim_reclaimable`` forever; the caller fails the run closed.
        """

        self._entry(version_id).reclaim_fatal = True

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def status(self) -> RegistryStatus:
        return RegistryStatus(
            deployment_epoch=self.deployment_epoch,
            default_version=self.default_version,
            default_revision=self.default_revision,
            versions={
                version_id: {
                    "digest": entry.digest,
                    "lora_name": entry.lora_name,
                    "state": entry.state.value,
                    "attempt_id": entry.current_attempt_id,
                    "reclaim_fatal": entry.reclaim_fatal,
                }
                for version_id, entry in self.versions.items()
            },
            session_bindings=dict(self.session_bindings),
            capacity_owning=self._capacity_owning(),
        )

    def bound_versions(self) -> Set[int]:
        return set(self.session_bindings.values())
