# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Fixed-cohort publication; execution ownership stays in the native engine.

Async entry points run on the RolloutManager's single control loop. A short
thread lock also protects synchronous bind/status calls from Ray concurrency
groups. No disk or network work runs under that lock.
"""

from __future__ import annotations

import asyncio
import math
import threading
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Protocol, Sequence
from uuid import NAMESPACE_URL, uuid4, uuid5

from relax.engine.lora.snapshot import AdapterSnapshot


class AdapterPublicationError(RuntimeError):
    def __init__(self, code: str, message: str = "", status_code: int = 409) -> None:
        super().__init__(message or code)
        self.code = code
        self.status_code = status_code


@dataclass(frozen=True)
class PublicationConfig:
    artifact_store: str
    target_model: str = "default"
    capacity: int = 2
    engines_per_gpu: int = 1
    bootstrap_version_id: str | None = None
    export_every_n_steps: int | None = None
    auto_publish: bool = False
    prepare_timeout_seconds: float = 120
    cleanup_timeout_seconds: float = 120
    artifact_max_bytes: int = 8589934592
    max_lifecycle_records: int = 100000

    @classmethod
    def read(cls, path: str) -> "PublicationConfig":
        import yaml

        with Path(path).open() as source:
            value = yaml.safe_load(source)
        if not isinstance(value, dict):
            raise ValueError("publication config must be an object")
        config = cls(**value)
        for name in ("capacity", "engines_per_gpu", "artifact_max_bytes", "max_lifecycle_records"):
            if type(getattr(config, name)) is not int or getattr(config, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if any(
            type(value) not in (int, float) or not math.isfinite(value) or value <= 0
            for value in (config.prepare_timeout_seconds, config.cleanup_timeout_seconds)
        ):
            raise ValueError("publication timeouts must be positive")
        if config.export_every_n_steps is not None and (
            type(config.export_every_n_steps) is not int or config.export_every_n_steps <= 0
        ):
            raise ValueError("export_every_n_steps must be a positive integer or null")
        if not config.artifact_store or not config.target_model or type(config.auto_publish) is not bool:
            raise ValueError("invalid publication store, target model or auto_publish")
        if config.bootstrap_version_id is not None:
            AdapterSnapshot(config.bootstrap_version_id, "0" * 64, "0" * 64, Path("."))
        return config

    def validate_args(self, args) -> None:
        required = {"use_agentic_rollout": True, "lora_adapter_mode": True}
        forbidden = (
            "agentic_program_admission",
            "use_fault_tolerance",
            "sglang_config",
            "rollout_engine_class_path",
            "use_opd",
            "debug_train_only",
            "debug_rollout_only",
            "true_on_policy_mode",
            "mask_offpolicy_in_partial_rollout",
        )
        if any(getattr(args, name, False) != value for name, value in required.items()):
            raise ValueError("publication requires Agentic rollout with LoRA adapter mode")
        # Hybrid reuses the colocate flag for local actor/ref switching,
        # while its rollout engines use a separate resource group.
        shared_rollout = getattr(args, "colocate", False) and not getattr(args, "hybrid", False)
        if shared_rollout and not (getattr(args, "offload_train", False) and getattr(args, "offload_rollout", False)):
            raise ValueError("shared training/rollout GPUs require both train and rollout offload")
        if any(getattr(args, name, False) for name in forbidden):
            raise ValueError(
                "publication requires managed inference engines without program admission/automatic recovery; "
                "step-based on-policy shortcuts cannot classify retained adapter sessions"
            )
        per_engine = args.rollout_num_gpus_per_engine
        if (
            per_engine < 1
            or args.rollout_num_gpus % per_engine
            or args.rollout_num_gpus * self.engines_per_gpu // per_engine < 2
        ):
            raise ValueError("publication requires resources for at least two complete engines")
        if self.engines_per_gpu > 1:
            fraction = getattr(args, "sglang_mem_fraction_static", None)
            if per_engine != 1 or type(fraction) not in (int, float) or not 0 < fraction < 1 / self.engines_per_gpu:
                raise ValueError(
                    "shared-GPU engines require TP=1 and an explicit per-engine "
                    "--sglang-mem-fraction-static below 1 / engines_per_gpu"
                )
        if self.target_model != "default":
            raise ValueError("initial profile manages the default policy engine cohort")
        if getattr(args, "train_backend", "megatron") != "megatron":
            raise ValueError("initial publication profile requires Megatron")


class VersionState(str, Enum):
    PREPARING = "PREPARING"
    PUBLISHED = "PUBLISHED"
    RETIRING = "RETIRING"
    ABORTED = "ABORTED"
    RETIRED = "RETIRED"


@dataclass(frozen=True)
class EngineIdentity:
    engine_id: str
    boot_id: str
    endpoint: str


@dataclass(frozen=True)
class EngineReceipt:
    engine: EngineIdentity
    cohort_id: str
    operation_id: str
    version_id: str
    digest: str
    native_lora_id: str | None
    state: str
    pinned: bool = False
    resident: bool = False
    fenced: bool = False
    actual_unload_count: int | None = None
    observation: dict | None = None


@dataclass(frozen=True)
class SessionReceipt:
    engine: EngineIdentity
    cohort_id: str
    owner_epoch: str
    session_id: str
    state: str


def native_instance_id(engine: EngineIdentity, cohort_id: str, operation_id: str) -> str:
    """Preallocate the same opaque instance before prepare or retire RPCs."""
    import json

    return uuid5(NAMESPACE_URL, json.dumps([cohort_id, engine.engine_id, engine.boot_id, operation_id])).hex


class AdapterEngine(Protocol):
    """Bounded transport calls to a fixed native boot, never health discovery.

    Native prepare/retire/close tasks survive an HTTP waiter's cancellation.
    ABSENT includes a fence for delayed prepare, and SESSION_DRAINED includes
    the Session fence, execution drain and KV-close completion.
    """

    @property
    def identity(self) -> EngineIdentity: ...

    async def prepare(self, snapshot: AdapterSnapshot, cohort_id: str, operation_id: str) -> EngineReceipt: ...

    async def status(self, snapshot: AdapterSnapshot, cohort_id: str, operation_id: str) -> EngineReceipt: ...

    async def retire(self, snapshot: AdapterSnapshot, cohort_id: str, operation_id: str) -> EngineReceipt: ...

    async def close_session(self, cohort_id: str, owner_epoch: str, session_id: str) -> SessionReceipt: ...


@dataclass(frozen=True)
class AdapterBinding:
    cohort_id: str
    version_id: str
    digest: str
    publication_id: str
    publication_epoch: int
    lora_path: str
    source_train_step: int | None


@dataclass
class _Version:
    snapshot: AdapterSnapshot
    operation_id: str
    expected_epoch: int | None
    source_train_step: int | None
    deadline: float
    publish_targets: tuple[str, ...] = ()
    targets: set[str] = field(default_factory=set)
    state: VersionState = VersionState.PREPARING
    binding: AdapterBinding | None = None
    sessions: int = 0
    ready: dict[str, EngineReceipt] = field(default_factory=dict)
    absent: set[str] = field(default_factory=set)
    absence_receipts: dict[str, EngineReceipt] = field(default_factory=dict)
    instances: dict[str, str] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)
    cancelled: bool = False
    started: float = field(default_factory=time.monotonic)
    publication_seconds: float | None = None
    prepare_task: asyncio.Task | None = None
    retire_task: asyncio.Task | None = None


@dataclass
class _Session:
    owner_epoch: str
    session_id: str
    binding: AdapterBinding | None = None
    engine: EngineIdentity | None = None
    native_lora_id: str | None = None
    targets: frozenset[str] = frozenset()
    closing: bool = False
    closed: bool = False
    drained: set[str] = field(default_factory=set)
    errors: dict[str, str] = field(default_factory=dict)
    close_task: asyncio.Task | None = None


def _observe(task: asyncio.Task) -> None:
    if not task.cancelled():
        task.exception()


class AdapterVersionManager:
    """One default pointer, bounded history, Session references and GPU slots.

    This owner is intentionally not recoverable from an empty table against old
    processes. Cohort and native boot identities must be recreated together
    after coordinator loss. It owns no per-request lease or output ledger.
    """

    def __init__(
        self,
        engines: Sequence[AdapterEngine],
        *,
        cohort_id: str,
        capacity: int,
        base_model_digest: str,
        max_lifecycle_records: int = 100000,
        prepare_timeout_seconds: float = 120,
        cleanup_timeout_seconds: float = 120,
        validate_snapshot: Callable[[AdapterSnapshot], int | None] | None = None,
    ) -> None:
        if not engines or not cohort_id or capacity < 1 or max_lifecycle_records < 1:
            raise ValueError("a cohort, engines and positive capacities are required")
        if prepare_timeout_seconds <= 0 or cleanup_timeout_seconds <= 0:
            raise ValueError("operation budgets must be positive")
        identities = [engine.identity for engine in engines]
        if len({identity.engine_id for identity in identities}) != len(engines):
            raise ValueError("duplicate engine ID")
        if any(not item.engine_id or not item.boot_id or not item.endpoint for item in identities):
            raise ValueError("engine ID, native boot ID and endpoint are required")
        self.cohort_id = cohort_id
        self._engines = {engine.identity.engine_id: engine for engine in engines}
        self._identities = {identity.engine_id: identity for identity in identities}
        self._initial_engine_count = len(engines)
        self._capacity = capacity
        self._base = base_model_digest
        self._limit = max_lifecycle_records
        self._prepare_timeout = prepare_timeout_seconds
        self._cleanup_timeout = cleanup_timeout_seconds
        self._validate_snapshot = validate_snapshot
        self._lock = threading.RLock()
        self._loop = asyncio.get_running_loop()
        self._versions: dict[str, _Version] = {}
        self._operations: dict[str, _Version] = {}
        self._intents: dict[str, tuple[tuple, str]] = {}
        self._sessions: dict[tuple[str, str], _Session] = {}
        self._session_owners: dict[str, str] = {}
        self._unavailable: set[str] = set()
        self._engine_sessions = dict.fromkeys(self._engines, 0)
        self._serving = set(self._engines)
        self._topology_epoch = 0
        self._detached: set[str] = set()
        self._topology_task: asyncio.Task | None = None
        self._topology_key: tuple | None = None
        self._topology_phase = "IDLE"
        self._live_operations: set[str] = set()
        self._closing_sessions: set[tuple[str, str]] = set()
        self._accepting = True
        self._suspended = False
        self._default: AdapterBinding | None = None
        self._epoch = 0
        self._active: str | None = None

    def _check_loop(self) -> None:
        if asyncio.get_running_loop() is not self._loop:
            raise RuntimeError("submit async publication operations to the owner's control loop")

    def _task(self, coroutine) -> asyncio.Task:
        task = self._loop.create_task(coroutine)
        task.add_done_callback(_observe)
        return task

    def _usage(self) -> int:
        return (
            len(self._versions)
            + len(self._operations)
            + len(self._intents)
            + len(self._sessions)
            + len(self._identities)
            - self._initial_engine_count
        )

    def _reserve_metadata(self, count: int) -> None:
        if not self._accepting or self._usage() + count > self._limit:
            raise AdapterPublicationError("LIFECYCLE_CAPACITY_EXCEEDED", status_code=507)

    def _occupancy(self) -> int:
        return len(self._live_operations)

    def _intent(self, request_id: str, payload: tuple) -> str | None:
        previous = self._intents.get(request_id)
        if previous is None:
            return None
        if previous[0] != payload:
            raise AdapterPublicationError("REQUEST_ID_CONFLICT")
        return previous[1]

    def publication_intent(
        self,
        request_id: str,
        version_id: str,
        digest: str | None = None,
        *,
        retry_of: str | None = None,
        expected_default_epoch: int | None = None,
    ) -> dict | None:
        """Replay accepted intent before touching files or reserving
        capacity."""
        if not isinstance(request_id, str) or not request_id or len(request_id) > 128:
            raise ValueError("request_id must be a nonempty string of at most 128 characters")
        with self._lock:
            previous = self._intents.get(request_id)
            if previous is None:
                return None
            accepted, operation = previous
            candidate = (
                version_id,
                accepted[1] if digest is None else digest,
                self._base,
                retry_of,
                expected_default_epoch,
            )
            if candidate != accepted:
                raise AdapterPublicationError("REQUEST_ID_CONFLICT")
            return self._operation_status(self._operations[operation])

    def _confirm(self, snapshot: AdapterSnapshot) -> int | None:
        if snapshot.base_model_digest != self._base:
            raise AdapterPublicationError("BASE_MODEL_MISMATCH")
        snapshot.confirm_sealed()
        return self._validate_snapshot(snapshot) if self._validate_snapshot is not None else None

    async def publish(
        self,
        snapshot: AdapterSnapshot,
        request_id: str,
        *,
        retry_of: str | None = None,
        expected_default_epoch: int | None = None,
    ) -> dict:
        self._check_loop()
        if not isinstance(request_id, str) or not request_id or len(request_id) > 128:
            raise ValueError("request_id must be a nonempty string of at most 128 characters")
        if retry_of is not None and (not isinstance(retry_of, str) or not retry_of or len(retry_of) > 128):
            raise ValueError("invalid retry_of")
        if expected_default_epoch is not None and (
            type(expected_default_epoch) is not int or expected_default_epoch < 0
        ):
            raise ValueError("expected_default_epoch must be a nonnegative integer")
        payload = (snapshot.version_id, snapshot.digest, snapshot.base_model_digest, retry_of, expected_default_epoch)
        with self._lock:
            operation = self._intent(request_id, payload)
            if operation is not None:
                return self._operation_status(self._operations[operation])
            previous = self._versions.get(snapshot.version_id)
            if previous is not None and previous.snapshot.digest != snapshot.digest:
                raise AdapterPublicationError("VERSION_CONTENT_CONFLICT")
        # No accepted operation exists yet; cancellation here only abandons
        # validation. Accepted tasks below have independent ownership.
        source_step = await asyncio.to_thread(self._confirm, snapshot)
        with self._lock:
            operation = self._intent(request_id, payload)
            if operation is not None:
                return self._operation_status(self._operations[operation])
            previous = self._versions.get(snapshot.version_id)
            if previous is not None:
                if previous.snapshot.digest != snapshot.digest:
                    raise AdapterPublicationError("VERSION_CONTENT_CONFLICT")
                if retry_of is None:
                    self._reserve_metadata(1)
                    self._intents[request_id] = (payload, previous.operation_id)
                    return self._operation_status(previous)
                if retry_of != previous.operation_id or previous.state != VersionState.ABORTED:
                    raise AdapterPublicationError("INVALID_PUBLICATION_RETRY")
            elif retry_of is not None:
                raise AdapterPublicationError("INVALID_PUBLICATION_RETRY")
            if self._suspended:
                raise AdapterPublicationError("ENGINE_SUSPENDED", status_code=503)
            if self._topology_busy():
                raise AdapterPublicationError("MEMBERSHIP_BUSY")
            if self._active is not None:
                raise AdapterPublicationError("PUBLICATION_BUSY")
            if expected_default_epoch is not None and expected_default_epoch != self._epoch:
                raise AdapterPublicationError("DEFAULT_EPOCH_CONFLICT")
            if self._unavailable & self._serving:
                raise AdapterPublicationError("ENGINE_UNAVAILABLE", status_code=503)
            if self._occupancy() >= self._capacity:
                raise AdapterPublicationError("ADAPTER_CAPACITY_EXCEEDED", status_code=507)
            self._reserve_metadata(2 + (previous is None))
            record = _Version(
                snapshot,
                uuid4().hex,
                expected_default_epoch,
                source_step,
                self._loop.time() + self._prepare_timeout,
                publish_targets=tuple(sorted(self._serving)),
                targets=set(self._serving),
            )
            record.instances = {
                key: native_instance_id(self._identities[key], self.cohort_id, record.operation_id)
                for key in record.targets
            }
            self._versions[snapshot.version_id] = record
            self._operations[record.operation_id] = record
            self._live_operations.add(record.operation_id)
            self._intents[request_id] = (payload, record.operation_id)
            self._active = record.operation_id
            record.prepare_task = self._task(self._prepare(record))
            return self._operation_status(record)

    def _validate_receipt_identity(self, record: _Version, engine_id: str, receipt: EngineReceipt) -> None:
        if (
            receipt.engine != self._identities[engine_id]
            or receipt.cohort_id != self.cohort_id
            or receipt.operation_id != record.operation_id
            or receipt.version_id != record.snapshot.version_id
            or receipt.digest != record.snapshot.digest
        ):
            raise AdapterPublicationError("ENGINE_RECEIPT_MISMATCH")

    def _validate_receipt(self, record: _Version, engine_id: str, receipt: EngineReceipt, *, absent: bool) -> None:
        self._validate_receipt_identity(record, engine_id, receipt)
        if receipt.native_lora_id != record.instances[engine_id]:
            raise AdapterPublicationError("NATIVE_INSTANCE_MISMATCH")
        if absent:
            if receipt.state != "ABSENT" or not receipt.fenced:
                raise AdapterPublicationError("ABSENCE_UNCONFIRMED")
        elif receipt.state != "READY" or not receipt.pinned or not receipt.resident or not receipt.native_lora_id:
            raise AdapterPublicationError("ADAPTER_NOT_READY")

    async def _prepare_engine(
        self, record: _Version, engine: AdapterEngine, *, deadline: float | None = None
    ) -> EngineReceipt:
        deadline = record.deadline if deadline is None else deadline
        first = True
        while True:
            with self._lock:
                remaining = deadline - self._loop.time()
                if remaining <= 0:
                    raise AdapterPublicationError("PREPARE_DEADLINE_EXCEEDED")
                if record.cancelled:
                    raise AdapterPublicationError("PUBLICATION_CANCELLED")
            call = engine.prepare if first else engine.status
            # Initial preparation includes target-side artifact verification.
            # Status polling is cheap; it must not set the preparation budget.
            timeout = remaining if first else min(5.0, remaining)
            first = False
            try:
                receipt = await asyncio.wait_for(call(record.snapshot, self.cohort_id, record.operation_id), timeout)
                if receipt.state == "READY":
                    return receipt
                self._validate_receipt_identity(record, engine.identity.engine_id, receipt)
                if receipt.state in {"ABSENT", "PREPARE_FAILED", "LOAD_FAILED", "FENCED", "CLEANUP_PENDING"}:
                    raise AdapterPublicationError("ENGINE_PREPARE_FAILED", receipt.state)
            except (TimeoutError, ConnectionError):
                pass  # A transport timeout is not the publication deadline.
            await asyncio.sleep(min(0.05, max(0, deadline - self._loop.time())))

    async def _prepare(self, record: _Version) -> None:
        try:
            # A single total budget; sequential loading keeps the other engine
            # available while native disk parsing runs on one scheduler.
            for engine_id in record.publish_targets:
                engine = self._engines[engine_id]
                with self._lock:
                    if record.cancelled:
                        raise AdapterPublicationError("PUBLICATION_CANCELLED")
                receipt = await self._prepare_engine(record, engine)
                with self._lock:
                    self._validate_receipt(record, engine_id, receipt, absent=False)
                    record.ready[engine_id] = receipt
            with self._lock:
                if self._loop.time() >= record.deadline:
                    record.cancelled = True
                    raise AdapterPublicationError("PREPARE_DEADLINE_EXCEEDED")
                if record.cancelled or self._active != record.operation_id:
                    raise AdapterPublicationError("PUBLICATION_CANCELLED")
                if self._unavailable & self._serving:
                    raise AdapterPublicationError("ENGINE_UNAVAILABLE", status_code=503)
                if record.expected_epoch is not None and record.expected_epoch != self._epoch:
                    raise AdapterPublicationError("DEFAULT_EPOCH_CONFLICT")
                self._epoch += 1
                record.binding = AdapterBinding(
                    self.cohort_id,
                    record.snapshot.version_id,
                    record.snapshot.digest,
                    record.operation_id,
                    self._epoch,
                    record.snapshot.lora_path,
                    record.source_train_step,
                )
                # The same lock serializes first bind and this single commit.
                self._default = record.binding
                record.state = VersionState.PUBLISHED
                record.publication_seconds = time.monotonic() - record.started
                self._active = None
        except Exception as error:
            with self._lock:
                record.errors["prepare"] = f"{type(error).__name__}: {error}"
                record.state = VersionState.RETIRING
                self._start_retire(record)
        await self.collect(report=False)

    async def cancel_publication(self, operation_id: str) -> dict:
        self._check_loop()
        with self._lock:
            record = self._operations[operation_id]
            if record.binding is not None:
                raise AdapterPublicationError("ALREADY_COMMITTED")
            record.cancelled = True
            if record.state != VersionState.ABORTED:
                record.state = VersionState.RETIRING
                self._start_retire(record)
            return self._operation_status(record)

    def bind_session(self, owner_epoch: str, session_id: str) -> dict:
        if not owner_epoch or not session_id:
            raise ValueError("owner_epoch and session_id are required")
        with self._lock:
            session = self._sessions.get((owner_epoch, session_id))
            if session is not None:
                if session.closing:
                    raise AdapterPublicationError("SESSION_CLOSED")
                return self._session_status(session)
            if session_id in self._session_owners:
                raise AdapterPublicationError("SESSION_ID_CONFLICT")
            if self._suspended:
                raise AdapterPublicationError("ENGINE_SUSPENDED", status_code=503)
            if self._default is None:
                raise AdapterPublicationError("NO_PUBLISHED_VERSION", status_code=503)
            self._reserve_metadata(1)
            record = self._operations[self._default.publication_id]
            candidates = [key for key in self._serving if key not in self._unavailable and key in record.ready]
            if not candidates:
                raise AdapterPublicationError("ENGINE_UNAVAILABLE", status_code=503)
            key = min(candidates, key=lambda item: (self._engine_sessions[item], item))
            session = _Session(
                owner_epoch,
                session_id,
                self._default,
                self._identities[key],
                record.ready[key].native_lora_id,
                targets=frozenset(self._serving),
            )
            self._sessions[(owner_epoch, session_id)] = session
            self._session_owners[session_id] = owner_epoch
            record.sessions += 1
            self._engine_sessions[key] += 1
            return self._session_status(session)

    async def close_session(self, owner_epoch: str, session_id: str) -> dict:
        self._check_loop()
        if not owner_epoch or not session_id:
            raise ValueError("owner_epoch and session_id are required")
        with self._lock:
            session = self._sessions.get((owner_epoch, session_id))
            if session is None:
                if session_id in self._session_owners:
                    raise AdapterPublicationError("SESSION_ID_CONFLICT")
                try:
                    self._reserve_metadata(1)
                except AdapterPublicationError:
                    # Unknown close must not be acknowledged without a fence.
                    # No binding ever existed for this ID; shut new RM binds so
                    # its delayed first-bind cannot revive it without storage.
                    self._accepting = False
                    raise
                session = _Session(owner_epoch, session_id, targets=frozenset(self._serving))
                self._sessions[(owner_epoch, session_id)] = session
                self._session_owners[session_id] = owner_epoch
            session.closing = True
            if not session.closed:
                self._closing_sessions.add((owner_epoch, session_id))
            self._start_close(session)
            return self._session_status(session)

    def _start_close(self, session: _Session) -> None:
        if not self._suspended and not session.closed and (session.close_task is None or session.close_task.done()):
            session.close_task = self._task(self._close(session))

    async def _close(self, session: _Session) -> None:
        async def close_one(key: str, engine: AdapterEngine | None) -> None:
            with self._lock:
                if key in self._detached:
                    # Every loaded instance was fenced and ABSENT before detach.
                    session.drained.add(key)
                    session.errors.pop(key, None)
                if key in session.drained:
                    return
            try:
                if engine is None:
                    raise AdapterPublicationError("ENGINE_OWNER_UNAVAILABLE")
                receipt = await asyncio.wait_for(
                    engine.close_session(self.cohort_id, session.owner_epoch, session.session_id),
                    self._cleanup_timeout,
                )
                if receipt != SessionReceipt(
                    self._identities[key], self.cohort_id, session.owner_epoch, session.session_id, "SESSION_DRAINED"
                ):
                    raise AdapterPublicationError("SESSION_DRAIN_UNCONFIRMED")
                with self._lock:
                    session.drained.add(key)
                    session.errors.pop(key, None)
            except Exception as error:
                with self._lock:
                    session.errors[key] = f"{type(error).__name__}: {error}"

        await asyncio.gather(*(close_one(key, self._engines.get(key)) for key in session.targets))
        with self._lock:
            if session.targets <= session.drained and not session.closed:
                session.closed = True
                self._closing_sessions.discard((session.owner_epoch, session.session_id))
                if session.engine is not None:
                    self._engine_sessions[session.engine.engine_id] -= 1
                if session.binding is not None:
                    self._operations[session.binding.publication_id].sessions -= 1
            # collect() must not restart this very task while it is returning.
        await self.collect(report=False)

    def _start_retire(self, record: _Version) -> None:
        if not self._suspended and (record.retire_task is None or record.retire_task.done()):
            record.retire_task = self._task(self._retire(record))

    async def _retire_engine(self, record: _Version, key: str) -> None:
        with self._lock:
            if key in record.absent:
                return
        try:
            receipt = await asyncio.wait_for(
                self._engines[key].retire(record.snapshot, self.cohort_id, record.operation_id),
                self._cleanup_timeout,
            )
            with self._lock:
                self._validate_receipt(record, key, receipt, absent=True)
                record.absent.add(key)
                record.absence_receipts[key] = receipt
                record.errors.pop(key, None)
        except Exception as error:
            with self._lock:
                record.errors[key] = f"{type(error).__name__}: {error}"

    async def _retire(self, record: _Version) -> None:
        # Fence every resident target, including failed joins with unknown ACKs.
        await asyncio.gather(*(self._retire_engine(record, key) for key in tuple(record.targets)))
        with self._lock:
            # A local prepare waiter may still be delivering a late READY. Its
            # task must settle before this attempt can be retried under a new ID.
            preparing = record.prepare_task is not None and not record.prepare_task.done()
            if record.targets <= record.absent and not preparing:
                record.state = VersionState.RETIRED if record.binding is not None else VersionState.ABORTED
                self._live_operations.discard(record.operation_id)
                if self._active == record.operation_id:
                    self._active = None

    async def collect(self, *, report: bool = True) -> dict:
        self._check_loop()
        with self._lock:
            if self._suspended:
                return self.status() if report else {}
            for key in self._closing_sessions:
                self._start_close(self._sessions[key])
            if self._topology_busy():
                return self.status() if report else {}
            for operation_id in self._live_operations:
                record = self._operations[operation_id]
                if record.state == VersionState.PUBLISHED and record.binding != self._default and record.sessions == 0:
                    record.state = VersionState.RETIRING
                if record.state == VersionState.RETIRING:
                    self._start_retire(record)
            return self.status() if report else {}

    def _topology_busy(self) -> bool:
        return self._topology_task is not None and not self._topology_task.done()

    async def join_engines(self, engines: Sequence[AdapterEngine]) -> None:
        """Prepare retained versions before atomically admitting new routes.

        The caller retains process handles even if its waiter exits. A failed
        join fences all attempted instances and owns cleanup until ABSENT.
        """
        self._check_loop()
        identities = tuple(engine.identity for engine in engines)
        key = ("join", identities)
        with self._lock:
            if self._topology_key == key:
                task = self._topology_task
            else:
                if self._topology_busy() or self._suspended:
                    raise AdapterPublicationError("MEMBERSHIP_BUSY")
                if not engines or len({item.engine_id for item in identities}) != len(engines):
                    raise ValueError("unique joining engines required")
                if any(item.engine_id in self._identities for item in identities):
                    raise AdapterPublicationError("ENGINE_ID_REUSED")
                if any(not item.engine_id or not item.boot_id or not item.endpoint for item in identities):
                    raise ValueError("joining engine identity is incomplete")
                self._reserve_metadata(len(engines))
                for engine in engines:
                    identity = engine.identity
                    self._engines[identity.engine_id] = engine
                    self._identities[identity.engine_id] = identity
                    self._engine_sessions[identity.engine_id] = 0
                self._topology_key = key
                self._topology_phase = "PREPARING"
                task = self._topology_task = self._task(self._join(identities))
        await asyncio.shield(task)

    async def _join(self, identities: tuple[EngineIdentity, ...]) -> None:
        keys = {identity.engine_id for identity in identities}
        try:
            await self.settle_control()
            with self._lock:
                records = [self._operations[key] for key in self._live_operations]
                if self._default is None or any(record.state != VersionState.PUBLISHED for record in records):
                    raise AdapterPublicationError("PUBLICATION_NOT_SETTLED")
            deadline = self._loop.time() + self._prepare_timeout
            for record in records:
                for key in sorted(keys):
                    # Own even a lost/late prepare before sending its first RPC.
                    with self._lock:
                        record.targets.add(key)
                        record.instances[key] = native_instance_id(
                            self._identities[key], self.cohort_id, record.operation_id
                        )
                    receipt = await self._prepare_engine(record, self._engines[key], deadline=deadline)
                    with self._lock:
                        self._validate_receipt(record, key, receipt, absent=False)
                        record.ready[key] = receipt
            with self._lock:
                if self._unavailable & keys:
                    raise AdapterPublicationError("ENGINE_UNAVAILABLE")
                self._serving.update(keys)
                self._topology_epoch += 1
        except Exception:
            await self._detach(keys)
            raise

    async def remove_engines(self, identities: Sequence[EngineIdentity]) -> None:
        """Stop new bindings; keep old readers and owned cleanup until
        drain."""
        self._check_loop()
        identities = tuple(identities)
        key = ("remove", identities)
        with self._lock:
            if self._topology_key == key:
                task = self._topology_task
            else:
                if self._topology_busy() or self._suspended:
                    raise AdapterPublicationError("MEMBERSHIP_BUSY")
                keys = {item.engine_id for item in identities}
                if not keys or any(self._identities.get(item.engine_id) != item for item in identities):
                    raise AdapterPublicationError("ENGINE_EPOCH_MISMATCH")
                if len(self._serving - keys) < 2:
                    raise AdapterPublicationError("MINIMUM_ENGINE_CAPACITY")
                self._serving.difference_update(keys)
                self._topology_epoch += 1
                self._topology_key = key
                task = self._topology_task = self._task(self._detach(keys))
        await asyncio.shield(task)

    async def _detach(self, keys: set[str]) -> None:
        await self.settle_control()
        with self._lock:
            self._topology_phase = "DRAINING"
        while True:
            with self._lock:
                busy = self._suspended or any(self._engine_sessions[key] for key in keys)
                if not busy:
                    self._topology_phase = "CLEANING"
                    break
            # Session close remains active while topology/publication is fenced.
            await self.collect(report=False)
            await asyncio.sleep(0.05)
        while True:
            with self._lock:
                pending = [
                    (self._operations[operation], key)
                    for operation in self._live_operations
                    for key in keys & self._operations[operation].targets - self._operations[operation].absent
                ]
            if not pending:
                with self._lock:
                    self._detached.update(keys)
                    self._unavailable.difference_update(keys)
                    for key in keys:
                        self._engines.pop(key, None)  # Keep evidence/identity, not a live Ray actor handle.
                return
            await asyncio.gather(*(self._retire_engine(record, key) for record, key in pending))
            await asyncio.sleep(0.05)

    @property
    def admission_topology(self) -> tuple[int, int]:
        with self._lock:
            return self._topology_epoch, len(self._serving)

    @property
    def engine_identities(self) -> dict[str, EngineIdentity]:
        with self._lock:
            return {key: identity for key, identity in self._identities.items() if key not in self._detached}

    @property
    def suspended(self) -> bool:
        with self._lock:
            return self._suspended

    def suspend(self) -> None:
        """Fence new binds/publications before an explicit GPU handoff.

        Existing cleanup remains owned. This is not retirement: references,
        identities and capacity reservations survive until resume.
        """
        with self._lock:
            if self._topology_busy() and self._topology_phase != "DRAINING":
                raise AdapterPublicationError("MEMBERSHIP_BUSY")
            self._suspended = True

    async def settle_control(self) -> None:
        self._check_loop()
        while True:
            with self._lock:
                tasks = [
                    task
                    for operation in self._live_operations
                    for task in (self._operations[operation].prepare_task, self._operations[operation].retire_task)
                    if task is not None and not task.done()
                ]
                tasks.extend(
                    self._sessions[key].close_task
                    for key in self._closing_sessions
                    if self._sessions[key].close_task is not None and not self._sessions[key].close_task.done()
                )
            if not tasks:
                return
            await asyncio.gather(*(asyncio.shield(task) for task in tasks))

    def resume(self) -> None:
        # Caller has obtained matching RESIDENT evidence from all target boots.
        with self._lock:
            self._suspended = False

    def mark_engine_unavailable(self, identity: EngineIdentity) -> None:
        with self._lock:
            if self._identities.get(identity.engine_id) != identity:
                raise AdapterPublicationError("ENGINE_EPOCH_MISMATCH")
            self._unavailable.add(identity.engine_id)

    def mark_engine_available(self, identity: EngineIdentity) -> None:
        """Only a fresh capability + own health probe for this boot may heal
        it."""
        with self._lock:
            if self._identities.get(identity.engine_id) != identity:
                raise AdapterPublicationError("ENGINE_EPOCH_MISMATCH")
            self._unavailable.discard(identity.engine_id)

    def _operation_status(self, record: _Version) -> dict:
        return {
            "operation_id": record.operation_id,
            "version_id": record.snapshot.version_id,
            "digest": record.snapshot.digest,
            "state": record.state.value,
            "cleanup_pending": record.state == VersionState.RETIRING,
            "session_refs": record.sessions,
            "binding": asdict(record.binding) if record.binding else None,
            "publish_targets": list(record.publish_targets),
            "resident_targets": sorted(record.targets),
            "native_instances": dict(record.instances),
            "ready": {key: asdict(receipt) for key, receipt in record.ready.items()},
            "absent": sorted(record.absent),
            "absence_receipts": {key: asdict(receipt) for key, receipt in record.absence_receipts.items()},
            "errors": dict(record.errors),
            "publication_seconds": record.publication_seconds,
        }

    def _session_status(self, session: _Session) -> dict:
        return {
            "owner_epoch": session.owner_epoch,
            "session_id": session.session_id,
            "binding": asdict(session.binding) if session.binding else None,
            "engine": asdict(session.engine) if session.engine else None,
            "native_lora_id": session.native_lora_id,
            "state": "CLOSED" if session.closed else ("CLOSING" if session.closing else "BOUND"),
            "accepted": session.closing,
            "errors": dict(session.errors),
        }

    def session_status(self, owner_epoch: str, session_id: str) -> dict:
        with self._lock:
            return self._session_status(self._sessions[(owner_epoch, session_id)])

    def status(self, operation_id: str | None = None) -> dict:
        with self._lock:
            if operation_id is not None:
                return self._operation_status(self._operations[operation_id])
            return {
                "cohort_id": self.cohort_id,
                "default": asdict(self._default) if self._default else None,
                "default_epoch": self._epoch,
                "capacity": self._capacity,
                "occupied": self._occupancy(),
                "metadata_used": self._usage(),
                "metadata_limit": self._limit,
                "accepting": self._accepting,
                "unavailable": sorted(self._unavailable),
                "suspended": self._suspended,
                "serving_engines": sorted(self._serving),
                "topology_epoch": self._topology_epoch,
                "detached_engines": sorted(self._detached),
                "membership_pending": self._topology_busy(),
                "membership_phase": self._topology_phase if self._topology_busy() else "IDLE",
                "versions": {key: self._operation_status(record) for key, record in self._versions.items()},
            }
