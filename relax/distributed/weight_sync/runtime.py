# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Explicit consumer operations: prepare, install, acknowledge and recover."""

from contextlib import ExitStack, contextmanager

from relax.utils.logging_utils import get_logger

from .budget import NodeBudget
from .consumer import PreparedVersion, VersionConsumer
from .deadline import checkpoint, time_budget, validate_timeout
from .diagnostics import save_diagnostic
from .metrics import SyncMetrics
from .session import InstallSession
from .store import atomic_write, canonical, digest, load_json, valid_id, writer_lock


logger = get_logger(__name__)


class ConsumptionError(RuntimeError):
    def __init__(self, state: dict) -> None:
        self.state = state
        super().__init__(f"{state['stage']}: {state['reason']}; {state['allowed_actions']}")

    def __reduce__(self):
        return type(self), (self.state,)


class ConsumptionRunner:
    def __init__(
        self,
        consumer: VersionConsumer,
        session: InstallSession,
        name: str,
        stream: str,
        *,
        start_version: int = 0,
        mode: str = "all_versions",
        operation_timeout: float = 1800,
        node_budget: NodeBudget | None = None,
        operation_resources: dict[str, int] | None = None,
    ) -> None:
        self.consumer, self.session = consumer, session
        self.node_budget, self.operation_resources = node_budget, operation_resources or {}
        self.name, self.stream = valid_id(name), valid_id(stream)
        self.directory = session.directory
        self.transport = consumer.transport
        self.operation_timeout = validate_timeout(operation_timeout)
        self._operation_state = None
        self.subscription = self._control("register", name, stream, start_version, mode=mode)
        self.mode = mode
        with writer_lock(self.directory / ".consumer"):
            intents = self.directory / "preparation-intents.json"
            owners = load_json(intents.read_bytes()) if intents.exists() else []
            retained_path = self.directory / "retained-checkpoints.json"
            retained = set(load_json(retained_path.read_bytes())) if retained_path.exists() else set()
            active = self._active()
            if active is not None:
                retained.add(active.owner)
            self.consumer.reconcile(owners, retained)
            atomic_write(intents, canonical(sorted(retained)))

    def _record(self, **state) -> dict:
        state.update(consumer=self.name, stream=self.stream)
        self._operation_state = state
        save_diagnostic(self.directory / "operation.json", canonical(state))
        return state

    def _control(self, operation, *args, **kwargs):
        """Keep control exchanges separate from checkpoint preparation
        bytes."""
        ledger = getattr(self.transport, "ledger", None)
        before = ledger.report() if ledger else None
        try:
            return getattr(self.transport, operation)(*args, **kwargs)
        finally:
            metrics = SyncMetrics(f"{self.stream}:consumer-control")
            if ledger is not None:
                after = ledger.report()
                count = sum(
                    after["bytes"][direction]["C"] - before["bytes"][direction]["C"]
                    for direction in ("sent", "received")
                )
                metrics.transferred("C", count, link=operation, scope="socket_bytes")
            else:
                metrics.unmeasured("C", "SharedFS control requires filesystem observations")
            report = {**metrics.report(), "stream": self.stream, "consumer": self.name, "operation": operation}
            try:
                save_diagnostic(self.directory / "control-metrics" / f"{metrics.operation_id}.json", canonical(report))
            except OSError:
                logger.warning("Consumer control diagnostic output failed", exc_info=True)

    def status(self) -> dict:
        if self._operation_state is not None:
            return dict(self._operation_state)
        # A new controller reports durable installation state, not a possibly
        # interrupted diagnostic write from the previous process.
        installation = self.session.status()
        return {
            "status": "READY",
            "consumer": self.name,
            "stream": self.stream,
            "installation": installation,
            "action": "consume_target_to_restore_or_confirm",
        }

    def available(self, limit: int = 100) -> list[dict]:
        cursor = self._control("consumer_status", self.name)["cursor"]
        return self._control(
            "versions",
            self.stream,
            after=cursor,
            limit=1 if self.mode == "latest" else limit,
            newest_first=self.mode == "latest",
        )

    def _active(self) -> PreparedVersion | None:
        state = self.session.status()
        record = state.get("committed_checkpoint") if state.get("committed") else state.get("active_checkpoint")
        path = self.directory / "active-checkpoint.json"
        if record is None and not path.exists():
            return None
        if record is None:
            record = load_json(path.read_bytes())
        checkpoint = (self.consumer.root / record["checkpoint"]).resolve()
        if checkpoint.parent != self.consumer.root.resolve():
            raise ValueError("Active checkpoint is outside the consumer directory")
        return PreparedVersion(checkpoint, record["manifest"], record["owner"], {})

    def _remember(self, prepared: PreparedVersion) -> dict:
        path = self.directory / "retained-checkpoints.json"
        retained = load_json(path.read_bytes()) if path.exists() else {}
        record = {"checkpoint": prepared.checkpoint.name, "manifest": prepared.manifest, "owner": prepared.owner}
        retained[prepared.owner] = record
        atomic_write(path, canonical(retained))
        return record

    def _release_previous(self, active: PreparedVersion) -> None:
        path = self.directory / "retained-checkpoints.json"
        retained = load_json(path.read_bytes())
        for owner, record in list(retained.items()):
            if owner != active.owner:
                previous = PreparedVersion(self.consumer.root / record["checkpoint"], record["manifest"], owner, {})
                self.consumer.release(previous, remove_checkpoint=True)
                del retained[owner]
        atomic_write(path, canonical(retained))
        intents = self.directory / "preparation-intents.json"
        owners = load_json(intents.read_bytes()) if intents.exists() else []
        self.consumer.reconcile(owners, set(retained))
        atomic_write(intents, canonical(sorted(retained)))

    def _cleanup(self, active: PreparedVersion) -> dict:
        try:
            self._release_previous(active)
        except Exception as error:
            logger.warning(
                "Version is active; retained checkpoint cleanup will retry on the next consumption: %s", error
            )
            return {
                "status": "PENDING",
                "reason": f"{type(error).__name__}: {error}",
                "action": "retry_cleanup_on_next_consumption",
            }
        return {"status": "COMPLETE"}

    @contextmanager
    def _reservation(self):
        if self.node_budget is None:
            reservation = writer_lock(self.consumer.root / ".node-operation", timeout=120)
        else:
            reservation = self.node_budget.operation(
                self.name,
                self.operation_resources,
                quiescent=lambda: (
                    self.session.status().get("phase")
                    not in ("PREPARED", "PAUSED", "LOADING", "VERIFIED", "COMMITTED", "UNAVAILABLE")
                ),
            )
        with ExitStack() as stack:
            try:
                stack.enter_context(reservation)
            except (OSError, TimeoutError) as error:
                state = self._record(
                    stage="RESERVING",
                    status="WAITING_ACTION",
                    active_version=self.session.status().get("serving"),
                    reason=f"{type(error).__name__}: {error}",
                    allowed_actions=[
                        "recover_retained_installation",
                        "increase_deployment_capacity",
                        "retry_same_target",
                    ],
                )
                raise ConsumptionError(state) from error
            yield

    def apply(self, version: int, *, full: bool = False) -> dict:
        """Apply one fixed target; an explicit FULL retry uses the same
        target."""
        with writer_lock(self.directory / ".consumer"), time_budget(self.operation_timeout), self._reservation():
            subscription = self._control("consumer_status", self.name)
            if type(version) is not int or version < max(subscription["cursor"], subscription["start_version"]):
                raise ValueError("Requested version precedes this subscription's active version or start")
            if self.mode == "all_versions" and version not in (subscription["cursor"], subscription["cursor"] + 1):
                raise ValueError("all_versions consumer must install the next contiguous version")
            owner = "consume-" + digest(canonical([self.name, self.stream, version, full]))[:48]
            operation_id = f"{self.name}:{self.stream}:{version}"
            stage = "PREPARING"
            self._record(
                operation_id=operation_id,
                stage=stage,
                status="RUNNING",
                target=version,
                active_version=self.session.status().get("serving"),
            )
            try:
                active = self._active()
                receipt = None if full else self.session.active_receipt(operation_id)
                if (
                    receipt is not None
                    and active is not None
                    and active.manifest["root"] == receipt["serving"]["root"]
                    and active.manifest["version"] == version
                    and active.manifest["stream"] == self.stream
                ):
                    # No new content is installed. Restart, offload and worker
                    # replacement invalidate the live receipt in InstallSession.
                    prepared = active
                    preparation = {"reused_active": True}
                else:
                    intents = self.directory / "preparation-intents.json"
                    owners = load_json(intents.read_bytes()) if intents.exists() else []
                    if owner not in owners:
                        atomic_write(intents, canonical([*owners, owner]))
                    prepared = self.consumer.prepare(self.stream, version, owner, full=full)
                    record = self._remember(prepared)
                    checkpoint("checkpoint preparation")
                    stage = "INSTALLING"
                    receipt = self.session.install(
                        prepared.checkpoint, prepared.manifest, operation_id, checkpoint_record=record
                    )
                    # Keep the actual serving checkpoint even if the source ACK is lost.
                    atomic_write(self.directory / "active-checkpoint.json", canonical(record))
                    preparation = prepared.metrics
                stage = "ACKNOWLEDGING"
                checkpoint("consumption acknowledgement")
                self.subscription = self._control("acknowledge", self.name, receipt)
                cleanup = self._cleanup(prepared)
                return self._record(
                    operation_id=operation_id,
                    stage="ACTIVE",
                    status="SUCCEEDED",
                    target=version,
                    active_version=receipt["serving"],
                    receipt=receipt,
                    preparation=preparation,
                    cursor=self.subscription["cursor"],
                    cleanup=cleanup,
                )
            except Exception as error:
                current = self.session.status()
                if stage == "ACKNOWLEDGING":
                    actions = ["retry_same_target_to_confirm_consumption"]
                elif stage == "PREPARING":
                    actions = ["check_source_and_capacity", "retry_same_target", "prepare_same_target_full"]
                else:
                    actions = [
                        "prepare_same_target_full",
                        "recover_committed_checkpoint"
                        if current.get("committed")
                        else "recover_previous_active_checkpoint",
                        "restart_unresponsive_engine_then_recover",
                    ]
                state = self._record(
                    operation_id=operation_id,
                    stage=stage,
                    status="WAITING_ACTION",
                    target=version,
                    active_version=current.get("serving"),
                    reason=f"{type(error).__name__}: {error}",
                    allowed_actions=actions,
                )
                raise ConsumptionError(state) from error

    def recover_active(self) -> dict:
        with writer_lock(self.directory / ".consumer"), time_budget(self.operation_timeout), self._reservation():
            active = self._active()
            if active is None:
                raise ValueError("No previously active checkpoint is retained")
            receipt = self.session.recover(active.checkpoint, active.manifest, "recover-" + active.owner)
            if receipt["phase"] == "ACTIVE":
                self.subscription = self._control("acknowledge", self.name, receipt)
                cleanup = self._cleanup(active)
                return self._record(
                    operation_id=receipt["installation_id"],
                    stage="ACTIVE",
                    status="SUCCEEDED",
                    target=receipt["target"]["version"],
                    active_version=receipt["serving"],
                    receipt=receipt,
                    cursor=self.subscription["cursor"],
                    cleanup=cleanup,
                )
            return self._record(
                operation_id=receipt["installation_id"],
                stage="RECOVERED",
                status="WAITING_ACTION",
                target=receipt["target"]["version"],
                active_version=receipt["serving"],
                allowed_actions=["prepare_same_target_full"],
                receipt=receipt,
            )

    def apply_through(self, version: int, *, full: bool = False) -> dict:
        """Install a catch-up range with one operation budget per target."""
        with time_budget(self.operation_timeout):
            subscription = self._control("consumer_status", self.name)
        start = max(subscription["start_version"], subscription["cursor"])
        pending = self.session.status().get("target")
        if pending and pending["stream"] == self.stream and pending["version"] > subscription["cursor"]:
            # A previous attempt already entered the next version. Resume
            # that target, including after recovery served the old one.
            # Restart/offload of the acknowledged version still starts at
            # the cursor and reinstalls it before advancing.
            start = max(start, subscription["cursor"] + 1)
        if type(version) is not int or version < start:
            raise ValueError("Catch-up target precedes the active or pending version, or subscription start")
        targets = [version] if self.mode == "latest" else range(start, version + 1)
        result = None
        for target in targets:
            result = self.apply(target, full=full)
        return result
