# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Durable installation coordinator for a fixed, locally fenced engine
group."""

import json
import time
from pathlib import Path
from typing import Protocol

from relax.utils.logging_utils import get_logger

from .checkpoint import validate_checkpoint
from .deadline import checkpoint as check_deadline
from .deadline import time_budget
from .diagnostics import save_diagnostic
from .metrics import SyncMetrics
from .store import atomic_write, canonical, writer_lock


logger = get_logger(__name__)


class EngineGroup(Protocol):
    """All generation entrypoints must obey close/open; methods must be
    bounded.

    verify checks actual engine bytes against the checkpoint, including every
    worker. An HTTP success or an engine's recorded version is insufficient.
    """

    def identity(self) -> str: ...
    def preflight(self, checkpoint: Path) -> None: ...
    def close(self) -> None: ...
    def pause(self) -> None: ...
    def load(self, checkpoint: Path, version: str) -> None: ...
    def verify(self, checkpoint: Path) -> None: ...
    def flush(self) -> None: ...
    def resume(self) -> None: ...
    def open(self) -> None: ...


class InstallSession:
    def __init__(self, directory: str | Path, engines: EngineGroup, *, node_install_lock: Path | None = None) -> None:
        self.directory = Path(directory)
        self.engines = engines
        self.node_install_lock = node_install_lock or self.directory / ".node-install"
        self.directory.mkdir(parents=True, exist_ok=True)
        # Restart never inherits permission to serve from an old ACTIVE marker.
        self.engines.close()
        self._active_target = None

    def status(self) -> dict:
        path = self.directory / "state.json"
        return json.loads(path.read_bytes()) if path.exists() else {"phase": "NEW"}

    def active_receipt(self, installation_id: str) -> dict | None:
        """Reuse an installation only while the same live workers serve it."""
        state = self.status()
        if (
            self._active_target is not None
            and state.get("phase") == "ACTIVE"
            and state.get("target") == self._active_target
            and state.get("serving") == self._active_target
            and state.get("installation_id") == installation_id
            and state.get("engines") == self.engines.identity()
        ):
            return state
        return None

    def suspend(self) -> None:
        """Drain generation before releasing engine memory; retain its
        checkpoint."""
        with writer_lock(self.directory / ".coordinator"), writer_lock(self.node_install_lock, timeout=120):
            self._active_target = None
            state = self.status()
            state.update(phase="SUSPENDING", serving=None)
            atomic_write(self.directory / "state.json", canonical(state))
            try:
                self.engines.close()
            except Exception as error:
                state.update(
                    phase="UNAVAILABLE",
                    error=f"{type(error).__name__}: {error}",
                    action="Drain or restart this engine, then recover its retained checkpoint before offload.",
                )
                atomic_write(self.directory / "state.json", canonical(state))
                raise
            state["phase"] = "SUSPENDED"
            atomic_write(self.directory / "state.json", canonical(state))

    def install(
        self,
        checkpoint: str | Path,
        manifest: dict,
        installation_id: str,
        *,
        checkpoint_record: dict | None = None,
        _recovery_for: dict | None = None,
    ) -> dict:
        checkpoint = Path(checkpoint).resolve()
        with writer_lock(self.directory / ".coordinator"), writer_lock(self.node_install_lock, timeout=120):
            validate_checkpoint(checkpoint, manifest)
            old = self.status()
            if _recovery_for is not None and old.get("target") != _recovery_for:
                raise ValueError("Requested target changed before recovery acquired the coordinator")
            identity = self.engines.identity()
            preflight = getattr(self.engines, "preflight", None)
            if preflight is not None:
                preflight(checkpoint)
            if identity != self.engines.identity():
                raise ValueError("Engine membership changed during checkpoint preparation")
            target = {"stream": manifest["stream"], "version": manifest["version"], "root": manifest["root"]}
            if (
                not _recovery_for
                and self._active_target == target
                and old.get("phase") == "ACTIVE"
                and old.get("installation_id") == installation_id
                and old.get("engines") == identity
            ):
                return old
            if old.get("committed") and old.get("phase") != "ACTIVE" and old.get("target") != target:
                raise ValueError("Committed installation must recover before changing target")
            if old.get("installation_id") == installation_id and old.get("target") != target:
                raise ValueError("Installation ID reused for a different target")
            if not _recovery_for and old.get("target", {}).get("stream") == target["stream"]:
                if target["version"] < old["target"]["version"]:
                    raise ValueError("Stale installation")
                if target["version"] == old["target"]["version"] and target["root"] != old["target"]["root"]:
                    raise ValueError("Version conflict")
            state = {
                "phase": "PREPARED",
                "target": target,
                "installation_id": installation_id,
                "engines": identity,
                "committed": bool(old.get("committed") and old.get("target") == target),
                "serving": old.get("serving"),
                "artifact_id": manifest["artifact_id"],
                "epoch": old.get("epoch", -1) + 1,
                "active_checkpoint": old.get("active_checkpoint"),
                "committed_checkpoint": old.get("committed_checkpoint") if old.get("target") == target else None,
            }
            metrics = SyncMetrics(manifest.get("update_id", str(manifest["version"])))

            def save(phase: str) -> None:
                state["phase"] = phase
                atomic_write(self.directory / "state.json", canonical(state))

            pause_start = time.monotonic()
            try:
                claim = getattr(self.engines, "claim_install", None)
                if claim is not None:
                    state["epoch"] = claim(installation_id, target["root"], state["epoch"])
                self.engines.close()
                check_deadline("admission close")
                save("PREPARED")
                with metrics.stage("pause"):
                    self.engines.pause()
                    begin = getattr(self.engines, "begin_install", None)
                    if begin is not None:
                        state["epoch"] = begin(installation_id, target["root"], state["epoch"])
                    if identity != self.engines.identity():
                        raise ValueError("Engine membership changed before checkpoint loading")
                save("PAUSED")
                check_deadline("engine pause")
                save("LOADING")
                with metrics.stage("load"):
                    self.engines.load(checkpoint, f"{target['stream']}:{target['version']}:{target['root']}")
                check_deadline("checkpoint load")
                state["workers"] = getattr(self.engines, "load_receipts", None)
                with metrics.stage("verify"):
                    self.engines.verify(checkpoint)
                check_deadline("physical verification")
                if identity != self.engines.identity():
                    raise ValueError("Engine membership changed during installation")
                save("VERIFIED")
                with metrics.stage("resume_and_probe"):
                    self.engines.flush()
                    self.engines.resume()
                check_deadline("generation probe")
                if identity != self.engines.identity():
                    raise ValueError("Engine membership changed before activation")
                state["committed"] = True
                state["serving"] = target
                record = checkpoint_record or {"checkpoint": str(checkpoint), "manifest": manifest, "owner": None}
                state["committed_checkpoint"] = record
                state["active_checkpoint"] = record
                save("COMMITTED")
                activate = getattr(self.engines, "commit_install", None)
                if activate is not None:
                    activate()
                # Persist before opening: a crash can only leave the gate closed.
                final_phase = "ACTIVE"
                if _recovery_for:
                    state["target"] = _recovery_for
                    state["committed"] = False
                    state["action"] = (
                        "Serving the recovered version; retry the requested target with a FULL checkpoint."
                    )
                    final_phase = "WAITING_ACTION"
                save(final_phase)
                self.engines.open()
                metrics.seconds["inference_pause"] = time.monotonic() - pause_start
                state["metrics"] = metrics.report()
                # The durable activation record already precedes open(). Timing
                # details are retained in the independent diagnostic report below.
                self._active_target = target
                return dict(state)
            except BaseException as error:
                try:
                    with time_budget(5, inherit=False):
                        self.engines.close()
                except Exception as gate_error:
                    state["gate_error"] = f"{type(gate_error).__name__}: {gate_error}"
                self._active_target = None
                state["error_type"] = type(error).__name__
                state["error"] = str(error)
                state["serving"] = None
                state["action"] = (
                    "Prepare a same-target FULL checkpoint and retry install with the same installation_id."
                )
                state["metrics"] = metrics.report()
                save("UNAVAILABLE")
                raise
            finally:
                record = {
                    **metrics.report(),
                    "stream": target["stream"],
                    "version": target["version"],
                    "installation_id": installation_id,
                    "phase": state.get("phase", "PREPARING"),
                    "workers": state.get("workers"),
                }
                try:
                    save_diagnostic(self.directory / "metrics" / f"{metrics.operation_id}.json", canonical(record))
                except OSError:
                    logger.warning("Installation diagnostic output failed", exc_info=True)

    def recover(self, checkpoint: str | Path, manifest: dict, installation_id: str) -> dict:
        """Restore a known checkpoint while retaining the failed requested
        target."""
        state = self.status()
        requested = state.get("target")
        if state.get("committed"):
            if requested != {key: manifest[key] for key in ("stream", "version", "root")}:
                raise ValueError("Committed installation must recover its committed checkpoint")
            return self.install(
                checkpoint, manifest, state["installation_id"], checkpoint_record=state.get("committed_checkpoint")
            )
        if not requested or requested["stream"] != manifest["stream"] or manifest["version"] > requested["version"]:
            raise ValueError("Recovery checkpoint must belong to the requested stream and an earlier version")
        return self.install(
            checkpoint,
            manifest,
            installation_id,
            checkpoint_record=state.get("active_checkpoint"),
            _recovery_for=requested,
        )
