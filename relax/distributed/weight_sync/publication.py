# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""One immutable candidate, CPU-only publication, explicit source
confirmation."""

import os
import shutil
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from contextvars import copy_context
from dataclasses import asdict, dataclass
from pathlib import Path

from relax.utils.logging_utils import get_logger

from .parallel_io import ordered_map
from .scratch import ScratchWorkspace
from .store import VersionStore, atomic_write, canonical, digest, load_json, sync_directory, valid_id, writer_lock
from .traffic import capture, current, file_write


logger = get_logger(__name__)


@dataclass(frozen=True)
class PublicationHandle:
    update_id: str
    candidate_id: str
    version: int
    source: dict | None = None


@dataclass(frozen=True)
class PublishedVersion:
    stream: str
    version: int
    root: str
    artifact_id: str
    update_id: str


class VersionPublisher:
    """The owning process serializes submit/finalize at training boundaries."""

    def __init__(
        self,
        store: VersionStore,
        stream: str,
        scratch: str | Path,
        *,
        model_config: dict,
        converter_revision: str,
        anchor_interval: int = 20,
        compression: bool = False,
        encode_workers: int = 4,
        full_only: bool = False,
    ) -> None:
        self.store, self.stream = store, valid_id(stream)
        self.scratch = Path(scratch)
        self.scratch.mkdir(parents=True, exist_ok=True)
        self.options = load_json(
            canonical(
                dict(
                    model_config=model_config,
                    converter_revision=converter_revision,
                    anchor_interval=anchor_interval,
                    compression=compression,
                    encode_workers=encode_workers,
                    full_only=full_only,
                )
            )
        )
        self._future: Future | None = None
        self._pending: PublicationHandle | None = None
        self._directory: Path | None = None
        self._table: dict | None = None
        self._attempt = 0
        self._traffic_metrics = None
        self._journal = self.scratch / f"publisher-{self.stream}.json"
        self._lease = writer_lock(store.root / ".publishers" / self.stream)
        self._lease.__enter__()
        self._closed = False
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="weight-publication")
        self.workspace = None
        try:
            # Protect the durable candidate before reclaiming pre-journal copies.
            protected = []
            if self._journal.exists():
                state = load_json(self._journal.read_bytes())
                protected = [self.scratch / state["directory"]]
            self.workspace = ScratchWorkspace(self.scratch, f"publication:{self.stream}", protected=protected)
            self._restore_pending()
        except BaseException:
            self._pool.shutdown(wait=True)
            self._lease.__exit__(None, None, None)
            if self.workspace:
                self.workspace.close()
            raise

    @property
    def pending(self) -> PublicationHandle | None:
        return self._pending

    def _restore_pending(self) -> None:
        if not self._journal.exists():
            return
        state = load_json(self._journal.read_bytes())
        self._attempt = state.get("attempt", 1)
        if state["options"] != self.options:
            raise ValueError("Pending publication uses another configuration; reopen with its recorded configuration")
        handle = PublicationHandle(**state["handle"])
        directory = (self.scratch / state["directory"]).resolve()
        if directory.parent != self.scratch.resolve() or not directory.name.startswith("candidate-"):
            raise ValueError("Invalid pending candidate directory")
        record_path = self.store.root / "publications" / self.stream / f"{handle.version}.json"
        record = load_json(record_path.read_bytes()) if record_path.exists() else None
        if record and record["phase"] == "SOURCE_CONFIRMED" and record["handle"] == asdict(handle):
            self._journal.unlink()
            sync_directory(self.scratch)
            if directory.exists():
                self._remove_candidate(directory)
            return
        if self.store.manifest_path(self.stream, handle.version).exists():
            manifest = self.store.manifest(self.stream, handle.version)
            if manifest["update_id"] != handle.update_id:
                raise ValueError("Pending candidate conflicts with the committed update")
            record = {
                "phase": "STORE_COMMITTED",
                "attempt": self._attempt,
                "handle": asdict(handle),
                "published": {k: manifest[k] for k in PublishedVersion.__dataclass_fields__},
            }
            atomic_write(record_path, canonical(record))
            self._pending, self._directory = handle, directory
            self._future = Future()
            self._future.set_result(record)
            return
        table = {}
        for name, (dtype, shape, filename) in state["tensors"].items():
            path = directory / filename
            if path.resolve().parent != directory or not path.is_file():
                raise ValueError(
                    "Pending candidate file is missing; restore the candidate before resuming publication"
                )
            table[name] = (dtype, shape, path)
        self._pending, self._directory, self._table = handle, directory, table
        self._advance_attempt()
        self._schedule_publish()

    def _schedule_publish(self) -> None:
        from .metrics import SyncMetrics

        if self._traffic_metrics is None:
            self._traffic_metrics = current() or SyncMetrics(self._pending.update_id)
        with capture(self._traffic_metrics, attempt=self._attempt):
            self._future = self._pool.submit(copy_context().run, self._publish)

    def _advance_attempt(self) -> None:
        self._attempt += 1
        state = load_json(self._journal.read_bytes())
        state["attempt"] = self._attempt
        atomic_write(self._journal, canonical(state))

    def next_version(self) -> int:
        versions = [int(p.stem) for p in (self.store.root / "versions" / self.stream).glob("*.json")]
        return max(versions, default=-1) + 1

    def submit(
        self, update_id: str, tensors: dict, *, version: int | None = None, source: dict | None = None
    ) -> PublicationHandle:
        if self._closed:
            raise RuntimeError("Publisher is closed")
        if self._pending is not None:
            raise RuntimeError("Publication pending; finalize or retry it before capturing another update")
        version = self.next_version() if version is None else version
        handle = PublicationHandle(str(update_id), uuid.uuid4().hex, version, load_json(canonical(source)))
        self._attempt = 1
        directory = self.workspace.temporary()
        table, total = {}, 0
        try:
            for dtype, shape, path in tensors.values():
                total += Path(path).stat().st_size
                if total > self.store.max_weight_bytes:
                    raise ValueError("Candidate exceeds max_weight_bytes")

            def persist_candidate(item):
                name, (dtype, shape, path) = item
                target = directory / (digest(name.encode("utf-8")) + ".bin")
                shutil.copyfile(path, target)
                file_write(target, target.stat().st_size)
                with target.open("r+b") as durable:
                    os.fsync(durable.fileno())
                return name, (dtype, shape, target)

            table.update(ordered_map(persist_candidate, tensors.items(), self.options["encode_workers"]))
            sync_directory(directory)
            atomic_write(
                self._journal,
                canonical(
                    {
                        "handle": asdict(handle),
                        "attempt": self._attempt,
                        "directory": directory.name,
                        "options": self.options,
                        "tensors": {
                            name: [dtype, list(shape), path.name] for name, (dtype, shape, path) in table.items()
                        },
                    }
                ),
            )
            self._pending, self._directory, self._table = handle, directory, table
            self._traffic_metrics = current()
            self._schedule_publish()
            return handle
        except BaseException as error:
            if self._pending == handle:
                # A durable journal already owns these files. Preserve the
                # same candidate if scheduling its worker fails locally.
                self._future = Future()
                self._future.set_exception(error)
            else:
                self._remove_candidate(directory)
                self._pending = self._directory = self._table = None
            raise

    def _remove_candidate(self, directory):
        if directory.name.startswith(self.workspace.prefix):
            self.workspace.remove(directory)
        else:
            # Earlier journal formats used an unscoped name. Only that exact
            # journal-owned directory is eligible; unrelated legacy scratch stays.
            try:
                shutil.rmtree(directory)
            except OSError:
                logger.warning("Legacy candidate cleanup pending: %s", directory, exc_info=True)

    def ready(self, handle: PublicationHandle) -> bool:
        if handle != self._pending:
            raise ValueError("Unknown publication handle")
        return self._future.done()

    def _publish(self) -> dict:
        try:
            with self._traffic_metrics.stage("store_publication", path=self.store.root):
                return self._publish_store()
        except BaseException:
            self._traffic_metrics.unmeasured("W", "Failed storage operation may have completed a partial write")
            raise

    def _publish_store(self) -> dict:
        handle = self._pending
        manifest = self.store.publish(
            self.stream, handle.version, self._table, update_id=handle.update_id, source=handle.source, **self.options
        )
        record = {
            "phase": "STORE_COMMITTED",
            "attempt": self._attempt,
            "handle": asdict(handle),
            "published": {k: manifest[k] for k in PublishedVersion.__dataclass_fields__},
        }
        atomic_write(self.store.root / "publications" / self.stream / f"{handle.version}.json", canonical(record))
        return record

    def wait_committed(self, handle: PublicationHandle, timeout: float | None = None) -> PublishedVersion:
        if handle != self._pending:
            raise ValueError("Unknown publication handle")
        record = self._future.result(timeout=timeout)
        published = PublishedVersion(**record["published"])
        actual = self.store.manifest(published.stream, published.version)
        if actual["root"] != published.root or actual["artifact_id"] != published.artifact_id:
            raise ValueError("Committed publication changed")
        return published

    def confirm(self, handle: PublicationHandle, published: PublishedVersion) -> PublishedVersion:
        if self.wait_committed(handle) != published:
            raise ValueError("Source confirmation refers to another publication")
        path = self.store.root / "publications" / self.stream / f"{published.version}.json"
        record = load_json(path.read_bytes())
        record["phase"] = "SOURCE_CONFIRMED"
        atomic_write(path, canonical(record))
        self._journal.unlink(missing_ok=True)
        sync_directory(self.scratch)
        if self._directory.exists():
            self._remove_candidate(self._directory)
        if self._traffic_metrics is not None:
            try:
                self._traffic_metrics.save(
                    self.scratch / f"publisher-metrics-v{published.version}-{self._traffic_metrics.operation_id}.json"
                )
            except OSError:
                logger.warning(
                    "Published version %s; communication report could not be saved", published.version, exc_info=True
                )
        self._future = self._pending = self._directory = self._table = None
        return published

    def finalize_pending(self, handle: PublicationHandle, timeout: float | None = None) -> PublishedVersion:
        return self.confirm(handle, self.wait_committed(handle, timeout))

    def retry(self, handle: PublicationHandle) -> None:
        if handle != self._pending or not self._future.done():
            raise ValueError("Only a completed failed publication can be retried")
        if self._future.exception() is None:
            raise ValueError("Publication already committed; finalize it")
        with capture(self._traffic_metrics, attempt=self._attempt + 1):
            self._advance_attempt()
            self._schedule_publish()

    def close(self, *, finalize: bool = True) -> None:
        """Stop the worker and release ownership.

        Coordinated producers use finalize=False on failed setup/shutdown so
        the durable candidate remains available for collective confirmation.
        """
        if self._closed:
            return
        try:
            if finalize and self._pending is not None:
                self.finalize_pending(self._pending)
        finally:
            self._pool.shutdown(wait=True)
            self._closed = True
            self.workspace.close()
            self._lease.__exit__(None, None, None)
