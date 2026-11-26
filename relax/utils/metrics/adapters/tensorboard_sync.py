# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Mirror a single, live TensorBoard event file onto its final filesystem."""

import fcntl
import os
import shutil
import struct
import tempfile
import threading
import time
from pathlib import Path
from typing import BinaryIO

from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)

_RECORD_SIZE = struct.Struct("<Q")
# A TFRecord has an 8-byte length, two 4-byte checksums, and its payload.
_RECORD_OVERHEAD = 16
_COPY_CHUNK_BYTES = 1024 * 1024
_FINAL_SYNC_TIMEOUT_SECONDS = 15
_EXIT_SYNC_RETRY_SECONDS = 30
_WRITER_LOCK = ".writer.lock"
_EXIT_SYNC_NAMESPACE = "relax-tensorboard-sync"


class _EventFileCopy:
    def __init__(self, local_dir: Path, destination: Path) -> None:
        self._local_dir = local_dir
        self._destination = destination
        self._source: Path | None = None
        self._complete_offset = 0
        self.synced_offset = 0

    def sync_once(self) -> bool:
        if self._source is None:
            files = list(self._local_dir.glob("events.out.tfevents.*"))
            if not files:
                return False
            if len(files) != 1:
                raise OSError(f"Expected a single TensorBoard event file in {self._local_dir}")
            self._source = files[0]

        with self._source.open("rb") as source:
            size = os.fstat(source.fileno()).st_size
            offset = self._complete_offset
            if size < offset:
                raise OSError("The local TensorBoard event file shrank")
            while offset + _RECORD_OVERHEAD <= size:
                source.seek(offset)
                (record_size,) = _RECORD_SIZE.unpack(source.read(_RECORD_SIZE.size))
                end = offset + _RECORD_OVERHEAD + record_size
                if end > size:
                    break
                offset = end
            self._complete_offset = offset

            if offset > self.synced_offset:
                self._destination.mkdir(parents=True, exist_ok=True)
                target_path = self._destination / self._source.name
                with target_path.open("ab", buffering=_COPY_CHUNK_BYTES) as target:
                    # A failed write may have appended some bytes. Read the
                    # actual target length to resume without duplicating them.
                    remote_offset = os.fstat(target.fileno()).st_size
                    if not self.synced_offset <= remote_offset <= offset:
                        raise OSError("The destination TensorBoard event length is inconsistent")
                    source.seek(remote_offset)
                    remaining = offset - remote_offset
                    while remaining:
                        chunk = source.read(min(remaining, _COPY_CHUNK_BYTES))
                        if not chunk:
                            raise OSError("The local TensorBoard event file was truncated during sync")
                        target.write(chunk)
                        remaining -= len(chunk)
                    target.flush()
                    os.fsync(target.fileno())
                self.synced_offset = offset

        return self.synced_offset == size


class _ExitSync:
    """A node-local, detached Ray actor that outlives the event writer."""

    def __init__(self, local_dir: Path, destination: Path) -> None:
        self._local_dir = local_dir
        self._destination = destination
        self._lease = (local_dir / _WRITER_LOCK).open("rb")

    def ready(self) -> None:
        """Confirm that the actor started and can see the writer's spool."""

    def wait_and_sync(self) -> None:
        import ray

        try:
            with self._lease as lease:
                # A Ray Job stop can kill the writer without calling finish().
                # The OS releases its lock when the process exits.
                fcntl.flock(lease, fcntl.LOCK_EX)
                if not self._local_dir.exists():
                    return
                self._sync_after_exit()
        finally:
            ray.actor.exit_actor()

    def _sync_after_exit(self) -> None:
        event_copy = _EventFileCopy(self._local_dir, self._destination)
        deadline = time.monotonic() + _EXIT_SYNC_RETRY_SECONDS
        while True:
            try:
                complete = event_copy.sync_once()
                break
            except OSError:
                if time.monotonic() >= deadline:
                    logger.warning(
                        "TensorBoard exit sync failed. Spool retained in %s.", self._local_dir, exc_info=True
                    )
                    return
                time.sleep(1)
        logger.info("TensorBoard exit sync wrote %s bytes to %s.", event_copy.synced_offset, self._destination)
        if complete:
            shutil.rmtree(self._local_dir, ignore_errors=True)
        else:
            logger.warning("TensorBoard spool contains an incomplete event. Retained in %s.", self._local_dir)


class TensorBoardEventSync:
    def __init__(self, destination: Path, interval_seconds: int) -> None:
        self.local_dir = Path(tempfile.mkdtemp(prefix="relax-tensorboard-"))
        self._destination = destination.absolute()
        self._interval_seconds = interval_seconds
        self._event_copy = _EventFileCopy(self.local_dir, self._destination)
        self._lease: BinaryIO | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="tensorboard-event-sync", daemon=True)
        self._finished = False

    def start(self) -> None:
        self._start_exit_sync()
        self._thread.start()

    def _start_exit_sync(self) -> None:
        try:
            import ray
        except ImportError:
            return
        if not ray.is_initialized():
            return

        from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

        lease = (self.local_dir / _WRITER_LOCK).open("xb")
        actor = None
        try:
            fcntl.flock(lease, fcntl.LOCK_EX)
            actor = (
                ray.remote(num_cpus=0, max_restarts=0)(_ExitSync)
                .options(
                    name=self.local_dir.name,
                    namespace=_EXIT_SYNC_NAMESPACE,
                    lifetime="detached",
                    scheduling_strategy=NodeAffinitySchedulingStrategy(
                        ray.get_runtime_context().get_node_id(), soft=False
                    ),
                )
                .remote(self.local_dir, self._destination)
            )
            ray.get(actor.ready.remote(), timeout=_FINAL_SYNC_TIMEOUT_SECONDS)
            actor.wait_and_sync.remote()
        except Exception:
            lease.close()
            if actor is not None:
                ray.kill(actor, no_restart=True)
            raise
        self._lease = lease

    def finish(self) -> None:
        self._stop.set()
        self._thread.join(timeout=_FINAL_SYNC_TIMEOUT_SECONDS)
        completed = self._finished
        if completed:
            shutil.rmtree(self.local_dir, ignore_errors=True)
        if self._lease is not None and not self._thread.is_alive():
            # Let the recovery actor take over only after the periodic writer
            # finished, or after this process dies and its lock is released.
            self._lease.close()
            self._lease = None
        if not completed:
            raise OSError(
                f"TensorBoard final sync to {self._destination} is incomplete. Local spool: {self.local_dir}."
            )

    def _run(self) -> None:
        while True:
            finishing = self._stop.wait(self._interval_seconds)
            try:
                complete = self._event_copy.sync_once()
            except OSError:
                logger.warning("TensorBoard sync to %s failed.", self._destination, exc_info=True)
            else:
                if finishing and complete:
                    self._finished = True
                    logger.info(
                        "TensorBoard final sync completed, %s bytes written to %s.",
                        self._event_copy.synced_offset,
                        self._destination,
                    )
            if finishing:
                return
