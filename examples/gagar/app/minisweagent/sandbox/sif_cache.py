# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""On-demand apptainer SIF builder with an LRU-bounded local cache.

Adapted from examples/multi-harness-dev/task/sif_cache.py at
882c7c34f7d8b57e3af6cc29fec427b25a38583d, including the shared-lease fix
6e179c4c554a2524a0324c6ab64ebef3f0240d93. Instance lifecycle stays in
agent_server.
"""

from __future__ import annotations

import fcntl
import logging
import os
import subprocess
import tempfile
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any


LOG = logging


def normalize_image_ref(image_ref: Any) -> str:
    """Collapse a full image reference to a bare ``namespace/repo:tag``.

    Strips a leading ``docker://`` scheme and an explicit ``docker.io/`` registry
    prefix so the SIF filename and the pull URI stay consistent across datasets:

      * SWE-rebench-V2 : ``image_name = docker.io/swerebenchv2/<repo>:<tag>``
      * SWE-bench_*    : ``image      = swebench/<name>:latest``
    """
    s = str(image_ref).strip()
    if s.startswith("docker://"):
        s = s[len("docker://") :]
    if s.startswith("docker.io/"):
        s = s[len("docker.io/") :]
    return s


def image_key(image_ref: Any) -> str:
    """Filesystem-safe SIF stem for an image reference (``<stem>.sif``)."""
    return normalize_image_ref(image_ref).replace("/", "__").replace(":", "__")


def _format_called_process_error(exc: subprocess.CalledProcessError) -> str:
    out = exc.stdout if exc.stdout is not None else getattr(exc, "output", None)
    if isinstance(out, bytes):
        out = out.decode("utf-8", errors="replace")
    out = str(out or "").strip()
    if len(out) > 4000:
        out = out[:4000] + f"\n... truncated {len(out) - 4000} chars"
    return f"returncode={exc.returncode}\n{out}"


class SifCache:
    """On-demand apptainer SIF builder with an LRU-bounded local cache.

    Concurrency contract:
      * ``acquire`` builds the SIF if missing (deduplicating concurrent builds of
        the SAME image via a per-image file lock), then pins it (refcount++) and
        returns its path. The caller MUST call ``release`` exactly once.
      * Eviction never removes a SIF that is pinned (refcount > 0) or currently
        building, so a live consumer can never lose its SIF underneath it.

    ``registry``:
      * non-empty  -> pull through it, e.g. a Docker Hub mirror
        (``docker://<registry>/<ref>``);
      * empty ("") -> pull directly from the image's own registry
        (``docker://<ref>``), e.g. Docker Hub via an http(s)_proxy.
    """

    def __init__(
        self,
        *,
        sif_dir: Path | str,
        build_tmp_root: Path | str,
        executable: str = "apptainer",
        registry: str = "",
        build_retries: int = 3,
        max_bytes: int = 0,
        disable_cache: bool = True,
        max_entries: int = 0,
        build_env: dict[str, str] | None = None,
        max_builds: int = 2,
        build_timeout: float = 1800,
        run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    ) -> None:
        self.sif_dir = Path(sif_dir)
        self.sif_dir.mkdir(parents=True, exist_ok=True)
        self.executable = executable
        self.registry = str(registry).rstrip("/")
        self._build_tmp_root = Path(build_tmp_root)
        self.build_retries = max(1, int(build_retries))
        self.max_bytes = int(max_bytes)  # 0 = unbounded
        self.max_entries = int(max_entries)  # 0 = unbounded
        self.disable_cache = bool(disable_cache)
        self.build_env = dict(build_env or {})
        self.max_builds = max(1, int(max_builds))
        self.build_timeout = build_timeout
        self._run = run
        self._lock = threading.Lock()
        self._building: dict[str, threading.Event] = {}
        self._build_errors: dict[str, str] = {}
        self._last_used: dict[str, float] = {}
        self._refcount: dict[str, int] = {}
        self._stats = {"hits": 0, "builds": 0, "build_failures": 0, "waits": 0, "evictions": 0}
        self._lease_files: dict[str, list[Any]] = {}
        # Seed LRU bookkeeping from any SIFs already on disk so they are reused
        # and eligible for eviction.
        for path in self.sif_dir.glob("*.sif"):
            self._last_used.setdefault(path.stem, 0.0)

    def _active_instance_sifs(self) -> set[Path]:
        """Return SIF paths mounted by live Apptainer instances on this
        node."""
        result = self._run(
            [self.executable, "instance", "list"],
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=10,
        )
        active: set[Path] = set()
        for line in (result.stdout or "").splitlines()[1:]:
            parts = line.split()
            if len(parts) < 3:
                continue
            image = parts[-1]
            if image.endswith(".sif"):
                active.add(Path(image).resolve())
        return active

    def _sif_path(self, key: str) -> Path:
        return self.sif_dir / f"{key}.sif"

    def _lock_path(self, key: str) -> Path:
        return self.sif_dir / f".{key}.lock"

    def _build_lock_path(self, key: str) -> Path:
        return self.sif_dir / f".{key}.build.lock"

    @staticmethod
    def _check_cancel(cancel_event: threading.Event | None) -> None:
        if cancel_event is not None and cancel_event.is_set():
            raise RuntimeError("SIF acquisition cancelled")

    def _flock(self, handle: Any, operation: int, cancel_event: threading.Event | None) -> None:
        while True:
            self._check_cancel(cancel_event)
            try:
                fcntl.flock(handle.fileno(), operation | fcntl.LOCK_NB)
                return
            except BlockingIOError:
                if cancel_event is not None:
                    cancel_event.wait(0.1)
                else:
                    time.sleep(0.1)

    @contextmanager
    def _build_slot(self, cancel_event: threading.Event | None) -> Iterator[None]:
        # Shared across servers using the same node-local cache directory.
        handles = []
        try:
            for slot in range(self.max_builds):
                handles.append((self.sif_dir / f".build-slot-{slot}.lock").open("a+b"))
            while True:
                self._check_cancel(cancel_event)
                for handle in handles:
                    try:
                        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        continue
                    yield
                    return
                if cancel_event is not None:
                    cancel_event.wait(0.1)
                else:
                    time.sleep(0.1)
        finally:
            for handle in handles:
                handle.close()

    def _build_and_lease(self, key: str, ref: str, lock_file: Any, cancel_event: threading.Event | None) -> bool:
        """Build one image, then pin it, all under a short-lived builder lock.

        The shared lease is taken before the builder lock is dropped, so a
        fresh SIF cannot be evicted in the window before it is pinned. The
        builder lock is never held for longer than the build itself, so it
        cannot serialize consumers of an already finished SIF. Returns True
        when this process performed the build.
        """
        build_lock = self._build_lock_path(key).open("a+b")
        try:
            # One builder per image across all managed-agent processes. Different
            # image keys use different locks and therefore build concurrently.
            self._flock(build_lock, fcntl.LOCK_EX, cancel_event)
            built = False
            if not self._sif_path(key).exists():
                with self._build_slot(cancel_event):
                    self._build(key, ref)
                built = True
            self._flock(lock_file, fcntl.LOCK_SH, cancel_event)
            return built
        finally:
            fcntl.flock(build_lock.fileno(), fcntl.LOCK_UN)
            build_lock.close()

    def acquire(self, image_ref: str, *, cancel_event: threading.Event | None = None) -> Path:
        """Build once across processes, then hold a shared lease until release.

        Sibling sessions of one Group share a task and therefore share a SIF,
        and every one of them must reach its first inference request before the
        Group clears the Prepare barrier. An exclusive lease would serialize
        them: the first holder would keep the lock for its whole episode while
        waiting at that barrier for siblings who are in turn waiting for the
        lock. Consumers therefore only ever take LOCK_SH, which they can all
        hold at once.
        """
        key = image_key(image_ref)
        ref = normalize_image_ref(image_ref)
        path = self._sif_path(key)
        attempts = self.build_retries + 1
        for _attempt in range(attempts):
            self._check_cancel(cancel_event)
            lock_file = self._lock_path(key).open("a+b")
            try:
                if path.exists():
                    self._stats["hits"] += 1
                    # Shared lease. Eviction requires a non-blocking exclusive
                    # lock, so a live Apptainer consumer can never lose its SIF.
                    self._flock(lock_file, fcntl.LOCK_SH, cancel_event)
                elif self._build_and_lease(key, ref, lock_file, cancel_event):
                    self._stats["builds"] += 1
                else:
                    self._stats["hits"] += 1
                self._check_cancel(cancel_event)
                if not path.exists():
                    # Evicted between the build and this lease. Rebuild and retry
                    # rather than handing out a path that no longer resolves.
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                    lock_file.close()
                    self._stats["waits"] += 1
                    continue
                os.utime(path, None)
                with self._lock:
                    self._lease_files.setdefault(key, []).append(lock_file)
                    self._refcount[key] = self._refcount.get(key, 0) + 1
                    self._last_used[key] = time.monotonic()
                return path
            except Exception:
                lock_file.close()
                raise
        raise RuntimeError(
            f"SIF {path.name} kept vanishing between build and lease after {attempts} attempt(s); "
            "the cache is most likely thrashing under disk pressure."
        )

    def release(self, image_ref: str) -> None:
        key = image_key(image_ref)
        with self._lock:
            leases = self._lease_files.get(key) or []
            if not leases:
                return
            lock_file = leases.pop()
            if not leases:
                self._lease_files.pop(key, None)
            count = self._refcount.get(key, 0)
            if count <= 1:
                self._refcount.pop(key, None)
            else:
                self._refcount[key] = count - 1
            self._last_used[key] = time.monotonic()
        try:
            path = self._sif_path(key)
            if path.exists():
                os.utime(path, None)
        finally:
            lock_file.close()
        self._evict_to_fit(exclude_key=key)

    def _build(self, key: str, ref: str) -> None:
        self._evict_to_fit(exclude_key=key)
        path = self._sif_path(key)
        if path.exists():
            return
        self._build_tmp_root.mkdir(parents=True, exist_ok=True)
        # Final tmp file lives in sif_dir so os.replace onto the target SIF is an
        # atomic rename on the same filesystem; the larger build scratch goes to
        # the node-local build_tmp_root via APPTAINER_TMPDIR.
        tmp_path = self.sif_dir / f"{key}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp"
        uri = f"docker://{self.registry}/{ref}" if self.registry else f"docker://{ref}"
        build_env = {
            **os.environ,
            **self.build_env,
        }
        if self.disable_cache:
            build_env["APPTAINER_DISABLE_CACHE"] = "true"
        last_error = ""
        for attempt in range(self.build_retries):
            try:
                with tempfile.TemporaryDirectory(prefix="sif-", dir=self._build_tmp_root) as scratch:
                    self._run(
                        [self.executable, "build", str(tmp_path), uri],
                        env={**build_env, "TMPDIR": scratch, "APPTAINER_TMPDIR": scratch},
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        check=True,
                        timeout=self.build_timeout,
                    )
                os.replace(tmp_path, path)
                with self._lock:
                    self._last_used[key] = time.monotonic()
                LOG.info("SIF cache built %s from %s", path.name, uri)
                return
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
                last_error = (
                    _format_called_process_error(exc) if isinstance(exc, subprocess.CalledProcessError) else str(exc)
                )
                Path(tmp_path).unlink(missing_ok=True)
                if attempt == self.build_retries - 1:
                    raise RuntimeError(
                        f"apptainer build failed after {self.build_retries} attempt(s) for {uri}.\n{last_error}"
                    ) from exc
                time.sleep(min(5.0, 1.0 + attempt))
            finally:
                tmp_path.unlink(missing_ok=True)

    def _evict_to_fit(self, *, exclude_key: str | None = None) -> None:
        if self.max_bytes <= 0 and self.max_entries <= 0:
            return
        global_lock = (self.sif_dir / ".eviction.lock").open("a+b")
        try:
            try:
                fcntl.flock(global_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return  # Another process is already trimming this cache.
            try:
                active_sifs = self._active_instance_sifs()
            except (OSError, subprocess.SubprocessError) as exc:
                LOG.warning("SIF cache skipped eviction: cannot list live instances: %s", exc)
                return
            entries: list[tuple[float, int, str, Path]] = []
            total = 0
            for path in self.sif_dir.glob("*.sif"):
                try:
                    stat = path.stat()
                except OSError:
                    continue
                total += stat.st_size
                entries.append((stat.st_mtime, stat.st_size, path.stem, path))
            over_budget = lambda: (
                (self.max_bytes > 0 and total > self.max_bytes)
                or (self.max_entries > 0 and len(entries) > self.max_entries)
            )
            entries.sort(key=lambda entry: entry[0])
            for entry in list(entries):
                if not over_budget():
                    break
                _last_used, size, key, path = entry
                if key == exclude_key or path.resolve() in active_sifs:
                    continue
                candidate_lock = self._lock_path(key).open("a+b")
                try:
                    fcntl.flock(candidate_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    candidate_lock.close()
                    continue
                try:
                    path.unlink(missing_ok=True)
                    total -= size
                    entries.remove(entry)
                    self._stats["evictions"] += 1
                    LOG.info("SIF cache evicted %s (%d bytes) under disk pressure", path.name, size)
                finally:
                    candidate_lock.close()
        finally:
            global_lock.close()

    def stats(self) -> dict[str, Any]:
        with self._lock:
            counters = dict(self._stats)
            pinned = sum(1 for count in self._refcount.values() if count > 0)
            building = len(self._building)
        total = 0
        count = 0
        for path in self.sif_dir.glob("*.sif"):
            try:
                total += path.stat().st_size
                count += 1
            except OSError:
                continue
        return {
            **counters,
            "cached_count": count,
            "total_bytes": total,
            "total_gb": round(total / 1024**3, 2),
            "max_gb": round(self.max_bytes / 1024**3, 2) if self.max_bytes else 0,
            "pinned_count": pinned,
            "building_count": building,
        }
