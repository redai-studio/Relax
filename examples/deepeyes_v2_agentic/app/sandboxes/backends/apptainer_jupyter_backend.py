# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Apptainer (Singularity) + in-container ipykernel sandbox backend.

Targets the deepeyes_v2 recipe on hosts that have ``apptainer`` (>=1.2) but
no docker daemon — common on shared HPC / Slurm clusters. Apptainer is
daemon-less and rootless, reads OCI / docker images via the
``docker://<ref>`` URI (transparently pulled and cached by apptainer
itself) or a prebuilt ``.sif`` file.

Architecture:
  * one container per session, running ``python -m ipykernel_launcher``
  * jupyter_client.AsyncKernelClient over ZMQ on **AF_UNIX (IPC)** sockets,
    not TCP. Each session writes its 5 ZMQ channels to socket files inside
    its own session-scoped tempdir; the tempdir is identity-bind-mounted
    into the container so the same path resolves to the same inode on
    both sides. Eliminates the 5-port allocation race that surfaced as
    ``zmq.error.ZMQError: Address already in use`` at >32 concurrent
    create_sessions on TCP transport.
  * file IO goes through host-side bind mounts (the recipe's image and
    output dirs are mapped from a session-scoped host tempdir into the
    container at well-known paths) — much cheaper than ``exec_run``-style
    round-trips and the only reason ``list_files`` / ``read_bytes`` /
    ``delete_files`` work at all without a long-lived control channel
    inside the container.

Image requirements: must include ``ipykernel`` (no defensive auto-install).
The user-supplied apptainer image is treated as authoritative.
"""

from __future__ import annotations

import asyncio
import atexit
import glob
import os
import shutil
import signal
import tempfile
import threading
import uuid
from datetime import timedelta
from pathlib import Path
from typing import Any, Mapping, Optional

from relax.utils.logging_utils import get_logger

from ..base import (
    BaseSandbox,
    BaseSandboxSession,
    ExecutionError,
    ExecutionResult,
    FileEntry,
    SandboxCapability,
)
from ..exceptions import SandboxCreateFailed, SandboxError
from . import register_backend


logger = get_logger(__name__)


def _translate(exc: BaseException, *, context: str) -> SandboxError:
    """Wrap an apptainer-/jupyter_client-/ZMQ-side exception in
    :class:`SandboxError` so recipes can catch a single type across backends.

    Apptainer failures are almost always permanent (kernel died, image
    missing, channel torn down), so we never tag as transient by default.
    The ``context`` string is appended to the message to make the post-mortem
    obvious in the rollout log.
    """
    return SandboxError(f"{context}: {type(exc).__name__}: {exc}")


_DEFAULT_CONNECTION_DIR = "/tmp/relax_sandbox"  # kept for backward-compat; unused under IPC transport
_DELETE_CHUNK = 256
# Pid-prefixed so the atexit sweep (below) only reaps this process's own
# leftovers — sibling Ray workers on the same host stay safe.
_SESSION_TMP_PREFIX = f"relax-apptainer-{os.getpid()}-"
# Linux AF_UNIX caps sockaddr_un.sun_path at 108 bytes incl. trailing \0.
# jupyter_client builds each of the 5 channels' socket files as
# f"{ip_prefix}-{random_int}" where random_int ∈ [1, 999999] (≤6 digits →
# ≤7 chars incl. dash). We reserve 8 bytes so the IPC prefix has headroom
# for the null terminator without risk of silent truncation.
_SUN_PATH_LIMIT = 108
_IPC_SUFFIX_BUDGET = 8
# Historical TCP-only failure mode: `jupyter_client.write_connection_file`
# allocated 5 ZMQ ports via a bind→close→write→ipykernel-rebind dance, racy
# under concurrency. Switched to AF_UNIX (IPC) transport so this marker is
# no longer expected to fire. The bounded detection and retry path remains for
# any EADDRINUSE reported during kernel startup.
_ZMQ_EADDRINUSE_MARKERS = ("Address already in use", "EADDRINUSE")
# jupyter_client.AsyncKernelClient.wait_for_ready raises this exact RuntimeError
# when its first kernel_info reply doesn't return within the heartbeat window.
# Observed at scale: cold-start of apptainer + ipykernel import can occasionally
# cross that window even though the kernel boots fine (banner reaches stdout,
# ZMQ sockets bind). Treated as transient — same recovery as a port collision:
# tear down + regenerate ports + respawn.
_KERNEL_HANDSHAKE_RACE_MARKER = "Kernel died before replying to kernel_info"
# AF_UNIX socket file-existence poll: cold-start of apptainer + ipykernel takes
# ~1-3s before ipykernel binds its 5 ZMQ channels. Without this pre-poll,
# AsyncKernelClient.start_channels races the bind and wait_for_ready times out
# in ~1s with the misleading "Kernel died" RuntimeError (kernel is alive, sockets
# just haven't been created yet).
_IPC_SOCKET_POLL_INTERVAL_S = 0.05
_IPC_SOCKET_POLL_TIMEOUT_S = 15.0


class _ZmqPortCollision(Exception):
    """Internal sentinel: kernel boot failed because a ZMQ socket couldn't
    bind."""

    def __init__(self, stderr_tail: str) -> None:
        super().__init__(stderr_tail)
        self.stderr_tail = stderr_tail


class _KernelHandshakeRace(Exception):
    """Internal sentinel: jupyter_client gave up waiting for the first
    kernel_info reply within its heartbeat window, even though the kernel
    process is alive and printed its banner.

    Distinct from a permanent boot failure (missing ipykernel, broken image,
    ...) which never reaches this branch.
    """

    def __init__(self, stdout_tail: str, stderr_tail: str) -> None:
        super().__init__(f"stdout_tail={stdout_tail!r} stderr_tail={stderr_tail!r}")
        self.stdout_tail = stdout_tail
        self.stderr_tail = stderr_tail


def _is_kernel_handshake_race(exc: BaseException) -> bool:
    return isinstance(exc, RuntimeError) and _KERNEL_HANDSHAKE_RACE_MARKER in str(exc)


class ApptainerJupyterBackend(BaseSandbox):
    """Provision sandboxes as apptainer ``exec`` subprocesses running
    ipykernel."""

    name = "apptainer_jupyter"
    capabilities = frozenset({SandboxCapability.STATEFUL_KERNEL, SandboxCapability.FILE_IO})

    def __init__(
        self,
        *,
        image: str,
        apptainer_bin: str = "apptainer",
        command_prefix: Optional[list[str]] = None,
        connection_dir: str = _DEFAULT_CONNECTION_DIR,
        bind_paths: Optional[Mapping[str, str]] = None,
        extra_binds: Optional[list[str]] = None,
        env: Optional[dict] = None,
        nv: bool = False,
        writable_tmpfs: bool = True,
        cleanenv: bool = True,
        pid_isolation: bool = True,
        home_dir: Optional[str] = "/root",
        create_max_retries: int = 5,
        kernel_ready_timeout_s: float = 120.0,
        kernel_python: str = "python",
        config_dir: Optional[str] = None,
    ) -> None:
        if not image:
            raise ValueError("ApptainerJupyterBackend requires a non-empty image (URI or .sif path)")
        self._image = self._resolve_image_path(image, config_dir)
        self._apptainer_bin = apptainer_bin
        self._command_prefix = list(command_prefix or [])
        self._connection_dir = connection_dir
        # bind_paths: { container_path: host_subdir_name }
        # The backend creates <session_tmp>/<host_subdir_name> and binds it
        # at <container_path>. File IO methods reverse-map a queried
        # container path to its host counterpart so callers can keep using
        # the in-container paths they already know about.
        self._bind_paths: dict[str, str] = dict(bind_paths or {})
        self._extra_binds = list(extra_binds or [])
        self._env = dict(env or {})
        self._nv = nv
        self._writable_tmpfs = writable_tmpfs
        self._cleanenv = cleanenv
        self._pid_isolation = pid_isolation
        self._home_dir = home_dir
        self._create_max_retries = create_max_retries
        self._kernel_ready_timeout_s = kernel_ready_timeout_s
        self._kernel_python = kernel_python

    @staticmethod
    def _resolve_image_path(image: str, config_dir: Optional[str]) -> str:
        # URI-form refs (docker://, library://, oras://, ...) and absolute
        # paths bypass resolution. Without a config_dir hint (e.g. direct
        # programmatic instantiation), keep the caller's CWD-relative
        # behavior so we don't silently change semantics.
        if "://" in image or os.path.isabs(image) or not config_dir:
            return image
        return os.path.normpath(os.path.join(config_dir, image))

    async def create_session(
        self,
        *,
        metadata: Optional[dict] = None,
        request_timeout: timedelta,
    ) -> "ApptainerJupyterSession":
        # Deferred so the package imports without jupyter_client installed.
        from jupyter_client import AsyncKernelClient  # type: ignore[import-not-found]
        from jupyter_client.connect import write_connection_file  # type: ignore[import-not-found]

        last_transient: Optional[Exception] = None
        for attempt in range(1, self._create_max_retries + 1):
            try:
                return await self._provision_session_once(
                    AsyncKernelClient=AsyncKernelClient,
                    write_connection_file=write_connection_file,
                    request_timeout=request_timeout,
                )
            except _ZmqPortCollision as exc:
                last_transient = exc
                logger.warning(
                    "apptainer_jupyter: ZMQ EADDRINUSE on attempt %d/%d; "
                    "regenerating connection file and retrying. stderr_tail=%r",
                    attempt,
                    self._create_max_retries,
                    exc.stderr_tail,
                )
                # Tiny back-off so the racing peers don't immediately re-collide
                # on the freshly-freed ports.
                await asyncio.sleep(0.05 * attempt)
            except _KernelHandshakeRace as exc:
                last_transient = exc
                logger.warning(
                    "apptainer_jupyter: kernel handshake race on attempt %d/%d; "
                    "kernel booted (banner in stdout) but heartbeat reply missed deadline; "
                    "tearing down + retrying. stdout_tail=%r stderr_tail=%r",
                    attempt,
                    self._create_max_retries,
                    exc.stdout_tail,
                    exc.stderr_tail,
                )
                # Longer back-off than port collision: this is a load/cold-cache
                # timing race, not an immediately-recoverable port conflict.
                # Giving the host a moment to drain in-flight kernel imports
                # before re-spawning sharply reduces back-to-back failures.
                await asyncio.sleep(0.5 * attempt)

        # Translate to SandboxCreateFailed (subclass of SandboxError) per the
        # exceptions.py translation contract so the recipe's catch-all can
        # isinstance-check and demote the log level — these are known-class
        # transient failures, not unhandled exceptions worth a full traceback.
        raise SandboxCreateFailed(
            f"apptainer_jupyter: transient kernel-launch failures persisted across "
            f"{self._create_max_retries} retries; last={type(last_transient).__name__}: "
            f"{last_transient}"
        ) from last_transient

    async def _provision_session_once(
        self,
        *,
        AsyncKernelClient: Any,
        write_connection_file: Any,
        request_timeout: timedelta,
    ) -> "ApptainerJupyterSession":
        """Single attempt: write_connection_file → spawn → wait_for_ready.

        On ZMQ EADDRINUSE the call site retries; everything else propagates.
        Cleans up partial state (process, kernel client, tmpdir) before raising
        on any failure so retries start from a clean slate.
        """
        session_tmp: Optional[Path] = None
        proc: Optional[asyncio.subprocess.Process] = None
        kernel_client: Any = None
        try:
            session_tmp = Path(tempfile.mkdtemp(prefix=_SESSION_TMP_PREFIX))
            os.chmod(session_tmp, 0o755)

            host_conn_dir = session_tmp / "conn"
            host_conn_dir.mkdir()
            os.chmod(host_conn_dir, 0o755)

            connection_basename = f"kernel-{uuid.uuid4().hex}.json"
            host_connection_path = host_conn_dir / connection_basename
            # IPC + identity bind: container resolves the same path as host,
            # so socket files written by ipykernel on one side appear on the
            # other side as the same inode (required for AF_UNIX rendezvous).
            in_container_connection_path = str(host_connection_path)

            ipc_prefix = str(host_conn_dir / "k")
            if len(ipc_prefix) + _IPC_SUFFIX_BUDGET > _SUN_PATH_LIMIT:
                raise SandboxError(
                    f"apptainer_jupyter: IPC socket path prefix too long for AF_UNIX "
                    f"(SUN_PATH={_SUN_PATH_LIMIT}): {ipc_prefix!r} "
                    f"({len(ipc_prefix)} bytes + {_IPC_SUFFIX_BUDGET} suffix budget). "
                    f"Set $TMPDIR to a shorter path."
                )
            # transport="ipc" makes write_connection_file emit AF_UNIX socket
            # paths rather than TCP host:port pairs. The 5 channels become
            # f"{ipc_prefix}-{rand}" files, no port allocation at all.
            await asyncio.to_thread(
                write_connection_file,
                fname=str(host_connection_path),
                ip=ipc_prefix,
                transport="ipc",
            )

            host_bind_dirs: dict[str, Path] = {}
            for container_path, host_subdir in self._bind_paths.items():
                host_dir = session_tmp / host_subdir
                host_dir.mkdir(parents=True, exist_ok=True)
                os.chmod(host_dir, 0o777)
                host_bind_dirs[container_path] = host_dir

            argv = self._build_apptainer_argv(
                host_conn_dir=host_conn_dir,
                host_bind_dirs=host_bind_dirs,
                in_container_connection_path=in_container_connection_path,
            )

            stderr_path = host_conn_dir / "kernel.err"
            stdout_path = host_conn_dir / "kernel.out"
            stderr_fh = open(stderr_path, "wb")
            stdout_fh = open(stdout_path, "wb")
            try:
                proc = await asyncio.create_subprocess_exec(
                    *argv,
                    stdout=stdout_fh,
                    stderr=stderr_fh,
                    stdin=asyncio.subprocess.DEVNULL,
                )
            finally:
                stderr_fh.close()
                stdout_fh.close()

            # Wait for ipykernel to bind its 5 AF_UNIX channels before opening
            # the client. Otherwise the heartbeat poll inside wait_for_ready
            # gives up in ~1s with "Kernel died before replying to kernel_info"
            # even though the kernel is mid-boot (banner already in stdout).
            await _wait_for_ipc_sockets(
                ipc_prefix=ipc_prefix,
                proc=proc,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                timeout_s=_IPC_SOCKET_POLL_TIMEOUT_S,
            )

            kernel_client = AsyncKernelClient()
            kernel_client.load_connection_file(str(host_connection_path))
            await asyncio.to_thread(kernel_client.start_channels)

            ready_deadline = min(request_timeout.total_seconds(), self._kernel_ready_timeout_s)
            try:
                await asyncio.wait_for(kernel_client.wait_for_ready(), timeout=ready_deadline)
            except Exception as wait_exc:
                stderr_tail = _read_stderr_file_tail(stderr_path)
                stdout_tail = _read_stderr_file_tail(stdout_path)
                if _is_port_collision(stderr_tail) or _is_port_collision(stdout_tail):
                    # Tear down the half-built session and surface a typed
                    # exception so the caller can retry. We deliberately do
                    # NOT re-raise the underlying ZMQError so the retry path
                    # is keyed on the recovery semantics, not the SDK type.
                    await _teardown_partial(proc, kernel_client, session_tmp)
                    raise _ZmqPortCollision(stderr_tail or stdout_tail)
                if _is_kernel_handshake_race(wait_exc):
                    # Always treat the "Kernel died before replying to
                    # kernel_info" RuntimeError as a transient race and retry.
                    #
                    # Earlier this branch gated on an "ipykernel" banner being
                    # present in captured stdout, to distinguish race from
                    # permanent boot failure (broken image, missing ipykernel,
                    # ...). Production data showed that gate was unreliable:
                    # under concurrent load, ipykernel's stdout buffer often
                    # had not flushed to the host file by the time we read it,
                    # so 20+ real races per run were misclassified as
                    # "non-transient" and surfaced full tracebacks upstream.
                    #
                    # Permanent failures are still caught — they exit the
                    # apptainer subprocess (rc != None), which `wait_ready`
                    # detects explicitly. The retry budget is bounded by
                    # `create_max_retries`, so a truly broken image surfaces
                    # as `transient kernel-launch failures persisted` after
                    # `create_max_retries × kernel_ready_timeout_s`.
                    await _teardown_partial(proc, kernel_client, session_tmp)
                    raise _KernelHandshakeRace(stdout_tail, stderr_tail)
                logger.warning(
                    "apptainer_jupyter: wait_for_ready failed (non-transient); rc=%s argv=%s "
                    "stderr_tail=%r stdout_tail=%r exc=%r",
                    proc.returncode if proc is not None else "N/A",
                    argv,
                    stderr_tail,
                    stdout_tail,
                    wait_exc,
                )
                raise

            session = ApptainerJupyterSession(
                process=proc,
                kernel_client=kernel_client,
                session_tmp=session_tmp,
                host_bind_dirs=host_bind_dirs,
                kernel_ready_timeout_s=self._kernel_ready_timeout_s,
                argv=argv,
                stderr_path=stderr_path,
                stdout_path=stdout_path,
            )
            # Mark the kernel as already verified ready so wait_ready becomes a
            # no-op (the recipe still calls it, but the actual probe has just
            # been done as part of the create-session retry loop).
            session._ready = True
            return session
        except (_ZmqPortCollision, _KernelHandshakeRace):
            raise
        except BaseException:
            await _teardown_partial(proc, kernel_client, session_tmp)
            raise

    def _build_apptainer_argv(
        self,
        *,
        host_conn_dir: Path,
        host_bind_dirs: dict[str, Path],
        in_container_connection_path: str,
    ) -> list[str]:
        argv: list[str] = [self._apptainer_bin, "exec"]
        if self._writable_tmpfs:
            argv.append("--writable-tmpfs")
        if self._cleanenv:
            argv.append("--cleanenv")
        if self._pid_isolation:
            argv.append("--pid")
        if self._nv:
            argv.append("--nv")
        if self._home_dir:
            # Apptainer 1.4+ rejects --env HOME=...; --home is the supported
            # way to set $HOME and silences the otherwise-fatal stderr warn.
            argv.extend(["--home", self._home_dir])

        # Identity bind: AF_UNIX socket files written by ipykernel inside the
        # container must resolve to the same inode the host AsyncKernelClient
        # is connecting to. Mounting host_conn_dir at a different container
        # path (the pre-IPC layout) breaks rendezvous; mount at the same path.
        argv.extend(["-B", f"{host_conn_dir}:{host_conn_dir}"])
        for container_path, host_dir in host_bind_dirs.items():
            argv.extend(["-B", f"{host_dir}:{container_path}"])
        for extra in self._extra_binds:
            argv.extend(["-B", extra])

        for k, v in self._env.items():
            argv.extend(["--env", f"{k}={v}"])

        argv.append(self._image)

        argv.extend(self._command_prefix)
        argv.extend(
            [
                self._kernel_python,
                "-m",
                "ipykernel_launcher",
                "-f",
                in_container_connection_path,
            ]
        )
        return argv


class ApptainerJupyterSession(BaseSandboxSession):
    capabilities = frozenset({SandboxCapability.STATEFUL_KERNEL, SandboxCapability.FILE_IO})

    def __init__(
        self,
        *,
        process: asyncio.subprocess.Process,
        kernel_client: Any,
        session_tmp: Path,
        host_bind_dirs: dict[str, Path],
        kernel_ready_timeout_s: float,
        argv: list[str],
        stderr_path: Optional[Path] = None,
        stdout_path: Optional[Path] = None,
    ) -> None:
        self._process: Optional[asyncio.subprocess.Process] = process
        self._kernel_client = kernel_client
        self._session_tmp: Optional[Path] = session_tmp
        self._host_bind_dirs = host_bind_dirs
        self._kernel_ready_timeout_s = kernel_ready_timeout_s
        self._argv = argv
        self._stderr_path = stderr_path
        self._stdout_path = stdout_path
        self._owner_loop: Optional[asyncio.AbstractEventLoop] = None
        self._closed = False
        self._exec_lock = asyncio.Lock()
        # Set to True by `_provision_session_once` once start_channels +
        # wait_for_ready have already been driven inside the create-session
        # retry loop. When True, `wait_ready` is a fast no-op (other than
        # capturing the owner loop for cross-loop close).
        self._ready: bool = False

    async def wait_ready(self, timeout: timedelta) -> None:
        self._owner_loop = asyncio.get_running_loop()
        if self._process is None or self._process.returncode is not None:
            stderr_tail = await self._read_stderr_tail()
            raise SandboxError(
                f"apptainer_jupyter: kernel process exited before wait_ready "
                f"(rc={self._process.returncode if self._process else 'N/A'}); "
                f"stderr_tail={stderr_tail!r}"
            )
        if self._ready:
            return
        try:
            await asyncio.to_thread(self._kernel_client.start_channels)
            ready_deadline = min(timeout.total_seconds(), self._kernel_ready_timeout_s)
            await asyncio.wait_for(self._kernel_client.wait_for_ready(), timeout=ready_deadline)
        except SandboxError:
            raise
        except Exception as exc:  # noqa: BLE001
            stderr_tail = await self._read_stderr_tail()
            logger.warning(
                "apptainer_jupyter: wait_for_ready failed; argv=%s stderr_tail=%r",
                self._argv,
                stderr_tail,
            )
            raise _translate(exc, context="wait_for_ready") from exc
        self._ready = True

    async def run_code(
        self,
        code: str,
        *,
        timeout: float,
        language: str = "python",
    ) -> ExecutionResult:
        if language != "python":
            raise ValueError(f"ApptainerJupyterSession.run_code: language={language!r} not supported (python only)")
        async with self._exec_lock:
            try:
                return await asyncio.wait_for(self._exec_one(code), timeout=timeout)
            except asyncio.TimeoutError:
                await self.interrupt()
                return ExecutionResult(status="timeout", stderr="Execution timed out.")
            except SandboxError:
                raise
            except Exception as exc:  # noqa: BLE001
                # If the kernel died mid-call, surface that explicitly.
                proc = self._process
                rc = proc.returncode if proc is not None else "N/A"
                stderr_tail = await self._read_stderr_tail()
                logger.warning(
                    "apptainer_jupyter: run_code failed; rc=%s stderr_tail=%r",
                    rc,
                    stderr_tail,
                )
                raise _translate(exc, context=f"run_code (rc={rc})") from exc

    async def _exec_one(self, code: str) -> ExecutionResult:
        msg_id = self._kernel_client.execute(code, allow_stdin=False, stop_on_error=False)
        stdout_parts: list[str] = []
        stderr_parts: list[str] = []
        err: Optional[ExecutionError] = None
        while True:
            msg = await self._kernel_client.get_iopub_msg()
            parent_id = msg.get("parent_header", {}).get("msg_id")
            if parent_id != msg_id:
                continue
            msg_type = msg.get("msg_type")
            content = msg.get("content", {})
            if msg_type == "stream":
                text = content.get("text", "")
                if content.get("name") == "stderr":
                    stderr_parts.append(text)
                else:
                    stdout_parts.append(text)
            elif msg_type == "error":
                tb = content.get("traceback") or []
                err = ExecutionError(
                    name=str(content.get("ename", "Error")),
                    value=str(content.get("evalue", "")),
                    traceback="\n".join(tb),
                )
            elif msg_type == "status" and content.get("execution_state") == "idle":
                break
            # display_data / execute_result intentionally ignored (recipes
            # that need image capture write to bind-mounted host dirs).

        stdout = "".join(stdout_parts)
        stderr = "".join(stderr_parts)
        if err is not None:
            stderr = (stderr + f"\n{err.name}: {err.value}\n{err.traceback}").strip()

        return ExecutionResult(
            status="error" if err is not None else "success",
            stdout=stdout,
            stderr=stderr,
            error=err,
        )

    async def interrupt(self) -> None:
        proc = self._process
        if proc is None or proc.returncode is not None:
            return
        try:
            proc.send_signal(signal.SIGINT)
        except (ProcessLookupError, OSError) as exc:
            logger.warning("apptainer_jupyter: interrupt failed: %s", exc)

    def _resolve_to_host(self, container_path: str) -> Optional[Path]:
        """Map a container-side path back to its host bind-mount path.

        Returns ``None`` if the path is not under any registered bind mount;
        the caller treats that as an empty / unreadable result.
        """
        cp = container_path.rstrip("/") or "/"
        for mount, host_dir in self._host_bind_dirs.items():
            mount_norm = mount.rstrip("/") or "/"
            if cp == mount_norm:
                return host_dir
            prefix = mount_norm + "/"
            if container_path.startswith(prefix):
                rel = container_path[len(prefix) :]
                return host_dir / rel
        return None

    async def list_files(self, path: str) -> list[FileEntry]:
        host = self._resolve_to_host(path)
        if host is None or not host.exists() or not host.is_dir():
            return []

        def _scan() -> list[FileEntry]:
            out: list[FileEntry] = []
            mount_root = self._mount_root_for(path)
            host_root = self._host_bind_dirs[mount_root] if mount_root else host
            for entry in host.iterdir():
                if not entry.is_file():
                    continue
                size: Optional[int]
                try:
                    size = entry.stat().st_size
                except OSError:
                    size = None
                # Translate host path back into container path-space so
                # callers see the same /tmp/_relax_imgs/foo.png they
                # passed in (regardless of whether ``path`` is the mount
                # root or a subdirectory of it).
                rel = entry.relative_to(host_root)
                container_root = mount_root if mount_root else path.rstrip("/")
                container_path = f"{container_root.rstrip('/')}/{rel.as_posix()}"
                out.append(FileEntry(path=container_path, size=size))
            return out

        return await asyncio.to_thread(_scan)

    def _mount_root_for(self, container_path: str) -> Optional[str]:
        cp = container_path.rstrip("/") or "/"
        for mount in self._host_bind_dirs.keys():
            mount_norm = mount.rstrip("/") or "/"
            if cp == mount_norm or container_path.startswith(mount_norm + "/"):
                return mount
        return None

    async def read_bytes(self, path: str) -> bytes:
        host = self._resolve_to_host(path)
        if host is None:
            raise FileNotFoundError(f"{path} is not under any registered bind mount")
        if not host.exists():
            raise FileNotFoundError(path)
        if host.is_dir():
            raise IsADirectoryError(path)
        return await asyncio.to_thread(host.read_bytes)

    async def delete_files(self, paths: list[str]) -> None:
        if not paths:
            return

        def _rm(targets: list[str]) -> None:
            for p in targets:
                host = self._resolve_to_host(p)
                if host is None:
                    continue
                try:
                    if host.is_dir():
                        shutil.rmtree(host, ignore_errors=True)
                    else:
                        host.unlink(missing_ok=True)
                except OSError as exc:
                    logger.warning("apptainer_jupyter: delete %s failed: %s", p, exc)

        for i in range(0, len(paths), _DELETE_CHUNK):
            chunk = paths[i : i + _DELETE_CHUNK]
            await asyncio.to_thread(_rm, chunk)

    async def _read_stderr_tail(self, n: int = 2000) -> str:
        if self._stderr_path is None:
            return ""
        return await asyncio.to_thread(_read_stderr_file_tail, self._stderr_path, n)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        process = self._process
        kernel_client = self._kernel_client
        session_tmp = self._session_tmp
        loop = self._owner_loop
        self._process = None
        self._kernel_client = None
        self._session_tmp = None
        self._owner_loop = None

        async def _cleanup() -> None:
            if kernel_client is not None:
                try:
                    await asyncio.to_thread(kernel_client.stop_channels)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("apptainer_jupyter: stop_channels failed: %s", exc)
            if process is not None and process.returncode is None:
                try:
                    process.send_signal(signal.SIGTERM)
                except (ProcessLookupError, OSError) as exc:
                    logger.debug("apptainer_jupyter: SIGTERM: %s", exc)
                try:
                    await asyncio.wait_for(process.wait(), timeout=10)
                except asyncio.TimeoutError:
                    try:
                        process.kill()
                    except (ProcessLookupError, OSError) as exc:
                        logger.debug("apptainer_jupyter: kill: %s", exc)
                    try:
                        await asyncio.wait_for(process.wait(), timeout=5)
                    except asyncio.TimeoutError:
                        logger.warning("apptainer_jupyter: process did not exit after kill")
            if session_tmp is not None and session_tmp.exists():
                try:
                    await asyncio.to_thread(shutil.rmtree, str(session_tmp), True)
                except Exception as exc:  # noqa: BLE001
                    logger.debug("apptainer_jupyter: rmtree session_tmp failed: %s", exc)

        if loop is not None and not loop.is_closed():
            try:
                current = asyncio.get_running_loop()
            except RuntimeError:
                current = None
            if current is loop:
                await _cleanup()
                return
            try:
                asyncio.run_coroutine_threadsafe(_cleanup(), loop)
                return
            except RuntimeError:
                pass

        def _run() -> None:
            try:
                asyncio.run(asyncio.wait_for(_cleanup(), timeout=15))
            except Exception as exc:  # noqa: BLE001
                logger.warning("apptainer_jupyter: cleanup thread aborted: %s", exc)

        threading.Thread(target=_run, daemon=True).start()


async def _wait_for_ipc_sockets(
    *,
    ipc_prefix: str,
    proc: asyncio.subprocess.Process,
    stdout_path: Path,
    stderr_path: Path,
    timeout_s: float,
) -> None:
    """Block until ipykernel binds k-1..k-5 socket files, or proc dies, or
    timeout.

    Raises ``_KernelHandshakeRace`` on timeout / early exit so the create-
    session retry loop treats it as transient (same recovery as a real
    handshake race).
    """
    socket_paths = [f"{ipc_prefix}-{i}" for i in range(1, 6)]
    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout_s
    while True:
        if proc.returncode is not None:
            raise _KernelHandshakeRace(
                stdout_tail=_read_stderr_file_tail(stdout_path),
                stderr_tail=(
                    f"kernel exited rc={proc.returncode} before binding IPC sockets; "
                    f"stderr_tail={_read_stderr_file_tail(stderr_path)!r}"
                ),
            )
        if all(os.path.exists(p) for p in socket_paths):
            return
        if loop.time() >= deadline:
            missing = [p for p in socket_paths if not os.path.exists(p)]
            raise _KernelHandshakeRace(
                stdout_tail=_read_stderr_file_tail(stdout_path),
                stderr_tail=(
                    f"timeout after {timeout_s}s waiting for IPC sockets; missing={missing}; "
                    f"stderr_tail={_read_stderr_file_tail(stderr_path)!r}"
                ),
            )
        await asyncio.sleep(_IPC_SOCKET_POLL_INTERVAL_S)


def _read_stderr_file_tail(path: Path, n: int = 8192) -> str:
    """Read up to the last ``n`` bytes from a kernel stderr capture file.

    We redirect kernel stderr to a host file (bind-mounted into the container)
    instead of a subprocess pipe. The pipe approach lost data when the kernel
    was alive-but-hung (heartbeat death without exit) because Python's internal
    stderr buffer only flushes on exit. Reading a file works regardless of
    process state.
    """
    try:
        with open(path, "rb") as f:
            try:
                f.seek(-n, os.SEEK_END)
            except OSError:
                f.seek(0)
            data = f.read()
        return data.decode("utf-8", errors="replace")
    except (FileNotFoundError, OSError):
        return ""


def _is_port_collision(stderr_tail: str) -> bool:
    return any(marker in stderr_tail for marker in _ZMQ_EADDRINUSE_MARKERS)


async def _teardown_partial(
    proc: Optional[asyncio.subprocess.Process],
    kernel_client: Any,
    session_tmp: Optional[Path],
) -> None:
    """Tear down a half-built session (process, kernel client, tmpdir) in a
    fail-tolerant way; safe to call with any subset of the args being None."""
    if kernel_client is not None:
        try:
            await asyncio.to_thread(kernel_client.stop_channels)
        except Exception as exc:  # noqa: BLE001
            logger.debug("apptainer_jupyter: teardown stop_channels failed: %s", exc)
    if proc is not None and proc.returncode is None:
        try:
            proc.send_signal(signal.SIGTERM)
        except (ProcessLookupError, OSError) as exc:
            logger.debug("apptainer_jupyter: teardown SIGTERM: %s", exc)
        try:
            await asyncio.wait_for(proc.wait(), timeout=5)
        except asyncio.TimeoutError:
            try:
                proc.kill()
            except (ProcessLookupError, OSError) as exc:
                logger.debug("apptainer_jupyter: teardown kill: %s", exc)
            try:
                await asyncio.wait_for(proc.wait(), timeout=3)
            except asyncio.TimeoutError:
                logger.warning("apptainer_jupyter: teardown process did not exit after kill")
    if session_tmp is not None and session_tmp.exists():
        try:
            await asyncio.to_thread(shutil.rmtree, str(session_tmp), True)
        except Exception as exc:  # noqa: BLE001
            logger.debug("apptainer_jupyter: teardown rmtree failed: %s", exc)


register_backend("apptainer_jupyter", ApptainerJupyterBackend)


def _sweep_leaked_session_tmpdirs() -> None:
    """Reap this process's leaked session tmpdirs on interpreter exit.

    ``ApptainerJupyterSession.close()`` normally rmtree's each session_tmp, but
    is skipped when the process dies before ``finally`` runs (SIGKILL, OOM,
    hard crash in the owner loop). Over hundreds of thousands of sessions per
    training run the leak — bind-mounted image dirs holding base64-decoded
    dataset images — fills the container disk and triggers eviction. Pid prefix
    keeps sibling Ray workers untouched.
    """
    pattern = os.path.join(tempfile.gettempdir(), f"{_SESSION_TMP_PREFIX}*")
    for path in glob.glob(pattern):
        try:
            shutil.rmtree(path, ignore_errors=True)
        except Exception:  # noqa: BLE001
            pass


atexit.register(_sweep_leaked_session_tmpdirs)
