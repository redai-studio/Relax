# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""IPC contracts between resident sessions and external agent processes."""

from __future__ import annotations

import argparse
import asyncio
import copy
import ctypes
import fcntl
import hashlib
import json
import os
import re
import shutil
import signal
import socket
import struct
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

from relax.agentic.profile import agentic_trace_events, mark_agentic_event
from relax.utils.env import Envs
from relax.utils.logging_utils import get_logger


# Repository root used as the launcher subprocess import root and local socket
# namespace. Its depth follows the package path ``relax/agentic/runner``;
# moving this module requires updating the relationship atomically.
_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
# Grace period between cooperative termination and forced process-group kill.
# Increasing it retains Session resources longer; decreasing it risks cutting
# off an agent while it is flushing output and cleanup.
_TERMINATION_GRACE_SECONDS = 5.0
_TERMINATION_RESPONSE_TIMEOUT_SECONDS = 2 * _TERMINATION_GRACE_SECONDS + 1.0
# Deadline for the node-local launcher daemon to publish its Unix socket.
# This covers process bootstrap rather than the agent's active timeout.
_LAUNCHER_BOOTSTRAP_TIMEOUT_SECONDS = 30.0
# Wire-format version shared by LauncherClient and the daemon. A mismatch is a
# hard compatibility break and must be rolled out atomically.
_LAUNCHER_PROTOCOL_VERSION = 1
# Four-byte network-order length prefix. Changing it makes every existing
# launcher connection undecodable.
_LAUNCHER_MESSAGE_SIZE = struct.Struct("!I")
# Pending Unix-socket connections queue at the OS boundary. A low value rejects
# bursty launches before application-level scheduling can observe them.
_LAUNCHER_SERVER_BACKLOG = max(1024, socket.SOMAXCONN)
_PR_SET_CHILD_SUBREAPER = 36

logger = get_logger(__name__)


class AgentExecutionError(RuntimeError):
    """Expected agent-level failure that drops its rollout group."""


class LauncherProtocolError(RuntimeError):
    """The node-local launcher returned an invalid message."""


def _encode_launcher_message(payload: Mapping[str, Any]) -> bytes:
    """Encode one length-prefixed launcher message."""

    body = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return _LAUNCHER_MESSAGE_SIZE.pack(len(body)) + body


async def _read_launcher_message(reader: asyncio.StreamReader) -> dict[str, Any]:
    """Decode and validate one launcher message."""

    (size,) = _LAUNCHER_MESSAGE_SIZE.unpack(await reader.readexactly(_LAUNCHER_MESSAGE_SIZE.size))
    if size <= 0:
        raise LauncherProtocolError("launcher message size must be positive")
    payload = json.loads((await reader.readexactly(size)).decode("utf-8"))
    if not isinstance(payload, dict):
        raise LauncherProtocolError("launcher message must be a JSON object")
    return payload


async def _write_launcher_message(writer: asyncio.StreamWriter, payload: Mapping[str, Any]) -> None:
    """Write and drain one launcher message."""

    writer.write(_encode_launcher_message(payload))
    await writer.drain()


class _LauncherProcess:
    """One live daemon connection that owns a process capability."""

    def __init__(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        self._reader = reader
        self._writer = writer
        self._exit = asyncio.create_task(self._read_exit(), name="launcher-process")

    async def wait(self) -> int:
        """Wait for the process exit frame."""

        return await asyncio.shield(self._exit)

    async def terminate(self) -> None:
        """Ask the daemon to terminate and join the complete process group."""

        try:
            if not self._exit.done():
                await _write_launcher_message(self._writer, {"op": "terminate"})
            await asyncio.wait_for(
                asyncio.shield(self._exit),
                timeout=_TERMINATION_RESPONSE_TIMEOUT_SECONDS,
            )
        except Exception:
            await self.close()

    async def close(self) -> None:
        """Release the process capability and trigger daemon cleanup."""

        self._writer.close()
        try:
            await asyncio.wait_for(self._writer.wait_closed(), timeout=_TERMINATION_GRACE_SECONDS)
        except Exception:
            pass
        if not self._exit.done():
            self._exit.cancel()
        await asyncio.gather(self._exit, return_exceptions=True)

    async def _read_exit(self) -> int:
        """Decode the daemon exit frame."""

        response = await _read_launcher_message(self._reader)
        if (
            response.get("op") != "exit"
            or not isinstance(response.get("return_code"), int)
            or not isinstance(response.get("group_closed"), bool)
        ):
            raise LauncherProtocolError(f"launcher expected an exit message, got {response!r}")
        return response["return_code"]


class LauncherClient:
    """Open one ownership connection for each node-local agent process."""

    def __init__(self, socket_path: str) -> None:
        self._socket_path = socket_path

    async def launch(
        self,
        *,
        command: str,
        cwd: str,
        env: Mapping[str, str],
        log_path: str,
        session_dir: str,
    ) -> _LauncherProcess:
        """Launch an agent and return its connection capability."""

        reader, writer = await asyncio.open_unix_connection(self._socket_path)
        try:
            await _write_launcher_message(
                writer,
                {
                    "op": "launch",
                    "command": command,
                    "cwd": cwd,
                    "env": dict(env),
                    "log_path": log_path,
                    "session_dir": session_dir,
                },
            )
            response = await _read_launcher_message(reader)
            error = response.get("error")
            if isinstance(error, str):
                raise RuntimeError(error)
            if response != {"op": "started"}:
                raise LauncherProtocolError(f"launcher expected a started message, got {response!r}")
            return _LauncherProcess(reader, writer)
        except BaseException:
            writer.close()
            try:
                await asyncio.wait_for(writer.wait_closed(), timeout=_TERMINATION_GRACE_SECONDS)
            except Exception:
                pass
            raise


RewardValue = float | dict[str, Any] | None


@dataclass(frozen=True)
class SessionOutput:
    """Agent output normalized at the process boundary."""

    metadata: dict[str, Any] = field(default_factory=dict)
    reward: RewardValue = None
    records: tuple[dict[str, Any], ...] = ()

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "SessionOutput":
        if not isinstance(payload, dict):
            raise TypeError(f"SessionOutput payload must be a JSON object, got {type(payload)}")
        if "messages" in payload:
            return cls.from_records([payload])
        unknown_keys = set(payload) - {"metadata", "reward"}
        if unknown_keys:
            logger.warning(
                "SessionOutput ignoring unknown top-level keys: %s. "
                "Only 'metadata' and 'reward' are consumed by Relax.",
                sorted(unknown_keys),
            )
        metadata = payload.get("metadata", {})
        reward = payload.get("reward")
        if not isinstance(metadata, dict):
            raise TypeError("SessionOutput 'metadata' must be a JSON object")
        if reward is not None and not isinstance(reward, (int, float, dict)):
            raise TypeError("SessionOutput 'reward' must be null, number, or JSON object")
        return cls(metadata=metadata, reward=reward)

    @classmethod
    def from_records(cls, records: list[dict[str, Any]]) -> "SessionOutput":
        normalized_records = tuple(_normalize_session_output_record(record) for record in records)
        names: set[str] = set()
        for record in normalized_records:
            name = record["name"]
            if name in names:
                raise TypeError(f"Session output record name must be unique within one explicit export: {name!r}")
            names.add(name)
        return cls(records=normalized_records)


def _normalize_session_output_record(record: dict[str, Any]) -> dict[str, Any]:
    from relax.agentic.session.state import check_messages, normalize_template_kwargs, normalize_tools

    if not isinstance(record, dict):
        raise TypeError(f"Session output record must be a JSON object, got {type(record)}")
    known_keys = {"name", "messages", "tools", "chat_template_kwargs", "metadata", "reward"}
    unknown_keys = set(record) - known_keys
    if unknown_keys:
        logger.warning(
            "Session output record ignoring unknown keys: %s. Only 'name', 'messages', 'tools', "
            "'chat_template_kwargs', 'metadata', and 'reward' are consumed by Relax.",
            sorted(unknown_keys),
        )
    if "messages" not in record:
        raise TypeError("Session output record must include 'messages'")
    name = record.get("name")
    if not isinstance(name, str) or not name:
        raise TypeError("Session output record must include non-empty string 'name'")
    metadata = record.get("metadata", {})
    if not isinstance(metadata, dict):
        raise TypeError("Session output record 'metadata' must be a JSON object")
    reward = record.get("reward")
    if reward is not None and not isinstance(reward, (int, float, dict)):
        raise TypeError("Session output record 'reward' must be null, number, or JSON object")
    normalized_record = {
        "name": name,
        "messages": check_messages(record["messages"]),
        "metadata": copy.deepcopy(metadata),
    }
    if "reward" in record:
        normalized_record["reward"] = copy.deepcopy(reward)
    if "tools" in record:
        normalized_record["tools"] = normalize_tools(record["tools"])
    if "chat_template_kwargs" in record:
        normalized_record["chat_template_kwargs"] = normalize_template_kwargs(record["chat_template_kwargs"])
    return normalized_record


def _session_output_from_text(raw_text: str) -> SessionOutput:
    stripped = raw_text.strip()
    if not stripped:
        return SessionOutput()
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        records = [json.loads(line) for line in stripped.split("\n") if line.strip()]
        return SessionOutput.from_records(records)
    if isinstance(payload, dict):
        return SessionOutput.from_payload(payload)
    if isinstance(payload, list):
        raise TypeError("Session output explicit records must be written as JSONL, not a JSON array")
    raise TypeError(f"Managed command output must be a JSON object or JSONL records, got {type(payload)}")


@dataclass(frozen=True)
class ManagedCommandAppSpec:
    """Normalized agent command, cwd, environment, and timeout."""

    command: str
    cwd: str
    env: Mapping[str, str] = field(default_factory=dict)
    timeout_s: float = 1800.0

    def __post_init__(self) -> None:
        if not self.command.strip():
            raise ValueError("managed agent command must be non-empty")
        if not Path(self.cwd).is_dir():
            raise ValueError(f"managed agent cwd must be an existing directory: {self.cwd}")
        if self.timeout_s <= 0:
            raise ValueError("managed agent timeout must be positive")


def load_agent_app_spec_from_args(args: argparse.Namespace) -> ManagedCommandAppSpec:
    """Normalize CLI-shaped agent settings once."""

    agent_env = {}
    for item in args.agent_env:
        key, value = item.split("=", 1)
        agent_env[key.strip()] = value
    return ManagedCommandAppSpec(
        command=args.agent_command.strip(),
        cwd=str(Path(args.agent_cwd).expanduser().resolve()),
        env=agent_env,
        timeout_s=float(args.agent_timeout),
    )


@dataclass(frozen=True)
class SessionInput:
    """Agent entrypoint input."""

    session_id: str
    rollout_mode: str
    group_id: str
    input_payload: dict[str, Any] = field(default_factory=dict)

    def to_agent_payload(self) -> dict[str, Any]:
        return copy.deepcopy(self.input_payload)


class ManagedAgentProcess:
    """One Session-owned command process with an active-time budget."""

    def __init__(
        self,
        launcher: LauncherClient,
        spec: ManagedCommandAppSpec,
        session_input: SessionInput,
    ) -> None:
        self._launcher = launcher
        self._spec = spec
        self._session_id = session_input.session_id
        self._group_id = session_input.group_id
        self._rollout_mode = session_input.rollout_mode
        self._remaining_timeout = spec.timeout_s
        self._process: Optional[_LauncherProcess] = None
        self._launch_finished = asyncio.Event()
        self._timeout_started: Optional[float] = None
        self._timeout_task: Optional["asyncio.Task[None]"] = None
        self._timed_out = False
        self._terminated = False
        self._output = SessionOutput()
        self._task = asyncio.create_task(
            self._run(session_input.to_agent_payload()),
            name=f"managed-agent:{session_input.session_id}",
        )
        self._task.add_done_callback(lambda _: self._launch_finished.set())

    async def wait(self) -> None:
        """Wait for the managed process task."""

        await asyncio.shield(self._task)

    @property
    def output(self) -> SessionOutput:
        """Expose terminal agent output after process completion."""

        return self._output

    async def set_timeout_active(self, active: bool) -> None:
        """Start or pause the process active-time budget."""

        if self._task.done():
            return
        if active:
            if self._timeout_started is not None:
                return
            self._timeout_started = time.monotonic()
            self._timeout_task = asyncio.create_task(
                self._expire_timeout(),
                name=f"managed-agent-timeout:{self._session_id}",
            )
            return
        await self._pause_timeout()

    async def terminate_and_join(self) -> None:
        """Terminate the process group and join its task."""

        self._terminated = True
        await self._pause_timeout()

        async def terminate() -> None:
            await self._launch_finished.wait()
            process = self._process
            if process is not None:
                await process.terminate()
            await asyncio.gather(self._task, return_exceptions=True)

        try:
            await asyncio.wait_for(
                terminate(),
                timeout=_TERMINATION_RESPONSE_TIMEOUT_SECONDS + _TERMINATION_GRACE_SECONDS,
            )
        except Exception:
            pass
        finally:
            if not self._task.done():
                self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)

    async def _run(self, input_payload: Mapping[str, Any]) -> None:
        """Run one external agent process for its owning Session."""

        process_events: dict[str, Any] = {}
        mark_agentic_event(process_events, "managed_prepare_start_at")
        with tempfile.TemporaryDirectory(prefix="relax-agentic-command-") as session_dir:
            session_path = Path(session_dir)
            input_path = session_path / "session_input.json"
            output_path = session_path / "session_output.json"
            log_path = session_path / "command.log"
            input_path.write_text(
                json.dumps(input_payload, separators=(",", ":"), ensure_ascii=False),
                encoding="utf-8",
            )
            del input_payload
            env = {
                **os.environ,
                **{str(key): str(value) for key, value in self._spec.env.items()},
                "RELAX_SESSION_ID": self._session_id,
                "RELAX_ROLLOUT_MODE": self._rollout_mode,
                "RELAX_GROUP_ID": self._group_id,
                "RELAX_SESSION_IO_DIR": session_dir,
                "RELAX_INPUT_JSON": str(input_path),
                "RELAX_OUTPUT_JSON": str(output_path),
            }
            mark_agentic_event(process_events, "managed_prepare_end_at")
            try:
                mark_agentic_event(process_events, "managed_launch_start_at")
                mark_agentic_event(process_events, "managed_process_start_at")
                self._process = await self._launcher.launch(
                    command=self._spec.command,
                    cwd=self._spec.cwd,
                    env=env,
                    log_path=str(log_path),
                    session_dir=session_dir,
                )
                mark_agentic_event(process_events, "managed_process_spawn_return_at")
                mark_agentic_event(process_events, "managed_launch_return_at")
                self._launch_finished.set()
                return_code = await self._process.wait()
                mark_agentic_event(process_events, "managed_process_exit_at")
                log_tail = _read_log_tail(log_path)
                if self._timed_out:
                    raise AgentExecutionError(
                        f"Managed command agent timed out after {self._spec.timeout_s} seconds.\n{log_tail}".rstrip()
                    )
                if return_code != 0 and not self._terminated:
                    raise AgentExecutionError(
                        f"Managed command agent exited with code {return_code}.\n{log_tail}".rstrip()
                    )
                if output_path.exists():
                    try:
                        output_text = output_path.read_text(encoding="utf-8")
                        mark_agentic_event(process_events, "managed_output_read_at")
                        self._output = _session_output_from_text(output_text)
                    except Exception as error:
                        raise AgentExecutionError(
                            f"Managed command agent produced invalid output.\n{log_tail}".rstrip()
                        ) from error
                mark_agentic_event(process_events, "managed_output_ready_at")
                agentic_trace_events(self._output.metadata).update(process_events)
            finally:
                self._launch_finished.set()
                if self._process is not None:
                    await self._process.close()
                    self._process = None
                await self._pause_timeout()

    async def _expire_timeout(self) -> None:
        """Terminate the process after its active-time budget expires."""

        await asyncio.sleep(self._remaining_timeout)
        self._timed_out = True
        await self._launch_finished.wait()
        process = self._process
        if process is not None:
            await process.terminate()

    async def _pause_timeout(self) -> None:
        """Retain the remaining timeout budget across generation gates."""

        started = self._timeout_started
        self._timeout_started = None
        if started is not None:
            self._remaining_timeout -= time.monotonic() - started
        timeout_task = self._timeout_task
        self._timeout_task = None
        if timeout_task is not None and timeout_task is not asyncio.current_task():
            timeout_task.cancel()
            await asyncio.gather(timeout_task, return_exceptions=True)


def _read_log_tail(path: Path, limit: int = 8192) -> str:
    """Read bounded managed-agent diagnostics."""

    with path.open("rb") as log_file:
        log_file.seek(0, os.SEEK_END)
        size = log_file.tell()
        log_file.seek(max(0, size - limit))
        return log_file.read().decode("utf-8", errors="replace")


class ManagedAgentLauncher:
    """Bind the managed command template to the node-local launcher."""

    def __init__(self, app: ManagedCommandAppSpec, base_url: str) -> None:
        self._app = ManagedCommandAppSpec(
            command=app.command,
            cwd=app.cwd,
            env={**app.env, "RELAX_BASE_URL": base_url},
            timeout_s=app.timeout_s,
        )
        self._launcher = LauncherClient(ensure_local_launcher_daemon())

    def start_agent(self, session_input: SessionInput) -> ManagedAgentProcess:
        """Create one process whose lifetime is owned by the Session record."""

        return ManagedAgentProcess(
            self._launcher,
            self._app,
            session_input,
        )


def _launcher_token(value: str) -> str:
    """Normalize one launcher namespace token."""

    return re.sub(r"[^0-9A-Za-z_.-]+", "-", value).strip("-") or "unknown"


def _launcher_namespace() -> str:
    """Derive the per-Ray-job launcher namespace."""

    explicit = Envs.RELAX_LAUNCHER_NAMESPACE
    if explicit:
        return _launcher_token(explicit)
    ray_job_id = Envs.RAY_JOB_ID
    if ray_job_id:
        return _launcher_token(ray_job_id)
    repository = str(_REPOSITORY_ROOT)
    return f"local-{hashlib.sha256(repository.encode()).hexdigest()[:12]}"


def launcher_socket_path() -> str:
    """Return the node-local launcher socket path."""

    user = _launcher_token(Envs.USER or Envs.LOGNAME or str(os.getuid()))
    return f"/tmp/relax-agentic-launcher-{user}-{_launcher_namespace()}.sock"


def _ping_launcher(socket_path: str) -> None:
    """Probe launcher readiness and protocol compatibility."""

    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.settimeout(1.0)
        client.connect(socket_path)
        client.sendall(_encode_launcher_message({"op": "ping"}))
        raw_size = bytearray()
        while len(raw_size) < _LAUNCHER_MESSAGE_SIZE.size:
            chunk = client.recv(_LAUNCHER_MESSAGE_SIZE.size - len(raw_size))
            if not chunk:
                raise LauncherProtocolError("launcher ping returned an incomplete size")
            raw_size.extend(chunk)
        (size,) = _LAUNCHER_MESSAGE_SIZE.unpack(raw_size)
        body = bytearray()
        while len(body) < size:
            chunk = client.recv(size - len(body))
            if not chunk:
                raise LauncherProtocolError("launcher ping returned an incomplete body")
            body.extend(chunk)
        if json.loads(body) != {"ok": True, "protocol": _LAUNCHER_PROTOCOL_VERSION}:
            raise LauncherProtocolError("launcher ping returned an invalid response")


def _wait_for_launcher(socket_path: str) -> None:
    """Wait for bounded launcher daemon bootstrap."""

    deadline = time.monotonic() + _LAUNCHER_BOOTSTRAP_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        try:
            _ping_launcher(socket_path)
            return
        except (OSError, LauncherProtocolError, json.JSONDecodeError):
            time.sleep(0.05)
    raise RuntimeError(f"launcher daemon did not become ready: {socket_path}")


def _spawn_launcher_daemon(socket_path: str) -> None:
    """Spawn the node-local launcher daemon."""

    with Path(f"{socket_path}.log").open("a", encoding="utf-8") as log_file:
        subprocess.Popen(
            [
                sys.executable,
                "-c",
                f"from {__name__} import _main; _main()",
                "--launcher-socket",
                socket_path,
            ],
            cwd=_REPOSITORY_ROOT,
            env=os.environ.copy(),
            stdin=subprocess.DEVNULL,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )


def ensure_local_launcher_daemon() -> str:
    """Return this node's launcher socket, starting its daemon when needed."""

    socket_path = launcher_socket_path()
    with open(f"{socket_path}.lock", "a+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        if Path(socket_path).exists():
            try:
                _ping_launcher(socket_path)
                return socket_path
            except (OSError, LauncherProtocolError, json.JSONDecodeError):
                Path(socket_path).unlink()
        _spawn_launcher_daemon(socket_path)
        _wait_for_launcher(socket_path)
        return socket_path


class _ProcessGroup:
    """Daemon-owned lifetime for one launcher command and all descendants."""

    def __init__(self, process: asyncio.subprocess.Process) -> None:
        self._process = process
        # start_new_session=True makes the child a session and process-group
        # leader, so its PID is the stable PGID even if the leader exits before
        # the daemon can query the kernel again.
        self.pgid = process.pid
        self.exit_task = asyncio.create_task(process.wait(), name=f"launcher-pgid:{self.pgid}")
        self._termination_task: Optional["asyncio.Task[tuple[int, bool]]"] = None

    def alive(self) -> bool:
        try:
            os.killpg(self.pgid, 0)
            return True
        except ProcessLookupError:
            return False

    def signal(self, signal_value: signal.Signals) -> None:
        try:
            os.killpg(self.pgid, signal_value)
        except ProcessLookupError:
            pass

    async def terminate_and_join(self) -> tuple[int, bool]:
        """Close the whole group once, escalating from TERM to KILL."""

        if self._termination_task is None:
            self._termination_task = asyncio.create_task(
                self._terminate_and_join(),
                name=f"launcher-terminate-pgid:{self.pgid}",
            )
        try:
            return await asyncio.shield(self._termination_task)
        except asyncio.CancelledError:
            await self._termination_task
            raise

    async def _terminate_and_join(self) -> tuple[int, bool]:
        if await self._wait_closed(0):
            return self._return_code(), True
        self.signal(signal.SIGTERM)
        if not await self._wait_closed(_TERMINATION_GRACE_SECONDS):
            self.signal(signal.SIGKILL)
            if not await self._wait_closed(_TERMINATION_GRACE_SECONDS):
                return self._return_code(), False
        return self._return_code(), True

    def _return_code(self) -> int:
        return_code = self._process.returncode
        return return_code if return_code is not None else -int(signal.SIGKILL)

    async def _wait_closed(self, timeout: float) -> bool:
        deadline = time.monotonic() + max(timeout, 0.0)
        while True:
            if self.exit_task.done():
                self._reap_exited_descendants()
                if not self.alive():
                    return True
            if time.monotonic() >= deadline:
                return False
            await asyncio.sleep(min(0.05, deadline - time.monotonic()))

    def _reap_exited_descendants(self) -> None:
        while True:
            try:
                pid, _ = os.waitpid(-self.pgid, os.WNOHANG)
            except ChildProcessError:
                return
            if pid <= 0:
                return


async def _supervise_launcher_process(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    process_group: _ProcessGroup,
    session_dir: Path,
) -> None:
    """Bind process lifetime and orphan cleanup to its owner connection."""

    request_task = asyncio.create_task(_read_launcher_message(reader))
    owner_disappeared = False
    try:
        done, _ = await asyncio.wait(
            (process_group.exit_task, request_task),
            return_when=asyncio.FIRST_COMPLETED,
        )
        if process_group.exit_task not in done:
            try:
                request = request_task.result()
            except asyncio.IncompleteReadError:
                owner_disappeared = True
                return
            if request.get("op") != "terminate":
                raise LauncherProtocolError(f"launcher expected a terminate message, got {request!r}")
        return_code, group_closed = await process_group.terminate_and_join()
        try:
            await _write_launcher_message(
                writer,
                {"op": "exit", "return_code": return_code, "group_closed": group_closed},
            )
        except (BrokenPipeError, ConnectionResetError):
            owner_disappeared = True
            return
        if not request_task.done():
            request_task.cancel()
            await asyncio.gather(request_task, return_exceptions=True)
        await reader.read()
    finally:
        await process_group.terminate_and_join()
        if not request_task.done():
            request_task.cancel()
        await asyncio.gather(request_task, return_exceptions=True)
        if owner_disappeared:
            try:
                shutil.rmtree(session_dir)
            except FileNotFoundError:
                pass


async def _launch_for_client(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    request: Mapping[str, Any],
) -> None:
    """Execute one command with file-backed diagnostics."""

    command = request.get("command")
    cwd = request.get("cwd")
    env = request.get("env")
    log_path = request.get("log_path")
    session_dir = request.get("session_dir")
    if not all(isinstance(value, str) for value in (command, cwd, log_path, session_dir)):
        raise LauncherProtocolError("launcher command, cwd, log_path, and session_dir must be strings")
    if not isinstance(env, dict):
        raise LauncherProtocolError("launcher env must be a JSON object")
    with Path(log_path).open("ab") as log_file:
        process = await asyncio.create_subprocess_exec(
            "/bin/bash",
            "-lc",
            command,
            cwd=cwd,
            env={str(key): str(value) for key, value in env.items()},
            stdin=asyncio.subprocess.DEVNULL,
            stdout=log_file,
            stderr=asyncio.subprocess.STDOUT,
            start_new_session=True,
        )
    process_group = _ProcessGroup(process)
    try:
        await _write_launcher_message(writer, {"op": "started"})
        await _supervise_launcher_process(reader, writer, process_group, Path(session_dir))
    except (BrokenPipeError, ConnectionResetError):
        try:
            shutil.rmtree(session_dir)
        except FileNotFoundError:
            pass
    finally:
        await process_group.terminate_and_join()


async def _handle_launcher_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    """Dispatch ping or launch for one daemon connection."""

    try:
        request = await _read_launcher_message(reader)
        if request.get("op") == "ping":
            await _write_launcher_message(
                writer,
                {"ok": True, "protocol": _LAUNCHER_PROTOCOL_VERSION},
            )
        elif request.get("op") == "launch":
            await _launch_for_client(reader, writer, request)
        else:
            raise LauncherProtocolError(f"unknown launcher operation: {request.get('op')!r}")
    except asyncio.IncompleteReadError:
        pass
    except Exception as error:
        try:
            await _write_launcher_message(writer, {"error": str(error)})
        except (BrokenPipeError, ConnectionResetError):
            pass
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except (BrokenPipeError, ConnectionResetError):
            pass


async def _run_launcher_daemon(socket_path: str) -> None:
    """Run the node-local launcher Unix server."""

    socket_file = Path(socket_path)
    socket_file.unlink(missing_ok=True)
    loop = asyncio.get_running_loop()
    stop_requested = asyncio.Event()
    for signal_value in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(signal_value, stop_requested.set)
    try:
        server = await asyncio.start_unix_server(
            _handle_launcher_client,
            path=socket_path,
            backlog=_LAUNCHER_SERVER_BACKLOG,
        )
        async with server:
            await stop_requested.wait()
    finally:
        for signal_value in (signal.SIGTERM, signal.SIGINT):
            loop.remove_signal_handler(signal_value)
        socket_file.unlink(missing_ok=True)


def _enable_child_subreaper() -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    prctl = libc.prctl
    prctl.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong]
    prctl.restype = ctypes.c_int
    if prctl(_PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0) != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number))


def _main() -> None:
    """Run the launcher daemon CLI entrypoint."""

    parser = argparse.ArgumentParser(description="Relax agentic launcher daemon")
    parser.add_argument("--launcher-socket", required=True)
    args = parser.parse_args()
    _enable_child_subreaper()
    asyncio.run(_run_launcher_daemon(args.launcher_socket))


if __name__ == "__main__":
    _main()
