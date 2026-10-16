# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Code-generation instance startup, tracked children and confirmed cleanup."""

import ctypes
import logging
import os
import signal
import subprocess
import tempfile
import threading
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from minisweagent.environments.singularity import SingularityEnvironment, SingularityEnvironmentConfig


INSTANCE_PREFIX = "mswe-"


CHILD_KILL_WAIT_TIMEOUT_SECONDS = 5


_INSTANCE_STOP_TIMEOUT_S = 10


_INSTANCE_STOP_VERIFY_ROUNDS = 40


_INSTANCE_STOP_VERIFY_INTERVAL_S = 0.25


MAX_ERROR_OUTPUT_CHARS = 4000


_owned_child_pids: set[int] = set()


_owned_lock = threading.Lock()


def _register_child(pid: int) -> None:
    with _owned_lock:
        _owned_child_pids.add(pid)


def _unregister_child(pid: int) -> None:
    with _owned_lock:
        _owned_child_pids.discard(pid)


def _kill_and_wait(proc: subprocess.Popen, *, process_group: bool = False) -> None:
    if process_group:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    else:
        proc.kill()
    try:
        proc.wait(timeout=CHILD_KILL_WAIT_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        pass


def _tracked_run(
    cmd: list[str],
    *,
    check: bool = False,
    timeout: float | None = None,
    cancel_event: threading.Event | None = None,
    **kwargs: Any,
) -> subprocess.CompletedProcess:
    """Drop-in for the subset of ``subprocess.run`` this module uses, but the
    child pid is registered in ``_owned_child_pids`` for the whole wait so the
    orphan reaper never reaps it out from under us (see
    ``_owned_child_pids``)."""
    kwargs.setdefault("stdout", subprocess.PIPE)
    proc = subprocess.Popen(cmd, **kwargs)
    _register_child(proc.pid)
    try:
        try:
            if cancel_event is None:
                stdout, stderr = proc.communicate(timeout=timeout)
            else:
                deadline = time.monotonic() + timeout if timeout is not None else float("inf")
                while True:
                    if cancel_event.is_set():
                        raise RuntimeError("command cancelled")
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise subprocess.TimeoutExpired(cmd, timeout)
                    try:
                        stdout, stderr = proc.communicate(timeout=min(0.2, remaining))
                        break
                    except subprocess.TimeoutExpired:
                        continue
        except subprocess.TimeoutExpired as exc:
            _kill_and_wait(proc, process_group=kwargs.get("start_new_session", False))
            raise subprocess.TimeoutExpired(cmd, timeout, output=exc.output, stderr=exc.stderr) from None
        except BaseException:
            _kill_and_wait(proc, process_group=kwargs.get("start_new_session", False))
            raise
        retcode = proc.returncode
    finally:
        _unregister_child(proc.pid)
    if check and retcode:
        raise subprocess.CalledProcessError(retcode, cmd, output=stdout, stderr=stderr)
    return subprocess.CompletedProcess(cmd, retcode, stdout, stderr)


def _install_child_subreaper_and_reaper() -> None:
    """Make this process reap orphaned apptainer descendants.

    PID 1 in the run container is a bare ``sleep`` that never wait()s, so
    apptainer ``starter`` processes orphaned on instance stop reparent to init
    and become permanent zombies (observed: thousands, load avg > 200, which in
    turn slows every apptainer exec until reward commands time out).
    Registering as a child subreaper causes those orphans to reparent to this
    process instead of init; a background thread then reaps them.
    """
    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        pr_set_child_subreaper = 36  # PR_SET_CHILD_SUBREAPER from <sys/prctl.h>
        if libc.prctl(pr_set_child_subreaper, 1, 0, 0, 0) != 0:
            logging.warning("prctl(PR_SET_CHILD_SUBREAPER) failed: errno=%d", ctypes.get_errno())
    except Exception as exc:  # ctypes/prctl unavailable — degrade gracefully
        logging.warning("could not set child subreaper: %s", exc)

    def _reap_loop() -> None:
        while True:
            try:
                _reap_orphan_children()
            except Exception as exc:
                logging.debug("reaper error: %s", exc)
            time.sleep(1.0)

    threading.Thread(target=_reap_loop, name="zombie-reaper", daemon=True).start()


def _reap_orphan_children() -> int:
    """Reap reparented (orphaned) descendants of this process.

    Enumerates our children via ``/proc`` and ``waitpid``s only those NOT in
    ``_owned_child_pids`` -- i.e. daemonized apptainer ``starter`` orphans that
    no ``_tracked_run`` is waiting on. This is deliberately NOT ``waitpid(-1)``:
    an indiscriminate reap steals a live subprocess's exit status (see
    ``_owned_child_pids``). Returns the number of children actually reaped.
    """
    my_pid = os.getpid()
    reaped = 0
    try:
        entries = os.listdir("/proc")
    except OSError:
        return 0
    for entry in entries:
        if not entry.isdigit():
            continue
        pid = int(entry)
        with _owned_lock:
            if pid in _owned_child_pids:
                continue
        if _ppid(pid) != my_pid:
            continue
        try:
            waited, _status = os.waitpid(pid, os.WNOHANG)
        except ChildProcessError:
            continue
        except OSError:
            continue
        if waited:
            reaped += 1
    return reaped


def _drain_orphans(deadline_s: float = 30.0) -> None:
    """Synchronously reap every orphaned descendant before the process exits.

    The background reaper (``_reap_loop``) is a daemon thread: once the main
    thread exits it dies immediately, so any orphan produced during shutdown
    (``_stop_apptainer_instance`` SIGKILLs each ``starter``, orphaning its
    descendants) reparents to the non-reaping PID 1 (``sleep``) and leaks as a
    permanent zombie. Called at the tail of ``shutdown()``, this blocks the
    main thread and keeps reaping until the tree is fully drained (three
    consecutive quiet rounds -> stragglers have settled) or the deadline hits,
    so children are reaped here instead of being handed to PID 1.
    """
    end = time.monotonic() + deadline_s
    idle_rounds = 0
    while time.monotonic() < end:
        reaped = _reap_orphan_children()
        idle_rounds = idle_rounds + 1 if reaped == 0 else 0
        if idle_rounds >= 3:  # 3 quiet rounds (~0.6s) -> stragglers settled
            return
        time.sleep(0.2)


def _apptainer_instances(executable: str, global_args: list[str]) -> dict[str, int]:
    result = _tracked_run(
        [executable, *global_args, "instance", "list"],
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
    )
    instances = {}
    for line in result.stdout.splitlines()[1:]:
        parts = line.split()
        if len(parts) >= 2 and parts[0].startswith(INSTANCE_PREFIX):
            instances[parts[0]] = int(parts[1])
    return instances


def _ppid(pid: int) -> int:
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("PPid:"):
                return int(line.split()[1])
    except OSError:
        # ENOENT / ESRCH: the pid vanished between /proc enumeration and this
        # read (routine in the reaper's scan). Treat as "not our child".
        pass
    return 0


def _stop_apptainer_instance(executable: str, global_args: list[str], instance_name: str) -> str:
    # Graceful stop FIRST. The previous `--force --timeout 1` SIGKILLed the
    # instance after ~1s; SIGKILL never unmounts the squashfs rootfs, so its
    # loop device stays attached and LEAKS. Under sustained load the leaked loop
    # devices exhaust the host (~104 available) and every new `instance start`
    # fails with "no loop devices available" -> rollout hangs. A graceful stop
    # lets apptainer unmount and detach the loop before we fall back to SIGKILL.
    result = _tracked_run(
        [executable, *global_args, "instance", "stop", "--timeout", str(_INSTANCE_STOP_TIMEOUT_S), instance_name],
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    # Do NOT early-return on a non-zero graceful stop: fall through and verify /
    # force-kill so a stubborn instance still gets torn down.
    pid = _apptainer_instances(executable, global_args).get(instance_name)
    if pid is None:
        return ""
    try:
        parent_pid = _ppid(pid)
        os.kill(parent_pid if parent_pid > 1 else pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except OSError as exc:
        return f"failed to kill stale Apptainer process: {exc}"
    for _ in range(_INSTANCE_STOP_VERIFY_ROUNDS):
        time.sleep(_INSTANCE_STOP_VERIFY_INTERVAL_S)
        if instance_name not in _apptainer_instances(executable, global_args):
            return ""
    return f"instance still listed after stop: {result.stdout.strip()}"


def _new_instance_name() -> str:
    # Embed the owning server pid so startup/shutdown cleanup can scope itself to
    # THIS server's instances (or a provably-dead server's) and never SIGKILL a
    # live co-tenant's sandboxes -- the previous flat `mswe-<uuid>` name carried
    # no owner, so any server sweeping the shared `mswe-` prefix wiped every
    # server's instances on the node.
    return f"{INSTANCE_PREFIX}{os.getpid()}-{uuid.uuid4().hex[:12]}"


def _instance_owner_pid(name: str) -> int | None:
    """Owner pid encoded by ``_new_instance_name`` (``None`` for legacy
    names)."""
    if not name.startswith(INSTANCE_PREFIX):
        return None
    owner, _, rest = name[len(INSTANCE_PREFIX) :].partition("-")
    if not rest:  # no `<uuid>` tail -> legacy `mswe-<uuid>` name, no owner
        return None
    try:
        return int(owner)
    except ValueError:
        return None


def _trim_error_output(output: str) -> str:
    if len(output) <= MAX_ERROR_OUTPUT_CHARS:
        return output
    return output[:MAX_ERROR_OUTPUT_CHARS] + f"\n... truncated {len(output) - MAX_ERROR_OUTPUT_CHARS} chars"


def _decode_process_output(output: Any) -> str:
    if isinstance(output, bytes):
        return output.decode("utf-8", errors="replace")
    return str(output or "")


def _format_called_process_error(exc: subprocess.CalledProcessError) -> str:
    stdout = _decode_process_output(getattr(exc, "stdout", None) or getattr(exc, "output", None))
    stderr = _decode_process_output(getattr(exc, "stderr", None))
    parts = [f"CalledProcessError: command={exc.cmd!r} returncode={exc.returncode}"]
    if stdout:
        parts.append("stdout/stderr:\n" + _trim_error_output(stdout.strip()))
    if stderr:
        parts.append("stderr:\n" + _trim_error_output(stderr.strip()))
    return "\n".join(parts)


class ApptainerInstanceEnvironment(SingularityEnvironment):
    def __init__(
        self,
        *,
        tmp_dir: Path,
        instance_name: str | None = None,
        release_sif: Callable[[], None] | None = None,
        process_env: dict[str, str] | None = None,
        **kwargs: Any,
    ):
        # instance_name may be supplied by the caller so it can stop a leaked
        # instance even if this constructor raises AFTER `_build_sandbox` has
        # already `instance start`ed one (in that case `env` is never bound in
        # the caller, so `env.cleanup()` would be skipped -> instance leak).
        self.instance_name = instance_name or _new_instance_name()
        self._cleanup_lock = threading.Lock()
        self._release_sif = release_sif
        self.apptainer_env = {
            **(os.environ if process_env is None else process_env),
            "TMPDIR": str(tmp_dir),
            "APPTAINER_TMPDIR": str(tmp_dir),
            "APPTAINER_DISABLE_CACHE": "true",
        }
        super().__init__(config_class=SingularityEnvironmentConfig, **kwargs)

    def _build_sandbox(self) -> str:
        cmd = [
            self.config.executable,
            *self.config.global_args,
            "instance",
            "start",
            *self.config.exec_args,
            "--writable-tmpfs",
            self.config.image,
            self.instance_name,
        ]
        max_retries = max(1, int(self.config.sandbox_build_retries))
        last_error = ""
        for attempt in range(max_retries):
            try:
                with tempfile.TemporaryFile() as start_output:
                    result = _tracked_run(
                        cmd,
                        env=self.apptainer_env,
                        stdout=start_output,
                        stderr=subprocess.STDOUT,
                    )
                    output_size = os.fstat(start_output.fileno()).st_size
                    start_output.seek(0)
                    output = start_output.read(output_size)
                if result.returncode:
                    raise subprocess.CalledProcessError(result.returncode, cmd, output=output)
                return f"instance://{self.instance_name}"
            except subprocess.CalledProcessError as exc:
                last_error = _format_called_process_error(exc)
                try:
                    _stop_apptainer_instance(self.config.executable, self.config.global_args, self.instance_name)
                except Exception as stop_exc:
                    last_error += f"\nfailed to stop partial instance after start failure: {stop_exc}"
                if attempt == max_retries - 1:
                    raise RuntimeError(
                        f"Apptainer instance start failed after {max_retries} attempt(s).\n{last_error}"
                    ) from exc
                time.sleep(min(5.0, 1.0 + attempt))
        raise RuntimeError(f"Apptainer instance start failed.\n{last_error}")

    def cleanup(self) -> None:
        with self._cleanup_lock:
            if self.instance_name:
                instance_name = self.instance_name
                error = _stop_apptainer_instance(self.config.executable, self.config.global_args, instance_name)
                if error:
                    raise RuntimeError(f"failed to stop Apptainer instance {instance_name}: {error}")
                self.instance_name = ""
            if self._release_sif is not None:
                self._release_sif()
                self._release_sif = None
