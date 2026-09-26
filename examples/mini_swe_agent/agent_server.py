#!/usr/bin/env python3
# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import argparse
import copy
import ctypes
import fcntl
import json
import logging
import os
import random
import re
import shutil
import signal
import subprocess
import tempfile
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from glob import glob
from pathlib import Path
from typing import Any, Iterator

import pyarrow.dataset as ds
from flask import Flask, jsonify, request
from litellm.exceptions import ContextWindowExceededError
from minisweagent.agents.default import DefaultAgent
from minisweagent.config import get_config_from_spec
from minisweagent.environments.singularity import SingularityEnvironment, SingularityEnvironmentConfig
from minisweagent.exceptions import FormatError, Submitted
from minisweagent.models.litellm_model import LitellmModel
from minisweagent.utils.serialize import recursive_merge


INSTANCE_PREFIX = "mswe-"
SETUP_AND_REWARD_TIMEOUT_SECONDS = 300
CHILD_KILL_WAIT_TIMEOUT_SECONDS = 5


# Reward (run_tests.sh = 4000+ pytest) is CPU-bound, unlike the IO-bound agent
# turns; both otherwise share the session ThreadPoolExecutor (train_concurrency).
# Cap concurrent reward execs independently so a reward wave can't saturate the
# node run-queue and starve the Ray dashboard / monitoring HTTP. Sized off node
# cores (cores//8); AGENT_SERVER_REWARD_CONCURRENCY overrides, 0 = unlimited.
def _reward_concurrency_limit() -> int:
    raw = os.environ.get("AGENT_SERVER_REWARD_CONCURRENCY", "").strip()
    if raw:
        return int(raw)
    return max(4, (os.cpu_count() or 32) // 8)


_REWARD_CONCURRENCY = _reward_concurrency_limit()
_reward_semaphore = threading.Semaphore(_REWARD_CONCURRENCY) if _REWARD_CONCURRENCY > 0 else None
# Graceful `instance stop` timeout (s). Must be generous enough for apptainer to
# unmount the squashfs rootfs and detach its loop device before we SIGKILL --
# a hard kill leaks the loop device (see _stop_apptainer_instance).
_INSTANCE_STOP_TIMEOUT_S = 10
# After a fallback SIGKILL, how long to wait for the instance to disappear.
_INSTANCE_STOP_VERIFY_ROUNDS = 40
_INSTANCE_STOP_VERIFY_INTERVAL_S = 0.25
TERMINAL_STATUSES = {"completed", "failed", "cancelled"}
MAX_ERROR_OUTPUT_CHARS = 4000
# Terminal SessionRecords are kept this long after finishing so a client GET is
# retryable across a dropped response, then reclaimed by the background sweeper.
# Overridable via AGENT_SERVER_RECORD_TTL_SECONDS.
SESSION_RECORD_TTL_SECONDS = 300


# PIDs this process is actively waiting on via `_tracked_run`. The orphan reaper
# must never `waitpid()` one of these: if it wins the race, `communicate()` sees
# ECHILD and reports returncode 0, silently turning a failed apptainer / setup /
# reward command into a false success (`check=True` never fires).
_owned_child_pids: set[int] = set()
_owned_lock = threading.Lock()


def _register_child(pid: int) -> None:
    with _owned_lock:
        _owned_child_pids.add(pid)


def _unregister_child(pid: int) -> None:
    with _owned_lock:
        _owned_child_pids.discard(pid)


def _kill_and_wait(proc: subprocess.Popen) -> None:
    proc.kill()
    try:
        proc.wait(timeout=CHILD_KILL_WAIT_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        pass


def _tracked_run(
    cmd: list[str], *, check: bool = False, timeout: float | None = None, **kwargs: Any
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
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            _kill_and_wait(proc)
            raise subprocess.TimeoutExpired(cmd, timeout, output=exc.output, stderr=exc.stderr) from None
        except BaseException:
            _kill_and_wait(proc)
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


def _image_key(docker_image: Any) -> str:
    return str(docker_image).replace("/", "__").replace(":", "__")


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


def _stop_mswe_instances(executable: str, global_args: list[str], *, keep: Any) -> None:
    """Stop every ``mswe-`` instance for which ``keep(name)`` is falsy."""
    errors = []
    for name in _apptainer_instances(executable, global_args):
        if keep(name):
            continue
        if error := _stop_apptainer_instance(executable, global_args, name):
            errors.append(f"{name}: {error}")
    if errors:
        raise RuntimeError("Failed to stop stale Apptainer instances:\n" + "\n".join(errors))


@contextmanager
def _startup_cleanup_lock(lock_path: Path) -> Iterator[None]:
    """Serialize startup reclaim across servers sharing a node so two starting
    servers cannot both race to reap the same dead server's
    instances/scratch."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


class ModeState:
    def __init__(
        self,
        data_path: str,
        mode: str,
        config: dict[str, Any],
        *,
        available_images: set[str] | None = None,
        shuffle: bool = False,
        seed: int = 42,
    ) -> None:
        files = sorted(glob(os.path.join(data_path, config["directory"], config["pattern"])))
        if not files:
            raise FileNotFoundError(f"No dataset files matched mode={mode} under {data_path}")
        self.dataset = ds.dataset(files, format="parquet")
        self.columns = config.get("columns")
        total = self.dataset.count_rows()
        indices = list(range(total))
        if available_images is not None:
            images = self.dataset.take(indices, columns=["docker_image"]).column("docker_image").to_pylist()
            indices = [i for i in indices if _image_key(images[i]) in available_images]
            dropped = total - len(indices)
            if dropped:
                logging.warning("mode=%s dropped %d/%d samples with no matching SIF image", mode, dropped, total)
        if "count" in config:
            indices = indices[: int(config["count"])]
        if not indices:
            raise RuntimeError(f"No usable samples for mode={mode} (missing SIF images or empty dataset)")
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self.order = indices
        if self.shuffle:
            random.Random(self.seed).shuffle(self.order)
        self.next_idx = 0
        self.count = len(self.order)
        self.samples = int(config["samples"])
        self.cache: dict[str, tuple[int, int]] = {}

    def lease(self) -> int:
        sample_idx = self.order[self.next_idx]
        self.next_idx += 1
        if self.next_idx >= self.count:
            self.next_idx = 0
            self.epoch += 1
            if self.shuffle:
                random.Random(self.seed + self.epoch).shuffle(self.order)
        return sample_idx

    def get(self, sample_idx: int) -> dict[str, Any]:
        return self.dataset.take([sample_idx], columns=self.columns).to_pylist()[0]

    def lease_sample(self, group_id: str) -> int:
        sample_idx, remaining = self.cache.get(group_id, (None, 0))
        if sample_idx is None:
            sample_idx, remaining = self.lease(), self.samples
        remaining -= 1
        if remaining:
            self.cache[group_id] = (sample_idx, remaining)
        else:
            self.cache.pop(group_id, None)
        return sample_idx


@dataclass
class SessionRecord:
    session_id: str
    group_id: str
    sample_idx: int
    mode: str
    base_url: str
    api_key: str
    cancel_event: threading.Event = field(default_factory=threading.Event)
    status: str = "queued"
    exit_code: int | None = None
    error: str = ""
    env: Any | None = None
    agent_exit_status: str = ""
    submission: str = ""
    reward: float | None = None
    n_calls: int = 0
    future: Future | None = None
    finished_at: float | None = None

    def finish(self, info: dict[str, Any], error: str) -> None:
        if self.cancel_event.is_set():
            if error:
                self.error = error
            self.cancel()
            return
        if error:
            self.status = "failed"
            self.exit_code = 1
            self.error = error
            self.finished_at = time.time()
            return
        self.status = "completed"
        self.exit_code = 0
        self.agent_exit_status = str(info.get("exit_status", ""))
        self.submission = str(info.get("submission", ""))
        self.reward = float(info.get("reward", 0.0))
        self.n_calls = int(info["n_calls"])
        self.finished_at = time.time()

    def cancel(self) -> None:
        self.status = "cancelled"
        self.exit_code = 143
        if not self.error:
            self.error = "session cancelled"
        self.finished_at = time.time()


def _parse_log_pytest(log: str | None) -> dict[str, str]:
    if log is None or "short test summary info" not in log:
        return {}
    status_map: dict[str, str] = {}
    for line in log.split("short test summary info", 1)[1].strip().splitlines():
        if "PASSED" in line:
            status_map[".".join(line.split("::")[1:])] = "PASSED"
        elif "FAILED" in line:
            status_map[".".join(line.split("::")[1:]).split(" - ")[0]] = "FAILED"
        elif "ERROR" in line:
            status_map[".".join(line.split("::")[1:]).split(" - ")[0]] = "ERROR"
    return status_map


def _decolor_dict_keys(data: dict[str, str]) -> dict[str, str]:
    return {re.sub(r"\u001b\[\d+m", "", key): value for key, value in data.items()}


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


def _format_command_result(label: str, result: dict[str, Any]) -> str:
    output = _trim_error_output(str(result.get("output") or "").strip())
    exception_info = str(result.get("exception_info") or "").strip()
    parts = [f"{label} command failed: returncode={result.get('returncode')}"]
    if exception_info:
        parts.append(f"exception_info={exception_info}")
    if output:
        parts.append("output:\n" + output)
    return "\n".join(parts)


def _r2e_reward(row: dict[str, Any], output: str) -> float:
    parsed = _decolor_dict_keys(_parse_log_pytest(output))
    expected = _decolor_dict_keys(json.loads(row["expected_output_json"]))
    parsed = {key.split(" - ")[0]: parsed[key] for key in sorted(parsed.keys())}
    expected = {key.split(" - ")[0]: expected[key] for key in sorted(expected.keys())}
    if len(parsed) != len(expected):
        return 0.0
    for key in parsed.keys():
        if not key:
            continue
        if key not in expected or parsed[key] != expected[key]:
            return 0.0
    return 1.0


def _swebench_reward(row: dict[str, Any], output: str) -> float:
    from swebench.harness.constants import (
        APPLY_PATCH_FAIL,
        FAIL_ONLY_REPOS,
        FAIL_TO_PASS,
        KEY_INSTANCE_ID,
        MAP_REPO_VERSION_TO_SPECS,
        PASS_TO_PASS,
        RESET_FAILED,
        TESTS_ERROR,
        TESTS_TIMEOUT,
        EvalType,
        ResolvedStatus,
    )
    from swebench.harness.grading import get_eval_tests_report, get_resolution_status
    from swebench.harness.log_parsers import MAP_REPO_TO_PARSER
    from swebench.harness.test_spec.test_spec import make_test_spec

    if any(code in output for code in [APPLY_PATCH_FAIL, RESET_FAILED, TESTS_ERROR, TESTS_TIMEOUT]):
        return 0.0
    test_spec = make_test_spec(row)
    test_cmd = MAP_REPO_VERSION_TO_SPECS[test_spec.repo][test_spec.version]["test_cmd"]
    test_cmd = test_cmd[-1] if isinstance(test_cmd, list) else test_cmd
    eval_status_map = MAP_REPO_TO_PARSER[test_spec.repo](output.split(test_cmd)[-1], test_spec)
    report = get_eval_tests_report(
        eval_status_map,
        {
            KEY_INSTANCE_ID: test_spec.instance_id,
            FAIL_TO_PASS: test_spec.FAIL_TO_PASS,
            PASS_TO_PASS: test_spec.PASS_TO_PASS,
        },
        eval_type=EvalType.FAIL_ONLY if test_spec.repo in FAIL_ONLY_REPOS else EvalType.PASS_AND_FAIL,
    )
    return 1.0 if get_resolution_status(report) == ResolvedStatus.FULL.value else 0.0


class ApptainerInstanceEnvironment(SingularityEnvironment):
    def __init__(self, *, tmp_dir: Path, instance_name: str | None = None, **kwargs: Any):
        # instance_name may be supplied by the caller so it can stop a leaked
        # instance even if this constructor raises AFTER `_build_sandbox` has
        # already `instance start`ed one (in that case `env` is never bound in
        # the caller, so `env.cleanup()` would be skipped -> instance leak).
        self.instance_name = instance_name or _new_instance_name()
        self.apptainer_env = {
            **os.environ,
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

    def execute(self, action: dict, cwd: str = "", *, timeout: int | None = None) -> dict[str, Any]:
        command = action["command"]
        cmd = [self.config.executable, *self.config.global_args, "exec"]
        work_dir = cwd or self.config.cwd
        if work_dir and work_dir != "/":
            cmd.extend(["--pwd", work_dir])
        for key in self.config.forward_env:
            if (value := os.getenv(key)) is not None:
                cmd.extend(["--env", f"{key}={value}"])
        for key, value in self.config.env.items():
            cmd.extend(["--env", f"{key}={value}"])
        cmd.extend([str(self.sandbox_dir), "bash", "-c", command])
        with tempfile.TemporaryFile() as command_output:
            try:
                result = _tracked_run(
                    cmd,
                    env=self.apptainer_env,
                    timeout=timeout if timeout is not None else self.config.timeout,
                    stdout=command_output,
                    stderr=subprocess.STDOUT,
                )
                error = None
            except Exception as exc:
                error = exc
            output_size = os.fstat(command_output.fileno()).st_size
            command_output.seek(0)
            stdout = command_output.read(output_size).decode("utf-8", errors="replace")
        if error is None:
            output = {"output": stdout, "returncode": result.returncode, "exception_info": ""}
        else:
            output = {
                "output": stdout,
                "returncode": -1,
                "exception_info": f"An error occurred while executing the command: {error}",
                "extra": {"exception_type": type(error).__name__, "exception": str(error)},
            }
        self._check_finished(output)
        return output

    def _check_finished(self, output: dict) -> None:
        """Raise ``Submitted`` when the completion sentinel appears anywhere in
        the output.

        Upstream ``SingularityEnvironment._check_finished`` only inspects the
        FIRST output line. Because this sandbox merges stderr into stdout, the
        container's bash emits a ``setlocale: LC_ALL: cannot change locale``
        warning as the first line of every command, so the sentinel is never at
        ``lines[0]`` and ``Submitted`` is never raised — the agent burns all
        ``step_limit`` steps and exits ``LimitsExceeded`` with an empty
        submission. Scan every line instead and take the submission from what
        follows the sentinel line.
        """
        lines = output.get("output", "").splitlines(keepends=True)
        for idx, line in enumerate(lines):
            if line.strip() == "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" and output["returncode"] == 0:
                submission = "".join(lines[idx + 1 :])
                raise Submitted(
                    {
                        "role": "exit",
                        "content": submission,
                        "extra": {"exit_status": "Submitted", "submission": submission},
                    }
                )

    def cleanup(self) -> None:
        if self.instance_name:
            instance_name = self.instance_name
            error = _stop_apptainer_instance(self.config.executable, self.config.global_args, instance_name)
            if error:
                raise RuntimeError(f"failed to stop Apptainer instance {instance_name}: {error}")
            self.instance_name = ""


class MyLitellmModel(LitellmModel):
    def _parse_actions(self, response: Any) -> list[dict]:
        try:
            return super()._parse_actions(response)
        except FormatError as exc:
            message = response.choices[0].message.model_dump()
            message["extra"] = {"actions": []}
            raise FormatError(message, *exc.messages) from exc


class MyAgent(DefaultAgent):
    def __init__(
        self,
        model: Any,
        env: Any,
        *,
        rollout_mode: str,
        row: dict[str, Any],
        setup_command: str,
        reward_command: str,
        **kwargs: Any,
    ):
        super().__init__(model, env, **kwargs)
        self.rollout_mode = rollout_mode
        self.row = row
        self.setup_command = setup_command
        self.reward_command = reward_command

    def query(self) -> dict:
        remaining = self.config.step_limit - self.n_calls
        step_message = (
            f"Steps Remaining: {remaining}"
            if remaining > 0
            else "You have reached the maximum number of steps. Please submit your answer NOW."
        )
        self.messages[-1]["content"] += f"\n{step_message}"
        return super().query()

    def run(self, task: str = "", **kwargs: Any) -> dict[str, Any]:
        setup_result = self.env.execute({"command": self.setup_command}, timeout=SETUP_AND_REWARD_TIMEOUT_SECONDS)
        if setup_result.get("returncode") != 0 or setup_result.get("exception_info"):
            raise RuntimeError(_format_command_result("setup", setup_result))
        try:
            info = super().run(task, **kwargs)
        except ContextWindowExceededError:
            info = self.messages[-1].get("extra", {})
        # Only score trajectories the agent formally submitted. The reward
        # command judges the working tree via ``git diff``, so an agent that
        # never submits (exit_status LimitsExceeded / ContextWindowExceeded)
        # can still score 1.0 off residual edits left in the sandbox. That
        # "luck" reward is off-policy noise: GRPO discovers it can earn reward
        # WITHOUT submitting and drifts toward long, non-terminating rambling
        # until every group scores 0 and the signal collapses. Gate scoring on
        # a real submission so reward only reflects "fixed it and submitted".
        if info.get("exit_status") == "Submitted":
            if _reward_semaphore is not None:
                _reward_semaphore.acquire()
            try:
                reward_result = self.env.execute(
                    {"command": self.reward_command}, timeout=SETUP_AND_REWARD_TIMEOUT_SECONDS
                )
                if reward_result.get("exception_info"):
                    raise RuntimeError(_format_command_result("reward", reward_result))
                output = reward_result.get("output", "")
                reward_func = _r2e_reward if self.rollout_mode == "train" else _swebench_reward
                reward = reward_func(self.row, output)
            finally:
                if _reward_semaphore is not None:
                    _reward_semaphore.release()
        else:
            reward = 0.0
        return {**info, "reward": reward, "n_calls": self.n_calls}


def _pid_alive(pid: int) -> bool:
    """Best-effort liveness check for a PID (used to reclaim only DEAD servers'
    scratch, never a concurrent live server's in-flight session dirs)."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class AgentServer:
    def __init__(
        self,
        *,
        data_path: str,
        sif_dir: str,
        work_dir: str,
        train_concurrency: int,
        eval_concurrency: int,
        shuffle: bool = False,
        seed: int = 42,
    ) -> None:
        self.work_dir = Path(work_dir)
        # Session scratch must live on NODE-LOCAL storage, NOT the shared FUSE
        # work_dir. Isolate it PER SERVER INSTANCE (srv-<pid>): the previous code
        # swept a shared base with glob("*-tmp"), so if a second agent_server ever
        # overlapped this one (a relaunch racing teardown, or a co-tenant on a
        # shared node) its construction rmtree'd THIS server's live session dirs
        # mid-setup -- mass-failing uv with "No such file or directory (os error
        # 2)" across dozens of groups at once, snowballing to a hung rollout.
        # Per-pid isolation makes that cross-deletion impossible. Overridable via
        # AGENT_SERVER_TMP_DIR (the root; each server still nests srv-<pid>).
        self._tmp_root = Path(os.environ.get("AGENT_SERVER_TMP_DIR", "/data/temp/mswe_agent_tmp"))
        self.tmp_base = self._tmp_root / f"srv-{os.getpid()}"
        self.sif_dir = Path(sif_dir)
        self.base_config = recursive_merge(
            get_config_from_spec("swebench.yaml"),
            get_config_from_spec(Path(__file__).resolve().parent / "agent_config.yaml"),
        )
        self.r2e_config = self.base_config.pop("r2e")
        environment_config = self.base_config["environment"]
        self.apptainer_args = (
            str(environment_config.get("executable", "apptainer")),
            list(environment_config.get("global_args", [])),
        )
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self._tmp_root.mkdir(parents=True, exist_ok=True)
        self.tmp_base.mkdir(parents=True, exist_ok=True)
        # Startup cleanup runs under a node-shared flock so co-tenant servers
        # cannot race the same reclaim. Reclaim ONLY dead/legacy owners -- never a
        # live server's instances or in-flight scratch (the previous unscoped
        # `mswe-` sweep + glob("*-tmp") cross-killed live co-tenants and rmtree'd
        # their session dirs mid-setup, mass-failing uv with os error 2).
        with _startup_cleanup_lock(self._tmp_root / "startup_cleanup.lock"):
            _stop_mswe_instances(
                *self.apptainer_args,
                keep=lambda name: (owner := _instance_owner_pid(name)) is not None and _pid_alive(owner),
            )
            for sib in self._tmp_root.glob("srv-*"):
                if sib == self.tmp_base:
                    continue
                try:
                    sib_pid = int(sib.name.split("-", 1)[1])
                except (ValueError, IndexError):
                    sib_pid = -1
                if _pid_alive(sib_pid):
                    continue
                shutil.rmtree(sib, ignore_errors=True)
            for stale in self._tmp_root.glob("*-tmp"):
                shutil.rmtree(stale, ignore_errors=True)
        self.wheel_dir = Path(data_path) / "wheels"
        self.sifs = {path.stem: path for path in self.sif_dir.glob("*.sif")}
        self.modes = {
            mode: ModeState(
                data_path,
                mode,
                config,
                available_images=set(self.sifs),
                shuffle=shuffle and mode == "train",
                seed=seed,
            )
            for mode, config in self.r2e_config["datasets"].items()
        }
        self.records: dict[str, SessionRecord] = {}
        self.executors = {
            "train": ThreadPoolExecutor(max_workers=train_concurrency),
            "eval": ThreadPoolExecutor(max_workers=eval_concurrency),
        }
        self.lock = threading.Lock()
        self._record_ttl = float(os.environ.get("AGENT_SERVER_RECORD_TTL_SECONDS", SESSION_RECORD_TTL_SECONDS))
        threading.Thread(target=self._record_sweep_loop, name="record-sweeper", daemon=True).start()

    def create_session(self, payload: dict[str, Any]) -> dict[str, Any]:
        mode = str(payload["mode"])
        with self.lock:
            group_id = str(payload["group_id"])
            sample_idx = self.modes[mode].lease_sample(group_id)
            server_session_id = f"{payload['session_id']}-{uuid.uuid4().hex[:8]}"
            record = SessionRecord(
                session_id=str(payload["session_id"]),
                group_id=group_id,
                sample_idx=sample_idx,
                mode=mode,
                base_url=str(payload["base_url"]),
                api_key=str(payload["api_key"]),
            )
            self.records[server_session_id] = record
            record.future = self.executors[mode].submit(self._run_session, server_session_id, sample_idx)
        return {"session_id": server_session_id, "status": "queued"}

    def get_session(self, server_session_id: str) -> dict[str, Any]:
        # Do NOT pop terminal records here. Popping on the first terminal GET made
        # the endpoint non-idempotent: if that response was lost, the client's
        # retry hit 404 and (agent_client.sh gates on `curl -fsS`) looped forever.
        # Records are instead reclaimed by the background TTL sweeper, so a GET is
        # retryable across a dropped response and cancelled/unfetched records
        # (whose clients exit without a final GET) no longer leak.
        with self.lock:
            record = self.records[server_session_id]
            return self._public_record(server_session_id, record)

    def _reap_terminal_records(self) -> int:
        """Evict terminal records older than the TTL.

        Bounds ``self.records`` growth from completed sessions and from
        cancelled ones whose client exited (after POST cancel) without ever
        issuing a reclaiming GET.
        """
        cutoff = time.time() - self._record_ttl
        with self.lock:
            stale = [
                sid
                for sid, record in self.records.items()
                if record.status in TERMINAL_STATUSES
                and record.finished_at is not None
                and record.finished_at < cutoff
            ]
            for sid in stale:
                self.records.pop(sid, None)
        return len(stale)

    def _record_sweep_loop(self) -> None:
        while True:
            time.sleep(30.0)
            try:
                self._reap_terminal_records()
            except Exception as exc:  # pragma: no cover - defensive
                logging.debug("record sweeper error: %s", exc)

    def cancel_session(self, server_session_id: str) -> None:
        with self.lock:
            record = self.records[server_session_id]
            record.cancel_event.set()
            if record.status == "queued":
                record.cancel()
            future = record.future
            env = record.env
        if future is not None:
            future.cancel()
        if env is not None:
            try:
                env.cleanup()
            except Exception as exc:
                logging.warning("failed to cleanup cancelled session %s: %s", server_session_id, exc)

    def shutdown(self) -> None:
        errors = []
        with self.lock:
            records = list(self.records.values())
            for record in records:
                record.cancel_event.set()
        for record in records:
            if record.env is not None:
                try:
                    record.env.cleanup()
                except Exception as exc:
                    errors.append(str(exc))
                    logging.warning("failed to cleanup session during shutdown: %s", exc)
        for executor in self.executors.values():
            executor.shutdown(wait=False, cancel_futures=True)
        try:
            _stop_mswe_instances(
                *self.apptainer_args,
                keep=lambda name: _instance_owner_pid(name) != os.getpid(),
            )
        except Exception as exc:
            errors.append(str(exc))
            logging.warning("failed to stop stale Apptainer instances during shutdown: %s", exc)
        # Reap orphans produced by the SIGKILLs above before we exit, so they do
        # not reparent to the non-reaping PID 1 and leak as permanent zombies.
        _drain_orphans()
        if errors:
            raise RuntimeError("Agent server shutdown cleanup failed:\n" + "\n".join(errors))

    def _run_session(self, server_session_id: str, sample_idx: int) -> None:
        with self.lock:
            record = self.records.get(server_session_id)
            if record is None or record.status == "cancelled":
                return
            record.status = "running"
        row = self.modes[record.mode].get(sample_idx)
        tmp_dir = self.tmp_base / f"{server_session_id}-tmp"
        env = None
        # Generate the instance name up-front so the finally block can stop the
        # apptainer instance even if `ApptainerInstanceEnvironment(...)` raises
        # AFTER `_build_sandbox` already started one (then `env` stays None and
        # `env.cleanup()` never runs -> the instance leaks, accumulates, and
        # eventually exhausts host resources so every new `instance start` fails
        # with exit 255, collapsing whole rollout groups to zero reward).
        instance_name = _new_instance_name()
        info: dict[str, Any] = {}
        error = ""
        tmp_dir.mkdir(parents=True, exist_ok=True)
        try:
            if record.cancel_event.is_set():
                raise RuntimeError("session cancelled")
            config = copy.deepcopy(self.base_config)
            image_key = _image_key(row["docker_image"])
            config["model"].setdefault("model_kwargs", {}).update(
                {"api_base": record.base_url, "api_key": record.api_key}
            )
            environment = config["environment"]
            sif_path = self.sifs.get(image_key)
            if sif_path is None:
                raise FileNotFoundError(
                    f"Missing SIF for docker_image={row['docker_image']!r}; expected {self.sif_dir / f'{image_key}.sif'}"
                )
            environment.update({"image": str(sif_path)})
            # Pin the sandbox's temp dirs to its in-memory /tmp (provided by
            # --writable-tmpfs). `uv pip install` writes atomic .tmpXXXX scratch
            # to $TMPDIR; leaving it unset let uv's temp resolve to the host
            # session scratch under production concurrency, so any cleanup of that
            # host dir surfaced as "No such file or directory (os error 2)"
            # mid-setup. Forcing /tmp decouples uv from host scratch entirely.
            environment.setdefault("env", {}).update({"TMPDIR": "/tmp", "UV_CACHE_DIR": "/tmp/uv-cache"})
            environment.setdefault("exec_args", []).extend(
                ["--bind", f"{self.wheel_dir}:{self.r2e_config['wheel_mount']}:ro"]
            )
            environment.pop("environment_class", None)
            model = MyLitellmModel(**config["model"])
            env = ApptainerInstanceEnvironment(tmp_dir=tmp_dir, instance_name=instance_name, **config["environment"])
            # Bind env FIRST (so a concurrent cancel_session can cleanup()), then
            # re-check cancel under the lock. A cancel arriving DURING the
            # constructor above (which `instance start`s a sandbox, seconds long)
            # saw record.env is None -> skipped cleanup, and future.cancel() no-ops
            # once running, so without this the agent would run the full 7200s
            # holding a sandbox + concurrency slot after being cancelled.
            with self.lock:
                record.env = env
                cancelled = record.cancel_event.is_set()
            if cancelled:
                raise RuntimeError("session cancelled")
            info = MyAgent(
                model,
                env,
                rollout_mode=record.mode,
                row=row,
                setup_command=self.r2e_config["setup_commands"][record.mode],
                reward_command=self.r2e_config["reward_commands"][record.mode],
                **config["agent"],
            ).run(str(row["problem_statement"]))
        except subprocess.CalledProcessError as exc:
            error = _format_called_process_error(exc)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        finally:
            cleanup_error = ""
            if env is not None:
                try:
                    env.cleanup()
                except Exception as exc:
                    cleanup_error = f"{type(exc).__name__}: {exc}"
            else:
                # env never got bound (constructor raised). If `_build_sandbox`
                # had already started the instance, `env.cleanup()` above was
                # skipped -- stop it explicitly by name so it does not leak.
                # Idempotent: a no-op if no instance with this name exists.
                try:
                    stop_error = _stop_apptainer_instance(*self.apptainer_args, instance_name)
                    if stop_error:
                        cleanup_error = (
                            f"RuntimeError: failed to stop Apptainer instance {instance_name}: {stop_error}"
                        )
                except Exception as exc:  # pragma: no cover - best-effort safety net
                    cleanup_error = f"{type(exc).__name__}: {exc}"
            # Stop the instance BEFORE removing tmp_dir: tmp_dir is the instance's
            # TMPDIR/APPTAINER_TMPDIR, so deleting it under a still-running
            # instance causes ENOENT on its temp files.
            if cleanup_error:
                logging.warning(
                    "session cleanup failed server_session_id=%s error=%s", server_session_id, cleanup_error
                )
                error = f"{error}\ncleanup failed: {cleanup_error}".strip()
            else:
                shutil.rmtree(tmp_dir, ignore_errors=True)

        with self.lock:
            record = self.records[server_session_id]
            record.env = None
            record.finish(info, error)
            if record.status in {"failed", "cancelled"}:
                event = {
                    "time": time.time(),
                    "session_id": server_session_id,
                    "client_session_id": record.session_id,
                    "group_id": record.group_id,
                    "mode": record.mode,
                    "sample_idx": record.sample_idx,
                    "docker_image": row.get("docker_image"),
                    "status": record.status,
                    "exit_code": record.exit_code,
                    "error": record.error,
                }
                with (self.work_dir / "agent_server_events.jsonl").open("a") as file:
                    file.write(json.dumps(event, ensure_ascii=False) + "\n")

    def _public_record(self, server_session_id: str, record: SessionRecord) -> dict[str, Any]:
        data: dict[str, Any] = {
            "session_id": server_session_id,
            "client_session_id": record.session_id,
            "status": record.status,
            "exit_code": record.exit_code,
            "error": record.error,
        }
        if record.status != "completed":
            return data
        data.update(
            {
                "agent_exit_status": record.agent_exit_status,
                "submission": record.submission,
                "reward": record.reward,
                "n_calls": record.n_calls,
            }
        )
        return data


def main() -> None:
    _install_child_subreaper_and_reaper()
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--train-concurrency", type=int, default=int(os.environ.get("AGENT_SERVER_TRAIN_CONCURRENCY", "128"))
    )
    parser.add_argument(
        "--eval-concurrency", type=int, default=int(os.environ.get("AGENT_SERVER_EVAL_CONCURRENCY", "64"))
    )
    parser.add_argument("--shuffle", action="store_true", default=os.environ.get("AGENT_SERVER_SHUFFLE", "0") == "1")
    parser.add_argument("--seed", type=int, default=int(os.environ.get("AGENT_SERVER_SEED", "42")))
    args = parser.parse_args()
    server = AgentServer(
        data_path=os.environ["R2E_DATA_PATH"],
        sif_dir=os.environ["R2E_SIF_DIR"],
        work_dir=os.environ["AGENT_SERVER_WORK_DIR"],
        train_concurrency=args.train_concurrency,
        eval_concurrency=args.eval_concurrency,
        shuffle=args.shuffle,
        seed=args.seed,
    )
    logging.getLogger("werkzeug").setLevel(logging.ERROR)

    shutting_down = threading.Event()

    def stop(signum, _frame):
        # Ignore repeat signals: a second SIGTERM must not restart shutdown or
        # interrupt the orphan-draining pass midway (which would leak zombies).
        if shutting_down.is_set():
            return
        shutting_down.set()
        try:
            server.shutdown()  # stops instances then drains orphans synchronously
        except Exception as exc:
            logging.warning("error during shutdown: %s", exc)
            _drain_orphans()  # best-effort reap even if instance-stop raised
        os._exit(128 + signum)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    app = Flask(__name__)

    @app.post("/sessions")
    def create_session():
        try:
            return jsonify(server.create_session(request.get_json(force=True)))
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400

    @app.get("/sessions/<server_session_id>")
    def get_session(server_session_id: str):
        try:
            return jsonify(server.get_session(server_session_id))
        except KeyError as exc:
            return jsonify({"error": str(exc)}), 404

    @app.post("/sessions/<server_session_id>/cancel")
    def cancel_session(server_session_id: str):
        try:
            server.cancel_session(server_session_id)
            return "", 204
        except KeyError as exc:
            return jsonify({"error": str(exc)}), 404

    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
