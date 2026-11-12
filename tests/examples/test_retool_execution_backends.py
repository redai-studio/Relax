# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""CPU ownership/selection tests; optional new Linux sandbox integrations."""

import asyncio
import errno
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from examples.retool import execution_backend, execution_service, landlock_sandbox, sandbox


def result(policy="test", **changes):
    value = dict(
        returncode=0,
        timed_out=False,
        guard=dict(
            profile=policy,
            uid=65534,
            gid=65534,
            cap_eff="0000000000000000",
            no_new_privs=1,
            seccomp_mode=2,
            process_creation_allowed=False,
        ),
        stderr="",
    )
    value.update(changes)
    return value


class TestBackendSelection(unittest.TestCase):
    def backend(self, name, runner):
        return execution_backend.Backend(name, "test", runner, mock.Mock(), {})

    def test_auto_probes_bootstrap_then_freezes_runner(self):
        first = self.backend("landlock", mock.Mock(side_effect=RuntimeError("Landlock ENOSYS")))
        runner = mock.Mock(return_value=result())
        second = self.backend("apptainer", runner)
        with mock.patch.object(execution_backend, "_build", side_effect=[first, second]) as build:
            selected = execution_backend.select_backend(Path("."), "auto")
            self.assertEqual(selected.name, "apptainer")
            self.assertEqual(build.call_count, 2)
            first.close.assert_called_once()
            self.assertFalse(selected.startup["student_code_executed"])
            runner.assert_called_once_with("", Path("."), timeout=10, cpu_seconds=3, memory_mb=1024, probe_only=True)
            runner.side_effect = RuntimeError("payload infrastructure fault")
            with self.assertRaisesRegex(RuntimeError, "payload infrastructure"):
                selected.run_isolated("new student program", Path("."))
            self.assertEqual(build.call_count, 2, "Payload failure must never build a fallback")

    def test_explicit_backend_failure_does_not_fallback(self):
        selected = self.backend("landlock", mock.Mock(return_value=result(guard=None, stderr="ENOSYS")))
        with mock.patch.object(execution_backend, "_build", return_value=selected) as build:
            with self.assertRaisesRegex(RuntimeError, "ENOSYS"):
                execution_backend.select_backend(Path("."), "landlock")
            build.assert_called_once_with("landlock")
            selected.close.assert_called_once()

    def test_invalid_choice_rejected_before_building(self):
        with mock.patch.object(execution_backend, "_build") as build:
            with self.assertRaises(ValueError):
                execution_backend.select_backend(Path("."), "unsafe")
            build.assert_not_called()

    def test_guard_failure_retains_bounded_diagnostic(self):
        with self.assertRaisesRegex(RuntimeError, "wrapper_returncode.*255"):
            execution_backend.verify_guard(result(guard=None, wrapper_returncode=255, stderr="mount failed"), "test")
        for field, bad in [("uid", 0), ("gid", 0), ("cap_eff", "1"), ("seccomp_mode", 0), ("no_new_privs", 0)]:
            value = result()
            value["guard"][field] = bad
            with self.subTest(field=field), self.assertRaises(RuntimeError):
                execution_backend.verify_guard(value, "test")


class TestGuardProcessOwnership(unittest.TestCase):
    def test_container_child_restores_thread_limits_after_cleanenv(self):
        from examples.retool import apptainer_sandbox

        observed = {}

        def child_main(**kwargs):
            observed.update(os.environ)

        base = SimpleNamespace(child_main=child_main)
        spec = SimpleNamespace(loader=SimpleNamespace(exec_module=lambda module: None))
        with mock.patch.dict(os.environ, {}, clear=True):
            with mock.patch.object(sys, "argv", ["child.py", "--supervise", "19"]):
                with mock.patch.object(apptainer_sandbox.os, "fork", return_value=0):
                    with mock.patch.object(
                        apptainer_sandbox.importlib.util, "spec_from_file_location", return_value=spec
                    ):
                        with mock.patch.object(
                            apptainer_sandbox.importlib.util, "module_from_spec", return_value=base
                        ):
                            apptainer_sandbox.supervise()
        self.assertEqual(observed["RETOOL_GUARD_FD"], "19")
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            self.assertEqual(observed[name], "1")

    def test_supervisor_preserves_cpython_systemexit_semantics(self):
        # Fresh synthetic payloads exercise the actual fork/wait wrapper only;
        # Linux integration separately verifies the real isolation bootstrap.
        harness = """
import os, sys, types
from unittest import mock
from examples.retool import apptainer_sandbox as sandbox
fd = int(sys.argv[2])
source = sys.argv[3]
def child_main(**kwargs):
    os.close(fd)
    exec(source)
base = types.SimpleNamespace(child_main=child_main)
with mock.patch.object(sandbox.importlib.util, 'spec_from_file_location', return_value=None):
    with mock.patch.object(sandbox.importlib.util, 'module_from_spec', return_value=base):
        base_spec = types.SimpleNamespace(loader=types.SimpleNamespace(exec_module=lambda module: None))
        sandbox.importlib.util.spec_from_file_location.return_value = base_spec
        sandbox.supervise()
"""
        for code in ("None", "False", "True", "'invalid hypothesis'"):
            source = f"print('fresh output', end=''); raise SystemExit({code})"
            oracle = subprocess.run([sys.executable, "-c", source], capture_output=True, text=True, check=False)
            reader, writer = os.pipe()
            try:
                wrapped = subprocess.run(
                    [sys.executable, "-c", harness, "--supervise", str(writer), source],
                    pass_fds=(writer,),
                    capture_output=True,
                    text=True,
                    check=False,
                )
                os.close(writer)
                writer = None
                records = os.read(reader, 4096).decode().splitlines()
                with self.subTest(exit_code=code):
                    self.assertEqual(wrapped.returncode, 0)
                    self.assertEqual((wrapped.stdout, wrapped.stderr), (oracle.stdout, oracle.stderr))
                    self.assertEqual(json.loads(records[-1])["returncode"], oracle.returncode)
            finally:
                os.close(reader)
                if writer is not None:
                    os.close(writer)

    def test_inner_signal_is_not_apptainer_exit_encoding(self):
        raw = json.dumps({"profile": "test"}).encode() + b'\n{"event":"process_exit","returncode":-24}\n'
        self.assertEqual(
            landlock_sandbox.parse_guard_stream(raw, require_exit_record=True), ({"profile": "test"}, -24)
        )
        raw = b'{"profile":"test"}\n{"event":"process_exit","returncode":137}\n'
        self.assertEqual(landlock_sandbox.parse_guard_stream(raw, require_exit_record=True)[1], 137)
        with self.assertRaisesRegex(ValueError, "Missing trusted"):
            landlock_sandbox.parse_guard_stream(b'{"profile":"test"}\n', require_exit_record=True)

    def test_service_classifies_exact_child_exit_and_preserves_infrastructure_signals(self):
        self.assertEqual(execution_service.execution_status(result(returncode=0)), "Finished")
        self.assertEqual(execution_service.execution_status(result(returncode=137)), "Error")
        self.assertEqual(execution_service.execution_status(result(returncode=-signal.SIGXCPU)), "TimeLimitExceeded")
        self.assertEqual(
            execution_service.execution_status(result(returncode=-signal.SIGKILL, timed_out=True)), "TimeLimitExceeded"
        )
        for signum in (
            signal.SIGSEGV,
            signal.SIGBUS,
            signal.SIGILL,
            signal.SIGFPE,
            signal.SIGABRT,
            signal.SIGXFSZ,
            signal.SIGTRAP,
        ):
            with self.subTest(signum=signum):
                self.assertEqual(execution_service.execution_status(result(returncode=-signum)), "Error")
        for signum in (signal.SIGKILL, signal.SIGTERM, signal.SIGINT):
            with (
                self.subTest(signum=signum),
                self.assertRaisesRegex(RuntimeError, f"Unexpected isolated process signal {signum}"),
            ):
                execution_service.execution_status(result(returncode=-signum))
        parsed = sandbox.parse_result(dict(status="Success", run_result=dict(status="Error")))
        self.assertEqual(parsed["status"], "code_error")

    def test_apptainer_owned_image_remains_traversable_under_umask077(self):
        from examples.retool.apptainer_sandbox import Runtime

        with tempfile.TemporaryDirectory() as folder:
            runtime = object.__new__(Runtime)
            runtime.image = Path(folder) / "image"
            runtime.image.mkdir(mode=0o755)
            original = os.umask(0o077)
            try:
                runtime._directory(runtime.image / "opt/runtime/lib")
            finally:
                os.umask(original)
            for name in ("opt", "opt/runtime", "opt/runtime/lib"):
                self.assertEqual((runtime.image / name).stat().st_mode & 0o777, 0o755)
            self.assertEqual(Path(folder).stat().st_mode & 0o777, 0o700)

    def test_popen_failure_closes_both_pipe_fds(self):
        original_pipe = os.pipe
        opened = []

        def pipe():
            pair = original_pipe()
            opened.extend(pair)
            return pair

        with tempfile.TemporaryDirectory() as folder:
            with mock.patch.object(landlock_sandbox.os, "pipe", side_effect=pipe):
                with mock.patch.object(landlock_sandbox.subprocess, "Popen", side_effect=OSError("start failed")):
                    with self.assertRaisesRegex(OSError, "start failed"):
                        landlock_sandbox.run_guarded_process(
                            lambda fd: ["not-started"],
                            {},
                            {},
                            Path(folder),
                            timeout=1,
                            policy="test",
                            started=time.monotonic(),
                        )
        for fd in opened:
            with self.assertRaises(OSError) as error:
                os.fstat(fd)
            self.assertEqual(error.exception.errno, errno.EBADF)

    def test_communicate_failure_kills_owned_group_and_waits(self):
        process = mock.Mock(pid=1234567)
        process.communicate.side_effect = RuntimeError("transport failure")
        process.poll.return_value = None
        with tempfile.TemporaryDirectory() as folder:
            with mock.patch.object(landlock_sandbox.subprocess, "Popen", return_value=process):
                with mock.patch.object(landlock_sandbox.os, "killpg") as kill:
                    with self.assertRaisesRegex(RuntimeError, "transport failure"):
                        landlock_sandbox.run_guarded_process(
                            lambda fd: ["not-started"],
                            {},
                            {},
                            Path(folder),
                            timeout=1,
                            policy="test",
                            started=time.monotonic(),
                        )
                    kill.assert_called_once_with(process.pid, signal.SIGKILL)
                    process.wait.assert_called_once()
                    process.stdin.close.assert_called_once()


class TestSandboxHTTPTransport(unittest.TestCase):
    def test_queue_deadline_is_not_the_execution_budget_and_never_retries(self):
        opener = mock.Mock()
        opener.open.side_effect = TimeoutError("new fixture timeout")
        with mock.patch.object(sandbox.urllib.request, "build_opener", return_value=opener):
            with self.assertRaisesRegex(RuntimeError, "completion is unknown.*not retried") as error:
                sandbox.SandboxClient("http://sandbox.invalid/run_code")._request("print(29 * 31)")
        self.assertIsInstance(error.exception.__cause__, TimeoutError)
        opener.open.assert_called_once()
        request = opener.open.call_args.args[0]
        payload = json.loads(request.data)
        self.assertEqual(opener.open.call_args.kwargs["timeout"], 300)
        self.assertEqual((payload["run_timeout"], payload["memory_limit_MB"]), (30, 1024))

    def test_shared_disk_persistence_does_not_block_http_event_loop(self):
        started, completed = threading.Event(), threading.Event()

        def slow_put(*args):
            started.set()
            time.sleep(0.06)
            completed.set()

        async def run():
            ticks = 0

            async def observe():
                nonlocal ticks
                while not completed.is_set():
                    if started.is_set():
                        ticks += 1
                    await asyncio.sleep(0.005)

            await asyncio.gather(execution_service.persist(Path("unused"), {}), observe())
            return ticks

        with mock.patch.object(execution_service, "put", side_effect=slow_put):
            self.assertGreater(asyncio.run(run()), 0)


@unittest.skipUnless(
    os.environ.get("RETOOL_SANDBOX_INTEGRATION") == "1" and sys.platform == "linux" and os.getuid() == 0,
    "Requires explicitly enabled fresh Linux root sandbox integration; never run historical programs",
)
class TestRealSandboxBackends(unittest.TestCase):
    def test_new_apptainer_compute_and_exact_signal_status(self):
        from examples.retool.apptainer_sandbox import POLICY, Runtime

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            runtime = Runtime()
            try:
                value = runtime.run_isolated("print(13 * 17)", root, timeout=10)
                execution_backend.verify_guard(value, POLICY)
                self.assertEqual((value["returncode"], value["stdout"]), (0, "221\n"))
                cpu = runtime.run_isolated("while True: pass", root, timeout=10, cpu_seconds=1)
                execution_backend.verify_guard(cpu, POLICY)
                self.assertEqual(cpu["returncode"], -signal.SIGXCPU)
                crash = runtime.run_isolated("import ctypes; ctypes.string_at(0)", root, timeout=10)
                execution_backend.verify_guard(crash, POLICY)
                self.assertEqual(crash["returncode"], -signal.SIGSEGV)
                explicit = runtime.run_isolated("raise SystemExit(137)", root, timeout=10)
                self.assertEqual(explicit["returncode"], 137)
                wall = runtime.run_isolated("import time; time.sleep(10)", root, timeout=3)
                execution_backend.verify_guard(wall, POLICY)
                self.assertTrue(wall["timed_out"])
            finally:
                runtime.close()

    def test_new_landlock_compute(self):
        with tempfile.TemporaryDirectory() as folder:
            value = landlock_sandbox.run_isolated("print(19 * 23)", Path(folder), timeout=10)
            execution_backend.verify_guard(value, landlock_sandbox.POLICY)
            self.assertEqual((value["returncode"], value["stdout"]), (0, "437\n"))


if __name__ == "__main__":
    unittest.main()
