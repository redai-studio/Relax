# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Exercise the real Action processes with synthetic nvidia-smi inventory."""

from __future__ import annotations

import fcntl
import hashlib
import os
import shutil
import signal
import subprocess
import tempfile
import time
import unittest
from pathlib import Path


ACTION = Path(__file__).resolve().parents[2] / ".github/actions/acquire-devices"
UUIDS = [f"GPU-00000000-0000-0000-0000-{index:012d}" for index in range(3)]


@unittest.skipUnless(shutil.which("node"), "Node.js is required to exercise Action main/post entrypoints")
class AcquireDevicesTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.locks = self.root / "locks"
        self.bin = self.root / "bin"
        self.bin.mkdir()
        self.inventory = "\n".join(f"{index}, {uuid}, Disabled" for index, uuid in enumerate(UUIDS))
        self.set_inventory(self.inventory)
        self.environments: list[dict[str, str]] = []
        self.processes: list[subprocess.Popen] = []

    def tearDown(self) -> None:
        for process in self.processes:
            if process.poll() is None:
                process.terminate()
            process.communicate(timeout=20)
        for env in self.environments:
            self.post(env)
        self.temporary.cleanup()

    def set_inventory(self, inventory: str, exit_code: int = 0) -> None:
        executable = self.bin / "nvidia-smi"
        executable.write_text(f"#!/bin/sh\ncat <<'INVENTORY'\n{inventory}\nINVENTORY\nexit {exit_code}\n")
        executable.chmod(0o755)

    def environment(self, **inputs: str) -> dict[str, str]:
        directory = self.root / f"action-{len(self.environments)}"
        directory.mkdir()
        env = {key: value for key, value in os.environ.items() if not key.startswith(("INPUT_", "STATE_"))}
        env.pop("CUDA_VISIBLE_DEVICES", None)
        env.update(
            PATH=f"{self.bin}{os.pathsep}{env['PATH']}",
            RUNNER_OS="Linux",
            RUNNER_TEMP=str(directory),
            GITHUB_STATE=str(directory / "state"),
            GITHUB_OUTPUT=str(directory / "output"),
            GITHUB_ENV=str(directory / "env"),
        )
        for name in ("GITHUB_STATE", "GITHUB_OUTPUT", "GITHUB_ENV"):
            Path(env[name]).touch()
        env["INPUT_LOCK-DIR"] = str(self.locks)
        env["INPUT_TIMEOUT"] = "2"
        for name, value in inputs.items():
            env[f"INPUT_{name.upper()}"] = value
        self.environments.append(env)
        return env

    def start(self, env: dict[str, str]) -> subprocess.Popen:
        process = subprocess.Popen(
            ["node", str(ACTION / "main.js")], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        self.processes.append(process)
        return process

    def run_main(self, env: dict[str, str], success: bool = True) -> str:
        process = self.start(env)
        stdout, stderr = process.communicate(timeout=20)
        if success:
            self.assertEqual(process.returncode, 0, stdout + stderr)
        else:
            self.assertNotEqual(process.returncode, 0, stdout + stderr)
        return stdout + stderr

    def post(self, env: dict[str, str]) -> None:
        state = self.commands(env["GITHUB_STATE"])
        result = subprocess.run(
            ["node", str(ACTION / "post.js")],
            env={**env, **{f"STATE_{key}": value for key, value in state.items()}},
            capture_output=True,
            text=True,
            timeout=20,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def commands(self, filename: str) -> dict[str, str]:
        return dict(line.split("=", 1) for line in Path(filename).read_text().splitlines())

    def selected(self, env: dict[str, str]) -> set[str]:
        return set(self.commands(env["GITHUB_OUTPUT"])["devices"].split(","))

    def wait_for(self, predicate) -> None:
        deadline = time.monotonic() + 10
        while not predicate():
            self.assertLess(time.monotonic(), deadline, "Timed out waiting for process state")
            time.sleep(0.02)

    def control_directory(self, env: dict[str, str]) -> Path:
        self.wait_for(lambda: "lease_directory" in self.commands(env["GITHUB_STATE"]))
        directory = Path(self.commands(env["GITHUB_STATE"])["lease_directory"])
        self.wait_for(lambda: (directory / "pid").exists())
        return directory

    def test_acquire_devices_holds_across_steps_and_releases_in_post(self) -> None:
        first = self.environment(count="2")
        self.run_main(first)
        self.assertEqual(self.selected(first), set(UUIDS[:2]))
        self.assertEqual(self.commands(first["GITHUB_ENV"])["CUDA_VISIBLE_DEVICES"], ",".join(UUIDS[:2]))
        second = self.environment(count="2", timeout="0")
        self.assertIn("Timed out", self.run_main(second, success=False))
        self.post(first)
        self.post(first)
        self.run_main(self.environment(count="3", timeout="0"))

    def test_acquire_devices_concurrent_jobs_never_overlap(self) -> None:
        environments = [self.environment() for _ in UUIDS]
        processes = [self.start(env) for env in environments]
        for process in processes:
            stdout, stderr = process.communicate(timeout=20)
            self.assertEqual(process.returncode, 0, stdout + stderr)
        selections = [self.selected(env) for env in environments]
        self.assertEqual(set.union(*selections), set(UUIDS))
        self.assertEqual(sum(map(len, selections)), len(UUIDS))

    def test_acquire_devices_partial_allocation_is_rolled_back_while_waiting(self) -> None:
        self.run_main(self.environment(count="2", devices="1,2"))
        waiting = self.environment(count="2", devices="0,1", timeout="1")
        process = self.start(waiting)
        self.control_directory(waiting)
        third = self.environment(devices="0", timeout="1")
        self.run_main(third)
        stdout, stderr = process.communicate(timeout=20)
        self.assertNotEqual(process.returncode, 0, stdout + stderr)
        self.assertIn("Timed out", stderr)

    def test_acquire_devices_index_and_uuid_share_one_lock(self) -> None:
        self.run_main(self.environment(devices="1"))
        error = self.run_main(self.environment(devices=UUIDS[1], timeout="0"), success=False)
        self.assertIn("Timed out", error)

    def test_acquire_devices_waiter_proceeds_after_release(self) -> None:
        first = self.environment(count="3")
        self.run_main(first)
        waiting = self.environment(count="3")
        process = self.start(waiting)
        self.control_directory(waiting)
        self.post(first)
        stdout, stderr = process.communicate(timeout=20)
        self.assertEqual(process.returncode, 0, stdout + stderr)
        self.assertEqual(self.selected(waiting), set(UUIDS))

    def test_acquire_devices_cancelled_acquisition_stops_holder(self) -> None:
        first = self.environment(count="3")
        self.run_main(first)
        for signum in (signal.SIGTERM, signal.SIGKILL):
            with self.subTest(signal=signum):
                env = self.environment(count="3", timeout="60")
                process = self.start(env)
                directory = self.control_directory(env)
                process.send_signal(signum)
                process.communicate(timeout=20)
                self.wait_for(lambda: (directory / "released").exists())
                self.post(env)
        self.post(first)
        self.run_main(self.environment(count="3", timeout="0"))

    def test_acquire_devices_lock_files_keep_the_same_inode(self) -> None:
        env = self.environment(devices="0")
        self.run_main(env)
        digest = hashlib.sha256(f"nvidia\0{UUIDS[0]}".encode()).hexdigest()
        lockfile = self.locks / f"{digest}.lock"
        with lockfile.open("a+") as handle:
            with self.assertRaises(BlockingIOError):
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            inode = lockfile.stat().st_ino
            self.post(env)
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.assertEqual(lockfile.stat().st_ino, inode)

    def test_acquire_devices_holder_termination_releases_kernel_locks(self) -> None:
        env = self.environment(count="3")
        self.run_main(env)
        directory = self.control_directory(env)
        os.kill(int((directory / "pid").read_text()), signal.SIGTERM)
        self.wait_for(lambda: (directory / "released").exists())
        self.run_main(self.environment(count="3", timeout="0"))

    def test_acquire_devices_failed_environment_export_releases_allocation(self) -> None:
        env = self.environment(count="3")
        env["GITHUB_ENV"] = str(self.root)
        self.run_main(env, success=False)
        self.post(env)
        self.run_main(self.environment(count="3", timeout="0"))

    def test_acquire_devices_invalid_requests_fail_without_leases(self) -> None:
        cases = [
            {"count": "0"},
            {"count": "4"},
            {"count": "1.5"},
            {"timeout": "-1"},
            {"backend": "npu"},
            {"devices": "0,0"},
            {"devices": f"0,{UUIDS[0]}"},
            {"devices": "99"},
            {"devices": "0,"},
            {"devices": "0\nINJECTED=1"},
            {"lock-dir": "relative"},
        ]
        for inputs in cases:
            with self.subTest(inputs=inputs):
                env = self.environment(**inputs)
                self.run_main(env, success=False)
                self.assertEqual(Path(env["GITHUB_OUTPUT"]).read_text(), "")
                self.post(env)
        self.run_main(self.environment(count="3", timeout="0"))

    def test_acquire_devices_rejects_mig_and_discovery_failures(self) -> None:
        for inventory, code in [(self.inventory.replace("Disabled", "Enabled"), 0), ("", 0), ("", 1)]:
            with self.subTest(inventory=inventory, code=code):
                self.set_inventory(inventory, code)
                self.run_main(self.environment(), success=False)

    def test_acquire_devices_requires_explicit_pool_when_visibility_is_preset(self) -> None:
        env = self.environment()
        env["CUDA_VISIBLE_DEVICES"] = "0"
        self.assertIn("specify devices", self.run_main(env, success=False))
        explicit = self.environment(devices="0")
        explicit["CUDA_VISIBLE_DEVICES"] = "0"
        self.run_main(explicit)
        self.assertEqual(self.selected(explicit), {UUIDS[0]})


if __name__ == "__main__":
    unittest.main()
