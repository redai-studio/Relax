# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Private read-only Apptainer runtime with the shared ReTool seccomp guard."""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


if __name__ != "__main__":
    from examples.retool import landlock_sandbox as guard

POLICY = "PURE_COMPUTE_APPTAINER_SECCOMP_V1"
CHILD = Path("/opt/retool_guard/child.py")
BASE = Path("/opt/retool_guard/base.py")


class Runtime:
    """Own a new private image, never bind a host root or an experiment
    tree."""

    def __init__(self) -> None:
        if sys.platform != "linux" or os.getuid() != 0:
            raise RuntimeError("Apptainer isolation requires a Linux root service dropping payload uid to 65534")
        self.executable = shutil.which("apptainer")
        if not self.executable:
            raise RuntimeError("Apptainer executable is not installed")
        self.directory = tempfile.TemporaryDirectory(prefix="retool_apptainer_")
        self.image = Path(self.directory.name) / "image"
        self.image.mkdir(mode=0o755)
        self.image.chmod(0o755)
        self.binds: list[tuple[Path, Path]] = []
        try:
            for name in ("dev", "proc", "sys", "tmp", "var/tmp", "root", "work"):
                self._directory(self.image / name)
            # Keep the executable's lexical venv path AND its actual target.
            python = Path(sys.executable).absolute()
            runtime = [*guard.read_roots(), python, python.resolve(), Path("/bin/sh"), Path("/bin/sh").resolve()]
            runtime += [Path("/lib"), Path("/lib64"), Path("/etc/ld.so.cache")]
            config = Path(sys.prefix) / "pyvenv.cfg"
            if config.is_file():
                runtime.append(config)
            for path in dict.fromkeys(runtime):
                if path.exists():
                    self._bind(path, path)
            self._bind(Path(__file__).resolve(), CHILD)
            self._bind(Path(guard.__file__).resolve(), BASE)
            self.python = str(python)
        except BaseException:
            self.close()
            raise

    def _bind(self, source: Path, destination: Path) -> None:
        if not destination.is_absolute() or destination == Path("/") or ".." in destination.parts:
            raise ValueError("Refusing an unsafe container bind destination")
        if any(char in str(path) for path in (source, destination) for char in (",", ":", "\n")):
            raise ValueError("Runtime paths cannot contain Apptainer bind separators")
        info = source.stat()
        if info.st_uid != 0 or info.st_mode & 0o022:
            raise RuntimeError(f"Runtime bind must be root-owned and not group/world writable: {source}")
        target = self.image / destination.relative_to("/")
        if source.is_dir():
            self._directory(target)
        else:
            self._directory(target.parent)
            target.touch(mode=0o644)
        if (source, destination) not in self.binds:
            self.binds.append((source, destination))

    def _directory(self, path: Path) -> None:
        current = self.image
        for part in path.relative_to(self.image).parts:
            current /= part
            current.mkdir(exist_ok=True, mode=0o755)
            current.chmod(0o755)  # Only newly owned image leaves; outer remains private0700.

    def close(self) -> None:
        self.directory.cleanup()

    def run_isolated(
        self,
        code: str,
        root: Path,
        *,
        timeout: float = 8,
        memory_mb: int = 1024,
        cpu_seconds: int = 3,
        probe_only: bool = False,
    ) -> dict[str, Any]:
        if not isinstance(code, str) or len(code.encode()) > 65536:
            raise ValueError("Expected at most 64 KiB of code")
        if not 0 < timeout <= 40 or not 1 <= cpu_seconds <= 30 or not 128 <= memory_mb <= 2048:
            raise ValueError("Invalid resource limit")
        root = Path(root).resolve(strict=True)
        started = time.monotonic()
        namespaces = {name: os.readlink(f"/proc/self/ns/{name}") for name in ("mnt", "net", "pid")}
        with tempfile.TemporaryDirectory(prefix="appt_request_", dir=root) as directory:
            outer = Path(directory)
            outer.chmod(0o711)
            work = outer / "work"
            work.mkdir(mode=0o700)
            os.chown(work, 65534, 65534)
            scratch = work / "scratch.txt"
            scratch.touch(mode=0o600)
            os.chown(scratch, 65534, 65534)
            binds = [f"{source}:{destination}:ro" for source, destination in self.binds]
            binds += [f"{work}:/work:ro", f"{scratch}:/work/scratch.txt:rw"]

            def command(fd: int) -> list[str]:
                return [
                    self.executable,
                    "exec",
                    "--containall",
                    "--no-home",
                    "--cleanenv",
                    "--no-eval",
                    "--no-mount",
                    "cwd,hostfs,bind-paths,tmp,dev,sys",
                    "--net",
                    "--network",
                    "none",
                    "--pwd",
                    "/work",
                    "--bind",
                    ",".join(binds),
                    str(self.image),
                    self.python,
                    "-I",
                    str(CHILD),
                    "--supervise",
                    str(fd),
                ]

            env = dict(
                PATH=os.defpath,
                HOME="/work",
                TMPDIR="/work",
                CUDA_VISIBLE_DEVICES="",
                PYTHONDONTWRITEBYTECODE="1",
                PYTHONIOENCODING="utf-8",
                OMP_NUM_THREADS="1",
                MKL_NUM_THREADS="1",
                OPENBLAS_NUM_THREADS="1",
            )
            request = dict(
                code=code, work="/work", memory_mb=memory_mb, cpu_seconds=cpu_seconds, probe_only=probe_only
            )
            result = guard.run_guarded_process(
                command,
                request,
                env,
                root,
                timeout=timeout,
                policy=POLICY,
                started=started,
                require_exit_record=True,
            )
            proof = result.get("guard")
            if proof and any(
                not proof.get("namespaces", {}).get(name) or proof["namespaces"][name] == value
                for name, value in namespaces.items()
            ):
                result["guard_error"] = "Apptainer failed to isolate mount/network/PID namespaces"
            return result


def supervise() -> None:
    """Report the exact inner wait status, not Apptainer's 128+signal code."""
    spec = importlib.util.spec_from_file_location("guard_base", BASE)
    base = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base)
    fd = int(sys.argv[2])
    # The payload never inherits a writer which remains open after its guard.
    # This root-owned supervisor cannot be signalled by the uid65534 payload.
    pid = os.fork()
    if pid == 0:
        os.environ["RETOOL_GUARD_FD"] = str(fd)
        # --cleanenv intentionally drops host thread limits. Restore them
        # inside the container before numerical libraries can be imported.
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ[name] = "1"

        def filesystem_guard(*unused: Any) -> dict[str, Any]:
            return dict(
                filesystem="private Apptainer root; readonly runtime/work; scratch-only writes",
                namespaces={name: os.readlink(f"/proc/self/ns/{name}") for name in ("mnt", "net", "pid")},
                Landlock_used=False,
            )

        # Preserve CPython exception, SystemExit, atexit and stream-flush
        # semantics. Only the separate trusted parent interprets wait status.
        base.child_main(filesystem_guard=filesystem_guard, policy=POLICY)
        return
    _, status = os.waitpid(pid, 0)
    record = dict(event="process_exit", returncode=os.waitstatus_to_exitcode(status))
    os.write(fd, json.dumps(record).encode() + b"\n")
    os.close(fd)


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "--supervise":
        raise SystemExit("Use the owned execution service")
    supervise()
