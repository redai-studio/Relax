# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Experimental Linux pure-computation runner: Landlock + seccomp + rlimits.

Adds restrictions to the existing platform profile; does not change host
policy, mount filesystems, or claim VM/container isolation. Only trusted
validation code may be used until the deployment-specific verification has been
accepted.
"""

from __future__ import annotations

import ctypes
import errno
import json
import os
import platform
import resource
import signal
import subprocess
import sys
import sysconfig
import tempfile
import time
from pathlib import Path
from typing import Any, Callable


READ_FILE = 1 << 2
READ_DIR = 1 << 3
WRITE_FILE = 1 << 1
HANDLED_FS_ABI1 = (1 << 13) - 1
ALLOW = 0x7FFF0000
DENY_EPERM = 0x00050000 | errno.EPERM
COMPARE_EQ = 4
POLICY = "PURE_COMPUTE_SINGLE_PROCESS_V1_R2"

# Intentionally no socket, clone/fork, exec, kill/ptrace, mount, ioctl, chmod,
# truncate, link/rename, io_uring, bpf, device creation or policy modification.
ALLOWED_SYSCALLS = (
    "read",
    "write",
    "readv",
    "writev",
    "close",
    "fstat",
    "lstat",
    "stat",
    "newfstatat",
    "statx",
    "lseek",
    "pread64",
    "pwrite64",
    "openat",
    "open",
    "readlink",
    "readlinkat",
    "getdents64",
    "access",
    "faccessat",
    "faccessat2",
    "mmap",
    "mprotect",
    "munmap",
    "mremap",
    "brk",
    "madvise",
    "rt_sigaction",
    "rt_sigprocmask",
    "rt_sigreturn",
    "sigaltstack",
    "futex",
    "futex_waitv",
    "set_robust_list",
    "rseq",
    "clock_gettime",
    "gettimeofday",
    "time",
    "nanosleep",
    "clock_nanosleep",
    "getrandom",
    "getpid",
    "getppid",
    "gettid",
    "getuid",
    "geteuid",
    "getgid",
    "getegid",
    "getgroups",
    "uname",
    "sysinfo",
    "sched_getaffinity",
    "sched_yield",
    "getcpu",
    "dup",
    "dup2",
    "dup3",
    "getcwd",
    "statfs",
    "fstatfs",
    "getrlimit",
    "getrusage",
    "exit",
    "exit_group",
    "restart_syscall",
)


class RuleSet(ctypes.Structure):
    _fields_ = [("handled_access_fs", ctypes.c_uint64)]


class PathRule(ctypes.Structure):
    _pack_ = 1
    _fields_ = [("allowed_access", ctypes.c_uint64), ("parent_fd", ctypes.c_int32)]


class ArgCompare(ctypes.Structure):
    _fields_ = [
        ("arg", ctypes.c_uint),
        ("op", ctypes.c_int),
        ("datum_a", ctypes.c_uint64),
        ("datum_b", ctypes.c_uint64),
    ]


def _checked(result: int, operation: str) -> int:
    if result < 0:
        code = ctypes.get_errno()
        raise OSError(code, f"{operation}: {os.strerror(code)}")
    return result


def read_roots() -> list[Path]:
    """Only runtime/stdlib/shared-library leaves, never experiment/model
    roots."""
    candidates = [Path(sysconfig.get_path(key)) for key in ("stdlib", "platstdlib", "purelib", "platlib")]
    candidates += [Path(sys.prefix) / "lib", Path(sys.base_prefix) / "lib", Path("/lib"), Path("/usr/lib")]
    resolved = sorted({path.resolve() for path in candidates if path.exists()}, key=lambda path: len(str(path)))
    result: list[Path] = []
    for path in resolved:
        if path == Path("/"):
            raise RuntimeError("Refusing a root-wide filesystem rule")
        if not any(path == ancestor or ancestor in path.parents for ancestor in result):
            result.append(path)
    return result


def install_landlock(libc: Any, readable: list[Path], work: Path, scratch: Path) -> int:
    if platform.machine() != "x86_64":
        raise RuntimeError("Only the verified Linux x86_64 syscall ABI is supported")
    libc.syscall.restype = ctypes.c_long
    abi = _checked(libc.syscall(444, ctypes.c_void_p(), ctypes.c_size_t(0), ctypes.c_uint(1)), "Landlock ABI")
    ruleset = RuleSet(HANDLED_FS_ABI1)
    fd = _checked(libc.syscall(444, ctypes.byref(ruleset), ctypes.sizeof(ruleset), 0), "create Landlock ruleset")
    try:
        rules = [(path, READ_FILE | READ_DIR) for path in readable]
        rules += [(work, READ_FILE | READ_DIR), (scratch, READ_FILE | WRITE_FILE)]
        for special in ("/etc/ld.so.cache", "/dev/null", "/dev/urandom"):
            path = Path(special)
            if path.exists():
                rules.append((path, READ_FILE))
        for path, rights in rules:
            path_fd = os.open(path, os.O_PATH | os.O_CLOEXEC)
            try:
                rule = PathRule(rights, path_fd)
                _checked(libc.syscall(445, fd, 1, ctypes.byref(rule), 0), f"add Landlock rule {path}")
            finally:
                os.close(path_fd)
        _checked(libc.syscall(446, fd, 0), "restrict Landlock")
    finally:
        os.close(fd)
    return abi


def install_seccomp(library: Any) -> None:
    library.seccomp_init.argtypes = [ctypes.c_uint32]
    library.seccomp_init.restype = ctypes.c_void_p
    library.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    library.seccomp_syscall_resolve_name.restype = ctypes.c_int
    library.seccomp_rule_add_array.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_int,
        ctypes.c_uint,
        ctypes.POINTER(ArgCompare),
    ]
    library.seccomp_load.argtypes = [ctypes.c_void_p]
    library.seccomp_release.argtypes = [ctypes.c_void_p]
    context = library.seccomp_init(DENY_EPERM)
    if not context:
        raise RuntimeError("seccomp_init failed")

    def add(name: str, comparisons: tuple[ArgCompare, ...] = ()) -> None:
        number = library.seccomp_syscall_resolve_name(name.encode())
        if number < 0:
            raise RuntimeError(f"The existing libseccomp cannot resolve required syscall {name}")
        array = (ArgCompare * len(comparisons))(*comparisons) if comparisons else None
        result = library.seccomp_rule_add_array(context, ALLOW, number, len(comparisons), array)
        if result != 0:
            raise RuntimeError(f"Cannot add seccomp rule {name}: {result}")

    try:
        for name in ALLOWED_SYSCALLS:
            add(name)
        # CPython 3.10 sleep uses select/pselect6 with zero descriptors.
        for name in ("select", "pselect6"):
            add(name, tuple(ArgCompare(index, COMPARE_EQ, 0, 0) for index in range(4)))
        # Only queries of this process's limits; no raising/removing limits.
        add("prlimit64", (ArgCompare(0, COMPARE_EQ, 0, 0), ArgCompare(2, COMPARE_EQ, 0, 0)))
        for option in (21, 39):  # PR_GET_SECCOMP, PR_GET_NO_NEW_PRIVS only.
            add("prctl", (ArgCompare(0, COMPARE_EQ, option, 0),))
        for option in (0, 1, 2, 3, 1030):  # dup/get/set CLOEXEC/get flags, not F_SETOWN/F_SETFL.
            add("fcntl", (ArgCompare(1, COMPARE_EQ, option, 0),))
        result = library.seccomp_load(context)
        if result != 0:
            raise RuntimeError(f"seccomp_load failed: {result}")
    finally:
        library.seccomp_release(context)


def child_main(*, filesystem_guard: Callable[..., dict[str, Any]] | None = None, policy: str = POLICY) -> None:
    """The code branch is reachable ONLY after all guards succeed."""
    request = json.load(sys.stdin)
    control_fd = int(os.environ["RETOOL_GUARD_FD"])
    libc = ctypes.CDLL(None, use_errno=True)
    library = ctypes.CDLL("libseccomp.so.2", use_errno=True)
    readable = read_roots()
    work = Path(request["work"]).resolve(strict=True)
    scratch = (work / "scratch.txt").resolve(strict=True)
    os.chdir(work)
    if os.getuid() == 0:
        os.setgroups([])
        os.setgid(65534)
        os.setuid(65534)
    if os.getuid() == 0 or os.getgid() == 0:
        raise RuntimeError("Payload must not run as root")
    _checked(libc.prctl(38, 1, 0, 0, 0), "no_new_privs")
    _checked(libc.prctl(4, 0, 0, 0, 0), "disable dumpability")
    cap_eff = next(
        line.split(":", 1)[1].strip()
        for line in Path("/proc/self/status").read_text().splitlines()
        if line.startswith("CapEff:")
    )
    if int(cap_eff, 16):
        raise RuntimeError("Payload capabilities were not dropped")
    memory = int(request["memory_mb"]) * 1024 * 1024
    for kind, limit in (
        (resource.RLIMIT_AS, memory),
        (resource.RLIMIT_DATA, memory),
        (resource.RLIMIT_FSIZE, 262144),
        (resource.RLIMIT_NOFILE, 64),
        (resource.RLIMIT_NPROC, 0),
        (resource.RLIMIT_CORE, 0),
    ):
        resource.setrlimit(kind, (limit, limit))
    resource.setrlimit(resource.RLIMIT_CPU, (int(request["cpu_seconds"]), int(request["cpu_seconds"]) + 1))
    filesystem = (
        {"landlock_abi": install_landlock(libc, readable, work, scratch)}
        if filesystem_guard is None
        else filesystem_guard(libc, readable, work, scratch)
    )
    install_seccomp(library)
    seccomp_mode = libc.prctl(21, 0, 0, 0, 0)
    no_new_privs = libc.prctl(39, 0, 0, 0, 0)
    if seccomp_mode != 2 or no_new_privs != 1:
        raise RuntimeError("Guard attestation failed")
    attestation = {
        "uid": os.getuid(),
        "gid": os.getgid(),
        "cap_eff": cap_eff,
        "seccomp_mode": seccomp_mode,
        "no_new_privs": no_new_privs,
        "read_roots": list(map(str, readable)),
        "profile": policy,
        "memory_mb": request["memory_mb"],
        "cpu_seconds": request["cpu_seconds"],
        "output_file_limit_bytes": 262144,
        "process_creation_allowed": False,
        **filesystem,
    }
    os.write(control_fd, json.dumps(attestation).encode() + b"\n")
    os.close(control_fd)  # Untrusted code cannot rewrite the parent's attestation.
    if request.get("probe_only", False):
        return
    scope = {"__name__": "__main__", "__file__": "<isolated-code>"}
    exec(compile(request["code"], "<isolated-code>", "exec"), scope, scope)


def run_isolated(
    code: str,
    root: Path,
    *,
    timeout: float = 8,
    memory_mb: int = 1024,
    cpu_seconds: int = 3,
    probe_only: bool = False,
) -> dict[str, Any]:
    """Run a fresh restricted child.

    Never resume or replay code history.
    """
    if sys.platform != "linux" or os.getuid() != 0:
        raise RuntimeError("Validation target is the verified Linux root driver dropping to uid/gid 65534")
    if not isinstance(code, str) or len(code.encode()) > 65536:
        raise ValueError("Expected at most 64 KiB of code")
    if not 0 < timeout <= 40 or not 1 <= cpu_seconds <= 30 or not 128 <= memory_mb <= 2048:
        raise ValueError("Invalid resource limit")
    root = root.resolve(strict=True)
    start = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="ll_request_", dir=root) as directory:
        outer = Path(directory)
        outer.chmod(0o711)
        work = outer / "work"
        work.mkdir(mode=0o700)
        os.chown(work, 65534, 65534)
        scratch = work / "scratch.txt"
        scratch.touch(mode=0o600)
        os.chown(scratch, 65534, 65534)
        request = {
            "code": code,
            "work": str(work),
            "memory_mb": memory_mb,
            "cpu_seconds": cpu_seconds,
            "probe_only": probe_only,
        }
        env = {
            "PATH": "/usr/bin:/bin",
            "HOME": str(work),
            "TMPDIR": str(work),
            "CUDA_VISIBLE_DEVICES": "",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONIOENCODING": "utf-8",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        }
        return run_guarded_process(
            lambda fd: [sys.executable, "-I", str(Path(__file__).resolve()), "--child"],
            request,
            env,
            root,
            timeout=timeout,
            policy=POLICY,
            started=start,
        )


def parse_guard_stream(raw: bytes, *, require_exit_record: bool) -> tuple[dict | None, int | None]:
    """Only trusted bootstrap writers retain the pipe after payload
    execution."""
    rows = [json.loads(line) for line in raw.splitlines()]
    guard = rows.pop(0) if rows and "profile" in rows[0] else None
    exit_code = None
    if rows:
        if len(rows) != 1 or set(rows[0]) != {"event", "returncode"} or rows[0]["event"] != "process_exit":
            raise ValueError("Unexpected guard control record")
        exit_code = rows[0]["returncode"]
        if type(exit_code) is not int or not -64 <= exit_code <= 255:
            raise ValueError("Invalid guarded child exit status")
    if require_exit_record and exit_code is None:
        raise ValueError("Missing trusted child wait status")
    return guard, exit_code


def run_guarded_process(
    command: Callable[[int], list[str]],
    request: dict,
    env: dict[str, str],
    root: Path,
    *,
    timeout: float,
    policy: str,
    started: float,
    require_exit_record: bool = False,
) -> dict[str, Any]:
    """Own one process group and both pipe FDs, including startup failures."""
    read_fd, write_fd = os.pipe()
    process = None
    timed_out = False
    try:
        with tempfile.TemporaryFile(dir=root) as stdout, tempfile.TemporaryFile(dir=root) as stderr:
            try:
                process = subprocess.Popen(
                    command(write_fd),
                    stdin=subprocess.PIPE,
                    stdout=stdout,
                    stderr=stderr,
                    env=dict(env, RETOOL_GUARD_FD=str(write_fd)),
                    cwd=root,
                    pass_fds=(write_fd,),
                    start_new_session=True,
                )
                os.close(write_fd)
                write_fd = -1
                try:
                    process.communicate(json.dumps(request).encode(), timeout=timeout)
                except subprocess.TimeoutExpired:
                    timed_out = True
            finally:
                if process is not None:
                    if process.poll() is None:
                        try:
                            os.killpg(process.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                    process.wait()
                    if process.stdin is not None:
                        process.stdin.close()
            with os.fdopen(read_fd, "rb", closefd=False) as control:
                raw = control.read(65537)
            guard_error = None
            try:
                guard, child_exit = parse_guard_stream(raw, require_exit_record=require_exit_record and not timed_out)
            except (ValueError, TypeError) as error:
                guard, child_exit, guard_error = None, None, str(error)
            if require_exit_record and process.returncode != 0 and not timed_out:
                guard_error = f"Apptainer wrapper failed exit{process.returncode}"
            stdout.seek(0)
            stderr.seek(0)
            out, err = stdout.read(262145), stderr.read(262145)
            return {
                "returncode": process.returncode if child_exit is None else child_exit,
                "wrapper_returncode": process.returncode,
                "timed_out": timed_out,
                "wall_seconds": time.monotonic() - started,
                "guard": guard,
                "guard_error": guard_error,
                "stdout": out.decode(errors="replace"),
                "stderr": err.decode(errors="replace"),
                "stdout_bytes": len(out),
                "stderr_bytes": len(err),
                "policy": policy,
                "used_gpu": False,
            }
    finally:
        os.close(read_fd)
        if write_fd >= 0:
            os.close(write_fd)


if __name__ == "__main__":
    if sys.argv[1:] != ["--child"]:
        raise SystemExit("Use the validation driver; this is not a public code execution service")
    child_main()
