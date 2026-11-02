# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Independent Kaggle supervisor: finite deadlines, new-log errors, owned
cleanup.

Never run on the user's computer. No evaluation is retried by this supervisor.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import sys
import time
from pathlib import Path


def save(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2))
    temp.replace(path)


def identity(pid):
    try:
        fields = Path(f"/proc/{int(pid)}/stat").read_text().rsplit(")", 1)[1].split()
        return dict(pid=int(pid), pgid=int(fields[2]), sid=int(fields[3]), start_ticks=fields[19], state=fields[0])
    except (OSError, IndexError, ValueError):
        return None


def matches(old):
    if not isinstance(old, dict) or not {"pid", "pgid", "sid", "start_ticks"} <= old.keys():
        return False
    now = identity(old["pid"])
    return now is not None and all(now[key] == old[key] for key in ("pid", "pgid", "sid", "start_ticks"))


def freeze_owned_tree(root, event, journal=None):
    """Follow kernel child links from one verified owner, never scan global
    PIDs."""
    seen = set()
    rows = [] if journal is None else journal

    def visit(row, depth):
        if not matches(row) or row["pid"] in seen or row["pid"] == os.getpid():
            return
        if depth > 32 or len(seen) >= 2048:
            event("owned_tree_safety_bound", dict(depth=depth, members=len(seen)))
            return
        seen.add(row["pid"])
        try:
            os.kill(row["pid"], signal.SIGSTOP)
        except (ProcessLookupError, PermissionError):
            return
        rows.append(row)
        # Wait for the process thread group to stop before enumerating all
        # threads' child links. The mutable journal already owns this SIGSTOP.
        deadline = time.monotonic() + 2
        tasks = []
        while matches(row):
            try:
                tasks = list(Path(f"/proc/{row['pid']}/task").iterdir())
                states = [path.joinpath("stat").read_text().rsplit(")", 1)[1].split()[0] for path in tasks]
            except OSError:
                tasks = []
                states = []
            if states and all(value in ("T", "t", "Z", "X") for value in states):
                break
            if time.monotonic() >= deadline:
                event("owned_stop_unconfirmed", row)
                return
            time.sleep(0.01)
        pids = set()
        for task in tasks:
            try:
                pids.update(int(value) for value in task.joinpath("children").read_text().split())
            except OSError:
                continue
        for pid in pids:
            child = identity(pid)
            if child:
                visit(child, depth + 1)

    visit(root, 0)
    event("owned_tree_frozen", dict(root=root, members=rows))
    return rows


def terminate_group(members, pgid, event):
    verified = [row for row in members if row.get("pgid") == pgid and row.get("sid") == pgid and matches(row)]
    if not verified:
        return
    try:
        os.killpg(pgid, signal.SIGTERM)
        # A timeout owner may have frozen these tasks to close the next-command
        # race. Let the pending TERM run before considering the bounded KILL.
        os.killpg(pgid, signal.SIGCONT)
        event("owned_group_signal", dict(pgid=pgid, signal="TERM", members=verified))
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline and any(matches(row) for row in verified):
            time.sleep(0.1)
        if any(matches(row) for row in verified):
            os.killpg(pgid, signal.SIGKILL)
            event("owned_group_signal", dict(pgid=pgid, signal="KILL"))
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            return
        event(
            "owned_group_remaining",
            dict(
                pgid=pgid,
                known_members_still_match=[row for row in verified if matches(row)],
                reason="Do not signal an unverified recycled group; native owner/parent must reap remaining tasks.",
            ),
        )
    except ProcessLookupError:
        pass
    finally:
        # A filesystem/signal/reporting error must not strand a verified member
        # in SIGSTOP. Keep it able to finish or be reaped by its native owner.
        for row in verified:
            if matches(row):
                try:
                    os.kill(row["pid"], signal.SIGCONT)
                except (ProcessLookupError, PermissionError):
                    pass


def cleanup_process_harness(directory, expected_source_root, event):
    """Only this run's validated ledger and private receipt paths authorize
    signals."""
    owner = directory / "process-ownership.json"
    if not owner.exists():
        return
    value = json.loads(owner.read_text())
    if Path(value["source_root"]).resolve() != Path(expected_source_root).resolve():
        raise ValueError("Ownership source root differs from frozen command source")
    scratch = Path(value["owned_temporary_root"]).resolve()
    if scratch.parent != Path("/tmp") or not scratch.name.startswith("rx14-"):
        raise ValueError("Ownership scratch root is not an isolated harness path")
    if scratch.exists() and scratch.stat().st_uid != os.getuid():
        raise ValueError("Harness scratch root has another owner")
    groups = {}
    daemon = value.get("daemon")
    daemon_row = None
    if daemon:
        expected_socket = str(scratch / "launcher.sock")
        argv = daemon.get("argv", [])
        if len(argv) != 5 or argv[1:] != [
            "-c",
            "from relax.agentic.runner.ipc import _main; _main()",
            "--launcher-socket",
            expected_socket,
        ]:
            raise ValueError("Daemon command does not match the owned launcher recipe")
        daemon_row = daemon["identity"]
        if daemon_row["pid"] != daemon_row["pgid"] or daemon_row["sid"] != daemon_row["pgid"]:
            raise ValueError("Invalid owned daemon identity")
        # Close the launch race and capture any pre-receipt command descendants.
        daemon_tree = []
        try:
            freeze_owned_tree(daemon_row, event, daemon_tree)
        except BaseException as error:
            event("owned_tree_capture_error", f"{type(error).__name__}: {error}")
        for row in daemon_tree:
            if row["pgid"] == row["sid"] and row["pgid"] != daemon_row["pgid"]:
                groups.setdefault(row["pgid"], []).append(row)
    for proof in value.get("ownership_proofs") or []:
        try:
            leader = proof["worker"]
            if leader["pid"] != leader["pgid"] or leader["sid"] != leader["pgid"]:
                raise ValueError("Invalid original command group leader")
            groups.setdefault(leader["pgid"], []).extend((leader, proof["child"]))
        except BaseException as error:
            event("invalid_owner_receipt", f"{type(error).__name__}: {error}")
    for name in value.get("receipt_directories") or []:
        try:
            receipt_dir = Path(name).resolve()
            if not receipt_dir.is_relative_to(directory.resolve()):
                raise ValueError("Receipt directory escapes this command output")
            leader_path = receipt_dir / "leader-identity.json"
            if not leader_path.exists():
                continue
            leader = json.loads(leader_path.read_text())["worker"]
            if leader["pid"] != leader["pgid"] or leader["sid"] != leader["pgid"]:
                raise ValueError("Invalid pre-ready command group leader")
            groups.setdefault(leader["pgid"], []).append(leader)
            child_path = receipt_dir / "child-ready.json"
            if child_path.exists():
                child = json.loads(child_path.read_text())
                if child["pgid"] == leader["pgid"] and child["sid"] == leader["sid"]:
                    groups[leader["pgid"]].append(child)
        except BaseException as error:
            event("invalid_owner_receipt", f"{type(error).__name__}: {error}")
    # Agents have separate process groups: finish them while the real subreaper
    # daemon still exists, rather than assuming the daemon's PGID contains them.
    for pgid, members in groups.items():
        try:
            terminate_group(members, pgid, event)
        except BaseException as error:
            event("owned_group_cleanup_error", dict(pgid=pgid, error=f"{type(error).__name__}: {error}"))
    if daemon_row:
        terminate_group([daemon_row], daemon_row["pgid"], event)


def stop_command(command, directory, event):
    row = command.get("identity")
    tree = []
    try:
        if row and row["pid"] == row["pgid"] == row["sid"]:
            freeze_owned_tree(row, event, tree)
        if command.get("process_harness"):
            cleanup_process_harness(Path(command["process_harness"]), command["cwd"], event)
    finally:
        groups = {}
        for member in tree:
            if member["pgid"] == member["sid"]:
                groups.setdefault(member["pgid"], []).append(member)
        if row:
            groups.setdefault(row["pgid"], []).append(row)
        for pgid, members in reversed(list(groups.items())):
            try:
                terminate_group(members, pgid, event)
            except BaseException as error:
                event("owned_group_cleanup_error", dict(pgid=pgid, error=f"{type(error).__name__}: {error}"))


def stop_runner(runner, directory, event):
    tree = []
    try:
        freeze_owned_tree(runner, event, tree)
        path = directory / "active-command.json"
        command = json.loads(path.read_text()) if path.exists() else {}
        stop_command(command, directory, event)
    finally:
        groups = {}
        for member in tree:
            if member["pgid"] == member["sid"]:
                groups.setdefault(member["pgid"], []).append(member)
        groups.setdefault(runner["pgid"], []).append(runner)
        for pgid, members in reversed(list(groups.items())):
            try:
                terminate_group(members, pgid, event)
            except BaseException as error:
                event("owned_group_cleanup_error", dict(pgid=pgid, error=f"{type(error).__name__}: {error}"))


def main():
    if not Path("/kaggle").exists():
        raise RuntimeError("Kaggle-only supervisor")
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--runner-identity", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--timeout", type=int, required=True)
    args = parser.parse_args()
    runner = json.loads(args.runner_identity)
    if runner["pid"] != runner["pgid"] or runner["sid"] != runner["pgid"] or not matches(runner):
        raise RuntimeError("Supervisor did not receive one live isolated runner identity")
    started = time.monotonic()
    offsets = {}
    reported = set()
    last_progress = time.monotonic()
    last_command = None

    def event(kind, detail):
        row = dict(at=time.time(), kind=kind, detail=detail)
        try:
            with (args.directory / "watchdog-events.jsonl").open("a") as stream:
                stream.write(json.dumps(row) + "\n")
            if kind not in ("complete", "owned_group_signal"):
                save(args.directory / "needs-attention.json", row)
        except BaseException as error:
            try:
                print("SUPERVISOR_EVENT_WRITE_ERROR", kind, repr(error), file=sys.stderr, flush=True)
            except BaseException:
                pass

    while True:
        state_path = args.directory / "state.json"
        state = json.loads(state_path.read_text()) if state_path.exists() else {}
        command_path = args.directory / "active-command.json"
        command = json.loads(command_path.read_text()) if command_path.exists() else {}
        terminal = state.get("phase") in ("complete", "error")
        current_runner = identity(runner["pid"])
        dead = not matches(runner) or current_runner.get("state") in ("Z", "X") if current_runner else True
        if terminal or dead or time.monotonic() - started > args.timeout:
            if not terminal or state.get("phase") == "error":
                try:
                    if not terminal and not dead:
                        stop_runner(runner, args.directory, event)
                    else:
                        stop_command(command, args.directory, event)
                except BaseException as error:
                    event("ownership_cleanup_error", f"{type(error).__name__}: {error}")
            kind = (
                "complete"
                if state.get("phase") == "complete"
                else "runner_error"
                if terminal
                else "runner_dead"
                if dead
                else "overall_timeout"
            )
            event(kind, state)
            save(args.directory / "watchdog-state.json", dict(phase="terminal", reason=kind, at=time.time()))
            return
        key = command.get("id")
        if key and key != last_command:
            last_command = key
            last_progress = time.monotonic()
        if key:
            log = Path(command["log"])
            if log.exists():
                with log.open("rb") as stream:
                    stream.seek(offsets.get(key, 0))
                    chunk = stream.read()
                    offsets[key] = stream.tell()
                if chunk:
                    last_progress = time.monotonic()
                    match = re.search(rb"Traceback|AssertionError|\bERROR\b|\bFAILED\b", chunk, re.I)
                    if match and ("log", key) not in reported:
                        reported.add(("log", key))
                        event(
                            "new_log_error",
                            dict(
                                command_id=key,
                                tail=chunk[max(0, match.start() - 100) : match.start() + 1600].decode(
                                    errors="replace"
                                ),
                            ),
                        )
            if command.get("phase") == "running" and time.time() - command["started_at"] > command["timeout_seconds"]:
                if ("timeout", key) not in reported:
                    reported.add(("timeout", key))
                    event("command_timeout", command)
                    try:
                        stop_command(command, args.directory, event)
                    except BaseException as error:
                        event("ownership_cleanup_error", f"{type(error).__name__}: {error}")
            if time.monotonic() - last_progress > 600 and ("stall", key) not in reported:
                reported.add(("stall", key))
                event("progress_stalled", dict(command_id=key, seconds=time.monotonic() - last_progress))
        if state.get("run_id") not in (None, args.run_id) or command.get("run_id") not in (None, args.run_id):
            event(
                "run_identity_mismatch",
                dict(expected=args.run_id, state=state.get("run_id"), command=command.get("run_id")),
            )
            stop_runner(runner, args.directory, event)
            save(
                args.directory / "watchdog-state.json",
                dict(phase="terminal", reason="run_identity_mismatch", at=time.time()),
            )
            return
        save(
            args.directory / "watchdog-state.json",
            dict(phase="watching", checked_at=time.time(), runner_identity=runner, run_id=args.run_id, command_id=key),
        )
        time.sleep(10)


if __name__ == "__main__":
    main()
