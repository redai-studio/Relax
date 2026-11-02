# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Reproduce the fixed 32-arm CPU study on Kaggle, with isolated revisions and
no retries."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from session_replay_watchdog import cleanup_process_harness, identity, save, stop_command


OUT = None
ROOT = None
COMMANDS = []
RUN_ID = None
MONITOR = None
GATE_CODE = """import os,sys,time
deadline=time.monotonic()+30
while not os.path.isfile(sys.argv[1]):
    if time.monotonic()>deadline:
        raise TimeoutError('Cloud owner did not release command gate')
    time.sleep(.01)
os.execvpe(sys.argv[2],sys.argv[2:],os.environ)
"""


def state(phase, **extra):
    row = dict(phase=phase, at=time.time(), run_id=RUN_ID, **extra)
    save(OUT / "state.json", row)
    print("TASK14_PROGRESS " + json.dumps(row), flush=True)


def event(kind, detail):
    try:
        with (OUT / "runner-cleanup-events.jsonl").open("a") as stream:
            stream.write(json.dumps(dict(at=time.time(), kind=kind, detail=detail)) + "\n")
    except BaseException as error:
        try:
            print("RUNNER_EVENT_WRITE_ERROR", kind, repr(error), file=sys.stderr, flush=True)
        except BaseException:
            pass


def run(argv, cwd=None, timeout=600, check=True, process_harness=None):
    key = str(time.time_ns())
    log_path = OUT / f"command-{key}.log"
    env = dict(
        os.environ,
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        TOKENIZERS_PARALLELISM="false",
        PYTHONPATH=str(cwd) if cwd else "",
    )
    gate = OUT / f".command-gate-{key}"
    with log_path.open("w") as log:
        child = None
        receipt = dict(
            id=key,
            run_id=RUN_ID,
            phase="preparing",
            command=argv,
            cwd=str(cwd),
            log=str(log_path),
            started_at=time.time(),
            timeout_seconds=timeout,
            process_harness=str(process_harness) if process_harness else None,
        )
        try:
            if MONITOR is not None and MONITOR.poll() is not None:
                raise RuntimeError("Independent cloud supervisor exited before command launch")
            child = subprocess.Popen(
                [sys.executable, "-c", GATE_CODE, str(gate), *argv],
                cwd=cwd,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            receipt.update(phase="running", pid=child.pid, identity=identity(child.pid))
            if (
                receipt["identity"] is None
                or receipt["identity"]["pid"] != receipt["identity"]["pgid"]
                or receipt["identity"]["sid"] != child.pid
            ):
                child.kill()  # Target argv is still behind the no-spawn gate.
                raise RuntimeError("Could not verify the gated command process identity")
            save(OUT / "active-command.json", receipt)
            gate.write_text("go")
            deadline = time.monotonic() + timeout
            while True:
                if MONITOR is not None and MONITOR.poll() is not None:
                    raise RuntimeError("Independent cloud supervisor exited during an active command")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(argv, timeout)
                try:
                    code = child.wait(timeout=min(10, remaining))
                    break
                except subprocess.TimeoutExpired:
                    continue
        except BaseException as error:
            try:
                if child is not None:
                    stop_command(receipt, OUT, event)
            finally:
                try:
                    if child is not None:
                        child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=5)
                receipt.update(
                    phase="error",
                    timed_out=isinstance(error, subprocess.TimeoutExpired),
                    error=f"{type(error).__name__}: {error}",
                    exit_code=child.returncode if child else None,
                    finished_at=time.time(),
                )
                COMMANDS.append(receipt)
                save(OUT / "commands.json", dict(commands=COMMANDS))
                save(OUT / "active-command.json", receipt)
            raise
        receipt.update(phase="complete" if code == 0 else "error", exit_code=code, finished_at=time.time())
        COMMANDS.append(receipt)
        save(OUT / "commands.json", dict(commands=COMMANDS))
        save(OUT / "active-command.json", receipt)
    if process_harness and code:
        cleanup_process_harness(Path(process_harness), str(cwd), event)
    if code and check:
        print(log_path.read_text()[-12000:], flush=True)
        raise RuntimeError(f"Command exit {code}: {argv}")
    return code


def source_hashes(repo):
    names = set(subprocess.check_output(["git", "ls-files", "-z"], cwd=repo).decode().split("\0"))
    names.update(
        subprocess.check_output(["git", "ls-files", "--others", "--exclude-standard", "-z"], cwd=repo)
        .decode()
        .split("\0")
    )
    return {
        name: hashlib.sha256((repo / name).read_bytes()).hexdigest()
        for name in sorted(names)
        if name and (repo / name).is_file()
    }


def main():
    global OUT, RUN_ID, MONITOR
    if sys.platform != "linux" or not Path("/kaggle/working").is_dir():
        raise RuntimeError("Run this reproduction only inside Kaggle")
    os.environ.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", TOKENIZERS_PARALLELISM="false")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--plan-only", action="store_true", help="Build both isolated sources and record32commands; do not measure"
    )
    args = parser.parse_args()
    source = args.source_root.resolve()
    assets = Path(__file__).resolve().parent / "session_replay_evidence"
    protocol = json.loads((assets / "replay-protocol.json").read_text())
    schedule = json.loads((assets / "planned-study-order.json").read_text())["rows"]
    expected = []
    workloads = [(length, c) for length in ("short", "long") for c in (1, 4)]
    for pair in range(4):
        for length, concurrency in workloads if pair % 2 == 0 else list(reversed(workloads)):
            for arm in ("baseline", "candidate") if pair % 2 == 0 else ("candidate", "baseline"):
                expected.append(dict(pair=pair, seed=pair, length=length, concurrency=concurrency, arm=arm))
    if schedule != expected:
        raise ValueError("Packaged schedule differs from the declared32-arm AB/BA protocol")
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip() != protocol["base_sha"]:
        raise ValueError(
            "Source checkout HEAD must be the documented pinned base; keep the candidate as its reviewed working-tree overlay"
        )
    benchmark = source / protocol["benchmark_path"]
    pipeline = source / protocol["pipeline_path"]
    if hashlib.sha256(benchmark.read_bytes()).hexdigest() != protocol["benchmark_sha256"]:
        raise ValueError("Replay benchmark differs from the inspected performance implementation")
    candidate_core = pipeline.read_bytes().replace(b"\r\n", b"\n")
    if hashlib.sha256(candidate_core).hexdigest() != protocol["pipeline_candidate_normalized_sha256"]:
        raise ValueError("Candidate compaction differs from the inspected optimization")
    OUT = args.output.resolve()
    if not OUT.is_relative_to(Path("/kaggle/working")) or OUT == Path("/kaggle/working"):
        raise ValueError("Output must be a fresh child of /kaggle/working")
    if OUT.exists():
        raise ValueError("Refuse existing outputs; failed arms are never resumed/retried automatically")
    if any(path.resolve().is_relative_to(OUT) for path in (args.corpus, args.tokenizer, source)):
        raise ValueError("Raw inputs/source must stay outside result outputs")
    if not (args.corpus / "provenance.json").is_file() or not (args.tokenizer / "tokenizer-source.json").is_file():
        raise ValueError("Prepare pinned inputs using prepare_session_replay_inputs.py first")
    OUT.mkdir(parents=True)
    RUN_ID = "task14-reproduction-" + str(time.time_ns())
    workspace = Path(tempfile.mkdtemp(prefix="rx14study-", dir="/tmp"))
    monitor_log = (OUT / "watchdog.log").open("w")
    MONITOR = subprocess.Popen(
        [
            sys.executable,
            str(Path(__file__).with_name("session_replay_watchdog.py")),
            "--directory",
            str(OUT),
            "--runner-identity",
            json.dumps(identity(os.getpid())),
            "--run-id",
            RUN_ID,
            "--timeout",
            "1800",
        ],
        stdout=monitor_log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    monitor_log.close()
    try:
        deadline = time.monotonic() + 10
        while True:
            if MONITOR.poll() is not None:
                raise RuntimeError("Supervisor exited before readiness")
            ready = OUT / "watchdog-state.json"
            observed = json.loads(ready.read_text()) if ready.exists() else {}
            current = identity(os.getpid())
            if (
                observed.get("phase") == "watching"
                and observed.get("run_id") == RUN_ID
                and current
                and all(
                    (observed.get("runner_identity") or {}).get(k) == current[k]
                    for k in ("pid", "pgid", "sid", "start_ticks")
                )
            ):
                break
            if time.monotonic() > deadline:
                raise TimeoutError("Supervisor readiness")
            time.sleep(0.05)
        state("environment", plan_only=args.plan_only, workspace=str(workspace), performance_run=not args.plan_only)
        run([sys.executable, "-m", "pip", "freeze"], timeout=60)
        save(
            OUT / "allocated-machine.json",
            dict(
                uname=list(os.uname()),
                python=sys.version,
                cpu_count=os.cpu_count(),
                cpu_affinity=sorted(os.sched_getaffinity(0)),
                thread_env={
                    k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "TOKENIZERS_PARALLELISM")
                },
            ),
        )
        baseline, candidate = workspace / "baseline", workspace / "candidate"
        run(["git", "clone", "--shared", str(source), str(baseline)])
        run(["git", "checkout", "--detach", protocol["base_sha"]], cwd=baseline)
        run(["git", "clone", "--shared", str(baseline), str(candidate)])
        for repo in (baseline, candidate):
            target = repo / protocol["benchmark_path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(benchmark.read_bytes())
        (candidate / protocol["pipeline_path"]).write_bytes(candidate_core)
        if (
            hashlib.sha256((baseline / protocol["pipeline_path"]).read_bytes()).hexdigest()
            != protocol["pipeline_baseline_sha256"]
        ):
            raise ValueError("Original pipeline is not the pinned unmodified source")
        checked = {arm: source_hashes(repo) for arm, repo in (("baseline", baseline), ("candidate", candidate))}
        differences = sorted(
            name
            for name in set(checked["baseline"]) | set(checked["candidate"])
            if checked["baseline"].get(name) != checked["candidate"].get(name)
        )
        if differences != [protocol["pipeline_path"]]:
            raise ValueError("Comparison sources must differ only in pipeline compaction")
        save(
            OUT / "source-provenance.json",
            dict(
                base_sha=protocol["base_sha"],
                files=checked,
                only_difference=differences,
                protocol=protocol,
                benchmark_copied_to_both=True,
                plan_only=args.plan_only,
            ),
        )
        save(OUT / "planned-study-order.json", dict(rows=schedule))
        plan = []
        for item in schedule:
            arm, pair, length, concurrency = (item[k] for k in ("arm", "pair", "length", "concurrency"))
            repo = baseline if arm == "baseline" else candidate
            argv = [
                sys.executable,
                protocol["benchmark_path"],
                "--corpus",
                str(args.corpus.resolve()),
                "--tokenizer",
                str(args.tokenizer.resolve()),
                "--length",
                length,
                "--exports",
                "64",
                "--concurrency",
                str(concurrency),
                "--order-seed",
                str(pair),
                "--output",
                str(OUT / f"arm-{pair:02d}-{length}-c{concurrency}-{arm}.json"),
            ]
            plan.append(dict(**item, command=argv, cwd=str(repo)))
        save(OUT / "command-plan.json", dict(rows=plan))
        executed = []
        if not args.plan_only:
            for item in plan:
                state(
                    "paired_arm",
                    completed_arms=len(executed),
                    total_arms=32,
                    **{k: item[k] for k in ("pair", "length", "concurrency", "arm")},
                )
                run(item["command"], cwd=Path(item["cwd"]), timeout=600)
                receipt = COMMANDS[-1]
                executed.append(
                    dict(
                        **item,
                        command_id=receipt["id"],
                        identity=receipt["identity"],
                        started_at=receipt["started_at"],
                        finished_at=receipt["finished_at"],
                        exit_code=receipt["exit_code"],
                    )
                )
                save(OUT / "actual-study-order.json", dict(rows=executed))
            run(
                [
                    sys.executable,
                    str(Path(__file__).with_name("summarize_session_replay.py")),
                    "--directory",
                    str(OUT),
                    "--output",
                    str(OUT / "paired-summary.json"),
                ],
                timeout=60,
            )
        if any(
            source_hashes(repo) != checked[arm] for arm, repo in (("baseline", baseline), ("candidate", candidate))
        ):
            raise ValueError("Comparison source changed during reproduction")
        save(
            OUT / "reproduction-summary.json",
            dict(
                passed=True,
                plan_only=args.plan_only,
                planned_arms=32,
                measured_arms=len(executed),
                automatic_retries=0,
                source_unchanged=True,
                workspace=str(workspace),
                scope="Actual32armstudy" if executed else "Source/command preparation only; no performance evidence",
            ),
        )
        state("complete", plan_only=args.plan_only, completed_arms=len(executed))
    except BaseException as error:
        state("error", error=f"{type(error).__name__}: {error}", automatic_retries=0)
        raise
    finally:
        try:
            MONITOR.wait(timeout=50)
        except subprocess.TimeoutExpired:
            MONITOR.terminate()
            MONITOR.wait(timeout=5)
        watcher_path = OUT / "watchdog-state.json"
        watcher = json.loads(watcher_path.read_text()) if watcher_path.exists() else {}
        state_path = OUT / "state.json"
        expected_reason = (
            "complete"
            if state_path.exists() and json.loads(state_path.read_text()).get("phase") == "complete"
            else "runner_error"
        )
        verified = (
            MONITOR.returncode == 0 and watcher.get("phase") == "terminal" and watcher.get("reason") == expected_reason
        )
        summary_path = OUT / "reproduction-summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text())
            summary["independent_watchdog_verified"] = verified
            if not verified:
                summary["passed"] = False
            save(summary_path, summary)
        if not verified:
            state("error", error="Independent supervisor terminal result did not verify", watchdog=watcher)
            raise RuntimeError("Independent supervisor did not verify terminal state")


if __name__ == "__main__":
    if sys.platform != "linux" or not Path("/kaggle/working").is_dir():
        raise RuntimeError("Run this reproduction only inside Kaggle")
    own = identity(os.getpid())
    if own and own["pid"] == own["pgid"] == own["sid"]:
        main()
    else:
        # A normal notebook shell need not provide an isolated session. Re-enter
        # once with a real OS-owned group before starting the independent monitor.
        child = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]], start_new_session=True
        )
        try:
            raise SystemExit(child.wait(timeout=1880))
        except (KeyboardInterrupt, subprocess.TimeoutExpired):
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait(timeout=5)
            raise
