# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Real-text, same-shard fault integration with native managed child processes.

Requires pinned inputs and an evidence directory supplied by the Kaggle runner.
Inference and the permit RPC transport are explicit local adapters. No actual
model inference or HTTP/Ray deployment is claimed. The native launcher, command
processes, compiler, Session IRs, builder, compactor and cleanup are production
code.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import os
import shlex
import subprocess
import sys
import tempfile
import time
import traceback
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

import pytest

from relax.agentic.pipeline import RuntimeGroupError, SessionExportTransport, SessionSpec, TrainingFieldArtifact
from relax.agentic.runner.ipc import LauncherClient, ManagedAgentLauncher, ManagedCommandAppSpec, _ping_launcher
from relax.agentic.session.service import GroupOwnership
from scripts.benchmarks.benchmark_session_replay import load_cases, make_shard, replay_turn, training_digest
from scripts.benchmarks.prepare_session_replay_inputs import TOKENIZER_FILES, TOKENIZER_ID, TOKENIZER_REVISION


WORKER = r"""
import json,os,pathlib,signal,subprocess,sys,time
root=pathlib.Path(os.environ['TASK14_FAULT_RECEIPTS'])/os.environ['RELAX_SESSION_ID']
def identity(pid):
    f=pathlib.Path('/proc/%s/stat'%pid).read_text().rsplit(')',1)[1].split()
    return dict(pid=pid,pgid=int(f[2]),sid=int(f[3]),start_ticks=f[19],state=f[0])
def save(name,value):
    p=root/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(value));t.replace(p)
save('leader-identity.json',dict(worker=identity(os.getpid())))
child_code="import json,os,pathlib,signal; f=pathlib.Path('/proc/self/stat').read_text().rsplit(')',1)[1].split(); p=pathlib.Path(os.environ['TASK14_CHILD_READY']); t=p.with_suffix('.tmp'); t.write_text(json.dumps(dict(pid=os.getpid(),pgid=int(f[2]),sid=int(f[3]),start_ticks=f[19],state=f[0]))); t.replace(p); signal.pause()"
child=subprocess.Popen([sys.executable,'-c',child_code],env={**os.environ,'TASK14_CHILD_READY':str(root/'child-ready.json')})
deadline=time.monotonic()+180
while not (root/'child-ready.json').exists():
    if child.poll() is not None or time.monotonic()>deadline:raise RuntimeError('Child readiness failed')
    time.sleep(.02)
save('ready.json',dict(worker=identity(os.getpid()),child=identity(child.pid),
    session_dir=os.environ['RELAX_SESSION_IO_DIR'],output_path=os.environ['RELAX_OUTPUT_JSON']))
while not (root/'go').exists():
    if time.monotonic()>deadline:raise TimeoutError('Completion control timed out')
    time.sleep(.02)
pathlib.Path(os.environ['RELAX_OUTPUT_JSON']).write_text(json.dumps({'reward':1.0,'metadata':{'test_session':os.environ['RELAX_SESSION_ID']}}))
"""


def identity(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return dict(pid=pid, pgid=int(fields[2]), sid=int(fields[3]), start_ticks=fields[19], state=fields[0])
    except (FileNotFoundError, ProcessLookupError):
        return None


def same_process(old):
    now = identity(old["pid"])
    return now is not None and all(now[key] == old[key] for key in ("pid", "pgid", "sid", "start_ticks"))


def fd_targets():
    values = []
    for path in Path("/proc/self/fd").iterdir():
        try:
            values.append(os.readlink(path))
        except FileNotFoundError:
            pass
    return dict(Counter(values))


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


async def until(predicate, label, timeout=20):
    deadline = time.monotonic() + timeout
    while True:
        result = predicate()
        if result:
            return result
        if time.monotonic() > deadline:
            raise TimeoutError(label)
        await asyncio.sleep(0.02)


@pytest.fixture(scope="module")
def real_inputs():
    values = {key: os.environ.get("RELAX_TASK14_" + key) for key in ("CORPUS", "TOKENIZER", "FAULT_OUTPUT")}
    if not all(values.values()):
        pytest.skip("Pinned private Kaggle replay inputs/evidence directory were not supplied")
    if sys.platform != "linux" or not Path("/kaggle/working").is_dir():
        pytest.skip("Native process integration is run on Kaggle, never on the user computer")
    from argparse import Namespace

    from transformers import AutoTokenizer

    root = Path(values["TOKENIZER"])
    assert {path.name for path in root.iterdir()} == set(TOKENIZER_FILES) | {"tokenizer-source.json"}
    metadata = json.loads((root / "tokenizer-source.json").read_text())
    assert metadata == dict(id=TOKENIZER_ID, revision=TOKENIZER_REVISION, files=TOKENIZER_FILES)
    for name, sha in metadata["files"].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == sha
    tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True, trust_remote_code=False)
    cases, selection = load_cases(Namespace(corpus=Path(values["CORPUS"]), length="long", exports=4), tokenizer)
    assert len(cases) == len({case.row for case in cases}) == 4
    assert all(sum(map(len, case.response_tokens)) >= 4000 for case in cases)
    return cases, tokenizer, Path(values["FAULT_OUTPUT"]), selection


class SessionRun:
    def __init__(self, inputs, label):
        cases, self.tokenizer, self.evidence, self.selection = inputs
        self.cases = {f"fault-{index}": case for index, case in enumerate(cases)}
        self.label = label
        self.receipts = self.evidence / label
        self.owned = Path(tempfile.mkdtemp(prefix="rx14-", dir="/tmp"))
        self.socket = self.owned / "launcher.sock"
        self.executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="fault-compiler")
        self.shard = make_shard(self.cases, self.tokenizer, self.executor)
        self.backend = self.shard._generation_backend
        self.sessions = {}
        self.resources = {}
        self.publications = {key: [] for key in self.cases}
        self.proofs = {}
        self.tasks = []
        self.gates = []
        self.events = []
        self.builder_errors = []
        self.builder_sessions = []
        self.training_hashes = {}
        self.daemon = None

    def ledger(self):
        source = Path(__file__).resolve().parents[2]
        names = (
            "relax/agentic/runner/ipc.py",
            "relax/agentic/session/service.py",
            "relax/agentic/pipeline/__init__.py",
        )
        save(
            self.evidence / "process-ownership.json",
            dict(
                source_root=str(source),
                source_sha256={name: hashlib.sha256((source / name).read_bytes()).hexdigest() for name in names},
                owned_temporary_root=str(self.owned),
                daemon=self.daemon,
                receipt_directories=[str(self.receipts / sid) for sid in self.cases],
                ownership_proofs=list(self.proofs.values()),
            ),
        )

    async def start(self):
        self.fd_before = fd_targets()
        self.other_tasks = {task for task in asyncio.all_tasks() if task is not asyncio.current_task()}
        self.receipts.mkdir(parents=True)
        for sid in self.cases:
            (self.receipts / sid).mkdir()
        worker = self.owned / "worker.py"
        worker.write_text(WORKER, encoding="utf-8")
        argv = [
            sys.executable,
            "-c",
            "from relax.agentic.runner.ipc import _main; _main()",
            "--launcher-socket",
            str(self.socket),
        ]
        self.ledger()
        with (self.receipts / "launcher.log").open("w") as log:
            self.daemon_process = subprocess.Popen(
                argv,
                cwd=Path(__file__).resolve().parents[2],
                env=os.environ.copy(),
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        self.daemon = dict(identity=identity(self.daemon_process.pid), argv=argv)
        assert self.daemon["identity"]["pid"] == self.daemon["identity"]["pgid"] == self.daemon["identity"]["sid"]
        self.ledger()

        def ready():
            assert self.daemon_process.poll() is None
            try:
                _ping_launcher(str(self.socket))
                return True
            except OSError:
                return False

        await until(ready, "Own launcher readiness")
        launcher = object.__new__(ManagedAgentLauncher)
        launcher._launcher = LauncherClient(str(self.socket))
        launcher._app = ManagedCommandAppSpec(
            command="exec " + shlex.quote(sys.executable) + " " + shlex.quote(str(worker)),
            cwd=str(self.owned),
            env={"TASK14_FAULT_RECEIPTS": str(self.receipts)},
            timeout_s=180,
        )
        self.shard._agent_launcher = launcher
        native = self.shard._build_session_export_transport

        def builder(**kwargs):
            self.builder_sessions.append(kwargs["session"].session_id)
            try:
                return native(**kwargs)
            except ValueError as error:
                self.builder_errors.append(
                    dict(
                        session_id=kwargs["session"].session_id,
                        message=str(error),
                        frames=[
                            dict(file=frame.filename, function=frame.name)
                            for frame in traceback.extract_tb(error.__traceback__)
                        ],
                    )
                )
                raise

        self.shard._build_session_export_transport = builder
        for sid, case in self.cases.items():
            await self.shard.start_group(
                "train",
                sid,
                (SessionSpec(session_id=sid, index=case.row, group_index=0, train_metadata={"source_row": case.row}),),
            )
            session = self.shard._session_records[sid]
            self.sessions[sid] = session
            self.resources[sid] = session.resources
            native_publish = session.result_cell.publish

            def publish(value, sid=sid, session=session, native_publish=native_publish):
                assert session.forest is None and session.resources is None
                assert not session.live_irs and not session.queued_irs and not session.resp_state_hash_by_request_id
                assert sid not in self.shard._session_records and self.resources[sid].process_wait.done()
                self.publications[sid].append(type(value).__name__)
                assert len(self.publications[sid]) == 1
                native_publish(value)

            session.result_cell.publish = publish
            await until(lambda: (self.receipts / sid / "ready.json").exists(), "Native worker readiness")
            self.proofs[sid] = json.loads((self.receipts / sid / "ready.json").read_text())
            self.ledger()

    async def prepare(self, exclude=()):
        for sid, case in self.cases.items():
            if sid not in exclude:
                for turn in range(len(case.assistant_positions) - 1):
                    await replay_turn(self.shard, sid, case, turn, first=turn == 0)

    async def final(self, sid):
        case = self.cases[sid]
        await replay_turn(self.shard, sid, case, len(case.assistant_positions) - 1, first=False)

    def chat(self, sid):
        case = self.cases[sid]
        return self.shard.chat(
            session_id=sid,
            messages=copy.deepcopy(case.messages[: case.assistant_positions[-1]]),
            tools=[],
            chat_template_kwargs={"enable_thinking": False},
            max_completion_tokens=len(case.response_tokens[-1]),
            stop=None,
            seed=None,
            logprobs=False,
        )

    def finish(self, *session_ids):
        for sid in session_ids:
            (self.receipts / sid / "go").write_text("complete", encoding="utf-8")

    def hold_close(self, sid):
        entered, release = asyncio.Event(), asyncio.Event()
        self.gates.append(release)
        native = self.backend.close_session

        async def close(session_id, **kwargs):
            if session_id == sid:
                entered.set()
                await release.wait()
            return await native(session_id, **kwargs)

        self.backend.close_session = close
        return entered, release

    async def result(self, sid):
        await asyncio.shield(self.resources[sid].watcher_task)
        (value,) = await self.shard.take_session_results(sid, (sid,))
        assert isinstance(value, SessionExportTransport)
        sample = TrainingFieldArtifact(sample_payload=value.exports[0]["sample_payload"]).to_sample()
        assert sample.index == self.cases[sid].row
        digest = training_digest(value)
        self.training_hashes[sid] = digest
        await self.shard.release_group(sid)
        return digest

    async def close(self, *, verify=True):
        for gate in self.gates:
            gate.set()
        try:
            await asyncio.wait_for(self.shard.shutdown(), 30)
            await asyncio.gather(*self.tasks, return_exceptions=True)
            await asyncio.gather(*(r.watcher_task for r in self.resources.values()), return_exceptions=True)
            if not verify:
                return
            for sid, session in self.sessions.items():
                assert session.forest is None and session.resources is None and session.finish_task.done()
                assert not session.live_irs and not session.queued_irs and not session.resp_state_hash_by_request_id
                assert len(self.publications[sid]) == 1
                assert self.resources[sid].process_wait.done()
                for key in ("worker", "child"):
                    assert not same_process(self.proofs[sid][key])
                assert not Path(self.proofs[sid]["session_dir"]).exists()
            assert not self.backend.tasks and self.backend.inflight == 0
            assert Counter(self.backend.closed) == Counter(self.cases.keys())
            assert not self.shard._groups and not self.shard._session_records
            assert not self.shard._permit_cleanup_tasks and not self.shard._sglang_request_limiter.owned
        finally:
            self.executor.shutdown(wait=True)
            if hasattr(self, "daemon_process"):
                self.daemon_process.terminate()
                try:
                    await asyncio.to_thread(self.daemon_process.wait, 10)
                except subprocess.TimeoutExpired:
                    self.daemon_process.kill()
                    await asyncio.to_thread(self.daemon_process.wait, 5)
                    raise AssertionError("Owned launcher required forced recovery; not clean lifecycle evidence")
                assert not same_process(self.daemon["identity"])
                self.socket.unlink(missing_ok=True)
        (self.owned / "worker.py").unlink()
        self.owned.rmdir()
        await asyncio.sleep(0)
        assert not {
            task
            for task in asyncio.all_tasks()
            if task is not asyncio.current_task() and task not in self.other_tasks and not task.done()
        }
        assert fd_targets() == self.fd_before
        save(
            self.receipts / "evidence.json",
            dict(
                label=self.label,
                cases=[case.describe() for case in self.cases.values()],
                events=self.events,
                publications=self.publications,
                builder_sessions=self.builder_sessions,
                builder_errors=self.builder_errors,
                native_process_proofs=self.proofs,
                daemon_reaped=True,
                training_sha256_by_session=self.training_hashes,
                input_selection=self.selection,
                native_groups_and_session_dirs_released=True,
                fd_targets_restored=True,
                remaining_permits=0,
                remaining_backend_work=0,
                scope="Real HF text/Session/builder/native process lifecycle; mock inference and local permit RPC shim",
            ),
        )
        self.daemon = None
        self.proofs = {}
        save(
            self.evidence / "process-ownership.json",
            dict(
                source_root=str(Path(__file__).resolve().parents[2]),
                owned_temporary_root=str(self.owned),
                daemon=None,
                receipt_directories=[],
                ownership_proofs=[],
            ),
        )


@asynccontextmanager
async def session_run(inputs, label):
    run = SessionRun(inputs, label)
    try:
        await run.start()
        yield run
    except BaseException:
        # Preserve the originating failure if startup only acquired a subset of
        # the four Sessions. The outer owner also has the persisted ledger.
        try:
            await run.close(verify=False)
        except BaseException as error:
            save(run.receipts / "teardown-error.json", dict(error=repr(error)))
        raise
    else:
        await run.close()


async def control(inputs, label):
    async with session_run(inputs, label + "-control") as run:
        await run.prepare()
        await asyncio.gather(*(run.final(sid) for sid in run.cases))
        run.finish(*run.cases)
        return {sid: await run.result(sid) for sid in run.cases}


async def test_long_session_watcher_cancel_preserves_other_exports(real_inputs):
    expected = await control(real_inputs, "watcher-cancel")
    async with session_run(real_inputs, "watcher-cancel") as run:
        await run.prepare()
        await asyncio.gather(*(run.final(sid) for sid in run.cases))
        entered, release = run.hold_close("fault-0")
        run.finish(*run.cases)
        await asyncio.wait_for(entered.wait(), 20)
        watcher = run.resources["fault-0"].watcher_task
        watcher.cancel()
        for sid in ("fault-1", "fault-2", "fault-3"):
            assert await run.result(sid) == expected[sid]
        assert not watcher.done() and run.sessions["fault-0"].finish_task is not None
        run.events.append("Three peer long exports complete while cancelled watcher waits for native cleanup")
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(watcher, 20)
        (value,) = await run.shard.take_session_results("fault-0", ("fault-0",))
        assert isinstance(value, SessionExportTransport) and training_digest(value) == expected["fault-0"]
        run.training_hashes["fault-0"] = training_digest(value)
        await run.shard.release_group("fault-0")


async def test_real_logprob_conversion_failure_isolated_from_peer_exports(real_inputs):
    expected = await control(real_inputs, "conversion-failure")
    async with session_run(real_inputs, "conversion-failure") as run:
        await run.prepare()
        await asyncio.gather(*(run.final(sid) for sid in run.cases))
        response = next(
            node
            for node in run.sessions["fault-0"].forest.nodes_by_hash.values()
            if node.kind == "resp" and node.logprob_delta
        )
        response.logprob_delta[0] = "task14-invalid-logprob"
        peers = ("fault-1", "fault-2", "fault-3")
        gates = [run.hold_close(sid) for sid in peers]
        run.finish(*peers)
        await asyncio.wait_for(asyncio.gather(*(entered.wait() for entered, _ in gates)), 20)
        assert all(
            run.sessions[sid].finish_task is not None and not run.sessions[sid].finish_task.done() for sid in peers
        )
        run.finish("fault-0")
        await asyncio.wait_for(run.resources["fault-0"].watcher_task, 20)
        error = run.sessions["fault-0"].result_cell.payload
        assert isinstance(error, RuntimeGroupError) and "task14-invalid-logprob" in str(error)
        assert len(run.builder_errors) == 1 and run.builder_errors[0]["session_id"] == "fault-0"
        assert any(frame["function"] == "_compact_sample_payload" for frame in run.builder_errors[0]["frames"])
        assert all(not run.sessions[sid].finish_task.done() for sid in peers)
        run.events.append("Actual compactor conversion failed while three peer long exports remained finalizing")
        for _, release in gates:
            release.set()
        for sid in peers:
            assert await run.result(sid) == expected[sid]
        await run.shard.drop_group("fault-0")
        run.events.append("Injected malformed logprob reached actual compactor; three peer long exports match control")


async def test_shutdown_mixed_finalizing_inflight_and_queued_long_sessions(real_inputs):
    expected = await control(real_inputs, "mixed-shutdown")
    async with session_run(real_inputs, "mixed-shutdown") as run:
        await run.prepare(exclude=("fault-2",))
        entered, release = run.hold_close("fault-0")
        await run.final("fault-0")
        run.finish("fault-0")
        await asyncio.wait_for(entered.wait(), 20)
        backend_entered = asyncio.Event()

        class HeldCohort:
            async def backend_enter(self, session_id):
                backend_entered.set()
                await asyncio.Event().wait()

        run.backend.cohorts["fault-1"] = HeldCohort()
        inflight = asyncio.create_task(run.chat("fault-1"))
        queued = asyncio.create_task(run.chat("fault-2"))
        run.tasks.extend((inflight, queued))
        await asyncio.wait_for(backend_entered.wait(), 20)
        await until(lambda: run.shard._groups["fault-2"].first_request_barrier_crossed, "PREPARE queue barrier")
        assert run.sessions["fault-2"].queued_irs and run.shard._groups["fault-2"].ownership is GroupOwnership.PREPARE
        assert any(ir.backend_started for ir in run.sessions["fault-1"].live_irs)
        assert run.sessions["fault-0"].finish_task is not None
        await run.final("fault-3")
        run.finish("fault-3")
        assert await run.result("fault-3") == expected["fault-3"]
        run.events.append(
            "Same shard contains finalizing, real backend-inflight, PREPARE queued long Sessions; peer completed"
        )
        shutdown = asyncio.create_task(run.shard.shutdown())
        run.tasks.append(shutdown)
        await until(lambda: not run.shard._train_generation_open, "Actual shutdown entry")
        assert not run.sessions["fault-0"].finish_task.done()
        assert any(ir.backend_started for ir in run.sessions["fault-1"].live_irs)
        assert run.sessions["fault-2"].queued_irs and run.shard._groups["fault-2"].ownership is GroupOwnership.PREPARE
        run.events.append("Shutdown entered with finalizing A still blocked, backend B active and C queued")
        release.set()
        await asyncio.wait_for(shutdown, 30)
        for pending in (inflight, queued):
            result = await pending
            assert result["_http_status"] == 404 and result["error"]["code"] == "session_discarded"
        # An already-owned finish_task preserves its successful outcome even if
        # shutdown removes the containing group. Do not assert all Sessions drop.
        assert isinstance(run.sessions["fault-0"].result_cell.payload, SessionExportTransport)
        assert training_digest(run.sessions["fault-0"].result_cell.payload) == expected["fault-0"]
        run.training_hashes["fault-0"] = training_digest(run.sessions["fault-0"].result_cell.payload)
        assert run.sessions["fault-1"].result_cell.dropped and run.sessions["fault-2"].result_cell.dropped
