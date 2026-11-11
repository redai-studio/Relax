# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Real Unix launcher and cleanup, with only receipt availability and Ray
transport controlled.

Existing owner tests check one confirmation; these regressions exercise retries
through Session ownership and physical capacity. Each case starts one short CPU
process. The clock override changes only IPC's cleanup clock, not asyncio time.
"""

import asyncio
import json
import shlex
import sys
from argparse import Namespace
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from relax.agentic.observability import ShardTrajectoryRecorder
from relax.agentic.pipeline import RuntimeGroupError
from relax.agentic.pipeline.runtime import RuntimeDomain, RuntimeGroupStream, _SessionResultRef, _SessionShardBinding
from relax.agentic.runner import ipc
from relax.agentic.runner.resources import ReceiptResourceOwner
from relax.agentic.session.service import (
    AgenticSessionShard,
    GroupOwnership,
    ResidentGroup,
    SessionResources,
    _SessionRecord,
    _SessionResultCell,
)


pytestmark = [pytest.mark.asyncio, pytest.mark.skipif(sys.platform != "linux", reason="Linux process groups")]


class RetryReceiptOwner(ReceiptResourceOwner):
    state = None

    def __init__(self, lease):
        super().__init__(lease)
        self.state.owner = self

    def confirm_release(self):
        receipt = super().confirm_release()
        assert receipt is not None and receipt.matches(self.lease)
        self.state.confirmations += 1
        return receipt if self.state.available else None


@asynccontextmanager
async def receipt_process(tmp_path, monkeypatch):
    state = SimpleNamespace(available=False, owner=None, confirmations=0)
    monkeypatch.setattr(RetryReceiptOwner, "state", state)
    handlers = set()

    async def handle(reader, writer):
        task = asyncio.current_task()
        handlers.add(task)
        try:
            await ipc._handle_launcher_client(reader, writer)
        finally:
            handlers.discard(task)

    socket_path = str(tmp_path / "launcher.sock")
    server = await asyncio.start_unix_server(handle, socket_path)
    command = (
        "import json,os;from pathlib import Path;e=os.environ;"
        "Path(e['RELAX_RESOURCE_RELEASE_PATH']).write_text(json.dumps(dict("
        "schema=1,session_id=e['RELAX_SESSION_ID'],nonce=e['RELAX_RESOURCE_RELEASE_NONCE'],status='released')))"
    )
    process = ipc.ManagedAgentProcess(
        ipc.LauncherClient(socket_path),
        ipc.ManagedCommandAppSpec(
            command=shlex.join([sys.executable, "-c", command]),
            cwd=str(tmp_path),
            resource_owner_path=f"{__name__}.RetryReceiptOwner",
        ),
        ipc.SessionInput("session", "train", "group"),
        require_resource_release=True,
    )
    try:
        await asyncio.wait_for(process.wait(), 5)
        assert state.confirmations == 1 and not process.resource_released
        yield process, state
    finally:
        state.available = True
        await asyncio.wait_for(process.terminate_and_join(), 2)
        server.close()
        await server.wait_closed()
        if handlers:
            await asyncio.wait_for(asyncio.gather(*handlers), 2)


async def test_unknown_confirmation_retries_original_lease(tmp_path, monkeypatch):
    async with receipt_process(tmp_path, monkeypatch) as (process, state):
        lease = state.owner.lease
        assert lease.release_path.exists()
        original_proof = json.loads(lease.release_path.read_text())
        state.available = True
        await process.terminate_and_join()
        assert process.resource_released and state.confirmations == 2
        assert original_proof["nonce"] == lease.nonce and original_proof["session_id"] == lease.session_id
        assert not lease.release_path.parent.exists()
        await process.terminate_and_join()
        assert state.confirmations == 2


async def test_confirmation_retry_does_not_extend_cleanup_deadline(tmp_path, monkeypatch):
    async with receipt_process(tmp_path, monkeypatch) as (process, state):
        deadline = process._cleanup_deadline
        assert deadline is not None
        clock = SimpleNamespace(now=deadline - 1)
        monkeypatch.setattr(ipc, "time", SimpleNamespace(monotonic=lambda: clock.now))
        await process.terminate_and_join()
        assert process._cleanup_deadline == deadline and not process.resource_released
        assert state.owner.lease.release_path.exists()
        confirmations = state.confirmations

        # The valid proof was present throughout, but cannot grant credit after expiry.
        clock.now = deadline + 1
        state.available = True
        await process.terminate_and_join()
        await process.terminate_and_join()
        assert process._cleanup_deadline == deadline and process._cleanup_timed_out
        assert not process.resource_released and state.confirmations == confirmations
        assert not state.owner.lease.release_path.parent.exists()


async def test_drop_retry_retains_session_and_capacity_until_confirmation(tmp_path, monkeypatch):
    async with receipt_process(tmp_path, monkeypatch) as (process, state):
        args = Namespace(agentic_session_capacity_per_node=1, agentic_session_lifecycle=False)
        shard = object.__new__(AgenticSessionShard.__ray_metadata__.modified_class)
        shard.args = args
        shard._trajectory_recorder = ShardTrajectoryRecorder(args, shard_ordinal=0)
        shard._groups = {}
        shard._session_records = {}
        shard._resource_release_receipts = {}
        shard._state_revision = 0
        shard._state_changed = asyncio.Event()
        shard._eager_gate_lock = asyncio.Lock()
        shard._eager_shutdown = False

        class Handle:
            def __getattr__(self, name):
                return SimpleNamespace(remote=lambda *a, **kw: asyncio.create_task(getattr(shard, name)(*a, **kw)))

        binding = _SessionShardBinding("owner", Handle(), node_id="node")
        runtime = RuntimeDomain(args, "train", [binding], concurrency=1)
        cell = _SessionResultCell()
        group = ResidentGroup(
            "train",
            "group",
            {"session": cell},
            [],
            ownership=GroupOwnership.RUNTIME,
            requires_resource_release=True,
            eager_session_generation=True,
        )
        session = _SessionRecord(group, "session", {}, cell, process_launch_attempted=True)
        process_wait = asyncio.create_task(process.wait())
        watcher = asyncio.create_task(asyncio.sleep(0))
        await asyncio.gather(process_wait, watcher)
        session.resources = SessionResources(process, process_wait, watcher)
        group.sessions.append(session)
        shard._groups[group.group_id] = group
        shard._session_records[session.session_id] = session
        await runtime._resident_group_permits.acquire()
        reservation = await runtime._reserve_group_shard(group.group_id, None, 1)
        runtime._session_capacity.bind(reservation.session_capacity, (session.session_id,))
        reservation.session_capacity.submitted = True
        loop = asyncio.get_running_loop()
        stream = RuntimeGroupStream(
            group.group_id,
            binding,
            loop.create_future(),
            (_SessionResultRef(session.session_id, loop.create_future()),),
            eager_session_generation=True,
        )
        binding.group_streams[group.group_id] = stream
        try:
            with pytest.raises(RuntimeError):
                await runtime.drop_group(stream)
            assert shard._session_records[session.session_id] is session
            assert session.resources.process is process and not cell.terminal
            assert runtime.session_capacity_snapshot()["outstanding"] == 1
            assert runtime.session_capacity_snapshot()["confirmed_releases"] == 0
            assert state.owner.lease.release_path.exists()

            state.available = True
            # Retry completes physical cleanup while preserving the original fatal outcome.
            with pytest.raises(RuntimeGroupError, match="resource release unconfirmed"):
                await runtime.drop_group(stream)
            assert process.resource_released and not shard._session_records
            assert not shard._groups and not state.owner.lease.release_path.parent.exists()
            assert runtime.session_capacity_snapshot()["outstanding"] == 1
            await runtime.drop_group(stream)  # Replay the committed release after the error response.
            assert runtime.session_capacity_snapshot()["outstanding"] == 0
            assert runtime.session_capacity_snapshot()["confirmed_releases"] == 1
            assert not runtime._reserved_group_shards and not binding.group_streams
            await runtime.drop_group(stream)
            assert runtime.session_capacity_snapshot()["confirmed_releases"] == 1
        finally:
            await shard._trajectory_recorder.aclose()
            await runtime._inventory.aclose()
