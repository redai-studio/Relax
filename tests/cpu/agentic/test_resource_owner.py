# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Resource scope adapters must never manufacture physical Session credit."""

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from relax.agentic.runner.ipc import ManagedAgentProcess, ManagedCommandAppSpec, SessionInput
from relax.agentic.runner.resources import (
    DEFAULT_RESOURCE_OWNER_PATH,
    ProcessGroupResourceOwner,
    ReceiptResourceOwner,
    ResourceLease,
    ResourceReleaseReceipt,
    SessionResourceOwner,
    load_resource_owner_type,
    read_release_receipt,
)


RECEIPT_OWNER = "relax.agentic.runner.resources.ReceiptResourceOwner"


class MarkerResourceOwner(SessionResourceOwner):
    """An unrelated sandbox adapter using its own local close
    acknowledgement."""

    requires_external_cleanup = True

    def launch_environment(self):
        return {
            **super().launch_environment(),
            "SANDBOX_CLOSED_MARKER": str(self.lease.release_path.with_suffix(".closed")),
        }

    def confirm_release(self):
        marker = self.lease.release_path.with_suffix(".closed")
        if marker.exists() and marker.read_text() == self.lease.nonce:
            return ResourceReleaseReceipt(self.lease.session_id, self.lease.nonce)
        return None


class BrokenConstructorOwner(ProcessGroupResourceOwner):
    def __init__(self, lease):
        raise RuntimeError("invalid sandbox configuration")


class BrokenVerifierOwner(ProcessGroupResourceOwner):
    def confirm_release(self):
        raise RuntimeError("sandbox close state unavailable")

    def diagnostics(self):
        raise RuntimeError("diagnostics unavailable")


class InvalidVerifierOwner(ProcessGroupResourceOwner):
    def confirm_release(self):
        return True


class AsyncVerifierOwner(ProcessGroupResourceOwner):
    async def confirm_release(self):
        return ResourceReleaseReceipt(self.lease.session_id, self.lease.nonce)


class WrongLeaseOwner(ProcessGroupResourceOwner):
    def confirm_release(self):
        return ResourceReleaseReceipt(self.lease.session_id, "another-launch")


class InvalidEnvironmentOwner(ProcessGroupResourceOwner):
    def launch_environment(self):
        return {"RELAX_SESSION_ID": "another-session"}


def owner_path(owner_type):
    return f"{owner_type.__module__}.{owner_type.__name__}"


def lease(tmp_path):
    return ResourceLease("session-1", "group-1", "attempt-1", tmp_path / "released.json")


def test_receipt_identity_and_repeated_confirmation_are_stable(tmp_path):
    identity = lease(tmp_path)
    receipt = ResourceReleaseReceipt(identity.session_id, identity.nonce)
    identity.release_path.write_text(json.dumps(receipt.to_payload()))
    owner = ReceiptResourceOwner(identity)
    assert owner.confirm_release() == owner.confirm_release() == receipt
    assert not receipt.matches(ResourceLease("session-2", identity.group_id, identity.nonce, identity.release_path))
    assert not receipt.matches(
        ResourceLease(identity.session_id, identity.group_id, "next-attempt", identity.release_path)
    )


@pytest.mark.parametrize(
    "malformation", ["missing", "partial", "oversized", "unknown", "schema_bool", "symlink", "fifo"]
)
def test_invalid_receipts_fail_closed_without_waiting_for_a_writer(tmp_path, malformation):
    identity = lease(tmp_path)
    payload = ResourceReleaseReceipt(identity.session_id, identity.nonce).to_payload()
    if malformation == "partial":
        identity.release_path.write_text("{")
    elif malformation == "oversized":
        identity.release_path.write_text(" " * 4097)
    elif malformation in {"unknown", "schema_bool"}:
        payload["status" if malformation == "unknown" else "schema"] = "unknown" if malformation == "unknown" else True
        identity.release_path.write_text(json.dumps(payload))
    elif malformation == "symlink":
        target = tmp_path / "other.json"
        target.write_text(json.dumps(payload))
        identity.release_path.symlink_to(target)
    elif malformation == "fifo":
        os.mkfifo(identity.release_path)
    started = time.monotonic()
    assert read_release_receipt(identity.release_path) is None
    assert time.monotonic() - started < 1


def test_custom_owner_is_loaded_explicitly_and_async_verifiers_are_rejected():
    assert load_resource_owner_type(owner_path(MarkerResourceOwner)) is MarkerResourceOwner
    with pytest.raises(TypeError, match="synchronous"):
        load_resource_owner_type(owner_path(AsyncVerifierOwner))
    with pytest.raises(TypeError, match="subclass"):
        load_resource_owner_type("pathlib.Path")


def test_owner_type_is_validated_before_session_admission(tmp_path):
    with pytest.raises(TypeError, match="subclass"):
        ManagedCommandAppSpec(command="exit 0", cwd=str(tmp_path), resource_owner_path="pathlib.Path")


async def run_managed(tmp_path, *, path=DEFAULT_RESOURCE_OWNER_PATH, group_closed=True, on_launch=None, on_close=None):
    handle = SimpleNamespace(
        group_closed=group_closed,
        wait=AsyncMock(return_value=0),
        terminate=AsyncMock(),
        close=AsyncMock(side_effect=on_close),
    )
    environments = []

    async def launch(**kwargs):
        env = kwargs["env"]
        environments.append(env)
        Path(kwargs["log_path"]).write_text("test command completed\n")
        Path(env["RELAX_OUTPUT_JSON"]).write_text('{"metadata":{"completed":true}}')
        if on_launch is not None:
            on_launch(env)
        return handle

    process = ManagedAgentProcess(
        SimpleNamespace(launch=launch),
        ManagedCommandAppSpec(command="unused", cwd=str(tmp_path), resource_owner_path=path),
        SessionInput("session-1", "train", "group-1"),
        require_resource_release=True,
    )
    return process, handle, environments


@pytest.mark.asyncio
@pytest.mark.parametrize("closed", [True, False, None])
async def test_process_scope_returns_credit_only_after_confirmed_group_close(tmp_path, closed):
    process, handle, environments = await run_managed(tmp_path, group_closed=closed)
    await process.wait()
    assert process.resource_released is (closed is True)
    assert environments[0]["RELAX_REQUIRE_RESOURCE_RELEASE"] == "0"
    handle.close.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("proof", ["valid", "missing", "wrong_nonce", "wrong_session"])
async def test_external_owner_requires_exact_release_proof(tmp_path, proof):
    def publish(env):
        if proof == "missing":
            return
        receipt = ResourceReleaseReceipt(
            "wrong-session" if proof == "wrong_session" else env["RELAX_SESSION_ID"],
            "wrong-nonce" if proof == "wrong_nonce" else env["RELAX_RESOURCE_RELEASE_NONCE"],
        )
        Path(env["RELAX_RESOURCE_RELEASE_PATH"]).write_text(json.dumps(receipt.to_payload()))

    process, handle, environments = await run_managed(tmp_path, path=RECEIPT_OWNER, on_launch=publish)
    await process.wait()
    assert process.resource_released is (proof == "valid")
    assert environments[0]["RELAX_REQUIRE_RESOURCE_RELEASE"] == "1"
    handle.close.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("publish,closed", [(True, True), (False, True), (True, False)])
async def test_unrelated_sandbox_owner_uses_its_own_release_contract(tmp_path, publish, closed):
    def release_sandbox(env):
        if publish:
            Path(env["SANDBOX_CLOSED_MARKER"]).write_text(env["RELAX_RESOURCE_RELEASE_NONCE"])

    process, handle, _ = await run_managed(
        tmp_path, path=owner_path(MarkerResourceOwner), group_closed=closed, on_launch=release_sandbox
    )
    await process.wait()
    assert process.resource_released is (publish and closed)
    handle.close.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("owner", [BrokenVerifierOwner, InvalidVerifierOwner, WrongLeaseOwner])
async def test_invalid_owner_confirmation_still_closes_process_without_credit(tmp_path, owner):
    process, handle, _ = await run_managed(tmp_path, path=owner_path(owner))
    await process.wait()
    assert not process.resource_released
    handle.close.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "owner,error", [(BrokenConstructorOwner, RuntimeError), (InvalidEnvironmentOwner, ValueError)]
)
async def test_owner_initialization_failure_returns_only_unspent_launch_credit(tmp_path, owner, error):
    process, handle, environments = await run_managed(tmp_path, path=owner_path(owner))
    with pytest.raises(error):
        await process.wait()
    await asyncio.wait_for(process.terminate_and_join(), 1)
    assert environments == [] and process.resource_released
    handle.close.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["payload_copy", "temporary_directory", "serialization", "input_write"])
async def test_local_preparation_failure_returns_unspent_launch_credit(tmp_path, monkeypatch, failure):
    from relax.agentic.runner import ipc

    class UncopyablePayload:
        def __deepcopy__(self, memo):
            raise ValueError("payload copy failed")

    payload = {}
    if failure == "payload_copy":
        payload["input"] = UncopyablePayload()
        expected_error, message = ValueError, "payload copy failed"
    elif failure == "temporary_directory":

        def fail_mkdtemp(*args, **kwargs):
            raise OSError("temporary directory creation failed")

        monkeypatch.setattr(ipc.tempfile, "mkdtemp", fail_mkdtemp)
        expected_error, message = OSError, "temporary directory creation failed"
    elif failure == "serialization":
        payload["input"] = object()
        expected_error, message = TypeError, "not JSON serializable"
    else:
        write_text = Path.write_text

        def fail_input_write(path, *args, **kwargs):
            if path.name == "session_input.json":
                raise OSError("session input write failed")
            return write_text(path, *args, **kwargs)

        monkeypatch.setattr(Path, "write_text", fail_input_write)
        expected_error, message = OSError, "session input write failed"

    launch = AsyncMock()
    # Even payload copying must fail inside the owned task so Session retains
    # a handle whose cleanup can prove that no launch consumed its credit.
    process = ManagedAgentProcess(
        SimpleNamespace(launch=launch),
        ManagedCommandAppSpec(command="unused", cwd=str(tmp_path)),
        SessionInput("s", "train", "g", input_payload=payload),
        require_resource_release=True,
    )
    with pytest.raises(expected_error, match=message):
        await process.wait()
    await asyncio.wait_for(process.terminate_and_join(), 1)
    await asyncio.wait_for(process.terminate_and_join(), 1)
    assert process.resource_released
    launch.assert_not_awaited()


@pytest.mark.asyncio
async def test_lost_launch_rpc_response_does_not_refund_uncertain_resources(tmp_path):
    launch = AsyncMock(side_effect=ConnectionError("launch reply lost"))
    process = ManagedAgentProcess(
        SimpleNamespace(launch=launch),
        ManagedCommandAppSpec(command="unused", cwd=str(tmp_path)),
        SessionInput("s", "train", "g"),
        require_resource_release=True,
    )
    with pytest.raises(ConnectionError, match="reply lost"):
        await process.wait()
    await asyncio.wait_for(process.terminate_and_join(), 1)
    assert not process.resource_released
    launch.assert_awaited_once()


@pytest.mark.asyncio
async def test_close_failure_invalidates_otherwise_valid_release(tmp_path):
    process, handle, _ = await run_managed(tmp_path, on_close=RuntimeError("connection close failed"))
    with pytest.raises(RuntimeError, match="connection close"):
        await process.wait()
    assert not process.resource_released
    handle.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_late_capability_close_cannot_publish_credit(tmp_path):
    release_close = asyncio.Event()
    close_started = asyncio.Event()

    async def close():
        close_started.set()
        await release_close.wait()

    process, handle, _ = await run_managed(tmp_path, on_close=close)
    await close_started.wait()
    process._cleanup_deadline = time.monotonic() - 1
    release_close.set()
    await process.wait()
    assert not process.resource_released and process._cleanup_timed_out
    handle.close.assert_awaited_once()


@pytest.mark.skipif(sys.platform != "linux", reason="The managed process launcher owns Linux process groups")
@pytest.mark.asyncio
async def test_real_local_command_needs_no_case_specific_release_file(tmp_path):
    from relax.agentic.runner import ipc

    socket_path = str(tmp_path / "launcher.sock")
    handlers = set()

    async def handle(reader, writer):
        task = asyncio.current_task()
        handlers.add(task)
        try:
            await ipc._handle_launcher_client(reader, writer)
        finally:
            handlers.discard(task)

    server = await asyncio.start_unix_server(handle, socket_path)
    try:
        process = ManagedAgentProcess(
            ipc.LauncherClient(socket_path),
            ManagedCommandAppSpec(command="exit 0", cwd=str(tmp_path)),
            SessionInput("local-session", "train", "local-group"),
            require_resource_release=True,
        )
        await asyncio.wait_for(process.wait(), 10)
        assert process.resource_released
        await process.terminate_and_join()
    finally:
        server.close()
        await server.wait_closed()
        if handlers:
            await asyncio.wait_for(asyncio.gather(*handlers), 5)
