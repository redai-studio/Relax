# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Linux subreaper regression with real local processes and Unix transport."""

import asyncio
import os
import shlex
import signal
import sys
from pathlib import Path

import pytest

from relax.agentic.runner import ipc


pytestmark = [pytest.mark.asyncio, pytest.mark.skipif(sys.platform != "linux", reason="requires Linux subreaper")]


async def eventually(predicate):
    async with asyncio.timeout(10):
        while not predicate():
            await asyncio.sleep(0.01)


@pytest.mark.parametrize("exit_mode", ["normal", "terminate", "nonzero"])
async def test_root_exit_returns_before_detached_descendant_cleanup(tmp_path, exit_mode):
    ready, release = tmp_path / "ready", tmp_path / "release"
    child_code = f"""import os, signal, time
from pathlib import Path
signal.signal(signal.SIGTERM, signal.SIG_IGN)
Path({str(ready)!r}).write_text(str(os.getpid()))
while not Path({str(release)!r}).exists():
    time.sleep(0.01)
"""
    root_code = (
        "import json,os,signal,subprocess,sys,time;from pathlib import Path;"
        f"subprocess.Popen([sys.executable,'-c',{child_code!r}],start_new_session=True);"
        "signal.signal(signal.SIGTERM,lambda *_:sys.exit(0));"
        f"\nwhile not Path({str(ready)!r}).exists(): time.sleep(0.01)\n"
        "Path(os.environ['RELAX_OUTPUT_JSON']).write_text(json.dumps({'reward':1}));"
    )
    root_code += (
        "\nwhile True: time.sleep(0.01)\n"
        if exit_mode == "terminate"
        else (f"sys.exit({7 if exit_mode == 'nonzero' else 0})")
    )
    handlers = set()

    async def handle(reader, writer):
        task = asyncio.current_task()
        handlers.add(task)
        try:
            await ipc._handle_launcher_client(reader, writer)
        finally:
            handlers.discard(task)

    socket_path = str(tmp_path / "launcher.sock")
    server = await asyncio.start_unix_server(handle, path=socket_path)
    process = ipc.ManagedAgentProcess(
        ipc.LauncherClient(socket_path),
        ipc.ManagedCommandAppSpec(
            command=f"{shlex.quote(sys.executable)} -c {shlex.quote(root_code)}", cwd=str(tmp_path)
        ),
        ipc.SessionInput(session_id="s", group_id="g", rollout_mode="train"),
    )
    child_pid = None
    try:
        await eventually(ready.exists)
        child_pid = int(ready.read_text())
        if exit_mode == "terminate":
            await asyncio.wait_for(process.terminate_and_join(), 10)
        elif exit_mode == "nonzero":
            with pytest.raises(ipc.AgentExecutionError) as caught:
                await asyncio.wait_for(process.wait(), 10)
            assert caught.value.reason_code == "agent_exit_nonzero"
        else:
            await asyncio.wait_for(process.wait(), 10)
            assert process.output.reward == 1
        os.kill(child_pid, 0)
        assert not release.exists()
        release.touch()
        await eventually(lambda: not Path(f"/proc/{child_pid}").exists())
        await eventually(lambda: not handlers)
    finally:
        release.touch()
        server.close()
        await server.wait_closed()
        if child_pid is not None and Path(f"/proc/{child_pid}").exists():
            os.kill(child_pid, signal.SIGKILL)
        if handlers:
            await asyncio.wait(tuple(handlers), timeout=10)
