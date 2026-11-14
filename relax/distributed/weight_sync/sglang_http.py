# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Weight-version control endpoint for an opt-in SGLang HTTP server."""

import asyncio
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

from .admission import AdmissionGate, AdmissionMiddleware, InstallationFence


def install_http_extension() -> None:
    import zmq
    from fastapi import HTTPException, Request
    from fastapi.responses import JSONResponse
    from sglang.srt.entrypoints import http_server
    from sglang.srt.managers.io_struct import GenerateReqInput, RpcReqInput, RpcReqOutput, sock_recv, sock_send
    from sglang.srt.managers.tokenizer_manager import TokenizerManager
    from sglang.srt.utils.auth import AuthLevel, auth_level
    from sglang.srt.utils.network import get_zmq_socket

    if getattr(http_server.app.state, "relax_weight_versions", False):
        return
    http_server.app.state.relax_weight_versions = True
    gate = AdmissionGate()
    fence = InstallationFence()
    control = asyncio.Lock()
    http_server.app.add_middleware(AdmissionMiddleware, gate=gate, fence=fence, control=control)
    original = TokenizerManager.init_ipc_channels

    def initialize(manager, port_args):
        original(manager, port_args)
        manager._relax_rpc_endpoint = port_args.rpc_ipc_name

    TokenizerManager.init_ipc_channels = initialize
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="weight-version-rpc")
    context, socket, failed = None, None, False

    def rpc(payload, timeout):
        nonlocal context, socket, failed
        if failed:
            raise RuntimeError("Restart this engine and restore a retained checkpoint after an RPC timeout")
        if socket is None:
            context = zmq.Context(1)
            endpoint = http_server.get_global_state().tokenizer_manager._relax_rpc_endpoint
            socket = get_zmq_socket(context, zmq.DEALER, endpoint, True)
        deadline = time.monotonic() + timeout
        socket.setsockopt(zmq.SNDTIMEO, max(1, int(timeout * 1000)))
        sock_send(socket, RpcReqInput(method="relax_weight_version", parameters={"payload": payload}))
        if not socket.poll(max(0, int((deadline - time.monotonic()) * 1000)), zmq.POLLIN):
            failed = True
            raise TimeoutError("Worker RPC timed out; restart the engine before recovery")
        reply = sock_recv(socket)
        if not isinstance(reply, RpcReqOutput) or not reply.success:
            raise RuntimeError(f"Worker RPC failed: {reply}")

    @asynccontextmanager
    async def bounded_control(timeout):
        if not math.isfinite(timeout) or timeout <= 0:
            raise HTTPException(status_code=400, detail="Control timeout must be positive and finite")
        deadline = time.monotonic() + timeout
        try:
            await asyncio.wait_for(control.acquire(), timeout)
        except TimeoutError as error:
            raise HTTPException(
                status_code=409,
                detail={"error": "Control queue deadline exceeded", "action": "inspect_consumer_status_and_retry"},
            ) from error
        try:

            def left():
                value = deadline - time.monotonic()
                if value <= 0:
                    raise TimeoutError("Control request expired before execution")
                return value

            left()
            yield left
        except TimeoutError as error:
            raise HTTPException(
                status_code=409,
                detail={"error": str(error), "action": "inspect_consumer_status_and_retry"},
            ) from error
        finally:
            control.release()

    async def control_request(request: Request):
        nonlocal failed
        body = await request.json()
        if body.get("action") == "status":
            return {
                "identity": fence.identity,
                "phase": fence.phase,
                "generation_open": gate.opened,
                "active_requests": gate.active,
                "unavailable": None if gate.opened else gate.unavailable(),
            }
        async with bounded_control(float(body.get("timeout", 120))) as left:
            try:
                action = body["action"]
                if failed and action != "close":
                    raise RuntimeError("Restart this engine and recover its checkpoint after a runtime timeout")
                identity = body.get("identity")
                request = json.loads(body["payload"]) if action == "rpc" else None
                if action == "claim":
                    fence.claim(identity)
                    await gate.close(left())
                    gate.failure = None
                elif action == "rpc" and request["phase"] == "identify":
                    await asyncio.get_running_loop().run_in_executor(executor, rpc, body["payload"], left())
                else:
                    fence.require(identity)
                if action in ("claim", "rpc") and (action == "claim" or request["phase"] == "identify"):
                    pass
                elif action == "close":
                    await gate.close(left())
                elif action == "open":
                    if fence.phase not in ("COMMITTED", "ACTIVE"):
                        raise ValueError("Commit the verified installation before opening generation")
                    await gate.open()
                    fence.phase = "ACTIVE"
                elif action == "commit":
                    if fence.phase not in ("PROBED", "COMMITTED"):
                        raise ValueError("Complete verification and generation probe before commit")
                    fence.phase = "COMMITTED"
                elif action == "rpc":
                    fence.check_workers(request)
                    if gate.opened and request["phase"] not in ("check", "dump"):
                        raise ValueError("Drain generation before mutating workers")
                    paths = [Path(request["receipts"]) / f"rank-{rank}.json" for rank in fence.identity["workers"]]
                    while True:
                        # A probe can return its final token before the scheduler
                        # retires its last batch. Let the scheduler advance between
                        # checks while keeping this installation's gate closed.
                        for path in paths:
                            path.unlink(missing_ok=True)
                        await asyncio.get_running_loop().run_in_executor(executor, rpc, body["payload"], left())
                        while not all(path.is_file() for path in paths):
                            left()
                            await asyncio.sleep(0.01)
                        receipts = [json.loads(path.read_bytes()) for path in paths]
                        if any(item["request_id"] != request["request_id"] for item in receipts):
                            raise RuntimeError("Worker receipt belongs to another request")
                        failures = [item for item in receipts if not item["success"]]
                        if not failures:
                            break
                        if request["phase"] in ("check", "flush") and all(
                            item.get("error_code") == "WORKER_BUSY" for item in failures
                        ):
                            await asyncio.sleep(min(0.01, left()))
                            continue
                        raise RuntimeError(f"Worker installation command failed: {failures}")
                    if request["phase"] == "verify":
                        fence.phase = "VERIFIED"
                elif action == "probe":
                    if gate.opened or fence.phase != "VERIFIED":
                        raise ValueError("Run the probe after draining this installation's requests")
                    manager = http_server.get_global_state().tokenizer_manager
                    manager._update_weight_version_if_provided(body["version"])
                    generator = manager.generate_request(
                        GenerateReqInput(input_ids=[0], sampling_params={"temperature": 0, "max_new_tokens": 1})
                    )
                    try:
                        await asyncio.wait_for(anext(generator), left())
                    except TimeoutError:
                        failed = True
                        raise
                    finally:
                        try:
                            await asyncio.wait_for(generator.aclose(), 5)
                        except TimeoutError:
                            failed = True
                            raise
                    fence.phase = "PROBED"
                else:
                    raise ValueError("Unknown weight-version action")
                return {"success": True, "generation_open": gate.opened, "active_requests": gate.active}
            except Exception as error:
                if not gate.opened and body.get("action") in ("claim", "rpc", "probe", "commit", "open", "close"):
                    gate.failure = f"{type(error).__name__}: {error}"
                raise HTTPException(
                    status_code=409,
                    detail={
                        "error": f"{type(error).__name__}: {error}",
                        "action": "restart_engine_and_recover_checkpoint"
                        if failed
                        else "inspect_consumer_status_and_retry",
                    },
                ) from error

    @http_server.app.post("/relax_weight_version")
    @auth_level(AuthLevel.ADMIN_OPTIONAL)
    async def weight_version(request: Request):
        try:
            return await control_request(request)
        except HTTPException as error:
            # SGLang's global handler expects a string detail. Preserve the
            # control endpoint's structured recovery action without changing
            # error handling for ordinary generation endpoints.
            return JSONResponse(status_code=error.status_code, content={"detail": error.detail})
