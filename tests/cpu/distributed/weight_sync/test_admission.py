import asyncio
import unittest

from relax.distributed.weight_sync.admission import AdmissionGate, AdmissionMiddleware


class AdmissionTests(unittest.IsolatedAsyncioTestCase):
    async def test_close_waits_for_stream_and_rejects_new_requests(self):
        gate = AdmissionGate()
        started, release = asyncio.Event(), asyncio.Event()
        output = []

        async def send(message):
            output.append(message)

        async def application(scope, receive, send):
            await send({"type": "http.response.start", "status": 200})
            await send({"type": "http.response.body", "body": b"partial", "more_body": True})
            started.set()
            await release.wait()
            await send({"type": "http.response.body", "body": b"done"})

        middleware = AdmissionMiddleware(application, gate)
        await gate.open()
        serving = asyncio.create_task(middleware({"type": "http", "path": "/v1/chat/completions"}, None, send))
        await started.wait()
        closing = asyncio.create_task(gate.close(1))
        await asyncio.sleep(0)
        await middleware({"type": "http", "path": "/generate"}, None, send)
        self.assertEqual(output[-2]["status"], 503)
        self.assertFalse(closing.done())
        release.set()
        await serving
        await closing
        self.assertEqual(gate.active, 0)

    async def test_drain_timeout_keeps_gate_closed_until_explicit_open(self):
        gate = AdmissionGate()
        await gate.open()
        self.assertTrue(await gate.enter())
        with self.assertRaises(TimeoutError):
            await gate.close(0.01)
        self.assertFalse(await gate.enter())
        await gate.leave()
        await gate.open()
        self.assertTrue(await gate.enter())
        await gate.leave()
