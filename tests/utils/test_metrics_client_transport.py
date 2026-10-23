#!/usr/bin/env python3
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Tests for the MetricsClient transport and the MetricsService report path.

Covered here, and not by the older tests (they patched the module-level
``requests`` functions and therefore never exercised the real transport):

* the client reuses one HTTP connection across calls instead of opening a new
  one per request, and it ignores ``HTTP_PROXY`` / ``HTTPS_PROXY`` for what is a
  cluster-local endpoint;
* the blocking backend fan-out in ``/report_step`` runs off the event loop, so a
  slow backend does not stall concurrent metric ingestion on the same replica;
* a metric ingested while that fan-out runs, and the batch of a request that is
  cancelled, each go out exactly once.

The connection-reuse test drives a real HTTP server and counts accepted TCP
connections, so it fails if the client goes back to a fresh connection per call.
"""

import asyncio
import contextlib
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

import pytest

from relax.utils.metrics.client import MetricsClient
from relax.utils.metrics.service import LogMetricsBatchRequest, MetricsBuffer, MetricsService, ReportStepRequest


class _CountingHandler(BaseHTTPRequestHandler):
    """Answer every metrics endpoint with success."""

    protocol_version = "HTTP/1.1"

    def _reply(self) -> None:
        # Drain the request body first. On a keep-alive connection, bytes left
        # unread here are parsed as the start of the next request and the server
        # answers 400 -- which is only reachable because reuse actually happens.
        length = int(self.headers.get("Content-Length") or 0)
        if length:
            self.rfile.read(length)
        body = b'{"status": "success"}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    do_POST = _reply
    do_GET = _reply

    def log_message(self, *args):  # silence the per-request stderr line
        pass


class _CountingHTTPServer(ThreadingHTTPServer):
    """Counts accepted connections, which is what connection reuse controls."""

    daemon_threads = True

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.accepted = 0

    def get_request(self):
        conn, addr = super().get_request()
        self.accepted += 1
        return conn, addr


class _CountingServer:
    """Threaded HTTP server exposing how many connections were accepted."""

    def __init__(self) -> None:
        self._server = _CountingHTTPServer(("127.0.0.1", 0), _CountingHandler)
        self._thread = threading.Thread(target=self._server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)

    def __enter__(self) -> "_CountingServer":
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)

    @property
    def url(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}/metrics"

    @property
    def accepted(self) -> int:
        return self._server.accepted


def test_client_reuses_one_connection_across_calls():
    """Repeat calls must not open a new TCP connection each time.

    Five POSTs plus a GET over one keep-alive session should ride a single
    connection. A client that calls the module-level ``requests.post`` per
    request opens one connection per call.
    """
    with _CountingServer() as server:
        client = MetricsClient(server.url)
        for step in range(5):
            assert client.log_metrics_batch(step, {"train/loss": float(step)}, immediate=True) is True
        assert client.health_check().get("status") == "success"
        accepted = server.accepted

    assert accepted == 1, f"expected one reused connection, the server accepted {accepted}"


def test_client_ignores_proxy_environment_for_cluster_local_endpoint(monkeypatch):
    """A proxy in the environment must not capture the metrics endpoint."""
    with _CountingServer() as server:
        # urllib prefers the lowercase names, and a NO_PROXY that covers
        # 127.0.0.1 would bypass the proxy anyway, so set and clear both cases.
        for name in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
            monkeypatch.setenv(name, "http://proxy.invalid:3128")
        for name in ("NO_PROXY", "no_proxy"):
            monkeypatch.delenv(name, raising=False)
        client = MetricsClient(server.url)
        # If trust_env were left True, this request would be sent to the proxy
        # and fail, because nothing is listening there.
        assert client.log_metrics_batch(1, {"train/loss": 1.0}, immediate=True) is True


def test_client_uses_a_session_rather_than_module_level_requests():
    client = MetricsClient("http://127.0.0.1:1/metrics")
    assert client._session.trust_env is False
    with patch("requests.post") as module_post:
        with patch.object(client._session, "post", return_value=_FakeResponse()) as session_post:
            client.log_metrics_batch(1, {"a": 1.0}, immediate=True)
            session_post.assert_called_once()
            module_post.assert_not_called()


class _FakeResponse:
    status_code = 200

    def json(self):
        return {"status": "success"}


def test_report_step_helper_is_synchronous_so_it_can_run_on_a_thread():
    """``report_step`` stays async while its blocking body is a plain function.

    ``asyncio.to_thread`` needs a sync callable; a coroutine helper would put
    the work straight back on the event loop.
    """
    cls = MetricsService.func_or_class
    assert not asyncio.iscoroutinefunction(cls._report_buffered_step)
    assert asyncio.iscoroutinefunction(cls.report_step)


def _service_with_slow_backend(adapter):
    """A MetricsService whose only backend is ``adapter``.

    The ``@serve.ingress`` wrapper makes the class unconstructible (its
    ``__init__`` is async and the real one would open wandb / TensorBoard /
    ClearML), so build it with ``__new__`` and set the attributes the handlers
    read. ``metrics_buffer`` is the real class, so the handler bodies run their
    real code.
    """
    cls = MetricsService.func_or_class
    service = cls.__new__(cls)
    service._use_wandb = False
    service._timeline_adapter = None
    service._report_lock = threading.Lock()
    service._adapters = {"tensorboard": adapter}
    service.metrics_buffer = MetricsBuffer()
    return service


class _SlowAdapter:
    """Stand-in for SummaryWriter / clearml / a network-mounted dump dir."""

    def __init__(self, sleep_s: float) -> None:
        self.sleep_s = sleep_s
        self.calls = 0

    def log(self, data=None, step=None) -> None:
        self.calls += 1
        time.sleep(self.sleep_s)


def test_slow_backend_does_not_block_concurrent_ingestion():
    """A slow report must not delay a ``/log_metrics_batch`` on the same
    replica.

    This drives the real ``report_step`` coroutine -- the thing that decides whether
    the fan-out runs on the loop. Calling ``_report_to_backends`` through
    ``asyncio.to_thread`` instead would test only that the test's own ``to_thread``
    works, and would pass even if ``report_step`` had no ``await`` at all.

    Two observers run while the report is in flight: a 10 ms ticker, and an ingest
    that arrives 50 ms in. On a blocked loop neither can run until the 300 ms
    fan-out returns, so both land at ~300 ms; with the fan-out on a worker thread
    the ingest completes on its own schedule at ~50 ms.
    """
    service = _service_with_slow_backend(_SlowAdapter(0.3))
    # report_step only enters the fan-out when the step has buffered metrics.
    service.metrics_buffer.add_metric(step=1, metric_name="perf/step_time", metric_value=1.0)

    ingest_delay_s = 0.05
    report_sleep_s = 0.3

    async def scenario():
        ticks = 0

        async def ticker():
            nonlocal ticks
            while True:
                await asyncio.sleep(0.01)
                ticks += 1

        async def ingest():
            await asyncio.sleep(ingest_delay_s)
            started = time.perf_counter()
            result = await service.log_metrics_batch(LogMetricsBatchRequest(step=1, metrics={"train/loss": 0.5}))
            return result, (time.perf_counter() - started) * 1e3

        ticker_task = asyncio.create_task(ticker())
        await asyncio.sleep(0.05)
        ticks_before = ticks

        started = time.perf_counter()
        report_task = asyncio.create_task(service.report_step(ReportStepRequest(step=1)))
        ingest_task = asyncio.create_task(ingest())

        # Read the tick count while the report is still running: awaiting the report
        # first would let the loop drain and hide the blockage.
        while not report_task.done():
            await asyncio.sleep(0.002)
        ticks_during = ticks - ticks_before

        report_result = await report_task
        ingest_result, ingest_ms = await ingest_task
        ticker_task.cancel()
        return report_result, ingest_result, ingest_ms, ticks_during, (time.perf_counter() - started) * 1e3

    report_result, ingest_result, ingest_waited_ms, ticks_during, report_ms = asyncio.run(scenario())

    assert report_result["results"] == {"tensorboard": "success"}
    assert ingest_result["status"] == "success"
    # The ingest landed after the report took its batch, so it waits for the next one.
    assert [m["name"] for m in service.metrics_buffer.get_metrics_for_step(1)] == ["train/loss"]
    assert report_ms >= report_sleep_s * 1e3 * 0.9, (
        f"the report finished too fast to have called the backend: {report_ms:.0f} ms"
    )

    # ~0.3 s of sleep at a 10 ms period: a free loop serves ~30 ticks, a blocked one ~0.
    assert ticks_during > 10, (
        f"the event loop made only {ticks_during} ticks during a {report_sleep_s * 1e3:.0f} ms backend call"
    )

    # The ingest was ready 50 ms in. If it had to wait for the 300 ms fan-out it
    # lands near 300 ms, so anything below half the sleep proves it did not queue.
    assert ingest_waited_ms < report_sleep_s * 1e3 / 2, (
        f"a /log_metrics_batch that was ready {ingest_delay_s * 1e3:.0f} ms into a "
        f"{report_sleep_s * 1e3:.0f} ms report waited {ingest_waited_ms:.0f} ms -- it queued behind the fan-out"
    )


class _GatedAdapter:
    """Records what it is sent and holds the first call until released."""

    def __init__(self) -> None:
        self.received = []
        self.entered = threading.Event()
        self.release = threading.Event()

    def log(self, data=None, step=None) -> None:
        self.received.append(dict(data))
        if len(self.received) == 1:
            self.entered.set()
            assert self.release.wait(5), "the test never released the backend"


def test_a_metric_ingested_during_a_report_goes_out_with_the_next_report_once():
    """The batch is taken before the fan-out, so a metric that arrives during
    it is neither cleared unreported nor sent twice."""
    adapter = _GatedAdapter()
    service = _service_with_slow_backend(adapter)
    service.metrics_buffer.add_metric(step=1, metric_name="perf/step_time", metric_value=1.0)

    async def scenario():
        report = asyncio.create_task(service.report_step(ReportStepRequest(step=1)))
        assert await asyncio.to_thread(adapter.entered.wait, 5), "the report never reached the backend"
        ingest = await service.log_metrics_batch(LogMetricsBatchRequest(step=1, metrics={"train/loss": 0.5}))
        adapter.release.set()
        first = await report
        second = await service.report_step(ReportStepRequest(step=1))
        third = await service.report_step(ReportStepRequest(step=1))
        return ingest, first, second, third

    ingest, first, second, third = asyncio.run(scenario())

    assert ingest["status"] == "success"
    assert first["results"] == {"tensorboard": "success"}
    assert second["results"] == {"tensorboard": "success"}
    assert third["results"] == {}
    assert adapter.received == [{"perf/step_time": 1.0}, {"train/loss": 0.5}]
    assert not service.metrics_buffer.has_metrics_for_step(1)


def test_a_report_that_raises_keeps_its_batch_for_the_next_report():
    service = _service_with_slow_backend(_GatedAdapter())
    service.metrics_buffer.add_metric(step=1, metric_name="perf/step_time", metric_value=1.0)

    def fail(step, metrics_dict):
        raise RuntimeError("backend thread failed")

    service._report_to_backends = fail
    result = asyncio.run(service.report_step(ReportStepRequest(step=1)))

    assert result["status"] == "error"
    assert [m["name"] for m in service.metrics_buffer.get_metrics_for_step(1)] == ["perf/step_time"]


@pytest.mark.parametrize("cancelled_while", ["queued", "running"])
def test_a_cancelled_report_neither_loses_nor_repeats_its_batch(cancelled_while):
    """Serve cancels the handler when the client disconnects.

    Work still queued then never runs and leaves the batch buffered; work
    already running reports it to the end.
    """
    adapter = _GatedAdapter()
    service = _service_with_slow_backend(adapter)
    service.metrics_buffer.add_metric(step=1, metric_name="perf/step_time", metric_value=1.0)
    worker_free = threading.Event()

    async def scenario():
        loop = asyncio.get_running_loop()
        # One worker: holding it keeps the report queued, and the next report
        # cannot start before the cancelled one is done.
        loop.set_default_executor(ThreadPoolExecutor(max_workers=1))
        if cancelled_while == "queued":
            loop.run_in_executor(None, worker_free.wait, 5)
        report = asyncio.create_task(service.report_step(ReportStepRequest(step=1)))
        await asyncio.sleep(0)
        # Blocking the loop is fine here: the worker does not need it.
        if cancelled_while == "running":
            assert adapter.entered.wait(5), "the report never reached the backend"
        report.cancel()
        worker_free.set()
        adapter.release.set()
        with contextlib.suppress(asyncio.CancelledError):
            await report
        await service.report_step(ReportStepRequest(step=1))

    asyncio.run(scenario())

    assert adapter.received == [{"perf/step_time": 1.0}]
    assert not service.metrics_buffer.has_metrics_for_step(1)
