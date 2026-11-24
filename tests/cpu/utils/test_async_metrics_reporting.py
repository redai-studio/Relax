# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from concurrent.futures import ThreadPoolExecutor
from threading import Event, get_ident
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from relax.utils.metrics.metrics_service_adapter import MetricsServiceAdapter
from relax.utils.metrics.ordered_reporting import OrderedMetricsReporter


@pytest.fixture
def adapter(monkeypatch):
    client = Mock()
    client.health_check.return_value = {"status": "healthy"}
    client.log_metrics_batch.return_value = True
    client.report_step.return_value = {"status": "success"}
    module = "relax.utils.metrics.metrics_service_adapter"
    monkeypatch.setattr(f"{module}.get_serve_url", lambda **kwargs: "http://metrics.test")
    monkeypatch.setattr(f"{module}.get_metrics_client", lambda url: client)
    instance = MetricsServiceAdapter(SimpleNamespace())
    yield instance
    instance.finish_async_reporting()


def test_async_metrics_overlap_preserves_step_order_and_payload(adapter):
    started, release, finishing = Event(), Event(), Event()
    calls = []

    def report(step):
        if step == 0:
            started.set()
            assert release.wait(5)
        calls.append(("flush", step))
        return {"status": "success"}

    def log(step, metrics, **kwargs):
        calls.append(("log", step, metrics))
        return True

    adapter.client.report_step.side_effect = report
    adapter.client.log_metrics_batch.side_effect = log
    with ThreadPoolExecutor(max_workers=1) as caller:
        try:
            adapter.flush(0, wait=False)
            assert started.wait(2)
            # This returns while step 0's HTTP call is still blocked.
            metrics = {"step": 1, "loss": [1.0]}
            assert adapter.log(metrics)
            metrics["loss"][0] = 99.0
            assert adapter.flush(1, wait=False)
            assert adapter.log({"step": 1, "eval/loss": 2.0})
            assert adapter.flush(1, wait=False)

            def finish():
                finishing.set()
                return adapter.finish_async_reporting()

            result = caller.submit(finish)
            assert finishing.wait(2)
            assert not result.done()
        finally:
            release.set()
        assert result.result(timeout=2)
    assert calls == [
        ("flush", 0),
        ("log", 1, {"loss": [1.0]}),
        ("flush", 1),
        ("log", 1, {"eval/loss": 2.0}),
        ("flush", 1),
    ]


def test_async_metrics_reports_network_failure_at_drain(adapter):
    adapter.client.report_step.return_value = {"status": "error", "message": "timeout"}
    assert adapter.flush(0, wait=False)  # Accepted, not a claim of successful delivery.
    assert adapter.finish_async_reporting() is False


def test_async_metrics_direct_log_waits_for_prior_reports(adapter):
    calls = []
    adapter.client.report_step.side_effect = lambda step: calls.append(("flush", step)) or {"status": "success"}
    adapter.client.log_metrics_batch.side_effect = lambda step, *args, **kwargs: calls.append(("log", step)) or True
    adapter.flush(0, wait=False)
    assert adapter.direct_log(1, {"loss": 1.0}) == {"status": "success"}
    assert calls == [("flush", 0), ("log", 1), ("flush", 1)]


def test_ordered_metrics_backpressure_bounds_pending_requests():
    started, release, submitting = Event(), Event(), Event()
    reporter = OrderedMetricsReporter(max_pending=1)

    def slow_report():
        started.set()
        assert release.wait(5)
        return True

    with ThreadPoolExecutor(max_workers=1) as caller:
        try:
            reporter.submit(slow_report)
            assert started.wait(2)

            def submit_next():
                submitting.set()
                return reporter.submit(lambda: True)

            result = caller.submit(submit_next)
            assert submitting.wait(2)
            assert not result.done()
        finally:
            release.set()
        assert result.result(timeout=2)
    assert reporter.close()


def test_ordered_metrics_exception_closes_worker_and_preserves_failure():
    reporter = OrderedMetricsReporter()

    def fail():
        raise RuntimeError("report failed")

    reporter.submit(fail)
    with pytest.raises(RuntimeError, match="report failed"):
        reporter.close()
    with pytest.raises(RuntimeError, match="closed"):
        reporter.submit(lambda: True)
    assert reporter.close() is False


def test_default_flush_waits_for_queued_reports_and_reports_failure(adapter):
    calls = []
    adapter.client.log_metrics_batch.side_effect = lambda *args, **kwargs: calls.append("log") or False
    adapter.client.report_step.side_effect = lambda step: calls.append("flush") or {"status": "success"}
    assert adapter.log({"step": 0, "loss": 1.0})
    assert adapter.flush(0) is False
    assert calls == ["log", "flush"]


def test_default_flush_includes_final_eval(adapter):
    adapter.log({"step": 0, "train/loss": 1.0})
    adapter.flush(0, wait=False)
    adapter.log({"step": 0, "eval/loss": 2.0})
    assert adapter.flush(0)
    assert adapter.client.log_metrics_batch.call_count == 2
    assert adapter.client.report_step.call_count == 2


def test_default_async_log_normalizes_values_on_caller_thread(adapter):
    caller_thread = get_ident()

    class Scalar:
        def tolist(self):
            assert get_ident() == caller_thread
            return 1.0

    assert adapter.log({"step": 0, "loss": Scalar()})
    assert adapter.flush(0)
    adapter.client.log_metrics_batch.assert_called_once_with(0, {"loss": 1.0}, immediate=True)


def test_ordered_metrics_accepts_concurrent_producers():
    reporter = OrderedMetricsReporter(max_pending=2)
    delivered = []
    with ThreadPoolExecutor(max_workers=4) as producers:
        results = list(
            producers.map(lambda value: reporter.submit(lambda: delivered.append(value) or True), range(20))
        )
    assert all(results)
    assert reporter.close()
    assert sorted(delivered) == list(range(20))
