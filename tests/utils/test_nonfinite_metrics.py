# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from types import SimpleNamespace
from unittest.mock import Mock, call
from urllib.parse import urlsplit

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from relax.utils.metrics.adapters.clearml import _ClearMLAdapter
from relax.utils.metrics.client import MetricsClient
from relax.utils.metrics.metrics_service_adapter import MetricsServiceAdapter
from relax.utils.metrics.ordered_reporting import OrderedMetricsReporter
from relax.utils.metrics.service import LogMetricsBatchRequest, MetricsBuffer, MetricsService, ReportStepRequest


@pytest.mark.parametrize("bad_norm", [None, float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("queued", [False, True])
def test_nonfinite_norm_does_not_drop_skip_counters(monkeypatch, bad_norm, queued):
    """Use real request validation, buffering and ClearML adapter; mock only
    HTTP transport and the external ClearML SDK logger."""
    sdk_logger = Mock()
    clearml = object.__new__(_ClearMLAdapter)
    clearml._task = SimpleNamespace(get_logger=lambda: sdk_logger)
    service = SimpleNamespace(
        metrics_buffer=MetricsBuffer(),
        _timeline_adapter=None,
        _use_wandb=False,
        _adapters={"clearml": clearml},
    )
    api = FastAPI()

    @api.post("/log_metrics_batch")
    async def log_batch(request: LogMetricsBatchRequest):
        return await MetricsService.func_or_class.log_metrics_batch(service, request)

    @api.post("/report_step")
    async def report_step(request: ReportStepRequest):
        return await MetricsService.func_or_class.report_step(service, request)

    with TestClient(api) as transport:

        def post(url, *, json, timeout):
            return transport.post(urlsplit(url).path, json=json)

        client = MetricsClient("http://metrics.test")
        monkeypatch.setattr(client._session, "post", post)
        metrics = {
            "train/grad_norm": bad_norm,
            "train/skipped_nonfinite_grad": 1,
            "train/nonfinite_skips_in_process": 2,
            "train/loss": 0.5,
            "train/zero": 0,
        }
        if queued:
            adapter = object.__new__(MetricsServiceAdapter)
            adapter.client = client
            adapter._timeline_enabled = False
            adapter._reporter = OrderedMetricsReporter()
            try:
                assert adapter.log({"train/step": 271, **metrics}, step_key="train/step")
                assert adapter.flush(271)
            finally:
                adapter.finish_async_reporting()
        else:
            assert client.log_metrics_batch(271, metrics)
            result = client.report_step(271)
            assert result["status"] == "success"
            assert result["results"]["clearml"] == "success"

    assert sdk_logger.report_scalar.call_args_list == [
        call(title="train", series="skipped_nonfinite_grad", value=1, iteration=271),
        call(title="train", series="nonfinite_skips_in_process", value=2, iteration=271),
        call(title="train", series="loss", value=0.5, iteration=271),
        call(title="train", series="zero", value=0, iteration=271),
    ]
    assert metrics["train/grad_norm"] is bad_norm
    assert not service.metrics_buffer.has_metrics_for_step(271)
