#!/usr/bin/env python3
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Basic tests for the Metrics Service components."""

import unittest
from unittest.mock import Mock, patch

import pytest
import requests

# MetricsBuffer is defined inside metrics.py, we need to import it correctly
from relax.utils.metrics.client import MetricsClient
from relax.utils.metrics.metrics_service_adapter import MetricsServiceAdapter
from relax.utils.metrics.service import MetricsBuffer
from relax.utils.misc import create_namespace


class TestMetricsBuffer(unittest.TestCase):
    """Test MetricsBuffer class."""

    def setUp(self):
        self.buffer = MetricsBuffer()

    def test_add_and_get_metrics(self):
        """Test adding and retrieving metrics."""
        # Add metrics
        self.buffer.add_metric(step=1, metric_name="loss", metric_value=0.5)
        self.buffer.add_metric(step=1, metric_name="accuracy", metric_value=0.9, tags={"type": "train"})
        self.buffer.add_metric(step=2, metric_name="loss", metric_value=0.4)

        # Get metrics for step 1
        metrics_step1 = self.buffer.get_metrics_for_step(1)
        self.assertEqual(len(metrics_step1), 2)
        self.assertEqual(metrics_step1[0]["name"], "loss")
        self.assertEqual(metrics_step1[0]["value"], 0.5)
        self.assertEqual(metrics_step1[1]["name"], "accuracy")
        self.assertEqual(metrics_step1[1]["value"], 0.9)
        self.assertEqual(metrics_step1[1]["tags"], {"type": "train"})

        # Get metrics for step 2
        metrics_step2 = self.buffer.get_metrics_for_step(2)
        self.assertEqual(len(metrics_step2), 1)
        self.assertEqual(metrics_step2[0]["name"], "loss")
        self.assertEqual(metrics_step2[0]["value"], 0.4)

        # Check non-existent step
        metrics_step3 = self.buffer.get_metrics_for_step(3)
        self.assertEqual(len(metrics_step3), 0)

    def test_has_metrics_for_step(self):
        """Test checking if metrics exist for a step."""
        self.assertFalse(self.buffer.has_metrics_for_step(1))

        self.buffer.add_metric(step=1, metric_name="loss", metric_value=0.5)
        self.assertTrue(self.buffer.has_metrics_for_step(1))

        self.assertFalse(self.buffer.has_metrics_for_step(2))

    def test_clear_step(self):
        """Test clearing metrics for a step."""
        self.buffer.add_metric(step=1, metric_name="loss", metric_value=0.5)
        self.buffer.add_metric(step=2, metric_name="accuracy", metric_value=0.9)

        self.assertTrue(self.buffer.has_metrics_for_step(1))
        self.assertTrue(self.buffer.has_metrics_for_step(2))

        self.buffer.clear_step(1)
        self.assertFalse(self.buffer.has_metrics_for_step(1))
        self.assertTrue(self.buffer.has_metrics_for_step(2))

        self.buffer.clear_step(2)
        self.assertFalse(self.buffer.has_metrics_for_step(2))


class TestMetricsClient(unittest.TestCase):
    """Test MetricsClient class."""

    def setUp(self):
        self.client = MetricsClient("http://test-server:8000/metrics")

    @patch("requests.Session.post")
    def test_log_metric_buffered(self, mock_post):
        """Test logging metrics with buffering."""
        # Log metrics with buffering (immediate=False)
        result = self.client.log_metric(
            step=1, metric_name="test_metric", metric_value=42.0, tags={"source": "test"}, immediate=False
        )

        # Should return True without making HTTP request
        self.assertTrue(result)
        mock_post.assert_not_called()

        # Check buffer count
        self.assertEqual(self.client.get_buffered_metrics_count(step=1), 1)
        self.assertEqual(self.client.get_buffered_metrics_count(), 1)

    @patch("requests.Session.post")
    def test_log_metrics_batch_buffered(self, mock_post):
        """Test logging batch metrics with buffering."""
        metrics = {
            "metric1": 1.0,
            "metric2": 2.0,
            "metric3": 3.0,
        }

        result = self.client.log_metrics_batch(step=2, metrics=metrics, tags={"batch": "test"}, immediate=False)

        self.assertTrue(result)
        mock_post.assert_not_called()
        self.assertEqual(self.client.get_buffered_metrics_count(step=2), 3)

    def test_clear_buffer(self):
        """Test clearing the buffer."""
        # Add some metrics
        self.client.log_metric(step=1, metric_name="m1", metric_value=1.0, immediate=False)
        self.client.log_metric(step=2, metric_name="m2", metric_value=2.0, immediate=False)

        self.assertEqual(self.client.get_buffered_metrics_count(), 2)

        # Clear step 1 only
        self.client.clear_buffer(step=1)
        self.assertEqual(self.client.get_buffered_metrics_count(), 1)

        # Clear all
        self.client.clear_buffer()
        self.assertEqual(self.client.get_buffered_metrics_count(), 0)


class TestMetricsServiceAdapter(unittest.TestCase):
    """Test MetricsServiceAdapter for backward compatibility."""

    def setUp(self):
        args_dict = {
            "use_metrics_service": True,
        }
        self.args = create_namespace(args_dict)

        # Mock the client to avoid actual HTTP calls
        self.mock_client = Mock(spec=MetricsClient)
        self.mock_client.log_metrics_batch.return_value = True
        self.mock_client.report_step.return_value = {"status": "success", "message": "Reported"}

    @patch("relax.utils.metrics.metrics_service_adapter.get_serve_url")
    @patch("relax.utils.metrics.metrics_service_adapter.get_metrics_client")
    def test_adapter_initialization(self, mock_get_client, mock_get_serve_url):
        """Test adapter initialization."""
        mock_get_serve_url.return_value = "http://test:8000/metrics"
        mock_get_client.return_value = self.mock_client
        self.mock_client.health_check.return_value = {"status": "healthy"}

        _ = MetricsServiceAdapter(self.args)

        mock_get_serve_url.assert_called_once_with(route_prefix="/metrics")
        mock_get_client.assert_called_once_with("http://test:8000/metrics")
        self.mock_client.health_check.assert_called_once()

    @patch("relax.utils.metrics.metrics_service_adapter.get_serve_url")
    @patch("relax.utils.metrics.metrics_service_adapter.get_metrics_client")
    def test_log_with_step_key(self, mock_get_client, mock_get_serve_url):
        """Test logging metrics with step key."""
        mock_get_serve_url.return_value = "http://test:8000/metrics"
        mock_get_client.return_value = self.mock_client

        adapter = MetricsServiceAdapter(self.args)

        # Test logging metrics
        metrics = {
            "step": 42,
            "train/loss": 0.5,
            "train/accuracy": 0.9,
        }

        result = adapter.log(metrics, step_key="step")
        self.assertTrue(result)

        self.mock_client.report_step.assert_not_called()

        # Flush should trigger reporting
        adapter.flush(metrics["step"])
        self.mock_client.log_metrics_batch.assert_called_once()
        self.mock_client.report_step.assert_called_once_with(42)

    @patch("relax.utils.metrics.metrics_service_adapter.get_serve_url")
    @patch("relax.utils.metrics.metrics_service_adapter.get_metrics_client")
    def test_direct_log(self, mock_get_client, mock_get_serve_url):
        """Test direct logging without buffering."""
        mock_get_serve_url.return_value = "http://test:8000/metrics"
        mock_get_client.return_value = self.mock_client

        adapter = MetricsServiceAdapter(self.args)

        metrics = {
            "train/loss": 0.5,
            "train/accuracy": 0.9,
        }

        result = adapter.direct_log(step=42, metrics=metrics)

        self.assertEqual(result, {"status": "success", "message": "Reported"})
        self.mock_client.log_metrics_batch.assert_called_once_with(42, metrics, immediate=False)
        self.mock_client.report_step.assert_called_once_with(42)


class TestTrackingUtilsIntegration(unittest.TestCase):
    """Test integration with tracking_utils."""

    def setUp(self):
        args_dict = {
            "use_metrics_service": True,
            "use_wandb": False,
            "use_tensorboard": False,
            "use_clearml": False,
        }
        self.args = create_namespace(args_dict)

    @patch("relax.utils.tracking_utils.get_metrics_service_adapter")
    def test_log_with_metrics_service(self, mock_get_adapter):
        """Test tracking_utils.log with metrics service enabled."""
        from relax.utils import tracking_utils

        # Mock adapter
        mock_adapter = Mock()
        mock_adapter.log.return_value = True
        mock_get_adapter.return_value = mock_adapter

        # Initialize tracking
        tracking_utils.init_tracking(self.args)

        # Log metrics
        metrics = {
            "step": 100,
            "train/loss": 0.42,
            "train/accuracy": 0.92,
        }

        result = tracking_utils.log(self.args, metrics, "step")

        self.assertTrue(result)
        mock_adapter.log.assert_called_once_with(metrics, "step")

    @patch("relax.utils.tracking_utils.get_metrics_service_adapter")
    def test_flush_metrics(self, mock_get_adapter):
        """Test flush_metrics function."""
        from relax.utils import tracking_utils

        # Mock adapter
        mock_adapter = Mock()
        mock_adapter.flush.return_value = True
        mock_get_adapter.return_value = mock_adapter

        result = tracking_utils.flush_metrics(self.args, 0)

        self.assertTrue(result)
        mock_adapter.flush.assert_called_once()

    @patch("relax.utils.tracking_utils.get_metrics_service_adapter")
    def test_log_direct(self, mock_get_adapter):
        """Test log_direct function."""
        from relax.utils import tracking_utils

        # Mock adapter
        mock_adapter = Mock()
        mock_adapter.direct_log.return_value = {"status": "success"}
        mock_get_adapter.return_value = mock_adapter

        metrics = {
            "train/loss": 0.42,
            "train/accuracy": 0.92,
        }

        result = tracking_utils.log_direct(self.args, step=100, metrics=metrics)

        self.assertEqual(result, {"status": "success"})
        mock_adapter.direct_log.assert_called_once_with(100, metrics)


@pytest.mark.parametrize("operation", ["metric", "batch", "report", "health"])
def test_metrics_requests_bypass_external_proxy(monkeypatch, operation):
    monkeypatch.setenv("HTTP_PROXY", "http://external-proxy.invalid:9999")
    monkeypatch.setenv("NO_PROXY", "")
    observed = []

    def send(_session, request, **kwargs):
        observed.append((request, kwargs))
        response = requests.Response()
        response.status_code = 200
        response._content = b'{"status":"healthy"}' if operation == "health" else b'{"status":"success"}'
        return response

    monkeypatch.setattr(requests.Session, "send", send)
    client = MetricsClient("http://internal-head:8000/metrics")
    if operation == "metric":
        assert client._send_metric(1, "loss", 0.5)
    elif operation == "batch":
        assert client._send_metrics_batch(1, {"loss": 0.5})
    elif operation == "report":
        assert client._send_report_step(1)["status"] == "success"
    else:
        assert client.health_check()["status"] == "healthy"
    assert len(observed) == 1
    assert not observed[0][1]["proxies"]
    assert observed[0][1]["timeout"] == (10 if operation == "report" else 5)


if __name__ == "__main__":
    unittest.main()
