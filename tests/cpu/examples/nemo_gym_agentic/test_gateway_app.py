# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import time

from fastapi.testclient import TestClient
from relax_nemo_gym_example.app.protocol import TrialStatus
from relax_nemo_gym_example.service.app import create_app
from relax_nemo_gym_example.service.run_adapter import AdapterResult, CleanupResult

from .test_gateway_service import _request, _settings


class ImmediateHandle:
    async def wait(self):
        return AdapterResult(status=TrialStatus.COMPLETED, reward=1.0)

    async def abort(self):
        return CleanupResult(confirmed=True)

    async def force_cleanup(self):
        return CleanupResult(confirmed=True)

    async def probe_cleanup(self):
        return CleanupResult(confirmed=True)


class ImmediateAdapter:
    async def start(self, context):
        return ImmediateHandle()

    async def ready(self):
        return True

    async def close(self):
        return None


def test_gateway_http_contract_and_health_metadata():
    app = create_app(settings=_settings(), adapter=ImmediateAdapter())

    with TestClient(app) as client:
        health = client.get("/healthz")
        created = client.post("/v1/trials", json=_request("request-one").to_payload())

        assert health.status_code == 200
        assert health.json()["gym_commit"] == "test-commit"
        assert created.status_code == 202

        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            result = client.get("/v1/trials/request-one")
            if result.json()["status"] == "completed":
                break
            time.sleep(0.001)
        assert result.json()["reward"] == 1.0

        ready = client.get("/readyz")
        assert ready.status_code == 200
        assert ready.json()["ready"] is True


def test_gateway_rejects_unregistered_callback_host_without_echoing_token():
    app = create_app(settings=_settings(), adapter=ImmediateAdapter())
    payload = _request("request-one").to_payload()
    payload["model_endpoint"]["base_url"] = "http://metadata.internal/latest"

    with TestClient(app) as client:
        response = client.post("/v1/trials", json=payload)

    assert response.status_code == 400
    assert payload["model_endpoint"]["api_key"] not in response.text
