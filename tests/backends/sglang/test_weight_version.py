# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from unittest.mock import MagicMock

import pytest
import requests

from tests.backends.sglang.test_router_registration import sglang_engine_module as sglang_engine_module


@pytest.mark.parametrize("version", ["7", 0, None])
def test_weight_version_uses_model_info_without_legacy_endpoint(sglang_engine_module, monkeypatch, version):
    engine = object.__new__(sglang_engine_module.SGLangEngine)
    engine.node_rank = 0
    engine.server_host = "backend.example"
    engine.server_port = 15000
    response = MagicMock()
    response.json.return_value = {"weight_version": version}
    get = MagicMock(return_value=response)
    monkeypatch.setattr(sglang_engine_module.requests, "get", get)

    assert engine.get_weight_version() == version
    get.assert_called_once_with("http://backend.example:15000/model_info", timeout=5.0)
    response.raise_for_status.assert_called_once_with()


def test_weight_version_propagates_http_failure(sglang_engine_module, monkeypatch):
    engine = object.__new__(sglang_engine_module.SGLangEngine)
    engine.node_rank = 0
    engine.server_host = "backend.example"
    engine.server_port = 15000
    error = requests.HTTPError("503 backend unavailable")
    response = MagicMock()
    response.raise_for_status.side_effect = error
    monkeypatch.setattr(sglang_engine_module.requests, "get", MagicMock(return_value=response))

    with pytest.raises(requests.HTTPError) as caught:
        engine.get_weight_version()

    assert caught.value is error
    response.json.assert_not_called()


def test_weight_version_follower_does_not_query_http(sglang_engine_module, monkeypatch):
    engine = object.__new__(sglang_engine_module.SGLangEngine)
    engine.node_rank = 1
    get = MagicMock()
    monkeypatch.setattr(sglang_engine_module.requests, "get", get)

    assert engine.get_weight_version() is None
    get.assert_not_called()
