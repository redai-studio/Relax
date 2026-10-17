# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import pytest
from fastapi import HTTPException
from starlette.requests import Request

from relax.components.inference_gateway import GATEWAY_REQUEST_HEADER
from relax.components.rollout import Rollout


def _request(headers: list[tuple[bytes, bytes]]) -> Request:
    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/v1/chat/completions",
            "headers": headers,
        }
    )


def test_rollout_rejects_direct_inference_request() -> None:
    with pytest.raises(HTTPException) as error:
        Rollout.func_or_class._require_gateway_request(_request([]))
    assert error.value.status_code == 403


def test_rollout_accepts_gateway_inference_request() -> None:
    Rollout.func_or_class._require_gateway_request(_request([(GATEWAY_REQUEST_HEADER.encode(), b"1")]))
