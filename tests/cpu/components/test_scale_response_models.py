# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Regression tests for scale-out API response-model serialization.

FastAPI's ``response_model`` silently drops any field not declared on the
Pydantic model. ``ScaleOutRequest.to_dict()`` carries ``failure_categories``
for the monitor to bucket failures without re-parsing ``error_message``; these
tests pin that the field survives the HTTP response models (both the single
status and the list wrapper) so it actually reaches /status, /scale_out/{id},
and the autoscaler poll-back that reads them.
"""

import pytest


try:
    from relax.components.rollout import ListScaleOutRequestsResponse, ScaleOutStatusResponse
    from relax.distributed.ray.rollout import ScaleOutRequest, ScaleOutStatus

    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False


pytestmark = pytest.mark.skipif(not HAS_DEPS, reason="Missing ray/serve/fastapi dependencies")


def _failed_request():
    req = ScaleOutRequest(request_id="r1", status=ScaleOutStatus.FAILED, num_replicas=1)
    req.error_message = "scale-out failed: NCCL transport mismatch: wrong_type"
    req.failure_categories = ["NCCL_PRECHECK_TRANSPORT_MISMATCH"]
    return req


def test_scale_out_status_response_preserves_failure_categories():
    req = _failed_request()
    resp = ScaleOutStatusResponse(**req.to_dict())
    assert resp.failure_categories == ["NCCL_PRECHECK_TRANSPORT_MISMATCH"]
    # model_dump() is what FastAPI serializes onto the wire.
    assert resp.model_dump()["failure_categories"] == ["NCCL_PRECHECK_TRANSPORT_MISMATCH"]


def test_scale_out_status_response_defaults_empty_when_absent():
    req = ScaleOutRequest(request_id="r2", status=ScaleOutStatus.ACTIVE, num_replicas=1)
    resp = ScaleOutStatusResponse(**req.to_dict())
    assert resp.failure_categories == []


def test_list_response_preserves_failure_categories():
    req = _failed_request()
    listed = ListScaleOutRequestsResponse(
        requests=[ScaleOutStatusResponse(**req.to_dict())],
        total_count=1,
    )
    assert listed.model_dump()["requests"][0]["failure_categories"] == ["NCCL_PRECHECK_TRANSPORT_MISMATCH"]
