# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Unit tests for teacher request routing in
``relax/engine/rollout/on_policy_distillation.py``.

A Relax-managed teacher is reached only through the Teacher Gateway: MOPD
selects the teacher model by sending the sample's ``data_source`` as the
Gateway ``route_key``, and every request carries its GRPO group as the Router
routing key, so the teacher's Router keeps a group on one replica. Only an
external teacher keeps its raw URL.

Run with: pytest tests/engine/rollout/test_on_policy_distillation_routing.py -v
"""

from argparse import Namespace

import pytest

from relax.engine.rollout import on_policy_distillation as opd
from relax.utils.types import Sample


TEXT_SOURCE = "dapo-math-17k"
VL_SOURCE = "multimodal-open-r1"


def _mopd_args() -> Namespace:
    return Namespace(
        opd_teacher_gateway_url="http://gateway/teacher/",
        opd_teacher_route_keys=(TEXT_SOURCE, VL_SOURCE),
        opd_teacher_key="data_source",
    )


def _sample(data_source: str) -> Sample:
    return Sample(group_index=0, metadata={"data_source": data_source})


def test_on_policy_distillation_mopd_routes_by_data_source_through_the_gateway():
    payload = {"input_ids": [1, 2], "return_logprob": True}
    url, forwarded = opd._teacher_request_target(_mopd_args(), _sample(VL_SOURCE), payload)
    assert url == "http://gateway/teacher/generate"
    assert forwarded == {**payload, "route_key": VL_SOURCE}
    assert "route_key" not in payload


def test_on_policy_distillation_single_managed_teacher_uses_the_gateway_default_model():
    args = Namespace(opd_teacher_gateway_url="http://gateway/teacher", opd_teacher_route_keys=None)
    payload = {"input_ids": [1]}
    assert opd._teacher_request_target(args, _sample(TEXT_SOURCE), payload) == (
        "http://gateway/teacher/generate",
        payload,
    )


def test_on_policy_distillation_external_teacher_keeps_its_raw_url():
    args = Namespace(opd_teacher_url="http://external/generate")
    payload = {"input_ids": [1]}
    assert opd._teacher_request_target(args, None, payload) == (args.opd_teacher_url, payload)


def test_on_policy_distillation_mopd_missing_routing_key_raises():
    with pytest.raises(ValueError, match="missing key 'data_source'"):
        opd._teacher_request_target(_mopd_args(), Sample(group_index=0, metadata={}), {})


def test_on_policy_distillation_mopd_unknown_data_source_raises():
    with pytest.raises(KeyError, match="no teacher route"):
        opd._teacher_request_target(_mopd_args(), _sample("not-a-source"), {})


def test_on_policy_distillation_same_group_shares_one_routing_key():
    args = _mopd_args()
    same_group = {
        opd._teacher_routing_headers(args, Sample(group_index=7, metadata={"data_source": source})).get(
            opd.TEACHER_ROUTING_KEY_HEADER
        )
        for source in (TEXT_SOURCE, VL_SOURCE)
        for _ in range(8)
    }
    assert same_group == {"7"}
    other_group = opd._teacher_routing_headers(args, Sample(group_index=8, metadata={"data_source": TEXT_SOURCE}))
    assert other_group == {opd.TEACHER_ROUTING_KEY_HEADER: "8"}


def test_on_policy_distillation_routing_key_needs_a_group_and_the_gateway():
    # Eval and replay samples without a group are left to the Router's balance.
    assert opd._teacher_routing_headers(_mopd_args(), Sample(group_index=None, metadata={})) == {}
    external = Namespace(opd_teacher_url="http://external/generate")
    assert opd._teacher_routing_headers(external, Sample(group_index=3, metadata={})) == {}


def test_inference_gateway_deploy_uses_one_cpu_ingress_per_role(monkeypatch):
    from unittest.mock import MagicMock

    from ray import serve

    from relax.components import inference_gateway
    from relax.engine.inference.types import Role
    from relax.utils import utils

    deployment = MagicMock()
    run = MagicMock()
    delete = MagicMock()
    monkeypatch.setattr(inference_gateway, "InferenceGatewayDeployment", deployment)
    monkeypatch.setattr(serve, "run", run)
    monkeypatch.setattr(serve, "delete", delete)
    monkeypatch.setattr(utils, "get_serve_url", lambda prefix: f"http://serve{prefix}")
    owner = object()

    assert inference_gateway.deploy_gateway(Role.TEACHER, manager_handle=owner) == "http://serve/teacher"
    deployment.bind.assert_called_once_with(
        "teacher", manager_handle=owner, upstream_url=None, genrm_backend_handle=None
    )
    run.assert_called_once_with(deployment.bind.return_value, name="teacher_gateway", route_prefix="/teacher")
    inference_gateway.delete_gateway("teacher")
    delete.assert_called_once_with("teacher_gateway")
