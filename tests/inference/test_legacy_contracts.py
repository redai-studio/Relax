# Copyright (c) 2026 Relax Authors. All Rights Reserved.


from __future__ import annotations

import json
from collections import Counter
from types import SimpleNamespace

import httpx
import numpy as np
import pytest


@pytest.fixture
async def genrm_http(monkeypatch):
    from relax.utils import genrm_client as module

    state = SimpleNamespace(requests=[], responses=[], sleeps=[])

    def handle(request: httpx.Request) -> httpx.Response:
        state.requests.append(request)
        status, payload = state.responses.pop(0)
        return httpx.Response(status, json=payload)

    async def sleep(delay: float) -> None:
        state.sleeps.append(delay)

    async_client_cls = httpx.AsyncClient
    sync_client_cls = httpx.Client
    transport = httpx.MockTransport(handle)
    monkeypatch.setattr(module.httpx, "AsyncClient", lambda **kw: async_client_cls(transport=transport, **kw))
    monkeypatch.setattr(module.httpx, "Client", lambda **kw: sync_client_cls(transport=transport, **kw))
    monkeypatch.setattr(module, "asyncio", SimpleNamespace(sleep=sleep))
    client = module.GenRMClient("http://inference.example/genrm///", timeout=1.0)
    try:
        yield client, state
    finally:
        await client._async_client.aclose()
        client._sync_client.close()


async def test_legacy_genrm_client_preserves_messages_sampling_and_route_key(genrm_http):
    client, state = genrm_http
    state.responses = [(200, {"response": "score: 0.75", "model": "judge"})]
    messages = [{"role": "user", "content": "Evaluate this answer."}]
    sampling_params = {"temperature": 0.0, "max_new_tokens": 8}

    response = await client.generate(messages, sampling_params, route_key="math")

    assert response == "score: 0.75"
    assert len(state.requests) == 1
    request = state.requests[0]
    assert request.method == "POST"
    assert str(request.url) == "http://inference.example/genrm/generate"
    assert json.loads(request.content) == {
        "messages": messages,
        "sampling_params": sampling_params,
        "route_key": "math",
    }


async def test_legacy_genrm_client_omits_unspecified_selectors(genrm_http):
    client, state = genrm_http
    state.responses = [(200, {"response": "ok"})]

    assert await client.generate([]) == "ok"
    assert json.loads(state.requests[0].content) == {"messages": []}


async def test_legacy_genrm_client_does_not_retry_client_errors(genrm_http):
    client, state = genrm_http
    state.responses = [(400, {"detail": "unknown route key"})]

    with pytest.raises(httpx.HTTPStatusError) as error:
        await client.generate([], route_key="missing")

    assert error.value.response.status_code == 400
    assert len(state.requests) == 1
    assert state.sleeps == []


async def test_legacy_genrm_client_retains_bounded_server_error_retries(genrm_http):
    client, state = genrm_http
    state.responses = [(503, {"detail": "unavailable"})] * 3

    with pytest.raises(httpx.HTTPStatusError):
        await client.generate([])

    assert len(state.requests) == 3
    assert state.sleeps == [0.5, 1.0]


@pytest.fixture
def opd_module(monkeypatch):
    from relax.engine.rollout import on_policy_distillation as module

    monkeypatch.setattr(module, "_TEACHER_URL_RR", {})
    monkeypatch.setattr(module, "_TEACHER_GROUP_REPLICA", {})
    return module


def _opd_args(token_selection: str, *, advantage: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        use_opd=True,
        opd_type="sglang",
        opd_token_selection=token_selection,
        opd_log_prob_top_k=2,
        opd_kl_coef=0.1 if advantage else 0.0,
        opd_loss_coef=0.0 if advantage else 0.1,
        opd_teacher_prompt_key=None,
        opd_teacher_image_key=None,
    )


def test_legacy_teacher_routes_keep_groups_sticky_and_balance_interleaved_sources(opd_module):
    routes = {
        "text": ["http://text-0.example/generate", "http://text-1.example/generate"],
        "vision": ["http://vision-0.example/generate", "http://vision-1.example/generate"],
    }
    args = SimpleNamespace(opd_teacher_routes_map=routes, opd_teacher_key="data_source")
    selected = {source: Counter() for source in routes}

    for group_index in range(16):
        source = "text" if group_index % 2 == 0 else "vision"
        sample = opd_module.Sample(group_index=group_index, metadata={"data_source": source})
        urls = [opd_module._pick_teacher_url(args, sample) for _ in range(4)]
        assert len(set(urls)) == 1
        assert urls[0] in routes[source]
        selected[source][urls[0]] += 1

    for source, replicas in routes.items():
        assert selected[source] == {replica: 4 for replica in replicas}


@pytest.mark.parametrize(
    ("metadata", "error_type"),
    [({}, ValueError), ({"data_source": "missing"}, KeyError)],
)
def test_legacy_teacher_invalid_route_never_falls_back(opd_module, metadata, error_type):
    args = SimpleNamespace(
        opd_teacher_routes_map={"text": ["http://text.example/generate"]},
        opd_teacher_key="data_source",
        opd_teacher_url="http://fallback.example/generate",
    )

    with pytest.raises(error_type):
        opd_module._pick_teacher_url(args, opd_module.Sample(metadata=metadata))


@pytest.mark.parametrize(
    ("token_selection", "advantage", "expected"),
    [
        ("student_sampled", False, ["teacher_log_probs", "rollout_log_probs"]),
        ("student_topk", False, ["opd_topk_token_ids", "opd_topk_teacher_log_probs"]),
        ("teacher_topk", False, ["opd_topk_token_ids", "opd_topk_teacher_log_probs"]),
        ("union", False, ["opd_topk_token_ids", "opd_topk_teacher_log_probs", "opd_topk_ksz"]),
        (
            "teacher_topk",
            True,
            ["opd_topk_token_ids", "opd_topk_teacher_log_probs", "opd_topk_student_log_probs"],
        ),
        (
            "union",
            True,
            ["opd_topk_token_ids", "opd_topk_teacher_log_probs", "opd_topk_student_log_probs", "opd_topk_ksz"],
        ),
    ],
)
def test_legacy_teacher_transfer_schema_matches_training_consumers(opd_module, token_selection, advantage, expected):
    from relax.utils.opd import opd_utils as utils

    args = _opd_args(token_selection, advantage=advantage)
    manager = opd_module.OpdManager(args)
    train_fields, advantage_fields = ["tokens"], ["tokens"]

    utils.consume_opd_train_data(train_fields, args)
    utils.consume_opd_advantage_data(advantage_fields, args)

    assert manager.schema_opd_transfer_data() == expected
    assert train_fields == ["tokens", *expected]
    assert advantage_fields == ["tokens", *expected]


def test_legacy_teacher_sampled_writeback_preserves_row_order_and_rollout_log_probs(opd_module):
    samples = [
        opd_module.Sample(teacher_log_probs=[-0.1, -0.2]),
        opd_module.Sample(teacher_log_probs=[]),
        opd_module.Sample(teacher_log_probs=[-0.3]),
    ]
    rollout_log_probs = [[-0.4, -0.5], [], [-0.6]]
    train_data = {"rollout_log_probs": rollout_log_probs}

    opd_module.OpdManager(_opd_args("student_sampled")).produce_opd_transfer_data(samples, train_data)

    assert train_data["teacher_log_probs"] == [[-0.1, -0.2], [], [-0.3]]
    assert train_data["rollout_log_probs"] is rollout_log_probs


def test_legacy_teacher_union_writeback_flattens_each_row_without_reordering(opd_module):
    sample = opd_module.Sample(
        opd_topk_token_ids=np.array([[11, 12, 0], [21, 22, 23]], dtype=np.int32),
        opd_topk_teacher_log_probs=np.array([[-1.0, -2.0, 0.0], [-3.0, -4.0, -5.0]], dtype=np.float32),
        opd_topk_student_log_probs=np.array([[-6.0, -7.0, 0.0], [-8.0, -9.0, -10.0]], dtype=np.float32),
        opd_topk_ksz=np.array([2, 3], dtype=np.int32),
    )
    train_data = {}

    opd_module.OpdManager(_opd_args("union", advantage=True)).produce_opd_transfer_data(
        [sample, opd_module.Sample()], train_data
    )

    assert train_data == {
        "opd_topk_token_ids": [[11, 12, 0, 21, 22, 23], []],
        "opd_topk_teacher_log_probs": [[-1.0, -2.0, 0.0, -3.0, -4.0, -5.0], []],
        "opd_topk_student_log_probs": [[-6.0, -7.0, 0.0, -8.0, -9.0, -10.0], []],
        "opd_topk_ksz": [[2, 3], []],
    }


def test_legacy_teacher_override_projection_preserves_policy_configuration():
    from relax.utils.opd import opd_utils as utils

    args = SimpleNamespace(
        teacher_hf_checkpoint="teacher-checkpoint",
        hf_checkpoint="policy-checkpoint",
        teacher_sglang_mem_fraction_static=0.7,
        teacher_sglang_moe_dense_tp_size=2,
        sglang_mem_fraction_static=0.4,
        sglang_moe_dense_tp_size=1,
        sglang_load_format="dummy",
    )
    original = vars(args).copy()

    overrides = utils.build_teacher_overrides(args, colocate_sync=True)
    teacher_args = utils.build_teacher_engine_args(args, overrides)

    assert overrides == {
        "model_path": "teacher-checkpoint",
        "mem_fraction_static": 0.7,
        "moe_dense_tp_size": 2,
        "load_format": "auto",
        "enable_memory_saver": True,
    }
    assert teacher_args is not args
    assert teacher_args.sglang_model_path == "teacher-checkpoint"
    assert teacher_args.sglang_load_format == "auto"
    assert teacher_args.sglang_mem_fraction_static == 0.7
    assert teacher_args.sglang_moe_dense_tp_size == 2
    assert vars(args) == original
