# Copyright (c) 2026 Relax Authors. All Rights Reserved.


import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest

from relax.engine.rollout import on_policy_distillation as module
from relax.inference.registry import InferenceRegistry
from relax.inference.specs import ModelSnapshot, ReplicaSnapshot, RoutingSpec


def _args(**overrides):
    values = dict(
        opd_token_selection="student_sampled",
        opd_teacher_timeout_s=1.0,
        opd_teacher_prompt_key=None,
        opd_teacher_image_key=None,
        opd_teacher_routes_map={"math": ["http://stale.example/generate"]},
        opd_teacher_key="data_source",
        _inference_teacher_discovery_url="http://gateway.example/teacher",
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _sample(index, source="math", group=1):
    return module.Sample(
        index=index, group_index=group, metadata={"data_source": source}, tokens=[1, 2, 3], response_length=2
    )


@pytest.fixture
def transport(monkeypatch):
    import relax.inference.client as client_module

    state = SimpleNamespace(requests=[], clients=[], responses={}, state="READY")
    registry = InferenceRegistry("teacher")
    client_class = httpx.AsyncClient

    def handle(request):
        state.requests.append(request)
        if request.method == "GET":
            model = ModelSnapshot(
                "math",
                state.state,
                "DIRECT",
                engines=(
                    ReplicaSnapshot("r0", "http://teacher-0.example", state.state, True),
                    ReplicaSnapshot("r1", "http://teacher-1.example", state.state, True),
                ),
            )
            snapshot = registry.publish([model], RoutingSpec(default_model="math", route_key_map={"math": "math"}))
            return httpx.Response(200, json=snapshot)
        payload = json.loads(request.content)
        if payload["input_ids"][0] == 99:
            return httpx.Response(503, json={"detail": "unavailable"})
        return httpx.Response(
            200, json={"meta_info": {"input_token_logprobs": [[0, 1, None], [-1, 2, None], [-2, 3, None]]}}
        )

    def client(**kwargs):
        instance = client_class(transport=httpx.MockTransport(handle), **kwargs)
        state.clients.append(instance)
        return instance

    monkeypatch.setattr(client_module.httpx, "AsyncClient", client)
    return state


async def test_managed_teacher_discovers_new_endpoint_and_preserves_group_affinity(transport):
    manager = module.OpdManager(_args())
    samples = [_sample(1), _sample(2)]

    await manager.prefill(samples)

    get_requests = [request for request in transport.requests if request.method == "GET"]
    posts = [request for request in transport.requests if request.method == "POST"]
    assert len(get_requests) == 1
    assert str(get_requests[0].url) == "http://gateway.example/teacher/engines?schema=2"
    assert posts[0].url.host == posts[1].url.host
    assert posts[0].headers["x-smg-routing-key"] == posts[1].headers["x-smg-routing-key"]
    assert posts[0].url.host != "stale.example"
    assert json.loads(posts[0].content) == {
        "input_ids": [1, 2, 3],
        "sampling_params": {"temperature": 0, "max_new_tokens": 0, "skip_special_tokens": False},
        "return_logprob": True,
        "logprob_start_len": 0,
    }
    assert samples[0].teacher_log_probs == [-1, -2]
    assert len(transport.clients) == 1
    assert transport.clients[0].is_closed
    assert manager._inference_clients == {}


async def test_managed_teacher_concurrent_prefills_have_separate_closed_clients(transport):
    manager = module.OpdManager(_args())

    await asyncio.gather(manager.prefill([_sample(1)]), manager.prefill([_sample(2, group=2)]))

    assert len(transport.clients) == 2
    assert all(client.is_closed for client in transport.clients)
    assert manager._inference_clients == {}


async def test_managed_teacher_sleeping_discovery_does_not_submit_logprob_request(transport):
    transport.state = "SLEEPING"
    manager = module.OpdManager(_args())

    with pytest.raises(RuntimeError, match="All OPD teacher fetches failed"):
        await manager.prefill([_sample(1)])

    assert not any(request.method == "POST" for request in transport.requests)
    assert all(client.is_closed for client in transport.clients)


async def test_managed_teacher_inline_partial_failure_retains_old_policy_but_deferred_fails(transport):
    manager = module.OpdManager(_args())
    bad, good = _sample(1), _sample(2)
    bad.tokens[0] = 99

    await manager.prefill([bad, good])

    assert bad.teacher_log_probs is None
    assert good.teacher_log_probs == [-1, -2]
    with pytest.raises(RuntimeError, match="ordinals"):
        await manager.teacher_prefill([bad, good])


@pytest.mark.parametrize("metadata,error_type", [({}, ValueError), ({"data_source": "missing"}, KeyError)])
async def test_managed_teacher_missing_or_unknown_routes_do_not_fallback(transport, metadata, error_type):
    manager = module.OpdManager(_args())
    sample = _sample(1)
    sample.metadata = metadata

    with pytest.raises(error_type):
        await manager.prefill([sample])

    assert not any(request.method == "POST" for request in transport.requests)
    assert all(client.is_closed for client in transport.clients)


async def test_external_teacher_keeps_legacy_url_transport_without_discovery(monkeypatch):
    manager = module.OpdManager(_args(_inference_teacher_discovery_url=None))
    calls = []

    async def post(session, url, payload, sample, err_tag):
        calls.append(url)
        return module.opd_main_worker.LogprobResponse(
            {"meta_info": {"input_token_logprobs": [[0, 1], [-1, 2], [-2, 3]]}}
        )

    monkeypatch.setattr(manager, "_post_logprob", post)
    sample = _sample(1)
    await manager.prefill([sample])

    assert calls == ["http://stale.example/generate"]
    assert sample.teacher_log_probs == [-1, -2]
    assert manager._inference_clients == {}
