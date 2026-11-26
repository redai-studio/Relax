# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from types import SimpleNamespace

import pytest
import yaml

from relax.utils.types import Sample


sglang_rollout = pytest.importorskip("relax.engine.rollout.sglang_rollout", exc_type=ModuleNotFoundError)


@pytest.mark.parametrize(
    "router_args,expected_keys",
    [
        ({"sglang_router_policy": "manual"}, ["0", "0", "1", "ungrouped"]),
        ({"sglang_router_policy": "consistent_hashing"}, ["first", "second", "third", "ungrouped"]),
        ({"sglang_router_policy": "round_robin"}, [None, None, None, None]),
        ({"router_policy": "manual"}, ["0", "0", "1", "ungrouped"]),
        (
            {"sglang_router_policy": "consistent_hashing", "router_policy": "manual"},
            ["first", "second", "third", "ungrouped"],
        ),
        (
            {"prefill_num_servers": 1, "router_prefill_policy": "manual", "router_decode_policy": "power_of_two"},
            ["0", "0", "1", "ungrouped"],
        ),
        (
            {"prefill_num_servers": 1, "router_decode_policy": "consistent_hashing"},
            ["first", "second", "third", "ungrouped"],
        ),
        (
            {"prefill_num_servers": 1, "sglang_router_policy": "manual", "router_prefill_policy": "cache_aware"},
            ["0", "0", "1", "ungrouped"],
        ),
        (
            {
                "prefill_num_servers": 1,
                "sglang_router_policy": "manual",
                "router_prefill_policy": "cache_aware",
                "router_decode_policy": "power_of_two",
            },
            [None, None, None, None],
        ),
    ],
)
async def test_generate_routes_prompt_groups_and_sessions(monkeypatch, router_args, expected_keys):
    """Exercise request construction; only tokenizer loading and HTTP transport
    are isolated."""
    args = SimpleNamespace(
        ci_test=False,
        fully_async=False,
        sglang_router_ip="localhost",
        sglang_router_port=1,
        sglang_router_policy=None,
        router_policy="cache_aware",
        sglang_speculative_algorithm=None,
        use_rollout_routing_replay=False,
        use_rollout_indexer_replay=False,
        lora_rank=0,
        lora_adapter_mode=False,
    )
    vars(args).update(router_args)
    await _assert_generation_routing(monkeypatch, args, expected_keys)


@pytest.mark.parametrize("first_model_pd", [False, True])
async def test_generate_uses_primary_yaml_model_topology(monkeypatch, tmp_path, first_model_pd):
    from relax.distributed.ray.rollout import _resolve_sglang_config

    pd_groups = [{"worker_type": "prefill", "num_gpus": 1}, {"worker_type": "decode", "num_gpus": 2}]
    regular_groups = [{"worker_type": "regular", "num_gpus": 3}]
    groups = [pd_groups, regular_groups] if first_model_pd else [regular_groups, pd_groups]
    config = tmp_path / "engines.yaml"
    config.write_text(yaml.safe_dump({"sglang": [{"name": str(i), "engine_groups": g} for i, g in enumerate(groups)]}))
    args = SimpleNamespace(
        ci_test=False,
        fully_async=False,
        sglang_config=str(config),
        rollout_num_gpus=6,
        sglang_router_ip="localhost",
        sglang_router_port=1,
        sglang_router_policy="manual",
        router_prefill_policy="cache_aware",
        router_decode_policy="consistent_hashing",
        sglang_speculative_algorithm=None,
        use_rollout_routing_replay=False,
        use_rollout_indexer_replay=False,
        lora_rank=0,
        lora_adapter_mode=False,
    )
    _resolve_sglang_config(args)
    expected_keys = ["first", "second", "third", "ungrouped"] if first_model_pd else ["0", "0", "1", "ungrouped"]
    await _assert_generation_routing(monkeypatch, args, expected_keys)


async def _assert_generation_routing(monkeypatch, args, expected_keys):
    state = SimpleNamespace(
        tokenizer=SimpleNamespace(encode=lambda *_args, **_kwargs: [10]),
        processor=None,
    )
    headers_sent = []

    async def post(url, payload, headers=None, **kwargs):
        headers_sent.append(headers)
        return {
            "text": "answer",
            "meta_info": {"output_token_logprobs": [[-0.1, 11]], "finish_reason": {"type": "stop"}},
        }

    monkeypatch.setattr(sglang_rollout, "GenerateState", lambda _args: state)
    monkeypatch.setattr(sglang_rollout, "post", post)
    for group_index, session_id in [(0, "first"), (0, "second"), (1, "third"), (None, "ungrouped")]:
        sample = Sample(prompt="prompt", group_index=group_index, session_id=session_id)
        result = await sglang_rollout.generate(args, sample, {"max_new_tokens": 1})
        assert result.tokens == [10, 11]
        assert result.status == Sample.Status.COMPLETED

    assert headers_sent == [{"X-SMG-Routing-Key": key} if key is not None else None for key in expected_keys]
