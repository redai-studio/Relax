# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Bounded ART lookahead ordering, ownership and failure cleanup."""

from threading import Event
from threading import enumerate as threads

import pytest

from relax.engine.rollout.art_http_lookahead import HeaderLookahead


class Response:
    def __init__(self, body, error=False):
        self.content = bytes([body["seed"]]) * 11
        self.error = error
        self.closed = False

    def raise_for_status(self):
        if self.error:
            raise RuntimeError("HTTP failed")

    def close(self):
        self.closed = True


def test_lookahead_starts_next_after_headers_before_current_decode_and_preserves_bytes():
    events = []
    responses = []
    next_started = Event()
    release_next = Event()

    def open_response(body):
        events.append(("start", body["seed"]))
        if body["seed"] == 2:
            next_started.set()
            assert release_next.wait(3)
        response = Response(body)
        responses.append(response)
        events.append(("headers", body["seed"]))
        return response

    cache = HeaderLookahead(open_response)

    def decode_first(r):
        assert next_started.wait(3), "next must run while current decode is active"
        assert events.index(("headers", 1)) < events.index(("start", 2))
        release_next.set()
        return r.content

    try:
        first = cache.consume({"seed": 1}, {"seed": 2}, [0, "abc"], decode_first)
        second = cache.consume({"seed": 2}, None, [0, "abc"], lambda r: r.content)
        assert first == bytes([1]) * 11
        assert second == bytes([2]) * 11
        assert [e for e in events if e[0] == "start"] == [("start", 1), ("start", 2)]
        assert all(r.closed for r in responses)
    finally:
        release_next.set()
        cache.close()
    assert not any(t.name.startswith("art-http-lookahead") for t in threads())


@pytest.mark.parametrize("change", [{"seed": 3}, "policy"])
def test_cached_request_or_policy_mismatch_never_resamples(change):
    calls = []
    responses = []

    def open_response(body):
        calls.append(body)
        r = Response(body)
        responses.append(r)
        return r

    cache = HeaderLookahead(open_response)
    cache.consume({"seed": 1}, {"seed": 2}, [0, "a"], lambda r, *_: r.content)
    with pytest.raises(RuntimeError, match="mismatch"):
        cache.consume(
            change if isinstance(change, dict) else {"seed": 2},
            None,
            [1, "a"] if change == "policy" else [0, "a"],
            lambda *a: None,
        )
    assert len(calls) == 2 and all(r.closed for r in responses)
    assert cache.closed


@pytest.mark.parametrize("failure", ["HTTP", "decode", "reward"])
def test_failure_drains_inflight_response_and_joins_thread(failure):
    responses = []

    def open_response(body):
        r = Response(body, error=failure == "HTTP" and body["seed"] == 2)
        responses.append(r)
        return r

    cache = HeaderLookahead(open_response)

    def decode(r, *args):
        if failure == "decode":
            raise ValueError("decode failed")
        return r.content

    if failure == "decode":
        with pytest.raises(ValueError):
            cache.consume({"seed": 1}, {"seed": 2}, 0, decode)
    else:
        cache.consume({"seed": 1}, {"seed": 2}, 0, decode)
        if failure == "HTTP":
            with pytest.raises(RuntimeError, match="HTTP"):
                cache.consume({"seed": 2}, None, 0, decode)
        else:
            # Driver finally calls clear on reward/prepare failure before offload.
            cache.close()
    assert all(r.closed for r in responses)
    assert not any(t.name.startswith("art-http-lookahead") for t in threads())


def test_engine_lookahead_uses_real_rpc_adapter_and_exact_candidate_order(monkeypatch):
    from types import SimpleNamespace

    import requests
    import torch

    from relax.backends.sglang.diffusion_engine import SGLangNativeGenerationEngine
    from relax.models.sensenova import rollout as sensenova_rollout
    from relax.utils.art_tensor_transport import TensorBuffer, unpack

    monkeypatch.setattr(sensenova_rollout, "pack_sensenova_rollout_requests", lambda reqs: reqs[0])
    calls = []
    responses = []

    def post(url, *, json, timeout, stream):
        assert stream and url.endswith("/rollout/generate")
        calls.append(json)
        r = Response(json)
        r.headers = {"content-type": "application/json"}
        r.json = lambda: [{"seed": json["seed"], "raw": r.content}]
        responses.append(r)
        return r

    monkeypatch.setattr(requests, "post", post)
    engine = SGLangNativeGenerationEngine(args=SimpleNamespace(), rank=0)
    engine._url = "http://unit-test"
    engine._active_version = 7
    engine._active_weight_manifest_sha256 = "abc"
    tensor = torch.arange(300000, dtype=torch.float32)
    engine._map_responses = lambda response, req: {"seed": req["seed"], "raw": response[0]["raw"], "tensor": tensor}

    def req(seed):
        return {
            "seed": seed,
            "rollout_return_dit_trajectory": True,
            "rollout_return_denoising_env": True,
            "extra_sampling_params": {"rollout_options": {"capture": "ode_states"}},
        }

    try:
        first = engine.generate_art_lookahead([req(1)], next_requests=[req(2)])
        second = engine.generate_art_lookahead([req(2)])
        for result, seed in ((first, 1), (second, 2)):
            assert isinstance(result[0]["tensor"], TensorBuffer)
            restored = unpack(result)[0]
            assert restored["seed"] == seed and restored["raw"] == bytes([seed]) * 11
            assert torch.equal(restored["tensor"], tensor)
            assert restored["tensor"].data_ptr() != tensor.data_ptr()
        assert [r["seed"] for r in calls] == [1, 2]
        assert all(r.closed for r in responses)
    finally:
        engine.clear_art_lookahead()


def test_engine_shutdown_reaps_server_even_if_prefetch_failed(monkeypatch):
    from types import SimpleNamespace

    from relax.backends.sglang.diffusion_engine import SGLangNativeGenerationEngine

    engine = SGLangNativeGenerationEngine(args=SimpleNamespace(), rank=0)
    reaped = []
    engine.clear_art_lookahead = lambda: (_ for _ in ()).throw(RuntimeError("prefetch failed"))
    engine._kill_server = lambda: reaped.append(True)
    with pytest.raises(RuntimeError, match="prefetch failed"):
        engine.shutdown()
    assert reaped == [True]


def test_dispatch_keeps_current_and_next_candidates_on_same_engine(monkeypatch):
    from types import SimpleNamespace

    import ray

    from relax.engine.rollout import native_generation as ng

    calls = []

    class Remote:
        def __init__(self, rank):
            self.rank = rank

        def remote(self, current, *, next_requests):
            calls.append((self.rank, current, next_requests))
            return current

    engines = [SimpleNamespace(generate_art_lookahead=Remote(i)) for i in range(8)]
    monkeypatch.setattr(ray, "get", lambda refs: refs)
    current = [{"seed": i} for i in range(8)]
    nxt = [{"seed": i + 10} for i in range(8)]
    assert ng._dispatch_groups(
        engines, [current], next_group_requests=[nxt], lookahead=True, use_art_transport=True
    ) == [current]
    assert calls == [(i, [current[i]], [nxt[i]]) for i in range(8)]


@pytest.mark.parametrize("failure", ["submit", "result"])
def test_driver_clear_failure_still_drains_every_actor_and_rolls_back(monkeypatch, failure):
    from types import SimpleNamespace

    import ray
    import ray._private.internal_api as api

    from relax.engine.rewards import generative as gr
    from relax.engine.rollout import native_generation as ng

    old = object()
    refs = {(2, 0): old}
    images = {}
    owners = {}
    calls = []
    freed = []
    groups = [[ng.Sample(index=g * 2 + i, group_index=g, prompt="p") for i in range(2)] for g in range(2)]
    args = SimpleNamespace(
        sampling_config={
            "algorithm": "art",
            "wave_groups": 1,
            "overlap_group_processing": True,
            "overlap_pending_groups": 3,
            "generation_prefetch": True,
            "art": {"train_step_indices": [0], "max_cache_bytes": 10000, "max_replay_bytes": 10000},
        },
        reward_runtime="cpu",
        reward_scorer_path="relax.engine.rewards.textpecker_reward.TextPeckerRewardScorer",
        custom_reward_post_process_path="relax.engine.rewards.generative.post_process",
        rollout_batch_size=2,
    )

    class Clear:
        def __init__(self, index):
            self.index = index

        def remote(self):
            calls.append(("submit", self.index))
            if self.index == 0 and failure == "submit":
                raise ValueError("clear submit failed")
            return self.index

    engines = [SimpleNamespace(clear_art_lookahead=Clear(i)) for i in range(2)]

    def get(ref):
        calls.append(("wait", ref))
        if ref == 0:
            raise ValueError("clear result failed")
        return {"success": True}

    monkeypatch.setattr(ray, "get", get)
    monkeypatch.setattr(ng, "_TRAJECTORY_REF_CACHE", refs)
    monkeypatch.setattr(ng, "_CANDIDATE_IMAGE_CACHE", images)
    monkeypatch.setattr(ng, "_CANDIDATE_IMAGE_ROLLOUT_IDS", owners)
    monkeypatch.setattr(
        ng,
        "_load_adapter",
        lambda _: SimpleNamespace(build_rollout_request=lambda *_a, **_k: {"height": 1, "width": 1}),
    )
    monkeypatch.setattr(ng, "_get_engines", lambda *_: engines)
    monkeypatch.setattr(ng, "_get_samples", lambda *_: groups)
    monkeypatch.setattr(ng, "prune_stale_artifacts", lambda *_: None)
    monkeypatch.setattr(ng, "_dispatch_groups", lambda *_a, **_k: [[{}, {}]])

    def prepare(_args, _sampling, _adapter, rollout, _fp, group, _responses, **kwargs):
        refs[(rollout, group[0].group_index)] = object()
        uri = f"memory://private/{group[0].group_index}"
        images[uri] = object()
        owners[uri] = rollout
        return group

    def score(_args, group):
        for sample in group:
            sample._prefetched_generative_reward = (4, {"ocr": 1.0}, True)

    def free(items, *, local_only):
        assert not any(t.name.startswith(("rollout-prepare", "rollout-reward")) for t in threads())
        freed.extend(items)

    monkeypatch.setattr(ng, "_prepare_generated_group", prepare)
    monkeypatch.setattr(gr, "prefetch_group_rewards", score)
    monkeypatch.setattr(api, "free", free)
    monkeypatch.setattr(
        "relax.utils.utils.transfer_batch_to_data_system", lambda *_a, **_k: pytest.fail("Early TQ put")
    )
    with pytest.raises(RuntimeError, match="lookahead cleanup failed"):
        ng.generate_rollout(args, 4, object(), object())
    assert ("submit", 1) in calls and ("wait", 1) in calls
    assert refs == {(2, 0): old} and freed and not images and not owners
    assert all(not hasattr(sample, "_prefetched_generative_reward") for group in groups for sample in group)


@pytest.mark.parametrize(
    "path,cleared",
    [("/get_weights_checksum", False), ("/update_weights_from_tensor", True), ("/release_memory_occupation", True)],
)
def test_control_queries_preserve_prefetch_but_mutations_drain_it(monkeypatch, path, cleared):
    from types import SimpleNamespace

    import requests

    from relax.backends.sglang.diffusion_engine import SGLangNativeGenerationEngine

    events = []
    engine = SGLangNativeGenerationEngine(args=SimpleNamespace(), rank=0)
    engine._url = "http://unit-test"
    engine.clear_art_lookahead = lambda: events.append("clear")

    def post(*args, **kwargs):
        events.append("post")
        return SimpleNamespace(
            status_code=200, headers={}, raise_for_status=lambda: None, json=lambda: {"success": True}
        )

    monkeypatch.setattr(requests, "post", post)
    assert engine._post(path, {}) == {"success": True}
    assert events == (["clear", "post"] if cleared else ["post"])
