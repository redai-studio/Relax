# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.engine.lora.acceptance.assertions import audit_evidence, compare, sample_scores, scores, worker_observations
from tests.engine.lora.acceptance.fixtures import make_fixtures


def test_numerical_comparison_uses_declared_asymmetric_tolerance():
    assert compare({1: -1.001}, {1: -1.0}, atol=0.002, rtol=0) < 0.002
    with pytest.raises(AssertionError):
        compare({1: -1.1}, {1: -1.0}, atol=0.002, rtol=0)
    with pytest.raises(AssertionError):
        compare({2: -1.0}, {1: -1.0}, atol=0.002, rtol=0)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_numerical_comparison_rejects_nonfinite_in_either_baseline(value):
    with pytest.raises(AssertionError):
        compare({1: value}, {1: value}, atol=0.001, rtol=0.001)
    with pytest.raises(AssertionError):
        scores({"meta_info": {"output_token_ids_logprobs": [[[value, 1]]]}})


@pytest.mark.parametrize("fault", [None, "token_fork", "decode_score", "missing", "nonfinite"])
def test_mixed_decode_checks_every_position_and_rejects_token_forks(fault):
    from copy import deepcopy

    from tests.engine.lora.acceptance.assertions import compare_decode

    reference = {"output_ids": [4, 5], "meta_info": {"output_token_ids_logprobs": [[[-1.0, 7]], [[-2.0, 7]]]}}
    actual = deepcopy(reference)
    if fault == "token_fork":
        actual["output_ids"][1] = 6
    elif fault == "decode_score":
        actual["meta_info"]["output_token_ids_logprobs"][1][0][0] = -3.0
    elif fault == "missing":
        actual["meta_info"]["output_token_ids_logprobs"].pop()
    elif fault == "nonfinite":
        actual["meta_info"]["output_token_ids_logprobs"][1][0][0] = float("nan")
    if fault:
        with pytest.raises(AssertionError):
            compare_decode(actual, reference, atol=1e-3, rtol=1e-3)
    else:
        assert compare_decode(actual, reference, atol=1e-3, rtol=1e-3) == [0, 0]


@pytest.mark.parametrize("cached_tokens", [0, 1, None])
def test_decode_reference_replays_full_sequence_in_fresh_cache_namespace(cached_tokens):
    import httpx

    from tests.engine.lora.acceptance.support import reference_decode

    requests = []
    output = {"output_ids": [4, 5], "meta_info": {"cached_tokens": cached_tokens}}

    def respond(request):
        body = json.loads(request.content)
        requests.append(body)
        assert body["input_ids"] == [1, 2, 3] and body["logprob_start_len"] == -1
        assert body["sampling_params"] == {"temperature": 0, "max_new_tokens": 2, "ignore_eos": True}
        return httpx.Response(200, json=output)

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            ctx = SimpleNamespace(
                client=client,
                report={},
                diagnostic_ids=[7],
                config={"baselines": {"A": {"url": "http://reference", "lora_path": "A", "cache_enabled": True}}},
            )
            for _ in range(2):
                if cached_tokens != 0:
                    with pytest.raises(AssertionError, match="diagnostic prefix"):
                        await reference_decode(ctx, "A", [1, 2, 3], tokens=2)
                else:
                    assert await reference_decode(ctx, "A", [1, 2, 3], tokens=2) == output
            assert requests[0]["extra_key"] != requests[1]["extra_key"]
            assert all(item["response"] == output for item in ctx.report["references"])

    asyncio.run(run())


def test_export_score_positions_exclude_inter_turn_observations():
    sample = SimpleNamespace(
        tokens=[1, 2, 3, 4, 5, 6], response_length=4, loss_mask=[1, 0, 0, 1], rollout_log_probs=[-0.5, 0, 0, -0.8]
    )
    assert sample_scores(sample) == {2: -0.5, 5: -0.8}


@pytest.mark.parametrize("expected", [-0.5, -1.0214])
@pytest.mark.parametrize("zero_output", [False, True])
def test_session_comparison_preserves_raw_evidence_before_failure(expected, zero_output):
    import httpx

    from tests.engine.lora.acceptance.support import score_export

    sample = SimpleNamespace(
        tokens=[10, 20, 30, 40, 50],
        rollout_tokens=[10, 20, 30, 40, 50],
        response_length=3,
        loss_mask=[1, 0, 1],
        rollout_log_probs=[-0.5, 0, -0.7],
        metadata={
            "lora_adapter": {"version_id": "A", "digest": "digest"},
            "lora_attempts": [
                {"adapter_version_id": "A", "adapter_digest": "digest", "token_start": start, "token_end": start + 1}
                for start in (2, 4)
            ],
        },
    )
    if zero_output:
        sample.metadata["lora_attempts"].insert(0, {**sample.metadata["lora_attempts"][0], "token_end": 2})
    requests = []

    def respond(request):
        payload = json.loads(request.content)
        start = (2, 4)[len(requests) % 2]
        requests.append(payload)
        assert payload["input_ids"] == sample.tokens[:start] and payload["lora_path"] == "A"
        assert payload["logprob_start_len"] == -1
        assert payload["sampling_params"] == {"temperature": 0, "max_new_tokens": 1, "ignore_eos": True}
        return httpx.Response(
            200,
            json={
                "output_ids": [sample.tokens[start]],
                "meta_info": {
                    "output_token_logprobs": [[expected if start == 2 else -0.7, sample.tokens[start]]],
                    "cached_tokens": 0 if start == 2 else 3,
                },
            },
        )

    report = {"samples": [], "numerical": []}

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            ctx = SimpleNamespace(
                client=client,
                report=report,
                atol=1e-3,
                rtol=1e-3,
                config={"baselines": {"A": {"url": "http://baseline", "lora_path": "A", "cache_enabled": True}}},
            )
            await score_export(ctx, SimpleNamespace(samples=[sample]), "A")

    if expected == -0.5:
        asyncio.run(run())
    else:
        with pytest.raises(AssertionError, match="Session sample 0, adapter A"):
            asyncio.run(run())
    assert report["samples"][0]["tokens"] == sample.tokens
    evidence = report["numerical"][0]
    assert evidence["actual"] == {2: -0.5, 4: -0.7}
    assert evidence["baseline_repeats"] == [{2: expected, 4: -0.7}, {2: expected, 4: -0.7}]
    assert evidence["method"] == "independent_cached_decode_v1"
    assert len(requests) == 4
    assert requests[0]["extra_key"] == requests[1]["extra_key"]
    assert requests[2]["extra_key"] == requests[3]["extra_key"] != requests[0]["extra_key"]
    assert evidence["baseline_attempts"][0][-1]["response"]["meta_info"]["cached_tokens"] == 3
    if zero_output:
        assert evidence["baseline_attempts"][0][0]["status"] == "NO_OUTPUT"
    assert evidence["max_error"] == pytest.approx(abs(expected + 0.5))
    assert evidence["status"] == ("PASS" if expected == -0.5 else "FAIL")
    if expected != -0.5:
        worst = evidence["worst_positions"][0]
        assert worst["position"] == 2 and worst["token_id"] == 30
        assert worst["baseline_repeat"] == expected
        assert worst["allowed_error"] == pytest.approx(1e-3 + 1e-3 * abs(expected))


@pytest.mark.parametrize(
    "fault", ["different_tokens", "misaligned_scores", "warm_start", "missing_span", "rollout_tokens"]
)
def test_session_replay_rejects_invalid_reference_without_rescoring(fault):
    import httpx

    from tests.engine.lora.acceptance.support import score_export

    sample = SimpleNamespace(
        tokens=[10, 20, 30],
        rollout_tokens=[10, 20, 30],
        response_length=1,
        loss_mask=[1],
        rollout_log_probs=[-0.5],
        metadata={
            "lora_adapter": {"version_id": "A", "digest": "digest"},
            "lora_attempts": [
                {"adapter_version_id": "A", "adapter_digest": "digest", "token_start": 2, "token_end": 3}
            ],
        },
    )
    if fault == "missing_span":
        sample.metadata["lora_attempts"][0]["token_end"] = 2
    if fault == "rollout_tokens":
        sample.rollout_tokens = [10, 99, 30]
    calls = []

    def respond(request):
        calls.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "output_ids": [99 if fault == "different_tokens" else 30],
                "meta_info": {
                    "cached_tokens": 1 if fault == "warm_start" else 0,
                    "output_token_logprobs": [[-0.5, 99 if fault == "misaligned_scores" else 30]],
                },
            },
        )

    report = {"samples": [], "numerical": []}

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            ctx = SimpleNamespace(
                client=client,
                report=report,
                atol=1e-3,
                rtol=1e-3,
                config={"baselines": {"A": {"url": "http://baseline", "lora_path": "A", "cache_enabled": True}}},
            )
            await score_export(ctx, SimpleNamespace(samples=[sample]), "A")

    with pytest.raises(AssertionError, match="Session sample 0, adapter A"):
        asyncio.run(run())
    assert report["numerical"][0]["status"] == "FAIL"
    assert len(calls) == (0 if fault in ("missing_span", "rollout_tokens") else 1)
    if calls:
        assert calls[0]["logprob_start_len"] == -1
        assert "response" in report["numerical"][0]["baseline_attempts"][0][0]


@pytest.mark.parametrize("diverge", [False, True])
def test_decode_diagnostic_separates_scoring_paths_and_excludes_divergent_prefixes(diverge):
    import httpx

    from tests.engine.lora.acceptance.diagnose import replay

    sample = {
        "tokens": [10, 20, 30, 40],
        "metadata": {"lora_attempts": [{"token_start": 2, "token_end": 4}]},
    }
    recorded = {"actual": {"2": -0.5, "3": -0.6}, "baseline": {"2": -1.0, "3": -1.1}}
    decoded = [30, 99 if diverge else 40]
    requests = []

    def respond(request):
        payload = json.loads(request.content)
        requests.append(payload)
        assert payload["lora_path"] == "A" and payload["sampling_params"]["temperature"] == 0
        if payload["logprob_start_len"] == -1:
            assert payload["input_ids"] == [10, 20]
            assert payload["sampling_params"]["max_new_tokens"] == 2
            return httpx.Response(
                200,
                json={"output_ids": decoded, "meta_info": {"output_token_logprobs": [[-0.5, 30], [-0.6, decoded[1]]]}},
            )
        assert payload["input_ids"] == ([10, 20, 30, 40] if len(requests) == 1 else [10, 20] + decoded)
        return httpx.Response(
            200,
            json={
                "meta_info": {
                    "input_token_logprobs": [[None, 10], [-2.0, 20], [-1.0, 30], [-1.1, payload["input_ids"][-1]]]
                }
            },
        )

    evidence = {"gpu": {"index": "2"}}
    with httpx.Client(base_url="http://baseline", transport=httpx.MockTransport(respond)) as client:
        replay(client, "A", sample, recorded, 1e-3, 1e-3, evidence)
    assert len(requests) == 3
    assert evidence["original_baseline_reproduced"]["status"] == "PASS"
    turn = evidence["turns"][0]
    assert turn["tokens_equal"] is (not diverge)
    assert turn["matching_prefix_tokens"] == (1 if diverge else 2)
    assert turn["recorded_vs_independent_decode"]["status"] == "PASS"
    assert set(turn["recorded_vs_independent_decode"]["actual"]) == ({0} if diverge else {0, 1})
    assert turn["independent_decode_vs_own_prefill"]["status"] == "FAIL"
    assert turn["decode"]["output_ids"] == decoded
    assert "own_sequence_prefill" in turn


@pytest.mark.parametrize("cached_tokens", [0, 3])
def test_warm_diagnostic_preserves_turn_order_without_prefill_probes(cached_tokens):
    import httpx

    from tests.engine.lora.acceptance.diagnose import replay

    sample = {
        "tokens": [10, 20, 30, 40, 50],
        "metadata": {"lora_attempts": [{"token_start": 2, "token_end": 3}, {"token_start": 4, "token_end": 5}]},
    }
    recorded = {"actual": {"2": -0.5, "4": -0.7}}
    requests = []

    def respond(request):
        payload = json.loads(request.content)
        start = (2, 4)[len(requests)]
        requests.append(payload)
        assert payload["input_ids"] == sample["tokens"][:start]
        assert payload["logprob_start_len"] == -1
        assert payload["sampling_params"]["max_new_tokens"] == 1
        token = sample["tokens"][start]
        return httpx.Response(
            200,
            json={
                "output_ids": [token],
                "meta_info": {
                    "output_token_logprobs": [[recorded["actual"][str(start)], token]],
                    "cached_tokens": 0 if start == 2 else cached_tokens,
                },
            },
        )

    evidence = {"gpu": {"index": "2"}}
    with httpx.Client(base_url="http://baseline", transport=httpx.MockTransport(respond)) as client:
        replay(client, "A", sample, recorded, 1e-3, 1e-3, evidence, cold=False)
    assert len(requests) == 2
    assert "original_sequence_prefill" not in evidence
    assert evidence["cache_reuse_observed"] is bool(cached_tokens)
    for turn in evidence["turns"]:
        assert turn["tokens_equal"] and turn["recorded_vs_independent_decode"]["status"] == "PASS"
        assert "own_sequence_prefill" not in turn
        assert "independent_decode_vs_own_prefill" not in turn


def test_diagnostic_selects_failed_sessions_after_publication_passed():
    from tests.engine.lora.acceptance.diagnose import failed_task

    publication = {"numerical": [{"status": "PASS"}]}
    sessions = {"numerical": [{"status": "FAIL"}]}
    assert failed_task({"tasks": {"publication": publication, "sessions": sessions}}) is sessions
    assert failed_task(sessions) is sessions
    with pytest.raises(ValueError, match="exactly one"):
        failed_task({"tasks": {"publication": sessions, "sessions": sessions}})


@pytest.mark.parametrize("fault", [None, "cached", "alignment"])
def test_concurrency_diagnostic_isolates_prefixes_and_never_compares_after_divergence(fault):
    import httpx

    from tests.engine.lora.acceptance.diagnose import replay_batch_sizes

    sample = {"tokens": [10, 20, 30, 40], "metadata": {"lora_attempts": [{"token_start": 2, "token_end": 4}]}}
    recorded = {"actual": {"2": -0.5, "3": -0.7}, "atol": 1e-3, "rtol": 1e-3}
    requests = []

    def respond(request):
        body = json.loads(request.content)
        requests.append(body)
        size = len(body["input_ids"])
        assert body["input_ids"] == [[10, 20]] * size and body["logprob_start_len"] == -1
        assert len(body["extra_key"]) == size
        token = 40 if size == 1 else 99
        return httpx.Response(
            200,
            json=[
                {
                    "output_ids": [30, token],
                    "meta_info": {
                        "cached_tokens": 1 if fault == "cached" else 0,
                        "output_token_logprobs": [
                            [-0.5 if size == 1 else -0.6, 30],
                            [-0.7, 88 if fault == "alignment" else token],
                        ],
                    },
                }
                for _ in range(size)
            ],
        )

    evidence = {"gpu": {"index": "2"}}
    with httpx.Client(base_url="http://baseline", transport=httpx.MockTransport(respond)) as client:
        if fault:
            with pytest.raises(AssertionError):
                replay_batch_sizes(client, "A", sample, recorded, [2], evidence)
            assert evidence["concurrent_replays"][0]["responses"]
            return
        replay_batch_sizes(client, "A", sample, recorded, [2], evidence)
    assert [len(body["input_ids"]) for body in requests] == [1, 1, 2, 2]
    keys = [key for body in requests for key in body["extra_key"]]
    assert len(keys) == len(set(keys))
    assert evidence["scheduler_batch_size"] == "NOT_OBSERVED"
    assert [row["matches_serial"] for row in evidence["concurrent_replays"]] == [True, True, False, False]
    for item in evidence["concurrent_replays"][-1]["comparisons"]:
        result = item["vs_serial"]
        assert result["tokens_equal"] is False and result["matching_prefix_tokens"] == 1
        assert result["scores"]["status"] == "FAIL"
        assert set(result["scores"]["actual"]) == {0}


def test_version_logs_and_core_session_pass_do_not_fake_full_acceptance():
    report = {"checks": {"old_session_logprobs": {"status": "PASS"}}}
    assert "cache_a_to_b" in audit_evidence(report)
    assert "continuous_traffic" in audit_evidence(report)


def test_control_reset_keeps_valid_json_visible_to_live_schedulers(tmp_path, monkeypatch):
    from tests.engine.lora.acceptance.support import configure_test, write_test_control

    path = tmp_path / "control.json"
    previous = {"capture_mixed_batches": True}
    path.write_text(json.dumps(previous))
    observed = []

    def write(target, content, *args, **kwargs):
        # Observe the live path while a writer has truncated its destination.
        with target.open("w") as stream:
            observed.append(json.loads(path.read_text()))
            stream.write(content)
        return len(content)

    monkeypatch.setattr(Path, "write_text", write)
    write_test_control(path, {})
    assert observed == [previous] and json.loads(path.read_text()) == {}
    ctx = SimpleNamespace(test_control=path, root=tmp_path, config={})
    configure_test(ctx, capture_graph=True)
    assert observed[-1] == {} and json.loads(path.read_text())["capture_graph"] is True


def test_slot_reuse_success_does_not_replace_stale_weight_negative_control():
    report = {"checks": {"slot_reuse": {"status": "PASS"}}}
    assert "slot_reuse_negative_control" in audit_evidence(report)


@pytest.mark.parametrize("fault", [None, "nonfinite", "missing", "positive", "restore"])
def test_stale_slot_control_requires_valid_negative_and_correct_restoration(tmp_path, monkeypatch, fault):
    from tests.engine.lora.acceptance import scenarios, support

    calls = []
    expected = {7: -1.0, 8: -2.0}

    async def reference(*args):
        return expected

    async def diagnostic(ctx, bound, ids, **kwargs):
        calls.append(kwargs["extra_key"])
        index = len(calls) - 1
        values = dict(expected)
        if index == 1 or (fault == "positive" and index == 0) or (fault == "restore" and index == 2):
            values[7] = -3.0
        if index == 1 and fault == "nonfinite":
            values[7] = float("nan")
        if index == 1 and fault == "missing":
            values.pop(8)
        injected = json.loads(ctx.test_control.read_text()).get("stale_slot")
        assert bool(injected) == (index == 1)
        if injected:
            (ctx.root / (kwargs["rid"] + ".stale-slot.json")).write_text(json.dumps(injected))
        return {
            "meta_info": {"output_token_ids_logprobs": [[[value, key] for key, value in values.items()]]},
            "verification_completion": {"execution_workers": [0]},
        }

    monkeypatch.setattr(support, "cold_diagnostic", reference)
    monkeypatch.setattr(support, "diagnostic", diagnostic)
    ctx = SimpleNamespace(
        report={"checks": {}},
        config={"timeout_seconds": 1},
        root=tmp_path,
        test_control=tmp_path / "control.json",
        versions={"B": "B"},
        prompts=["prefix"],
        tokenizer=SimpleNamespace(encode=lambda *a, **k: [1, 2]),
        atol=1e-3,
        rtol=1e-3,
    )
    old = {"engine": "E1", "native_lora_id": "old"}
    current = {"engine": "E1", "native_lora_id": "new", "execution_workers": [0]}
    support.configure_test(ctx)
    if fault:
        with pytest.raises(AssertionError):
            asyncio.run(scenarios.stale_slot_negative_control(ctx, [old], [current]))
        assert ctx.report["checks"]["slot_reuse_negative_control"]["status"] == "INCOMPLETE"
    else:
        asyncio.run(scenarios.stale_slot_negative_control(ctx, [old], [current]))
        assert ctx.report["checks"]["slot_reuse_negative_control"]["status"] == "PASS"
        assert len(calls) == 3
    assert len(calls) == len(set(calls))


def test_worker_resources_cannot_be_proven_by_only_the_leader():
    observation = {"worker_boots": ["rank0", "rank1"], "workers": {"0": {"slot": 1}}}
    with pytest.raises(AssertionError, match="incomplete"):
        worker_observations({"observation": observation})
    observation["workers"]["1"] = {"slot": 2}
    assert set(worker_observations({"observation": observation})) == {"0", "1"}
    observation["worker_boots"] = ["same", "same"]
    with pytest.raises(AssertionError, match="incomplete"):
        worker_observations({"observation": observation})


def test_performance_counts_completions_and_detects_stalled_individual_engine():
    from tests.engine.lora.acceptance.performance import compare_windows, summarize_window

    requests = [
        {"accepted": 1.1, "completed": 1.5, "first_token_at": 1.2},
        {"accepted": 1.6, "completed": 2.1, "first_token_at": 1.8},
    ]
    progress = [
        {"at": 0.9, "engine": "E1", "tokens": 100},
        {"at": 1.2, "engine": "E1", "tokens": 3},
        {"at": 1.8, "engine": "E1", "tokens": 2},
        {"at": 2, "engine": "E2", "tokens": 100},
    ]
    metrics = summarize_window(requests, progress, ("E1", "E2"), 1, 2)
    assert metrics["token_throughput"] == {"all": 5, "E1": 5, "E2": 0}
    assert metrics["accepted"] == 2 and metrics["throughput"] == 1
    assert metrics["p95"] == pytest.approx(0.5)
    assert metrics["max_progress_gaps"]["E2"] == 1
    with pytest.raises(AssertionError, match="pre-registered"):
        compare_windows(metrics, metrics, {"max_progress_gap": 0.9})


@pytest.mark.parametrize("completed,error", [(None, None), (2, "timeout")])
def test_performance_cannot_hide_censored_or_failed_requests(completed, error):
    from tests.engine.lora.acceptance.performance import summarize_window

    with pytest.raises(AssertionError, match="failed, censored"):
        summarize_window([{"accepted": 1, "completed": completed, "error": error}], [], ("E1",), 0, 2)


@pytest.mark.parametrize(
    "managed,finish,tokens,publication",
    [
        (True, "length", 1, None),
        (False, "length", 1, None),
        (True, "abort", 1, None),
        (True, "length", 0, None),
        (True, "length", 1, "B"),
    ],
)
@pytest.mark.parametrize("observe", [True, False])
def test_traffic_uses_native_instance_for_managed_and_registered_alias_for_baseline(
    tmp_path, monkeypatch, managed, finish, tokens, publication, observe
):
    import asyncio

    import httpx

    from tests.engine.lora.acceptance import performance, support

    async def scenario():
        bound = {
            "native_lora_id": "instance-A",
            "engine": {"engine_id": "E1", "boot_id": "boot", "endpoint": "http://engine"},
            "session_id": "session",
            "dp_rank": 0,
            "binding": {"version_id": "A", "lora_path": "registered-A", "digest": "digest", "cohort_id": "cohort"},
        }
        calls = []

        async def handle(request):
            body = json.loads(request.content)
            calls.append(request.url.path)
            assert "operation_id" not in body and "version_id" not in body
            if request.url.path == "/lora_version_status":
                assert body["native_lora_id"] == "instance-A"
                return httpx.Response(200, json={"state": "READY"})
            assert body["lora_path"] == ("instance-A" if managed else "registered-A")
            if managed:
                assert body["lora_binding"]["native_lora_id"] == "instance-A"
                assert "operation_id" not in body["lora_binding"] and "version_id" not in body["lora_binding"]
            else:
                assert "lora_binding" not in body and body["routed_dp_rank"] == 0
            await asyncio.sleep(0.001)
            event = {
                "meta_info": {
                    "completion_tokens": tokens,
                    "finish_reason": {"type": finish},
                    "lora_adapter": {
                        "native_lora_id": "instance-A",
                        "adapter_digest": "digest",
                        "engine_boot_id": "boot",
                    },
                }
            }
            return httpx.Response(200, text="data: " + json.dumps(event) + "\n\ndata: [DONE]\n\n")

        async def state(*args):
            return {
                "cohort_id": "cohort",
                "versions": {"A": {"digest": "digest", "ready": {"E1": bound}}},
            }

        monkeypatch.setattr(support, "request", state)

        async def publish(*args, **kwargs):
            await asyncio.sleep(0.002)
            return {"state": "PUBLISHED"}

        monkeypatch.setattr(support, "publish", publish)
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            ctx = SimpleNamespace(
                client=client,
                root=tmp_path,
                test_control=tmp_path / "control.json",
                report={},
                native_owner="owner",
                config={"timeout_seconds": 2, "traffic_window_seconds": 0.03, "traffic_tokens": 1},
                tokenizer=SimpleNamespace(encode=lambda *args, **kwargs: [1, 2]),
            )
            if finish != "length" or tokens != 1:
                with pytest.raises(AssertionError, match="ended early"):
                    await performance.traffic_window(
                        ctx, [bound], managed=managed, observe=observe, diagnostic=not observe
                    )
                failed = ctx.report["traffic_raw"][0]["requests"][0]
                assert failed["finish_reason"]["type"] == finish and failed["completed"] is None
                return
            if publication:
                with pytest.raises(AssertionError, match="lacks coverage"):
                    await performance.traffic_window(
                        ctx, [bound], publication, managed=managed, observe=observe, diagnostic=not observe
                    )
                assert ctx.report["traffic_raw"][0]["publication"]["crossing_requests"] == {"E1": []}
                return
            result = await performance.traffic_window(
                ctx, [bound], managed=managed, observe=observe, diagnostic=not observe
            )
        assert result["accepted"] > 0 and result["failed"] == 0
        record = ctx.report["traffic_raw"][0]["requests"][0]
        if not observe:
            assert record["last_token_at"] <= record["completed"]
            assert record["headers_at"] <= record["terminal_at"] and "server_timing" in record
        else:
            assert "server_timing" not in record
        assert ("/lora_version_status" in calls) == (managed and observe)

    asyncio.run(scenario())


def test_diagnostic_binding_covers_each_engine_dp_group_and_closes_extras(monkeypatch):
    import asyncio

    from tests.engine.lora.acceptance import support

    async def scenario():
        routes = [(0, "e1"), (0, "e2"), (0, "e1"), (1, "e1"), (1, "e2")]
        candidates = iter(
            [
                {"dp_rank": dp, "dp_size": 2, "engine": {"engine_id": engine}, "session_id": str(index)}
                for index, (dp, engine) in enumerate(routes)
            ]
        )
        closed = []

        async def request(*args):
            return {"serving_engines": ["e1", "e2"]}

        async def bind(ctx):
            return next(candidates)

        async def close(ctx, bound):
            closed.append(bound["session_id"])

        monkeypatch.setattr(support, "request", request)
        monkeypatch.setattr(support, "native_bind", bind)
        monkeypatch.setattr(support, "close_native", close)
        result = await support.native_bindings(SimpleNamespace(config={}))
        assert [(bound["dp_rank"], bound["engine"]["engine_id"]) for bound in result] == [
            (0, "e1"),
            (0, "e2"),
            (1, "e1"),
            (1, "e2"),
        ]
        assert closed == ["2"]

    asyncio.run(scenario())


def test_fixture_writer_is_repeatable_and_changes_value_projection(tmp_path):
    torch = pytest.importorskip("torch", reason="fixture serialization requires CPU PyTorch")
    serialization = pytest.importorskip("safetensors.torch", reason="fixture serialization requires safetensors")
    model = tmp_path / "base"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen2",
                "hidden_size": 16,
                "intermediate_size": 32,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "num_hidden_layers": 2,
            }
        )
    )
    # Only the fingerprint is consumed here, not a runnable base checkpoint.
    (model / "model.safetensors").write_bytes(b"fingerprint-only-test-data")
    first, repeated = tmp_path / "first", tmp_path / "repeated"
    make_fixtures(model, first, rank=4)
    make_fixtures(model, repeated, rank=4)
    a = serialization.load_file(first / "A/adapter_model.safetensors")
    b = serialization.load_file(first / "B/adapter_model.safetensors")
    again = serialization.load_file(repeated / "A/adapter_model.safetensors")
    assert a.keys() == b.keys() == again.keys()
    for name in a:
        assert torch.equal(a[name], again[name])
        assert torch.equal(b[name], -a[name] if ".lora_B." in name else a[name])
    for name in a:
        if name.endswith("v_proj.lora_B.weight"):
            left = name.replace(".lora_B.", ".lora_A.")
            assert torch.count_nonzero(a[name] @ a[left]) > 0
    assert (first / "A/producer_manifest.json").read_bytes() == (repeated / "A/producer_manifest.json").read_bytes()
    from tests.engine.lora.acceptance.prepare import prepare

    bundle = tmp_path / "bundle"
    prepare(model, bundle)
    artifacts = json.loads((bundle / "artifacts.json").read_text())
    assert artifacts["versions"]["A"]["digest"] != artifacts["versions"]["B"]["digest"]
    assert artifacts["versions"]["B"]["digest"] == artifacts["versions"]["C"]["digest"]
    assert len(json.loads((bundle / "verification.json").read_text())["prompts"]) == 32
    with pytest.raises(FileExistsError):
        prepare(model, bundle)


@pytest.mark.parametrize("busy_uuid,allowed", [("GPU-other", True), ("GPU-selected-3", False)])
def test_auto_deployment_gpu_selection_preserves_other_jobs(monkeypatch, busy_uuid, allowed):
    from tests.engine.lora.acceptance import processes

    def query(command, **kwargs):
        if "--query-compute-apps=gpu_uuid,pid" in command:
            return f"{busy_uuid}, 123\n"
        return "0, GPU-other, card, 48000, driver\n2, GPU-selected-2, card, 48000, driver\n3, GPU-selected-3, card, 48000, driver\n"

    monkeypatch.setattr(processes.subprocess, "check_output", query)
    if allowed:
        assert [gpu["uuid"] for gpu in processes.select_gpus(["2", "3"])] == ["GPU-selected-2", "GPU-selected-3"]
    else:
        with pytest.raises(RuntimeError, match="No existing process was stopped"):
            processes.select_gpus(["2", "3"])


def test_auto_deployment_failed_task_reaps_only_owned_process(tmp_path):
    from tests.engine.lora.acceptance.processes import owned_process

    records = []
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        with pytest.raises(RuntimeError, match="injected"):
            with owned_process(
                [sys.executable, "-c", "import time; time.sleep(60)"],
                dict(os.environ),
                tmp_path / "owned.log",
                records,
            ) as owned:
                raise RuntimeError("injected")
        assert owned.poll() is not None
        assert unrelated.poll() is None
        assert records[0]["cleanup"] == "SIGNALLED_AND_LEADER_REAPED"
    finally:
        unrelated.terminate()
        unrelated.wait(timeout=10)


def test_auto_deployment_preflight_failure_still_writes_report(tmp_path, monkeypatch):
    from tests.engine.lora.acceptance import __main__ as runner

    def busy(*args):
        raise RuntimeError("GPU busy")

    monkeypatch.setattr(runner, "select_gpus", busy)
    output = tmp_path / "run"
    args = SimpleNamespace(model=tmp_path, output=output, gpus=["2", "3"], tasks=["capacity"])
    with pytest.raises(RuntimeError, match="GPU busy"):
        runner.run_suite(args)
    report = json.loads((output / "report.json").read_text())
    assert report["status"] == "FAIL" and report["processes"] == []
    assert report["error"] == "RuntimeError: GPU busy"


def test_auto_deployment_reaps_owned_child_in_a_separate_process_group(tmp_path):
    import time

    import psutil

    from tests.engine.lora.acceptance.processes import owned_process

    marker = tmp_path / "child.pid"
    script = (
        "import subprocess,sys,time; from pathlib import Path; "
        "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'],start_new_session=True); "
        "Path(sys.argv[1]).write_text(str(p.pid)); time.sleep(60)"
    )
    records = []
    with owned_process([sys.executable, "-c", script, str(marker)], dict(os.environ), tmp_path / "child.log", records):
        deadline = time.monotonic() + 5
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert marker.exists()
        child = psutil.Process(int(marker.read_text()))
        assert child.is_running()
    assert not child.is_running() or child.status() == psutil.STATUS_ZOMBIE
    assert records[0]["detached_processes_reaped"] >= 1


def test_independent_task_files_cover_required_full_acceptance():
    import importlib

    from tests.engine.lora.acceptance.assertions import REQUIRED_EVIDENCE
    from tests.engine.lora.acceptance.deployment import TASKS

    covered = set()
    for name in TASKS:
        task = importlib.import_module(f"tests.engine.lora.acceptance.test_{name}")
        assert callable(task.run) and callable(getattr(task, "test_" + name))
        covered.update(task.CHECKS)
    assert set(REQUIRED_EVIDENCE) <= covered


@pytest.mark.parametrize("returncode,deterministic", [(0, False), (0, True), (1, True)])
def test_auto_deployment_preserves_worker_evidence_without_claiming_full_acceptance(
    tmp_path, monkeypatch, returncode, deterministic
):
    from contextlib import contextmanager

    from tests.engine.lora.acceptance import __main__ as runner

    def prepare(model, output):
        output.mkdir()
        (output / "verification.json").write_text("{}")

    @contextmanager
    def process(command, env, log, records):
        assert env["NO_PROXY"] == env["no_proxy"] == "*"
        assert "http_proxy" not in env and "HTTPS_PROXY" not in env
        config = json.loads(Path(command[-1]).read_text())
        assert config["deterministic"] is deterministic
        output = Path(config["output"])
        (output / "report.json").write_text(
            json.dumps(
                {
                    "status": "FAIL" if returncode else "PASS",
                    "checks": {"capacity": {"status": "PASS", "observations": [1, 2]}},
                }
            )
        )
        log.write_text("injected worker exit")
        yield SimpleNamespace(poll=lambda: returncode, returncode=returncode)

    monkeypatch.setattr(runner, "prepare", prepare)
    monkeypatch.setattr(runner, "select_gpus", lambda _: [{"uuid": "gpu2"}, {"uuid": "gpu3"}])
    monkeypatch.setattr(runner, "owned_process", process)
    monkeypatch.setenv("http_proxy", "http://proxy.invalid:8080")
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.invalid:8080")
    output = tmp_path / "run"
    args = SimpleNamespace(
        model=tmp_path, output=output, gpus=["2", "3"], tasks=["capacity"], deterministic=deterministic
    )
    if returncode:
        with pytest.raises(RuntimeError, match="exit 1"):
            runner.run_suite(args)
    else:
        runner.run_suite(args)
    report = json.loads((output / "report.json").read_text())
    assert report["checks"]["capacity"]["observations"] == [1, 2]
    assert report["full_acceptance"] == "INCOMPLETE"
    assert report["status"] == ("FAIL" if returncode else "PASS")
    assert report["inference_profile"] == ("deterministic_triton" if deterministic else "default")


@pytest.mark.parametrize("deterministic,cold", [(False, False), (True, False), (True, True)])
def test_inference_profile_rejects_silent_engine_configuration_changes(deterministic, cold):
    from tests.engine.lora.acceptance.processes import inference_profile

    info = {
        "enable_deterministic_inference": deterministic,
        "attention_backend": "triton" if deterministic else "flashinfer",
        "disable_radix_cache": cold,
        "disable_cuda_graph": False,
        "disable_decode_cuda_graph": False,
    }
    assert (
        inference_profile(info, deterministic=deterministic, cold=cold)["attention_backend"]
        == info["attention_backend"]
    )
    for key in info:
        if key == "attention_backend" and not deterministic:
            continue
        changed = dict(info, **{key: "flashinfer" if key == "attention_backend" else not info[key]})
        with pytest.raises(AssertionError, match=key):
            inference_profile(changed, deterministic=deterministic, cold=cold)
        missing = {name: value for name, value in info.items() if name != key}
        with pytest.raises(AssertionError, match=key):
            inference_profile(missing, deterministic=deterministic, cold=cold)


@pytest.mark.parametrize("second_engine_graph_disabled", [False, True])
def test_managed_profile_checks_each_serving_engine(second_engine_graph_disabled):
    import httpx

    from tests.engine.lora.acceptance.deployment import check_managed_profiles

    visited = []

    def respond(request):
        visited.append(request.url.host)
        return httpx.Response(
            200,
            json={
                "enable_deterministic_inference": True,
                "attention_backend": "triton",
                "disable_radix_cache": False,
                "disable_cuda_graph": request.url.host == "engine1" and second_engine_graph_disabled,
                "disable_decode_cuda_graph": False,
            },
        )

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            ctx = SimpleNamespace(
                client=client,
                config={"deterministic": True},
                report={
                    "initial": {
                        "default": {"version_id": "A"},
                        "serving_engines": ["0", "1"],
                        "versions": {
                            "A": {
                                "ready": {
                                    str(index): {"engine": {"endpoint": f"http://engine{index}"}} for index in range(2)
                                }
                            }
                        },
                    }
                },
            )
            if second_engine_graph_disabled:
                with pytest.raises(AssertionError, match="disable_cuda_graph"):
                    await check_managed_profiles(ctx)
            else:
                await check_managed_profiles(ctx)
                assert set(ctx.report["engine_profiles"]) == {"0", "1"}
            assert visited == ["engine0", "engine1"]

    asyncio.run(run())


@pytest.mark.parametrize(
    "center,spread,verdict", [(0, 0.005, "PASS"), (0.2, 0.005, "FAIL"), (0.05, 0.5, "INCONCLUSIVE")]
)
def test_overhead_target_uses_block_uncertainty_and_checks_each_engine(center, spread, verdict):
    from tests.engine.lora.acceptance.overhead import analyze

    def blocks_for(engine_only=False):
        blocks = []
        for index in range(12):
            cost = center + spread * (1 if index % 2 else -1)
            windows = []
            for mode in ("ordinary", "managed", "managed", "ordinary"):
                factor = 1 - cost / 100 if mode == "managed" else 1
                rates = {"all": 200 if engine_only else 200 * factor, "E1": 100 * factor, "E2": 100}
                if engine_only:
                    rates["E2"] = 200 - rates["E1"]
                else:
                    rates["E2"] *= factor
                windows.append({"mode": mode, "metrics": {"token_throughput": rates}})
            blocks.append({"windows": windows})
        return blocks

    result = analyze(blocks_for(), 0.05)
    assert result["verdict"] == verdict
    assert result["metrics"]["all"]["overhead_percent"] == pytest.approx(center, abs=0.002)
    if verdict == "FAIL":
        assert analyze(blocks_for(engine_only=True), 0.05)["verdict"] == "FAIL"
    if verdict == "PASS":
        assert result["metrics"]["all"]["overhead_ci_percent"][1] < 0.05
    blocks = blocks_for()
    blocks[-1]["windows"].pop()
    with pytest.raises(ValueError, match="complete"):
        analyze(blocks, 0.05)
    with pytest.raises(ValueError, match="six"):
        analyze(blocks[:5], 0.05)


def test_overhead_experiment_alternates_balanced_blocks_without_publication_or_polling(tmp_path, monkeypatch):
    import asyncio

    from tests.engine.lora.acceptance import overhead

    calls = []
    bindings = ["E1", "E2"]
    ctx = SimpleNamespace(
        config={
            "output": str(tmp_path),
            "gpus": [],
            "overhead_profile": {
                "rounds": 6,
                "diagnostics": False,
                "warmup_seconds": 30,
                "concurrency": 3,
                "target_percent": 0.05,
            },
        },
        report={"traffic_raw": []},
    )

    async def bind(_):
        return bindings

    async def ordinary(_, given, **options):
        assert given == bindings and options == {"concurrency": 3, "observe": False, "warmup_seconds": 30}
        return record("ordinary")

    async def managed(_, given, **options):
        assert given == bindings * 3 and options == {"observe": False, "warmup_seconds": 30}
        return record("managed")

    def record(mode):
        calls.append(mode)
        ctx.report["traffic_raw"].append({"requests": [mode]})
        return {"token_throughput": {"all": 100, "E1": 50, "E2": 50}}

    monkeypatch.setattr(overhead, "gpu_state", lambda gpus: {})
    monkeypatch.setattr(overhead.support, "native_bindings", bind)
    monkeypatch.setattr(overhead.performance, "ordinary_baseline", ordinary)
    monkeypatch.setattr(overhead.performance, "traffic_window", managed)
    asyncio.run(overhead.run(ctx))
    assert calls == ["ordinary", "managed", "managed", "ordinary", "managed", "ordinary", "ordinary", "managed"] * 3
    assert not ctx.report["traffic_raw"]  # Stream raw evidence to disk instead of accumulating hours of tokens.
    result = json.loads((tmp_path / "overhead.json").read_text())
    assert result["verdict"] == "INCONCLUSIVE"  # Identical numbers do not prove perfect measurement precision.
    assert len(list(tmp_path.glob("overhead-*.json"))) == 24
    assert set(ctx.report["checks"]) == {"steady_state_overhead"}


def test_diagnostic_trace_separates_overlapping_gpu_and_nested_cpu_time(tmp_path):
    import gzip

    from tests.engine.lora.acceptance.overhead_diagnostics import summarize_samples, summarize_trace

    path = tmp_path / "profile.trace.json.gz"
    events = [
        {"cat": "kernel", "ph": "X", "name": "lora_kernel", "ts": 0, "dur": 100, "args": {"device": 0}},
        {"cat": "kernel", "ph": "X", "name": "dense_kernel", "ts": 50, "dur": 100, "args": {"device": 0}},
        {"cat": "gpu_memcpy", "ph": "X", "name": "copy", "ts": 200, "dur": 10, "args": {"device": 0}},
        {"cat": "cpu_op", "ph": "X", "name": "parent", "ts": 0, "dur": 200},
        {"cat": "cpu_op", "ph": "X", "name": "child", "ts": 50, "dur": 100},
    ]
    with gzip.open(path, "wt") as sink:
        json.dump({"traceEvents": events}, sink)
    report = summarize_trace(path)
    assert report["status"] == "COMPLETE"
    gpu = report["gpu_activity"]["0"]
    assert gpu["union_busy_ms"] == pytest.approx(0.16)
    assert gpu["max_internal_gap_ms"] == pytest.approx(0.05)
    assert gpu["busy_fraction_within_span"] == pytest.approx(160 / 210)
    assert report["top_events"]["cpu_op"][0]["inclusive_total_ms"] == pytest.approx(0.2)
    path = tmp_path / "cpu-only.json"
    path.write_text(json.dumps({"traceEvents": events[3:]}))
    assert summarize_trace(path)["status"] == "INCOMPLETE"
    samples = tmp_path / "samples.json"
    samples.write_text(
        json.dumps(
            {
                "shared": {"frames": [{"name": "outer"}, {"name": "bind", "file": "control.py", "line": 20}]},
                "profiles": [{"type": "sampled", "samples": [[0, 1], [0, 1], [0]]}],
            }
        )
    )
    report = summarize_samples(samples)
    assert report["top_leaf_frames"][0] == {"function": "bind", "file": "control.py", "line": 20, "samples": 2}


def test_diagnostic_request_breakdown_never_subtracts_client_and_server_clocks():
    from tests.engine.lora.acceptance.overhead_diagnostics import request_breakdown

    raw = {
        "requests": [
            {
                "rid": "r",
                "engine": "e",
                "accepted": 1,
                "headers_at": 1.1,
                "first_token_at": 2,
                "last_token_at": 4,
                "completed": 4.5,
                "tokens": 3,
                "server_timing": {
                    "request_received_ts": 1000,
                    "api_server_dispatch_finish_ts": 1000.25,
                    "forward_entry_time": 2000,
                    "prefill_finished_time": 2000.5,
                    "queue_time": 0.125,
                },
            }
        ],
        "progress": [{"rid": "r", "at": 2}, {"rid": "r", "at": 4}],
    }
    report = request_breakdown(raw, 0, 2)
    row = report["requests"][0]
    assert row["server_api_dispatch_s"] == 0.25 and row["server_queue_s"] == 0.125
    assert row["server_prefill_span_s"] == 0.5 and row["client_mean_decode_s_per_token"] == 1
    assert row["client_response_tail_s"] == 0.5 and "server_after_prefill_s" not in row
    assert row["chunk_gaps_seconds"]["p50"] == 2


@pytest.mark.parametrize("failure", [None, "start", "stop"])
def test_diagnostics_stop_attempted_profiles_and_preserve_partial_evidence(tmp_path, monkeypatch, failure):
    import asyncio

    import httpx

    from tests.engine.lora.acceptance import overhead_diagnostics as diagnostic

    starts, stops, directories = [], [], {}
    real_sleep = asyncio.sleep

    async def immediate(seconds):
        await real_sleep(0)

    async def handler(request):
        host = request.url.host
        if request.url.path == "/start_profile":
            starts.append(host)
            directories[host] = Path(json.loads(request.content)["output_dir"])
            if failure == "start":
                raise httpx.ReadTimeout("start may already have taken effect", request=request)
        elif request.url.path == "/stop_profile":
            stops.append(host)
            (directories[host] / "gpu.trace.json").write_text(
                json.dumps(
                    {
                        "traceEvents": [
                            {"ph": "X", "cat": "kernel", "name": "kernel", "ts": 0, "dur": 10},
                        ]
                    }
                )
            )
            if failure == "stop":
                raise httpx.ReadTimeout("trace exists but stop ACK unknown", request=request)
        return httpx.Response(200, json={})

    async def bind_times(ctx):
        return {"status": "COMPLETE"}

    async def ordinary(ctx, bindings):
        return bindings

    async def traffic(ctx, bindings, **options):
        assert options["observe"] is False and options["diagnostic"] is True
        await options["on_ready"]()
        ctx.report.setdefault("traffic_raw", []).append({"requests": [], "progress": []})
        return {"start": 0, "end": 2}

    monkeypatch.setattr(diagnostic, "binding_timings", bind_times)
    monkeypatch.setattr(diagnostic.shutil, "which", lambda name: None)
    monkeypatch.setattr(diagnostic.asyncio, "sleep", immediate)
    monkeypatch.setattr(diagnostic.performance, "ordinary_bindings", ordinary)
    monkeypatch.setattr(diagnostic.performance, "traffic_window", traffic)
    monkeypatch.delenv("SGLANG_PROFILE_V2", raising=False)
    config = {"output": str(tmp_path), "overhead_profile": {"concurrency": 2}}

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            ctx = SimpleNamespace(config=config, client=client, report={})
            report = await diagnostic.collect(
                ctx, [{"engine": {"endpoint": "http://engine0"}}, {"engine": {"endpoint": "http://engine1"}}]
            )
            assert ctx.config is config
            assert report["status"] == "INCOMPLETE"  # Missing py-spy cannot become a clean bill of health.
            assert not report["included_in_overhead_ci"]
            gpu = [phase for phase in report["phases"] if phase["kind"] == "gpu"]
            assert len(gpu) == 2
            assert all(phase["status"] == ("INCOMPLETE" if failure else "COMPLETE") for phase in gpu)
            assert len(starts) == len(stops) == 4  # Even start timeouts require a stop attempt.
            assert len(list((tmp_path / "diagnostics").glob("*/requests.json"))) == 4
            assert (tmp_path / "diagnostics/report.json").is_file()

    asyncio.run(run())


@pytest.mark.parametrize("fail_bind", [False, True])
def test_binding_diagnostic_retains_cleanup_ownership_on_failed_rpc(fail_bind):
    from tests.engine.lora.acceptance.overhead_diagnostics import binding_timings

    calls = []

    async def remote(action, payload):
        calls.append((action, payload["session_id"]))
        if action == "close":
            return {"accepted": True}
        if fail_bind:
            raise TimeoutError("unknown bind outcome")
        return {"binding": "A", "native_lora_id": "instance-A"}

    ctx = SimpleNamespace(
        native_owner="owner", native_sessions=[], manager=SimpleNamespace(lora_control=SimpleNamespace(remote=remote))
    )
    result = asyncio.run(binding_timings(ctx))
    assert result["status"] == ("INCOMPLETE" if fail_bind else "COMPLETE")
    assert len(ctx.native_sessions) == (1 if fail_bind else 12)
    for bound in ctx.native_sessions:
        actions = [action for action, sid in calls if sid == bound["session_id"]]
        assert actions == (["bind", "close"] if fail_bind else ["bind", "bind", "close"])
    assert all(row["close"] == "ACCEPTED" for row in result["raw"])
