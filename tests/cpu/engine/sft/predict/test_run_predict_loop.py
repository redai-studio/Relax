# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Unit tests for the SFT periodic predict loop."""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from relax.engine.sft.dataset.sample import CanonicalMessage, CanonicalSample
from relax.engine.sft.predict import loop as predict_loop
from relax.utils.http_utils import DistributedPostUnconfirmed
from relax.utils.types import Sample


@pytest.mark.asyncio
async def test_unconfirmed_predict_dispatch_blocks_gpu_handoff(monkeypatch):
    from relax.backends.megatron import initialize
    from relax.engine.sft.predict import runner

    config = SimpleNamespace(loss_type="sft", offload_train=True, offload_rollout=True)
    rollout = SimpleNamespace(
        config=config,
        _logger=Mock(),
        healthy=Mock(),
        rollout_manager=SimpleNamespace(
            run_predict=SimpleNamespace(remote=AsyncMock(side_effect=DistributedPostUnconfirmed("poster lost"))),
            offload=SimpleNamespace(remote=AsyncMock()),
        ),
    )
    reply = await runner.handle_predict(rollout, 1)
    assert reply["fatal"] and reply["status"] == "error"
    rollout.rollout_manager.offload.remote.assert_not_awaited()
    rollout.healthy.report_error.remote.assert_called_once_with("rollout", reply["message"], fatal=True)

    monkeypatch.setattr(initialize, "is_megatron_main_rank", lambda: True)
    monkeypatch.setattr(runner, "get_serve_url", lambda role: "http://rollout")
    monkeypatch.setattr(runner, "get_gloo_group", lambda: "test-gloo")
    monkeypatch.setattr(runner.dist, "barrier", lambda **kwargs: None)
    monkeypatch.setattr(runner.dist, "get_world_size", lambda **kwargs: 1)
    monkeypatch.setattr(runner.dist, "all_gather_object", lambda result, error, **kwargs: result.__setitem__(0, error))
    response = Mock()
    response.json.return_value = reply
    monkeypatch.setattr(runner.requests, "get", lambda *args, **kwargs: response)
    actor = Mock(args=config)
    with pytest.raises(RuntimeError, match="poster lost"):
        runner.run_sft_predict(actor, 1)
    actor.sleep.assert_called_once()
    actor.wake_up.assert_not_called()


def _sample(messages: list[tuple[str, str, bool]]) -> CanonicalSample:
    return CanonicalSample(
        messages=[CanonicalMessage(role=r, content=c, learn=l) for r, c, l in messages],
        metadata={"source_dataset": "fake", "row_index": 0},
    )


def test_split_prompt_and_reference_drops_last_assistant():
    sample = _sample(
        [
            ("system", "be helpful", False),
            ("user", "hello", False),
            ("assistant", "hi there", True),
        ]
    )
    prompt_msgs, reference = predict_loop.split_prompt_and_reference(sample)
    assert prompt_msgs == [
        {"role": "system", "content": "be helpful"},
        {"role": "user", "content": "hello"},
    ]
    assert reference == "hi there"


def test_split_prompt_and_reference_uses_last_assistant_in_multi_turn():
    sample = _sample(
        [
            ("user", "q1", False),
            ("assistant", "a1", True),
            ("user", "q2", False),
            ("assistant", "a2", True),
        ]
    )
    prompt_msgs, reference = predict_loop.split_prompt_and_reference(sample)
    assert [m["role"] for m in prompt_msgs] == ["user", "assistant", "user"]
    assert reference == "a2"


def test_split_prompt_and_reference_no_assistant_returns_all_and_empty_ref():
    sample = _sample([("user", "hi", False)])
    prompt_msgs, reference = predict_loop.split_prompt_and_reference(sample)
    assert prompt_msgs == [{"role": "user", "content": "hi"}]
    assert reference == ""


def test_split_prompt_and_reference_serializes_list_content_reference():
    sample = _sample(
        [
            ("user", "describe this", False),
        ]
    )
    sample.messages.append(
        CanonicalMessage(
            role="assistant",
            content=[{"type": "text", "text": "a cat"}],
            learn=True,
        )
    )
    _, reference = predict_loop.split_prompt_and_reference(sample)
    assert json.loads(reference) == [{"type": "text", "text": "a cat"}]


@pytest.mark.parametrize("scoring", [False, True])
def test_render_eval_prompts_uses_seeded_random_eval_rows(monkeypatch, scoring):
    accessed_indices = []
    canonical_samples = []

    class _FakeDataset:
        def __init__(self, **kwargs):  # noqa: ARG002
            pass

        def __len__(self):
            return 10

        def get_canonical_sample(self, idx):
            accessed_indices.append(idx)
            sample = _sample([("user", f"q{idx}", False), ("assistant", f"a{idx}", True)])
            sample.metadata["row_index"] = idx
            canonical_samples.append(sample)
            return sample

        def stop(self):
            pass

    class _FakeTokenizer:
        def apply_chat_template(self, messages, **kwargs):  # noqa: ARG002
            return messages[-1]["content"]

        def encode(self, prompt, *, add_special_tokens):
            assert not add_special_tokens
            return [100, int(prompt.removeprefix("q"))]

    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", lambda *args, **kwargs: _FakeTokenizer())
    monkeypatch.setattr("relax.engine.sft.dataset.streaming.SFTStreamingDataset", _FakeDataset)
    config = SimpleNamespace(
        hf_checkpoint="/fake/model",
        context_parallel_size=1,
        max_tokens_per_gpu=128,
        seed=42,
        eval_prompt_data=None,
        eval_size=0.2,
        prompt_data="/fake/train.jsonl",
        input_key="messages",
        label_key=None,
        multimodal_keys=None,
        metadata_key="metadata",
        tool_key=None,
        system_prompt=None,
    )

    samples = [] if scoring else None
    rendered = predict_loop.render_eval_prompts(config, samples=samples)
    _, expected_eval_indices = predict_loop.resolve_sft_split_indices(10, 0.2, seed=42)

    assert accessed_indices == list(expected_eval_indices)
    assert [reference for _, reference, _ in rendered] == [f"a{idx}" for idx in expected_eval_indices]
    if scoring:
        for position, (sample, canonical, idx) in enumerate(
            zip(samples, canonical_samples, expected_eval_indices, strict=True)
        ):
            assert sample.index == sample.group_index == position
            assert sample.prompt == f"q{idx}"
            assert sample.label == f"a{idx}"
            assert sample.tokens == [100, idx]
            assert sample.response_length == 0
            assert sample.metadata == canonical.metadata
            assert sample.metadata is not canonical.metadata


class _FakePolicyEnginePool:
    def __init__(self, completion_factory=None):
        self.calls: list[list[str]] = []
        self.submitted = []
        self.generated_samples = None
        self.completion_factory = completion_factory or (lambda prompts: [f"answer_{p}" for p in prompts])

    async def generate_predict(self, prompts: list[str], multimodal_inputs_list=None, *, samples=None) -> list[str]:
        self.calls.append(list(prompts))
        completions = self.completion_factory(prompts)
        self.generated_samples = samples
        if samples is not None:
            for sample, completion in zip(samples, completions):
                sample.response = completion
                sample.response_length = 1
                sample.status = Sample.Status.COMPLETED if sample.index == 0 else Sample.Status.TRUNCATED
        return completions

    async def submit_eval_result(self, train_step, result, pending):
        self.submitted.append((train_step, result, pending))


@pytest.mark.asyncio
async def test_generate_and_write_predictions_writes_jsonl_with_three_fields(tmp_path: Path):
    rm = _FakePolicyEnginePool()
    prompts_and_refs = [(f"p{i}", f"r{i}", None) for i in range(5)]
    out_path = tmp_path / "predict" / "predictions_step_42.jsonl"

    await predict_loop.generate_and_write_predictions(rm, prompts_and_refs, out_path=out_path)

    assert out_path.exists()
    lines = out_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 5
    rows = [json.loads(line) for line in lines]
    assert {"prompt", "reference", "completion"} == set(rows[0].keys())
    assert [r["prompt"] for r in rows] == [f"p{i}" for i in range(5)]
    assert [r["reference"] for r in rows] == [f"r{i}" for i in range(5)]
    assert [r["completion"] for r in rows] == [f"answer_p{i}" for i in range(5)]


@pytest.mark.asyncio
async def test_generate_and_write_predictions_fires_all_in_one_call(tmp_path: Path):
    rm = _FakePolicyEnginePool()
    prompts_and_refs = [(f"p{i}", "ref", None) for i in range(7)]
    out_path = tmp_path / "predictions.jsonl"

    await predict_loop.generate_and_write_predictions(rm, prompts_and_refs, out_path=out_path)

    # All 7 prompts in a single generate_predict call (no batching).
    assert [len(c) for c in rm.calls] == [7]
    assert rm.calls[0] == [f"p{i}" for i in range(7)]


@pytest.mark.asyncio
async def test_generate_and_write_predictions_pads_short_completion_list(tmp_path: Path):
    rm = _FakePolicyEnginePool(completion_factory=lambda prompts: ["only_one"])
    prompts_and_refs = [("p0", "r0", None), ("p1", "r1", None)]
    out_path = tmp_path / "out.jsonl"

    await predict_loop.generate_and_write_predictions(rm, prompts_and_refs, out_path=out_path)

    rows = [json.loads(line) for line in out_path.read_text().splitlines()]
    assert rows[0]["completion"] == "only_one"
    assert rows[1]["completion"] == ""


@pytest.mark.asyncio
async def test_run_predict_loop_writes_to_save_predict_dir(tmp_path: Path, monkeypatch):
    rm = _FakePolicyEnginePool()
    config = SimpleNamespace(save=str(tmp_path))
    monkeypatch.setattr(
        predict_loop,
        "render_eval_prompts",
        lambda cfg: [("hello prompt", "hello ref", None), ("bye prompt", "bye ref", None)],
    )

    await predict_loop.run_predict_loop(rm, config, train_step=17)

    out_path = tmp_path / "predict" / "predictions_step_17.jsonl"
    assert out_path.exists()
    rows = [json.loads(line) for line in out_path.read_text().splitlines()]
    assert len(rows) == 2
    assert rows[0]["prompt"] == "hello prompt"
    assert rows[0]["completion"] == "answer_hello prompt"
    assert all(set(row) == {"prompt", "reference", "completion"} for row in rows)
    assert rm.generated_samples is None
    assert rm.submitted == []


@pytest.mark.asyncio
async def test_run_predict_loop_skips_when_no_prompts(tmp_path: Path, monkeypatch):
    rm = _FakePolicyEnginePool()
    config = SimpleNamespace(save=str(tmp_path))
    monkeypatch.setattr(predict_loop, "render_eval_prompts", lambda cfg: [])

    await predict_loop.run_predict_loop(rm, config, train_step=3)

    assert not (tmp_path / "predict" / "predictions_step_3.jsonl").exists()
    assert rm.calls == []


@pytest.mark.asyncio
async def test_run_predict_loop_overwrites_existing_file(tmp_path: Path, monkeypatch):
    out_path = tmp_path / "predict" / "predictions_step_5.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("stale content from previous round\n")

    rm = _FakePolicyEnginePool()
    config = SimpleNamespace(save=str(tmp_path))
    monkeypatch.setattr(predict_loop, "render_eval_prompts", lambda cfg: [("p", "r", None)])

    await predict_loop.run_predict_loop(rm, config, train_step=5)

    rows = [json.loads(line) for line in out_path.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["prompt"] == "p"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("group_rm", "disable_async", "deferrable"),
    [(False, False, True), (False, True, True), (True, False, True), (True, True, True), (False, False, False)],
)
async def test_run_predict_loop_submits_eval_rewards(tmp_path, monkeypatch, group_rm, disable_async, deferrable):
    from relax.engine.rewards.async_eval import defer_eval_reward, eval_reward_deferral_enabled

    rm = _FakePolicyEnginePool()
    config = SimpleNamespace(
        save=str(tmp_path),
        custom_rm_path="example.reward_func",
        group_rm=group_rm,
        disable_eval_async_judge=disable_async,
        reward_key="train_score",
        eval_reward_key="eval_score",
    )
    scored = []
    requests = []

    def render(config, *, samples):
        samples.extend(Sample(index=i, group_index=i, prompt=f"p{i}", label=f"r{i}") for i in range(2))
        return [(sample.prompt, sample.label, None) for sample in samples]

    async def reward(args, sample_or_group, **kwargs):
        assert args is config
        assert kwargs == {"evaluation": True}
        requests.append(sample_or_group)
        if group_rm:
            assert len(sample_or_group) == 1
            sample = sample_or_group[0]
        else:
            sample = sample_or_group
        assert sample.response == f"answer_{sample.prompt}"
        assert sample.response_length == 1
        # Emulate an eligible scorer at the executor boundary, using the real
        # context collector so deferred work remains owned by the framework.
        if deferrable and defer_eval_reward(sample_or_group, kwargs):
            return [None] if group_rm else None
        scored.append(sample.index)
        value = {"eval_score": sample.index + 0.25}
        return [value] if group_rm else value

    monkeypatch.setattr(predict_loop, "render_eval_prompts", render)
    monkeypatch.setattr("relax.engine.rewards.async_rm", reward)
    monkeypatch.setattr("relax.engine.rewards.batched_async_rm", reward)

    await predict_loop.run_predict_loop(rm, config, train_step=17)

    assert not eval_reward_deferral_enabled()
    assert len(requests) == 2
    assert len(rm.submitted) == 1
    step, result, pending = rm.submitted[0]
    dataset = result.data["sft_predict"]
    assert step == 17
    assert dataset["samples"] is rm.generated_samples
    assert dataset["num_prompts"] == dataset["num_valid_prompts"] == 2
    assert dataset["group_size"] == 1
    assert dataset["truncated"] == [False, True]
    if deferrable and not disable_async:
        # Predict has returned while judge execution is still pending.
        assert scored == []
        assert dataset["rewards"] == [None, None]
        assert len(pending) == 2
        for call, request, sample in zip(pending, requests, dataset["samples"], strict=True):
            assert call.samples is request
            assert (call.samples[0] if group_rm else call.samples) is sample
            assert call.kwargs == {"evaluation": True}
    else:
        assert scored == [0, 1]
        assert dataset["rewards"] == [0.25, 1.25]
        assert pending == []
    rows = [json.loads(line) for line in (tmp_path / "predict" / "predictions_step_17.jsonl").read_text().splitlines()]
    assert all(set(row) == {"prompt", "reference", "completion"} for row in rows)


@pytest.mark.asyncio
async def test_score_predictions_propagates_errors_and_cancels_siblings(monkeypatch):
    from relax.engine.rewards.async_eval import eval_reward_deferral_enabled

    rm = _FakePolicyEnginePool()
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def reward(args, sample, **kwargs):
        if sample.index == 0:
            await started.wait()
            raise ValueError("judge failed")
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    monkeypatch.setattr("relax.engine.rewards.async_rm", reward)
    with pytest.raises(ValueError, match="judge failed"):
        await asyncio.wait_for(
            predict_loop.score_predictions(rm, SimpleNamespace(), 7, [Sample(index=0), Sample(index=1)]), timeout=5
        )

    assert cancelled.is_set()
    assert rm.submitted == []
    assert not eval_reward_deferral_enabled()


@pytest.mark.asyncio
async def test_generate_and_write_predictions_rejects_missing_scored_completions(tmp_path):
    rm = _FakePolicyEnginePool(completion_factory=lambda prompts: ["only_one"])
    out_path = tmp_path / "out.jsonl"

    with pytest.raises(ValueError, match="one completion per prompt"):
        await predict_loop.generate_and_write_predictions(
            rm, [("p0", "r0", None), ("p1", "r1", None)], out_path, samples=[Sample(index=0), Sample(index=1)]
        )

    assert not out_path.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("scoring", [False, True])
async def test_predict_workload_uses_managed_admission_and_returns_permits(monkeypatch, scoring):
    import httpx

    from relax.engine.inference.lifecycle import AdmissionLedger
    from relax.engine.inference.types import (
        LifecycleState,
        ModelSnapshot,
        ReplicaSnapshot,
        Role,
        RoleSnapshot,
        RoutingSpec,
    )
    from relax.engine.rollout.workload import RolloutWorkload
    from relax.utils import http_utils

    ledger = AdmissionLedger()
    snapshot = RoleSnapshot(
        Role.ROLLOUT,
        "task-epoch",
        models=(
            ModelSnapshot(
                "default",
                (ReplicaSnapshot("replica", LifecycleState.READY, "http://engine", direct_eligible=True),),
                state=LifecycleState.READY,
                admission=True,
            ),
        ),
        routing=RoutingSpec(default_model="default"),
    )

    async def observe(*args):
        return snapshot

    async def admit(role, model, trace):
        return ledger.admit(model)

    async def complete(permit, *args):
        await ledger.complete(permit)

    async def ready(*args):
        return None

    owner = SimpleNamespace(
        **{
            name: SimpleNamespace(remote=method)
            for name, method in {
                "snapshot": observe,
                "admit_request": admit,
                "complete_request": complete,
                "wait_ready": ready,
            }.items()
        }
    )
    requests = []

    def backend(request):
        assert ledger.permits
        payload = json.loads(request.content)
        requests.append(payload)
        return httpx.Response(
            200,
            json={
                "text": payload["text"] + " done",
                "meta_info": {"output_token_logprobs": [[-0.5, 42, None]], "finish_reason": {"type": "stop"}},
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(backend)) as client:
        monkeypatch.setattr(http_utils, "_http_client", client)
        workload = RolloutWorkload.__new__(RolloutWorkload)
        workload.args = SimpleNamespace(
            _inference_manager_handle=owner,
            lora_rank=8,
            lora_adapter_mode=True,
            use_distributed_post=scoring,
            sglang_speculative_algorithm=None,
        )
        remote_requests = []

        async def remote_send(url, body, headers, request_id):
            remote_requests.append(url)
            return await http_utils._request_once(client, url, body, headers)

        monkeypatch.setattr(
            http_utils, "_post_actors", [SimpleNamespace(do_request=SimpleNamespace(remote=remote_send))]
        )
        workload.inference = SimpleNamespace(resume_health_monitoring=ready)
        samples = [Sample(index=i, tokens=[10]) for i in range(2)] if scoring else None
        assert await workload.generate_predict(["one", "two"], samples=samples) == ["one done", "two done"]
        if scoring:
            assert all(sample.tokens == [10, 42] and sample.rollout_tokens == [10, 42] for sample in samples)
            assert all(sample.response_length == 1 and sample.rollout_log_probs == [-0.5] for sample in samples)
            assert all(sample.status is Sample.Status.COMPLETED for sample in samples)
            assert all(request["return_logprob"] for request in requests)
            from relax.engine.rollout.sglang_rollout import _post_generation

            output = await _post_generation(workload.args, None, "unused", {"text": "native"}, None, evaluation=True)
            assert output["text"] == "native done"
            assert len(remote_requests) == 3
        else:
            assert not remote_requests
    assert not ledger.permits
    assert all(request["sampling_params"]["max_new_tokens"] == 512 for request in requests[:2])
    assert all(request["lora_path"] == "relax_policy_lora" for request in requests[:2])
