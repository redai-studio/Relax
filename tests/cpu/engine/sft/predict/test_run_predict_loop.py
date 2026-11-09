# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Unit tests for the SFT periodic predict loop."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from relax.engine.sft.dataset.sample import CanonicalMessage, CanonicalSample
from relax.engine.sft.predict import loop as predict_loop


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


def test_render_eval_prompts_uses_seeded_random_eval_rows(monkeypatch):
    accessed_indices = []

    class _FakeDataset:
        def __init__(self, **kwargs):  # noqa: ARG002
            pass

        def __len__(self):
            return 10

        def get_canonical_sample(self, idx):
            accessed_indices.append(idx)
            return _sample([("user", f"q{idx}", False), ("assistant", f"a{idx}", True)])

        def stop(self):
            pass

    class _FakeTokenizer:
        def apply_chat_template(self, messages, **kwargs):  # noqa: ARG002
            return messages[-1]["content"]

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

    rendered = predict_loop.render_eval_prompts(config)
    _, expected_eval_indices = predict_loop.resolve_sft_split_indices(10, 0.2, seed=42)

    assert accessed_indices == list(expected_eval_indices)
    assert [reference for _, reference, _ in rendered] == [f"a{idx}" for idx in expected_eval_indices]


class _FakeRolloutManager:
    def __init__(self, completion_factory=None):
        self.calls: list[list[str]] = []
        self.completion_factory = completion_factory or (lambda prompts: [f"answer_{p}" for p in prompts])

    async def generate_predict(self, prompts: list[str], multimodal_inputs_list=None) -> list[str]:
        self.calls.append(list(prompts))
        return self.completion_factory(prompts)


@pytest.mark.asyncio
async def test_generate_and_write_predictions_writes_jsonl_with_three_fields(tmp_path: Path):
    rm = _FakeRolloutManager()
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
    rm = _FakeRolloutManager()
    prompts_and_refs = [(f"p{i}", "ref", None) for i in range(7)]
    out_path = tmp_path / "predictions.jsonl"

    await predict_loop.generate_and_write_predictions(rm, prompts_and_refs, out_path=out_path)

    # All 7 prompts in a single generate_predict call (no batching).
    assert [len(c) for c in rm.calls] == [7]
    assert rm.calls[0] == [f"p{i}" for i in range(7)]


@pytest.mark.asyncio
async def test_generate_and_write_predictions_pads_short_completion_list(tmp_path: Path):
    rm = _FakeRolloutManager(completion_factory=lambda prompts: ["only_one"])
    prompts_and_refs = [("p0", "r0", None), ("p1", "r1", None)]
    out_path = tmp_path / "out.jsonl"

    await predict_loop.generate_and_write_predictions(rm, prompts_and_refs, out_path=out_path)

    rows = [json.loads(line) for line in out_path.read_text().splitlines()]
    assert rows[0]["completion"] == "only_one"
    assert rows[1]["completion"] == ""


@pytest.mark.asyncio
async def test_run_predict_loop_writes_to_save_predict_dir(tmp_path: Path, monkeypatch):
    rm = _FakeRolloutManager()
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


@pytest.mark.asyncio
async def test_run_predict_loop_skips_when_no_prompts(tmp_path: Path, monkeypatch):
    rm = _FakeRolloutManager()
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

    rm = _FakeRolloutManager()
    config = SimpleNamespace(save=str(tmp_path))
    monkeypatch.setattr(predict_loop, "render_eval_prompts", lambda cfg: [("p", "r", None)])

    await predict_loop.run_predict_loop(rm, config, train_step=5)

    rows = [json.loads(line) for line in out_path.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["prompt"] == "p"
