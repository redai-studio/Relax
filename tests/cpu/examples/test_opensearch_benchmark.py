# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Benchmark boundaries retain complete history and count every selected
case."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


TOOLS = Path(__file__).resolve().parents[3] / "examples/opensearch_vl/tools"


@pytest.fixture
def benchmark(monkeypatch):
    monkeypatch.syspath_prepend(str(TOOLS))
    spec = importlib.util.spec_from_file_location("opensearch_benchmark", TOOLS / "benchmark.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "input_tokens,finish_reason,raises", [(32760, "stop", False), (32768, "stop", True), (32760, "length", True)]
)
@pytest.mark.parametrize("chat_kwargs", [{"enable_thinking": True, "preserve_thinking": True}, {}])
def test_http_runner_keeps_history_within_32k(
    monkeypatch, benchmark, input_tokens, finish_reason, raises, chat_kwargs
):
    captured = []

    def create(**kwargs):
        captured.append(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="<response>answer</response>"), finish_reason=finish_reason
                )
            ],
            model="baseline",
            usage=None,
        )

    runner = benchmark.HTTPRunner.__new__(benchmark.HTTPRunner)
    runner.processor = SimpleNamespace(
        apply_chat_template=lambda *_args, **_kwargs: {"input_ids": SimpleNamespace(shape=(1, input_tokens))}
    )
    runner.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    runner.model, runner.requests, runner.error = "baseline", [], None
    runner.chat_template_kwargs = chat_kwargs
    history = [
        {"role": "user", "parts": [{"text": "original question"}]},
        {"role": "model", "parts": [{"text": "<tool_call>earlier call</tool_call>"}]},
        {"role": "user", "parts": [{"text": "entire earlier observation"}]},
    ]
    if raises:
        with pytest.raises(ValueError):
            runner.infer(history, "system")
        assert runner.error == "context_limit"
        if finish_reason == "length":
            assert runner.last_response == "<response>answer</response>"
    else:
        runner.infer(history, "system")
    if input_tokens < 32768:
        assert captured[0]["max_tokens"] == 7
        assert captured[0]["extra_body"]["chat_template_kwargs"] == chat_kwargs
        assert len(captured[0]["messages"]) == 4
        assert captured[0]["messages"][1]["content"][0]["text"] == "original question"
        assert captured[0]["messages"][-1]["content"][0]["text"] == "entire earlier observation"
    else:
        assert not captured


@pytest.mark.parametrize("invalid", [None, "protocol", "resume", "partial", "smoke"])
def test_comparison_needs_only_full_vstar_results(monkeypatch, tmp_path, invalid):
    import json

    monkeypatch.syspath_prepend(str(TOOLS))
    spec = importlib.util.spec_from_file_location("opensearch_comparison", TOOLS / "compare_benchmark.py")
    comparison = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(comparison)
    for label, score in [("base", 0.4), ("relax", 0.5)]:
        summary = {
            "protocol": {"smoke": False, "evaluator_sha256": "same-source"},
            "expected_cases": 191,
            "model_source": {"kind": "base" if label == "base" else "trained-final"},
            "deduplicated_cases": 191,
            "pass_at_1": score,
            "deduplicated_pass_at_1": score,
        }
        if label == "relax":
            if invalid == "protocol":
                summary["protocol"]["evaluator_sha256"] = "different-source"
            elif invalid == "resume":
                summary["protocol"]["resume_evaluator_sha256s"] = ["different-source"]
            elif invalid == "partial":
                summary["expected_cases"] = 100
            elif invalid == "smoke":
                summary["protocol"]["smoke"] = True
        directory = tmp_path / label / "vstar"
        directory.mkdir(parents=True)
        (directory / "summary.json").write_text(json.dumps(summary))
    output = tmp_path / "comparison.json"
    monkeypatch.setattr(sys, "argv", ["compare", "--root", str(tmp_path), "--output", str(output)])
    if invalid:
        with pytest.raises(ValueError):
            comparison.main()
        assert not output.exists()
    else:
        comparison.main()
        result = json.loads(output.read_text())
        assert set(result) == {"vstar"}
        assert result["vstar"]["pass_at_1_difference"] == pytest.approx(0.1)


def test_qwen3_vl_instruct_keeps_native_history_without_thinking_flags(benchmark):
    from jinja2 import Environment

    template = "{% for message in messages %}{{ message.content }}{% endfor %}"
    processor = type("Qwen3VLProcessor", (), {})()
    processor.chat_template = template
    processor.tokenizer = SimpleNamespace(chat_template=template)
    assert benchmark.inference_chat_kwargs(processor) == {}
    actual = benchmark.preserve_history_template(processor)
    history = "<think>original reasoning</think><tool_call>original call</tool_call>"
    assert Environment().from_string(actual).render(messages=[{"content": history}]) == history
    assert processor.tokenizer.chat_template == template


@pytest.mark.parametrize(
    "processor_type,template", [("UnknownProcessor", "{{ content }}"), ("Qwen3VLProcessor", "<think>")]
)
def test_unsupported_reasoning_templates_still_fail(benchmark, processor_type, template):
    processor = type(processor_type, (), {})()
    processor.chat_template = template
    processor.tokenizer = SimpleNamespace(chat_template=template)
    assert benchmark.inference_chat_kwargs(processor)["enable_thinking"] is True
    with pytest.raises(ValueError, match="supported Qwen history template"):
        benchmark.preserve_history_template(processor)
