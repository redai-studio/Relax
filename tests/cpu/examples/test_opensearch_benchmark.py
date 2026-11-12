# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Benchmark boundaries retain complete history and count every selected
case."""

import importlib.util
import io
import json
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


@pytest.fixture
def vstar(monkeypatch):
    monkeypatch.syspath_prepend(str(TOOLS))
    spec = importlib.util.spec_from_file_location("opensearch_vstar", TOOLS / "vstar_benchmark.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "input_tokens,finish_reason,error,chat_kwargs",
    [
        (32760, "stop", None, {"enable_thinking": True, "preserve_thinking": True}),
        (32760, "stop", None, {}),
        (32768, "stop", "context_limit", {}),
        (32760, "length", "context_limit", {}),
        (32760, "abort", "server_finish_error", {}),
        (32760, None, "server_finish_error", {}),
    ],
)
def test_http_runner_keeps_history_within_32k(monkeypatch, benchmark, input_tokens, finish_reason, error, chat_kwargs):
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
    if error:
        with pytest.raises(ValueError):
            runner.infer(history, "system")
        assert runner.error == error
        if input_tokens < 32768:
            assert runner.last_response == "<response>answer</response>"
            assert runner.requests[-1]["finish_reason"] == finish_reason
    else:
        runner.infer(history, "system")
    if input_tokens < 32768:
        assert captured[0]["max_tokens"] == 7
        assert captured[0]["extra_body"]["chat_template_kwargs"] == chat_kwargs
        assert len(captured[0]["messages"]) == 4
        assert captured[0]["messages"][1]["content"][0]["text"] == "original question"
        assert captured[0]["messages"][2] == {
            "role": "assistant",
            "content": [{"type": "text", "text": "<tool_call>earlier call</tool_call>"}],
        }
        assert captured[0]["messages"][-1]["content"][0]["text"] == "entire earlier observation"
    else:
        assert not captured


@pytest.mark.parametrize("invalid", [None, "protocol", "partial", "smoke", "infrastructure", "server_abort"])
def test_comparison_needs_only_full_vstar_results(monkeypatch, tmp_path, invalid):
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
            elif invalid == "partial":
                summary["expected_cases"] = 100
            elif invalid == "smoke":
                summary["protocol"]["smoke"] = True
            elif invalid == "infrastructure":
                summary["errors"] = {"APIConnectionError": 1}
            elif invalid == "server_abort":
                summary["errors"] = {"server_finish_error": 1}
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


@pytest.mark.parametrize(
    "text,answer",
    [
        ("<response>A</response>", "A"),
        ("(B) blue", "B"),
        ("C. red", "C"),
        ("<response>Explanation.\nThe correct answer is: **(A) rubber**</response>", "A"),
        ("Explanation.\n**Answer: (B)**", "B"),
        (r"<response>Explanation." + "\n" + r"\boxed{D}</response>", "D"),
        ("Considered (C).</think><response>A</response>", "A"),
        ("<response>(A) red\n(B) blue</response>", None),
        ("Answer: (A) or B", None),
        ("<response>A</response><response>B</response>", None),
        ("<response>A</response><response>B", None),
        ("<response>A</response></response>", None),
        ("<response>\nA", None),
        ("No option selected.", None),
    ],
)
def test_final_option_extracts_explicit_answers_without_guessing(vstar, text, answer):
    assert vstar.final_option(text) == answer


def test_crop_workflow_maps_display_coordinates_to_original_regions(vstar, monkeypatch, tmp_path):
    from PIL import Image, ImageDraw

    data = tmp_path / "data"
    data.mkdir()
    image = Image.new("RGB", (2560, 1280), "blue")
    ImageDraw.Draw(image).rectangle((640, 319, 1279, 639), fill="red")
    image.save(data / "scene.png")
    case = {"question_id": 0, "category": "direct_attributes", "image": "scene.png", "text": "Color?", "label": "A"}
    questions = data / "test_questions.jsonl"
    questions.write_text(json.dumps(case) + "\n")
    manifest = {"files_sha256": {questions.name: vstar.file_hash(questions)}}
    (data / "manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(vstar, "prepare", lambda _: manifest)
    provenance = tmp_path / "model.json"
    provenance.write_text(json.dumps({"hashes_verified": True, "kind": "base", "served_model_name": "test-model"}))
    overlap = tmp_path / "overlap.json"
    overlap.write_text('{"questions":[],"images":[]}')
    calls = [
        {"image": "img_1", "x": 128, "y": 64, "width": 768, "height": 384},
        {"image": "img_2", "x": 171, "y": 85, "width": 341, "height": 171},
    ]
    observed_pixels = []

    class Runner:
        def __init__(self, *_args):
            self.processor = SimpleNamespace(image_processor=SimpleNamespace(patch_size=16, merge_size=2))
            self.template_sha256 = "test-template"
            self.requests, self.error, self.last_response = [], None, None

        def infer(self, history, **_kwargs):
            turn = len(self.requests)
            self.requests.append({})
            if turn < 2:
                text = "<tool_call>" + json.dumps({"name": "crop", "arguments": calls[turn]}) + "</tool_call>"
            else:
                import base64

                payload = next(part["inline_data"]["data"] for part in history[-1]["parts"] if "inline_data" in part)
                with Image.open(io.BytesIO(base64.b64decode(payload))) as shown:
                    observed_pixels.append(shown.getextrema())
                text = "<response>A</response>"
            return {"candidates": [{"content": {"parts": [{"text": text}]}}]}

    monkeypatch.setattr(vstar, "HTTPRunner", Runner)
    monkeypatch.setattr("openai.OpenAI", lambda **kwargs: SimpleNamespace(close=kwargs["http_client"].close))
    monkeypatch.setenv("AGENT_MODEL", "test-model")
    monkeypatch.setenv("AGENT_BASE_URL", "http://127.0.0.1:1/v1")
    monkeypatch.setenv("RELAX_GIT_COMMIT", "0" * 40)
    monkeypatch.chdir(tmp_path)  # Archive checkouts may have no Git metadata.
    output = tmp_path / "result"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "vstar",
            "--data",
            str(data),
            "--processor",
            str(tmp_path),
            "--provenance",
            str(provenance),
            "--overlap-index",
            str(overlap),
            "--output",
            str(output),
            "--label",
            "base",
            "--workers",
            "1",
            "--max-cases",
            "1",
            "--max-turns",
            "3",
            "--image-tokens",
            "512",
        ],
    )
    vstar.main()
    assert json.loads((output / "evaluator.json").read_text())["git_commit"] == "0" * 40
    record = json.loads((output / "trajectories.jsonl").read_text())
    assert record["correct"] and "error" not in record
    assert [call["source_box"] for call in record["tool_calls"]] == [[320, 160, 2240, 1120], [320, 159, 960, 480]]
    assert all(call["success"] for call in record["tool_calls"])
    images = [entry for entry in record["trajectory"] if "image" in entry]
    assert [entry["source_size"] for entry in images] == [[2560, 1280], [1920, 960], [640, 321]]
    assert [entry["display_size"] for entry in images] == [[1024, 512], [1024, 512], [640, 320]]
    assert observed_pixels == [((255, 255), (0, 0), (0, 0))]


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
