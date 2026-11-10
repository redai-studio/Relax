# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Benchmark boundaries retain complete history and count every selected
case."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

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
    upstream = ModuleType("opensearch_infer.messages")
    upstream.to_qwen3vl_messages = lambda contents: list(contents)
    monkeypatch.setitem(sys.modules, "opensearch_infer.messages", upstream)
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


def test_missing_tools_cannot_silently_become_formal_benchmark(monkeypatch, benchmark, tmp_path):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark",
            "--label",
            "base",
            "--dataset",
            "vdr",
            "--data",
            str(tmp_path / "data.parquet"),
            "--processor",
            str(tmp_path / "model"),
            "--output",
            str(tmp_path / "result"),
            "--tool-cache",
            str(tmp_path / "cache"),
            "--allow-missing-tools",
        ],
    )
    with pytest.raises(SystemExit) as error:
        benchmark.main()
    assert error.value.code == 2
    assert not (tmp_path / "result").exists()


def test_image_search_cache_matches_pixels_across_urls(monkeypatch, benchmark, tmp_path):
    import io
    import json

    from PIL import Image

    image = io.BytesIO()
    Image.new("RGB", (32, 32), "red").save(image, format="PNG")
    monkeypatch.setattr(benchmark, "url_image_bytes", lambda _url: image.getvalue())
    monkeypatch.delenv("SR_MODEL_PATH", raising=False)
    calls = []

    def execute(*args, **_kwargs):
        calls.append(args)
        return "<observation>result</observation>", {}

    tools = SimpleNamespace(execute_tool=execute)
    benchmark.connect_tools(tmp_path, tools, SimpleNamespace())
    for url in ["https://example.com/first.png", "https://example.com/second.png"]:
        call = json.dumps({"name": "image_search", "parameters": {"url": url}})
        assert tools.execute_tool(call, {}, "case", 0, 0, str(tmp_path))[0] == "<observation>result</observation>"
    assert len(calls) == 1


def test_relax_evaluation_rejects_upload_validation_model(monkeypatch, benchmark, tmp_path):
    import json

    provenance = tmp_path / "model.json"
    provenance.write_text(json.dumps({"served_model_name": "baseline", "kind": "upload-validation", "epochs": 0}))
    monkeypatch.setenv("AGENT_MODEL", "baseline")
    monkeypatch.setenv("AGENT_PROVENANCE_FILE", str(provenance))
    monkeypatch.setattr(benchmark, "missing_tools", lambda: [])
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark",
            "--label",
            "relax",
            "--dataset",
            "vdr",
            "--max-cases",
            "1",
            "--data",
            str(tmp_path / "data.parquet"),
            "--processor",
            str(tmp_path / "model"),
            "--output",
            str(tmp_path / "result"),
            "--tool-cache",
            str(tmp_path / "cache"),
        ],
    )
    with pytest.raises(ValueError, match="not an upload-validation model"):
        benchmark.main()
    assert not (tmp_path / "result").exists()


@pytest.mark.parametrize(
    "text,answer",
    [
        ("red", "red"),
        ("<think>private reasoning</think>red", "red"),
        ("private reasoning</think><response>red</response>", "red"),
    ],
)
def test_direct_answer_excludes_reasoning(benchmark, text, answer):
    assert benchmark.direct_answer(text) == answer


@pytest.mark.parametrize("text", ["<think>unfinished", '<tool_call>{"name":"search"}</tool_call>'])
def test_direct_answer_rejects_unfinished_reasoning_or_tool_call(benchmark, text):
    with pytest.raises(ValueError):
        benchmark.direct_answer(text)


def test_direct_workflow_sends_one_question_without_reference_or_tools(benchmark):
    import json

    calls = []

    def infer(contents, system_instruction):
        calls.append((contents, system_instruction))
        return {"candidates": [{"content": {"parts": [{"text": "red"}]}}]}

    case = {"id": "one", "question": "What color?", "images": [b"image bytes"], "answer": "HIDDEN_REFERENCE"}
    trajectory = benchmark.direct_trajectory(case, SimpleNamespace(infer=infer))
    assert len(calls) == 1
    assert len(calls[0][0]) == 1
    assert calls[0][0][0]["role"] == "user"
    assert calls[0][0][0]["parts"][0]["text"] == case["question"]
    assert "HIDDEN_REFERENCE" not in json.dumps(calls)
    assert "<tools>" not in json.dumps(calls)
    assert len(trajectory["turns"]) == 1


@pytest.mark.parametrize("model_failure,judge_failure", [(False, False), (True, False), (False, True)])
def test_direct_main_requires_no_tools_and_retains_failed_cases(
    monkeypatch, benchmark, tmp_path, model_failure, judge_failure
):
    import json
    from threading import Barrier

    import httpx
    from openai import OpenAI

    barrier = Barrier(2)
    inference_calls, judge_calls = [], []

    class Runner:
        def __init__(self, *_args):
            self.template_sha256 = "template"
            self.chat_template_kwargs = {"enable_thinking": True, "preserve_thinking": True}
            self.requests, self.error, self.last_response = [], None, None

        def infer(self, contents, **_kwargs):
            inference_calls.append(contents)
            if model_failure and contents[0]["parts"][0]["text"] == "fail":
                raise ConnectionError("model unavailable")
            return {"candidates": [{"content": {"parts": [{"text": "answer"}]}}]}

    def unexpected_tools(*_args):
        pytest.fail("Direct workflow must not inspect or connect tools")

    monkeypatch.setattr(benchmark, "missing_tools", unexpected_tools)
    monkeypatch.setattr(benchmark, "connect_tools", unexpected_tools)
    monkeypatch.setattr(benchmark, "HTTPRunner", Runner)
    monkeypatch.setattr(benchmark.subprocess, "check_output", lambda *_args, **_kwargs: benchmark.UPSTREAM_REVISION)
    monkeypatch.setattr(
        benchmark,
        "read_cases",
        lambda *_args: [
            {"id": question, "question": question, "answer": "reference", "images": []}
            for question in ["pass", "fail"]
        ],
    )
    upstream = ModuleType("eval_with_gpt4o")
    upstream.JUDGE_PROMPT = "{question} {correct_answer} {response}"
    upstream.parse_judge_response = lambda _text: {"acc": 1}
    monkeypatch.setitem(sys.modules, "eval_with_gpt4o", upstream)

    def judge_request(**_kwargs):
        if not model_failure:
            barrier.wait(timeout=5)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="correct: yes"))], model="judge"
        )

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=judge_request)))

    def transport(request):
        judge_calls.append(request)
        if len(judge_calls) == 1:
            return httpx.Response(500, json={"error": {"message": "temporary judge failure"}})
        return httpx.Response(
            200,
            json={
                "id": "verdict",
                "object": "chat.completion",
                "created": 0,
                "model": "judge",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "correct: yes"}}],
            },
        )

    def make_client(**kwargs):
        if judge_failure and kwargs["base_url"].endswith(":2/v1"):
            kwargs["http_client"].close()
            kwargs["http_client"] = httpx.Client(transport=httpx.MockTransport(transport))
            return OpenAI(**kwargs)
        return client

    monkeypatch.setattr(sys.modules["openai"], "OpenAI", make_client)
    for name, value in {
        "AGENT_MODEL": "baseline",
        "AGENT_BASE_URL": "http://127.0.0.1:1/v1",
        "JUDGE_API_KEY": "test",
        "JUDGE_BASE_URL": "http://127.0.0.1:2/v1",
        "JUDGE_MODEL": "judge",
        "OPENSEARCH_ROOT": str(tmp_path),
    }.items():
        monkeypatch.setenv(name, value)
    provenance = tmp_path / "model.json"
    provenance.write_text(json.dumps({"served_model_name": "baseline", "hashes_verified": True}))
    monkeypatch.setenv("AGENT_PROVENANCE_FILE", str(provenance))
    data = tmp_path / "data.parquet"
    data.write_bytes(b"dataset")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark",
            "--workflow",
            "direct",
            "--workers",
            "2",
            "--judge-qps",
            "0",
            "--dataset",
            "vdr",
            "--label",
            "base",
            "--data",
            str(data),
            "--processor",
            str(tmp_path),
            "--output",
            str(tmp_path / "result"),
            "--max-cases",
            "2",
        ],
    )
    benchmark.main()
    result = json.loads((tmp_path / "result/summary.json").read_text())
    assert result["expected_cases"] == 2 and result["failures"] == int(model_failure)
    assert result["pass_at_1"] == (0.5 if model_failure else 1.0)
    assert result["tool_errors"] == 0 and result["protocol"]["tool_cache"] is None
    assert len(inference_calls) == 2
    if judge_failure:
        assert len(judge_calls) == 3


@pytest.mark.parametrize("invalid", [None, "protocol", "resume", "partial", "smoke"])
def test_comparison_needs_only_full_simplevqa_results(monkeypatch, tmp_path, invalid):
    import json

    monkeypatch.syspath_prepend(str(TOOLS))
    spec = importlib.util.spec_from_file_location("opensearch_comparison", TOOLS / "compare_benchmark.py")
    comparison = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(comparison)
    for label, score in [("base", 0.4), ("relax", 0.5)]:
        summary = {
            "protocol": {"smoke": False, "evaluator_sha256": "same-source"},
            "expected_cases": 1013,
            "model_source": {"kind": "base" if label == "base" else "trained-final"},
            "deduplicated_cases": 981,
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
        directory = tmp_path / label / "simplevqa"
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
        assert set(result) == {"simplevqa"}
        assert result["simplevqa"]["pass_at_1_difference"] == pytest.approx(0.1)


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


@pytest.fixture
def unfinished_benchmark(monkeypatch, benchmark, tmp_path):
    import json

    calls, verdicts = [], []
    cases = [{"id": str(index), "question": str(index), "answer": "reference", "images": []} for index in range(1013)]

    class Runner:
        def __init__(self, *_args):
            self.template_sha256, self.chat_template_kwargs = "native-template", {}
            self.requests, self.error, self.last_response = [], None, None

        def infer(self, contents, **_kwargs):
            question = contents[0]["parts"][0]["text"]
            calls.append(question)
            if question == "0":
                self.error = "context_limit"
                raise ValueError("Context exhausted")
            return {"candidates": [{"content": {"parts": [{"text": "answer"}]}}]}

    def judge_request(**kwargs):
        verdicts.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="correct: yes"))], model="judge"
        )

    upstream = tmp_path / "upstream"
    monkeypatch.setattr(benchmark, "HTTPRunner", Runner)
    monkeypatch.setattr(benchmark, "read_cases", lambda *_args: cases)
    monkeypatch.setattr(benchmark.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(
        benchmark.subprocess,
        "check_output",
        lambda command, **_kwargs: (
            benchmark.UPSTREAM_REVISION if command[2] == str(upstream) else "current-evaluator-commit"
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "openai",
        SimpleNamespace(
            OpenAI=lambda **_kwargs: SimpleNamespace(
                chat=SimpleNamespace(completions=SimpleNamespace(create=judge_request))
            )
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "eval_with_gpt4o",
        SimpleNamespace(
            JUDGE_PROMPT="{question} {correct_answer} {response}", parse_judge_response=lambda _text: {"acc": 1}
        ),
    )
    provenance = tmp_path / "model-provenance.json"
    provenance.write_text(
        json.dumps(
            {
                "kind": "base",
                "served_model_name": "baseline",
                "hashes_verified": True,
                "model_id": 5859,
                "version": 1,
                "sub_version": 1761219912,
                "files_sha256": {"config.json": "trusted-model-hash"},
            }
        )
    )
    for key, value in {
        "OPENSEARCH_ROOT": str(upstream),
        "AGENT_MODEL": "baseline",
        "AGENT_BASE_URL": "http://127.0.0.1:1/v1",
        "AGENT_PROVENANCE_FILE": str(provenance),
        "JUDGE_MODEL": "judge",
        "JUDGE_BASE_URL": "http://127.0.0.1:2/v1",
        "JUDGE_API_KEY": "fixture",
    }.items():
        monkeypatch.setenv(key, value)
    data, overlap, output = tmp_path / "dataset.parquet", tmp_path / "overlap.json", tmp_path / "output"
    data.write_bytes(b"complete-dataset")
    overlap.write_text('{"questions":[],"images":[]}')
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark",
            "--dataset",
            "simplevqa",
            "--label",
            "base",
            "--data",
            str(data),
            "--processor",
            str(tmp_path),
            "--output",
            str(output),
            "--overlap-index",
            str(overlap),
            "--judge-qps",
            "0",
        ],
    )
    benchmark.main()
    assert json.loads((output / "model-provenance.json").read_text()) == json.loads(provenance.read_text())
    (output / "summary.json").unlink()
    records = output / "trajectories.jsonl"
    records.write_text(
        "".join(
            line
            for line in records.read_text().splitlines(keepends=True)
            if json.loads(line)["id"] not in {"1011", "1012"}
        )
    )
    protocol = json.loads((output / "protocol.json").read_text())
    protocol["evaluator_sha256"] = "previous-source-hash"
    (output / "protocol.json").write_text(json.dumps(protocol))
    (output / "evaluator.json").write_text(
        json.dumps(
            {"git_branch": "original", "git_commit": "original-commit", "source_sha256": "previous-source-hash"}
        )
    )
    calls.clear()
    verdicts.clear()
    monkeypatch.setattr(sys, "argv", [*sys.argv, "--resume"])
    return SimpleNamespace(output=output, data=data, provenance=provenance, calls=calls, verdicts=verdicts)


def test_resume_appends_without_repeating_completed_cases_and_keeps_full_denominator(benchmark, unfinished_benchmark):
    import json

    run = unfinished_benchmark
    original = {
        name: (run.output / name).read_bytes() for name in ("protocol.json", "evaluator.json", "model-provenance.json")
    }
    prefix = (run.output / "trajectories.jsonl").read_bytes()
    benchmark.main()
    assert sorted(run.calls) == ["1011", "1012"] and len(run.verdicts) == 2
    assert (run.output / "trajectories.jsonl").read_bytes().startswith(prefix)
    for name, payload in original.items():
        assert (run.output / name).read_bytes() == payload
    records = [json.loads(line) for line in (run.output / "trajectories.jsonl").read_text().splitlines()]
    assert len(records) == len({record["id"] for record in records}) == 1013
    summary = json.loads((run.output / "summary.json").read_text())
    assert summary["expected_cases"] == summary["deduplicated_cases"] == 1013
    assert summary["correct"] == 1012 and summary["failures"] == summary["truncations"] == 1
    assert summary["pass_at_1"] == pytest.approx(1012 / 1013)
    assert summary["protocol"]["evaluator_sha256"] == "previous-source-hash"
    assert summary["protocol"]["resume_evaluator_sha256s"] == [benchmark.sha256((TOOLS / "benchmark.py").read_bytes())]
    assert summary["evaluator"]["git_commit"] == "original-commit"
    source = json.loads((run.output / "resume-source.json").read_text())
    assert source["attempts"][0]["completed_cases"] == 1011
    assert source["attempts"][0]["protocol_differences"] == ["evaluator_sha256"]
    assert source["attempts"][0]["evaluator"]["git_commit"] == "current-evaluator-commit"
    assert source["attempts"][0]["evaluator"]["source_sha256"] == benchmark.sha256(
        (TOOLS / "benchmark.py").read_bytes()
    )
    with pytest.raises(ValueError, match="unfinished"):
        benchmark.main()
    assert len(run.calls) == 2


@pytest.mark.parametrize(
    "damage", ["model", "data", "judge", "protocol", "duplicate", "unknown", "partial", "incomplete"]
)
def test_resume_rejects_mismatched_or_invalid_saved_runs(monkeypatch, benchmark, unfinished_benchmark, damage):
    import json

    run = unfinished_benchmark
    trajectories = run.output / "trajectories.jsonl"
    if damage == "model":
        source = json.loads(run.provenance.read_text())
        source["files_sha256"]["config.json"] = "different-model"
        run.provenance.write_text(json.dumps(source))
    elif damage == "data":
        run.data.write_bytes(b"different-dataset")
    elif damage == "judge":
        monkeypatch.setenv("JUDGE_MODEL", "different-judge")
    elif damage == "protocol":
        protocol = json.loads((run.output / "protocol.json").read_text())
        protocol["temperature"] = 1
        (run.output / "protocol.json").write_text(json.dumps(protocol))
    else:
        rows = trajectories.read_text().splitlines(keepends=True)
        if damage == "duplicate":
            rows.append(rows[0])
        elif damage == "partial":
            rows.append('{"id":')
        else:
            record = json.loads(rows[0])
            if damage == "unknown":
                record["id"] = "unknown-case"
            else:
                record.pop("correct")
            rows[0] = json.dumps(record) + "\n"
        trajectories.write_text("".join(rows))
    original = trajectories.read_bytes()
    with pytest.raises(ValueError):
        benchmark.main()
    assert not run.calls and not run.verdicts
    assert trajectories.read_bytes() == original
    assert not (run.output / "summary.json").exists()
    assert not (run.output / "resume-source.json").exists()
