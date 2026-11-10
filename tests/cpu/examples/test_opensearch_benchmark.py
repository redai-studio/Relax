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
def test_http_runner_keeps_history_within_32k(monkeypatch, benchmark, input_tokens, finish_reason, raises):
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
    history = [
        {"role": "user", "parts": [{"text": "original question"}]},
        {"role": "model", "parts": [{"text": "<tool_call>earlier call</tool_call>"}]},
        {"role": "user", "parts": [{"text": "entire earlier observation"}]},
    ]
    if raises:
        with pytest.raises(ValueError):
            runner.infer(history, "system")
        assert runner.error == "context_limit"
    else:
        runner.infer(history, "system")
    if input_tokens < 32768:
        assert captured[0]["max_tokens"] == 7
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
