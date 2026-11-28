# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import pytest

from examples.agentic_weight_audit.demo import run_demo, summarize


async def test_summary_loads_debug_dump_with_transfer_queue_metadata(tmp_path, monkeypatch):
    import json
    import sys

    import numpy as np
    import torch
    from transfer_queue.metadata import BatchMeta

    from examples.agentic_weight_audit.demo import main

    report = await run_demo()
    dump, output = tmp_path / "train.pt", tmp_path / "summary.json"
    torch.save(
        {
            "samples": [{"weight_version_audit": report}],
            "batch_meta": BatchMeta(global_indexes=[0], partition_ids=["train"], production_status=np.array([[True]])),
        },
        dump,
    )
    monkeypatch.setattr(sys, "argv", ["demo", "--train-dump", str(dump), "--output", str(output)])
    main()
    result = json.loads(output.read_text())
    assert result["reports"] == [report]
    assert result["summary"]["known_age_token_mean"] == 1.8


async def test_tool_wait_demo_preserves_masked_observation_and_versions():
    report = await run_demo()
    assert [(s["start"], s["end"]) for s in report["spans"]] == [(0, 4), (7, 13)]
    assert [s["weight_version"] for s in report["spans"]] == ["10", "12"]
    assert report["effective_tokens"] == 10
    assert report["version_missing_token_rate"] == 0
    assert summarize([report])["known_age_token_mean"] == 1.8
    assert summarize([report], unaudited_rows=1)["version_missing_token_rate"] is None


@pytest.mark.parametrize("gate,barrier_size", [(None, 2), ("train-1-0.pt", 2), ("long-ready", 2), ("long-ready", 4)])
def test_smoke_agent_preserves_full_history_and_emits_implicit_reward(tmp_path, monkeypatch, gate, barrier_size):
    import json

    import httpx

    from examples.agentic_weight_audit import agent

    original = [{"role": "user", "content": "Read 3."}]
    input_path, output_path = tmp_path / "input.json", tmp_path / "output.json"
    metadata = {"value": 3, "wait_for": gate, "signal_first_turn": gate == "train-1-0.pt"}
    metadata["first_turn_barrier_size"] = barrier_size
    input_path.write_text(json.dumps({"messages": original, "metadata": metadata}))
    if gate:
        monkeypatch.setenv("TASK15_CONTROL_DIR", str(tmp_path))
        for name in ["train-1-0.pt"] if gate == "train-1-0.pt" else [f"long-ready.{i}" for i in range(2)]:
            (tmp_path / name).touch()
    else:
        monkeypatch.delenv("TASK15_CONTROL_DIR", raising=False)
    for name, value in {
        "RELAX_INPUT_JSON": str(input_path),
        "RELAX_OUTPUT_JSON": str(output_path),
        "RELAX_BASE_URL": "http://test/agentic",
        "RELAX_SESSION_ID": "test-session",
    }.items():
        monkeypatch.setenv(name, value)
    requests = []
    waits = []

    def finish_remaining_first_turns(delay):
        waits.append(delay)
        for i in range(2, barrier_size):
            (tmp_path / f"long-ready.{i}").touch()

    monkeypatch.setattr(agent.time, "sleep", finish_remaining_first_turns)

    def respond(request):
        assert str(request.url) == "http://test/agentic/v1/chat/completions"
        assert request.headers["authorization"] == "Bearer test-session"
        requests.append(json.loads(request.content))
        content = "Ready." if len(requests) == 1 else "6."
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"role": "assistant", "content": content}}],
                "usage": {"completion_tokens": 2},
            },
        )

    client = httpx.Client
    monkeypatch.setattr(
        agent.httpx, "Client", lambda **kwargs: client(transport=httpx.MockTransport(respond), **kwargs)
    )
    agent.main()
    assert requests[0]["messages"] == original
    assert requests[1]["messages"] == original + [
        {"role": "assistant", "content": "Ready."},
        {"role": "user", "content": "The doubling tool returned 6. State this result."},
    ]
    output = json.loads(output_path.read_text())
    assert "messages" not in output
    assert output["reward"] == 4
    assert output["metadata"]["model_turns"] == 2
    assert waits == ([0.2] if gate == "long-ready" and barrier_size == 4 else [])
    if gate == "train-1-0.pt":
        assert (tmp_path / "long-ready.test-session").exists()
