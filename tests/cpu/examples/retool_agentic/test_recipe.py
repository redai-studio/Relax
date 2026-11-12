# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Data preparation, verifier, sandbox client and checkpoint contracts."""

import asyncio
import json
import random
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer, ThreadingHTTPServer
from threading import BoundedSemaphore, Thread

import pytest

from examples.retool_agentic.checkpoint import validate_checkpoint
from examples.retool_agentic.prepare_data import LEAD, TAIL, deduplicate
from examples.retool_agentic.protocol import PREFIX, feedback_for_turn
from examples.retool_agentic.reward import score
from examples.retool_agentic.sandbox import SandboxClient, parse_result


@pytest.mark.parametrize(
    "text,label,counts,expected",
    [
        ("Answer: 2", "2", {}, 1),
        ("Answer: 3", "2", {"timeout": 3}, -1.3),
        ("Answer: 2", "2", {"ok": 10, "code_error": 2}, 0.8),
        ("Answer: 3 then \\boxed{2}", "2", {}, -1),
        ("\\boxed{2}", "2", {}, 1),
    ],
)
def test_retool_reward_matches_yuanfan(text, label, counts, expected):
    result = score(text, label, counts)
    assert result["score"] == expected and result["reward_floor"] is None
    assert result["acc"] == (result["math_score"] == 1)


def test_retool_unknown_executor_status_is_not_negative_reward():
    with pytest.raises(ValueError):
        score("Answer: 2", "2", {"http_500": 1})
    with pytest.raises(RuntimeError):
        parse_result(dict(status="InternalError"))


@pytest.mark.parametrize("return_code,expected", [(0, "ok"), (1, "code_error")])
def test_retool_sandbox_finished_checks_process_exit_code(return_code, expected):
    result = parse_result(
        dict(status="Success", run_result=dict(status="Finished", return_code=return_code, stderr="execution output"))
    )
    assert result["status"] == expected and result["stderr"] == "execution output"
    assert score("Answer: 2", "2", {result["status"]: 1})["score"] == (1 if expected == "ok" else 0.9)


def test_retool_http_client_uses_actual_response():

    received = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            received.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            self.send_response(200)
            self.end_headers()
            self.wfile.write(
                json.dumps(dict(status="Success", run_result=dict(status="Finished", stdout="2"))).encode()
            )

        def log_message(self, *args):
            pass

    with HTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = Thread(target=server.serve_forever)
        thread.start()
        try:
            result = asyncio.run(SandboxClient(f"http://127.0.0.1:{server.server_port}/run_code").execute("print(2)"))
        finally:
            server.shutdown()
            thread.join()
    assert result == dict(status="ok", stdout="2", stderr="")
    assert len(received) == 1 and received[0]["code"] == "print(2)"
    assert received[0]["run_timeout"] == 30 and received[0]["memory_limit_MB"] == 1024


def test_retool_http_client_waits_for_queued_execution_without_retry(monkeypatch):
    slots = BoundedSemaphore(1)
    received = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            received.append(body["code"])
            with slots:
                time.sleep(0.2)
                payload = dict(status="Success", run_result=dict(status="Finished", stdout=body["code"]))
                self.send_response(200)
                self.end_headers()
                self.wfile.write(json.dumps(payload).encode())

        def log_message(self, *args):
            pass

    original_open = urllib.request.OpenerDirector.open

    def scaled_open(opener, request, *, timeout):
        # Exercise real socket deadlines and queueing without a 70-second test.
        return original_open(opener, request, timeout=None if timeout is None else timeout * 0.005)

    monkeypatch.setattr(urllib.request.OpenerDirector, "open", scaled_open)
    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = Thread(target=server.serve_forever)
        thread.start()

        async def queued_calls():
            client = SandboxClient(f"http://127.0.0.1:{server.server_port}/run_code")
            return await asyncio.gather(*(client.execute(str(i)) for i in range(3)))

        try:
            results = asyncio.run(queued_calls())
        finally:
            server.shutdown()
            thread.join()
    assert [result["stdout"] for result in results] == ["0", "1", "2"]
    assert sorted(received) == ["0", "1", "2"]


@pytest.mark.parametrize("character", ["a", "界"])
@pytest.mark.parametrize("extra_bytes", [0, 1])
def test_retool_code_limit_counts_utf8_bytes_before_submission(character, extra_bytes):
    width = len(character.encode("utf-8"))
    code = "#" + character * (65535 // width) + "a" * (65535 % width + extra_bytes)
    calls = []

    class Client:
        async def execute(self, source):
            calls.append(source)
            return dict(status="ok", stdout="2", stderr="")

    assert len(code.encode("utf-8")) == 65536 + extra_bytes
    feedback, executed = asyncio.run(feedback_for_turn(f"<code>\n```python\n{code}\n```\n</code>", Client()))
    if extra_bytes:
        assert feedback == dict(
            status="invalid_call", stdout="", stderr="Python source exceeds sandbox 65536-byte limit"
        )
        assert executed is None and calls == []
        assert score("Answer: 2", "2", {feedback["status"]: 1})["score"] == 0.9
    else:
        assert feedback == dict(status="ok", stdout="2", stderr="")
        assert executed == code and calls == [code]


def test_retool_dapo_dedup_and_conflicting_labels():
    def row(uid, question, gold):
        return dict(
            data_source="math_dapo",
            prompt=[dict(role="user", content=LEAD + question + TAIL)],
            reward_model=dict(ground_truth=gold, style="rule"),
            extra_info=dict(index=uid),
        )

    a = row(1, "1+1?", "2")
    cases, report = deduplicate([a, a, row(2, "bad gold", "3"), row(3, "bad gold", "4")])
    assert len(cases) == 1 and cases[0]["label"] == "2"
    assert cases[0]["prompt"][0]["content"] == PREFIX + "1+1?"
    assert {r["source_id"] for r in report["quarantine_records"]} == {2, 3}
    with pytest.raises(ValueError, match="payload"):
        deduplicate([a, row(1, "1+1?", "5")])


def test_retool_resume_requires_matching_sampling_state(tmp_path):
    assert validate_checkpoint(tmp_path) is None
    (tmp_path / "junk").touch()
    with pytest.raises(ValueError, match="Nonempty"):
        validate_checkpoint(tmp_path)
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("24")
    with pytest.raises(ValueError, match="sampling cursor"):
        validate_checkpoint(tmp_path)
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    state_path = dataset / "retool_24.json"
    state = dict(cursor=24, rng_state_version=1, rng_state=random.Random(0).getstate())
    state_path.write_text(json.dumps(state))
    with pytest.raises(ValueError, match="matching ReTool cursor"):
        validate_checkpoint(tmp_path)
    state["cursor"] = 25
    state_path.write_text(json.dumps(state))
    # Native model/optimizer validation belongs to the framework load path.
    assert validate_checkpoint(tmp_path) == 24


def test_retool_resume_accepts_only_initialization_without_rollouts(tmp_path):
    (tmp_path / "transformer_config.json").write_text("{}")
    (tmp_path / "tensorboard_log").mkdir()
    assert validate_checkpoint(tmp_path) is None
    results = tmp_path / "rollout_result/train"
    results.mkdir(parents=True)
    (results / "0.jsonl").write_text("{}\n")
    with pytest.raises(ValueError, match="Uncheckpointed"):
        validate_checkpoint(tmp_path)


@pytest.mark.parametrize("signal", [9, 11])
def test_payload_signal_is_tool_feedback_without_hiding_service_failures(signal):
    result = parse_result(dict(status="Failed", run_result=dict(status="Finished", return_code=-signal)))
    assert result == dict(status="code_error", stdout="", stderr=f"Python program terminated by signal {signal}")
    assert score("Answer: 2", "2", {result["status"]: 1})["score"] == 0.9
    with pytest.raises(RuntimeError, match="infrastructure"):
        parse_result(dict(status="InternalError"))
    with pytest.raises(RuntimeError, match="execution result"):
        parse_result(dict(status="Failed", message="sandbox service crashed"))
