# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""CPU regressions for ReTool's trajectory, reward and recovery contracts."""

import asyncio
import json
import random
from types import SimpleNamespace

import pytest
import torch

from examples.retool.checkpoint import validate_checkpoint
from examples.retool.data_source import ReToolDataSource
from examples.retool.prepare_data import LEAD, TAIL, deduplicate, make_case
from examples.retool.prepare_sft import observation_spans
from examples.retool.reward import score
from examples.retool.rollout import PREFIX, VERSION, feedback_for_turn, run_episode
from examples.retool.sandbox import SandboxClient, parse_result
from examples.retool.sft_dataset import read_author_sample
from examples.retool.vocab import mask_shard
from relax.utils.types import Sample


class Tokenizer:
    def encode(self, text, **kwargs):
        return [ord(char) for char in text]

    def decode(self, ids, **kwargs):
        return "".join(chr(i) for i in ids)

    def apply_chat_template(self, messages, **kwargs):
        return "user:" + messages[0]["content"] + "\nassistant:"


class Client:
    def __init__(self, status="ok", error=None):
        self.calls, self.closed = [], False
        self.status, self.error = status, error

    async def execute(self, code):
        self.calls.append(code)
        if self.error:
            raise self.error
        return dict(status=self.status, stdout="2", stderr="bad code" if self.status != "ok" else "")

    async def close(self):
        self.closed = True


def episode(texts, client, *, budget=4096, tool_enabled=True, max_turns=16):
    tokenizer = Tokenizer()
    sample = Sample(prompt="Question", label="2", metadata=dict(protocol_id=VERSION, tool_enabled=tool_enabled))
    args = SimpleNamespace(rollout_max_prompt_len=2048, rollout_max_context_len=10240, max_turns=max_turns)
    payloads = []

    async def post(payload):
        payloads.append(payload)
        text, finish = texts[len(payloads) - 1]
        return dict(
            meta_info=dict(
                finish_reason=dict(type=finish),
                output_token_logprobs=[[-0.25, i, None] for i in tokenizer.encode(text)],
            )
        )

    result = asyncio.run(
        run_episode(
            args,
            sample,
            dict(max_new_tokens=budget),
            tokenizer=tokenizer,
            post=post,
            client=client,
            statuses=Sample.Status,
        )
    )
    return result, payloads


def test_retool_feedback_mask_and_exact_ids():
    code = "<code>\n```python\nprint(1 + 1)\n```\n</code>"
    final = "<answer>\\boxed{2}</answer>"
    client = Client()
    sample, payloads = episode([(code, "stop"), (final, "stop")], client)
    observation = "\n<interpreter>\n2\n</interpreter>\n"
    assert sample.tokens == Tokenizer().encode("Question" + code + observation + final)
    assert sample.loss_mask == [1] * len(code) + [0] * len(observation) + [1] * len(final)
    assert sample.rollout_log_probs == [-0.25] * len(code) + [0] * len(observation) + [-0.25] * len(final)
    assert payloads[1]["input_ids"] == Tokenizer().encode("Question" + code + observation)
    assert payloads[1]["sampling_params"]["max_new_tokens"] == 4096 - len(code + observation)
    assert client.calls == ["print(1 + 1)"] and client.closed
    assert sample.metadata["tool_feedback_counts"] == {"ok": 1}
    assert sample.status == Sample.Status.COMPLETED


@pytest.mark.parametrize("status", ["code_error", "timeout"])
def test_retool_error_feedback_can_recover(status):
    code = "<code>\n```python\nprint(1/0)\n```\n</code>"
    sample, _ = episode([(code, "stop"), ("Answer: 2", "stop")], Client(status))
    assert sample.metadata["tool_feedback_counts"] == {status: 1}
    assert score(sample.metadata["retool_final_answer"], "2", sample.metadata["tool_feedback_counts"])["score"] == 0.9


def test_retool_invalid_call_is_penalized_without_execution():
    client = Client()
    sample, _ = episode([("<code>oops</code>", "stop"), ("Answer: 2", "stop")], client)
    assert not client.calls and sample.metadata["tool_feedback_counts"] == {"invalid_call": 1}


@pytest.mark.parametrize("source,allowed", [("x" * 65536, True), ("x" * 65537, False), ("é" * 32769, False)])
def test_retool_sandbox_source_byte_limit(source, allowed):
    client = Client()
    text = "<code>\n```python\n" + source + "\n```\n</code>"
    feedback, executed = asyncio.run(feedback_for_turn(text, client))
    assert (feedback["status"] == "ok") is allowed
    assert client.calls == ([source] if allowed else [])
    assert executed == (source if allowed else None)


def test_retool_infrastructure_fault_propagates_without_retry():
    client = Client(error=OSError("service unavailable"))
    with pytest.raises(OSError, match="service unavailable"):
        episode([("<code>\n```python\nprint(2)\n```\n</code>", "stop")], client)
    assert len(client.calls) == 1 and client.closed


def test_retool_disabled_eval_never_executes_or_sets_code_stop():
    client = Client()
    sample, payloads = episode([("<code>ignored</code> Answer: 2", "stop")], client, tool_enabled=False)
    assert not client.calls and not sample.metadata["tool_feedback_counts"]
    assert "stop" not in payloads[0]["sampling_params"]
    assert "Answer: 2" in sample.metadata["retool_final_answer"]


@pytest.mark.parametrize("finish", ["length", "abort"])
def test_retool_incomplete_code_is_not_executed(finish):
    client = Client()
    sample, _ = episode([("<code>\n```python\nprint(2)", finish)], client)
    assert not client.calls
    assert sample.status == (Sample.Status.TRUNCATED if finish == "length" else Sample.Status.ABORTED)


def test_retool_last_turn_does_not_execute_unused_code():
    client = Client()
    sample, _ = episode([("<code>\n```python\nprint(2)\n```\n</code>", "stop")], client, max_turns=1)
    assert not client.calls and sample.status == Sample.Status.TRUNCATED


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
def test_retool_reward_matches_failure_penalty(text, label, counts, expected):
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


def test_retool_unexpected_process_signal_is_infrastructure_failure():
    with pytest.raises(RuntimeError, match="Unexpected sandbox process termination"):
        parse_result(dict(status="Success", run_result=dict(status="Finished", return_code=-9)))


def test_retool_http_client_uses_actual_response():
    from http.server import BaseHTTPRequestHandler, HTTPServer
    from threading import Thread

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


def test_retool_sampler_preserves_replacement_groups_and_resume(tmp_path, monkeypatch):
    import transformers

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: Tokenizer())
    data = tmp_path / "train.jsonl"
    data.write_text(json.dumps(make_case("1+1?", "2")) + "\n")
    args = SimpleNamespace(
        prompt_data=str(data),
        hf_checkpoint="unused",
        rollout_max_prompt_len=2048,
        rollout_seed=20261031,
        rollout_batch_size=3,
        n_samples_per_prompt=2,
        num_rollout=3,
        save=str(tmp_path),
        load=str(tmp_path),
    )
    source = ReToolDataSource(args)
    assert source.lengths() == len(source) == 1
    batch = source.get_samples(3)
    assert len(batch) == 3 and all(len(group) == 2 for group in batch)
    assert [g[0].group_index for g in batch] == [0, 1, 2]
    assert {s.index for g in batch for s in g} == set(range(6))
    source.save(0)
    expected = source.get_samples(3)
    restored = ReToolDataSource(args)
    restored.load(0)
    actual = restored.get_samples(3)
    assert [s.to_dict() for g in actual for s in g] == [s.to_dict() for g in expected for s in g]
    data.write_text(json.dumps(make_case("different data", "2")) + "\n")
    with pytest.raises(ValueError, match="Resume data"):
        ReToolDataSource(args).load(0)


def test_retool_sampler_draws_match_original_schedule(tmp_path, monkeypatch):
    import transformers

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: Tokenizer())
    path = tmp_path / "train.jsonl"
    path.write_text("".join(json.dumps(make_case(str(i), "2")) + "\n" for i in range(7)))
    args = SimpleNamespace(
        prompt_data=str(path),
        hf_checkpoint="unused",
        rollout_max_prompt_len=2048,
        rollout_seed=20261031,
        rollout_batch_size=64,
        n_samples_per_prompt=16,
        num_rollout=200,
    )
    data = ReToolDataSource(args)
    rng = random.Random(20261031)
    for _ in range(3):
        assert [group[0].metadata["scheduled_question"] for group in data.get_samples(64)] == [
            rng.randrange(7) for _ in range(64)
        ]


def test_retool_vocab_mask_matches_unsplit_distribution():
    logits = torch.tensor([[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]])
    parts = [mask_shard(part, limit=6, rank=rank, world_size=2) for rank, part in enumerate(logits.chunk(2, -1))]
    assert torch.equal(torch.log_softmax(torch.cat(parts, -1), -1)[:, :6], torch.log_softmax(logits[:, :6], -1))
    assert torch.equal(torch.softmax(torch.cat(parts, -1), -1)[:, 6:], torch.zeros(1, 2))


def test_retool_sft_spans_preserve_malformed_author_text():
    text = "think<interpreter>2<code>python</code><answer>2</answer>"
    spans, notes = observation_spans(text)
    assert text[spans[0][0] : spans[0][1]] == "<interpreter>2" and notes


def test_retool_sft_reader_keeps_feedback_context(tmp_path):
    import numpy as np

    ids, mask = [1, 2, 3, 4, 5], [0, 1, 0, 1, 1]
    (tmp_path / "tokens").write_bytes(np.array(ids, dtype="<u4").tobytes())
    (tmp_path / "mask").write_bytes(bytes(mask))
    row = dict(
        positions=5,
        token_offset=0,
        mask_offset=0,
        tokens_path=str(tmp_path / "tokens"),
        mask_path=str(tmp_path / "mask"),
        assistant_loss_tokens=3,
        eos_token_id=5,
        supervised_assistant_EOS=1,
    )
    sample = read_author_sample(row, 16, 0)
    assert sample.tokens.tolist() == ids and sample.loss_mask.tolist() == mask


def test_retool_resume_refuses_nonempty_or_missing_checkpoint(tmp_path):
    assert validate_checkpoint(tmp_path / "missing") is None
    with pytest.raises(ValueError, match="committed"):
        validate_checkpoint(tmp_path)
    (tmp_path / "junk").touch()
    with pytest.raises(ValueError, match="committed"):
        validate_checkpoint(tmp_path)
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("24")
    with pytest.raises(ValueError, match="Incomplete"):
        validate_checkpoint(tmp_path)


@pytest.mark.parametrize("evaluation", [False, True])
def test_retool_generate_uses_native_sglang_parameters(monkeypatch, evaluation):
    pytest.importorskip("sglang")
    from sglang.srt.sampling.sampling_params import SamplingParams

    from examples.retool.rollout import generate
    from relax.engine.rollout import sglang_rollout

    requests = []

    async def post(url, payload):
        SamplingParams(**payload["sampling_params"])
        requests.append(payload)
        return dict(
            meta_info=dict(
                weight_version="1",
                finish_reason=dict(type="stop"),
                output_token_logprobs=[[-0.25, ord(c), None] for c in "Answer: 2"],
            )
        )

    monkeypatch.setattr(
        sglang_rollout, "GenerateState", lambda args: SimpleNamespace(tokenizer=Tokenizer(), post_generate=post)
    )
    args = SimpleNamespace(
        rollout_seed=20261031,
        start_rollout_id=0,
        sglang_router_ip="127.0.0.1",
        sglang_router_port=1,
        rollout_max_prompt_len=2048,
        rollout_max_context_len=10240,
        max_turns=16,
    )
    sample = Sample(index=0, prompt="Question", metadata=dict(policy_outer=0, tool_enabled=False, question="1+1?"))
    result = asyncio.run(
        generate(args, sample, dict(temperature=1, top_p=1, top_k=-1, max_new_tokens=100), evaluation=evaluation)
    )
    assert result.status == Sample.Status.COMPLETED
    assert "seed" not in requests[0]["sampling_params"]
    assert requests[0]["sampling_params"]["sampling_seed"] == args.rollout_seed
    assert requests[0]["custom_logit_processor"]
    assert requests[0]["sampling_params"]["custom_params"]["token_ids"] == list(range(151665, 152064))
