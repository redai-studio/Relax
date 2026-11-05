# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""CPU contract tests through the real compiler and SessionForest exporter."""

import asyncio
import copy
import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

from examples.retool.data_source import ReToolDataSource
from examples.retool.rollout import PREFIX, VERSION
from examples.retool.rollout import run_episode as native_episode
from examples.retool_agentic import data_source as source_module
from examples.retool_agentic.agent import ModelClient, run_episode
from examples.retool_agentic.data_source import ReToolAgenticDataSource
from examples.retool_agentic.protocol import CHAT_TEMPLATE, observation_text, render_prompt
from examples.retool_agentic.reward import reward_func
from relax.agentic.pipeline.runtime import SGLangMessageCompiler, _build_session_specs
from relax.agentic.runner.ipc import SessionOutput
from relax.agentic.session.service import AgenticSessionShard, _normalized_chat_request
from relax.agentic.session.state import SessionForest, check_messages
from relax.utils.types import Sample


ROOT = Path(__file__).resolve().parents[3]
CODE = "<code>\n```python\nprint(1 + 1)\n```\n</code>"
FINAL = "<answer>\\boxed{2}</answer>"


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer, PreTrainedTokenizerFast

    if model := os.environ.get("RETOOL_TEST_MODEL_DIR"):
        return AutoTokenizer.from_pretrained(model, local_files_only=True)
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    raw = Tokenizer(models.BPE(unk_token="<unk>"))
    raw.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    raw.decoder = decoders.ByteLevel()
    raw.train_from_iterator(
        [PREFIX + "What is 1+1?", CODE, FINAL, observation_text(dict(status="ok", stdout="2", stderr=""))],
        trainers.BpeTrainer(
            vocab_size=512,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            special_tokens=["<unk>", "<|im_start|>", "<|im_end|>"],
        ),
    )
    return PreTrainedTokenizerFast(tokenizer_object=raw, chat_template=CHAT_TEMPLATE)


class Sandbox:
    def __init__(self, status="ok", stdout="2", error=None):
        self.status, self.stdout, self.error = status, stdout, error
        self.calls, self.closed = [], False

    async def execute(self, code):
        self.calls.append(code)
        if self.error:
            raise self.error
        return dict(status=self.status, stdout=self.stdout, stderr="bad code" if self.status != "ok" else "")

    async def close(self):
        self.closed = True


def input_payload(*, turns=16, budget=8192, context=10240):
    return dict(
        messages=[dict(role="user", content=PREFIX + "What is 1+1?")],
        metadata=dict(retool_agentic=dict(max_turns=turns, response_budget=budget, context_budget=context, seed=123)),
    )


class ForestBackend:
    """Only model generation is replaced; context matching/export use core
    code."""

    def __init__(self, tokenizer, replies):
        self.tokenizer, self.replies = tokenizer, replies
        self.compiler = SGLangMessageCompiler(tokenizer=tokenizer, processor=None, cpu_executor=None)
        self.forest = SessionForest.create_empty(session_id="test", label="2")
        self.requests, self.prompts = [], []
        self.last = None

    def request(self, payload):
        request = _normalized_chat_request(payload)
        self.requests.append(copy.deepcopy(payload))
        kwargs = request["chat_template_kwargs"]
        shard = AgenticSessionShard.__ray_metadata__.modified_class
        parent, delta = shard._match_parent_state(
            forest=self.forest, messages=request["messages"], tools=[], chat_template_kwargs=kwargs
        )
        initial = parent.state_hash == self.forest.root_state_hash
        if not initial:
            assert parent == self.last, "History must extend the exact previous response"
            assert len(delta) == 1 and delta[0]["role"] == "tool"
        build = self.compiler._build_prompt_sync if initial else self.compiler._build_observation_prompt_sync
        _, ids, _, _ = build(delta, [], kwargs)
        obs = self.forest.append_obs(
            parent_state_hash=parent.state_hash,
            rollout_id=0,
            abort_count=0,
            messages_delta=delta,
            train_token_delta=ids,
            rollout_token_delta=ids,
            tools=[] if initial else None,
            chat_template_kwargs=kwargs if initial else None,
        )
        prefix = [token for node in self.forest.lineage(obs.state_hash) for token in node.rollout_token_delta]
        self.prompts.append(prefix)
        text, finish = self.replies[len(self.requests) - 1]
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        message = dict(role="assistant", content=text)
        self.last = self.forest.append_resp(
            parent_state_hash=obs.state_hash,
            rollout_id=0,
            abort_count=0,
            messages_delta=check_messages([message]),
            token_delta=tokens,
            logprob_delta=[-0.25] * len(tokens),
            weight_version_delta=["1"],
            status="truncated" if finish == "length" else "completed",
        )
        return dict(
            choices=[dict(message=message, finish_reason=finish)],
            usage=dict(prompt_tokens=len(prefix), completion_tokens=len(tokens)),
        )

    async def complete(self, payload):
        return self.request(payload)

    def export(self, output):
        sample = self.forest.build_sample(leaf_state_hash=self.last.state_hash, tokenizer=self.tokenizer)
        sample.metadata.update(SessionOutput.from_payload(output).metadata)
        return sample


@pytest.mark.parametrize(
    "texts,status,stdout,turns,budget",
    [
        ([(FINAL, "stop")], "ok", "2", 16, 8192),
        ([(CODE, "stop")] * 3 + [(FINAL, "stop")], "ok", "2", 16, 8192),
        ([(CODE, "stop"), (FINAL, "stop")], "ok", "", 16, 8192),
        ([(CODE, "stop"), (FINAL, "stop")], "ok", "中文\n\n", 16, 8192),
        ([(CODE, "stop"), (FINAL, "stop")], "code_error", "", 16, 8192),
        ([("<code>invalid</code>", "stop"), (FINAL, "stop")], "ok", "", 16, 8192),
        ([(CODE, "stop")], "ok", "2", 1, 8192),
        ([("partial answer", "length")], "ok", "2", 16, 8192),
        ([(CODE, "stop")], "ok", "x" * 1024, 16, 32),
    ],
)
def test_agentic_export_matches_native_episode(tokenizer, texts, status, stdout, turns, budget):
    # Keep the observation-budget case independent of tokenizer compression.
    if budget == 32:
        budget = len(tokenizer.encode(CODE, add_special_tokens=False)) + 2
    payload = input_payload(turns=turns, budget=budget)
    args = SimpleNamespace(
        rollout_max_prompt_len=2048, rollout_max_context_len=10240, max_turns=turns, start_rollout_id=0
    )
    initial = render_prompt(payload["messages"])
    native = Sample(prompt=initial, label="2", metadata=dict(protocol_id=VERSION))
    native_requests = []

    async def post(request):
        native_requests.append(request)
        text, finish = texts[len(native_requests) - 1]
        return dict(
            meta_info=dict(
                finish_reason=dict(type="eos" if finish == "stop" else finish),
                output_token_logprobs=[[-0.25, i, None] for i in tokenizer.encode(text, add_special_tokens=False)],
            )
        )

    asyncio.run(
        native_episode(
            args,
            native,
            dict(max_new_tokens=budget),
            tokenizer=tokenizer,
            post=post,
            client=Sandbox(status, stdout),
            statuses=Sample.Status,
        )
    )
    backend, sandbox = ForestBackend(tokenizer, texts), Sandbox(status, stdout)
    output = asyncio.run(
        run_episode(
            payload,
            count_tokens=lambda text: len(tokenizer.encode(text, add_special_tokens=False)),
            complete=backend.complete,
            client=sandbox,
        )
    )
    exported = backend.export(output)
    reward = asyncio.run(reward_func(args, exported))
    for name in (
        "tokens",
        "rollout_tokens",
        "response",
        "response_length",
        "loss_mask",
        "rollout_log_probs",
        "status",
    ):
        assert getattr(exported, name) == getattr(native, name), name
    for key in (
        "retool_final_answer",
        "tool_feedback_counts",
        "tool_calls",
        "retool_assistant_turns",
        "retool_stop_reason",
    ):
        assert exported.metadata[key] == native.metadata[key], key
    assert exported.metadata["rollout_turns"] == len(texts)
    assert exported.metadata["agentic_trace"]["terminal_status"] == exported.status.value
    assert sandbox.closed
    assert backend.prompts == [r["input_ids"] for r in native_requests]
    assert [r["max_completion_tokens"] for r in backend.requests] == [
        r["sampling_params"]["max_new_tokens"] for r in native_requests
    ]
    assert [r["seed"] for r in backend.requests] == [123 + i * 1000003 for i in range(len(texts))]
    assert reward["tool_failure_count"] == sum(
        n for key, n in exported.metadata["tool_feedback_counts"].items() if key != "ok"
    )


def test_infrastructure_error_is_not_feedback_or_retried(tokenizer):
    sandbox = Sandbox(error=RuntimeError("sandbox unavailable"))
    backend = ForestBackend(tokenizer, [(CODE, "stop")])
    with pytest.raises(RuntimeError, match="sandbox unavailable"):
        asyncio.run(
            run_episode(
                input_payload(),
                count_tokens=lambda text: len(tokenizer.encode(text, add_special_tokens=False)),
                complete=backend.complete,
                client=sandbox,
            )
        )
    assert sandbox.calls == ["print(1 + 1)"] and sandbox.closed
    assert len(backend.requests) == 1


def test_agent_process_http_contract_and_prefix(tokenizer, tmp_path):
    backend = ForestBackend(tokenizer, [(CODE, "stop"), (FINAL, "stop")])
    seen_code = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if self.path == "/relax-prefix/v1/chat/completions":
                assert self.headers["Authorization"] == "Bearer test-session"
                result = backend.request(payload)
            else:
                assert self.path == "/run_code"
                seen_code.append(payload["code"])
                result = dict(status="Success", run_result=dict(status="Finished", stdout="2", stderr=""))
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps(result).encode())

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    input_path, output_path = tmp_path / "input.json", tmp_path / "output.json"
    input_path.write_text(json.dumps(input_payload()))
    tokenizer.backend_tokenizer.save(str(tmp_path / "tokenizer.json"))
    env = dict(
        os.environ,
        RELAX_INPUT_JSON=str(input_path),
        RELAX_OUTPUT_JSON=str(output_path),
        RELAX_BASE_URL=base + "/relax-prefix",
        RELAX_SESSION_ID="test-session",
        RETOOL_TOKENIZER_PATH=str(tmp_path / "tokenizer.json"),
        SANDBOX_FUSION_URL=base + "/run_code",
        PYTHONPATH=str(ROOT),
    )
    try:
        result = subprocess.run(
            [sys.executable, "-m", "examples.retool_agentic.agent"],
            env=env,
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, result.stderr
        sample = backend.export(json.loads(output_path.read_text()))
        assert sample.metadata["rollout_turns"] == 2
        assert sample.metadata["retool_final_answer"] == FINAL
        assert 0 in sample.loss_mask and seen_code == ["print(1 + 1)"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_chunked_draws_and_checkpoint_resume_match_original(tokenizer, tmp_path, monkeypatch):
    model = tmp_path / "model"
    tokenizer.save_pretrained(model)
    data = tmp_path / "train.jsonl"
    data.write_text(
        "\n".join(
            json.dumps(dict(prompt=[dict(role="user", content=PREFIX + f"question {i}")], label=str(i)))
            for i in range(17)
        )
    )
    monkeypatch.setattr(source_module, "TOKENIZER_LIMIT", len(tokenizer))
    monkeypatch.setattr(source_module, "MODEL_VOCAB", len(tokenizer) + 3)
    args = SimpleNamespace(
        prompt_data=str(data),
        hf_checkpoint=str(model),
        rollout_max_prompt_len=2048,
        rollout_seed=123,
        rollout_batch_size=4,
        n_samples_per_prompt=2,
        num_rollout=3,
        max_turns=16,
        rollout_max_response_len=8192,
        rollout_max_context_len=10240,
        save=str(tmp_path / "save"),
        load=None,
    )
    Path(args.save).mkdir()
    source, native = ReToolAgenticDataSource(args), ReToolDataSource(args)
    groups = source.get_samples(1) + source.get_samples(2) + source.get_samples(8)
    baseline = native.get_samples(4)
    assert [[s.label for s in g] for g in groups] == [[s.label for s in g] for g in baseline]
    assert [[s.index for s in g] for g in groups] == [[s.index for s in g] for g in baseline]
    assert groups[0][0].prompt[0]["role"] == "user"
    assert len(groups[0][0].sampling_params["logit_bias"]) == 3
    source.save(0)
    args.load = args.save
    restored = ReToolAgenticDataSource(args)
    restored.load(0)
    next_groups = restored.get_samples(3) + restored.get_samples(1)
    expected = native.get_samples(4)
    assert [[s.label for s in g] for g in next_groups] == [[s.label for s in g] for g in expected]
    assert [[s.index for s in g] for g in next_groups] == [[s.index for s in g] for g in expected]
    restored.get_samples(1)
    with pytest.raises(RuntimeError, match="complete Agentic prompt batch"):
        restored.save(1)


def test_model_client_preserves_relax_route_prefix():
    assert ModelClient("http://host/agentic/", "sid").url == "http://host/agentic/v1/chat/completions"


def test_session_sampling_keeps_padded_vocab_mask():
    import cloudpickle

    mask = {str(i): -1e30 for i in range(151665, 152064)}
    sample = Sample(prompt=[dict(role="user", content="question")], index=7)
    sample.sampling_params = dict(logit_bias=mask)
    sample = cloudpickle.loads(cloudpickle.dumps(sample))
    spec = _build_session_specs([sample], session_ids=["session-7"], include_input_payload=True)[0]
    shard = object.__new__(AgenticSessionShard.__ray_metadata__.modified_class)
    shard.args = SimpleNamespace(
        rollout_temperature=1,
        rollout_top_p=1,
        rollout_top_k=-1,
        rollout_max_response_len=8192,
        rollout_stop=None,
        rollout_stop_token_ids=None,
        rollout_skip_special_tokens=False,
        sglang_enable_deterministic_inference=False,
    )
    params = shard._session_sampling_params(spec, 0)
    assert params["logit_bias"] == mask and params["no_stop_trim"] is True


def test_reward_checks_resumed_policy_version():
    sample = Sample(
        label="2",
        weight_versions=["1", "1"],
        status=Sample.Status.COMPLETED,
        metadata=dict(
            policy_outer=25, retool_stop_reason="final_answer", retool_final_answer=FINAL, tool_feedback_counts={}
        ),
    )
    args = SimpleNamespace(start_rollout_id=25)
    assert asyncio.run(reward_func(args, sample))["score"] == 1
    sample.weight_versions.append("2")
    with pytest.raises(RuntimeError, match="policy version"):
        asyncio.run(reward_func(args, sample))


def test_agentic_launch_uses_local_entrypoint_and_distinct_checkpoint(tmp_path):
    tools = tmp_path / "bin"
    tools.mkdir()
    record = tmp_path / "ray.jsonl"
    for name in ("pkill", "sleep", "nvidia-smi"):
        path = tools / name
        path.write_text("#!/bin/sh\nexit 0\n")
        path.chmod(0o755)
    ray = tools / "ray"
    ray.write_text(
        f"#!{sys.executable}\nimport json, os, sys\n"
        "with open(os.environ['TEST_RAY_RECORD'], 'a') as f:\n"
        "    f.write(json.dumps(sys.argv[1:]) + '\\n')\n"
    )
    ray.chmod(0o755)
    model, data, save = [tmp_path / name for name in ("model with spaces", "data", "save")]
    model.mkdir()
    data.mkdir()
    (model / "config.json").write_text("{}")
    (model / "tokenizer.json").write_text("{}")
    (data / "train.jsonl").write_text("{}\n")
    env = {key: value for key, value in os.environ.items() if not key.startswith(("RAY_", "RELAX_", "SANDBOX_"))}
    for name in ("EVAL_CONFIG", "EXP_NAME"):
        env.pop(name, None)
    env.update(
        PATH=f"{tools}:{os.environ['PATH']}",
        MODEL_DIR=str(model),
        DATA_DIR=str(data),
        SAVE_DIR=str(save),
        TEST_RAY_RECORD=str(record),
    )
    script = ROOT / "examples/retool_agentic/scripts/run-qwen25-7B-grpo-8xgpu.sh"
    # All process/cluster executables above are stubs: never clean up real Ray.
    result = subprocess.run(["bash", str(script)], env=env, cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    commands = [json.loads(line) for line in record.read_text().splitlines()]
    assert any(command[0] == "start" for command in commands)
    command = next(command for command in commands if command[:2] == ["job", "submit"])
    assert "--no-wait" not in command
    assert "--use-agentic-rollout" in command
    assert "--custom-generate-function-path" not in command
    assert command[command.index("--agentic-concurrency") + 1] == "16"
    assert command[command.index("--agent-env") + 1] == f"RETOOL_TOKENIZER_PATH={model}/tokenizer.json"
    expected = str(save / "qwen25-7b-retool-agentic-grpo-8xgpu")
    assert command[command.index("--load") + 1] == command[command.index("--save") + 1] == expected
    runtime = json.loads(next(arg.split("=", 1)[1] for arg in command if arg.startswith("--runtime-env-json=")))
    assert runtime["env_vars"]["DATA_DIR"] == str(data)
    assert runtime["env_vars"]["SAVE_DIR"] == str(save)
