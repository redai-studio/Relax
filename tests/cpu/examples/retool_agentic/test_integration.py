# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""CPU contract tests through the real compiler and SessionForest exporter."""

import asyncio
import copy
import json
import os
import random
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from examples.retool_agentic.agent import ModelClient, run_episode
from examples.retool_agentic.data_source import ReToolAgenticDataSource
from examples.retool_agentic.protocol import CHAT_TEMPLATE, PREFIX, VERSION, observation_text, render_prompt
from examples.retool_agentic.reward import reward_func
from relax.agentic.inventory import InventoryRecorder
from relax.agentic.pipeline.runtime import SGLangMessageCompiler
from relax.agentic.runner.ipc import SessionOutput
from relax.agentic.session.service import AgenticSessionShard, _normalized_chat_request
from relax.agentic.session.state import SessionForest, check_messages
from relax.utils.types import Sample
from tests.cpu.examples.retool_agentic.native_reference import run_episode as native_episode


ROOT = Path(__file__).resolve().parents[4]
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


def test_chunked_draws_and_checkpoint_resume_match_original(tokenizer, tmp_path):
    model = tmp_path / "model"
    tokenizer.save_pretrained(model)
    data = tmp_path / "train.jsonl"
    data.write_text(
        "\n\n".join(
            json.dumps(dict(prompt=[dict(role="user", content=PREFIX + f"question {i}")], label=str(i)))
            for i in range(17)
        )
    )
    args = SimpleNamespace(
        policy_vocab_size=len(tokenizer),
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
    source = ReToolAgenticDataSource(args)
    rng = random.Random(args.rollout_seed)
    groups = source.get_samples(1) + source.get_samples(2) + source.get_samples(8)
    assert [g[0].metadata["scheduled_question"] for g in groups] == [rng.randrange(17) for _ in range(4)]
    assert [s.index for g in groups for s in g] == list(range(8))
    assert groups[0][0].prompt[0]["role"] == "user"
    for group in groups:
        assert all(s.agentic_prompt_source is not None for s in group)
        assert len({s.agentic_prompt_source["prompt_id"] for s in group}) == 1
        question = group[0].metadata["scheduled_question"]
        provenance = group[0].agentic_prompt_source
        assert provenance["source_row_id"] == str(2 * question)
        assert provenance["input_payload"] == source.cases[question]
        assert [s.metadata["retool_agentic"]["seed"] for s in group] == [
            args.rollout_seed + s.index * 1009 for s in group
        ]

    args.dump_agentic_obs = str(tmp_path / "obs")
    args.agentic_obs_run_id = "test_provenance"

    async def record_inventory():
        recorder = InventoryRecorder(args, "train")
        for i, group in enumerate(groups):
            recorder.record_group(f"group_{i}", group, [f"test_session_{s.index}" for s in group])
        await recorder.aclose()

    asyncio.run(record_inventory())
    obs = tmp_path / "obs/test_provenance"
    inventories = {
        stream: [json.loads(line) for path in (obs / stream).glob("*.jsonl") for line in path.read_text().splitlines()]
        for stream in ("groups", "sessions")
    }
    group_prompts = {g["group_id"]: g["prompt_id"] for g in inventories["groups"]}
    assert len(group_prompts) == 4 and len(inventories["sessions"]) == 8
    assert all(s["prompt_id"] == group_prompts[s["group_id"]] for s in inventories["sessions"])
    source.save(0)
    args.load = args.save
    args.start_rollout_id = 1
    restored = ReToolAgenticDataSource(args)
    next_groups = restored.get_samples(3) + restored.get_samples(1)
    assert [g[0].metadata["scheduled_question"] for g in next_groups] == [rng.randrange(17) for _ in range(4)]
    assert [s.index for g in next_groups for s in g] == list(range(8, 16))
    # Replacement groups consume independent draws, not a new policy version.
    replacement = restored.get_samples(1)[0]
    restored.save(1)
    assert replacement[0].group_index == 8
    assert "policy_outer" not in replacement[0].metadata
    args.start_rollout_id = 2
    after_drop = ReToolAgenticDataSource(args)
    # The framework's later load of this boundary is idempotent.
    after_drop.load(1)
    assert after_drop.group_cursor == 9
    continued, replayed = restored.get_samples(1), after_drop.get_samples(1)
    assert continued[0][0].metadata == replayed[0][0].metadata
    assert continued[0][0].agentic_prompt_source == replayed[0][0].agentic_prompt_source
    assert continued[0][0].index == replayed[0][0].index == 18
    with pytest.raises(RuntimeError, match="complete Agentic prompt batch"):
        restored.save(2)
    # Original checkpoints predate next_group_index and remain loadable.
    state_path = Path(args.save) / "dataset/retool_0.json"
    state = json.loads(state_path.read_text())
    state.pop("next_group_index")
    state_path.write_text(json.dumps(state))
    args.start_rollout_id = 1
    compatible = ReToolAgenticDataSource(args)
    assert compatible.group_cursor == 4
    # Validate at construction: the framework catches errors from load().
    invalid_states = [
        dict(seed=-1),
        dict(sha256="changed"),
        dict(batch=8),
        dict(K=3),
        dict(rng_state=[1, [], None]),
        dict(next_group_index=-1),
        dict(execution_mode="hybrid"),
    ]
    for change in invalid_states:
        state_path.write_text(json.dumps(dict(state, **change)))
        with pytest.raises(ValueError):
            ReToolAgenticDataSource(args)


def test_model_client_preserves_relax_route_prefix():
    assert ModelClient("http://host/agentic/", "sid").url == "http://host/agentic/v1/chat/completions"


def test_reward_checks_resumed_policy_version():
    sample = Sample(
        label="2",
        weight_versions=["1", "1"],
        status=Sample.Status.COMPLETED,
        metadata=dict(
            policy_outer=54,
            start_rollout_id=25,
            retool_stop_reason="final_answer",
            retool_final_answer=FINAL,
            tool_feedback_counts={},
        ),
    )
    args = SimpleNamespace(start_rollout_id=25)
    assert asyncio.run(reward_func(args, sample))["score"] == 1
    sample.weight_versions.append("2")
    with pytest.raises(RuntimeError, match="policy version"):
        asyncio.run(reward_func(args, sample))


@pytest.mark.parametrize("versions", [["1", "2", "2"], ["1", "1"]])
def test_hybrid_reward_accepts_monotonic_policy_versions(versions):
    sample = Sample(
        label="2",
        weight_versions=versions,
        metadata=dict(retool_stop_reason="final_answer", retool_final_answer=FINAL, tool_feedback_counts={}),
    )
    assert asyncio.run(reward_func(SimpleNamespace(hybrid=True), sample))["score"] == 1
    assert sample.metadata["retool_policy_version_span"] == int(versions[-1]) - int(versions[0])
    sample.weight_versions = ["2", "1"]
    with pytest.raises(RuntimeError, match="went backwards"):
        asyncio.run(reward_func(SimpleNamespace(hybrid=True), sample))


@pytest.mark.parametrize(
    "statuses,timeout_limit,wall_budget,reason,expected_calls",
    [
        (["timeout"] * 5, 3, 0, "consecutive_tool_timeouts", 3),
        (["timeout", "ok", "timeout", "timeout", "timeout"], 3, 0, "consecutive_tool_timeouts", 5),
        (["timeout"] * 5, 0, 0, "final_answer", 5),
        (["ok"] * 5, 0, 90, "tool_wall_time_budget", 3),
    ],
)
def test_tool_tail_budget_preserves_feedback_and_export(
    tokenizer, monkeypatch, statuses, timeout_limit, wall_budget, reason, expected_calls
):
    from examples.retool_agentic import agent

    payload = input_payload()
    payload["metadata"]["retool_agentic"].update(
        max_consecutive_timeouts=timeout_limit, tool_wall_time_budget_seconds=wall_budget
    )
    backend = ForestBackend(tokenizer, [(CODE, "stop")] * len(statuses) + [(FINAL, "stop")])
    sandbox = Sandbox()
    clock = iter(value for i in range(len(statuses)) for value in (i * 31.0, (i + 1) * 31.0))
    monkeypatch.setattr(agent.time, "perf_counter", lambda: next(clock))
    original_execute = sandbox.execute

    async def execute(code):
        sandbox.status = statuses[len(sandbox.calls)]
        return await original_execute(code)

    sandbox.execute = execute
    output = asyncio.run(
        run_episode(
            payload,
            count_tokens=lambda text: len(tokenizer.encode(text, add_special_tokens=False)),
            complete=backend.complete,
            client=sandbox,
        )
    )
    exported = backend.export(output)
    reward = asyncio.run(reward_func(SimpleNamespace(hybrid=True), exported))
    assert exported.metadata["retool_stop_reason"] == reason
    assert len(sandbox.calls) == exported.metadata["tool_calls"] == expected_calls
    assert exported.metadata["retool_tool_wall_seconds"] == expected_calls * 31
    assert reward["tool_failure_count"] == statuses[:expected_calls].count("timeout")
    assert exported.status == (Sample.Status.COMPLETED if reason == "final_answer" else Sample.Status.TRUNCATED)
    assert len(backend.requests) == expected_calls + (reason == "final_answer")
    assert sandbox.closed


def test_hybrid_draws_cross_batch_boundary_and_refuse_incomplete_resume(tokenizer, tmp_path):
    model = tmp_path / "model"
    tokenizer.save_pretrained(model)
    data = tmp_path / "train.jsonl"
    data.write_text(json.dumps(dict(prompt=[dict(role="user", content=PREFIX + "What is 1+1?")], label="2")))
    args = SimpleNamespace(
        policy_vocab_size=len(tokenizer),
        prompt_data=str(data),
        hf_checkpoint=str(model),
        rollout_max_prompt_len=2048,
        rollout_seed=123,
        rollout_batch_size=4,
        n_samples_per_prompt=2,
        max_turns=16,
        rollout_max_response_len=8192,
        rollout_max_context_len=10240,
        save=str(tmp_path / "save"),
        load=None,
        hybrid=True,
        fully_async=True,
        agentic_prelaunch=True,
        start_rollout_id=0,
        num_rollout=1,
        over_sampling_batch_size=6,
        retool_max_consecutive_timeouts=3,
        retool_tool_wall_time_budget_seconds=90,
    )
    Path(args.save).mkdir()
    source = ReToolAgenticDataSource(args)
    groups = source.get_samples(3) + source.get_samples(3)
    assert [g[0].group_index for g in groups] == list(range(6))
    assert groups[-1][0].metadata["retool_agentic"]["max_consecutive_timeouts"] == 3
    source.save(0)
    state = json.loads((Path(args.save) / "dataset/retool_0.json").read_text())
    assert state["next_group_index"] == 6 and state["execution_mode"] == "hybrid"
    # The terminal backfill drains existing debt; it consumes no new batch.
    source.save(args.num_rollout)
    final_state = json.loads((Path(args.save) / "dataset/retool_1.json").read_text())
    assert final_state["cursor"] == 1 and final_state["next_group_index"] == 6
    args.load = args.save
    with pytest.raises(ValueError, match="cannot resume resident"):
        source.load(0)
    args.start_rollout_id = 1
    with pytest.raises(ValueError, match="starts fresh"):
        ReToolAgenticDataSource(args)


@pytest.mark.parametrize("nvls", [None, "0"])
@pytest.mark.parametrize("hybrid,score_centering", [(False, False), (True, False), (True, True), (False, True)])
def test_agentic_launch_uses_local_entrypoint_and_distinct_checkpoint(tmp_path, nvls, hybrid, score_centering):
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
    for name in ("EVAL_CONFIG", "EXP_NAME", "NCCL_NVLS_ENABLE"):
        env.pop(name, None)
    env.update(
        PATH=f"{tools}:{os.environ['PATH']}",
        MODEL_DIR=str(model),
        DATA_DIR=str(data),
        SAVE_DIR=str(save),
        TEST_RAY_RECORD=str(record),
    )
    if nvls is not None:
        # Existing-cluster entrypoints auto-detect NVLink. An explicit override
        # must still win in the actual submitted worker environment.
        env.update(
            RELAX_ENTRYPOINT_MODE="ray-job",
            RUNTIME_ENV_JSON='{"env_vars":{"NCCL_NVLS_ENABLE":"1"}}',
            NCCL_NVLS_ENABLE=nvls,
        )
    suffix = "-hybrid-async" if hybrid else ""
    script = ROOT / f"examples/retool_agentic/scripts/run-qwen25-7B-grpo-8xgpu{suffix}.sh"
    if score_centering:
        mode = "hybrid-6a2r" if hybrid else "colocate"
        script = ROOT / f"examples/retool_agentic/scripts/run-qwen25-7B-sc-8xgpu-{mode}.sh"
    # All process/cluster executables above are stubs: never clean up real Ray.
    result = subprocess.run(["bash", str(script)], env=env, cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    commands = [json.loads(line) for line in record.read_text().splitlines()]
    assert any(command[0] == "start" for command in commands) == (nvls is None)
    command = next(command for command in commands if command[:2] == ["job", "submit"])
    assert "--no-wait" not in command
    assert "--use-agentic-rollout" in command
    assert "--custom-generate-function-path" not in command
    assert command[command.index("-m") + 1] == "examples.retool_agentic.train"
    assert "--custom-megatron-before-train-step-hook-path" not in command
    assert "--custom-megatron-before-log-prob-hook-path" not in command
    config = yaml.safe_load(Path(command[command.index("--custom-config-path") + 1]).read_text())
    assert config["policy_vocab_size"] == 151665
    assert command[command.index("--agentic-concurrency") + 1] == "16"
    assert command[command.index("--agent-env") + 1] == f"RETOOL_TOKENIZER_PATH={model}/tokenizer.json"
    expected = str(save / f"qwen25-7b-retool-agentic-grpo-8xgpu{suffix}")
    if score_centering:
        expected = str(save / f"qwen25-7b-retool-agentic-sc-8xgpu-{mode}")
    assert command[command.index("--load") + 1] == command[command.index("--save") + 1] == expected
    runtime = json.loads(next(arg.split("=", 1)[1] for arg in command if arg.startswith("--runtime-env-json=")))
    assert runtime["env_vars"]["DATA_DIR"] == str(data)
    assert runtime["env_vars"]["SAVE_DIR"] == str(save)
    if nvls is not None:
        assert runtime["env_vars"]["NCCL_NVLS_ENABLE"] == nvls
    if hybrid:
        assert "--hybrid" in command and "--fully-async" not in command and "--partial-rollout" not in command
        actor, rollout = (6, 2) if score_centering else (4, 4)
        assert json.loads(command[command.index("--resource") + 1]) == {"actor": [1, actor], "rollout": [1, rollout]}
        assert command[command.index("--start-rollout-id") + 1] == "0"
    if score_centering:
        assert runtime["env_vars"]["SGLANG_RETURN_ORIGINAL_LOGPROB"] == "0"
        assert "--sglang-router-disable-circuit-breaker" in command
        assert command[command.index("--advantage-estimator") + 1] == "score_centering"
        assert command[command.index("--score-centering-is") + 1] == "tis"
        assert "--use-tis" not in command and "--calculate-per-token-loss" in command
        batch = int(command[command.index("--global-batch-size") + 1])
        groups = int(command[command.index("--rollout-batch-size") + 1])
        dp = 3 if hybrid else 4
        micro = int(command[command.index("--micro-batch-size") + 1])
        assert batch == groups * 16 == 1008 and batch % (dp * micro) == 0
        if not hybrid:
            assert json.loads(command[command.index("--resource") + 1]) == {"actor": [1, 8], "rollout": [1, 8]}
            assert "--colocate" in command and "--selective-offload" in command
            assert "--hybrid" not in command and "--agentic-prelaunch" not in command
            assert "--over-sampling-batch-size" not in command
            assert command[command.index("--max-staleness") + 1] == "0"
            assert micro == 4
    if hybrid or score_centering:
        (Path(expected)).mkdir(parents=True)
        (Path(expected) / "latest_checkpointed_iteration.txt").write_text("1")
        retry = subprocess.run(
            ["bash", str(script)], env=env, cwd=tmp_path, capture_output=True, text=True, timeout=30
        )
        assert retry.returncode != 0 and "Fresh training" in retry.stderr
