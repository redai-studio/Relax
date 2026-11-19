# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""CPU contract tests: actual HTTP adapter -> Forest -> explicit exports ->
loss mask."""

import ast
import asyncio
import base64
import copy
import importlib.util
import json
import logging
import os
import signal
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import httpx
import pytest

from examples.osworld_agentic.app.reward_post_process import compute_advantages, compute_legacy_raw_reward
from examples.osworld_agentic.app.visual_context import VisualContext, image_count
from relax.agentic.pipeline import TrainingFieldArtifact
from relax.agentic.session.service import AgenticSessionShard
from relax.agentic.session.state import SessionForest, check_messages, normalize_tools
from relax.utils.logging_utils import get_logger
from relax.utils.utils import convert_samples_to_train_data


ROOT = Path(__file__).resolve().parents[2]


def observation(turn):
    return {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64," + base64.b64encode(bytes([turn])).decode()},
            },
            {"type": "text", "text": f"Tool result: {turn}; Instruction: task; Previous actions: {list(range(turn))}"},
        ],
    }


class MockUpstream:
    """Pinned upstream shape: rebuilt window, dynamic system tools, same-prompt
    parse retries."""

    def __init__(self, **kwargs):
        self.max_steps = kwargs.get("max_steps", 30)
        self.actions, self.responses = [], []
        self._input_toks_per_step, self._output_toks_per_step = [], []

    def reset(self, *args):
        pass

    def record_exec_result(self, *args):
        pass

    def predict(self, instruction, obs, step_idx):
        messages = [{"role": "system", "content": obs.get("system", "rules and tool definitions")}]
        # The old user message loses its original instruction in upstream's rebuild.
        if step_idx and self.responses:
            old = observation(step_idx - 1)
            old["content"] = old["content"][:1]
            messages.extend([old, {"role": "assistant", "content": self.responses[-1]}])
        messages.append(observation(step_idx))
        payload = {
            "messages": messages,
            "tools": obs.get("tools", []),
            "chat_template_kwargs": obs.get("template", {}),
        }
        for _ in range(5):
            response = self.call_llm(payload, "mock")
            if response != "bad parse":
                break
        self.responses.append(response)
        self.actions.append(response)
        return response, ["DONE"] if step_idx + 1 == self.max_steps else ["click()"], {"action_type": "gui"}


@pytest.fixture
def hybrid(monkeypatch):
    external = ModuleType("mm_agents.hybrid_agent_local")
    external.HybridAgentLocal = MockUpstream
    monkeypatch.setitem(sys.modules, "mm_agents", ModuleType("mm_agents"))
    monkeypatch.setitem(sys.modules, "mm_agents.hybrid_agent_local", external)
    name = "examples.osworld_agentic.app.hybrid_agent"
    spec = importlib.util.spec_from_file_location(name, ROOT / "examples/osworld_agentic/app/hybrid_agent.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setenv("OPENAI_BASE_URL", "http://mock-only/v1")
    return module


class MockBackend:
    """Deterministic fake codec and generation, real Forest
    matching/commit/export.

    Each input byte is one token, each image 32 tokens. Generated token IDs are
    deliberately different from re-encoding answer text, exposing
    retokenization. No image processing, HTTP, VM, tokenizer download or GPU is
    involved.
    """

    apply_chat_template_kwargs = {}

    def __init__(self, monkeypatch, hybrid, replies=None):
        self.requests, self.generated, self.parents = [], [], []
        self.replies = iter(replies) if replies else None
        self.forest = SessionForest.create_empty(session_id="session", group_index=7, index=42)
        self.shard = object.__new__(AgenticSessionShard.__ray_metadata__.modified_class)
        self.shard.args = SimpleNamespace(use_rollout_routing_replay=False)
        from relax.agentic.observability import ShardTrajectoryRecorder

        self.shard._trajectory_recorder = ShardTrajectoryRecorder(self.shard.args, shard_ordinal=0)
        self.shard._generation_backend = SimpleNamespace(compiler=self, tokenizer=self)
        self.shard._raise_if_observation_exceeds_context = lambda **kwargs: None
        monkeypatch.setattr(hybrid.httpx, "post", self.post)

    def decode(self, token_ids, **kwargs):
        return " ".join(map(str, token_ids))

    async def encode_messages(self, messages, **kwargs):
        ids = []
        for message in messages:
            ids.extend(message["role"].encode())
            content = message["content"]
            for block in content if isinstance(content, list) else [{"type": "text", "text": content}]:
                ids.extend([999] * 32 if block["type"] == "image_url" else block["text"].encode())
        return SimpleNamespace(
            train_prompt_ids=ids,
            backend_prompt_ids=ids,
            multimodal_train_inputs=None,
            backend_image_data=[],
            backend_audio_data=[],
            backend_video_data=[],
            timing={},
        )

    encode_observation_delta = encode_messages

    def post(self, *args, json, **kwargs):
        self.requests.append(copy.deepcopy(json))
        messages = check_messages(json["messages"])
        tools, template = normalize_tools(json.get("tools")), json.get("chat_template_kwargs") or {}
        parent, delta = self.shard._match_parent_state(
            forest=self.forest,
            messages=messages,
            tools=tools,
            chat_template_kwargs=template,
        )
        self.parents.append(parent)
        obs, _ = asyncio.run(
            self.shard._append_observation(
                session=SimpleNamespace(rollout_id=0),
                forest=self.forest,
                parent=parent,
                messages_delta=delta,
                tools=tools,
                chat_template_kwargs=template,
            )
        )
        i = len(self.generated)
        content = next(self.replies) if self.replies else f"answer-{i}"
        ids, probs = [100_000 + 2 * i, 100_001 + 2 * i], [-0.1 * (i + 1), -0.1 * (i + 1) - 0.01]
        self.generated.append((ids, probs))
        self.forest.append_resp(
            parent_state_hash=obs.state_hash,
            rollout_id=0,
            abort_count=0,
            messages_delta=[{"role": "assistant", "content": content}],
            token_delta=ids,
            logprob_delta=probs,
        )
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": self.forest.train_token_count(obs.state_hash),
                    "completion_tokens": len(ids),
                },
            },
        )

    def export(self, rows):
        from relax.agentic.runner.ipc import _normalize_session_output_record

        transport = self.shard._build_session_export_transport(
            session=SimpleNamespace(
                forest=self.forest, group=SimpleNamespace(rollout_mode="train", group_id="group"), observation=None
            ),
            reward=None,
            metadata={},
            output_records=tuple(_normalize_session_output_record(row) for row in rows),
            finalize_events={},
        )
        return [TrainingFieldArtifact(sample_payload=e["sample_payload"]).to_sample() for e in transport.exports]


def adapter_functions(**extras):
    path = ROOT / "examples/osworld_agentic/app/agent.py"
    names = {"run_session", "_predict_one_step", "write_session_output", "_extract_task_config"}
    nodes = [
        n
        for n in ast.parse(path.read_text()).body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in names
    ]
    tree = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *nodes],
        type_ignores=[],
    )
    ns = dict(asyncio=asyncio, signal=signal, logging=logging, os=os, json=json, Path=Path, **extras)
    exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), ns)
    return ns


def export_agent(agent, backend, tmp_path):
    vcs = agent.visual_context
    payload = {"_turn_records": agent.turn_records, "reward": 1, "metadata": {"agent_turns": len(agent.actions)}}
    if vcs:
        vcs.finish("agent_done")
        payload["_segment_records"] = vcs.records
        payload["metadata"].update(context_mode="vcs", model_calls=len(agent.turn_records))
    output = tmp_path / "records.jsonl"
    adapter_functions()["write_session_output"](output, payload)
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    return rows, backend.export(rows)


def trained_pairs(samples):
    return [
        (token, prob)
        for sample in samples
        for token, prob, mask in zip(
            sample.tokens[-sample.response_length :],
            sample.rollout_log_probs,
            sample.loss_mask,
            strict=True,
        )
        if mask
    ]


@pytest.mark.parametrize("mode", ["baseline", "vcs"])
def test_long_trajectory_actual_export_tokens_masks_and_normalization(hybrid, monkeypatch, tmp_path, mode):
    backend = MockBackend(monkeypatch, hybrid)
    agent = hybrid.RelaxHybridAgent(context_mode=mode)
    for turn in range(10):
        agent.predict("task", {}, step_idx=turn)
    rows, samples = export_agent(agent, backend, tmp_path)
    expected = [(t, p) for ids, probs in backend.generated for t, p in zip(ids, probs, strict=True)]
    actual = trained_pairs(samples)
    assert [x[0] for x in actual] == [x[0] for x in expected]
    assert [x[1] for x in actual] == pytest.approx([x[1] for x in expected])
    assert len({token for token, _ in actual}) == 20
    assert {s.index for s in samples} == {42}
    assert {s.group_index for s in samples} == {7}
    assert sum(s.reward is not None for s in samples) == 1
    for sample in samples:
        assert len(sample.loss_mask) == len(sample.rollout_log_probs) == sample.response_length
        assert all(p == 0 for m, p in zip(sample.loss_mask, sample.rollout_log_probs, strict=True) if not m)
        sample.custom_advantage = 1.0
    data = convert_samples_to_train_data(
        SimpleNamespace(
            custom_reward_post_process_path=None,
            agentic_custom_advantage_path="mock",
            use_opd=False,
            debug_train_only=True,
        ),
        samples,
    )
    assert data["sample_index_mask_sums"] == [20] * len(samples)
    comparison = {
        "mode": mode,
        "model_calls": len(backend.generated),
        "environment_steps": len(agent.actions),
        "training_rows": len(samples),
        "request_images": sum(image_count(r["messages"]) for r in backend.requests),
        "training_images": sum(image_count(r["messages"]) for r in rows),
        "trained_answer_tokens": sum(sum(s.loss_mask) for s in samples),
        "rollout_input_tokens": sum(agent._input_toks_per_step),
        "training_tokens": sum(len(s.tokens) for s in samples),
    }
    (tmp_path / f"comparison-{mode}.json").write_text(json.dumps(comparison, indent=2))
    if mode == "baseline":
        assert len(rows) == 10
        assert [image_count(r["messages"]) for r in backend.requests] == [1] + [2] * 9
        assert all(s.metadata["agentic_trace"]["turn_count"] == 1 for s in samples)
    else:
        assert len(rows) == 4
        assert [r["metadata"]["vcs_segment"]["new_answers"] for r in rows] == [4, 2, 2, 2]
        assert [image_count(r["messages"]) for r in backend.requests] == [1, 2, 3, 4, 3, 4, 3, 4, 3, 4]
        for i, parent in enumerate(backend.parents):
            if i in (0, 4, 6, 8):
                assert parent.state_hash == backend.forest.root_state_hash
            else:
                assert backend.requests[i]["messages"][:-1] == agent.turn_records[i - 1]["messages"]
        # Replayed answers are before the first generated node: prompt, not targets.
        assert [s.metadata["agentic_trace"]["turn_count"] for s in samples] == [4, 2, 2, 2]
        after_trim = backend.requests[4]["messages"]
        assert [m["content"] for m in after_trim if m["role"] == "assistant"] == [f"answer-{i}" for i in range(4)]
        for i in range(5):
            assert observation(i)["content"][-1] in [
                b for m in after_trim if isinstance(m["content"], list) for b in m["content"]
            ]
        assert rows[0]["metadata"]["vcs_segment"]["trigger_images"] == 5
        assert rows[0]["metadata"]["vcs_segment"]["pruned_images"] == 2
        assert rows[-1]["metadata"]["vcs_segment"]["reason"] == "agent_done"


@pytest.mark.parametrize("keep,prune", [(2, 2), (4, 4), (1, 2)])
def test_exact_window_threshold(hybrid, monkeypatch, keep, prune):
    backend = MockBackend(monkeypatch, hybrid)
    agent = hybrid.RelaxHybridAgent(context_mode="vcs", vcs_keep_images=keep, vcs_prune_images=prune)
    threshold = keep + prune
    for turn in range(threshold):
        agent.predict("task", {}, step_idx=turn)
        assert agent.visual_context.records == []
    assert image_count(backend.requests[-1]["messages"]) == threshold
    agent.predict("task", {}, step_idx=threshold)
    assert len(agent.visual_context.records) == 1
    assert image_count(backend.requests[-1]["messages"]) == keep + 1


def test_identical_parse_retries_and_tools_reversion_have_unique_loss(hybrid, monkeypatch, tmp_path):
    backend = MockBackend(monkeypatch, hybrid, replies=["bad parse", "bad parse", "ok", "ok", "ok", "ok"])
    agent = hybrid.RelaxHybridAgent(context_mode="vcs")
    agent.predict("task", {}, step_idx=0)
    tool = {"type": "function", "function": {"name": "click", "parameters": {"type": "object"}}}
    agent.predict("task", {"tools": [tool], "system": "new tool definitions"}, step_idx=1)
    agent.predict("task", {}, step_idx=2)
    agent.predict("task", {"template": {"enable_thinking": False}}, step_idx=3)
    rows, samples = export_agent(agent, backend, tmp_path)
    assert len(samples) == 6
    assert len(trained_pairs(samples)) == 12
    assert [r["metadata"]["vcs_segment"]["reason"] for r in rows[:2]] == ["parse_retry"] * 2
    assert all(p.state_hash == backend.forest.root_state_hash for p in backend.parents)
    assert "bad parse" not in str(backend.requests[2])
    assert backend.requests[3]["tools"] == [tool]
    assert backend.requests[-1]["chat_template_kwargs"] == {"enable_thinking": False}
    assert [c["parse_attempt"] for c in agent.turn_records] == [0, 1, 2, 0, 0, 0]
    agent.visual_context.finish("duplicate_finish")
    assert len(agent.visual_context.records) == 6


@pytest.mark.parametrize("norm", ["0", "1"])
def test_segments_share_trajectory_advantage_and_legacy_metric(monkeypatch, norm):
    monkeypatch.setenv("OSWORLD_PER_TRAJ_NORM", norm)
    baseline, vcs = [], []
    for outcome, calls in [(1, 10), (0, 6)]:
        md = {"outcome": outcome, "agent_turns": calls}
        baseline.append({f"turn_{i}": dict(md) for i in range(calls)})
        vcs.append({f"segment_{i}": dict(md, context_mode="vcs", model_calls=calls) for i in range(2)})
    old, new = compute_advantages(baseline), compute_advantages(vcs)
    assert [set(s.values()) for s in old] == [set(s.values()) for s in new]

    def metric_samples(sessions):
        return [
            SimpleNamespace(session_id=str(i), metadata=md)
            for i, session in enumerate(sessions)
            for md in session.values()
        ]

    assert compute_legacy_raw_reward(metric_samples(baseline)) == compute_legacy_raw_reward(metric_samples(vcs))


@pytest.mark.parametrize("mode,fail_at", [("baseline", None), ("vcs", None), ("vcs", 5)])
def test_real_session_driver_mock_environment_tail_and_early_stop(hybrid, monkeypatch, tmp_path, mode, fail_at):
    backend = MockBackend(monkeypatch, hybrid)
    actions = []
    obs = SimpleNamespace(
        screenshot_bytes=b"mock", cur_app=None, app_info=None, tool_list=[], done=False, exec_ok=True, exec_result={}
    )

    class Env:
        def __init__(self, **kwargs):
            pass

        async def reset(self):
            return obs

        async def exec_pyautogui(self, code):
            actions.append(code)
            return obs

        async def evaluate(self):
            return 1.0

        async def close(self):
            actions.append("close")

    post = backend.post

    def maybe_fail(*args, **kwargs):
        if fail_at is not None and len(backend.generated) == fail_at:
            raise RuntimeError("mock model unavailable")
        return post(*args, **kwargs)

    monkeypatch.setattr(hybrid.httpx, "post", maybe_fail)
    ns = adapter_functions(
        time=SimpleNamespace(monotonic=lambda: 1),
        _load_config=lambda: {"max_turns": 10, "context_mode": mode},
        OSWorldEnv=Env,
        RelaxHybridAgent=hybrid.RelaxHybridAgent,
        parse_mcp_action=lambda a: None,
        SessionDiscardedError=hybrid.SessionDiscardedError,
        ContextLengthExceededError=hybrid.ContextLengthExceededError,
        logger=get_logger(__name__),
        emit=lambda *a, **k: None,
    )
    result = asyncio.run(ns["run_session"]({"task_config": {"id": "task", "instruction": "task"}}, {}))
    count = fail_at or 10
    assert result["metadata"]["agent_turns"] == count
    assert result["metadata"]["_session_metrics"]["osworld/model_calls"] == count
    assert len(actions) == count + 1 and actions[-1] == "close"
    if mode == "vcs":
        assert result["_segment_records"][-1]["vcs_segment"]["reason"] == (
            "predict_error" if fail_at else "agent_done"
        )
        assert result["metadata"]["_session_metrics"]["osworld/answer_tokens"] == count * 2
    else:
        assert "_segment_records" not in result
        assert not any("vcs" in key for key in result["metadata"]["_session_metrics"])
    path = tmp_path / "session.jsonl"
    ns["write_session_output"](path, result)
    samples = backend.export([json.loads(line) for line in path.read_text().splitlines()])
    assert len(trained_pairs(samples)) == count * 2


def test_vcs_configuration_is_explicit_and_validated(hybrid):
    assert hybrid.RelaxHybridAgent().visual_context is None
    with pytest.raises(ValueError, match="positive"):
        VisualContext(0, 2)
    with pytest.raises(ValueError, match="context_mode"):
        hybrid.RelaxHybridAgent(context_mode="unknown")
    with pytest.raises(ValueError, match="context_policy"):
        hybrid.RelaxHybridAgent(context_mode="vcs", context_policy="skip_on_no_change")


def test_changed_retry_prompt_is_not_silently_discarded(hybrid, monkeypatch):
    MockBackend(monkeypatch, hybrid)
    agent = hybrid.RelaxHybridAgent(context_mode="vcs")
    agent.predict("task", {}, step_idx=0)
    request = {"messages": [{"role": "system", "content": "rules"}, observation(1)]}
    with pytest.raises(ValueError, match="retries must reuse"):
        agent.call_llm(request, "mock")


def test_tool_change_at_image_boundary_records_both_causes(hybrid, monkeypatch, tmp_path):
    backend = MockBackend(monkeypatch, hybrid)
    agent = hybrid.RelaxHybridAgent(context_mode="vcs")
    for turn in range(5):
        agent.predict("task", {"system": "changed"} if turn == 4 else {}, step_idx=turn)
    rows, samples = export_agent(agent, backend, tmp_path)
    boundary = rows[0]["metadata"]["vcs_segment"]
    assert boundary["reason"] == "tools_or_system_or_template_changed+image_window"
    assert boundary["pruned_images"] == 2
    assert agent.visual_context.metrics()["osworld/vcs_pruned_images"] == 2
    assert len(trained_pairs(samples)) == 10
    assert image_count(backend.requests[-1]["messages"]) == 3
