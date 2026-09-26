# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""CPU accounting integration; no Ray cluster, SGLang engine or GPU is started.

Hand-checkable report (also asserted below): exported A->B and A->C cover
A=(1,2,1,2), B=(9,10,2,11), C=(2,4,1,3), ordered as accepted, proposed,
verify, completion. Unique totals are (12,16,4,16), giving 75% acceptance
and 4 tokens/verify. Four record occurrences represent three generations.
These are accounting results, not evidence of speculative speedup.

Run on the review server with:
pytest tests/integration/test_agentic_speculative_metrics.py
"""

import asyncio
import importlib.util
import json
import sys
import time
from argparse import Namespace
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

import relax.utils
from relax.agentic.pipeline import TrainingFieldArtifact
from relax.agentic.pipeline.runtime import BackendGenerateResult
from relax.agentic.session.service import AgenticSessionShard
from relax.agentic.session.state import InflightRequest, MsgNode, RequestKind, SessionForest
from relax.utils.metrics.speculative_metrics import compute_speculative_metrics
from relax.utils.speculative import SpeculativeCounts
from relax.utils.types import Sample


class _Tokenizer:
    def decode(self, tokens: list[int], **kwargs: Any) -> str:
        return "".join(chr(token) for token in tokens)


def _session(session_id: str) -> tuple[Any, SimpleNamespace, MsgNode]:
    # Invoke the real actor implementation locally, without .remote() or init.
    shard_cls = AgenticSessionShard.__ray_metadata__.modified_class
    shard = object.__new__(shard_cls)
    shard.args = Namespace(
        agentic_reasoning_parser=None,
        agentic_tool_call_parser=None,
        use_rollout_routing_replay=False,
    )
    shard._generation_backend = SimpleNamespace(
        tokenizer=_Tokenizer(), compiler=SimpleNamespace(apply_chat_template_kwargs={})
    )
    forest = SessionForest.create_empty(session_id=session_id)
    root = forest.append_obs(
        parent_state_hash=forest.root_state_hash,
        rollout_id=0,
        abort_count=0,
        messages_delta=[{"role": "user", "content": "prompt"}],
        train_token_delta=[112],
        rollout_token_delta=[112],
    )
    session = SimpleNamespace(
        forest=forest, resp_state_hash_by_request_id={}, group=SimpleNamespace(rollout_mode="train")
    )
    return shard, session, root


def _request(parent: MsgNode, request_id: str) -> InflightRequest:
    return InflightRequest(
        request_id=request_id,
        parent_state_hash=parent.state_hash,
        rollout_id=0,
        kind=RequestKind.FRESH,
        abort_count=0,
        waiter=asyncio.get_running_loop().create_future(),
        wall_started_at=time.monotonic(),
    )


def _attempt(text: str, counts: SpeculativeCounts, finish_type: str = "stop") -> BackendGenerateResult:
    values = {
        "spec_num_correct_drafts": counts.accepted,
        "spec_num_proposed_drafts": counts.proposed,
        "spec_verify_ct": counts.verify,
        "completion_tokens": counts.completion,
    }
    meta = {key: value for key, value in values.items() if value is not None}
    meta["finish_reason"] = {"type": finish_type}
    return BackendGenerateResult(
        new_tokens=[ord(char) for char in text],
        new_log_probs=[-0.1] * len(text),
        finish_type=finish_type,
        meta_info=meta,
        elapsed=0.01,
    )


def _commit(
    shard: Any,
    session: SimpleNamespace,
    parent: MsgNode,
    request_id: str,
    text: str,
    counts: SpeculativeCounts,
) -> MsgNode:
    request = _request(parent, request_id)
    shard._apply_generate_result(request, _attempt(text, counts))
    request.pending_status = "completed"
    shard._terminal_response_locked(session=session, ir=request, finish_type="stop")
    request.waiter.cancel()
    return session.forest.nodes_by_hash[session.resp_state_hash_by_request_id[request_id]]


def _export(shard: Any, session: SimpleNamespace, leaves: list[MsgNode]) -> list[Sample]:
    transport = shard._build_session_export_transport(
        session=session,
        reward=None,
        metadata={},
        finalize_events={},
        output_records=tuple(
            {
                "name": str(index),
                "metadata": {},
                "messages": session.forest.full_messages(leaf.state_hash),
            }
            for index, leaf in enumerate(leaves)
        ),
    )
    samples = []
    for export in transport.exports:
        restored = TrainingFieldArtifact(sample_payload=export["sample_payload"]).to_sample()
        samples.append(Sample.from_dict(json.loads(json.dumps(restored.to_dict()))))
    return samples


@pytest.fixture
def rollout_logger(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Load the real logging entrypoints, replacing only external boundaries.

    A private module name prevents cached mocks from leaking into other tests.
    Neither compute_metrics_from_samples nor the speculative reducer is mocked.
    """
    constants = ModuleType("sglang.srt.constants")
    for name in ("GPU_MEMORY_TYPE_CUDA_GRAPH", "GPU_MEMORY_TYPE_KV_CACHE", "GPU_MEMORY_TYPE_WEIGHTS"):
        setattr(constants, name, name)
    engine = ModuleType("relax.backends.sglang.sglang_engine")
    engine.SGLangEngine = type("UnusedEngine", (), {})
    tracking = ModuleType("relax.utils.tracking_utils")
    tracking.log = Mock()
    tracking.flush_metrics = Mock()
    tracking.init_tracking = Mock()
    monkeypatch.setitem(sys.modules, constants.__name__, constants)
    monkeypatch.setitem(sys.modules, engine.__name__, engine)
    monkeypatch.setitem(sys.modules, "transfer_queue", ModuleType("transfer_queue"))
    monkeypatch.setitem(sys.modules, tracking.__name__, tracking)
    monkeypatch.setattr(relax.utils, "tracking_utils", tracking, raising=False)
    path = Path(__file__).resolve().parents[2] / "relax/distributed/ray/rollout.py"
    name = "relax.distributed.ray._speculative_logging_test"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "save_rollout_result_jsonl", Mock())
    # Performance accounting is unrelated to speculative accounting.
    monkeypatch.setattr(module, "compute_perf_metrics_from_samples", lambda *args: {})
    return module


def _log(module: ModuleType, samples: list[Sample], enabled: bool = False) -> dict[str, Any]:
    args = Namespace(
        custom_rollout_log_function_path=None,
        load_debug_rollout_data=None,
        sglang_speculative_algorithm="EAGLE" if enabled else None,
        log_reward_category=None,
        log_passrate=False,
        reward_key=None,
        advantage_estimator="grpo",
        n_samples_per_prompt=1,
        partial_rollout=False,
        fully_async=False,
        wandb_always_use_train_step=False,
    )
    module._log_rollout_data(7, args, samples, {}, 1.0)
    call = module.tracking_utils.log.call_args
    assert call.kwargs["step_key"] == "rollout/step"
    module.tracking_utils.flush_metrics.assert_called_with(args, 7)
    return call.args[1]


@pytest.mark.asyncio
async def test_agentic_speculative_metrics_branch_report_and_logging(rollout_logger: ModuleType) -> None:
    shard, session, prompt = _session("branches")
    a = _commit(shard, session, prompt, "a", "aa", SpeculativeCounts(1, 2, 1, 2))
    b = _commit(shard, session, a, "b", "b" * 11, SpeculativeCounts(9, 10, 2, 11))
    c = _commit(shard, session, a, "c", "ccc", SpeculativeCounts(2, 4, 1, 3))
    _commit(shard, session, prompt, "discarded", "x", SpeculativeCounts(99, 99, 1, 100))
    samples = _export(shard, session, [b, c])
    expected = compute_speculative_metrics(samples)
    logged = _log(rollout_logger, samples)
    for key, value in expected.items():
        assert logged[f"rollout/{key}"] == value
    assert logged["rollout/spec/unique_generation_count"] == 3
    assert logged["rollout/spec/record_occurrence_count"] == 4
    assert logged["rollout/spec/accepted_total"] == 12
    assert logged["rollout/spec/proposed_total"] == 16
    assert logged["rollout/spec/accept_rate"] == 0.75
    assert logged["rollout/spec/tokens_per_verify"] == 4
    assert logged["rollout/spec/accept_count_coverage"] == 1
    assert _log(rollout_logger, samples) == logged


@pytest.mark.asyncio
async def test_agentic_speculative_metrics_equal_content_independent_requests() -> None:
    shard, session, prompt = _session("same-content")
    node = _commit(shard, session, prompt, "one", "same", SpeculativeCounts(1, 2, 1, 4))
    other = _commit(shard, session, prompt, "two", "same", SpeculativeCounts(9, 10, 2, 4))
    assert node is other
    metrics = compute_speculative_metrics(_export(shard, session, [node]))
    assert metrics["spec/unique_generation_count"] == 2
    assert metrics["spec/accept_rate"] == pytest.approx(10 / 12)


@pytest.mark.asyncio
async def test_agentic_speculative_metrics_same_request_name_across_sessions() -> None:
    samples = []
    for session_id, counts in (("one", SpeculativeCounts(1, 2)), ("two", SpeculativeCounts(9, 10))):
        shard, session, prompt = _session(session_id)
        node = _commit(shard, session, prompt, "same-request", "same", counts)
        samples.extend(_export(shard, session, [node]))
    metrics = compute_speculative_metrics(samples)
    assert metrics["spec/unique_generation_count"] == 2
    assert metrics["spec/accept_rate"] == pytest.approx(10 / 12)


@pytest.mark.asyncio
async def test_agentic_speculative_metrics_request_normalizes_backend_values() -> None:
    shard, session, prompt = _session("normalization")
    request = _request(prompt, "values")
    result = _attempt("a", SpeculativeCounts())
    result.meta_info.update(
        {
            "spec_num_correct_drafts": "bad",
            "spec_num_proposed_drafts": "10",
            "spec_verify_ct": "2",
            "completion_tokens": "5",
        }
    )
    shard._apply_generate_result(request, result)
    assert request.pending_spec_counts == SpeculativeCounts(None, 10, 2, 5)
    assert request.pending_spec_delta["completion_token_num"] == 5
    assert session.forest.committed_generations == {}
    request.waiter.cancel()


@pytest.mark.asyncio
async def test_agentic_speculative_metrics_request_ignores_invalid_backend_counters() -> None:
    shard, session, prompt = _session("invalid-counters")
    request = _request(prompt, "invalid-values")
    result = _attempt("a", SpeculativeCounts())

    result.meta_info.update(
        {
            "spec_num_correct_drafts": "bad",
            "spec_num_proposed_drafts": "10",
            "spec_verify_ct": "bad",
            "completion_tokens": -1,
        }
    )

    shard._apply_generate_result(request, result)

    assert request.pending_spec_counts == SpeculativeCounts(
        None,
        10,
        None,
        None,
    )
    assert request.pending_spec_delta["spec_accept_token_num"] == 0
    assert request.pending_spec_delta["spec_draft_token_num"] == 10
    assert request.pending_spec_delta["spec_verify_ct"] == 0
    assert request.pending_spec_delta["completion_token_num"] == 0
    assert session.forest.committed_generations == {}

    request.waiter.cancel()


@pytest.mark.asyncio
async def test_agentic_speculative_metrics_resume_missing_attempt_and_uncommitted() -> None:
    shard, session, prompt = _session("resume")
    request = _request(prompt, "stable")
    shard._apply_generate_result(request, _attempt("a", SpeculativeCounts(1, 2, None, 1), "abort"))
    assert session.forest.committed_generations == {}
    request.abort_count += 1
    request.kind = RequestKind.RESUMED
    shard._apply_generate_result(request, _attempt("b", SpeculativeCounts(9, 10, 2, 1)))
    request.pending_status = "completed"
    shard._terminal_response_locked(session=session, ir=request, finish_type="stop")
    shard._terminal_response_locked(session=session, ir=request, finish_type="stop")
    request.waiter.cancel()
    node = session.forest.nodes_by_hash[session.resp_state_hash_by_request_id["stable"]]
    pending = _request(node, "not-committed")
    shard._apply_generate_result(pending, _attempt("x", SpeculativeCounts(100, 100, 1, 100), "abort"))
    metrics = compute_speculative_metrics(_export(shard, session, [node]))
    pending.waiter.cancel()
    assert metrics["spec/unique_generation_count"] == 1
    assert metrics["spec/accept_rate"] == pytest.approx(10 / 12)
    assert metrics["spec/verify_count_coverage"] == 0
    assert "spec/tokens_per_verify" not in metrics


def test_agentic_speculative_metrics_empty_logging(rollout_logger: ModuleType) -> None:
    logged = _log(rollout_logger, [], enabled=True)
    assert logged["rollout/spec/unique_generation_count"] == 0
    assert "rollout/spec/accept_rate" not in logged


@pytest.mark.asyncio
async def test_agentic_speculative_metrics_missing_and_legacy_logging(rollout_logger: ModuleType) -> None:
    shard, session, prompt = _session("unknown")
    node = _commit(shard, session, prompt, "unknown", "a", SpeculativeCounts())
    samples = _export(shard, session, [node])
    samples.append(Sample.from_dict({"status": "completed"}))
    logged = _log(rollout_logger, samples)
    assert logged["rollout/spec/legacy_sample_count"] == 1
    assert logged["rollout/spec/accept_count_coverage"] == 0
    assert "rollout/spec/accept_rate" not in logged
    assert "rollout/spec_accept_rate" not in logged
