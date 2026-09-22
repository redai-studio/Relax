# Copyright (c) 2026 Relax Authors. All Rights Reserved.
from __future__ import annotations

import asyncio
import json
import sys
import time
from argparse import Namespace
from collections import deque
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from relax.agentic.pipeline import (
    GroupExport,
    GroupInput,
    SampleExport,
    SessionExport,
)
from relax.agentic.pipeline import runtime as runtime_mod
from relax.agentic.pipeline.prepare import PrepareDomain
from relax.agentic.pipeline.reward import RewardDomain
from relax.agentic.pipeline.runtime import RuntimeDomain, SGLangBackendAdapter, _build_session_specs
from relax.agentic.pipeline.transfer import TransferBatch, TransferDomain
from relax.agentic.session.admission import (
    AdmissionAction,
    AdmissionReason,
    BudgetState,
    WorkerSnapshot,
    compute_reservation_tokens,
)
from relax.agentic.session.admission_coordinator import AdmissionCoordinator, _parse_engine_kv_gauges
from relax.agentic.session.service import (
    AgenticChatRequestError,
    AgenticSessionShard,
    ResidentGroup,
    _normalized_chat_request,
    _openai_token_logprobs_payload,
    _SessionRecord,
    _SessionResultCell,
)
from relax.agentic.session.state import InflightRequest, RequestKind, SessionForest, check_messages
from relax.utils.types import Sample


# ``relax.agentic.rollout`` only needs this logging helper at import time. Keep
# the agentic unit tests collectible in environments without the optional
# SGLang backend, while avoiding a process-wide stub after this import.
_rollout_module_name = "relax.distributed.ray.rollout"
_existing_rollout_module = sys.modules.get(_rollout_module_name)
if _existing_rollout_module is None:
    _rollout_module_stub = ModuleType(_rollout_module_name)
    _rollout_module_stub._log_rollout_data = MagicMock()
    sys.modules[_rollout_module_name] = _rollout_module_stub
try:
    from relax.agentic.rollout import AgenticResidentPipeline, _StepContext  # noqa: E402
finally:
    if _existing_rollout_module is None:
        sys.modules.pop(_rollout_module_name, None)


def _runtime_args(**overrides):
    base = {
        "agent_command": "python -c 'pass'",
        "agent_cwd": None,
        "agent_env": [],
        "agentic_custom_advantage_path": None,
        "hf_checkpoint": "/tmp/relax-test-model",
        "mm_processor_pool_size": 0,
        "rollout_batch_size": 2,
        "n_samples_per_prompt": 2,
        "over_sampling_batch_size": None,
        "rollout_max_context_len": 4096,
        "rollout_max_response_len": 128,
        "rollout_temperature": 1.0,
        "rollout_top_p": 1.0,
        "rollout_top_k": -1,
        "rollout_stop": None,
        "rollout_stop_token_ids": None,
        "rollout_skip_special_tokens": False,
        "group_rm": False,
        "reward_max_concurrency": None,
        "partial_rollout": False,
        "fully_async": False,
        "max_staleness": 0,
        "colocate": True,
        "global_batch_size": 2,
        "num_iters_per_train_update": 1,
    }
    base.update(overrides)
    if base["over_sampling_batch_size"] is None:
        base["over_sampling_batch_size"] = base["rollout_batch_size"]
    return SimpleNamespace(**base)


async def test_reward_domain_delegates_sample_reward_to_executor(monkeypatch) -> None:
    calls: list[Sample] = []

    async def fake_async_rm(args, sample):
        del args
        calls.append(sample)
        return 1.0

    monkeypatch.setattr("relax.agentic.pipeline.reward.async_rm", fake_async_rm)
    reward_domain = RewardDomain(
        args=_runtime_args(reward_max_concurrency=1),
        rollout_mode="train",
        group_filter=None,
    )
    sample = Sample(index=0, group_index=0, session_id="sample-0", metadata={})

    await reward_domain._score_sample(sample)

    assert calls == [sample]
    assert sample.reward == 1.0


async def test_reward_domain_delegates_group_reward_to_executor(monkeypatch) -> None:
    calls: list[list[Sample]] = []

    async def fake_batched_async_rm(args, samples):
        del args
        calls.append(samples)
        return [float(sample.index) for sample in samples]

    monkeypatch.setattr("relax.agentic.pipeline.reward.batched_async_rm", fake_batched_async_rm)
    reward_domain = RewardDomain(
        args=_runtime_args(group_rm=True, reward_max_concurrency=1),
        rollout_mode="train",
        group_filter=None,
    )
    samples = [Sample(index=index, group_index=0, session_id=f"sample-{index}", metadata={}) for index in range(3)]
    group = GroupExport(
        group_id="group-reward",
        sessions=tuple(SessionExport(exports=(SampleExport(name=None, sample=sample),)) for sample in samples),
    )

    await reward_domain.score_group(group)

    assert calls == [samples]
    assert [sample.reward for sample in samples] == [0.0, 1.0, 2.0]


class _FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [ord(ch) for ch in str(text)]

    def decode(self, token_ids, skip_special_tokens=False):
        del skip_special_tokens
        return "".join(chr(token_id) for token_id in token_ids)


def _chars(text: str) -> list[int]:
    return [ord(ch) for ch in text]


def _forest_with_initial_obs(
    *,
    session_id: str,
    messages: list[dict[str, Any]],
    train_token_delta: list[int],
    rollout_token_delta: list[int],
    rollout_id: int = 0,
    metadata: dict[str, Any] | None = None,
    group_index: int | None = None,
    index: int | None = None,
    label: str | None = None,
    train_metadata: dict[str, Any] | None = None,
):
    forest = SessionForest.create_empty(
        session_id=session_id,
        group_index=group_index,
        index=index,
        label=label,
        train_metadata=train_metadata,
        metadata=metadata,
    )
    initial_obs = forest.append_obs(
        parent_state_hash=forest.root_state_hash,
        rollout_id=rollout_id,
        abort_count=0,
        messages_delta=check_messages(messages),
        train_token_delta=list(train_token_delta),
        rollout_token_delta=list(rollout_token_delta),
    )
    return forest, initial_obs


def test_session_parent_match_canonicalizes_python_typescript_tool_arguments() -> None:
    def tool_call_message(arguments: str) -> dict[str, Any]:
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "run", "arguments": arguments},
                }
            ],
        }

    system_message = {"role": "system", "content": "Use tools."}
    user_message = {"role": "user", "content": "Run the tool"}
    python_arguments = json.dumps({"c": "d", "a": "b"})
    typescript_arguments = '{"a":"b","c":"d"}'

    forest, observation = _forest_with_initial_obs(
        session_id="python-typescript-tool-arguments",
        messages=[system_message, user_message],
        train_token_delta=[1, 2],
        rollout_token_delta=[1, 2],
    )
    response = forest.append_resp(
        parent_state_hash=observation.state_hash,
        rollout_id=0,
        abort_count=0,
        messages_delta=check_messages([tool_call_message(python_arguments)]),
        token_delta=[3],
        logprob_delta=[-0.1],
    )
    tool_result = {"role": "tool", "tool_call_id": "call-1", "content": "done"}
    replayed_messages = check_messages(
        [system_message, user_message, tool_call_message(typescript_arguments), tool_result]
    )

    parent, unmatched = AgenticSessionShard._match_parent_state(
        forest=forest,
        messages=replayed_messages,
        tools=[],
        chat_template_kwargs={},
    )

    assert python_arguments == '{"c": "d", "a": "b"}'
    assert forest.full_messages(response.state_hash)[-1]["tool_calls"][0]["function"]["arguments"] == (
        typescript_arguments
    )
    assert parent is response
    assert unmatched == [tool_result]


def _group_export(group_id: str, *, row_count: int = 2) -> GroupExport:
    exports = tuple(
        SampleExport(
            name=f"{group_id}/row-{row_number}",
            sample=Sample(response=f"message-{row_number}", reward=1.0, metadata={}),
        )
        for row_number in range(row_count)
    )
    return GroupExport(
        group_id=group_id,
        sessions=(SessionExport(exports=exports),),
    )


def _transfer_args(**overrides: Any) -> Namespace:
    args = {
        "fully_async": False,
        "partial_rollout": False,
        "use_dynamic_global_batch_size": False,
        "colocate": True,
        "rollout_batch_size": 8,
        "over_sampling_batch_size": 8,
        "global_batch_size": 16,
        "num_iters_per_train_update": 4,
        "n_samples_per_prompt": 2,
    }
    args.update(overrides)
    return Namespace(**args)


class _RecordingTransferQueue:
    def __init__(self) -> None:
        self.received: list[tuple[tuple[GroupExport, ...], int, bool]] = []
        self.release_writes = asyncio.Event()

    async def async_put(self, *, data: tuple[GroupExport, ...], partition_id: int, is_last: bool) -> None:
        self.received.append((data, partition_id, is_last))
        await self.release_writes.wait()


def _submit_batches(
    sink: _RecordingTransferQueue,
    *batches: TransferBatch,
) -> list["asyncio.Task[None]"]:
    return [
        asyncio.create_task(sink.async_put(data=groups, partition_id=partition_id, is_last=is_last))
        for partition_id, groups, is_last in batches
    ]


async def _wait_until(predicate) -> None:
    for _ in range(100):
        if predicate():
            return
        await asyncio.sleep(0)
    raise AssertionError("condition did not become true")


def test_session_forest_build_sample_and_session_spec() -> None:
    forest, initial_obs = _forest_with_initial_obs(
        session_id="sess-build",
        messages=[{"role": "user", "content": [{"type": "text", "text": "hello"}]}],
        train_token_delta=_chars("hello"),
        rollout_token_delta=_chars("hello"),
        rollout_id=11,
        group_index=3,
        index=7,
        label="lab",
        train_metadata={"loss": "grpo"},
        metadata={"seed_stage": "bootstrap"},
    )
    response_kwargs = {
        "parent_state_hash": initial_obs.state_hash,
        "rollout_id": 11,
        "abort_count": 0,
        "messages_delta": [{"role": "assistant", "content": [{"type": "text", "text": "ok"}]}],
        "token_delta": _chars("ok"),
        "logprob_delta": [-0.1, -0.2],
        "status": "completed",
        "export_metadata_patch": {"request_id": "req-build", "base_state_hash": initial_obs.state_hash},
    }
    leaf = forest.append_resp(**response_kwargs)
    duplicate_leaf = forest.append_resp(**response_kwargs)
    assert duplicate_leaf.state_hash == leaf.state_hash
    assert forest.export_leaf_hashes() == [leaf.state_hash]
    sample = forest.build_sample(leaf_state_hash=leaf.state_hash, tokenizer=_FakeTokenizer())
    assert (sample.prompt, sample.response, sample.group_index, sample.index) == ("hello", "ok", 3, 7)
    assert sample.train_metadata == {"loss": "grpo"}
    assert sample.metadata["agentic_trace"]["turn_count"] == 1
    sample.sampling_params = {"temperature": 0.2}
    (session_spec,) = _build_session_specs(
        [sample],
        session_ids=["sess-build"],
        include_input_payload=True,
    )
    assert (session_spec.session_id, session_spec.group_index, session_spec.index, session_spec.train_metadata) == (
        "sess-build",
        3,
        7,
        {"loss": "grpo"},
    )
    assert session_spec.sampling_params == {"temperature": 0.2}
    assert session_spec.input_payload["messages"] == [{"role": "user", "content": "hello"}]


async def test_prepare_gate_defers_unstarted_groups_and_adhoc_refills_current_gap() -> None:
    handle = SimpleNamespace(
        start_group=SimpleNamespace(remote=AsyncMock()),
        drop_group=SimpleNamespace(remote=AsyncMock()),
    )
    shard = SimpleNamespace(actor_name="shard", handle=handle, group_streams={}, next_session_number=0)
    runtime = RuntimeDomain(Namespace(rollout_global_dataset=False), "train", [shard], concurrency=2)

    async def exit_before_first_ir(_mode: str, group_id: str, _requests: Any) -> None:
        shard.group_streams[group_id].first_request_barrier.set_result(False)

    handle.start_group.remote.side_effect = exit_before_first_ir
    first = GroupInput("first", [Sample(group_index=0, index=0)])
    second = GroupInput("second", [Sample(group_index=1, index=1)])
    prepare = PrepareDomain("train", None, runtime)
    prepare.set_progress_callback(lambda: None)
    prepare.open_step(prelaunch_next_step=False)

    with (
        patch.object(runtime, "_start_shard_observers"),
        patch("relax.agentic.pipeline.runtime.logger.warning") as warning,
    ):
        prepare._start_prepare(first)
        prepare._start_prepare(second)
        prepare.close_step()
        assert list(prepare._pending_group_inputs) == [first, second]

        prepare.fill(2)
        assert not prepare._pending_group_inputs
        await asyncio.gather(*tuple(prepare._warming_prepare_tasks))
        await asyncio.sleep(0)

    assert handle.start_group.remote.await_count == 2
    handle.drop_group.remote.assert_any_await("first")
    handle.drop_group.remote.assert_any_await("second")
    assert runtime.resident_group_count == 0
    assert warning.call_count == 2
    assert not prepare._warming_prepare_tasks
    assert not prepare._ready_prepare_tasks


def test_chat_request_validation_and_logprob_payload() -> None:
    normalized = _normalized_chat_request({"messages": [{"role": "user", "content": "hello"}], "logprobs": True})
    assert normalized["logprobs"] is True

    with pytest.raises(AgenticChatRequestError, match="logprobs must be a boolean"):
        _normalized_chat_request({"messages": [{"role": "user", "content": "hello"}], "logprobs": "true"})

    payload = _openai_token_logprobs_payload(
        tokenizer=_FakeTokenizer(),
        token_ids=_chars("ok"),
        token_logprobs=[-0.1, -0.2],
    )
    assert payload["content"][1]["logprob"] == -0.2


@pytest.mark.parametrize(
    ("abort_count", "resumed", "protected_threshold", "expected_kind"),
    [
        (0, False, None, RequestKind.FRESH),
        (0, True, None, RequestKind.RESUMED),
        (1, False, None, RequestKind.RESUMED),
        (2, True, 2, RequestKind.PROTECTED),
    ],
)
def test_partial_resume_request_kind(
    abort_count: int,
    resumed: bool,
    protected_threshold: int | None,
    expected_kind: RequestKind,
) -> None:
    assert (
        SessionForest.resolve_request_kind(
            abort_count=abort_count,
            resumed=resumed,
            protected_abort_count_threshold=protected_threshold,
        )
        is expected_kind
    )


@pytest.mark.parametrize(
    ("fully_async", "final_backfill", "debt", "interrupted_count", "expected"),
    [
        (False, False, 0, 0, True),
        (False, False, 2, 2, False),
        (True, False, 2, 2, True),
        (True, False, 3, 2, False),
        (True, True, 2, 2, False),
    ],
)
def test_finish_eligibility_uses_physical_debt_and_interrupted_resident_groups(
    fully_async: bool,
    final_backfill: bool,
    debt: int,
    interrupted_count: int,
    expected: bool,
) -> None:
    pipeline = object.__new__(AgenticResidentPipeline)
    pipeline.args = SimpleNamespace(fully_async=fully_async)
    pipeline.transfer_domain = SimpleNamespace(total_debt=debt)
    pipeline.runtime_domain = SimpleNamespace(interrupted_group_ids=tuple(range(interrupted_count)))

    assert pipeline._step_can_close(_StepContext(rollout_id=3, final_backfill=final_backfill)) is expected


def test_runtime_gap_counts_current_debt_oversampling_and_resident_tail() -> None:
    pipeline = object.__new__(AgenticResidentPipeline)
    pipeline.args = SimpleNamespace(over_sampling_batch_size=6, rollout_batch_size=4)
    pipeline.transfer_domain = SimpleNamespace(total_debt=3)
    pipeline._active_group_tasks = [object()]
    pipeline._finalized_groups = deque([object()])

    assert pipeline._runtime_group_gap(_StepContext(rollout_id=3, final_backfill=False)) == 3
    assert pipeline._runtime_group_gap(_StepContext(rollout_id=4, final_backfill=True)) == 1


def test_transfer_batch_waterline_follows_fully_async_mode() -> None:
    cases = (
        (False, True, 8),
        (False, False, 8),
        (True, True, 2),
        (True, False, 2),
    )
    for fully_async, colocate, expected_group_count in cases:
        transfer = TransferDomain(args=_transfer_args(fully_async=fully_async, colocate=colocate))
        assert transfer._transfer_batch_group_count == expected_group_count


def test_dynamic_global_requires_partial_oversampling() -> None:
    invalid_overrides = (
        {"partial_rollout": False},
        {"fully_async": True},
        {"over_sampling_batch_size": 4},
    )
    for overrides in invalid_overrides:
        args = _transfer_args(
            partial_rollout=True,
            use_dynamic_global_batch_size=True,
            rollout_batch_size=4,
            over_sampling_batch_size=8,
        )
        for key, value in overrides.items():
            setattr(args, key, value)
        with pytest.raises(ValueError, match="requires partial rollout and over-sampling"):
            TransferDomain(args=args)


async def test_transfer_fifo_routes_groups_by_arrival_and_preserves_physical_rows() -> None:
    sink = _RecordingTransferQueue()
    transfer = TransferDomain(args=_transfer_args(fully_async=True, rollout_batch_size=2))
    transfer.open_partition(30, target_groups=2)
    transfer.open_partition(31, target_groups=1)

    first_group = _group_export("later-step/group-z", row_count=3)
    transfer.accept(first_group)
    tasks = _submit_batches(sink, *transfer.detach_ready_batches())
    second_group = _group_export("earlier-step/group-a")
    transfer.accept(second_group)
    tasks.extend(_submit_batches(sink, *transfer.detach_ready_batches()))
    transfer.accept(_group_export("unrelated/group-m"))
    tasks.extend(_submit_batches(sink, *transfer.detach_ready_batches()))
    assert transfer.total_debt == 0
    await _wait_until(lambda: len(sink.received) == 2)
    received_groups, partition_id, is_last = sink.received[0]
    assert received_groups == (first_group, second_group)
    assert partition_id == 30
    assert is_last
    assert [export.name for session in received_groups[0].sessions for export in session.exports] == [
        f"later-step/group-z/row-{row_number}" for row_number in range(3)
    ]

    sink.release_writes.set()
    tasks.extend(_submit_batches(sink, *transfer.flush_batches()))
    await asyncio.gather(*tasks)
    transfer.release_finished_partitions()

    assert [group.group_id for groups, _partition_id, _is_last in sink.received for group in groups] == [
        "later-step/group-z",
        "earlier-step/group-a",
        "unrelated/group-m",
    ]
    assert [is_last for _groups, _partition_id, is_last in sink.received] == [True, True]


def test_transfer_is_last_only_when_partition_target_is_met() -> None:
    transfer = TransferDomain(args=_transfer_args(fully_async=True, rollout_batch_size=2))
    transfer.open_partition(40, target_groups=4)

    transfer.accept(_group_export("group-0"))
    transfer.accept(_group_export("group-1"))
    first_batch = transfer.detach_ready_batches()
    assert len(first_batch) == 1
    assert first_batch[0][2] is False
    assert transfer.total_debt == 2

    transfer.accept(_group_export("group-2"))
    transfer.accept(_group_export("group-3"))
    second_batch = transfer.detach_ready_batches()
    assert len(second_batch) == 1
    assert second_batch[0][2] is True
    assert transfer.total_debt == 0


def test_oversampling_surplus_extends_and_seals_dynamic_partition() -> None:
    transfer = TransferDomain(
        args=_transfer_args(
            partial_rollout=True,
            use_dynamic_global_batch_size=True,
            rollout_batch_size=2,
            over_sampling_batch_size=4,
        )
    )
    transfer.open_partition(50, target_groups=2, accepts_surplus=True)
    transfer.accept(_group_export("base-0"))
    transfer.accept(_group_export("base-1"))
    base_batch = transfer.detach_ready_batches()
    assert [group.group_id for group in base_batch[0][1]] == ["base-0", "base-1"]
    assert base_batch[0][2] is False

    surplus_batch = transfer.finish_current_partition((_group_export("surplus-0"), _group_export("surplus-1")))
    assert [group.group_id for group in surplus_batch[0][1]] == ["surplus-0", "surplus-1"]
    assert transfer.total_debt == 0


def test_previous_partition_debt_is_derived_from_its_remaining_target() -> None:
    transfer = TransferDomain(args=_transfer_args(fully_async=True, rollout_batch_size=2, global_batch_size=32))
    transfer.open_partition(60, target_groups=4)
    for number in range(3):
        transfer.accept(_group_export(f"previous-{number}"))
    assert transfer.total_debt == 1

    transfer.open_partition(61, target_groups=4)
    assert transfer.total_debt == 5
    transfer.accept(_group_export("backfill"))
    assert transfer.total_debt == 4


def test_previous_partition_must_finish_before_a_third_partition_opens() -> None:
    transfer = TransferDomain(args=_transfer_args(rollout_batch_size=2, global_batch_size=32))
    transfer.open_partition(70, target_groups=1)
    transfer.open_partition(71, target_groups=1)

    with pytest.raises(RuntimeError, match="previous partition must be produced"):
        transfer.open_partition(72, target_groups=1)


async def test_transfer_preserves_original_sample_refs_and_training_fields() -> None:
    sink = _RecordingTransferQueue()
    sink.release_writes.set()
    transfer = TransferDomain(args=_transfer_args(colocate=False, rollout_batch_size=4, global_batch_size=32))
    transfer.open_partition(80, target_groups=2)
    first = _group_export("group-fields")
    sample = first.samples[0]
    sample.tokens = [11, 12, 13]
    sample.rollout_log_probs = [-0.1, -0.2]
    sample.multimodal_inputs = {"image": "raw"}
    sample.metadata = {"source": "test"}
    sample.train_metadata = {"loss": "policy"}
    sample.custom_advantage = [[0.3], [0.7, 0.9]]

    transfer.accept(first)
    transfer.accept(_group_export("group-tail"))
    await asyncio.gather(*_submit_batches(sink, *transfer.detach_ready_batches()))

    transferred = sink.received[0][0][0].samples[0]
    assert transferred is sample
    assert transferred.tokens == [11, 12, 13]
    assert transferred.rollout_log_probs == [-0.1, -0.2]
    assert transferred.multimodal_inputs == {"image": "raw"}
    assert transferred.metadata["source"] == "test"
    transfer_events = transferred.metadata["agentic_trace"]["events"]
    assert "transfer_buffer_enter_at" in transfer_events
    assert "transfer_enqueue_at" in transfer_events
    assert transferred.train_metadata == {"loss": "policy"}
    assert transferred.custom_advantage == [[0.3], [0.7, 0.9]]


def _backend_adapter(*, lifecycle_enabled: bool) -> SGLangBackendAdapter:
    adapter = object.__new__(SGLangBackendAdapter)
    adapter._args = SimpleNamespace(
        sglang_router_ip="10.0.0.1",
        sglang_router_port=8000,
        use_rollout_routing_replay=False,
        sglang_router_policy="cache_aware",
        slime_router_sticky=False,
    )
    adapter._session_lifecycle = lifecycle_enabled
    adapter.tokenizer = _FakeTokenizer()
    adapter.compiler = SimpleNamespace(processor=None)
    return adapter


async def test_generate_sends_session_id_only_when_lifecycle_enabled(monkeypatch) -> None:
    payloads: list[dict[str, Any]] = []

    async def fake_post(url, payload, headers=None):
        del url, headers
        payloads.append(dict(payload))
        return {
            "output_ids": [4],
            "meta_info": {"output_token_logprobs": [], "finish_reason": {"type": "stop"}},
        }

    monkeypatch.setattr(runtime_mod, "post", fake_post)
    adapter = _backend_adapter(lifecycle_enabled=True)
    await adapter.generate(
        input_ids=[1, 2, 3],
        sampling_params={"max_new_tokens": 4},
        session_id="session-9",
        request_id="request-1:0",
    )
    assert payloads[-1]["session_id"] == "session-9"
    assert payloads[-1]["rid"] == "request-1:0"

    adapter._session_lifecycle = False
    await adapter.generate(
        input_ids=[1, 2, 3],
        sampling_params={"max_new_tokens": 4},
        session_id="session-9",
        request_id="request-2:0",
    )
    assert "session_id" not in payloads[-1]


async def test_close_session_fans_out_and_is_fail_open(monkeypatch) -> None:
    adapter = _backend_adapter(lifecycle_enabled=True)
    adapter._worker_urls = AsyncMock(return_value=["http://engine-0", "http://engine-1"])
    posts: list[tuple[str, dict[str, Any]]] = []

    async def fake_post(url, payload):
        posts.append((url, payload))
        if url.startswith("http://engine-1"):
            raise RuntimeError("engine unavailable")

    monkeypatch.setattr(runtime_mod, "post", fake_post)
    assert not await adapter.close_session("session-1", timeout_s=1.0)
    assert sorted(posts) == [
        ("http://engine-0/close_session", {"session_id": "session-1"}),
        ("http://engine-1/close_session", {"session_id": "session-1"}),
    ]

    adapter._session_lifecycle = False
    posts.clear()
    assert await adapter.close_session("session-1", timeout_s=1.0)
    assert posts == []


async def test_terminal_session_closes_lifecycle_once() -> None:
    shard_cls = AgenticSessionShard.__ray_metadata__.modified_class
    shard = object.__new__(shard_cls)
    shard.args = SimpleNamespace(agentic_session_lifecycle=True)
    shard._generation_backend = SimpleNamespace(
        abort_request=AsyncMock(),
        close_session=AsyncMock(return_value=True),
    )
    shard._lifecycle_close_count = 0
    shard._lifecycle_close_failure_count = 0
    shard._notify_state_change = lambda group=None: None

    result_cell = _SessionResultCell()
    group = ResidentGroup(
        rollout_mode="train",
        group_id="group-1",
        result_cells={"session-1": result_cell},
        sessions=[],
    )
    session = _SessionRecord(
        group=group,
        session_id="session-1",
        session_sampling_params={},
        result_cell=result_cell,
    )
    group.sessions.append(session)
    shard._session_records = {session.session_id: session}

    assert await shard_cls._finish_session(shard, session, None) is None
    assert await shard_cls._finish_session(shard, session, None) is None
    shard._generation_backend.close_session.assert_awaited_once()
    assert shard._lifecycle_close_count == 1
    assert shard._lifecycle_close_failure_count == 0


def test_compute_reservation_tokens_uses_expanded_prompt_and_remaining_decode() -> None:
    assert compute_reservation_tokens(prompt_tokens=100, remaining_completion_tokens=50) == 150
    assert compute_reservation_tokens(prompt_tokens=-1, remaining_completion_tokens=-1) == 0


def test_budget_state_reserve_release_and_exhaust() -> None:
    budget = BudgetState(headroom=1.0, pressure_threshold=1.0, lease_ttl_s=10.0, staleness_s=30.0)
    budget.reconcile([WorkerSnapshot("engine-0", 1000, 0.1)], now=100.0)
    grant = budget.reserve(ticket_id="request-1:0", tokens=600, now=100.0)
    assert grant.granted
    assert grant.lease_id == "1:request-1:0"
    assert budget.reserve(ticket_id="request-1:0", tokens=700, now=100.0).lease_id == grant.lease_id
    assert budget.reserved == 600

    exhausted = budget.reserve(ticket_id="request-2:0", tokens=600, now=100.0)
    assert not exhausted.granted
    assert exhausted.reason is AdmissionReason.CAPACITY_EXHAUSTED
    budget.release(grant.lease_id)
    budget.release(grant.lease_id)
    assert budget.reserved == 0


def test_budget_state_pressure_ttl_and_degraded_contracts() -> None:
    budget = BudgetState(headroom=1.0, pressure_threshold=0.9, lease_ttl_s=5.0, staleness_s=10.0)
    budget.reconcile([WorkerSnapshot("engine-0", 1000, 0.95)], now=0.0)
    pressured = budget.reserve(ticket_id="pressured", tokens=10, now=0.0)
    assert not pressured.granted
    assert pressured.reason is AdmissionReason.PRESSURE_GUARD

    budget.reconcile([WorkerSnapshot("engine-0", 1000, 0.1)], now=1.0)
    grant = budget.reserve(ticket_id="expiring", tokens=100, now=1.0)
    assert grant.granted
    assert budget.expire_ttl(now=7.0) == (grant.lease_id,)
    assert budget.reserved == 0

    stale = budget.reserve(ticket_id="stale", tokens=10, now=20.0)
    assert not stale.granted
    assert stale.reason is AdmissionReason.DEGRADED


def test_parse_engine_kv_gauges_uses_max_across_tp_ranks() -> None:
    text = "\n".join(
        [
            'sglang:max_total_num_tokens{tp_rank="0"} 262144.0',
            'sglang:max_total_num_tokens{tp_rank="1"} 262144.0',
            'sglang:num_used_tokens{tp_rank="0"} 239901.0',
            'sglang:num_used_tokens{tp_rank="1"} 239901.0',
            'sglang:token_usage{tp_rank="0"} 0.915',
            'sglang:token_usage{tp_rank="1"} 0.915',
        ]
    )
    gauges = _parse_engine_kv_gauges(text)
    assert gauges["sglang:max_total_num_tokens"] == 262144.0
    assert gauges["sglang:num_used_tokens"] == 239901.0
    assert gauges["sglang:token_usage"] == 0.915


def test_budget_state_usage_window_tracks_peak_and_resets() -> None:
    budget = BudgetState(headroom=1.0, pressure_threshold=1.0, staleness_s=100.0)
    budget.reconcile([WorkerSnapshot("engine-0", 1000, 0.3)], now=0.0)
    budget.reconcile([WorkerSnapshot("engine-0", 1000, 0.9)], now=1.0)
    budget.reconcile([WorkerSnapshot("engine-0", 1000, 0.05)], now=2.0)

    snapshot = budget.snapshot(now=2.0, reset_usage_window=True)
    assert snapshot["kv_usage_max"] == 0.9
    assert snapshot["kv_usage_mean"] == pytest.approx((0.3 + 0.9 + 0.05) / 3)

    budget.reconcile([WorkerSnapshot("engine-0", 1000, 0.1)], now=3.0)
    next_snapshot = budget.snapshot(now=3.0)
    assert next_snapshot["kv_usage_max"] == 0.1
    assert next_snapshot["kv_usage_mean"] == 0.1


def _admission_coordinator(*, max_wait_s: float = 30.0, capacity: int = 100):
    coordinator_cls = AdmissionCoordinator.__ray_metadata__.modified_class
    coordinator = object.__new__(coordinator_cls)
    coordinator._max_wait_s = max_wait_s
    coordinator._state = BudgetState(headroom=1.0, pressure_threshold=1.0, staleness_s=100.0)
    coordinator._state.reconcile([WorkerSnapshot("engine-0", capacity, 0.0)], now=time.monotonic())
    coordinator._waiters = {}
    coordinator._waiter_order = deque()
    coordinator._cancelled_tickets = {}
    coordinator._counters = {}
    return coordinator_cls, coordinator


async def test_admission_coordinator_bypasses_protected_and_degraded_requests() -> None:
    coordinator_cls, coordinator = _admission_coordinator()
    protected = await coordinator_cls.acquire(
        coordinator,
        {"ticket_id": "protected", "reservation_tokens": 1000, "protected": True},
    )
    assert protected["action"] == AdmissionAction.BYPASS.value
    assert protected["reason"] == AdmissionReason.PROTECTED.value

    coordinator._state.invalidate()
    degraded = await coordinator_cls.acquire(
        coordinator,
        {"ticket_id": "degraded", "reservation_tokens": 1000},
    )
    assert degraded["action"] == AdmissionAction.BYPASS.value
    assert degraded["reason"] == AdmissionReason.DEGRADED.value


async def test_admission_coordinator_grants_waiters_in_global_fifo_order() -> None:
    coordinator_cls, coordinator = _admission_coordinator(capacity=100)
    first = await coordinator_cls.acquire(coordinator, {"ticket_id": "first", "reservation_tokens": 100})
    second_task = asyncio.create_task(
        coordinator_cls.acquire(coordinator, {"ticket_id": "second", "reservation_tokens": 70})
    )
    third_task = asyncio.create_task(
        coordinator_cls.acquire(coordinator, {"ticket_id": "third", "reservation_tokens": 40})
    )
    await _wait_until(lambda: len(coordinator._waiters) == 2)
    assert list(coordinator._waiter_order) == ["second", "third"]

    await coordinator_cls.release(coordinator, first["lease_id"])
    second = await second_task
    assert second["action"] == AdmissionAction.ADMIT.value
    assert not third_task.done()

    await coordinator_cls.release(coordinator, second["lease_id"])
    third = await third_task
    assert third["action"] == AdmissionAction.ADMIT.value


async def test_admission_coordinator_cancellation_removes_waiter_or_granted_lease() -> None:
    coordinator_cls, coordinator = _admission_coordinator(capacity=100)
    owner = await coordinator_cls.acquire(coordinator, {"ticket_id": "owner", "reservation_tokens": 100})
    waiter_task = asyncio.create_task(
        coordinator_cls.acquire(coordinator, {"ticket_id": "waiter", "reservation_tokens": 100})
    )
    await asyncio.sleep(0)
    await coordinator_cls.cancel(coordinator, "waiter")
    cancelled = await waiter_task
    assert cancelled["reason"] == AdmissionReason.CANCELLED.value
    assert "waiter" not in coordinator._waiters

    await coordinator_cls.release(coordinator, owner["lease_id"])
    await coordinator_cls.acquire(coordinator, {"ticket_id": "granted", "reservation_tokens": 100})
    await coordinator_cls.cancel(coordinator, "granted")
    assert coordinator._state.lease_for_ticket("granted") is None
    assert coordinator._state.reserved == 0

    await coordinator_cls.cancel(coordinator, "cancelled-before-acquire")
    cancelled_before_acquire = await coordinator_cls.acquire(
        coordinator,
        {"ticket_id": "cancelled-before-acquire", "reservation_tokens": 1},
    )
    assert cancelled_before_acquire["reason"] == AdmissionReason.CANCELLED.value


async def test_admission_coordinator_ages_waiters_without_leasing_capacity() -> None:
    coordinator_cls, coordinator = _admission_coordinator(max_wait_s=0.0, capacity=100)
    owner = await coordinator_cls.acquire(coordinator, {"ticket_id": "owner", "reservation_tokens": 100})
    aged = await coordinator_cls.acquire(coordinator, {"ticket_id": "aged", "reservation_tokens": 100})
    assert aged["action"] == AdmissionAction.BYPASS.value
    assert aged["reason"] == AdmissionReason.AGED.value
    assert coordinator._state.lease_for_ticket("aged") is None
    await coordinator_cls.release(coordinator, owner["lease_id"])


async def test_admission_lease_uses_train_prefix_and_releases_after_backend_attempt() -> None:
    class RecordingAdmissionClient:
        def __init__(self) -> None:
            self.requests: list[dict[str, Any]] = []
            self.releases: list[str] = []

        async def acquire(self, request):
            self.requests.append(request)
            return {"lease_id": "lease-1"}

        async def release(self, lease_id):
            self.releases.append(lease_id)

    shard_cls = AgenticSessionShard.__ray_metadata__.modified_class
    shard = object.__new__(shard_cls)
    client = RecordingAdmissionClient()
    shard._admission_client = client
    shard.args = SimpleNamespace(
        agentic_admission_scope="all",
        agentic_admission_expected_decode_cap=20,
    )
    group = SimpleNamespace(rollout_mode="train")
    session = SimpleNamespace(group=group, protected_until_finalize=False)
    ir = InflightRequest(
        request_id="request-1",
        parent_state_hash="parent",
        rollout_id=0,
        kind=RequestKind.FRESH,
        abort_count=2,
        waiter=asyncio.get_running_loop().create_future(),
        wall_started_at=time.monotonic(),
        sampling_params={"max_new_tokens": 64},
        history_train_token_prefix=list(range(100)),
        pending_token_delta=list(range(8)),
    )

    with pytest.raises(RuntimeError, match="backend failure"):
        async with shard_cls._admission_lease(shard, session, ir):
            assert client.releases == []
            raise RuntimeError("backend failure")

    assert client.requests == [
        {
            "ticket_id": "request-1:2",
            "reservation_tokens": 128,
            "protected": False,
        }
    ]
    assert client.releases == ["lease-1"]
