# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Actual Session commit/export and numeric training serialization on CPU."""

import asyncio
import copy
import pickle
from types import SimpleNamespace

import pytest

from relax.agentic.pipeline import TrainingFieldArtifact
from relax.agentic.pipeline.runtime import BackendGenerateResult
from relax.agentic.session.state import InflightRequest, RequestKind
from relax.utils import weight_version_audit as audit
from relax.utils.utils import convert_samples_to_train_data
from tests.cpu.test_agentic_rollout import _FakeTokenizer, _forest_with_initial_obs, _router_recovery_shard


def _ir(session, parent, request_id):
    import asyncio

    return InflightRequest(
        request_id=request_id,
        parent_state_hash=parent,
        rollout_id=0,
        kind=RequestKind.FRESH,
        abort_count=0,
        waiter=asyncio.get_running_loop().create_future(),
        wall_started_at=0,
        sampling_params={"max_new_tokens": 10},
    )


def _apply(shard, ir, text, version, finish="stop"):
    shard._apply_generate_result(
        ir,
        BackendGenerateResult(
            new_tokens=[ord(c) for c in text],
            new_log_probs=[-0.5] * len(text),
            finish_type=finish,
            meta_info={
                "prompt_tokens": 4,
                "finish_reason": {"type": finish},
                "weight_version": version,
                "weight_versions": [{"version": version, "start": 0, "end": len(text)}] if text else [],
            },
            elapsed=0.1,
        ),
    )


@pytest.fixture
def publisher(tmp_path, monkeypatch):
    monkeypatch.setenv("RELAX_WEIGHT_AUDIT_DIR", str(tmp_path))
    ledger = audit.PublicationLedger(str(tmp_path), "policy-main")
    versions = [ledger.publish() for _ in range(10)]
    return ledger, versions[-1]


def _session():
    shard, group, session, _ = _router_recovery_shard()
    shard.args.agentic_reasoning_parser = None
    shard.args.agentic_tool_call_parser = None
    shard._generation_backend = SimpleNamespace(tokenizer=_FakeTokenizer())
    forest, observation = _forest_with_initial_obs(
        session_id=session.session_id,
        messages=[{"role": "user", "content": "pppp"}],
        train_token_delta=[112] * 4,
        rollout_token_delta=[112] * 4,
        index=0,
        group_index=0,
    )
    session.forest = forest
    return shard, session, forest, observation


async def _two_turn_sample(publisher):
    ledger, first_version = publisher
    shard, session, forest, observation = _session()
    first = _ir(session, observation.state_hash, "request-0")
    _apply(shard, first, "abc", first_version)
    first.pending_status = "completed"
    shard._terminal_response_locked(session=session, ir=first, finish_type="stop")
    first_hash = session.resp_state_hash_by_request_id[first.request_id]
    ledger.publish()
    second_version = ledger.publish()
    tool = forest.append_obs(
        parent_state_hash=first_hash,
        rollout_id=0,
        abort_count=0,
        messages_delta=[{"role": "tool", "tool_call_id": "tool-0", "content": "ok"}],
        train_token_delta=[111, 107],
        rollout_token_delta=[111, 107],
    )
    second = _ir(session, tool.state_hash, "request-1")
    _apply(shard, second, "de", second_version)
    second.pending_status = "completed"
    shard._terminal_response_locked(session=session, ir=second, finish_type="stop")
    sample = forest.build_sample(
        leaf_state_hash=session.resp_state_hash_by_request_id[second.request_id],
        tokenizer=_FakeTokenizer(),
    )
    sample.reward = 1.0
    return sample


def _converter_args():
    return SimpleNamespace(
        custom_reward_post_process_path=None,
        reward_key=None,
        rewards_normalization=False,
        advantage_estimator="grpo",
        use_opd=False,
        multimodal_keys=None,
        debug_train_only=False,
    )


async def test_real_session_to_training_row_round_trip(publisher, monkeypatch):
    sample = await _two_turn_sample(publisher)
    payload = sample.metadata[audit.AUDIT_KEY]
    assert sample.loss_mask == [1, 1, 1, 0, 0, 1, 1]
    assert [(s["token_start"], s["token_end"], s["published_seq"], s["turn_index"]) for s in payload["spans"]] == [
        (4, 7, 10, 0),
        (9, 11, 12, 1),
    ]
    restored = TrainingFieldArtifact.from_sample(sample).to_sample()
    assert restored.metadata[audit.AUDIT_KEY] == payload
    args = _converter_args()
    enabled_data = convert_samples_to_train_data(args, [copy.deepcopy(restored)])
    monkeypatch.delenv("RELAX_WEIGHT_AUDIT_DIR")
    disabled_data = convert_samples_to_train_data(args, [copy.deepcopy(restored)])
    assert set(enabled_data.keys()) - set(disabled_data.keys()) == {audit.AUDIT_FIELD}
    for key in disabled_data.keys():
        assert [v.tolist() for v in enabled_data[key].unbind()] == [v.tolist() for v in disabled_data[key].unbind()]
    monkeypatch.setenv("RELAX_WEIGHT_AUDIT_DIR", str(publisher[0].directory))
    transported = dict(pickle.loads(pickle.dumps(enabled_data)).items())
    audit.prepare_batch(transported)
    publisher[0].publish()
    report = audit.audit_row(transported[audit.PREPARED_FIELD][0], audit.read_frontier())
    assert report["eligible_tokens"] == 5
    assert report["token_weighted_age"] == 2.2
    assert report["observed_generation_span"] == 2
    assert report["post_generation_published_advance"] == 1


async def test_transfer_queue_codec_and_storage_preserve_reordered_audit_rows(publisher):
    from transfer_queue.storage.simple_storage import StorageUnitData
    from transfer_queue.utils.zmq_utils import ZMQMessage, ZMQRequestType

    known = await _two_turn_sample(publisher)
    unknown = copy.deepcopy(known)
    unknown.tokens = unknown.tokens[:-1]
    unknown.response_length -= 1
    unknown.loss_mask = unknown.loss_mask[:-1]
    unknown.rollout_log_probs = unknown.rollout_log_probs[:-1]
    unknown.metadata[audit.AUDIT_KEY] = {"session_id": "缺失版本", "spans": []}
    removed = copy.deepcopy(known)
    removed.remove_sample = True
    removed.metadata[audit.AUDIT_KEY]["session_id"] = "已过滤"
    data = convert_samples_to_train_data(_converter_args(), [known, unknown, removed])
    original_payloads = [row.tolist() for row in data[audit.AUDIT_FIELD]]

    def round_trip(request_type, value):
        message = ZMQMessage.create(request_type, "audit-test", {"data": value})
        frames = [bytes(frame) for frame in message.serialize()]
        assert frames[0] != b"\xc1\xfe\xed" and len(frames) > 1
        return ZMQMessage.deserialize(frames).body["data"]

    received = round_trip(ZMQRequestType.PUT_DATA, data)
    storage = StorageUnitData(storage_size=3)
    storage.put_data(dict(received.items()), [7, 13, 19])
    fetched = storage.get_data(list(data.keys()), [19, 13, 7])
    transported = round_trip(ZMQRequestType.GET_DATA_RESPONSE, fetched)
    assert [row.tolist() for row in transported[audit.AUDIT_FIELD]] == original_payloads[::-1]
    assert [row.tolist() for row in transported["tokens"]] == [removed.tokens, unknown.tokens, known.tokens]
    assert [row.tolist() for row in transported["loss_masks"]] == [
        [0] * removed.response_length,
        unknown.loss_mask,
        known.loss_mask,
    ]
    audit.prepare_batch(transported)
    reports = [audit.audit_row(row, audit.read_frontier()) for row in transported[audit.PREPARED_FIELD]]
    assert [row["physical_row_index"] for row in transported[audit.PREPARED_FIELD]] == [2, 1, 0]
    assert [report["eligible_tokens"] for report in reports] == [0, 4, 5]
    assert [report["missing_token_rate"] for report in reports] == [None, 1, 0]
    assert reports[0]["session_id"] == "已过滤" and reports[1]["session_id"] == "缺失版本"


async def test_partial_attempts_preserve_only_new_tokens(publisher):
    ledger, version = publisher
    shard, session, forest, observation = _session()
    ir = _ir(session, observation.state_hash, "resume")
    _apply(shard, ir, "ab", version, finish="abort")
    ir.abort_count += 1
    ledger.publish()
    new_version = ledger.publish()
    _apply(shard, ir, "c", new_version)
    ir.pending_status = "completed"
    shard._terminal_response_locked(session=session, ir=ir, finish_type="stop")
    sample = forest.build_sample(
        leaf_state_hash=session.resp_state_hash_by_request_id[ir.request_id], tokenizer=_FakeTokenizer()
    )
    spans = sample.metadata[audit.AUDIT_KEY]["spans"]
    assert sample.tokens == [112] * 4 + [97, 98, 99]
    assert [(s["token_start"], s["token_end"], s["published_seq"]) for s in spans] == [(4, 6, 10), (6, 7, 12)]
    assert [s["attempt_id"] for s in spans] == ["resume:0", "resume:1"]


async def test_cancelled_uncommitted_response_does_not_enter_export(publisher):
    _, version = publisher
    shard, session, forest, observation = _session()
    cancelled = _ir(session, observation.state_hash, "cancelled")
    _apply(shard, cancelled, "discard", version)
    replacement = _ir(session, observation.state_hash, "replacement")
    _apply(shard, replacement, "ok", version)
    replacement.pending_status = "completed"
    shard._terminal_response_locked(session=session, ir=replacement, finish_type="stop")
    sample = forest.build_sample(
        leaf_state_hash=session.resp_state_hash_by_request_id[replacement.request_id], tokenizer=_FakeTokenizer()
    )
    assert sample.response == "ok"
    assert {s["request_id"] for s in sample.metadata[audit.AUDIT_KEY]["spans"]} == {"replacement"}
    assert "cancelled" not in session.resp_state_hash_by_request_id


async def test_reused_node_keeps_its_original_provenance(publisher):
    ledger, version = publisher
    shard, session, forest, observation = _session()
    first = _ir(session, observation.state_hash, "original")
    _apply(shard, first, "same", version)
    first.pending_status = "completed"
    shard._terminal_response_locked(session=session, ir=first, finish_type="stop")
    later_version = ledger.publish()
    later = _ir(session, observation.state_hash, "later")
    _apply(shard, later, "same", later_version)
    later.pending_status = "completed"
    shard._terminal_response_locked(session=session, ir=later, finish_type="stop")
    assert (
        session.resp_state_hash_by_request_id[first.request_id]
        == session.resp_state_hash_by_request_id[later.request_id]
    )
    sample = forest.build_sample(
        leaf_state_hash=session.resp_state_hash_by_request_id[later.request_id], tokenizer=_FakeTokenizer()
    )
    assert sample.metadata[audit.AUDIT_KEY]["spans"][0]["request_id"] == "original"
    assert sample.metadata[audit.AUDIT_KEY]["spans"][0]["published_seq"] == 10


def test_bad_numeric_payload_is_diagnosed_without_changing_training_fields():
    data = {
        audit.AUDIT_FIELD: [[255]],
        "tokens": [[1, 2]],
        "total_lengths": [2],
        "response_lengths": [1],
        "loss_masks": [[1]],
    }
    audit.prepare_batch(data)
    assert data["tokens"] == [[1, 2]] and data["loss_masks"] == [[1]]
    report = audit.audit_row(data[audit.PREPARED_FIELD][0], None)
    assert report["missing_token_rate"] == 1
    assert report["audit_error"] == "invalid_provenance_payload"


async def test_late_result_of_replaced_runner_cannot_add_provenance(publisher):
    shard, _, session, request = _router_recovery_shard()
    started, release = asyncio.Event(), asyncio.Event()

    async def generate(**_kwargs):
        started.set()
        await release.wait()
        return BackendGenerateResult(
            new_tokens=[97],
            new_log_probs=[-0.5],
            finish_type="stop",
            meta_info={"weight_versions": [{"start": 0, "end": 1, "version": publisher[1]}]},
            elapsed=0.1,
        )

    shard._generation_backend = SimpleNamespace(generate=generate, tokenizer=_FakeTokenizer())
    task = asyncio.create_task(shard._run_ir(session, request))
    request.runner_task = task
    await asyncio.wait_for(started.wait(), timeout=2)
    async with session.lock:
        session.live_irs.remove(request)
    release.set()
    await asyncio.wait_for(task, timeout=2)
    assert request.pending_token_delta == [300]
    assert audit.AUDIT_KEY not in request.pending_export_metadata_patch


def test_unavailable_audit_transport_preserves_original_valid_mask_counts():
    data = {
        audit.AUDIT_FIELD: [],
        "tokens": [[1, 2, 3]],
        "total_lengths": [3],
        "response_lengths": [2],
        "loss_masks": [[1, 1]],
    }
    rows = audit.unavailable_rows(data)
    report = audit.audit_row(rows[0], None)
    assert report["eligible_tokens"] == 2
    assert report["missing_token_rate"] == 1
    assert report["audit_errors"] == 1
    assert audit.aggregate([report])["aggregate_complete"] is False


def test_removed_sample_does_not_affect_consumption_wait(publisher):
    async def run():
        removed = await _two_turn_sample(publisher)
        retained = await _two_turn_sample(publisher)
        removed.remove_sample = True
        publisher[0].publish()
        data = convert_samples_to_train_data(_converter_args(), [removed, retained])
        data["total_lengths"] = [len(tokens) for tokens in data["tokens"]]
        audit.prepare_batch(data)
        step = audit.AuditStep("filtered-sample")
        step.observe(0, data[audit.PREPARED_FIELD])
        summary = step.commit(0)
        assert summary["consumed_rows"] == 1
        assert summary["post_generation_advance_count"] == 1
        assert summary["post_generation_advance_mean"] == 1
        assert summary["post_generation_advance_max"] == 1

    asyncio.run(run())
