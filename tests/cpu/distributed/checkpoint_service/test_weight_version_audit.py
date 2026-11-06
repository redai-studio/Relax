# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Exercise the production DCS update and publication methods with CPU I/O
doubles."""

from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from relax.distributed.checkpoint_service.backends import device_direct as dcs
from relax.utils import weight_version_audit as audit


@pytest.fixture
def backend(tmp_path, monkeypatch):
    monkeypatch.setenv("RELAX_WEIGHT_AUDIT_DIR", str(tmp_path))
    monkeypatch.setattr(dcs.dist, "get_rank", lambda **_kwargs: 0)
    monkeypatch.setattr(dcs.dist, "barrier", lambda **_kwargs: None)
    monkeypatch.setattr(dcs.dist, "all_reduce", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(dcs, "get_gloo_group", lambda: object())
    monkeypatch.setattr(dcs.ray, "get", lambda value: value)
    monkeypatch.setattr(dcs.device_module, "empty_cache", lambda: None)
    monkeypatch.setattr(dcs, "tqdm", lambda **_kwargs: SimpleNamespace(update=lambda *_args: None))
    value = object.__new__(dcs.DeviceDirectBackend)
    value.args = SimpleNamespace(num_experts=0)
    value.model_name = "policy-main"
    value.model = []
    value.weight_version = 0
    value._use_bridge = value._lora_merge_mode = value._lora_adapter_mode = False
    value._lora_sync = SimpleNamespace(base_sync_done=True, adapter_loaded=True)
    value._is_pp_src_rank = True
    value._group_name = "slime-pp_0"
    value.rollout_engines = {}
    value._megatron = SimpleNamespace(named_params_and_buffers=lambda *_args, **_kwargs: [])
    value._run_expert_pass = lambda **_kwargs: None
    value._push_lora_adapter_distributed = lambda **_kwargs: None
    value.http_client = SimpleNamespace(post=lambda *_args, **_kwargs: SimpleNamespace(raise_for_status=lambda: None))
    value.coordinator_url = "http://coordinator"
    value._weight_audit_publisher = audit.PublicationLedger(str(tmp_path), "policy-main")
    events = []

    def request(endpoint, payload=None):
        events.append((endpoint, payload))
        return [{"success": True}]

    value._batch_request = request
    return value, events


def test_success_publishes_after_loads_before_resume(backend):
    value, events = backend
    value._run_expert_pass = lambda **_kwargs: events.append(("weights-loaded", None))
    value.update_weights_for_rollout()
    endpoints = [event[0] for event in events]
    assert endpoints.index("weights-loaded") < endpoints.index("/update_weight_version")
    assert endpoints.index("/update_weight_version") < endpoints.index("/continue_generation")
    assert events[0] == ("/pause_generation", {"mode": "abort"})
    assert audit.read_frontier()["published_seq"] == 1


def test_actor_fwd_only_does_not_advance_policy_publication(backend):
    value, events = backend
    value.update_weights_for_rollout(actor_fwd_only=True)
    assert value.weight_version == 1
    assert value._weight_audit_publisher.published_seq == 0
    assert not any(endpoint == "/update_weight_version" for endpoint, _ in events)
    assert audit.read_frontier() is None


def test_load_failure_preserves_original_exception_without_publication(backend):
    value, _ = backend
    error = RuntimeError("load failed")

    def fail(**_kwargs):
        raise error

    value._run_expert_pass = fail
    with pytest.raises(RuntimeError) as caught:
        value.update_weights_for_rollout()
    assert caught.value is error
    assert audit.read_frontier() is None


def test_failure_on_other_pp_source_prevents_confirmation(backend, monkeypatch):
    value, events = backend
    value._weight_audit_load_ok = True
    verdicts = iter([1, 0])
    monkeypatch.setattr(dcs.dist, "all_reduce", lambda flag, **_kwargs: flag.fill_(next(verdicts)))
    value._publish_weight_audit_version(push_failed=False)
    assert value._weight_audit_publisher.published_seq == 0
    assert audit.read_frontier() is None
    assert events == []


@pytest.mark.parametrize("failure", ["tag_response", "tag_exception", "frontier_write", "invalidation_write"])
def test_failed_publication_marks_old_tags_unknown_before_resume(backend, monkeypatch, failure):
    value, events = backend
    old_tag = value._weight_audit_publisher.publish()
    original_request = value._batch_request
    original_write = audit._atomic_json

    def request(endpoint, payload=None):
        if endpoint == "/update_weight_version":
            if failure == "tag_exception":
                raise OSError("tag RPC failed")
            return [{"success": True}, {"success": False}]
        if endpoint == "/continue_generation":
            assert audit.read_frontier() is None
        return original_request(endpoint, payload)

    def write(path, data):
        if failure == "frontier_write" and "published_seq" in data:
            raise OSError("frontier write failed")
        if failure == "invalidation_write" and data.get("evidence_valid") is False:
            raise OSError("invalidation write failed")
        return original_write(path, data)

    with monkeypatch.context() as patch:
        patch.setattr(value, "_batch_request", request)
        patch.setattr(audit, "_atomic_json", write)
        value.update_weights_for_rollout()
        assert audit.read_frontier() is None
        spans = audit.generation_spans(
            {"weight_versions": [{"start": 0, "end": 1, "version": old_tag}]},
            1,
            offset=0,
            request_id="request",
            attempt_id="attempt",
            piece_index=0,
        )
        assert spans[0]["published_seq"] is None
        assert spans[0]["unknown_reason"] == "publication_frontier_unavailable"
    assert any(endpoint == "/continue_generation" for endpoint, _ in events)
    value.update_weights_for_rollout()
    assert audit.read_frontier()["published_seq"] == 3
    late = audit.generation_spans(
        {"weight_version": old_tag, "_weight_audit_scalar_bound": True},
        1,
        offset=0,
        request_id="late",
        attempt_id="late:0",
        piece_index=0,
    )
    assert late[0]["published_seq"] is None
    assert late[0]["unknown_reason"] == "publication_scope_mismatch"


def test_http_200_failed_bucket_is_not_a_successful_publication(backend, monkeypatch):
    value, _ = backend
    value._weight_audit_load_ok = True
    value._model_update_groups = object()
    value.lock = SimpleNamespace(
        acquire=SimpleNamespace(remote=lambda: True), release=SimpleNamespace(remote=lambda: True)
    )
    value._batch_request = lambda *_args: [{"success": False}]
    monkeypatch.setattr(dcs.dist, "broadcast", lambda *_args, **_kwargs: SimpleNamespace(wait=lambda: None))
    value._update_bucket_weights_from_distributed([("weight", torch.ones(1))])
    assert value._weight_audit_load_ok is False
    value._publish_weight_audit_version(push_failed=False)
    assert audit.read_frontier() is None


def test_adapter_invalidates_old_tag_and_publishes_after_adapter_load(backend):
    value, events = backend
    value._lora_adapter_mode = True
    value._push_lora_adapter_distributed = lambda **_kwargs: events.append(("adapter-loaded", None))
    value.update_weights_for_rollout()
    endpoints = [event[0] for event in events]
    tag_positions = [i for i, event in enumerate(events) if event[0] == "/update_weight_version"]
    assert len(tag_positions) == 2
    assert tag_positions[0] < endpoints.index("adapter-loaded") < tag_positions[1]
    assert events[tag_positions[0]][1]["new_version"].startswith("relax-audit-pending:")
    assert events[tag_positions[1]][1]["new_version"].endswith(":1")


def test_audit_io_failure_does_not_strand_normal_adapter_update(backend, tmp_path):
    value, events = backend
    value._lora_adapter_mode = True
    blocker = tmp_path / "file"
    blocker.write_text("not a directory")
    value._weight_audit_publisher.directory = blocker
    original_request = value._batch_request

    def failed_metadata(endpoint, payload=None):
        if endpoint == "/update_weight_version":
            raise OSError("audit metadata unavailable")
        return original_request(endpoint, payload)

    value._batch_request = failed_metadata
    value.update_weights_for_rollout()
    assert any(endpoint == "/continue_generation" for endpoint, _ in events)


def _gloo_publication_worker(rank, rendezvous, directory, failed_rank):
    torch.set_num_threads(1)
    dcs.dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2, timeout=timedelta(seconds=20)
    )
    try:
        dcs.get_gloo_group = lambda: dcs.dist.group.WORLD
        dcs.ray.get = lambda value: value
        value = object.__new__(dcs.DeviceDirectBackend)
        value._weight_audit_load_ok = rank != failed_rank
        value._batch_request = lambda *_args: [{"success": True}]
        if rank == 0:
            value._weight_audit_publisher = audit.PublicationLedger(directory, "policy")
        if failed_rank == -2:
            if rank == 0:
                value._weight_audit_publisher.publish()

            def fail_io(*_args, **_kwargs):
                raise PermissionError("frontier cannot be written or removed")

            with pytest.MonkeyPatch.context() as patch:
                patch.setattr(audit, "_atomic_json", fail_io)
                patch.setattr(Path, "unlink", fail_io)
                with pytest.raises(RuntimeError, match="generation remains paused"):
                    value._publish_weight_audit_version(push_failed=False)
        else:
            value._publish_weight_audit_version(push_failed=False)
        dcs.dist.barrier(group=dcs.dist.group.WORLD)
    finally:
        dcs.dist.destroy_process_group()


@pytest.mark.parametrize("failed_rank", [-1, 1, -2])
def test_real_gloo_collective_confirms_all_sources(tmp_path, monkeypatch, failed_rank):
    monkeypatch.setenv("RELAX_WEIGHT_AUDIT_DIR", str(tmp_path))
    torch.multiprocessing.spawn(
        _gloo_publication_worker, args=(str(tmp_path / "rendezvous"), str(tmp_path), failed_rank), nprocs=2, join=True
    )
    frontier = audit.read_frontier()
    assert (frontier is not None) == (failed_rank in (-1, -2))
    if frontier is not None:
        assert frontier["published_seq"] == 1


def test_late_adapter_response_stays_unknown_after_recovery(backend):
    import asyncio

    from relax.agentic.pipeline.runtime import BackendGenerateResult
    from tests.cpu.test_agentic_weight_version_audit import _ir, _session

    value, _ = backend
    value._lora_adapter_mode = True
    old_tag = value._weight_audit_publisher.publish()
    engine_tag = old_tag
    pending_fails = True
    actual_policy = 1
    original_request = value._batch_request

    def request(endpoint, payload=None):
        nonlocal engine_tag
        if endpoint == "/update_weight_version":
            if pending_fails and payload["new_version"].startswith("relax-audit-pending:"):
                return [{"success": False}]
            engine_tag = payload["new_version"]
            return [{"success": True}]
        return original_request(endpoint, payload)

    def push(**_kwargs):
        nonlocal actual_policy
        actual_policy += 1

    value._batch_request = request
    value._push_lora_adapter_distributed = push
    value.update_weights_for_rollout()
    assert actual_policy == 2 and engine_tag == old_tag
    assert audit.read_frontier() is None
    delayed = BackendGenerateResult(
        new_tokens=[97],
        new_log_probs=[-0.5],
        finish_type="stop",
        meta_info={
            "prompt_tokens": 4,
            "finish_reason": {"type": "stop"},
            "weight_version": engine_tag,
            "_weight_audit_scalar_bound": True,
        },
        elapsed=0.1,
    )
    pending_fails = False
    value.update_weights_for_rollout()
    assert actual_policy == 3

    async def export():
        shard, session, forest, observation = _session()
        ir = _ir(session, observation.state_hash, "late-response")
        shard._apply_generate_result(ir, delayed)
        ir.pending_status = "completed"
        shard._terminal_response_locked(session=session, ir=ir, finish_type="stop")
        return forest.build_sample(
            leaf_state_hash=session.resp_state_hash_by_request_id[ir.request_id],
            tokenizer=shard._generation_backend.tokenizer,
        )

    sample = asyncio.run(export())
    span = sample.metadata[audit.AUDIT_KEY]["spans"][0]
    assert span["published_seq"] is None
    assert span["unknown_reason"] == "publication_scope_mismatch"
