# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Exercise the production DCS update and publication methods with CPU I/O
doubles."""

from datetime import timedelta
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
    monkeypatch.setattr(dcs.dist, "all_reduce", lambda flag, **_kwargs: flag.fill_(1))
    value._publish_weight_audit_version(push_failed=False)
    assert value._weight_audit_publisher.published_seq == 0
    assert audit.read_frontier() is None
    assert events == []


def test_terminal_tag_failure_does_not_reuse_a_successful_policy_sequence(backend):
    value, _ = backend
    value._weight_audit_load_ok = True
    value._batch_request = lambda *_args: [{"success": False}]
    value._publish_weight_audit_version(push_failed=False)
    assert audit.read_frontier()["published_seq"] == 1
    tags = []
    value._batch_request = lambda _, payload: tags.append(payload["new_version"]) or [{"success": True}]
    value._publish_weight_audit_version(push_failed=False)
    assert tags[0].endswith(":2")
    assert audit.read_frontier()["published_seq"] == 2


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
        value._publish_weight_audit_version(push_failed=False)
        dcs.dist.barrier()
    finally:
        dcs.dist.destroy_process_group()


@pytest.mark.parametrize("failed_rank", [-1, 1])
def test_real_gloo_collective_confirms_all_sources(tmp_path, monkeypatch, failed_rank):
    monkeypatch.setenv("RELAX_WEIGHT_AUDIT_DIR", str(tmp_path))
    torch.multiprocessing.spawn(
        _gloo_publication_worker, args=(str(tmp_path / "rendezvous"), str(tmp_path), failed_rank), nprocs=2, join=True
    )
    frontier = audit.read_frontier()
    assert (frontier is not None) == (failed_rank == -1)
    if frontier is not None:
        assert frontier["published_seq"] == 1
