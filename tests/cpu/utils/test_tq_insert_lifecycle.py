# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Exercise the pinned TransferQueue metadata implementation without
storage."""

import pytest


@pytest.fixture
def controller_factory(monkeypatch):
    module = pytest.importorskip("transfer_queue.controller")
    controller_type = module.TransferQueueController.__ray_metadata__.modified_class
    # Only isolate network setup: allocation, completion, clear and checkpoint
    # still execute the dependency's actual implementation.
    for method in ("_init_zmq_socket", "_start_process_handshake", "_start_process_request"):
        monkeypatch.setattr(controller_type, method, lambda _self: None)
    monkeypatch.setenv("TQ_PRE_ALLOC_SAMPLE_NUM", "1")
    return lambda: controller_type(polling_mode=True)


def _insert(controller, request_id, *, count=1, last=False):
    return controller.get_metadata(
        data_fields=["tokens"],
        batch_size=count,
        partition_id="train",
        mode="insert",
        is_last=last,
        insert_request_id=request_id,
    )


def test_insert_retry_returns_original_allocation(controller_factory):
    controller = controller_factory()
    first = _insert(controller, "lost-reply")
    retry = _insert(controller, "lost-reply")
    assert retry.global_indexes == first.global_indexes
    assert controller.partitions["train"].actual_sample_count == 1

    # A caller must not mutate the saved response used by subsequent retries.
    retry.global_indexes.append(123456)
    assert _insert(controller, "lost-reply").global_indexes == first.global_indexes
    with pytest.raises(RuntimeError, match="different arguments"):
        _insert(controller, "lost-reply", count=2)
    assert controller.partitions["train"].actual_sample_count == 1


def test_completed_partition_retains_terminal_state_and_rejects_old_insert(controller_factory, tmp_path):
    import torch

    controller = controller_factory()
    metadata = _insert(controller, "completed-insert", last=True)
    assert controller.update_production_status(
        "train",
        metadata.global_indexes,
        {"tokens": {"dtype": torch.float32, "shape": (1,), "is_nested": False, "is_non_tensor": False}},
    )
    controller.get_metadata(["tokens"], batch_size=1, partition_id="train", mode="fetch", task_name="actor")
    assert controller.check_stream_drained("train", "actor")
    controller.clear_partition("train")
    assert controller.check_stream_drained("train", "actor")
    assert not controller.check_stream_drained("train", "unknown-task")
    assert not controller.check_stream_drained("future", "actor")

    checkpoint = str(tmp_path / "queue.pkl")
    controller.save_checkpoint(checkpoint)
    restored = controller_factory()
    restored.load_checkpoint(checkpoint)
    assert restored.check_stream_drained("train", "actor")
    restored.create_partition("train")
    assert not restored.check_stream_drained("train", "actor")
    with pytest.raises(RuntimeError, match="retired"):
        _insert(restored, "completed-insert", last=True)
    assert restored.partitions["train"].actual_sample_count == 0
    assert _insert(restored, "new-generation").size == 1


def test_partially_failed_insert_cannot_allocate_again(controller_factory, monkeypatch):
    controller = controller_factory()
    original = controller.generate_batch_meta

    def fail_after_allocation(*_args, **_kwargs):
        raise RuntimeError("injected metadata packaging failure")

    monkeypatch.setattr(controller, "generate_batch_meta", fail_after_allocation)
    with pytest.raises(RuntimeError):
        _insert(controller, "partial-failure")
    assert controller.partitions["train"].actual_sample_count == 1
    monkeypatch.setattr(controller, "generate_batch_meta", original)
    with pytest.raises(RuntimeError, match="retired"):
        _insert(controller, "partial-failure")
    assert controller.partitions["train"].actual_sample_count == 1
