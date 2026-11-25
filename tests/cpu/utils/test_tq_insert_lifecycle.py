# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from types import SimpleNamespace
from uuid import uuid4

import pytest
from transfer_queue import controller as tq_controller


@pytest.fixture
def controller(monkeypatch):
    # Exercise the real allocator and checkpoint format. The wire integration
    # in test_tq_dataplane_behavior runs the same controller inside Ray.
    cls = tq_controller.TransferQueueController.__ray_metadata__.modified_class
    for method in ("_init_zmq_socket", "_start_process_handshake", "_start_process_request"):
        monkeypatch.setattr(cls, method, lambda self: None)
    return cls(polling_mode=True)


def insert_params(controller):
    params = dict(data_fields=["a"], batch_size=1, partition_id="p", insert_id=uuid4().hex, is_last=True)
    ticket = controller.get_metadata(**params, mode="prepare_insert").extra_info["insert_ticket"]
    return {**params, "mode": "insert", "insert_ticket": ticket}


def test_controller_replayed_insert_is_atomic_and_validates_arguments(controller):
    params = insert_params(controller)
    ready = Barrier(2)

    def replay():
        ready.wait(timeout=5)
        return controller.get_metadata(**params).global_indexes

    with ThreadPoolExecutor(max_workers=2) as pool:
        requests = [pool.submit(replay) for _ in range(2)]
        assert [request.result(timeout=10) for request in requests] == [[0], [0]]
    for change in ({"is_last": False}, {"data_fields": ["b"]}, {"partition_id": "other"}, {"batch_size": 2}):
        with pytest.raises(ValueError, match="different parameters"):
            controller.get_metadata(**{**params, **change})
    assert controller.get_partition_snapshot("p").actual_sample_count == 1
    # A legacy GET_META without a ticket still allocates independent rows.
    assert controller.get_metadata(["a"], "p", mode="insert", batch_size=1).global_indexes == [1]


@pytest.mark.parametrize("retire", ["partition", "row", "restore", "expiry"])
def test_controller_retired_insert_cannot_reallocate(controller, retire, tmp_path, monkeypatch):
    now = [100.0]
    if retire == "expiry":
        monkeypatch.setattr(tq_controller.time, "monotonic", lambda: now[0])
    params = insert_params(controller)
    row = controller.get_metadata(**params)
    if retire == "partition":
        controller.clear_partition("p")
        controller.create_partition("p")
    elif retire == "row":
        controller.clear_meta(row.global_indexes, row.partition_ids)
    elif retire == "restore":
        checkpoint = str(tmp_path / "state.pkl")
        controller.save_checkpoint(checkpoint)
        controller.load_checkpoint(checkpoint)
    else:
        now[0] += 1e12

    fresh = controller.get_metadata(**insert_params(controller)).global_indexes
    live_rows = controller.get_partition_snapshot("p").actual_sample_count
    with pytest.raises(ValueError, match="retired"):
        controller.get_metadata(**params)
    assert controller.get_partition_snapshot("p").actual_sample_count == live_rows
    assert controller.get_metadata(**insert_params(controller)).global_indexes != fresh


def test_controller_processes_notification_queued_behind_empty_poll(controller, monkeypatch):
    from transfer_queue.storage.managers.base import TQ_DATA_UPDATE_RESPONSE_TIMEOUT
    from transfer_queue.utils.zmq_utils import ZMQMessage, ZMQRequestType

    allocation = controller.get_metadata(["a"], "p", mode="insert", batch_size=1, is_last=True)
    # Drive the real shared ROUTER dispatcher with ordered wire frames and a
    # virtual clock. A blocking read consumes the entire notification deadline.
    now = [0.0]

    def sleep(seconds):
        now[0] += seconds

    monkeypatch.setattr(tq_controller, "time", SimpleNamespace(time=lambda: now[0], sleep=sleep))
    monkeypatch.setattr(tq_controller, "TQ_CONTROLLER_GET_METADATA_TIMEOUT", TQ_DATA_UPDATE_RESPONSE_TIMEOUT + 1)
    fetch = ZMQMessage.create(
        request_type=ZMQRequestType.GET_META,
        sender_id="reader",
        body={"partition_id": "waiting", "data_fields": ["a"], "task_name": "actor", "token_budget": 100},
    )
    notify = ZMQMessage.create(
        request_type=ZMQRequestType.NOTIFY_DATA_UPDATE,
        sender_id="writer",
        body={
            "partition_id": "p",
            "global_indexes": allocation.global_indexes,
            "field_schema": {"a": {"dtype": "torch.float32", "shape": (2,)}},
        },
    )
    messages = iter(([b"reader", *fetch.serialize()], [b"writer", *notify.serialize()]))
    replies = []

    def record_reply(frames):
        replies.append((now[0], ZMQMessage.deserialize(frames[1:])))

    controller.request_handle_socket = SimpleNamespace(
        recv_multipart=lambda copy: next(messages), send_multipart=record_reply
    )
    with pytest.raises(StopIteration):
        controller._process_request()
    read, ack = replies
    assert read[1].body["metadata"].size == 0
    assert ack[0] < TQ_DATA_UPDATE_RESPONSE_TIMEOUT
    assert ack[1].request_type == ZMQRequestType.NOTIFY_DATA_UPDATE_ACK
    assert ack[1].body["success"] is True
    assert controller.check_production_completed("p")
