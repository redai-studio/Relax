"""Replay compact host-side indices as gather-compatible micro-batches."""

import os

import pytest
import torch

from relax.distributed.replay_manager.backend import megatron


@pytest.fixture
def replay_type(monkeypatch):
    monkeypatch.setattr(megatron.RoutingReplay, "all_routing_replays", [])
    return megatron.RoutingReplay


def test_routing_replay_upcasts_only_requested_microbatch_and_preserves_replay_order(replay_type, monkeypatch) -> None:
    monkeypatch.setattr(megatron.device_utils, "make_current_torch_device", lambda: torch.device("cpu"))
    replay = replay_type()
    recorded = [
        torch.tensor([[7, 2], [4, 0]], dtype=torch.int32),
        torch.tensor([[1, 8]], dtype=torch.int32),
    ]
    # No CUDA pinning allocator is required to exercise the real pop methods.
    replay.top_indices_list = recorded
    scores = [torch.arange(20).reshape(2, 10).float(), torch.arange(10).reshape(1, 10).float()]

    for i, expected in enumerate(recorded):
        actual = replay.pop_forward()
        assert actual.dtype == torch.int64
        assert torch.equal(scores[i].gather(1, actual), scores[i].gather(1, expected.long()))
        assert replay.forward_index == i + 1
        assert replay.backward_index == 0
    replay.clear_forward()
    assert torch.equal(replay.pop_forward(), recorded[0].long())
    for i, expected in enumerate(recorded):
        assert torch.equal(replay.pop_backward(), expected.long())
        assert replay.backward_index == i + 1
    assert all(value.dtype == torch.int32 and value.device.type == "cpu" for value in recorded)
    replay.clear()
    assert replay.forward_index == replay.backward_index == 0
    assert replay.top_indices_list == []


@pytest.mark.skipif(
    os.environ.get("RELAX_ROUTING_REPLAY_GPU_TEST") != "1",
    reason="set RELAX_ROUTING_REPLAY_GPU_TEST=1 on an available GPU",
)
def test_routing_replay_record_pins_int32_and_replays_long_on_current_gpu(replay_type, monkeypatch) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the pinned-memory allocator")
    monkeypatch.setattr(megatron.device_utils, "make_current_torch_device", lambda: torch.device("cuda", 0))
    replay = replay_type()
    recorded = torch.tensor([[7, 2], [4, 0]], dtype=torch.int32)

    replay.record(recorded)

    assert replay.top_indices_list[0].dtype == torch.int32
    assert replay.top_indices_list[0].device.type == "cpu"
    assert replay.top_indices_list[0].is_pinned()
    for actual in (replay.pop_forward(), replay.pop_backward()):
        assert actual.dtype == torch.int64 and actual.is_cuda
        assert torch.equal(actual.cpu(), recorded.long())
