import pytest
import torch

from relax.utils.training.indexer_replay import IndexerReplay, maybe_replay_indexer_topk


@pytest.fixture(autouse=True)
def clear_indexer_replay():
    IndexerReplay.clear()
    yield
    IndexerReplay.clear()


def test_indexer_replay_preserves_padding_and_replays_forward_and_backward():
    rollout_topk = torch.tensor([[9, 8], [-1, -1]], dtype=torch.int32)
    computed_topk = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)
    row_valid = torch.tensor([True, False])

    IndexerReplay.begin_step([3])
    IndexerReplay.record(3, rollout_topk, row_valid)

    IndexerReplay.set_stage("replay_forward")
    assert torch.equal(maybe_replay_indexer_topk(3, computed_topk), torch.tensor([[9, 8], [3, 4]]))
    IndexerReplay.reset_forward()
    assert torch.equal(maybe_replay_indexer_topk(3, computed_topk), torch.tensor([[9, 8], [3, 4]]))

    IndexerReplay.set_stage("replay_backward")
    assert torch.equal(maybe_replay_indexer_topk(3, computed_topk), torch.tensor([[9, 8], [3, 4]]))
    IndexerReplay.finish_step(expect_backward=True)


def test_indexer_replay_falls_through_when_disabled_or_in_fallthrough_stage():
    computed_topk = torch.tensor([[1, 2]], dtype=torch.int32)
    assert maybe_replay_indexer_topk(3, computed_topk) is computed_topk

    IndexerReplay.begin_step([3])
    assert maybe_replay_indexer_topk(3, computed_topk) is computed_topk


def test_indexer_replay_rejects_shape_mismatch():
    IndexerReplay.begin_step([3])
    IndexerReplay.record(
        3,
        torch.tensor([[9, 8]], dtype=torch.int32),
        torch.tensor([True]),
    )
    IndexerReplay.set_stage("replay_forward")

    with pytest.raises(RuntimeError, match="shape mismatch"):
        maybe_replay_indexer_topk(3, torch.tensor([[1, 2], [3, 4]], dtype=torch.int32))


def test_indexer_replay_forward_stage_restores_after_failure():
    IndexerReplay.begin_step([3])
    IndexerReplay.set_stage("replay_backward")

    with pytest.raises(RuntimeError, match="forward failed"):
        with IndexerReplay.forward_stage():
            assert IndexerReplay.get_stage() == "replay_forward"
            raise RuntimeError("forward failed")

    assert IndexerReplay.get_stage() == "replay_backward"


def test_indexer_replay_forward_stage_preserves_fallthrough():
    IndexerReplay.begin_step([3])

    with IndexerReplay.forward_stage():
        assert IndexerReplay.get_stage() == "fallthrough"

    assert IndexerReplay.get_stage() == "fallthrough"
