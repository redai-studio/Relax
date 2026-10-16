import ast
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
import torch

from relax.utils.training.data_fields import build_data_fields
from relax.utils.types import Sample


@pytest.fixture
def replay_values_and_offsets():
    # Exercise the real tensor helper without loading the distributed runtime.
    source = Path(__file__).resolve().parents[3] / "relax/utils/data/stream_dataloader.py"
    tree = ast.parse(source.read_text())
    function = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_replay_values_and_offsets"
    )
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
    return namespace["_replay_values_and_offsets"]


def test_replay_values_and_offsets_accepts_dense_equal_length_batch(replay_values_and_offsets):
    field_data = torch.arange(2 * 3 * 4, dtype=torch.int32).reshape(2, 3, 4)

    values, offsets = replay_values_and_offsets(field_data, "rollout_indexer_topk")

    assert torch.equal(values, field_data.reshape(6, 4))
    assert torch.equal(offsets, torch.tensor([0, 3, 6]))


def test_replay_values_and_offsets_accepts_jagged_batch(replay_values_and_offsets):
    samples = [torch.ones(2, 4, dtype=torch.int32), torch.full((3, 4), 2, dtype=torch.int32)]
    field_data = torch.nested.as_nested_tensor(samples, layout=torch.jagged)

    values, offsets = replay_values_and_offsets(field_data, "rollout_indexer_topk")

    assert values.shape == (5, 4)
    assert torch.equal(offsets, torch.tensor([0, 2, 5]))


def test_indexer_replay_field_is_actor_only():
    args = Namespace(
        advantage_estimator="ppo",
        fully_async=False,
        hybrid=False,
        kl_coef=0.0,
        loss_type="policy_loss",
        multimodal_keys=None,
        use_kl_loss=False,
        use_opd=False,
        use_rollout_indexer_replay=True,
        use_rollout_routing_replay=False,
    )

    assert "rollout_indexer_topk" in build_data_fields(args, consumer="actor")
    assert "rollout_indexer_topk" not in build_data_fields(args, consumer="critic")
    assert "rollout_indexer_topk" not in build_data_fields(args, consumer="advantages")


def test_indexer_replay_disabled_does_not_change_sample_serialization():
    sample = Sample(tokens=[1, 2])

    assert "rollout_indexer_topk" not in sample.to_dict()


def test_indexer_replay_dynamic_sample_field_round_trips_when_enabled():
    sample = Sample(tokens=[1, 2])
    replay = np.array([[7, 8]], dtype=np.int32)
    setattr(sample, "rollout_indexer_topk", replay)

    restored = Sample.from_dict(sample.to_dict())

    assert np.array_equal(getattr(restored, "rollout_indexer_topk"), replay)


def test_indexer_replay_field_is_absent_when_flag_is_false_or_missing():
    base_args = dict(
        advantage_estimator="grpo",
        loss_type="policy_loss",
        multimodal_keys=None,
        use_opd=False,
        use_rollout_routing_replay=False,
    )

    without_flag = build_data_fields(Namespace(**base_args), consumer="actor")
    with_false_flag = build_data_fields(
        Namespace(**base_args, use_rollout_indexer_replay=False),
        consumer="actor",
    )

    assert without_flag == with_false_flag
    assert "rollout_indexer_topk" not in without_flag
