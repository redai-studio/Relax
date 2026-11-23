# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Sampling support is an owned CSR contract, independent of top-k scoring."""

import io
import json
from argparse import Namespace

import pytest
import torch

from relax.utils.sampling_mask import SAMPLING_MASK_FIELDS, RolloutSamplingMask
from relax.utils.training.data_fields import build_data_fields
from relax.utils.types import Sample
from relax.utils.utils import convert_samples_to_train_data


def replay_args(**overrides):
    return Namespace(
        **{
            "advantage_estimator": "grpo",
            "loss_type": "policy_loss",
            "fully_async": False,
            "hybrid": False,
            "kl_coef": 0.0,
            "multimodal_keys": None,
            "use_opd": False,
            "use_kl_loss": False,
            "reward_key": None,
            "rewards_normalization": False,
            "custom_reward_post_process_path": None,
            "debug_train_only": False,
            "use_rollout_sampling_replay": True,
            **overrides,
        }
    )


def replay_samples():
    return [
        Sample(
            index=0,
            tokens=[1, 11, 7, 3],
            response_length=3,
            reward=1.0,
            rollout_log_probs=[-1.0] * 3,
            rollout_sampling_mask=RolloutSamplingMask.from_rows([[11, 25, 91], [7, 12], [3, 8, 14, 27]]),
        ),
        Sample(
            index=1,
            tokens=[1, 42],
            response_length=1,
            reward=0.0,
            rollout_log_probs=[0.0],
            rollout_sampling_mask=RolloutSamplingMask.from_rows([[42]]),
        ),
        Sample(
            index=2,
            tokens=[1],
            response_length=0,
            reward=0.0,
            rollout_log_probs=[],
            rollout_sampling_mask=RolloutSamplingMask.from_rows([]),
        ),
    ]


def assert_support_rows(data):
    # Hand-written CSR, including distinct response and sample boundaries.
    expected = [([11, 25, 91, 7, 12, 3, 8, 14, 27], [0, 3, 5, 9]), ([42], [0, 1]), ([], [0])]
    for field, dtype, column in zip(SAMPLING_MASK_FIELDS, (torch.int32, torch.int64), (0, 1), strict=True):
        rows = list(data[field])
        assert len(rows) == 3
        for row, values in zip(rows, expected, strict=True):
            assert isinstance(row, torch.Tensor)
            assert row.dtype == dtype
            assert row.device.type == "cpu"
            assert row.tolist() == values[column]


def test_sampling_mask_operations_own_storage():
    ids = torch.tensor([11, 25, 7], dtype=torch.int64)
    offsets = torch.tensor([0, 2, 3], dtype=torch.int32)
    mask = RolloutSamplingMask(ids, offsets)
    ids[0], offsets[1] = 99, 1
    assert mask.ids.tolist() == [11, 25, 7]
    assert mask.offsets.tolist() == [0, 2, 3]
    assert mask.ids.dtype == torch.int32
    assert mask.offsets.dtype == torch.int64
    assert len(mask) == 2
    empty = RolloutSamplingMask.from_rows([])
    assert len(empty) == 0
    assert empty.ids.tolist() == []
    assert empty.offsets.tolist() == [0]
    joined = RolloutSamplingMask.concatenate([mask, empty, RolloutSamplingMask.from_rows([[42]])])
    assert joined.ids.tolist() == [11, 25, 7, 42]
    assert joined.offsets.tolist() == [0, 2, 3, 4]
    assert RolloutSamplingMask.concatenate([]).offsets.tolist() == [0]
    prefix = joined.prefix(1)
    assert prefix.ids.tolist() == [11, 25]
    assert prefix.offsets.tolist() == [0, 2]
    assert joined.prefix(0).offsets.tolist() == [0]
    copies = [mask.prefix(len(mask)), RolloutSamplingMask.concatenate([mask]), prefix, joined]
    for copy in copies:
        copy.ids[0] = 0
        copy.offsets[1] = 1
    assert mask.ids.tolist() == [11, 25, 7]
    assert mask.offsets.tolist() == [0, 2, 3]


@pytest.mark.parametrize(
    "ids,offsets",
    [
        ([1], []),
        ([1], [1, 1]),
        ([1], [0, 2]),
        ([1], [0, 0, 1]),
        ([[1]], [0, 1]),
        ([1.5], [0, 1]),
        ([1], [0.0, 1.0]),
        ([-1], [0, 1]),
        ([2**31], [0, 1]),
    ],
)
def test_sampling_mask_rejects_invalid_csr(ids, offsets):
    with pytest.raises(ValueError):
        RolloutSamplingMask(ids, offsets)


@pytest.mark.parametrize("length", [-1, 2])
def test_sampling_mask_rejects_invalid_prefix(length):
    with pytest.raises(ValueError, match="prefix length"):
        RolloutSamplingMask.from_rows([[1]]).prefix(length)


def test_sampling_mask_sample_serialization_compatibility():
    legacy = Sample(tokens=[1]).to_dict()
    assert "rollout_sampling_mask" not in legacy
    assert Sample.from_dict(legacy).rollout_sampling_mask is None
    sample = replay_samples()[0]
    payload = sample.to_dict()
    assert payload["rollout_sampling_mask"] == {"ids": [11, 25, 91, 7, 12, 3, 8, 14, 27], "offsets": [0, 3, 5, 9]}
    assert sample.metadata == {}
    restored = Sample.from_dict(json.loads(json.dumps(payload)))
    assert restored.rollout_sampling_mask.ids.tolist() == payload["rollout_sampling_mask"]["ids"]
    assert restored.rollout_sampling_mask.offsets.dtype == torch.int64
    buffer = io.BytesIO()
    torch.save(sample, buffer)
    buffer.seek(0)
    restored = torch.load(buffer, weights_only=False)
    assert restored.rollout_sampling_mask.offsets.tolist() == [0, 3, 5, 9]


def test_sampling_mask_tensordict_preserves_ragged_rows():
    data = convert_samples_to_train_data(replay_args(), replay_samples())
    assert_support_rows(data)
    assert data["response_lengths"].tolist() == [3, 1, 0]
    # The batch owns storage independently of the original Sample CSR.
    samples = replay_samples()
    data = convert_samples_to_train_data(replay_args(), samples)
    data[SAMPLING_MASK_FIELDS[0]].values()[0] = 0
    assert samples[0].rollout_sampling_mask.ids[0] == 11


@pytest.mark.parametrize("estimator", ["ppo", "grpo"])
def test_sampling_mask_field_selection_is_actor_only(estimator):
    args = replay_args(advantage_estimator=estimator)
    assert set(SAMPLING_MASK_FIELDS) <= set(build_data_fields(args, consumer="actor"))
    for consumer in ("critic", "advantages"):
        assert not set(SAMPLING_MASK_FIELDS) & set(build_data_fields(args, consumer=consumer))
    args.use_rollout_sampling_replay = False
    disabled = build_data_fields(args)
    del args.use_rollout_sampling_replay
    assert build_data_fields(args) == disabled
    assert not set(SAMPLING_MASK_FIELDS) & set(disabled)
    assert not set(SAMPLING_MASK_FIELDS) & set(convert_samples_to_train_data(args, replay_samples()).keys())


@pytest.mark.parametrize("failure", ["missing", "mixed", "length"])
def test_sampling_mask_conversion_rejects_incomplete_support(failure):
    samples = replay_samples()
    if failure == "missing":
        for sample in samples:
            sample.rollout_sampling_mask = None
    elif failure == "mixed":
        samples[1].rollout_sampling_mask = None
    else:
        samples[0].rollout_sampling_mask = RolloutSamplingMask.from_rows([[11]])
    with pytest.raises(ValueError, match="missing rollout_sampling_mask|match response_length"):
        convert_samples_to_train_data(replay_args(), samples)


def test_sampling_mask_custom_converter_cannot_silently_drop_support():
    with pytest.raises(ValueError, match="standard token sample converter"):
        convert_samples_to_train_data(replay_args(custom_convert_samples_to_train_data_path="custom.convert"), [])


def _transfer_round_trip(ray_temp_dir):
    import ray
    import transfer_queue as tq
    from omegaconf import OmegaConf

    from relax.utils.data.stream_dataloader import get_data_from_transfer_queue

    # A private CPU-only cluster; never attach to or stop a training cluster.
    ray.init(
        address="local",
        num_cpus=3,
        num_gpus=0,
        include_dashboard=False,
        object_store_memory=80 * 1024**2,
        _temp_dir=ray_temp_dir,
    )
    try:
        tq.init(
            OmegaConf.create(
                {"backend": {"storage_backend": "SimpleStorage", "SimpleStorage": {"num_data_storage_units": 1}}}
            )
        )
        client = tq.get_client()
        args = replay_args()
        data = convert_samples_to_train_data(args, replay_samples())
        client.put(data, partition_id="sampling_contract")
        received, meta = get_data_from_transfer_queue(
            args=args,
            tq_client=client,
            data_fields=build_data_fields(args),
            batch_size=3,
            partition_id="sampling_contract",
            task_name="actor",
            sampling_config={},
            batch_index=0,
            per_rank_fetch=True,
            synchronize_per_rank_fetch=False,
            post_process=False,
        )
        assert meta.size == 3
        assert received["response_lengths"] == [3, 1, 0]
        assert_support_rows(received)
    finally:
        tq.close()
        ray.shutdown()


def test_sampling_mask_transfer_queue_reaches_training_side():
    import multiprocessing
    import tempfile

    # Isolate Ray/TQ globals; short paths also fit Unix socket limits.
    with tempfile.TemporaryDirectory(prefix="relax-tq-") as ray_temp_dir:
        process = multiprocessing.get_context("spawn").Process(target=_transfer_round_trip, args=(ray_temp_dir,))
        process.start()
        try:
            process.join(timeout=120)
            assert not process.is_alive(), "TransferQueue round-trip timed out"
            assert process.exitcode == 0, "TransferQueue round-trip failed in the isolated worker"
        finally:
            if process.is_alive():
                process.terminate()
                process.join(timeout=10)
            process.close()
