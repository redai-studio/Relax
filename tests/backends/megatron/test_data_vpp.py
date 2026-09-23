import importlib
import sys
import types
from argparse import Namespace

import pytest
import torch


def _load_data_module(monkeypatch):
    # Importing a module also assigns attributes on its parent package. Restore
    # both caches so fake process groups cannot leak into later integration tests.
    parent = importlib.import_module("relax.backends.megatron")
    for leaf in ("data", "cp_utils"):
        name = f"relax.backends.megatron.{leaf}"
        monkeypatch.setitem(sys.modules, name, sys.modules.get(name))
        sys.modules.pop(name)
        monkeypatch.setattr(parent, leaf, None, raising=False)

    megatron = types.ModuleType("megatron")
    core = types.ModuleType("megatron.core")
    mpu = types.ModuleType("megatron.core.mpu")
    packed_seq_params = types.ModuleType("megatron.core.packed_seq_params")
    training = types.ModuleType("megatron.training")
    global_vars = types.ModuleType("megatron.training.global_vars")
    tracking_utils = types.ModuleType("relax.utils.tracking_utils")

    class _PackedSeqParams:
        pass

    core.mpu = mpu
    packed_seq_params.PackedSeqParams = _PackedSeqParams
    global_vars.get_args = lambda: None

    modules = {
        "megatron": megatron,
        "megatron.core": core,
        "megatron.core.mpu": mpu,
        "megatron.core.packed_seq_params": packed_seq_params,
        "megatron.training": training,
        "megatron.training.global_vars": global_vars,
        "relax.utils.tracking_utils": tracking_utils,
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    sys.modules.pop("relax.backends.megatron.data", None)
    return importlib.import_module("relax.backends.megatron.data")


@pytest.mark.parametrize("already_imported", [False, True])
def test_data_module_loader_restores_import_caches(monkeypatch, already_imported):
    parent = importlib.import_module("relax.backends.megatron")
    originals = {}
    for leaf in ("data", "cp_utils"):
        name = f"relax.backends.megatron.{leaf}"
        if already_imported:
            originals[leaf] = types.ModuleType(name)
            monkeypatch.setitem(sys.modules, name, originals[leaf])
            monkeypatch.setattr(parent, leaf, originals[leaf], raising=False)
        else:
            monkeypatch.delitem(sys.modules, name, raising=False)
            monkeypatch.delattr(parent, leaf, raising=False)

    with pytest.MonkeyPatch.context() as isolated:
        data_module = _load_data_module(isolated)
        assert data_module.mpu is sys.modules["megatron.core.mpu"]
        assert parent.cp_utils.mpu is data_module.mpu

    for leaf in ("data", "cp_utils"):
        name = f"relax.backends.megatron.{leaf}"
        if already_imported:
            assert sys.modules[name] is originals[leaf]
            assert getattr(parent, leaf) is originals[leaf]
        else:
            assert name not in sys.modules
            assert not hasattr(parent, leaf)


def test_vpp_microbatch_rounding_uses_ceil_multiple(monkeypatch):
    data_module = _load_data_module(monkeypatch)

    rounded = data_module._round_up_to_microbatch_group(torch.tensor([1, 2, 3, 5]), microbatch_group_size=4)

    assert rounded.tolist() == [4, 4, 4, 8]


@pytest.mark.parametrize(
    "seqlens, capacity, num_partitions, first_fit_count, expect_fallback",
    [
        pytest.param([2, 2, 3, 3, 5, 2], 6, 3, 3, True, id="kk-over-capacity"),
        pytest.param([6, 6, 10, 10, 7, 1] * 4, 20, 9, 8, True, id="split-fallback-to-target"),
        pytest.param([2, 2, 3, 3, 5, 2], 6, 4, 3, False, id="keep-valid-kk"),
    ],
)
def test_seqlen_partitions_respect_capacity(
    monkeypatch: pytest.MonkeyPatch,
    seqlens: list[int],
    capacity: int,
    num_partitions: int,
    first_fit_count: int,
    expect_fallback: bool,
) -> None:
    data_module = _load_data_module(monkeypatch)
    kk_partitions = data_module.get_seqlen_balanced_partitions(seqlens, num_partitions, equal_size=False)
    assert any(sum(seqlens[index] for index in partition) > capacity for partition in kk_partitions) == expect_fallback

    first_fit = data_module.get_first_fit_partitions(seqlens, capacity)
    assert len(first_fit) == first_fit_count
    assert data_module.get_minimum_num_micro_batch_size(seqlens, capacity) == first_fit_count

    partitions, dummy_offsets = data_module._get_seqlen_partitions_with_dummy_padding(
        seqlens, num_partitions, capacity
    )

    assert len(partitions) == num_partitions
    assert all(partitions)
    assert all(sum(seqlens[index] for index in partition) <= capacity for partition in partitions)
    assert sorted(index for partition in partitions for index in partition) == list(range(len(seqlens)))
    assert not dummy_offsets
    if not expect_fallback:
        assert partitions == kk_partitions


def test_rollout_minibatch_plan_derives_from_global_batch(monkeypatch):
    data_module = _load_data_module(monkeypatch)
    args = Namespace(
        rollout_batch_size=8,
        n_samples_per_prompt=8,
        global_batch_size=32,
        num_steps_per_rollout=None,
    )

    plan = data_module.build_rollout_minibatch_plan(args, dp_size=2)

    assert plan.num_rollout_minis == 2
    assert plan.mini_rollout_batch_size == 4
    assert plan.mini_global_samples == 32
    assert plan.mini_local_sample_request == 16


def test_rollout_minibatch_plan_prefers_explicit_steps(monkeypatch):
    data_module = _load_data_module(monkeypatch)
    args = Namespace(
        rollout_batch_size=12,
        n_samples_per_prompt=8,
        global_batch_size=None,
        num_steps_per_rollout=3,
    )

    plan = data_module.build_rollout_minibatch_plan(args, dp_size=2)

    assert plan.num_rollout_minis == 3
    assert plan.mini_rollout_batch_size == 4
    assert plan.mini_global_samples == 32
    assert plan.mini_local_sample_request == 16


def test_rollout_minibatch_plan_rejects_non_divisible_prompt_groups(monkeypatch):
    data_module = _load_data_module(monkeypatch)
    args = Namespace(
        rollout_batch_size=9,
        n_samples_per_prompt=8,
        global_batch_size=None,
        num_steps_per_rollout=3,
    )

    with pytest.raises(ValueError, match="mini_rollout_batch_size must be divisible"):
        data_module.build_rollout_minibatch_plan(args, dp_size=2)


def test_log_rollout_data_creates_collective_stats_on_training_device(monkeypatch):
    data_module = _load_data_module(monkeypatch)
    loss_masks = [torch.tensor([0, 1]), torch.tensor([0, 1, 1])]
    requested_devices = []
    original_tensor = torch.tensor

    monkeypatch.setattr(data_module.mpu, "get_tensor_model_parallel_rank", lambda: 0, raising=False)
    monkeypatch.setattr(data_module.mpu, "is_pipeline_last_stage", lambda: True, raising=False)
    monkeypatch.setattr(data_module.mpu, "get_context_parallel_world_size", lambda: 1, raising=False)
    monkeypatch.setattr(
        data_module.mpu,
        "get_data_parallel_group",
        lambda with_context_parallel=True: object(),
        raising=False,
    )
    monkeypatch.setattr(data_module.device_utils, "make_current_torch_device", lambda: "training-device")

    def capture_tensor(*args, **kwargs):
        requested_devices.append(kwargs.get("device"))
        kwargs["device"] = "cpu"
        return original_tensor(*args, **kwargs)

    monkeypatch.setattr(data_module.torch, "tensor", capture_tensor)
    monkeypatch.setattr(data_module.dist, "all_reduce", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(data_module, "gather_log_data", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(data_module, "maybe_padded_total_lengths", lambda *_args, **_kwargs: None)

    args = Namespace(
        qkv_format="thd",
        is_vl_model=False,
        uses_unsplit_forward=False,
        dynamic_context_parallel=False,
        use_opd=False,
        rollout_batch_size=2,
        n_samples_per_prompt=1,
        ci_test=False,
        log_multi_turn=False,
        log_correct_samples=False,
    )
    rollout_data = {
        "total_lengths": [4, 6],
        "response_lengths": [2, 3],
        "loss_masks": loss_masks,
    }

    data_module.log_rollout_data(rollout_id=0, args=args, rollout_data=rollout_data)

    assert requested_devices == ["training-device"]


def test_concat_rollout_batches_preserves_order_and_scalar_metadata(monkeypatch):
    data_module = _load_data_module(monkeypatch)

    merged = data_module.concat_rollout_batches(
        [
            {
                "tokens": ["a", "b"],
                "total_lengths": [1, 2],
                "scores": torch.tensor([[1], [2]]),
                "weight_version": 7,
            },
            {
                "tokens": ["c"],
                "total_lengths": [3],
                "scores": torch.tensor([[3]]),
                "weight_version": 7,
            },
        ]
    )

    assert merged["tokens"] == ["a", "b", "c"]
    assert merged["total_lengths"] == [1, 2, 3]
    assert torch.equal(merged["scores"], torch.tensor([[1], [2], [3]]))
    assert merged["weight_version"] == 7


def test_get_data_iterator_uses_rollout_mini_boundaries_with_balance_data(monkeypatch):
    data_module = _load_data_module(monkeypatch)
    monkeypatch.setattr(
        data_module.mpu,
        "get_data_parallel_world_size",
        lambda with_context_parallel=False: 2,
        raising=False,
    )
    monkeypatch.setattr(data_module.mpu, "get_data_parallel_group", lambda: object(), raising=False)
    monkeypatch.setattr(data_module.mpu, "get_virtual_pipeline_model_parallel_world_size", lambda: None, raising=False)
    monkeypatch.setattr(data_module.mpu, "get_context_parallel_world_size", lambda: 1, raising=False)
    monkeypatch.setattr(data_module.device_utils, "make_current_torch_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(data_module.dist, "all_reduce", lambda tensor, op=None, group=None: None)

    args = Namespace(
        balance_data=True,
        global_batch_size=32,
        micro_batch_size=4,
        use_dynamic_batch_size=False,
    )
    rollout_data = {
        "total_lengths": list(range(32)),
        data_module.ROLLOUT_MINI_LOCAL_SAMPLE_COUNTS_KEY: [16, 16],
    }

    data_iterators, num_microbatches = data_module.get_data_iterator(args, object(), rollout_data)

    assert num_microbatches == [4, 4]
    iterator = data_iterators[0]
    first_step = [iterator.get_next(["total_lengths"])["total_lengths"] for _ in range(4)]
    second_step = [iterator.get_next(["total_lengths"])["total_lengths"] for _ in range(4)]
    assert first_step[0] == [0, 1, 2, 3]
    assert first_step[-1] == [12, 13, 14, 15]
    assert second_step[0] == [16, 17, 18, 19]
    assert second_step[-1] == [28, 29, 30, 31]


def test_get_data_iterator_balance_data_without_boundaries_uses_regular_steps(monkeypatch):
    data_module = _load_data_module(monkeypatch)
    monkeypatch.setattr(
        data_module.mpu,
        "get_data_parallel_world_size",
        lambda with_context_parallel=False: 2,
        raising=False,
    )
    monkeypatch.setattr(data_module.mpu, "get_data_parallel_group", lambda: object(), raising=False)
    monkeypatch.setattr(data_module.mpu, "get_virtual_pipeline_model_parallel_world_size", lambda: None, raising=False)
    monkeypatch.setattr(data_module.mpu, "get_context_parallel_world_size", lambda: 1, raising=False)
    monkeypatch.setattr(data_module.device_utils, "make_current_torch_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(data_module.dist, "all_reduce", lambda tensor, op=None, group=None: None)

    args = Namespace(
        balance_data=True,
        global_batch_size=16,
        micro_batch_size=4,
        use_dynamic_batch_size=False,
    )
    rollout_data = {"total_lengths": list(range(16))}

    _, num_microbatches = data_module.get_data_iterator(args, object(), rollout_data)

    assert num_microbatches == [2, 2]
