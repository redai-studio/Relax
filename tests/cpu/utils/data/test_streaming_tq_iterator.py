import importlib
import json
import sys
import types
from argparse import Namespace
from datetime import timedelta

import pytest
import torch


def _load_stream_module(monkeypatch):
    megatron = types.ModuleType("megatron")
    core = types.ModuleType("megatron.core")
    mpu = types.ModuleType("megatron.core.mpu")
    transfer_queue = types.ModuleType("transfer_queue")
    tq_dataloader = types.ModuleType("transfer_queue.dataloader")
    streaming_dataloader = types.ModuleType("transfer_queue.dataloader.streaming_dataloader")
    streaming_dataset = types.ModuleType("transfer_queue.dataloader.streaming_dataset")
    tensordict = types.ModuleType("tensordict")

    mpu.get_tensor_model_parallel_rank = lambda: 1
    mpu.get_context_parallel_rank = lambda: 0
    mpu.get_data_parallel_group = lambda with_context_parallel=True: object()
    core.mpu = mpu

    class _StreamingDataLoader:
        pass

    class _StreamingDataset:
        pass

    streaming_dataloader.StreamingDataLoader = _StreamingDataLoader
    streaming_dataset.StreamingDataset = _StreamingDataset
    tensordict.TensorDict = dict

    modules = {
        "megatron": megatron,
        "megatron.core": core,
        "megatron.core.mpu": mpu,
        "transfer_queue": transfer_queue,
        "transfer_queue.dataloader": tq_dataloader,
        "transfer_queue.dataloader.streaming_dataloader": streaming_dataloader,
        "transfer_queue.dataloader.streaming_dataset": streaming_dataset,
        "tensordict": tensordict,
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    sys.modules.pop("relax.utils.data.stream_dataloader", None)
    return importlib.import_module("relax.utils.data.stream_dataloader")


def _make_iterator(stream_module, *, get_data_fn, all_consumed_fn, **kwargs):
    monkeypatch_kwargs = dict(
        args=Namespace(),
        tq_client=object(),
        data_fields=["tokens"],
        rollout_id=52,
        token_budget=1024,
        loss_scale=0.5,
        all_consumed_fn=all_consumed_fn,
        dp_rank=1,
        dp_size=2,
        window_quota=64,
        rollout_mini_index=0,
    )
    monkeypatch_kwargs.update(kwargs)
    return stream_module.StreamingTQIterator(**monkeypatch_kwargs)


def test_micro_batch_list_iterator_observability_excludes_vpp_padding(monkeypatch, tmp_path):
    from relax.utils.agentic_obs import resolve_agentic_obs_config
    from relax.utils.agentic_obs.train_hooks import (
        capture_training_input,
        close_train_appenders,
        observe_train_step,
        record_optimizer_result,
    )

    stream_module = _load_stream_module(monkeypatch)
    mpu = stream_module.mpu
    mpu.get_tensor_model_parallel_rank = lambda: 0
    mpu.is_pipeline_last_stage = lambda **kwargs: True
    mpu.get_context_parallel_world_size = lambda: 1
    mpu.get_data_parallel_rank = lambda **kwargs: 0
    mpu.get_data_parallel_world_size = lambda **kwargs: 1
    batch = {
        "sample_export_ids": [7, 8],
        "agentic_group_ids": ["group-7", "group-8"],
        "response_lengths": [2, 2],
        "total_lengths": [4, 4],
        "loss_masks": [torch.ones(2), torch.ones(2)],
        "advantages": [torch.ones(2), torch.ones(2)],
    }
    iterator = stream_module.MicroBatchListIterator([(batch, None), (batch, None)], dummy_after=1)
    args = Namespace(dump_agentic_obs=str(tmp_path), agentic_obs_run_id="run_test")
    with observe_train_step(args, 1, 0):
        actual, _ = next(iterator)
        capture_training_input(actual)
        # Actual capture precedes dynamic CP's destructive subgroup selection.
        actual["sample_export_ids"] = [8]
        dummy, _ = next(iterator)
        assert dummy["__is_dummy__"]
        capture_training_input(dummy)
        record_optimizer_result(True)
    assert len(iterator.get_buffer()) == 2
    close_train_appenders()
    path = resolve_agentic_obs_config(args).train_scalar_path(1, 0)
    (record,) = [json.loads(line) for line in path.read_text().splitlines()]
    assert [(row["sample_export_id"], row["effective_tokens"]) for row in record["samples"]] == [("7", 2), ("8", 2)]


def test_streaming_tq_iterator_sampling_config_carries_window_quota(monkeypatch):
    """The consumer forwards window_quota + rollout_mini_index (and NOT any
    per-DP count keys) so the sampler can enforce the per-window global cap."""
    stream_module = _load_stream_module(monkeypatch)
    monkeypatch.setattr(stream_module.device_utils, "make_current_torch_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(stream_module.dist, "all_reduce", lambda tensor, op=None, group=None: None)

    iterator = _make_iterator(
        stream_module,
        get_data_fn=lambda **k: (None, None),
        all_consumed_fn=lambda: True,
        window_quota=64,
        rollout_mini_index=3,
    )
    config = iterator._sampling_config(7)
    assert config["window_quota"] == 64
    assert config["rollout_mini_index"] == 3
    assert config["batch_index"] == 7
    # Removed per-DP count keys must be gone.
    assert "max_samples" not in config
    assert "remaining_samples" not in config
    assert "consumed_samples" not in config


def test_get_data_from_transfer_queue_can_skip_per_rank_agreement(monkeypatch):
    stream_module = _load_stream_module(monkeypatch)

    class _Meta:
        size = 0

    class _Client:
        def get_meta(self, **kwargs):
            return _Meta()

    def fail_agreement(*args, **kwargs):
        raise AssertionError("_agree_on_fetch should not run")

    monkeypatch.setattr(stream_module, "_agree_on_fetch", fail_agreement)

    data, meta = stream_module.get_data_from_transfer_queue(
        args=Namespace(),
        tq_client=_Client(),
        data_fields=["tokens"],
        batch_size=1,
        partition_id="partition",
        task_name="task",
        sampling_config={},
        batch_index=0,
        broadcast_pp=False,
        per_rank_fetch=True,
        synchronize_per_rank_fetch=False,
    )

    assert data is None
    assert meta.size == 0


def test_get_data_from_transfer_queue_rejects_disabled_agreement_without_per_rank_fetch(monkeypatch):
    stream_module = _load_stream_module(monkeypatch)

    with pytest.raises(ValueError, match="requires per_rank_fetch=True"):
        stream_module.get_data_from_transfer_queue(
            args=Namespace(),
            tq_client=object(),
            data_fields=["tokens"],
            batch_size=1,
            partition_id="partition",
            task_name="task",
            sampling_config={},
            batch_index=0,
            per_rank_fetch=False,
            synchronize_per_rank_fetch=False,
        )


def test_fetch_data_from_transfer_queue_returns_raw_cpu_payload(monkeypatch):
    stream_module = _load_stream_module(monkeypatch)

    class _Meta:
        size = 1

    class _Client:
        def __init__(self):
            self.meta_kwargs = None

        def get_meta(self, **kwargs):
            self.meta_kwargs = kwargs
            return _Meta()

        def get_data(self, meta):
            assert isinstance(meta, _Meta)
            return {"tokens": [torch.tensor([1])]}

    client = _Client()
    raw_data, elapsed = stream_module.fetch_data_from_transfer_queue(
        tq_client=client,
        data_fields=["tokens"],
        batch_size=2,
        partition_id="sft_3",
        task_name="sft_train",
        sampling_config={"dp_rank": 1},
        batch_index=0,
    )

    assert raw_data[0]["tokens"][0].tolist() == [1]
    assert raw_data[1].size == 1
    assert elapsed >= 0
    assert client.meta_kwargs["sampling_config"] == {"dp_rank": 1, "batch_index": 0, "partition_id": "sft_3"}


def test_prefetched_raw_payload_skips_second_tq_rpc(monkeypatch):
    stream_module = _load_stream_module(monkeypatch)
    monkeypatch.setattr(stream_module.device_utils, "make_current_torch_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(stream_module.dist, "all_reduce", lambda tensor, op=None, group=None: None)
    monkeypatch.setattr(stream_module.mpu, "get_tensor_and_context_parallel_group", lambda: object(), raising=False)

    class _Meta:
        size = 0

    class _Client:
        def get_meta(self, **kwargs):
            raise AssertionError("prefetched payload must not fetch metadata again")

        def get_data(self, meta):
            raise AssertionError("prefetched payload must not fetch data again")

    meta = _Meta()
    rollout_data, batch_meta = stream_module.get_data_from_transfer_queue(
        args=Namespace(),
        tq_client=_Client(),
        data_fields=["tokens"],
        batch_size=2,
        partition_id="sft_3",
        task_name="sft_train",
        sampling_config={"dp_rank": 1},
        batch_index=0,
        broadcast_pp=False,
        per_rank_fetch=True,
        prefetched_rollout_data=[None, meta],
        prefetched_fetch_time_s=0.1,
    )

    assert rollout_data is None
    assert batch_meta is meta


def test_streaming_tq_iterator_finishes_on_window_drained_with_underfill(monkeypatch):
    """Regression for the fully-async DP-imbalance deadlock.

    Per-DP sample counts are uneven (token-budget driven). The iterator
    finishes when the per-window drained signal fires — NOT on a per-DP count
    target — so a DP may finish underfilled and the dummy-pad aligns cross-DP
    mb counts.
    """
    stream_module = _load_stream_module(monkeypatch)
    monkeypatch.setattr(stream_module.device_utils, "make_current_torch_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(stream_module.dist, "all_reduce", lambda tensor, op=None, group=None: None)

    batches = [{"tokens": [torch.tensor([0]), torch.tensor([1])]}]
    drained = {"flag": False}

    def fake_get_data_from_transfer_queue(**kwargs):
        if batches:
            return batches.pop(0), object()
        drained["flag"] = True  # window quota met → controller reports drained
        return None, None

    monkeypatch.setattr(stream_module, "get_data_from_transfer_queue", fake_get_data_from_transfer_queue)

    iterator = _make_iterator(
        stream_module,
        get_data_fn=fake_get_data_from_transfer_queue,
        all_consumed_fn=lambda: drained["flag"],
    )

    assert len(next(iterator)[0]["tokens"]) == 2
    # Underfilled (2 of 64 window quota) but window drained → clean StopIteration.
    with pytest.raises(StopIteration):
        next(iterator)
    assert iterator._sample_count == 2


def test_streaming_tq_iterator_polls_until_window_drained(monkeypatch):
    """Empty fetch while the window is NOT yet drained -> keep polling; once
    the per-window drained signal flips, finish (no per-DP count involved)."""
    stream_module = _load_stream_module(monkeypatch)
    monkeypatch.setattr(stream_module.device_utils, "make_current_torch_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(stream_module.dist, "all_reduce", lambda tensor, op=None, group=None: None)
    monkeypatch.setattr(stream_module.time, "sleep", lambda _s: None)

    # One real mb, then several empty polls, then drained.
    events = [("data", {"tokens": [torch.tensor([0])]})] + [("empty", None)] * 3
    drained = {"flag": False}

    def fake_get_data_from_transfer_queue(**kwargs):
        if events:
            kind, payload = events.pop(0)
            if kind == "data":
                return payload, object()
        if not events:
            drained["flag"] = True
        return None, None

    monkeypatch.setattr(stream_module, "get_data_from_transfer_queue", fake_get_data_from_transfer_queue)

    iterator = _make_iterator(
        stream_module,
        get_data_fn=fake_get_data_from_transfer_queue,
        all_consumed_fn=lambda: drained["flag"],
    )
    assert len(next(iterator)[0]["tokens"]) == 1
    with pytest.raises(StopIteration):
        next(iterator)


def test_streaming_tq_iterator_raises_on_stall_timeout(monkeypatch):
    """Deadlock guard: no data + window not drained past the stall timeout must
    raise (fast, diagnosable failure) instead of polling forever."""
    stream_module = _load_stream_module(monkeypatch)
    monkeypatch.setattr(stream_module.device_utils, "make_current_torch_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(stream_module.dist, "all_reduce", lambda tensor, op=None, group=None: None)
    monkeypatch.setattr(stream_module.time, "sleep", lambda _s: None)

    monkeypatch.setattr(stream_module, "get_data_from_transfer_queue", lambda **k: (None, None))

    iterator = _make_iterator(
        stream_module,
        get_data_fn=lambda **k: (None, None),
        all_consumed_fn=lambda: False,  # never drains → would hang forever
        max_stream_stall_s=0.0001,  # trip almost immediately
    )
    with pytest.raises(RuntimeError, match="stalled"):
        next(iterator)


def test_streaming_tq_iterator_rejects_nonpositive_window_quota(monkeypatch):
    stream_module = _load_stream_module(monkeypatch)
    monkeypatch.setattr(stream_module.device_utils, "make_current_torch_device", lambda: torch.device("cpu"))
    with pytest.raises(ValueError, match="window_quota"):
        _make_iterator(
            stream_module,
            get_data_fn=lambda **k: (None, None),
            all_consumed_fn=lambda: True,
            window_quota=0,
        )


def test_streaming_tq_iterator_per_round_dummy_then_end(monkeypatch):
    """Train path (per_round_dummy): on an empty fetch the consumer reads the
    DUMMY-round flag from the fetched meta's extra_info and emits a zero-grad
    dummy, then StopIteration when the fetched meta carries STREAM_END.

    The end verdict rides the SAME empty fetch (stamped atomically by the
    controller), NOT a separate all_consumed_fn() RPC — so all_consumed_fn is
    pinned False here to prove the stop is driven by the fetch, not the RPC.
    Guarantees equal micro-batch counts across DP/PP without a cross-DP
    all_reduce (MoE lockstep) and without the two-RPC TOCTOU that let PP stages
    diverge across the drain boundary.
    """
    stream_module = _load_stream_module(monkeypatch)
    monkeypatch.setattr(stream_module.device_utils, "make_current_torch_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(stream_module.dist, "all_reduce", lambda tensor, op=None, group=None: None)
    monkeypatch.setattr(stream_module.time, "sleep", lambda _s: None)

    class _Meta:
        def __init__(self, extra_info):
            self.extra_info = extra_info

    # 2 real fetches, then 2 DUMMY-round empties (meta carries the flag), then an
    # end empty whose meta carries STREAM_END (window drained).
    events = [
        ({"tokens": [torch.tensor([0])]}, _Meta({})),
        ({"tokens": [torch.tensor([1])]}, _Meta({})),
        (None, _Meta({"dummy_round": True})),
        (None, _Meta({"dummy_round": True})),
        (None, _Meta({"stream_end": True})),
    ]

    def fake_get(**kwargs):
        return events.pop(0) if events else (None, _Meta({"stream_end": True}))

    monkeypatch.setattr(stream_module, "get_data_from_transfer_queue", fake_get)

    iterator = _make_iterator(
        stream_module,
        get_data_fn=fake_get,
        all_consumed_fn=lambda: False,  # never consulted on the train path anymore
        per_round_dummy=True,
    )

    assert not next(iterator)[0].get("__is_dummy__")  # real mb 0
    assert not next(iterator)[0].get("__is_dummy__")  # real mb 1
    assert next(iterator)[0]["__is_dummy__"] is True  # dummy round
    assert next(iterator)[0]["__is_dummy__"] is True  # dummy round
    with pytest.raises(StopIteration):  # empty, no dummy flag, stream_end → END
        next(iterator)
    assert iterator._mb_count == 4  # 2 real + 2 dummy → equal count with a busier DP
    assert iterator._sample_count == 2  # dummies don't count as samples


@pytest.mark.parametrize("dense_ids", [False, True])
def test_get_data_from_transfer_queue_converts_nested_length_and_reward_fields(monkeypatch, dense_ids):
    stream_module = _load_stream_module(monkeypatch)
    monkeypatch.setattr(stream_module.mpu, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(stream_module.mpu, "get_tensor_model_parallel_world_size", lambda: 1, raising=False)
    monkeypatch.setattr(stream_module.mpu, "get_context_parallel_world_size", lambda: 1, raising=False)
    monkeypatch.setattr(stream_module.device_utils, "make_current_torch_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(stream_module, "_maybe_log_tgd_pickle_diag", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(stream_module, "post_process_rollout_data", lambda *_args, **_kwargs: None)

    class _Meta:
        size = 2

    class _TQClient:
        def get_meta(self, **kwargs):
            return _Meta()

        def get_data(self, batch_meta):
            return {
                "dataset_ids": torch.tensor([0, 1])
                if dense_ids
                else torch.nested.nested_tensor([torch.tensor([0]), torch.tensor([1])], layout=torch.jagged),
                "response_lengths": torch.nested.nested_tensor(
                    [
                        torch.tensor([512]),
                        torch.tensor([135]),
                    ],
                    layout=torch.jagged,
                ),
                "total_lengths": torch.nested.nested_tensor(
                    [
                        torch.tensor([1309]),
                        torch.tensor([842]),
                    ],
                    layout=torch.jagged,
                ),
                "rewards": torch.nested.nested_tensor(
                    [
                        torch.tensor([1.5]),
                        torch.tensor([2.5]),
                    ],
                    layout=torch.jagged,
                ),
            }

    rollout_data, batch_meta = stream_module.get_data_from_transfer_queue(
        args=Namespace(),
        tq_client=_TQClient(),
        data_fields=["response_lengths", "total_lengths", "rewards", "dataset_ids"],
        batch_size=2,
        partition_id="train_0",
        task_name="ref_log_probs",
        sampling_config={},
        batch_index=0,
        broadcast_pp=False,
    )

    assert batch_meta.size == 2
    assert rollout_data["response_lengths"] == [512, 135]
    assert rollout_data["total_lengths"] == [1309, 842]
    assert rollout_data["rewards"] == [1.5, 2.5]
    assert rollout_data["dataset_ids"] == [0, 1]
    assert all(type(value) is int for value in rollout_data["dataset_ids"])


def test_tensor_to_python_values_dense_tensor(monkeypatch):
    stream_module = _load_stream_module(monkeypatch)

    value = torch.tensor([512, 135])

    assert stream_module._tensor_to_python_values(value) == [512, 135]


def test_tensor_to_python_values_jagged_tensor(monkeypatch):
    stream_module = _load_stream_module(monkeypatch)

    value = torch.nested.nested_tensor(
        [
            torch.tensor([1, 2]),
            torch.tensor([3, 4, 5]),
        ],
        layout=torch.jagged,
    )

    assert stream_module._tensor_to_python_values(value) == [
        [1, 2],
        [3, 4, 5],
    ]


def test_tensor_to_python_values_singleton_nested_rows(monkeypatch):
    stream_module = _load_stream_module(monkeypatch)

    value = torch.nested.nested_tensor(
        [
            torch.tensor([512]),
            torch.tensor([135]),
        ],
        layout=torch.jagged,
    )

    assert stream_module._tensor_to_python_values(value) == [512, 135]


def test_tensor_to_python_values_uniform_nested_rows(monkeypatch):
    stream_module = _load_stream_module(monkeypatch)

    value = torch.nested.nested_tensor(
        [
            torch.tensor([1, 2]),
            torch.tensor([3, 4]),
        ],
        layout=torch.jagged,
    )

    assert stream_module._tensor_to_python_values(value) == [
        [1, 2],
        [3, 4],
    ]


def test_tensor_to_python_values_mixed_ragged_rows(monkeypatch):
    stream_module = _load_stream_module(monkeypatch)

    value = torch.nested.nested_tensor(
        [
            torch.tensor([1]),
            torch.tensor([2, 3]),
        ],
        layout=torch.jagged,
    )

    assert stream_module._tensor_to_python_values(value) == [
        [1],
        [2, 3],
    ]


@pytest.mark.parametrize("as_tensor", [False, True])
def test_sft_token_lengths_count_raw_masks_on_cpu(monkeypatch, as_tensor):
    stream_module = _load_stream_module(monkeypatch)
    masks = [[0, 1, 1, 0, 1, 0], [0, 0, 0], [1, 1]]
    batch = {
        "total_lengths": [6, 3, 2],
        "response_lengths": [6, 3, 2],
        "loss_masks": [torch.tensor(mask) for mask in masks] if as_tensor else masks,
    }
    stream_module._add_sft_token_lengths(Namespace(loss_type="sft"), batch)
    assert batch["learn_lengths"] == [3, 0, 2]
    assert batch["non_learn_lengths"] == [3, 3, 0]
    assert batch["response_lengths"] == batch["total_lengths"]  # Keep internal alignment semantics.
    assert all(
        total == learn + non_learn
        for total, learn, non_learn in zip(
            batch["total_lengths"], batch["learn_lengths"], batch["non_learn_lengths"], strict=True
        )
    )
    # Subsequent postprocessing must reuse CPU counts after masks move to a device.
    batch["loss_masks"] = [torch.empty(6, device="meta")]
    stream_module._add_sft_token_lengths(Namespace(loss_type="sft"), batch)
    assert batch["learn_lengths"] == [3, 0, 2]


def test_sft_token_lengths_leave_rl_batch_unchanged(monkeypatch):
    stream_module = _load_stream_module(monkeypatch)
    batch = {"total_lengths": [6], "loss_masks": [[1, 1]]}
    stream_module._add_sft_token_lengths(Namespace(loss_type="policy_loss"), batch)
    assert "learn_lengths" not in batch
    assert "non_learn_lengths" not in batch


@pytest.mark.parametrize(
    "dynamic_cp, allgather_cp, cp_rank, expected_rows",
    [
        (False, False, 0, [6, 7]),
        (False, False, 1, [0, 1, 2, 3, 4, 5]),
        (False, True, 1, [0, 1, 2, 3, 4, 5]),
        (True, False, 0, list(range(8))),
    ],
)
def test_score_centering_head_ingestion_preserves_dtype_and_cp_alignment(
    monkeypatch, dynamic_cp, allgather_cp, cp_rank, expected_rows
):
    stream_module = _load_stream_module(monkeypatch)
    from relax.backends.megatron import cp_utils

    monkeypatch.setattr(cp_utils, "mpu", stream_module.mpu)
    monkeypatch.setattr(stream_module.mpu, "get_context_parallel_world_size", lambda: 2, raising=False)
    monkeypatch.setattr(stream_module.mpu, "get_context_parallel_rank", lambda: cp_rank)
    monkeypatch.setattr(stream_module.device_utils, "make_current_torch_device", lambda: torch.device("cpu"))
    token_ids = torch.arange(16).reshape(8, 2) + 2**24
    log_probs = -torch.arange(16, dtype=torch.float32).reshape(8, 2)
    rollout_data = {
        "tokens": [list(range(12))],
        "total_lengths": [12],
        "response_lengths": [8],
        "loss_masks": [[1] * 8],
        "rollout_log_probs": [list(range(8))],
        "rollout_topk_token_ids": [token_ids.flatten().tolist()],
        "rollout_topk_log_probs": [log_probs.flatten().tolist()],
    }
    args = Namespace(
        loss_type="policy_loss",
        qkv_format="thd",
        use_opd=False,
        dynamic_context_parallel=dynamic_cp,
        allgather_cp=allgather_cp,
        rollout_top_logprobs_num=2,
    )

    stream_module.post_process_rollout_data(args, rollout_data)

    assert rollout_data["rollout_topk_token_ids"][0].dtype == torch.long
    assert rollout_data["rollout_topk_log_probs"][0].dtype == torch.float32
    torch.testing.assert_close(rollout_data["rollout_topk_token_ids"][0], token_ids[expected_rows])
    torch.testing.assert_close(rollout_data["rollout_topk_log_probs"][0], log_probs[expected_rows])
    assert rollout_data["rollout_log_probs"][0].tolist() == expected_rows


def _known_tensor_broadcast_worker(rank, rendezvous):
    import torch.distributed as dist

    torch.set_num_threads(1)
    timeout = timedelta(seconds=90)
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=8, timeout=timeout)
    try:
        coordinates = (rank // 4, rank // 2 % 2, rank % 2)  # PP2, TP2, CP2
        axes = ("pipeline_model", "tensor_model", "context")
        groups = {}
        for index, axis in enumerate(axes):
            for first in range(8):
                if (first // (4, 2, 1)[index]) % 2:
                    continue
                members = [first, first + (4, 2, 1)[index]]
                group = dist.new_group(members, backend="gloo", timeout=timeout)
                if rank in members:
                    groups[axis] = group

        with pytest.MonkeyPatch.context() as monkeypatch:
            stream_module = _load_stream_module(monkeypatch)
            for index, axis in enumerate(axes):
                monkeypatch.setattr(
                    stream_module.mpu,
                    f"get_{axis}_parallel_rank",
                    lambda index=index: coordinates[index],
                    raising=False,
                )
                monkeypatch.setattr(stream_module.mpu, f"get_{axis}_parallel_world_size", lambda: 2, raising=False)
                monkeypatch.setattr(
                    stream_module.mpu, f"get_{axis}_parallel_group", lambda axis=axis: groups[axis], raising=False
                )
            calls = []
            original_broadcast = dist.broadcast

            def broadcast(tensor, src, group):
                calls.append(group)
                return original_broadcast(tensor, src=src, group=group)

            monkeypatch.setattr(stream_module.dist, "broadcast", broadcast)
            pp, tp, cp = coordinates
            for broadcast_pp in (True, False):
                calls.clear()
                source = torch.arange(6, dtype=torch.bfloat16).reshape(2, 3) + (0 if broadcast_pp else 100 * pp)
                expected = source.clone()
                is_src = tp == cp == 0 and (pp == 0 or not broadcast_pp)
                result = stream_module._bcast_known_tensor(
                    source if is_src else None, is_src, source.dtype, source.shape, torch.device("cpu"), broadcast_pp
                )
                assert torch.equal(result, expected)
                assert torch.equal(source, expected), "Source CPU tensor must remain unchanged"
                assert calls.count(groups["context"]) == int(tp == 0)
                assert calls.count(groups["tensor_model"]) == 1
                assert calls.count(groups["pipeline_model"]) == int(broadcast_pp)
    finally:
        dist.destroy_process_group(dist.group.WORLD)


def test_known_tensor_broadcast_cpu_processes(tmp_path):
    torch.multiprocessing.spawn(
        _known_tensor_broadcast_worker, args=(str(tmp_path / "rendezvous"),), nprocs=8, join=True
    )
