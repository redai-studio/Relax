# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Unit tests for SFT producer component (loop-only, no Ray runtime)."""

import asyncio
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import torch

from relax.engine.sft.dataset.streaming import ProcessedSample
from relax.engine.sft.runtime import resolve_sft_split_indices


@pytest.fixture(autouse=True)
def stub_processor_pool_module(monkeypatch):
    module = ModuleType("relax.utils.data.processor_pool")
    module.ProcessorPool = object
    monkeypatch.setitem(sys.modules, "relax.utils.data.processor_pool", module)


def _make_processed(idx: int = 0, n_tokens: int = 8) -> ProcessedSample:
    return ProcessedSample(
        tokens=torch.arange(n_tokens, dtype=torch.long),
        loss_mask=torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.long),
        total_length=n_tokens,
        multimodal_train_inputs=None,
        source_idx=idx,
    )


def _make_args(global_batch_size=4, max_tokens_per_gpu=128, num_rollout=1):
    return SimpleNamespace(
        global_batch_size=global_batch_size,
        max_tokens_per_gpu=max_tokens_per_gpu,
        context_parallel_size=1,
        num_rollout=num_rollout,
        prompt_data="/fake/train.jsonl",
        input_key="messages",
        label_key=None,
        multimodal_keys=None,
        metadata_key="metadata",
        tool_key=None,
        system_prompt=None,
        eval_prompt_data=None,
        eval_size=None,
        sft_prefetch_buffer_size=0,
        sft_prefetch_chunk_size=32,
        sft_prefetch_num_workers=4,
        loss_type="sft",
        tq_config=SimpleNamespace(),
        hf_checkpoint="/fake/model",
        start_rollout_id=0,
        seed=42,
        max_staleness=0,
        sft_async_prepack=False,
        custom_dataset_class_path=None,
    )


def _patch_pipeline_dependencies(monkeypatch, n_samples: int = 8):
    fake_processed = [_make_processed(i) for i in range(n_samples)]

    fake_ds = MagicMock()
    fake_ds.__len__ = MagicMock(return_value=len(fake_processed))
    fake_ds.shuffle = MagicMock(return_value=None)
    fake_ds.restrict_training_size = MagicMock(return_value=None)
    fake_ds.restrict_training_indices = MagicMock(return_value=None)
    fake_ds.stop = MagicMock(return_value=None)
    fake_ds.index_manager = SimpleNamespace(current_epoch=0)

    async def _get_batch_async(n):
        return fake_processed[:n], False

    fake_ds.get_batch_async = AsyncMock(side_effect=_get_batch_async)
    fake_ds.get_batch_in_order = MagicMock(side_effect=lambda start, n: fake_processed[start : start + n])
    fake_ds.get_batch_by_indices = MagicMock(side_effect=lambda indices: [fake_processed[i] for i in indices])

    fake_tok = MagicMock()
    fake_tok.chat_template = "{% generation %}assistant{% endgeneration %}"

    monkeypatch.setattr("relax.components.sft.SFTStreamingDataset", lambda **kw: fake_ds)
    monkeypatch.setattr("relax.components.sft.AutoTokenizer.from_pretrained", lambda *a, **kw: fake_tok)
    monkeypatch.setattr("relax.components.sft.ProcessorPool", MagicMock())
    monkeypatch.setattr("relax.components.sft._resolve_pad_token_ids_from_config", lambda *a, **kw: frozenset())
    monkeypatch.setattr("relax.components.sft.print_first_sample", lambda **kw: None)
    return fake_ds, fake_tok


def test_sft_eval_size_randomly_splits_and_restricts_the_shuffled_train_pool(monkeypatch):
    from relax.components.sft import SFT

    fake_ds, _ = _patch_pipeline_dependencies(monkeypatch, n_samples=10)
    monkeypatch.setattr("relax.components.sft.tq.init", lambda *a, **kw: None)
    monkeypatch.setattr("relax.components.sft.tq.get_client", MagicMock())

    args = _make_args(global_batch_size=2)
    args.eval_size = 0.2
    SFTCls = SFT.func_or_class
    sft = SFTCls.__new__(SFTCls)
    sft.config = args
    sft.role = "sft"
    sft.step = 0
    sft._dataset = None
    sft._eval_dataset = None
    sft._eval_indices = None
    sft._train_size = 0
    sft._tokenizer = None
    sft._processor_pool = None
    sft._logger_instance = None

    sft._init_data_pipeline()

    train_indices, eval_indices = resolve_sft_split_indices(10, 0.2, seed=args.seed)
    assert sft._train_size == 8
    assert sft._eval_indices == eval_indices
    assert eval_indices != (8, 9)
    fake_ds.restrict_training_indices.assert_called_once_with(train_indices)
    fake_ds.shuffle.assert_called_once_with(0, position=0)

    assert [sample.source_idx for sample in sft._build_eval_batches()] == list(eval_indices)
    fake_ds.get_batch_by_indices.assert_called_once_with(eval_indices)


def test_sft_component_imports_without_ray():
    from relax.components.sft import SFT  # noqa: F401


def test_sft_step_pushes_one_batch_to_tq(monkeypatch):
    from relax.components.sft import SFT

    _patch_pipeline_dependencies(monkeypatch)

    fake_client = MagicMock()
    fake_client.async_put = AsyncMock(return_value=None)
    monkeypatch.setattr("relax.components.sft.tq.init", lambda *a, **kw: None, raising=False)
    monkeypatch.setattr("relax.components.sft.tq.get_client", lambda: fake_client, raising=False)

    args = _make_args(global_batch_size=4)
    SFTCls = SFT.func_or_class
    sft = SFTCls.__new__(SFTCls)
    sft.config = args
    sft.role = "sft"
    sft._healthy = True
    sft.step = 0
    sft.data_system_client = fake_client
    sft._dataset = None
    sft._eval_dataset = None
    sft._eval_indices = None
    sft._train_size = 0
    sft._tokenizer = None
    sft._processor_pool = None
    sft._logger_instance = None
    sft._stop_event = MagicMock()
    sft._stop_event.is_set = MagicMock(return_value=False)

    sft._init_data_pipeline()
    asyncio.run(sft._produce_one_step())
    assert fake_client.async_put.await_count == 1
    args_call, kwargs_call = fake_client.async_put.call_args
    pushed_data = kwargs_call.get("data")
    assert "tokens" in pushed_data
    assert "loss_masks" in pushed_data
    assert "total_lengths" in pushed_data
    assert "response_lengths" in pushed_data
    assert kwargs_call.get("partition_id") == "sft_0"
    assert kwargs_call.get("custom_meta") == [{"total_lengths": 8}] * 4


def test_sft_step_pushes_sharded_batches_to_tq(monkeypatch):
    from relax.components.sft import SFT, _sft_train_partitions_in_flight

    _patch_pipeline_dependencies(monkeypatch)
    monkeypatch.setenv("RELAX_SFT_TQ_SHARDS", "2")
    monkeypatch.setattr("relax.components.sft.ray.is_initialized", lambda: False)

    fake_client = MagicMock()
    fake_client.async_put = AsyncMock(return_value=None)
    monkeypatch.setattr("relax.components.sft.tq.init", lambda *a, **kw: None)
    monkeypatch.setattr("relax.components.sft.tq.get_client", lambda: fake_client)

    args = _make_args(global_batch_size=4)
    args.sft_async_prepack = True
    SFTCls = SFT.func_or_class
    sft = SFTCls.__new__(SFTCls)
    sft.config = args
    sft.role = "sft"
    sft._healthy = True
    sft.step = 0
    sft.data_system_client = fake_client
    sft._dataset = None
    sft._eval_dataset = None
    sft._eval_indices = None
    sft._train_size = 0
    sft._tokenizer = None
    sft._processor_pool = None
    sft._logger_instance = None
    sft._stop_event = MagicMock()
    sft._stop_event.is_set = MagicMock(return_value=False)
    sft._runtime_env = None

    sft._init_data_pipeline()
    asyncio.run(sft._produce_one_step())

    assert fake_client.async_put.await_count == 2
    seen_partitions = [c.kwargs.get("partition_id") for c in fake_client.async_put.call_args_list]
    assert seen_partitions == ["sft_0_shard_0_of_2", "sft_0_shard_1_of_2"]
    seen_meta = [c.kwargs.get("custom_meta") for c in fake_client.async_put.call_args_list]
    assert seen_meta == [[{"total_lengths": 8}] * 2, [{"total_lengths": 8}] * 2]
    assert _sft_train_partitions_in_flight(seen_partitions) == 1


@pytest.mark.parametrize("returned_count", [0, 3])
def test_sft_step_rejects_empty_or_partial_batch(monkeypatch, returned_count):
    from relax.components.sft import SFT

    fake_ds, _ = _patch_pipeline_dependencies(monkeypatch)
    fake_ds.get_batch_async = AsyncMock(return_value=([_make_processed(i) for i in range(returned_count)], False))

    fake_client = MagicMock()
    fake_client.async_put = AsyncMock(return_value=None)
    monkeypatch.setattr("relax.components.sft.tq.init", lambda *a, **kw: None, raising=False)
    monkeypatch.setattr("relax.components.sft.tq.get_client", lambda: fake_client, raising=False)

    args = _make_args(global_batch_size=4)
    SFTCls = SFT.func_or_class
    sft = SFTCls.__new__(SFTCls)
    sft.config = args
    sft.role = "sft"
    sft._healthy = True
    sft.step = 0
    sft.data_system_client = fake_client
    sft._dataset = None
    sft._eval_dataset = None
    sft._eval_indices = None
    sft._train_size = 0
    sft._tokenizer = None
    sft._processor_pool = None
    sft._logger_instance = None
    sft._stop_event = MagicMock()
    sft._stop_event.is_set = MagicMock(return_value=False)
    sft._init_data_pipeline()

    with pytest.raises(RuntimeError, match=rf"dataset returned {returned_count}/4 samples"):
        asyncio.run(sft._produce_one_step())

    fake_client.async_put.assert_not_awaited()
    assert sft.step == 0


def test_sft_eval_rejects_source_with_no_valid_samples(monkeypatch):
    from relax.components.sft import SFT

    _patch_pipeline_dependencies(monkeypatch)
    fake_client = MagicMock()
    fake_client.async_put = AsyncMock(return_value=None)
    monkeypatch.setattr("relax.components.sft.tq.init", lambda *a, **kw: None, raising=False)
    monkeypatch.setattr("relax.components.sft.tq.get_client", lambda: fake_client, raising=False)

    args = _make_args(global_batch_size=4)
    args.eval_interval = 1
    SFTCls = SFT.func_or_class
    sft = SFTCls.__new__(SFTCls)
    sft.config = args
    sft.role = "sft"
    sft._healthy = True
    sft.step = 0
    sft.data_system_client = fake_client
    sft._dataset = MagicMock()
    sft._eval_dataset = MagicMock()
    sft._eval_dataset.get_batch_in_order.return_value = []
    sft._eval_indices = None
    sft._logger_instance = None
    sft._stop_event = MagicMock()
    sft._stop_event.is_set = MagicMock(return_value=False)

    with pytest.raises(RuntimeError, match="source produced 0 valid samples"):
        asyncio.run(sft._maybe_produce_eval())

    fake_client.async_put.assert_not_awaited()


@pytest.mark.parametrize("n_real", [1, 3, 4, 5, 8])
def test_classification_eval_pads_without_dropping_real_samples(n_real):
    from relax.components.sft import SFT

    samples = [
        ProcessedSample(
            tokens=torch.tensor([idx + 1, 99], dtype=torch.long),
            loss_mask=torch.ones(1, dtype=torch.long),
            total_length=2,
            multimodal_train_inputs=None,
            source_idx=idx,
            classification_label=torch.tensor(idx % 2, dtype=torch.long),
        )
        for idx in range(n_real)
    ]
    fake_client = MagicMock()
    fake_client.async_put = AsyncMock(return_value=None)

    SFTCls = SFT.func_or_class
    sft = SFTCls.__new__(SFTCls)
    sft.config = SimpleNamespace(
        eval_interval=1,
        global_batch_size=4,
        task_type="seq_cls",
        multimodal_keys=None,
        sft_eval_chunk_drain_timeout_sec=1,
    )
    sft.step = 0
    sft.data_system_client = fake_client
    sft._logger_instance = MagicMock()
    sft._build_eval_batches = MagicMock(return_value=samples)
    sft._wait_for_partition_drained = AsyncMock(return_value=True)

    asyncio.run(sft._maybe_produce_eval())

    expected_chunks = (n_real + 3) // 4
    assert fake_client.async_put.await_count == expected_chunks
    weights = torch.cat([call.kwargs["data"]["sample_weights"] for call in fake_client.async_put.call_args_list])
    assert weights.tolist() == [1.0] * n_real + [0.0] * (expected_chunks * 4 - n_real)
    partition_ids = [call.kwargs["partition_id"] for call in fake_client.async_put.call_args_list]
    assert partition_ids == [f"sft_eval_0_n{expected_chunks}_{idx}" for idx in range(expected_chunks)]


def test_sft_loop_advances_step(monkeypatch):
    from relax.components.sft import SFT

    _patch_pipeline_dependencies(monkeypatch)

    fake_client = MagicMock()
    fake_client.async_put = AsyncMock(return_value=None)
    fake_client.async_get_partition_list = AsyncMock(return_value=[])
    monkeypatch.setattr("relax.components.sft.tq.init", lambda *a, **kw: None, raising=False)
    monkeypatch.setattr("relax.components.sft.tq.get_client", lambda: fake_client, raising=False)

    args = _make_args(global_batch_size=2, num_rollout=3)
    SFTCls = SFT.func_or_class
    sft = SFTCls.__new__(SFTCls)
    sft.config = args
    sft.role = "sft"
    sft._healthy = True
    sft.step = 0
    sft.data_system_client = fake_client
    sft._dataset = None
    sft._eval_dataset = None
    sft._eval_indices = None
    sft._train_size = 0
    sft._tokenizer = None
    sft._processor_pool = None
    sft._logger_instance = None
    sft._stop_event = MagicMock()
    sft._stop_event.is_set = MagicMock(return_value=False)
    sft._init_data_pipeline()

    for _ in range(3):
        asyncio.run(sft._produce_one_step())
    assert sft.step == 3
    assert fake_client.async_put.await_count == 3
    seen_partitions = [c.kwargs.get("partition_id") for c in fake_client.async_put.call_args_list]
    assert seen_partitions == ["sft_0", "sft_1", "sft_2"]


def test_sft_resume_only_produces_remaining_steps():
    from relax.components.sft import SFT

    SFTCls = SFT.func_or_class
    sft = SFTCls.__new__(SFTCls)
    sft.config = _make_args(num_rollout=5)
    sft.step = 2
    sft._stop_event = MagicMock()
    sft._stop_event.is_set = MagicMock(return_value=False)

    async def _produce_one_step():
        sft.step += 1

    sft._produce_one_step = AsyncMock(side_effect=_produce_one_step)

    asyncio.run(sft._async_run())

    assert sft.step == 5
    assert sft._produce_one_step.await_count == 3


@pytest.mark.parametrize("start_step", [5, 6])
def test_sft_resume_at_or_after_end_produces_nothing(start_step):
    from relax.components.sft import SFT

    SFTCls = SFT.func_or_class
    sft = SFTCls.__new__(SFTCls)
    sft.config = _make_args(num_rollout=5)
    sft.step = start_step
    sft._stop_event = MagicMock()
    sft._stop_event.is_set = MagicMock(return_value=False)
    sft._produce_one_step = AsyncMock()

    asyncio.run(sft._async_run())

    sft._produce_one_step.assert_not_awaited()
