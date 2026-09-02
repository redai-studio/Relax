# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Unit tests for SFT producer component (loop-only, no Ray runtime)."""

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


@pytest.mark.asyncio
async def test_sft_step_pushes_one_batch_to_tq(monkeypatch):
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
    await sft._produce_one_step()
    assert fake_client.async_put.await_count == 1
    args_call, kwargs_call = fake_client.async_put.call_args
    pushed_data = kwargs_call.get("data")
    assert "tokens" in pushed_data
    assert "loss_masks" in pushed_data
    assert "total_lengths" in pushed_data
    assert "response_lengths" in pushed_data
    assert kwargs_call.get("partition_id") == "sft_0"
    assert kwargs_call.get("custom_meta") == [{"total_lengths": 8}] * 4


@pytest.mark.asyncio
async def test_sft_step_pushes_sharded_batches_to_tq(monkeypatch):
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
    await sft._produce_one_step()

    assert fake_client.async_put.await_count == 2
    seen_partitions = [c.kwargs.get("partition_id") for c in fake_client.async_put.call_args_list]
    assert seen_partitions == ["sft_0_shard_0_of_2", "sft_0_shard_1_of_2"]
    seen_meta = [c.kwargs.get("custom_meta") for c in fake_client.async_put.call_args_list]
    assert seen_meta == [[{"total_lengths": 8}] * 2, [{"total_lengths": 8}] * 2]
    assert _sft_train_partitions_in_flight(seen_partitions) == 1


@pytest.mark.parametrize(
    ("attribute", "value"),
    [
        ("task_type", "seq_cls"),
    ],
)
def test_sft_remote_batch_producer_falls_back_for_target_only_modes(monkeypatch, attribute, value):
    from relax.components.sft import SFT

    monkeypatch.setenv("RELAX_SFT_TQ_SHARDS", "2")
    monkeypatch.setattr("relax.components.sft.ray.is_initialized", lambda: True)

    args = _make_args(global_batch_size=4)
    args.sft_async_prepack = True
    setattr(args, attribute, value)
    SFTCls = SFT.func_or_class
    sft = SFTCls.__new__(SFTCls)
    sft.config = args
    sft._logger_instance = MagicMock()

    assert sft._should_use_remote_batch_producer() is False


def test_sft_remote_batch_producer_allows_eval_size(monkeypatch):
    from relax.components.sft import SFT

    monkeypatch.setenv("RELAX_SFT_TQ_SHARDS", "2")
    monkeypatch.setattr("relax.components.sft.ray.is_initialized", lambda: True)

    args = _make_args(global_batch_size=4)
    args.sft_async_prepack = True
    args.eval_size = 0.25
    SFTCls = SFT.func_or_class
    sft = SFTCls.__new__(SFTCls)
    sft.config = args
    sft._logger_instance = MagicMock()

    assert sft._should_use_remote_batch_producer() is True


def test_sft_remote_batch_producer_resolves_model_on_its_node(monkeypatch):
    from relax.components import sft as sft_module

    call_order = []
    fake_client = MagicMock()
    fake_tokenizer = MagicMock()

    def _prepare_model(config, *, completeness):
        call_order.append("prepare")
        assert completeness == "metadata"
        config.hf_checkpoint = "/dev/shm/resolved-sft-model"

    def _load_tokenizer(path, **kwargs):
        call_order.append("tokenizer")
        assert path == "/dev/shm/resolved-sft-model"
        assert kwargs == {"trust_remote_code": True}
        return fake_tokenizer

    class _FakeIndexManager:
        total_size = 4

        def __init__(self):
            self.current_epoch = -1
            self.position = 0
            self.indices = list(range(self.total_size))

        def shuffle(self, epoch):
            self.current_epoch = epoch
            self.position = 0
            self.indices = list(range(self.total_size))

    fake_dataset = MagicMock()
    fake_dataset.__len__ = MagicMock(return_value=4)
    fake_dataset.index_manager = _FakeIndexManager()
    fake_dataset._prefetch = None

    monkeypatch.setattr(sft_module.tq, "init", MagicMock())
    monkeypatch.setattr(sft_module.tq, "get_client", MagicMock(return_value=fake_client))
    monkeypatch.setattr(sft_module, "prepare_model_maybe_update_args", _prepare_model)
    monkeypatch.setattr(sft_module.AutoTokenizer, "from_pretrained", _load_tokenizer)
    monkeypatch.setattr(sft_module, "ProcessorPool", MagicMock())
    monkeypatch.setattr(sft_module, "_resolve_pad_token_ids_from_config", MagicMock(return_value=frozenset()))
    create_dataset = MagicMock(return_value=fake_dataset)
    monkeypatch.setattr(sft_module, "_create_sft_train_dataset", create_dataset)

    args = _make_args(global_batch_size=2)
    producer_cls = sft_module._SFTBatchProducerActor.__ray_metadata__.modified_class
    producer = producer_cls(args, shard_id=0, num_shards=2, prefetch_num_workers=1)

    state = producer.initialize(start_step=0)

    assert call_order == ["prepare", "tokenizer"]
    assert args.hf_checkpoint == "/dev/shm/resolved-sft-model"
    assert state["train_size"] == 4
    assert create_dataset.call_args.kwargs["task_type"] == "causal_lm"


def test_sft_remote_batch_producer_restricts_train_pool_for_eval_size(monkeypatch):
    from relax.components import sft as sft_module

    class _FakeIndexManager:
        total_size = 10

        def __init__(self):
            self.current_epoch = -1
            self.position = 0
            self.indices = list(range(self.total_size))

        def shuffle(self, epoch):
            self.current_epoch = epoch
            self.position = 0
            self.indices = list(range(self.total_size))

    fake_dataset = MagicMock()
    fake_dataset.__len__ = MagicMock(return_value=10)
    fake_dataset.index_manager = _FakeIndexManager()
    fake_dataset._prefetch = None

    monkeypatch.setattr(sft_module.tq, "init", MagicMock())
    monkeypatch.setattr(sft_module.tq, "get_client", MagicMock())
    monkeypatch.setattr(sft_module, "prepare_model_maybe_update_args", MagicMock())
    monkeypatch.setattr(sft_module.AutoTokenizer, "from_pretrained", MagicMock(return_value=MagicMock()))
    monkeypatch.setattr(sft_module, "ProcessorPool", MagicMock())
    monkeypatch.setattr(sft_module, "_resolve_pad_token_ids_from_config", MagicMock(return_value=frozenset()))
    monkeypatch.setattr(sft_module, "_create_sft_train_dataset", MagicMock(return_value=fake_dataset))

    args = _make_args(global_batch_size=2)
    args.eval_size = 0.2
    producer_cls = sft_module._SFTBatchProducerActor.__ray_metadata__.modified_class
    producer = producer_cls(args, shard_id=0, num_shards=2, prefetch_num_workers=1)

    state = producer.initialize(start_step=0)

    train_indices, _eval_indices = resolve_sft_split_indices(10, 0.2, seed=args.seed)
    assert state["train_size"] == 8
    fake_dataset.restrict_training_indices.assert_called_once_with(train_indices)


@pytest.mark.asyncio
async def test_sft_remote_batch_producer_reprimes_before_second_step(monkeypatch):
    from relax.components import sft as sft_module

    class _FakeIndexManager:
        total_size = 16

        def __init__(self):
            self.current_epoch = 0
            self.position = 0
            self.indices = list(range(self.total_size))

        def shuffle(self, epoch):
            if epoch != self.current_epoch:
                self.current_epoch = epoch
                self.position = 0
                self.indices = list(range(self.total_size))

    class _FakeDataset:
        def __init__(self):
            self.index_manager = _FakeIndexManager()
            self._prefetch = MagicMock()

        async def get_batch_async(self, batch_size):
            start = self.index_manager.position
            self.index_manager.position += batch_size
            return [_make_processed(idx) for idx in range(start, start + batch_size)], False

    monkeypatch.setattr(sft_module, "print_first_sample", MagicMock())
    args = _make_args(global_batch_size=4)
    producer_cls = sft_module._SFTBatchProducerActor.__ray_metadata__.modified_class
    producer = producer_cls(args, shard_id=0, num_shards=2, prefetch_num_workers=1)
    producer._dataset = _FakeDataset()
    producer._tokenizer = MagicMock()
    producer.data_system_client = MagicMock()
    producer.data_system_client.async_put = AsyncMock()
    producer._train_size = 16

    await producer.produce_partition(0, "sft_0_shard_0_of_2", 4, False)
    producer._dataset._prefetch.set_index_order.assert_not_called()

    await producer.produce_partition(1, "sft_1_shard_0_of_2", 4, False)
    producer._dataset._prefetch.set_index_order.assert_called_once_with([4, 5])
    assert producer.data_system_client.async_put.await_count == 2


def test_sft_remote_eval_size_initializes_eval_dataset_without_train_overlap(monkeypatch):
    from relax.components.sft import SFT

    fake_ds, _ = _patch_pipeline_dependencies(monkeypatch, n_samples=10)
    monkeypatch.setenv("RELAX_SFT_TQ_SHARDS", "2")
    monkeypatch.setattr("relax.components.sft.ray.is_initialized", lambda: True)
    monkeypatch.setattr("relax.components.sft.tq.init", lambda *a, **kw: None)
    monkeypatch.setattr("relax.components.sft.tq.get_client", MagicMock())

    args = _make_args(global_batch_size=2)
    args.sft_async_prepack = True
    args.eval_size = 0.2
    SFTCls = SFT.func_or_class
    sft = SFTCls.__new__(SFTCls)
    sft.config = args
    sft.role = "sft"
    sft.step = 0
    sft._dataset = None
    sft._eval_dataset = None
    sft._eval_indices = None
    sft._batch_producers = []
    sft._train_size = 0
    sft._tokenizer = None
    sft._processor_pool = None
    sft._logger_instance = MagicMock()
    sft._runtime_env = None
    sft._init_remote_batch_producers = MagicMock(side_effect=lambda: setattr(sft, "_train_size", 8))

    sft._init_data_pipeline()

    train_indices, eval_indices = resolve_sft_split_indices(10, 0.2, seed=args.seed)
    assert sft._train_size == len(train_indices)
    assert sft._eval_indices == eval_indices
    fake_ds.restrict_training_indices.assert_not_called()
    assert [sample.source_idx for sample in sft._build_eval_batches()] == list(eval_indices)


@pytest.mark.asyncio
async def test_sft_remote_step_produces_eval_before_advancing_step(monkeypatch):
    from relax.components import sft as sft_module
    from relax.components.sft import SFT

    class _RemoteProduce:
        def __init__(self):
            self.calls = []

        def remote(self, step, partition_id, global_batch_size, force_multimodal_field):
            self.calls.append((step, partition_id, global_batch_size, force_multimodal_field))
            return object()

    class _RemoteProducer:
        def __init__(self):
            self.produce_partition = _RemoteProduce()

    monkeypatch.setenv("RELAX_SFT_TQ_SHARDS", "2")
    monkeypatch.setattr(
        sft_module,
        "_ray_get_many_async",
        AsyncMock(
            return_value=[
                {"crossed_epoch": False, "epoch": 0},
                {"crossed_epoch": False, "epoch": 0},
            ]
        ),
    )

    args = _make_args(global_batch_size=4)
    args.sft_async_prepack = True
    SFTCls = SFT.func_or_class
    sft = SFTCls.__new__(SFTCls)
    sft.config = args
    sft.role = "sft"
    sft.step = 0
    sft.data_system_client = MagicMock()
    sft._dataset = None
    sft._eval_dataset = MagicMock()
    sft._eval_indices = None
    sft._batch_producers = [_RemoteProducer(), _RemoteProducer()]
    sft._train_size = 8
    sft._tokenizer = None
    sft._processor_pool = None
    sft._logger_instance = MagicMock()
    sft._stop_event = MagicMock()
    sft._stop_event.is_set = MagicMock(return_value=False)

    async def _maybe_produce_eval():
        assert sft.step == 0

    sft._maybe_produce_eval = AsyncMock(side_effect=_maybe_produce_eval)

    await sft._produce_one_step()

    sft._maybe_produce_eval.assert_awaited_once()
    assert sft.step == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("returned_count", [0, 3])
async def test_sft_step_rejects_empty_or_partial_batch(monkeypatch, returned_count):
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
        await sft._produce_one_step()

    fake_client.async_put.assert_not_awaited()
    assert sft.step == 0


@pytest.mark.asyncio
async def test_sft_eval_rejects_source_with_no_valid_samples(monkeypatch):
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
        await sft._maybe_produce_eval()

    fake_client.async_put.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("n_real", [1, 3, 4, 5, 8])
async def test_classification_eval_pads_without_dropping_real_samples(n_real):
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

    await sft._maybe_produce_eval()

    expected_chunks = (n_real + 3) // 4
    assert fake_client.async_put.await_count == expected_chunks
    weights = torch.cat([call.kwargs["data"]["sample_weights"] for call in fake_client.async_put.call_args_list])
    assert weights.tolist() == [1.0] * n_real + [0.0] * (expected_chunks * 4 - n_real)
    partition_ids = [call.kwargs["partition_id"] for call in fake_client.async_put.call_args_list]
    assert partition_ids == [f"sft_eval_0_n{expected_chunks}_{idx}" for idx in range(expected_chunks)]


@pytest.mark.asyncio
async def test_sft_loop_advances_step(monkeypatch):
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
        await sft._produce_one_step()
    assert sft.step == 3
    assert fake_client.async_put.await_count == 3
    seen_partitions = [c.kwargs.get("partition_id") for c in fake_client.async_put.call_args_list]
    assert seen_partitions == ["sft_0", "sft_1", "sft_2"]


@pytest.mark.asyncio
async def test_sft_resume_only_produces_remaining_steps():
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

    await sft._async_run()

    assert sft.step == 5
    assert sft._produce_one_step.await_count == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("start_step", [5, 6])
async def test_sft_resume_at_or_after_end_produces_nothing(start_step):
    from relax.components.sft import SFT

    SFTCls = SFT.func_or_class
    sft = SFTCls.__new__(SFTCls)
    sft.config = _make_args(num_rollout=5)
    sft.step = start_step
    sft._stop_event = MagicMock()
    sft._stop_event.is_set = MagicMock(return_value=False)
    sft._produce_one_step = AsyncMock()

    await sft._async_run()

    sft._produce_one_step.assert_not_awaited()
