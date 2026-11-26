# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from relax.distributed.replay_manager.config import ReplayConfig, validate_reference_training
from relax.distributed.replay_manager.consumer import ConsumerBatchLayout, ConsumerLayerReader, ConsumerTensorCache
from relax.distributed.replay_manager.core import ReplayInventory, ReplayManager
from relax.distributed.replay_manager.storage import TQReplayStorage, _clear_simple_storage_samples


def test_debug_dump_replay_uses_self_contained_fields(tmp_path):
    from relax.utils.training.train_dump_utils import load_debug_rollout_samples
    from relax.utils.types import Sample

    path = tmp_path / "rollout.pt"
    sample = Sample(tokens=[1, 2], rollout_replay_refs={"moe": {"ref": "old-run"}})
    torch.save({"samples": [sample.to_dict()]}, path)
    [restored] = load_debug_rollout_samples(path, SimpleNamespace())
    assert restored.tokens == [1, 2] and not restored.rollout_replay_refs
    with pytest.raises(ValueError, match="moe.*another run"):
        load_debug_rollout_samples(path, SimpleNamespace(use_rollout_routing_replay=True))

    sample.rollout_replay_refs = {"mm_embedding": {"media": [{"modality": "video"}]}}
    sample.multimodal_train_inputs = {"pixel_values": torch.ones(2)}
    torch.save({"samples": [sample.to_dict()]}, path)
    with pytest.raises(ValueError, match="lacks pixel_values_videos"):
        load_debug_rollout_samples(path, SimpleNamespace())


class MemoryBody:
    """Replace remote body I/O; publication, ownership and reads stay real."""

    def __init__(self):
        self.blocks = {}
        self.reads = 0

    def write_many(self, keys, values):
        self.blocks.update((key, value.copy()) for key, value in zip(keys, values, strict=True))

    def read(self, keys):
        self.reads += len(keys)
        return [self.blocks[key].copy() for key in keys]

    def delete(self, keys):
        for key in keys:
            self.blocks.pop(key, None)

    def close(self):
        pass


@pytest.mark.parametrize("configured", [None, "0"])
def test_cli_reference_enable_preserves_manager_and_retires_after_successful_tq_clear(monkeypatch, configured):
    from unittest.mock import AsyncMock

    from relax.distributed import replay_manager
    from relax.distributed.replay_manager import core
    from relax.utils.tq.lifecycle import clear_training_partition

    monkeypatch.setenv("RELAX_REPLAY_REFERENCES", configured or "0")
    monkeypatch.setenv("RELAX_REPLAY_RUN_ID", "")
    if configured is None:
        monkeypatch.delenv("RELAX_REPLAY_REFERENCES")
    config = ReplayConfig(delivery="full", run_id="abcd", capacity_bytes=123456)
    store = SimpleNamespace(invoke_async=AsyncMock())
    manager = ReplayManager(config=config, store_factory=lambda: store)
    monkeypatch.setattr(core, "replay_data_manager", manager)
    monkeypatch.setattr(replay_manager, "replay_data_manager", manager)
    core.enable_reference_replay()
    assert core.replay_data_manager is manager and manager.uses_references
    assert manager.config.run_id == config.run_id and manager.config.capacity_bytes == config.capacity_bytes
    assert os.environ["RELAX_REPLAY_RUN_ID"] == config.run_id

    client = SimpleNamespace(async_clear_partition=AsyncMock(side_effect=[RuntimeError("TQ clear failed"), 1]))
    with pytest.raises(RuntimeError, match="TQ clear failed"):
        asyncio.run(clear_training_partition(client, "train_0"))
    store.invoke_async.assert_not_awaited()
    assert asyncio.run(clear_training_partition(client, "train_0")) == 1
    store.invoke_async.assert_awaited_once_with("retire_partition", "train_0")


def test_rollout_staleness_counts_only_pending_training_steps():
    from relax.components.rollout import satisfy_staleness
    from relax.distributed.replay_manager.storage import BODY_PARTITION_PREFIX

    other = [f"{BODY_PARTITION_PREFIX}abcd", "sft_0"]
    assert satisfy_staleness(other, current_rollout_id=6, max_staleness=2)
    assert satisfy_staleness([*other, "train_5"], current_rollout_id=6, max_staleness=2)
    assert not satisfy_staleness([*other, "train_5", "train_4"], current_rollout_id=6, max_staleness=2)


def test_cli_reference_enable_refuses_initialized_full_manager_without_mutation(monkeypatch):
    from relax.distributed.replay_manager import core

    monkeypatch.setenv("RELAX_REPLAY_REFERENCES", "0")
    monkeypatch.setenv("RELAX_REPLAY_RUN_ID", "ffff")
    config = ReplayConfig(delivery="full", run_id="abcd")
    manager = ReplayManager(config=config)
    manager.initialize(kinds=["moe"])
    monkeypatch.setattr(core, "replay_data_manager", manager)
    with pytest.raises(RuntimeError, match="initialization"):
        core.enable_reference_replay()
    assert manager.config is config
    assert not manager.uses_references
    assert os.environ["RELAX_REPLAY_REFERENCES"] == "0"
    assert os.environ["RELAX_REPLAY_RUN_ID"] == "ffff"


@pytest.mark.parametrize("delivery", ["full", "reference"])
def test_cli_reference_enable_generates_storage_identity_before_initialization(monkeypatch, delivery):
    from relax.distributed.replay_manager import core

    monkeypatch.setenv("RELAX_REPLAY_REFERENCES", "0")
    monkeypatch.setenv("RELAX_REPLAY_RUN_ID", "")
    manager = ReplayManager(config=ReplayConfig(delivery=delivery, run_id="", capacity_bytes=123456))
    monkeypatch.setattr(core, "replay_data_manager", manager)
    early_import = manager
    core.enable_reference_replay()
    identity = manager.config.run_id
    assert identity
    assert core.replay_data_manager is early_import and manager.uses_references
    assert os.environ["RELAX_REPLAY_RUN_ID"] == identity
    assert os.environ["RELAX_REPLAY_REFERENCES"] == "1"
    assert manager.config.capacity_bytes == 123456
    worker_config = ReplayConfig.from_env()
    assert worker_config.delivery == "reference" and worker_config.run_id == identity
    # Execute the same constructor that rejected an empty/non-hex ID in the
    # driver, replacing only the external registry to avoid starting Ray.
    store = TQReplayStorage(manager.config, inventory=ReplayInventory(manager.config, lambda _: None))
    assert store.run == identity
    manager.initialize(kinds=["moe"])
    core.enable_reference_replay()
    assert manager.config.run_id == identity
    assert os.environ["RELAX_REPLAY_RUN_ID"] == identity


def test_debug_dump_uses_full_routes_and_rejects_explicit_references(monkeypatch):
    args = SimpleNamespace(
        train_backend="megatron", colocate=True, use_rollout_routing_replay=True, load_debug_rollout_data="dump.pt"
    )
    monkeypatch.setenv("RELAX_REPLAY_REFERENCES", "0")
    validate_reference_training(args)
    monkeypatch.setenv("RELAX_REPLAY_REFERENCES", "1")
    with pytest.raises(ValueError, match="--load-debug-rollout-data.*RELAX_REPLAY_REFERENCES=0"):
        validate_reference_training(args)


@pytest.mark.parametrize("mode", ["colocate", "hybrid", "fully_async", "separate", "fsdp"])
def test_reference_training_rejects_unwired_consumers_before_startup(monkeypatch, mode):
    monkeypatch.setenv("RELAX_REPLAY_REFERENCES", "1")
    args = SimpleNamespace(
        train_backend="fsdp" if mode == "fsdp" else "megatron",
        colocate=mode == "colocate",
        hybrid=mode == "hybrid",
        fully_async=mode == "fully_async",
        use_rollout_routing_replay=True,
        use_rollout_indexer_replay=False,
    )
    if mode in {"colocate", "hybrid", "fully_async"}:
        validate_reference_training(args)
    else:
        with pytest.raises(ValueError, match="colocated Megatron"):
            validate_reference_training(args)
    args.use_rollout_routing_replay = False
    args.use_rollout_indexer_replay = False
    validate_reference_training(args)


@pytest.mark.parametrize(
    ("field", "reason"),
    [("dynamic_context_parallel", "dynamic context"), ("overlap_moe_expert_parallel_comm", "combined 1F1B")],
)
def test_reference_r3_rejects_unsupported_router_layout(monkeypatch, field, reason):
    monkeypatch.setenv("RELAX_REPLAY_REFERENCES", "1")
    args = SimpleNamespace(train_backend="megatron", fully_async=True, use_rollout_routing_replay=True)
    setattr(args, field, True)
    with pytest.raises(ValueError, match=reason):
        validate_reference_training(args)


@pytest.fixture
def published():
    config = ReplayConfig(delivery="reference", run_id="a", chunk_rows=2, publication_bytes=64)
    body = MemoryBody()
    inventory = ReplayInventory(config, body.delete)
    store = TQReplayStorage(config, inventory=inventory)
    store._backend = body
    values = np.arange(4 * 3 * 2, dtype=np.int32).reshape(4, 3, 2)
    publication = store.begin_publication(kind="moe", layers=3, topk=2)
    for start in (0, 2):
        publication.append(2, lambda layer, start=start: values[start : start + 2, layer].copy())
    ref = publication.finish([10, 11, 12, 13, 14])
    yield config, store, body, inventory, ref, values
    store._writer.close()


@pytest.mark.parametrize("dump_kind", ["rollout", "train"])
async def test_debug_dump_materializes_routes_for_another_run(published, monkeypatch, tmp_path, dump_kind):
    from relax.distributed import replay_manager
    from relax.utils.training import train_dump_utils
    from relax.utils.types import Sample

    config, store, _, inventory, ref, expected = published
    manager = ReplayManager(config=config, store_factory=lambda: store)
    monkeypatch.setattr(replay_manager, "replay_data_manager", manager)
    sample = Sample(tokens=[10, 11, 12, 13, 14], rollout_replay_refs={"moe": ref.to_dict()})
    path = tmp_path / "replay.pt"
    args = SimpleNamespace(save_debug_rollout_data=str(path), save_debug_train_data=str(path))
    if dump_kind == "rollout":
        from relax.utils import utils

        monkeypatch.setattr(utils, "replay_data_manager", manager)

        async def consume(data, partition_id, **kwargs):
            assert "rollout_routed_experts" not in data
            await manager.on_partition_cleared(partition_id)

        batch = {"total_lengths": [5], "sample_indices": [0], "rollout_replay_refs": [sample.rollout_replay_refs]}
        await utils.put_rollout_batch(args, SimpleNamespace(async_put=consume), "train_0", [sample], batch, True)
        assert not inventory.state["objects"]
        train_dump_utils.save_debug_rollout_data(args, [sample], 0, False)
    else:
        monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
        train_dump_utils.save_debug_train_data(
            args, rollout_id=0, rollout_data={key: [value] for key, value in sample.to_dict().items()}
        )
        assert sample.rollout_routed_experts is None
        assert not inventory.state["objects"][ref.record]["readers"]
        store.discard([ref])
    assert sample.rollout_replay_refs == {"moe": ref.to_dict()}
    assert not inventory.state["objects"]
    monkeypatch.setattr(replay_manager, "replay_data_manager", ReplayManager(config=ReplayConfig(run_id="other")))
    for replay in (False, True):
        [restored] = train_dump_utils.load_debug_rollout_samples(
            str(path), SimpleNamespace(use_rollout_routing_replay=replay)
        )
        np.testing.assert_array_equal(restored.rollout_routed_experts, expected)
        assert restored.tokens == sample.tokens
        assert not restored.rollout_replay_refs


async def test_native_cancel_reclaims_exclusive_route_while_peer_commits(published, native_write, monkeypatch):
    from dataclasses import replace
    from unittest.mock import AsyncMock

    from relax.utils.utils import transfer_batch_to_data_system

    _, store, body, inventory, shared, _ = published
    args, sample, producer = native_write
    await producer.adopt_native_response(sample, args, {"routed_experts": shared.to_dict()})
    publication = store.begin_publication(kind="moe", layers=1, topk=1)
    publication.append(2, lambda _: np.zeros((2, 1), dtype=np.int32))
    other = publication.finish([1, 2, 3])
    exclusive = replace(sample, index=1, tokens=[1, 2, 3], rollout_replay_refs={"moe": other.to_dict()})
    await producer.adopt_native_response(exclusive, args, {"routed_experts": other.to_dict()})
    attached, acknowledge = asyncio.Event(), asyncio.Event()
    invoke = store.invoke_async

    async def delay_ack(method, *args, **kwargs):
        result = await invoke(method, *args, **kwargs)
        if method == "attach":
            attached.set()
            await acknowledge.wait()
        return result

    monkeypatch.setattr(store, "invoke_async", delay_ack)
    client = SimpleNamespace(async_put=AsyncMock())
    task = asyncio.create_task(transfer_batch_to_data_system(args, [[sample, exclusive]], 0, 0, client))
    await asyncio.wait_for(attached.wait(), 5)
    store.attach([shared], "train_0")
    store.commit([shared])
    task.cancel()
    acknowledge.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    client.async_put.assert_not_awaited()
    assert set(inventory.state["objects"]) == {shared.record}
    assert inventory.state["objects"][shared.record]["partitions"] == ["train_0"]
    store.retire_partition("train_0")
    assert not body.blocks


@pytest.fixture
def native_write(published, monkeypatch):
    from relax.utils import utils
    from relax.utils.types import Sample

    config, store, _, _, ref, _ = published
    manager = ReplayManager(config=config, store_factory=lambda: store)
    monkeypatch.setattr(utils, "replay_data_manager", manager)
    monkeypatch.setattr(utils, "CURRENT_ROLLOUT_BATCH", [])
    sample = Sample(
        index=0,
        group_index=0,
        session_id="native",
        tokens=[10, 11, 12, 13, 14],
        response_length=1,
        reward=1.0,
        rollout_replay_refs={"moe": ref.to_dict()},
    )
    args = SimpleNamespace(
        custom_convert_samples_to_train_data_path=None,
        use_rollout_indexer_replay=False,
        custom_reward_post_process_path=None,
        reward_key=None,
        rewards_normalization=False,
        use_rollout_routing_replay=True,
        multimodal_keys=None,
        use_opd=False,
        debug_train_only=False,
        save_debug_rollout_data=None,
    )
    return args, sample, manager.register_producer({"moe"})


@pytest.mark.parametrize("failure", ["invalid-sample", "uninitialized-client", "lost-ack"])
async def test_native_queue_failure_keeps_only_possibly_delivered_bodies(published, native_write, failure):
    from unittest.mock import AsyncMock

    from transfer_queue.client import AsyncTransferQueueClient

    from relax.utils.utils import transfer_batch_to_data_system

    _, _, body, inventory, ref, _ = published
    args, sample, producer = native_write
    await producer.adopt_native_response(sample, args, {"routed_experts": ref.to_dict()})
    put = AsyncMock(side_effect=ConnectionError("TQ write ACK lost"))
    client = SimpleNamespace(async_put=put)
    if failure == "invalid-sample":
        sample.tokens[-1] = 99
        error, message = ValueError, "tokens"
    elif failure == "uninitialized-client":
        client = object.__new__(AsyncTransferQueueClient)
        client.storage_manager = None
        client.async_get_meta = AsyncMock()
        error, message = RuntimeError, "storage manager"
    else:
        error, message = ConnectionError, "ACK lost"

    with pytest.raises(error, match=message):
        await transfer_batch_to_data_system(args, [[sample]], 0, 0, client)
    if failure == "lost-ack":
        put.assert_awaited_once()
        assert inventory.state["objects"][ref.record]["partitions"] == ["train_0"]
        assert body.blocks
    else:
        if failure == "uninitialized-client":
            client.async_get_meta.assert_not_awaited()
        put.assert_not_awaited()
        assert not inventory.state["objects"] and not body.blocks


async def test_buffered_native_route_releases_only_for_returned_owner(published, native_write):
    from copy import copy
    from unittest.mock import AsyncMock

    from ray import cloudpickle

    from relax.engine.rollout.data_source import RolloutDataSourceWithBuffer
    from relax.utils.utils import transfer_batch_to_data_system

    _, _, body, inventory, ref, _ = published
    args, sample, producer = native_write
    await producer.adopt_native_response(sample, args, {"routed_experts": ref.to_dict()})
    buffer = RolloutDataSourceWithBuffer(
        SimpleNamespace(
            rollout_global_dataset=False, use_streaming_dataset=False, buffer_filter_path=None, n_samples_per_prompt=1
        )
    )
    groups = [[sample]]
    producer.manager.handoff_buffered_routes(groups)
    buffer.add_samples(cloudpickle.loads(cloudpickle.dumps(groups)))
    returned = cloudpickle.loads(cloudpickle.dumps(buffer.get_samples(1)))
    producer.manager.receive_buffered_routes(returned)
    (owner,) = returned[0]
    owner.loss_mask = [1, 1]
    client = SimpleNamespace(async_put=AsyncMock())
    with pytest.raises(AssertionError, match="loss mask"):
        await transfer_batch_to_data_system(args, [[copy(owner)]], 0, 0, client)
    assert body.blocks
    with pytest.raises(AssertionError, match="loss mask"):
        await transfer_batch_to_data_system(args, [[owner]], 0, 0, client)
    client.async_put.assert_not_awaited()
    assert not inventory.state["objects"] and not body.blocks


@pytest.mark.parametrize("custom", [False, True], ids=["borrowed-native-route", "custom-fanout"])
async def test_invalid_sibling_preserves_route_for_later_write(published, native_write, custom):
    from dataclasses import replace
    from unittest.mock import AsyncMock

    from relax.utils.utils import transfer_batch_to_data_system

    _, store, body, _, ref, _ = published
    args, sample, producer = native_write
    if custom:
        args.custom_generate_function_path = "custom_generator"
    await producer.adopt_native_response(sample, args, {"routed_experts": ref.to_dict()})
    # In a custom generator even the originating row may fail before its sibling submits.
    sibling = replace(sample)
    bad, good = (sample, sibling) if custom else (sibling, sample)
    bad.loss_mask = [1, 1]
    client = SimpleNamespace(async_put=AsyncMock())
    with pytest.raises(AssertionError, match="loss mask"):
        await transfer_batch_to_data_system(args, [[bad]], 0, 0, client)
    await transfer_batch_to_data_system(args, [[good]], 1, 1, client)
    client.async_put.assert_awaited_once()
    assert body.blocks
    store.retire_partition("train_1")
    assert not body.blocks


@pytest.fixture
def run_custom(native_write, monkeypatch):
    from contextlib import nullcontext

    from relax.engine.rollout import sglang_rollout

    args, _, producer = native_write
    monkeypatch.setenv("RELAX_REPLAY_REFERENCES", "1")
    args.custom_generate_function_path = "custom"
    args.partial_rollout = False
    args.group_rm = True
    args.use_dynamic_batch_size = True
    args.sglang_enable_deterministic_inference = False
    state = SimpleNamespace(aborted=False, dp_rank_context=nullcontext, semaphore=asyncio.Semaphore(2))
    monkeypatch.setattr(sglang_rollout, "GenerateState", lambda _: state)
    monkeypatch.setattr(sglang_rollout, "replay_data_manager", producer.manager)
    monkeypatch.setattr("relax.distributed.replay_manager.replay_data_manager", producer.manager)

    async def run(generate, source):
        monkeypatch.setattr(sglang_rollout, "load_function", lambda _: generate)
        if isinstance(source, list):
            return await sglang_rollout.generate_and_rm_group(args, source, {})
        return await sglang_rollout.generate_and_rm(args, source, {})

    return run


@pytest.mark.parametrize("case", ["raw", "normalized", "conflicting-rewards", "fixed-batch", "gspo"])
async def test_custom_group_hands_off_or_releases_all_rows(published, native_write, run_custom, monkeypatch, case):
    from dataclasses import replace
    from unittest.mock import AsyncMock

    from relax.engine.rollout import sglang_rollout
    from relax.utils.utils import transfer_batch_to_data_system

    _, store, body, inventory, ref, values = published
    args, source, producer = native_write
    args.use_dynamic_batch_size = case != "fixed-batch"
    args.rewards_normalization = case in ("normalized", "conflicting-rewards", "gspo")
    args.advantage_estimator = "gspo" if case == "gspo" else "grpo"
    args.n_samples_per_prompt = 2
    args.grpo_std_normalization = True
    rewards = [1.0, 1.0 if case == "normalized" else 2.0, 3.0]
    score = AsyncMock(return_value=rewards)
    monkeypatch.setattr(sglang_rollout, "batched_async_rm", score)

    async def generate(args, row, params):
        if row.index == 0:
            await producer.adopt_native_response(row, args, {"routed_experts": ref.to_dict()})
            return [replace(row, index=None), replace(row, index=None)]
        route = producer._publish_response(values, kind="moe", tokens=row.tokens)
        row.rollout_replay_refs = {"moe": route}
        await producer.adopt_native_response(row, args, {"routed_experts": route})
        return row

    second = replace(source, index=1, rollout_replay_refs=None)
    if not args.use_dynamic_batch_size:
        with pytest.raises(ValueError, match="use-dynamic-batch-size"):
            await run_custom(generate, [source, second])
        assert not inventory.state["objects"] and not body.blocks
        return

    rows = await run_custom(generate, [source, second])
    score.assert_awaited_once_with(args, rows, evaluation=False)
    assert [row.reward for row in rows] == rewards
    put = AsyncMock()
    transfer = transfer_batch_to_data_system(args, [rows], 1, 0, SimpleNamespace(async_put=put), is_last=True)
    if case in ("gspo", "conflicting-rewards"):
        message = "3 samples, expected 2" if case == "gspo" else "custom-reward-post-process-path"
        with pytest.raises(ValueError, match=message):
            await transfer
        put.assert_not_awaited()
        assert not inventory.state["objects"] and not body.blocks
        return
    await transfer
    put.assert_awaited_once()
    sent = put.await_args.kwargs
    expected_rewards = [-1 / 2**0.5, -1 / 2**0.5, 1 / 2**0.5] if case == "normalized" else rewards
    assert sent["data"]["rewards"].tolist() == pytest.approx(expected_rewards)
    assert [meta["sample_index"] for meta in sent["custom_meta"]] == [0, 0, 1]
    assert sent["data"]["sample_index_mask_sums"].tolist() == [2, 2, 1]
    assert sent["is_last"]
    assert inventory.state["objects"][ref.record]["partitions"] == ["train_0"]
    store.retire_partition("train_0")
    assert not inventory.state["objects"] and not body.blocks


@pytest.mark.parametrize(
    "outcomes", [("reject",), ("reject", "reject"), ("reject", "ack"), ("ack", "ack"), ("unknown", "reject")]
)
async def test_custom_route_lives_until_last_output_finishes(published, native_write, run_custom, outcomes):
    from dataclasses import replace
    from unittest.mock import AsyncMock

    from relax.utils.utils import transfer_batch_to_data_system

    _, store, body, inventory, ref, expected = published
    args, sample, producer = native_write

    async def generate(args, source, sampling_params):
        await producer.adopt_native_response(source, args, {"routed_experts": ref.to_dict()})
        # The generator may return copies and omit the sample that received the response.
        copies = (
            [replace(source)] * len(outcomes) if outcomes == ("reject", "ack") else [replace(source) for _ in outcomes]
        )
        return copies[0] if len(copies) == 1 else copies

    result = await run_custom(generate, sample)
    outputs = result if isinstance(result, list) else [result]
    for index, (outcome, row) in enumerate(zip(outcomes, outputs, strict=True)):
        put = AsyncMock()
        client = SimpleNamespace(async_put=put)
        if outcome == "reject":
            row.loss_mask = [1, 1]
            with pytest.raises(AssertionError, match="loss mask"):
                await transfer_batch_to_data_system(args, [[row]], 0, index, client)
            put.assert_not_awaited()
        elif outcome == "unknown":
            put.side_effect = ConnectionError("lost ACK")
            with pytest.raises(ConnectionError, match="lost ACK"):
                await transfer_batch_to_data_system(args, [[row]], 0, index, client)
            put.assert_awaited_once()
        else:
            await transfer_batch_to_data_system(args, [[row]], 0, index, client)
            put.assert_awaited_once()
            store.retire_partition(f"train_{index}")
        if index + 1 < len(outputs):
            assert body.blocks

    if "unknown" in outcomes:
        assert inventory.state["objects"][ref.record]["partitions"] == ["train_0"]
        with store.reader([ref], partition="train_0") as reader:
            output = np.empty((ref.rows, ref.topk), dtype=np.int32)
            rows = np.arange(ref.rows)
            reader.read_into(0, 0, rows, output, rows)
            np.testing.assert_array_equal(output, expected[:, 0])
        store.retire_partition("train_0")
        store.close_run()
    assert not inventory.state["objects"] and not body.blocks


async def test_custom_route_skips_collected_candidate(published, native_write, run_custom):
    from dataclasses import replace

    _, store, body, inventory, ref, values = published
    _, sample, producer = native_write

    async def select(args, row, params):
        await producer.adopt_native_response(row, args, {"routed_experts": ref.to_dict()})
        candidate = producer._publish_response(values, kind="moe", tokens=row.tokens)
        rejected = replace(row, rollout_replay_refs={"moe": candidate})
        await producer.adopt_native_response(rejected, args, {"routed_experts": candidate})
        await producer.manager.discard_async([rejected])
        return row

    selected = await run_custom(select, sample)
    assert set(inventory.state["objects"]) == {ref.record}
    await producer.manager.discard_async([selected], unsent=True, native=True)
    assert not body.blocks


async def test_custom_route_resume_splits_only_input_hold(published, native_write, run_custom):
    from copy import copy

    _, _, body, _, ref, _ = published
    _, sample, producer = native_write

    async def first(args, row, params):
        await producer.adopt_native_response(row, args, {"routed_experts": ref.to_dict()})
        return [copy(row), copy(row)]

    async def resume(args, row, params):
        return [copy(row), copy(row)]

    original, peer = await run_custom(first, sample)
    left, right = await run_custom(resume, original)
    for dropped in (left, peer):
        await producer.manager.discard_async([dropped], unsent=True, native=True)
        assert body.blocks
    await producer.manager.discard_async([right], unsent=True, native=True)
    assert not body.blocks


@pytest.mark.parametrize("late", ["native-response", "nested-response", "nested-output"])
async def test_late_route_after_custom_failure_is_reclaimed(published, native_write, run_custom, late):
    from copy import copy

    _, _, body, inventory, ref, _ = published
    _, sample, producer = native_write
    sample.rollout_replay_refs = None
    started, release, finished = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def wait_for_group_exit():
        started.set()
        await release.wait()

    async def receive(args, row, params):
        if late != "nested-output":
            await wait_for_group_exit()
        row.rollout_replay_refs = {"moe": ref.to_dict()}
        await producer.adopt_native_response(row, args, {"routed_experts": ref.to_dict()})
        if late == "nested-output":
            await wait_for_group_exit()
        return [row, copy(row)]

    async def fail(args, row, params):
        candidate = copy(row)
        work = (
            receive(SimpleNamespace(), candidate, params)
            if late == "native-response"
            else run_custom(receive, candidate)
        )
        task = asyncio.create_task(work)
        task.add_done_callback(lambda _: finished.set())
        await asyncio.wait_for(started.wait(), 5)
        raise ValueError("other request failed")

    with pytest.raises(ValueError, match="other request failed"):
        await run_custom(fail, [sample])
    assert body.blocks
    release.set()
    await asyncio.wait_for(finished.wait(), 5)
    assert not inventory.state["objects"] and not body.blocks


def test_normalized_sample_roundtrip_keeps_capture_manifest(published):
    from relax.distributed.replay_manager.ref_data import decode_refs, encode_refs, iter_refs, validate_sample
    from relax.utils.types import Sample

    config, store, _, inventory, ref, _ = published
    producer = ReplayManager(config=config, store_factory=lambda: store).register_producer(["moe"])
    sample = Sample(tokens=[10, 99, 12, 13, 14])
    meta = {"routed_experts": ref.to_dict()}
    producer.accept_response(sample, meta, SimpleNamespace(), captured_tokens=[10, 11, 12, 13, 14])
    received = decode_refs(encode_refs(sample.rollout_replay_refs))
    validate_sample(received, sample.tokens)
    assert list(iter_refs(received)) == [ref]
    assert meta["routed_experts"] == inventory.state["objects"][ref.record]["manifest"] == ref.to_dict()
    with pytest.raises(ValueError, match="tokens"):
        validate_sample(received, [10, 77, 12, 13, 14])


@pytest.mark.parametrize("capacity", [0, 8, 4096])
def test_reference_repeated_reads_preserve_values_and_retired_reader_ownership(published, capacity):
    _, store, body, inventory, ref, values = published
    store.attach([ref], "train_0")
    store.commit([ref])
    cache = ConsumerTensorCache(capacity)
    # Include padding and a non-contiguous row order, as after rank slicing.
    layout = ConsumerBatchLayout.from_coordinates(np.array([[0, 3], [0, 0], [-1, 0], [0, 2]]))
    with store.reader([ref], partition="train_0") as session:
        sources = [
            ConsumerLayerReader(session, layout, layer=layer, kind="moe", device="cpu", num_experts=128, cache=cache)
            for layer in range(3)
        ]
    try:
        store.retire_partition("train_0")
        assert body.blocks  # Retiring the queue must not invalidate recompute.
        first_reads = None
        for phase in range(3):
            for layer, source in enumerate(sources):
                expected = np.stack((values[3, layer], values[0, layer], [layer * 2, layer * 2 + 1], values[2, layer]))
                actual = source.read()
                assert actual.dtype == torch.int64
                np.testing.assert_array_equal(actual.numpy(), expected)
                actual.fill_(99)  # Returned tensors must not mutate the cache.
            if phase == 0:
                first_reads = body.reads
            elif capacity == 4096:
                assert body.reads == first_reads
        if capacity in (0, 8):
            assert body.reads > first_reads
    finally:
        for source in sources:
            source.release()
    assert not inventory.state["objects"]
    assert not body.blocks


@pytest.mark.parametrize("capacity", [0, 32, 4096])
def test_reference_uint8_recompute_preserves_unsigned_routes_and_padding(capacity):
    from relax.distributed.replay_manager.ref_data import decode_refs, encode_refs

    # Odd top-k, two layers, two chunks, and IDs on both sides of int8's limit.
    values = np.array(
        [[[0, 127, 255], [128, 255, 1]], [[255, 128, 0], [127, 0, 128]], [[128, 3, 127], [255, 1, 0]]],
        dtype=np.int32,
    )
    config = ReplayConfig(delivery="reference", run_id="a", chunk_rows=2, capacity_bytes=18)
    body = MemoryBody()
    inventory = ReplayInventory(config, body.delete)
    store = TQReplayStorage(config, inventory=inventory)
    store._backend = body
    sources = []
    try:
        publication = store.begin_publication(kind="moe", layers=2, topk=3, codec="uint8")
        for start, count in ((0, 2), (2, 1)):
            publication.append(count, lambda layer, start=start, count=count: values[start : start + count, layer])
        ref = publication.finish([10, 11, 12, 13])
        assert list(decode_refs(encode_refs({"moe": ref.to_dict()})).values()) == [ref.to_dict()]
        assert ref.codec == "uint8" and inventory._used_bytes == 18
        assert sum(block.nbytes for block in body.blocks.values()) == 18
        assert all(block.dtype == np.uint8 for block in body.blocks.values())
        store.attach([ref], "train_0")
        store.commit([ref])
        layout = ConsumerBatchLayout.from_coordinates(np.array([[0, 2], [-1, 0], [0, 0], [0, 1]]))
        cache = ConsumerTensorCache(capacity)
        with store.reader([ref], partition="train_0") as session:
            sources = [
                ConsumerLayerReader(
                    session, layout, layer=layer, kind="moe", device="cpu", num_experts=256, cache=cache
                )
                for layer in range(2)
            ]
        store.retire_partition("train_0")
        assert body.blocks
        first_reads = None
        for phase in range(3):
            for layer, source in enumerate(sources):
                actual = source.read()
                assert actual.dtype == torch.int64
                padding = [0, 1, 2] if layer == 0 else [3, 4, 5]
                np.testing.assert_array_equal(
                    actual.numpy(), np.stack([values[2, layer], padding, *values[:2, layer]])
                )
                actual.fill_(-1)
            if phase == 0:
                first_reads = body.reads
            elif capacity:
                assert body.reads == first_reads
    finally:
        for source in sources:
            source.release()
        store.close_run()
    assert not body.blocks and inventory._used_bytes == 0


@pytest.mark.parametrize("invalid_id", [-1, 256])
def test_reference_uint8_rejects_out_of_range_before_narrowing(invalid_id):
    config = ReplayConfig(delivery="reference", run_id="a")
    body = MemoryBody()
    inventory = ReplayInventory(config, body.delete)
    store = TQReplayStorage(config, inventory=inventory)
    store._backend = body
    try:
        publication = store.begin_publication(kind="moe", layers=1, topk=2, codec="uint8")
        with pytest.raises(ValueError, match="indices must be in"):
            publication.append(1, lambda _: np.array([[0, invalid_id]], dtype=np.int32))
        assert not body.blocks and inventory._used_bytes == 0
        publication.abort()
    finally:
        store.close_run()


def test_reference_retire_before_put_ack_keeps_reader_until_final_release(published):
    _, store, body, inventory, ref, _ = published
    store.attach([ref], "train_0")
    leases = store.acquire_many([ref], partition="train_0")
    store.retire_partition("train_0")
    store.commit([ref])
    assert body.blocks
    store.release_many([ref], leases)
    assert not body.blocks and not inventory.state["objects"]


def test_reference_prefetches_next_microbatch_without_blocking_cached_layers(published, monkeypatch) -> None:
    _, store, body, inventory, first_ref, first_values = published
    second_values = np.arange(4 * 5 * 2, dtype=np.int32).reshape(4, 5, 2) + 216
    publication = store.begin_publication(kind="moe", layers=5, topk=2)
    for start in (0, 2):
        publication.append(2, lambda layer, start=start: second_values[start : start + 2, layer].copy())
    second_ref = publication.finish([20, 21, 22, 23, 24])
    refs = [first_ref, second_ref]
    store.attach(refs, "train_0")
    store.commit(refs)
    cache = ConsumerTensorCache(4096)
    groups = []
    for ref, rows, layers in ((first_ref, [3, 0], 3), (second_ref, [1, 2], 5)):
        layout = ConsumerBatchLayout.from_coordinates(np.array([[0, row] for row in rows]))
        with store.reader([ref], partition="train_0") as session:
            groups.append(
                [
                    ConsumerLayerReader(
                        session, layout, layer=layer, kind="moe", device="cpu", num_experts=256, cache=cache
                    )
                    for layer in range(layers)
                ]
            )
    started, resume = Event(), Event()
    read_layers = store._read_layers_into

    def gated_read(layers: list, **kwargs) -> None:
        if layers[0][0][0][0].record == second_ref.record:
            started.set()
            assert resume.wait(5), "Next-microbatch read was never released"
        return read_layers(layers, **kwargs)

    monkeypatch.setattr(store, "_read_layers_into", gated_read)
    try:
        groups[0][0].read()
        groups[0][1].read()
        assert started.wait(2), "Next-microbatch prefetch did not start"
        assert len(cache._read_ahead.pending.sources) == 5
        assert cache.bytes + cache.prefetch_bytes <= cache.capacity_bytes
        # A pending read for another microbatch must not stall cached layers.
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(groups[0][2].read)
            try:
                np.testing.assert_array_equal(future.result(timeout=1).numpy(), first_values[[3, 0], 2])
            finally:
                resume.set()
        for layer, source in enumerate(groups[1]):
            np.testing.assert_array_equal(source.read().numpy(), second_values[[1, 2], layer])
        first_reads = body.reads
        for sources, values, rows in ((groups[0], first_values, [3, 0]), (groups[1], second_values, [1, 2])):
            for layer, source in enumerate(sources):
                np.testing.assert_array_equal(source.read().numpy(), values[rows, layer])
        assert body.reads == first_reads
        assert cache._read_ahead.cold_reads == 1
        assert cache.prefetch_peak_bytes <= cache.capacity_bytes
    finally:
        resume.set()
        store.retire_partition("train_0")
        for sources in groups:
            for source in sources:
                source.release()
    assert not body.blocks and not inventory.state["objects"]


def test_reference_prefetch_keeps_uncached_current_layers_ahead_of_next_microbatch(published) -> None:
    _, store, body, inventory, ref, values = published
    store.attach([ref], "train_0")
    store.commit([ref])
    cache = ConsumerTensorCache(600)
    groups = []
    for rows, layers in (([0, 1, 2, 3, 0, 1, 2, 3, 0, 1], (0, 1)), ([2], (0,))):
        layout = ConsumerBatchLayout.from_coordinates(np.array([[0, row] for row in rows]))
        with store.reader([ref], partition="train_0") as session:
            groups.append(
                [
                    ConsumerLayerReader(
                        session, layout, layer=layer, kind="moe", device="cpu", num_experts=256, cache=cache
                    )
                    for layer in layers
                ]
            )
    try:
        np.testing.assert_array_equal(groups[0][0].read().numpy(), values[[0, 1, 2, 3, 0, 1, 2, 3, 0, 1], 0])
        # The next current layer cannot fit the conservative reservation.
        # A smaller future microbatch must not jump ahead and make it wait.
        assert cache._read_ahead.pending is None
        groups[0][1].read()
        np.testing.assert_array_equal(groups[1][0].read().numpy(), values[[2], 0])
        assert cache.prefetch_peak_bytes <= cache.capacity_bytes
    finally:
        store.retire_partition("train_0")
        for sources in groups:
            for source in sources:
                source.release()
    assert not body.blocks and not inventory.state["objects"]


def test_reference_prefetch_reclaims_old_cache_without_evicting_current_recompute(published, monkeypatch) -> None:
    _, store, body, inventory, ref, values = published
    store.attach([ref], "train_0")
    store.commit([ref])
    cache = ConsumerTensorCache(512)
    sources = []
    with store.reader([ref], partition="train_0") as session:
        for _ in range(64):
            layout = ConsumerBatchLayout.from_coordinates(np.array([[0, row] for row in range(4)]))
            sources.append(
                ConsumerLayerReader(session, layout, layer=0, kind="moe", device="cpu", num_experts=128, cache=cache)
            )
        layout = ConsumerBatchLayout.from_coordinates(np.array([[0, row] for row in range(4)]))
        current = [
            ConsumerLayerReader(session, layout, layer=layer, kind="moe", device="cpu", num_experts=128, cache=cache)
            for layer in range(3)
        ]
        sources.extend(current)
    # Model a full cache left by the previous pass, using real body reads and
    # materialization. The next layer must still read ahead under this budget.
    for source in sources[:-3]:
        cache.put(source._cache_key, source._materialize())
    assert cache.bytes == cache.capacity_bytes
    started, resume = Event(), Event()
    read_layers = store._read_layers_into

    def gated_read(layers: list, **kwargs) -> None:
        if kwargs.get("on_layer") is not None:
            started.set()
            assert resume.wait(5)
        return read_layers(layers, **kwargs)

    monkeypatch.setattr(store, "_read_layers_into", gated_read)
    try:
        np.testing.assert_array_equal(current[0].read().numpy(), values[:, 0])
        assert started.wait(2), "A full old cache disabled read-ahead"
        assert cache.bytes + cache.prefetch_bytes <= cache.capacity_bytes
        # Recompute must hit the current layer while the next read is blocked.
        reads = body.reads
        np.testing.assert_array_equal(current[0].read().numpy(), values[:, 0])
        assert body.reads == reads
        resume.set()
        for layer in (1, 2):
            np.testing.assert_array_equal(current[layer].read().numpy(), values[:, layer])
        assert cache.prefetch_peak_bytes <= cache.capacity_bytes
    finally:
        resume.set()
        store.retire_partition("train_0")
        for source in sources:
            source.release()
    assert not body.blocks and not inventory.state["objects"]


@pytest.mark.parametrize("fail_after_ready", [False, True])
def test_reference_prefetch_returns_ready_layer_and_propagates_later_failure(
    published, monkeypatch, fail_after_ready
) -> None:
    _, store, body, inventory, ref, values = published
    store.attach([ref], "train_0")
    store.commit([ref])
    cache = ConsumerTensorCache(4096)
    layout = ConsumerBatchLayout.from_coordinates(np.array([[0, 0], [0, 3]]))
    with store.reader([ref], partition="train_0") as session:
        sources = [
            ConsumerLayerReader(session, layout, layer=layer, kind="moe", device="cpu", num_experts=128, cache=cache)
            for layer in range(3)
        ]
    first_ready, resume = Event(), Event()
    read_layers = store._read_layers_into

    def gated_read(layers: list, *, on_layer=None) -> None:
        def publish(index: int) -> None:
            if on_layer is not None:
                on_layer(index)
            if index == 0:
                first_ready.set()
                assert resume.wait(5)
                if fail_after_ready:
                    raise RuntimeError("later layer read failed")

        return read_layers(layers, on_layer=publish if on_layer is not None else None)

    monkeypatch.setattr(store, "_read_layers_into", gated_read)
    try:
        np.testing.assert_array_equal(sources[0].read().numpy(), values[[0, 3], 0])
        assert first_ready.wait(2)
        window = cache._read_ahead.pending
        with ThreadPoolExecutor(max_workers=1) as executor:
            result = executor.submit(sources[1].read)
            try:
                np.testing.assert_array_equal(result.result(timeout=1).numpy(), values[[0, 3], 1])
                assert not window.future.done()
            finally:
                resume.set()
        if fail_after_ready:
            with pytest.raises(RuntimeError, match="later layer read failed"):
                window.future.result(timeout=2)
            # Even a warm-cache read must observe the background failure.
            with pytest.raises(RuntimeError, match="later layer read failed"):
                sources[0].read()
        else:
            window.future.result(timeout=2)
            np.testing.assert_array_equal(sources[2].read().numpy(), values[[0, 3], 2])
        assert cache.bytes + cache.prefetch_bytes <= cache.capacity_bytes
    finally:
        resume.set()
        store.retire_partition("train_0")
        for source in sources:
            source.release()
    assert not body.blocks and not inventory.state["objects"]


def test_reference_int32_read_rejects_lossy_uint8_destination(published) -> None:
    _, store, body, _, ref, _ = published
    output = np.zeros((1, ref.topk), dtype=np.uint8)
    with store.reader([ref]) as session:
        with pytest.raises(ValueError, match="Invalid replay read layout"):
            session.read_into(0, 0, [0], output, [0])
    assert body.reads == 0


def test_reference_wire_round_trip_validates_token_identity_without_body_reads(published):
    config, store, body, _, ref, _ = published
    manager = ReplayManager(config=config, store_factory=lambda: store)
    consumer = manager.register_consumer({"moe"})
    assert consumer.fields(["tokens", "rollout_routed_experts"]) == ["tokens", "rollout_replay_refs"]
    packed = manager.pack("rollout_replay_refs", [{"moe": ref.to_dict()}])
    batch = {"tokens": [[10, 11, 12, 13, 14]], "rollout_replay_refs": list(packed.unbind())}
    consumer.receive_references(batch)
    assert batch["rollout_replay_refs"] == [{"moe": ref.to_dict()}]
    assert body.reads == 0
    batch = {"tokens": [[10, 11, 12, 13, 99]], "rollout_replay_refs": list(packed.unbind())}
    with pytest.raises(ValueError, match="does not match sample tokens"):
        consumer.receive_references(batch)


@pytest.fixture
def r3_batches(published, monkeypatch):
    from megatron.core import mpu

    from relax.distributed.replay_manager.backend import megatron

    config, store, body, _, first, values = published
    monkeypatch.setenv("RELAX_REPLAY_REFERENCES", "1")
    publication = store.begin_publication(kind="moe", layers=3, topk=2)
    for start in (0, 2):
        publication.append(2, lambda layer, start=start: values[start : start + 2, layer] + 100)
    second = publication.finish([20, 21, 22, 23, 24])
    refs = [first, second]
    store.attach(refs, "train_0")
    store.commit(refs)
    manager = ReplayManager(config=config, store_factory=lambda: store)
    monkeypatch.setattr(megatron, "replay_data_manager", manager)
    monkeypatch.setattr(megatron.device_utils, "make_current_torch_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(mpu, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(mpu, "get_context_parallel_world_size", lambda: 1)
    monkeypatch.setattr(mpu, "get_context_parallel_rank", lambda: 0)
    monkeypatch.setattr(megatron.RoutingReplay, "all_routing_replays", [])
    routers = [megatron.RoutingReplay() for _ in range(3)]
    args = SimpleNamespace(
        num_layers=3,
        moe_router_topk=2,
        num_experts=256,
        sequence_parallel=False,
        qkv_format="thd",
        allgather_cp=False,
        data_pad_size_multiplier=1,
    )
    batches = [
        {
            "tokens": [torch.arange(start, start + 5)],
            "total_lengths": [5],
            "rollout_replay_refs": [{"moe": ref.to_dict()}],
        }
        for start, ref in zip((10, 20), refs, strict=True)
    ]
    try:
        yield manager, store, body, routers, args, batches, values
    finally:
        megatron.RoutingReplay.clear_all()
        store.retire_partition("train_0")
        assert not body.blocks


def test_streaming_r3_keeps_virtual_chunks_microbatches_and_recompute_in_order(r3_batches):
    from relax.distributed.replay_manager.backend import megatron

    _, store, body, routers, args, batches, values = r3_batches

    def model(layers):
        return SimpleNamespace(
            modules=lambda: [SimpleNamespace(layer_number=i + 1, routing_replay=routers[i]) for i in layers]
        )

    models = [model((0,)), model((2, 1))]
    batches = [{**batch, "unconcat_tokens": batch["tokens"]} for batch in batches]
    bind = megatron.streaming_moe_replay(args, models, "train_0")
    # The first chunk can be ahead of the next chunk on the same rank.
    for model, layer_indices in zip(models, ((0,), (1, 2)), strict=True):
        for batch, offset in zip([*batches, {**batches[1], "__is_dummy__": True}], (0, 100, 100), strict=True):
            bind(model, batch)
            for layer in layer_indices:
                expected = np.concatenate((values[:, layer] + offset, [[2 * layer, 2 * layer + 1]]))
                np.testing.assert_array_equal(routers[layer].pop_forward().numpy(), expected)
    store.retire_partition("train_0")
    assert body.blocks
    for layer, router in enumerate(routers):
        for offset in (0, 100, 100):
            expected = np.concatenate((values[:, layer] + offset, [[2 * layer, 2 * layer + 1]]))
            np.testing.assert_array_equal(router.pop_backward().numpy(), expected)


@pytest.mark.parametrize("mode", ["fixed", "streaming", "hybrid"])
def test_r3_forward_uses_each_batch_routes_and_writes_only_real_rows(r3_batches, monkeypatch, mode):
    from megatron.core import mpu

    from relax.backends.megatron import actor as backend

    manager, store, body, routers, args, batches, values = r3_batches
    actor = object.__new__(backend.MegatronTrainRayActor)
    actor.model = [
        SimpleNamespace(
            modules=lambda: [SimpleNamespace(layer_number=i + 1, routing_replay=r) for i, r in enumerate(routers)]
        )
    ]
    actor.prof = SimpleNamespace(step=lambda **_: None)
    actor.weights_backuper = SimpleNamespace(backup_tags=["actor"])
    actor._switch_model = actor.recv_weight_fully_async = lambda _: None
    actor.data_system_client = None
    actor.args = args
    vars(args).update(
        use_routing_replay=True,
        use_rollout_routing_replay=True,
        use_dynamic_batch_size=False,
        global_batch_size=1,
        num_iters_per_train_update=1,
        max_tokens_per_gpu=8,
        log_probs_max_tokens_per_gpu=8,
        multimodal_keys=None,
        compute_advantages_and_returns=True,
        keep_old_actor=False,
        use_rollout_logprobs=False,
    )
    monkeypatch.setattr(backend, "replay_data_manager", manager)
    monkeypatch.setattr(backend, "log_perf_data_fwd", lambda *_: None)
    monkeypatch.setattr(mpu, "get_data_parallel_rank", lambda: 0)
    monkeypatch.setattr(mpu, "get_data_parallel_world_size", lambda **_: 1)
    monkeypatch.setattr(
        backend,
        "get_data_iterator",
        lambda args, model, data, **_: ([backend.DataIterator(data, micro_batch_size=1)], [1]),
    )

    observed, written = [], []

    def forward(iterators, _, *, store_prefix):
        batch = iterators[0].get_next(["tokens"])
        start = batch["tokens"][0][0].item()
        observed.append(start)
        for layer, router in enumerate(routers):
            expected = np.concatenate((values[:, layer] + (0 if start == 10 else 100), [[2 * layer, 2 * layer + 1]]))
            np.testing.assert_array_equal(router.pop_forward().numpy(), expected)
        return {"log_probs": [start]}

    actor.compute_log_prob = forward
    actor._put_data_to_transfer_queue = lambda output, meta, data: written.extend(output["log_probs"])
    if mode == "hybrid":
        region = manager.region("train_0")
        cache = ConsumerTensorCache(manager.config.materialization_bytes)
        for batch in batches:
            actor._hybrid_forward_subbatch(batch, replay_region=region, replay_cache=cache)
        assert [batch["log_probs"] for batch in batches] == [[10], [20]]
        assert observed == [10, 20]
        return

    def fetched(fields):
        assert manager.register_consumer().fields(fields)[-1] == "rollout_replay_refs"

    def fetch(_, rollout_id, fields, size, index):
        assert rollout_id == 0 and size == 1
        fetched(fields)
        return batches[index], index

    def streaming(**kwargs):
        fetched(kwargs["data_fields"])
        return [(batches[0], 0), (batches[1], 1), ({**batches[-1], "__is_dummy__": True}, 1)]

    actor._use_streaming_fwd = lambda: mode == "streaming"
    actor.all_consumed = lambda *_: len(written) == len(batches)
    actor._get_data_from_transfer_queue = fetch
    monkeypatch.setattr(backend, "StreamingTQIterator", streaming)
    actor.compute_actor_log_prob(0)
    assert written == [10, 20]
    assert observed == ([10, 20, 20] if mode == "streaming" else [10, 20])
    store.retire_partition("train_0")
    assert not body.blocks


@pytest.mark.asyncio
async def test_reference_delete_failure_drains_units_and_preserves_retry_metadata(published):
    _, _, _, inventory, ref, _ = published
    metadata = SimpleNamespace(global_indexes=[1, 2])
    remaining = {1, 2}
    events = []
    fail = True

    async def clear(indexes, *, target_storage_unit):
        if target_storage_unit == 1 and fail:
            raise RuntimeError("storage unit unavailable")
        await asyncio.sleep(0)
        remaining.difference_update(indexes)
        events.append(target_storage_unit)

    async def clear_meta(meta):
        assert meta is metadata and not remaining
        events.append("metadata")

    client = SimpleNamespace(
        storage_manager=SimpleNamespace(
            _group_by_hash=lambda _: {i: SimpleNamespace(global_indexes=[i]) for i in (1, 2)},
            _clear_single_storage_unit=clear,
        ),
        _clear_meta_in_controller=clear_meta,
    )
    with pytest.raises(RuntimeError, match="storage unit unavailable"):
        await _clear_simple_storage_samples(client, metadata)
    assert events == [2] and remaining == {1}

    # Exercise the inventory contract: a failed deletion must retain its record.
    def fail_delete(keys):
        raise RuntimeError("storage unit unavailable")

    inventory._delete = fail_delete
    with pytest.raises(RuntimeError, match="storage unit unavailable"):
        inventory.discard([ref])
    assert ref.record in inventory.state["objects"]
    fail = False
    await _clear_simple_storage_samples(client, metadata)
    assert events[-1] == "metadata"
    inventory._delete = lambda keys: None
    inventory.discard([ref])
    assert not inventory.state["objects"]


def test_reference_restart_replaces_registry_and_rejects_previous_generation(published, monkeypatch):
    config, store, _, _, ref, _ = published
    monkeypatch.setenv("RELAX_REPLAY_RUN_ID", config.run_id)
    manager = ReplayManager(config=config)
    manager._store_instance = store
    manager.start_run()
    # A dead consumer retains a lease; normal close would reject this state.
    store.acquire_many([ref])
    runtime_env = {"env_vars": {"EXISTING": "value"}}
    manager.prepare_restart(runtime_env)
    assert manager.config.run_id != config.run_id
    assert runtime_env["env_vars"] == {"EXISTING": "value", "RELAX_REPLAY_RUN_ID": manager.config.run_id}
    assert os.environ["RELAX_REPLAY_RUN_ID"] == manager.config.run_id
    assert not manager._owns_run and manager._store_instance is None
    new_inventory = ReplayInventory(manager.config, lambda keys: None)
    replacement = TQReplayStorage(manager.config, inventory=new_inventory)
    manager._store_instance = replacement
    manager.start_run()
    with pytest.raises(RuntimeError, match="another replay run"):
        replacement.acquire_many([ref])
    assert not new_inventory.state["objects"] and not new_inventory.state["closed_partitions"]
    manager.finish_run()


def test_reference_failed_finalization_is_reported_without_retrying_closed_tq(published):
    config, store, _, _, ref, _ = published
    manager = ReplayManager(config=config)
    manager._store_instance = store
    manager.start_run()
    store.acquire_many([ref])
    with pytest.raises(RuntimeError, match="writers, readers or queue holds"):
        manager.finish_run()
    assert not manager._owns_run and manager._run_stopped
    # The second call is the entrypoint after Controller has destroyed TQ.
    store.close_run = lambda: pytest.fail("must not access the closed TQ")
    with pytest.raises(RuntimeError, match="previously failed"):
        manager.finish_run()


def test_reference_capture_reads_modern_request_kv_slots(published, monkeypatch):
    from relax.distributed.replay_manager.backend import sglang

    config, store, body, _, _, _ = published
    manager = ReplayManager(config=config, store_factory=lambda: store)
    monkeypatch.setattr(sglang, "replay_data_manager", manager)
    buffer = torch.arange(6 * 3 * 2, dtype=torch.int32).reshape(6, 3, 2)
    capturer = SimpleNamespace(host_cache=SimpleNamespace(buffer=buffer))
    pool = SimpleNamespace(req_to_token=torch.tensor([[4, 1, 5, 0]]))
    processor = SimpleNamespace(req_to_token_pool=pool)
    req = SimpleNamespace(
        kv=SimpleNamespace(req_pool_idx=0),
        return_routed_experts=True,
        origin_input_ids=[10, 11],
        output_ids_through_stop=[12, 13, 14],
    )
    publisher = (0, "engine")
    store.invoke("register_engine_publisher", *publisher)
    refs = sglang._advance_publication(
        processor, req, 4, {"moe": lambda: capturer}, final=True, engine_publisher=publisher
    )
    ref = sglang.ReplayRef.from_dict(refs["moe"])
    with store.reader([ref]) as reader:
        for layer in range(3):
            output = np.empty((4, 2), np.int32)
            reader.read_into(0, layer, np.arange(4), output, np.arange(4))
            np.testing.assert_array_equal(output, buffer[[4, 1, 5, 0], layer].numpy())
    store.discard([ref])
    assert all(not key.startswith(ref.record) for key in body.blocks)


def test_dense_routed_expert_decode_preserves_alignment_diagnostics():
    import base64
    import binascii

    from relax.distributed.replay_manager.backend.sglang import decode_routed_experts

    values = np.array([1, 2], dtype=np.int32)
    payload = base64.b64encode(values.tobytes()).decode("ascii")
    meta = {"routed_experts": payload, "finish_reason": {"type": "stop"}, "prompt_tokens": 2}
    np.testing.assert_array_equal(decode_routed_experts(meta, 2, 1, 1).reshape(-1), values)
    with pytest.raises(ValueError, match=r"Routed-expert alignment mismatch: .*prompt_tokens=2"):
        decode_routed_experts(meta, 3, 1, 1)
    with pytest.raises(binascii.Error):
        decode_routed_experts({**meta, "routed_experts": payload + "!"}, 2, 1, 1)


@pytest.mark.asyncio
async def test_registry_reclaims_batch_without_blocking_publications():
    from threading import Event

    from relax.distributed.replay_manager.storage import _registry_type

    started, release = Event(), Event()

    class SlowBody(MemoryBody):
        def __init__(self):
            super().__init__()
            self.deletes = []
            self.fail = True

        def delete(self, keys):
            self.deletes.append(tuple(keys))
            started.set()
            if not release.wait(5):
                raise TimeoutError("Test did not release body deletion")
            if self.fail:
                raise RuntimeError("storage unavailable")
            super().delete(keys)

    config = ReplayConfig(delivery="reference", run_id="abcd", chunk_rows=2)
    body = SlowBody()
    registry = _registry_type()(config, "driver", backend=body)
    store = TQReplayStorage(config, inventory=registry)
    store._backend = body
    refs = []
    for offset in (0, 10):
        publication = store.begin_publication(kind="moe", layers=1, topk=2)
        publication.append(2, lambda _: np.full((2, 2), offset, dtype=np.int32))
        refs.append(publication.finish([1, 2, 3]))
    store.attach(refs, "train_0")
    store.commit(refs)
    used = registry._used_bytes
    retirement = asyncio.create_task(registry.invoke("retire_partition", ("train_0",), {}))
    duplicate = None
    try:
        assert await asyncio.to_thread(started.wait, 5)
        assert not retirement.done()
        assert registry._used_bytes == used
        assert len(body.deletes) == 1 and len(body.deletes[0]) == 2
        with pytest.raises(RuntimeError, match="keys cannot be reused"):
            await registry.invoke("begin", (refs[0].record, "moe", 1, 2, 2), {})
        # Exercise the real production dispatch while the delete worker is held.
        claim = await asyncio.wait_for(registry.invoke("claim_media", ("a" * 64, [1, 2, 2], "b" * 32, "image"), {}), 1)
        assert claim["status"] == "claimed"
        await asyncio.wait_for(registry.invoke("begin", ("c" * 32, "moe", 1, 2, 2), {}), 1)
        await asyncio.wait_for(registry.invoke("reserve_layers", ("c" * 32, 0, (0,), 2), {}), 1)
        await asyncio.wait_for(registry.invoke("written_layers", ("c" * 32, 0, (0,)), {}), 1)
        duplicate = asyncio.create_task(registry.invoke("retire_partition", ("train_0",), {}))
        await asyncio.sleep(0)
        assert not duplicate.done()
        release.set()
        for task in (retirement, duplicate):
            with pytest.raises(RuntimeError, match="storage unavailable"):
                await task
        # Delete failure keeps both bodies and their charged bytes retryable.
        assert all(ref.record in registry.state["objects"] for ref in refs)
        assert registry._used_bytes == used + 16
        body.fail = False
        await registry.invoke("retire_partition", ("train_0",), {})
        assert not any(ref.record in registry.state["objects"] for ref in refs)
        assert registry._used_bytes == 16
        await registry.invoke("abort", ("c" * 32,), {})
        await registry.invoke("abort_media_claim", ("a" * 64, "b" * 32), {})
        await registry.invoke("finish", ("driver",), {})
        assert registry._used_bytes == 0 and not body.blocks and not registry._retiring
    finally:
        release.set()
        await asyncio.gather(*(task for task in (retirement, duplicate) if task is not None), return_exceptions=True)
        if store._writer is not None:
            store._writer.close()
        registry._reclaim_executor.shutdown(wait=True)


@pytest.mark.asyncio
async def test_registry_reclaim_survives_repeated_cancellation_and_waits_for_capacity():
    from threading import Event

    from relax.distributed.replay_manager.storage import _registry_type

    started, release = Event(), Event()
    body = MemoryBody()
    delete = body.delete

    def slow_delete(keys):
        started.set()
        if not release.wait(5):
            raise TimeoutError("Test did not release body deletion")
        delete(keys)

    body.delete = slow_delete
    config = ReplayConfig(delivery="reference", run_id="abcd", capacity_bytes=16, chunk_rows=2)
    registry = _registry_type()(config, "driver", backend=body)
    store = TQReplayStorage(config, inventory=registry)
    store._backend = body
    publication = store.begin_publication(kind="moe", layers=1, topk=2)
    publication.append(2, lambda _: np.ones((2, 2), dtype=np.int32))
    ref = publication.finish([1, 2, 3])
    store.attach([ref], "train_0")
    store.commit([ref])
    retirement = asyncio.create_task(registry.invoke("retire_partition", ("train_0",), {}))
    reserve = None
    try:
        assert await asyncio.to_thread(started.wait, 5)
        for _ in range(2):
            retirement.cancel()
            await asyncio.sleep(0)
        assert not retirement.done() and ref.record in registry._retiring
        assert registry._used_bytes == 16 and body.blocks
        await registry.invoke("begin", ("d" * 32, "moe", 1, 2, 2), {})
        reserve = asyncio.create_task(registry.invoke("reserve_layers", ("d" * 32, 0, (0,), 2), {}))
        await asyncio.sleep(0)
        assert not reserve.done() and registry._used_bytes == 16
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await retirement
        await asyncio.wait_for(reserve, 2)
        assert not registry._retiring and registry._used_bytes == 16
        await registry.invoke("written_layers", ("d" * 32, 0, (0,)), {})
        await registry.invoke("abort", ("d" * 32,), {})
        await registry.invoke("finish", ("driver",), {})
        assert registry._used_bytes == 0 and not body.blocks
    finally:
        release.set()
        await asyncio.gather(*(task for task in (retirement, reserve) if task is not None), return_exceptions=True)
        store._writer.close()
        registry._reclaim_executor.shutdown(wait=True)
