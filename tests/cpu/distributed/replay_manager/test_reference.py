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


def test_r3_rollout_dump_round_trips_routes_across_runs(tmp_path, monkeypatch):
    import base64

    from relax.distributed.replay_manager.config import default_reference_r3
    from relax.utils.training.train_dump_utils import load_debug_rollout_samples, save_debug_rollout_data
    from relax.utils.types import Sample

    monkeypatch.delenv("RELAX_REPLAY_REFERENCES", raising=False)
    args = SimpleNamespace(
        train_backend="megatron",
        colocate=True,
        use_rollout_routing_replay=True,
        num_layers=1,
        moe_router_topk=1,
        save_debug_rollout_data=None,
    )
    assert default_reference_r3(args)
    args.save_debug_rollout_data = str(tmp_path / "rollout-{rollout_id}.pt")
    manager = ReplayManager(config=ReplayConfig(delivery="reference" if default_reference_r3(args) else "full"))
    sample = Sample(tokens=[1, 2, 3])
    routes = np.array([[[7]], [[3]]], dtype=np.int32)
    producer = manager.register_producer(["moe"])
    producer.accept_response(sample, {"routed_experts": base64.b64encode(routes).decode()}, args)
    save_debug_rollout_data(args, [sample], 0, evaluation=False)
    [restored] = load_debug_rollout_samples(tmp_path / "rollout-0.pt", args)
    np.testing.assert_array_equal(restored.rollout_routed_experts, routes)


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


@pytest.mark.asyncio
async def test_cancel_before_queue_put_releases_only_this_writer(published, monkeypatch):
    from unittest.mock import AsyncMock

    config, store, body, inventory, ref, _ = published
    manager = ReplayManager(config=config, store_factory=lambda: store)
    sample = SimpleNamespace(rollout_replay_refs={"moe": ref.to_dict()})
    attached, acknowledge = asyncio.Event(), asyncio.Event()
    invoke = store.invoke_async

    async def delay_ack(method, *args, **kwargs):
        result = await invoke(method, *args, **kwargs)
        if method == "attach":
            attached.set()
            await acknowledge.wait()
        return result

    monkeypatch.setattr(store, "invoke_async", delay_ack)
    write = AsyncMock()
    task = asyncio.create_task(manager.region("train_0").put([sample], write))
    await asyncio.wait_for(attached.wait(), 5)
    # Another sample sharing the same body can finish while the first waits.
    store.attach([ref], "train_0")
    store.commit([ref])
    task.cancel()
    acknowledge.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    write.assert_not_awaited()
    assert inventory.state["objects"][ref.record]["partitions"] == ["train_0"]
    store.retire_partition("train_0")
    assert not body.blocks


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

    def gated_read(layers: list) -> None:
        if layers[0][0][0][0].record == second_ref.record:
            started.set()
            assert resume.wait(5), "Next-microbatch read was never released"
        return read_layers(layers)

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
