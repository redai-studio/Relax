# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Parallel publication contracts, isolating only external body transport."""

from concurrent.futures import ThreadPoolExecutor
from queue import Empty, Queue
from threading import Barrier, Event, Lock, get_ident
from types import SimpleNamespace

import numpy as np
import pytest

from relax.distributed.replay_manager.config import ReplayConfig
from relax.distributed.replay_manager.core import ReplayInventory
from relax.distributed.replay_manager.producer import ProducerChunkWriter
from relax.distributed.replay_manager.storage import TQBodyBackend, TQReplayStorage
from tests.cpu.distributed.replay_manager.test_reference import MemoryBody


def test_engine_replacement_reclaims_acknowledged_unsealed_replay():
    body = MemoryBody()
    rows = np.array([[1], [2]], dtype=np.int32)
    config = ReplayConfig(delivery="reference", run_id="a", chunk_rows=2, capacity_bytes=3 * rows.nbytes)
    inventory = ReplayInventory(config, body.delete)
    store = TQReplayStorage(config, inventory=inventory)
    store._backend = body

    old, survivor, replacement = (0, "old"), (1, "survivor"), (0, "replacement")
    store.invoke("register_engine_publisher", *old)
    store.invoke("register_engine_publisher", *survivor)

    def publish(kind, publisher):
        publication = store.begin_publication(kind=kind, layers=1, topk=1, engine_publisher=publisher)
        publication.append(len(rows), lambda _: rows.copy())
        publication._futures[0].result(timeout=5)
        return publication

    try:
        handed_off = publish("moe", old).finish([10, 11, 12])
        store.attach([handed_off], "train_0")
        orphan = publish("moe", old)
        live = publish("moe", survivor)

        store.invoke("register_engine_publisher", *replacement)
        assert f"{orphan.record}:0:0" not in body.blocks
        assert f"{live.record}:0:0" in body.blocks
        with pytest.raises(RuntimeError, match="generation has been retired"):
            store.invoke("register_engine_publisher", *old)
        with pytest.raises(RuntimeError, match="generation has been retired"):
            store.begin_publication(kind="moe", layers=1, topk=1, engine_publisher=old)

        # Rotation preserves an adopted response while the queue put completes.
        store.commit([handed_off])
        store.validate_partition([handed_off], "train_0")
        assert f"{handed_off.record}:0:0" in body.blocks
        renewed = publish("moe", replacement)
        assert f"{renewed.record}:0:0" in body.blocks
    finally:
        store.retire_partition("train_0")
        store.close_run()


def test_engine_replacement_waits_for_sealed_reference_handoff():
    body = MemoryBody()
    rows = np.array([[1], [2]], dtype=np.int32)
    config = ReplayConfig(delivery="reference", run_id="a", chunk_rows=2, capacity_bytes=rows.nbytes)
    store = TQReplayStorage(config, inventory=ReplayInventory(config, body.delete))
    store._backend = body
    old, replacement = (0, "old"), (0, "replacement")
    store.invoke("register_engine_publisher", *old)

    try:
        publication = store.begin_publication(kind="moe", layers=1, topk=1, engine_publisher=old)
        publication.append(len(rows), lambda _: rows.copy())
        ref = publication.finish_async([10, 11, 12]).result(timeout=5)

        for _ in range(2):
            with pytest.raises(RuntimeError, match="sealed references without a confirmed handoff"):
                store.invoke("register_engine_publisher", *replacement)
        assert body.blocks

        store.invoke("adopt_response", [ref], "rollout")
        store.invoke("adopt_response", [ref], "rollout")
        with pytest.raises(RuntimeError, match="another owner"):
            store.invoke("adopt_response", [ref], "late")
        store.invoke("discard", [ref], unadopted=True)
        store.invoke("register_engine_publisher", *replacement)
        assert body.blocks
        store.attach([ref], "train_0")
        store.commit([ref])
    finally:
        store.retire_partition("train_0")
        store.close_run()


def test_engine_replacement_fences_ambiguous_write_until_acknowledged():
    body = MemoryBody()
    config = ReplayConfig(delivery="reference", run_id="a", chunk_rows=2)
    store = TQReplayStorage(config, inventory=ReplayInventory(config, body.delete))
    store._backend = body
    old, replacement = (0, "old"), (0, "replacement")
    store.invoke("register_engine_publisher", *old)
    publication = store.begin_publication(kind="moe", layers=1, topk=1, engine_publisher=old)
    entered, resume = Event(), Event()
    write = body.write_many
    delete = body.delete
    deleted = Event()

    def pause_write(keys, values):
        write(keys, values)
        entered.set()
        assert resume.wait(5)

    def note_delete(keys):
        delete(keys)
        deleted.set()

    body.write_many = pause_write
    store._inventory._delete = note_delete
    try:
        publication.append(2, lambda _: np.array([[1], [2]], dtype=np.int32))
        assert entered.wait(5)
        with pytest.raises(RuntimeError, match="unacknowledged writes; restart the replay run"):
            store.invoke("register_engine_publisher", *replacement)
        with pytest.raises(RuntimeError, match="unacknowledged writes; restart the replay run"):
            store.invoke("register_engine_publisher", *replacement)
        assert body.blocks and not deleted.is_set()

        resume.set()
        publication._futures[0].result(timeout=5)
        assert deleted.is_set() and not body.blocks
        with pytest.raises(RuntimeError, match="publication no longer exists"):
            publication.finish([10, 11, 12])
        store.invoke("register_engine_publisher", *replacement)
    finally:
        resume.set()
        store.close_run()


def test_parallel_publications_seal_independently_abort_drains_and_preserve_codecs():
    body = MemoryBody()
    config = ReplayConfig(delivery="reference", run_id="a", chunk_rows=2, publication_bytes=32768)
    inventory = ReplayInventory(config, body.delete)
    store = TQReplayStorage(config, inventory=inventory)
    store._backend = body
    specs = [("moe", 2, "int32", 2), ("moe", 2, "int32", 2), ("moe", 3, "int32", 3)]
    publications = [
        store.begin_publication(kind=kind, layers=1, topk=topk, codec=codec) for kind, topk, codec, _ in specs
    ]
    media_record = "d" * 32
    media_claim = store.invoke("claim_media", "a" * 64, [1, 2, 2], media_record, "image")
    publications.append(
        store.begin_publication(
            kind="mm_embedding", layers=1, topk=2560, codec="bf16-raw-v1", record=media_record, media_hash="a" * 64
        )
    )
    expected = [np.arange(2 * width, dtype=np.int32).reshape(2, width) for _, _, _, width in specs]
    expected.append(np.arange(5120, dtype=np.uint8).reshape(1, 5120))
    resume = {publication.record: Event() for publication in publications}
    entered, lock = set(), Lock()
    all_writers = Event()
    original_write = body.write_many

    def write(keys, values):
        record = keys[0].split(":")[0]
        with lock:
            entered.add(record)
            if len(entered) == 4:
                all_writers.set()
        assert resume[record].wait(5)
        original_write(keys, values)

    body.write_many = write
    try:
        inputs = [values.copy() for values in expected]
        for publication, values in zip(publications, inputs, strict=True):
            publication.append(len(values), lambda _, values=values: values.copy(), encoded_width=values.shape[1])
        assert all_writers.wait(5), "publications did not write concurrently"
        for values in inputs:
            values.fill(99)  # Reused capture inputs must not mutate admitted snapshots.
        sealed = publications[0].finish_async([10, 11, 12])
        assert not sealed.done()
        resume[publications[0].record].set()
        assert sealed.result(timeout=5).codec == "int32"
        assert all(not resume[p.record].is_set() for p in publications[1:])
        with ThreadPoolExecutor(max_workers=1) as executor:
            aborting = Event()

            def abort():
                aborting.set()
                publications[1].abort()

            aborted = executor.submit(abort)
            try:
                assert aborting.wait(5)
                assert not aborted.done()
            finally:
                resume[publications[1].record].set()
            aborted.result(timeout=5)
        assert not any(key.startswith(publications[1].record) for key in body.blocks)
        resume[publications[2].record].set()
        assert publications[2].finish([10, 11, 12]).codec == "int32"
        resume[publications[3].record].set()
        assert publications[3].finish_media("a" * 64).codec == "bf16-raw-v1"
        for index in (0, 2, 3):
            np.testing.assert_array_equal(body.blocks[f"{publications[index].record}:0:0"], expected[index])
        assert store._writer.peak_bytes <= config.publication_bytes
    finally:
        for event in resume.values():
            event.set()
        store.invoke("rollback_acquire", [media_claim["lease"]])
        for publication in publications:
            publication.abort()
        store._writer.close()
    assert not body.blocks and not inventory.state["objects"]


def test_writer_reserves_shared_bytes_before_taking_another_snapshot():
    writer = ProducerChunkWriter(32)
    resume, started, snapshot_taken = Event(), Event(), Event()
    values = np.arange(4, dtype=np.int32)
    first = [writer.submit(16, values.copy, lambda _: resume.wait(5)) for _ in range(2)]

    def submit():
        started.set()

        def snapshot():
            snapshot_taken.set()
            return values.copy()

        return writer.submit(16, snapshot, lambda _: None)

    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            pending = executor.submit(submit)
            try:
                assert started.wait(5)
                assert not snapshot_taken.wait(0.05)
            finally:
                resume.set()
            pending.result(timeout=5).result(timeout=5)
        for future in first:
            future.result(timeout=5)
        assert snapshot_taken.is_set()
        assert writer.peak_bytes == 32
    finally:
        resume.set()
        writer.close()
    assert writer.pending_bytes == 0


def test_ambiguous_body_write_failure_retains_unacknowledged_keys_for_recovery():
    body = MemoryBody()
    config = ReplayConfig(delivery="reference", run_id="a")
    inventory = ReplayInventory(config, body.delete)
    store = TQReplayStorage(config, inventory=inventory)
    store._backend = body
    original_write = body.write_many

    def fail_after_put(keys, values):
        original_write(keys, values)
        raise RuntimeError("injected lost put reply")

    body.write_many = fail_after_put
    publication = store.begin_publication(kind="moe", layers=1, topk=2)
    try:
        publication.append(2, lambda _: np.ones((2, 2), dtype=np.int32))
        with pytest.raises(RuntimeError, match="lost put reply"):
            publication.finish([10, 11, 12])
        assert body.blocks
        assert not inventory.state["objects"][publication.record]["ready"]
        with pytest.raises(RuntimeError, match="acknowledgement is uncertain"):
            publication.abort()
        assert body.blocks and publication.record in inventory.state["objects"]
    finally:
        store._writer.close()
    assert store._writer.pending_bytes == 0


def test_parallel_body_writes_have_independent_clients_and_close_retries_failures():
    clients, barrier = [], Barrier(4)

    class Client:
        def __init__(self):
            self.owner = get_ident()
            self.closes = 0
            self.blocks = []
            clients.append(self)

        def kv_retrieve_meta(self, keys, **kwargs):
            assert get_ident() == self.owner
            return SimpleNamespace(size=len(keys))

        def put(self, data, metadata):
            assert get_ident() == self.owner
            barrier.wait(timeout=5)
            self.blocks.append(data["replay_body"].clone())

        def close(self):
            self.closes += 1
            if self is clients[0] and self.closes == 1:
                raise RuntimeError("injected client close failure")

    backend = TQBodyBackend("a", Client)
    values = (np.arange(6, dtype=np.int32).reshape(2, 3), np.ones((2, 3), dtype=np.int32))
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(backend.write_many, [f"{i}:0", f"{i}:1"], values) for i in range(4)]
        for future in futures:
            future.result(timeout=5)
    assert len(clients) == 4
    for client in clients:
        wire = client.blocks[0].numpy()
        assert wire.dtype == np.uint8 and wire.nbytes == sum(value.nbytes for value in values)
        np.testing.assert_array_equal(wire.view(np.int32), np.stack(values))
    with pytest.raises(RuntimeError, match="client close failure"):
        backend.close()
    assert all(client.closes == 1 for client in clients)
    backend.close()
    assert clients[0].closes == 2 and all(client.closes == 1 for client in clients[1:])
    assert not backend._write_clients


@pytest.mark.parametrize("storage_backend,client_count", [("MooncakeStore", 1), ("SimpleStorage", 4)])
def test_publication_limits_body_clients_for_resolved_backend(monkeypatch, storage_backend, client_count):
    created, resume = Queue(), Event()
    writes = []

    class Client:
        def __init__(self):
            created.put(self)

        def kv_retrieve_meta(self, keys, **kwargs):
            return SimpleNamespace(size=len(keys), keys=keys)

        def put(self, data, metadata):
            assert resume.wait(5)
            writes.extend(metadata.keys)

        def close(self):
            pass

    config = ReplayConfig(delivery="reference", run_id="a")
    inventory = ReplayInventory(config, lambda _: None)
    store = TQReplayStorage(config, client_factory=Client)
    connection = SimpleNamespace(backend=SimpleNamespace(storage_backend=storage_backend))

    def invoke(method, *args, **kwargs):
        if method == "get_connection":
            return connection
        with inventory._lock:
            return getattr(inventory, method)(*args, **kwargs)

    monkeypatch.setattr(store, "invoke", invoke)
    publications = [store.begin_publication(kind="moe", layers=1, topk=2) for _ in range(4)]
    try:
        for publication in publications:
            publication.append(2, lambda _: np.ones((2, 2), dtype=np.int32))
        for _ in range(client_count):
            created.get(timeout=5)
        if client_count == 1:
            with pytest.raises(Empty):
                created.get(timeout=0.2)
        resume.set()
        refs = [publication.finish([10, 11, 12]) for publication in publications]
        assert sorted(writes) == sorted(f"{ref.record}:0:0" for ref in refs)
        assert created.empty()
    finally:
        resume.set()
        store._writer.close()
        store._backend.close()


@pytest.mark.parametrize("positions", [[0, 2, 4], [4, 2, 0], [0, 1, 2], [1, 4], [3]])
@pytest.mark.parametrize("dtype", ["int32", "uint8"])
def test_body_strided_selection_matches_original_tensor_indexing(positions, dtype):
    import torch

    from relax.distributed.replay_manager.storage import _select_body_rows

    values = torch.arange(72, dtype=getattr(torch, dtype)).reshape(6, 3, 4)
    actual = _select_body_rows(values, positions, lambda data, rows: data[rows])
    torch.testing.assert_close(actual, values[positions])


def test_concurrent_first_publication_shares_one_writer_and_byte_budget(monkeypatch):
    from relax.distributed.replay_manager import producer

    started, resume, second, duplicate = Event(), Event(), Event(), Event()
    writers = []
    original = producer.ProducerChunkWriter

    class DelayedWriter(original):
        def __init__(self, capacity, *, max_workers):
            super().__init__(capacity, max_workers=max_workers)
            writers.append(self)
            if len(writers) == 1:
                started.set()
                assert resume.wait(5)
            else:
                duplicate.set()

    monkeypatch.setattr(producer, "ProducerChunkWriter", DelayedWriter)
    config = ReplayConfig(delivery="reference", run_id="a")
    store = TQReplayStorage(config, inventory=ReplayInventory(config, lambda _: None))

    def begin_again():
        second.set()
        return store.begin_publication(kind="moe", layers=1, topk=2)

    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            first = executor.submit(store.begin_publication, kind="moe", layers=1, topk=2)
            try:
                assert started.wait(5)
                other = executor.submit(begin_again)
                assert second.wait(5)
                assert not duplicate.wait(0.05)
            finally:
                resume.set()
            one, two = first.result(timeout=5), other.result(timeout=5)
        assert one.writer is two.writer
        one.abort()
        two.abort()
    finally:
        resume.set()
        for writer in writers:
            writer.close()


def test_media_and_route_bodies_share_a_tq_field_without_dtype_conflicts():
    from transfer_queue.controller import FieldMeta

    schema, bodies = FieldMeta(), {}

    class Client:
        def kv_retrieve_meta(self, *, keys, **kwargs):
            metadata = SimpleNamespace(size=len(keys), keys=keys)
            metadata.select_fields = lambda fields: metadata
            return metadata

        def put(self, data, metadata):
            values = data["replay_body"]
            schema.update(
                {"dtype": values.dtype, "shape": tuple(values.shape[1:])},
                list(range(len(bodies), len(bodies) + len(values))),
            )
            bodies.update(zip(metadata.keys, values.unbind(), strict=True))

        def get_data(self, metadata):
            return {"replay_body": [bodies[key] for key in metadata.keys]}

        def close(self):
            pass

    backend = TQBodyBackend("mixed", Client)
    config = ReplayConfig(delivery="reference", run_id="a", chunk_rows=2, capacity_bytes=26)
    inventory = ReplayInventory(config, lambda _: None)
    store = TQReplayStorage(config, inventory=inventory)
    store._backend = backend
    media = np.arange(32, dtype=np.uint8).reshape(2, 16)
    routes = np.array([[1, 5], [7, 3]], dtype=np.int32)
    compact = np.array([[0, 127, 128, 255, 3], [255, 128, 127, 0, 1]], dtype=np.uint8)
    try:
        backend.write_many(["media"], (media,))
        refs = []
        for codec, values in (("int32", routes), ("uint8", compact.astype(np.int32))):
            publication = store.begin_publication(kind="moe", layers=1, topk=values.shape[1], codec=codec)
            publication.append(2, lambda _, values=values: values.copy())
            refs.append(publication.finish([10, 11, 12]))
        actual_media, actual_routes, actual_compact = backend.read(["media", *(f"{ref.record}:0:0" for ref in refs)])
        assert all(value.dtype == np.uint8 for value in (actual_media, actual_routes, actual_compact))
        assert actual_compact.nbytes == 10  # No alignment padding for an odd top-k.
        np.testing.assert_array_equal(actual_media, media)
        np.testing.assert_array_equal(actual_routes.view(np.int32), routes)
        np.testing.assert_array_equal(actual_compact, compact)
        assert inventory._used_bytes == 26  # 16 raw bytes plus 10 compact bytes.
        for ref, expected in zip(refs, (routes, compact), strict=True):
            output = np.empty((2, ref.topk), dtype=np.int64)
            with store.reader([ref]) as session:
                session.read_into(0, 0, [1, 0], output, [0, 1])
            np.testing.assert_array_equal(output, expected[[1, 0]])
    finally:
        if store._writer is not None:
            store._writer.close()
        backend.close()


def test_publication_sealing_wait_does_not_occupy_body_writer():
    writer = ProducerChunkWriter(32, max_workers=1)
    sealing, release, written = Event(), Event(), Event()

    def complete():
        sealing.set()
        assert release.wait(5)
        return "sealed"

    try:
        result = writer.submit_completion(complete)
        assert sealing.wait(5)
        body = np.ones((2, 2), dtype=np.int32)
        write = writer.submit(body.nbytes, body.copy, lambda _: written.set())
        assert written.wait(1), "A pending seal occupied the body transport worker"
        write.result(timeout=2)
        assert not result.done()
        release.set()
        assert result.result(timeout=2) == "sealed"
    finally:
        release.set()
        writer.close()
    assert writer.pending_bytes == 0
