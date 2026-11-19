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


def test_parallel_publications_seal_independently_abort_drains_and_preserve_codecs():
    body = MemoryBody()
    config = ReplayConfig(delivery="reference", run_id="a", chunk_rows=2, publication_bytes=32768)
    inventory = ReplayInventory(config, body.delete)
    store = TQReplayStorage(config, inventory=inventory)
    store._backend = body
    specs = [("moe", 2, "int32", 2), ("indexer", 2, "int32", 2), ("indexer", 2048, "keye-bitmap-v1", 3)]
    publications = [
        store.begin_publication(kind=kind, layers=1, topk=topk, codec=codec) for kind, topk, codec, _ in specs
    ]
    media_record = "d" * 32
    media_claim = store.invoke("claim_media", "a" * 64, [1, 2, 2], media_record, "image")
    publications.append(
        store.begin_publication(kind="mm_embedding", layers=1, topk=2560, codec="bf16-raw-v1", record=media_record)
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
        assert publications[2].finish([10, 11, 12]).codec == "keye-bitmap-v1"
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
        np.testing.assert_array_equal(client.blocks[0].numpy(), np.stack(values))
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
def test_body_strided_selection_matches_original_tensor_indexing(positions):
    import torch

    from relax.distributed.replay_manager.storage import _select_body_rows

    values = torch.arange(72, dtype=torch.int32).reshape(6, 3, 4)
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
            first = executor.submit(store.begin_publication, kind="indexer", layers=1, topk=2)
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
