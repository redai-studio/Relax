# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Node-shared frozen-condition store: exactness, bounds, faults and
lifecycle."""

import os
import socket
import time
from datetime import timedelta
from pathlib import Path

import pytest
import torch

from relax.models import condition_cache as cc
from relax.models.condition_cache import (
    ConditionFeatureCache,
    SharedConditionStore,
    all_ranks_hit,
    build_condition_cache,
    model_files_identity,
)


def _store(root, run_id="run", **kwargs):
    kwargs.setdefault("max_bytes", 1 << 20)
    return SharedConditionStore(root=root, run_id=run_id, **kwargs)


def _payload(fill=1.0):
    return {
        "kv": (torch.full((2, 3), fill, dtype=torch.bfloat16), torch.arange(5, dtype=torch.int64)),
        "mask": None,
        "flags": [torch.tensor([True, False]), torch.zeros(0, dtype=torch.float16)],
        "scalar": torch.tensor(fill, dtype=torch.float32),
    }


def _assert_tree_equal(left, right):
    assert type(left) is type(right)
    if isinstance(left, torch.Tensor):
        assert left.dtype == right.dtype and left.shape == right.shape and torch.equal(left, right)
    elif isinstance(left, dict):
        assert list(left) == list(right)
        for key in left:
            _assert_tree_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            _assert_tree_equal(a, b)
    else:
        assert left == right


def test_shared_store_roundtrip_is_bit_exact_private_and_shared_across_ranks(tmp_path):
    writer, reader = _store(tmp_path), _store(tmp_path)  # two ranks of one node
    payload = _payload(0.3)
    expected = cc._copy_tree(payload, None)
    assert writer.put("sample", payload) and writer.flush()
    assert reader.contains("sample")
    first = reader.get("sample")
    _assert_tree_equal(expected, first)
    first["kv"][0].add_(1)  # private copy: later readers are unaffected
    _assert_tree_equal(expected, reader.get("sample"))
    assert writer.stats()["committed"] == 1 and reader.stats()["hits"] == 2
    assert writer.put("sample", payload) and writer.stats()["deduplicated"] == 1
    writer.close()
    reader.close()


def test_staged_entry_survives_eviction_after_hit_agreement(tmp_path):
    store = _store(tmp_path)
    assert store.put("sample", _payload()) and store.flush()
    assert store.contains("sample")  # the probe preceding the collective
    for path in (tmp_path / "run").glob("*.rcc"):
        path.unlink()  # another rank evicts it before this rank reads
    _assert_tree_equal(cc._copy_tree(_payload(), None), store.get("sample"))
    assert not store.contains("sample")
    store.close()


def test_shared_store_lru_bounds_whole_node_directory(tmp_path):
    probe = _store(tmp_path, run_id="probe")
    probe.put("a", {"x": torch.zeros(1000, dtype=torch.uint8)})
    probe.flush()
    size = probe.stats()["bytes"]
    probe.close()
    writer, other = _store(tmp_path, max_bytes=2 * size), _store(tmp_path, max_bytes=2 * size)
    for index, key in enumerate("abc"):
        store = writer if index % 2 == 0 else other  # the budget is shared
        assert store.put(key, {"x": torch.full((1000,), index, dtype=torch.uint8)}) and store.flush()
        time.sleep(0.02)
        if key == "b":
            assert writer.contains("a")  # refresh "a"; "b" becomes LRU
            time.sleep(0.02)
    assert sorted(writer.keys()) == ["a", "c"]
    assert writer.stats()["bytes"] <= 2 * size
    assert writer.stats()["evictions"] == 1
    capped = _store(tmp_path, run_id="capped", max_entries=1)
    for key in "ab":
        capped.put(key, {"x": torch.ones(1)})
        capped.flush()
    assert capped.keys() == ("b",)
    for store in (writer, other, capped):
        store.close()


@pytest.mark.parametrize("damage", ["truncate", "garbage", "rename", "trailer"])
def test_corrupted_or_mismatched_entry_is_a_miss_and_deleted(tmp_path, damage):
    store = _store(tmp_path)
    store.put("sample", _payload())
    store.flush()
    path = tmp_path / "run" / "sample.rcc"
    data = path.read_bytes()
    if damage == "truncate":
        path.write_bytes(data[: len(data) // 2])
    elif damage == "garbage":
        path.write_bytes(b"RCC1" + b"\xff" * 64)
    elif damage == "trailer":
        path.write_bytes(data[:-4] + b"XXXX")
    else:
        # A file whose embedded key differs from its name is never served.
        os.replace(path, tmp_path / "run" / "other.rcc")
    key = "other" if damage == "rename" else "sample"
    assert store.get(key) is None
    assert not (tmp_path / "run" / f"{key}.rcc").exists()
    assert store.stats()["corruptions"] == 1
    store.close()


def test_missing_oversized_and_staging_limits_fall_back_online(tmp_path):
    store = _store(tmp_path, max_bytes=256, staging_bytes=128)
    assert store.get("missing") is None and store.stats()["misses"] == 1
    assert not store.put("big", {"x": torch.zeros(1024, dtype=torch.uint8)})
    assert store.stats()["oversized"] == 1
    gate = store._pending_lock
    with gate:  # a full staging budget drops instead of queueing unbounded
        store._pending_bytes = store.staging_bytes
    assert not store.put("small", {"x": torch.zeros(8, dtype=torch.uint8)})
    assert store.stats()["write_dropped"] == 1
    with gate:
        store._pending_bytes = 0
    with pytest.raises(ValueError, match="detached"):
        store.put("grad", {"x": torch.ones(2, requires_grad=True)})
    store.close()


def test_clear_drops_entries_and_late_writes(tmp_path):
    store = _store(tmp_path)
    store.put("a", {"x": torch.ones(2)})
    store.flush()
    stale = cc._PendingWrite("late", {"t": 0}, [torch.ones(2)], None, 64, store._generation)
    store.clear()
    store._pending_bytes += stale.size
    store._queue.put(stale)  # encoded before the invalidation
    assert store.flush()
    assert store.keys() == () and store.stats()["entries"] == 0
    store.close()


def test_last_owner_removes_run_directory_and_close_is_idempotent(tmp_path):
    first, second = _store(tmp_path), _store(tmp_path)
    first.put("a", {"x": torch.ones(2)})
    first.flush()
    first.close()
    assert second.contains("a")  # another live rank keeps the directory
    second.close()
    second.close()
    assert not (tmp_path / "run").exists()
    with pytest.raises(RuntimeError, match="closed"):
        second.put("a", {"x": torch.ones(2)})


@pytest.mark.skipif(os.name == "nt", reason="PID liveness probing is POSIX-only")
def test_stale_run_directories_of_dead_owners_are_reaped(tmp_path):
    import subprocess
    import sys

    dead = subprocess.run([sys.executable, "-c", "import os; print(os.getpid())"], capture_output=True, text=True)
    stale = tmp_path / "crashed" / "owners"
    stale.mkdir(parents=True)
    (stale / f"{socket.gethostname()}-{int(dead.stdout)}-deadbeef").touch()
    (tmp_path / "crashed" / "x.rcc").write_bytes(b"leftover")
    live = tmp_path / "foreign" / "owners"
    live.mkdir(parents=True)
    (live / "other-host-1-cafe").touch()
    store = _store(tmp_path)
    assert not (tmp_path / "crashed").exists()
    assert (tmp_path / "foreign").exists()
    store.close()


def _lifecycle_first(root, kind, gap, proceed, finish, results):
    root = Path(root)
    store = SharedConditionStore(1 << 20, root)
    try:
        if kind == "start":
            touch = Path.touch

            def paused_touch(path, *args, **kwargs):
                if path.parent == root / "first" / "owners":
                    gap.set()
                    assert proceed.wait(20), "Owner registration was not released"
                return touch(path, *args, **kwargs)

            Path.touch = paused_touch
            store.bind(run_id="first")
            results.put(("first", "ok"))
            assert finish.wait(20), "First owner was not released"
        else:
            rmtree = cc.shutil.rmtree

            def paused_rmtree(path, *args, **kwargs):
                if Path(path) == root / "same":
                    gap.set()
                    assert proceed.wait(20), "Last-owner removal was not released"
                return rmtree(path, *args, **kwargs)

            cc.shutil.rmtree = paused_rmtree
            store.bind(run_id="same")
            assert store.put("old", {"x": torch.ones(2)}) and store.flush()
            store.close()
            results.put(("first", "ok"))
    except Exception as error:
        results.put(("first", repr(error)))
    finally:
        store.close()


def _lifecycle_second(root, kind, bound, check, observations, results):
    root = Path(root)
    enter = cc._NodeLock.__enter__

    def observed_enter(lock):
        if lock._path.parent == root:
            # With a lifecycle lock this is reached before blocking on the
            # first owner. Without one, bind completes in the forced gap.
            observations.put("root-lock-attempt")
        return enter(lock)

    cc._NodeLock.__enter__ = observed_enter
    store = SharedConditionStore(1 << 20, root)
    try:
        store.bind(run_id="second" if kind == "start" else "same")
        bound.set()
        observations.put("bound")
        assert check.wait(20), "Second owner was not released"
        assert store.put("new", {"x": torch.arange(4)}) and store.flush()
        assert store.contains("new")
        assert torch.equal(store.get("new")["x"], torch.arange(4))
        results.put(("second", "ok"))
    except Exception as error:
        results.put(("second", repr(error)))
    finally:
        store.close()


def _exercise_lifecycle_interleaving(tmp_path, kind):
    import multiprocessing as mp

    context = mp.get_context("spawn")
    gap, proceed, finish, bound, check = (context.Event() for _ in range(5))
    observations, results = context.Queue(), context.Queue()
    root = tmp_path / "cache"
    first = context.Process(target=_lifecycle_first, args=(root, kind, gap, proceed, finish, results))
    second = context.Process(target=_lifecycle_second, args=(root, kind, bound, check, observations, results))
    started = []
    try:
        first.start()
        started.append(first)
        assert gap.wait(20), "First process did not reach the lifecycle gap"
        second.start()
        started.append(second)
        # Event-driven in both implementations: old bind registers/reaps
        # during the gap; fixed bind first attempts the held root lock.
        assert observations.get(timeout=20) in ("bound", "root-lock-attempt")
        proceed.set()
        assert bound.wait(20), "Second process did not bind"
        first_result = results.get(timeout=20)
        check.set()
        second_result = results.get(timeout=20)
        finish.set()
        for process in started:
            process.join(20)
            assert not process.is_alive(), "Lifecycle regression worker hung"
            assert process.exitcode == 0
        assert dict([first_result, second_result]) == {"first": "ok", "second": "ok"}
        assert not (root / "first").exists()
        assert not (root / "second").exists()
        assert not (root / "same").exists()
    finally:
        proceed.set()
        check.set()
        finish.set()
        for process in started:
            if process.is_alive():
                process.terminate()
            process.join(5)
        for channel in (observations, results):
            channel.close()
            channel.join_thread()


@pytest.mark.skipif(os.name == "nt", reason="Cross-process lifecycle locks require POSIX")
def test_concurrent_start_keeps_owner_registration_directory(tmp_path):
    _exercise_lifecycle_interleaving(tmp_path, "start")


@pytest.mark.skipif(os.name == "nt", reason="Cross-process lifecycle locks require POSIX")
def test_last_owner_close_cannot_remove_concurrent_bind(tmp_path):
    _exercise_lifecycle_interleaving(tmp_path, "close")


def test_factory_selects_storage_and_disables_at_zero(tmp_path, monkeypatch):
    monkeypatch.setenv("RELAX_CONDITION_CACHE_DIR", str(tmp_path))
    assert build_condition_cache(0) is None
    assert isinstance(build_condition_cache(1024, "cpu"), ConditionFeatureCache)
    monkeypatch.delenv("RELAX_CONDITION_CACHE_DEVICE", raising=False)
    shared = build_condition_cache(1024)
    assert isinstance(shared, SharedConditionStore) and shared.root == tmp_path
    shared.close()
    with pytest.raises(ValueError, match="negative"):
        build_condition_cache(-1)


def test_model_identity_hashes_small_files_and_stats_large_shards(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "config.json").write_text("{}")
    shard = tmp_path / "sub" / "model.safetensors"
    shard.write_bytes(b"0" * 32)
    base = model_files_identity(tmp_path, small_file_bytes=16)
    assert base == model_files_identity(tmp_path, small_file_bytes=16)
    (tmp_path / "config.json").write_text('{"a": 1}')
    changed = model_files_identity(tmp_path, small_file_bytes=16)
    assert changed != base
    shard.write_bytes(b"1" * 33)  # large files are identified by size/mtime
    assert model_files_identity(tmp_path, small_file_bytes=16) != changed
    with pytest.raises(FileNotFoundError):
        model_files_identity(tmp_path / "missing")


def _epoch_swap_worker(rank, rendezvous, root, shared, output):
    """Epoch 0: rank r encodes sample r. Epoch 1: samples swap ranks.

    Mirrors ImageBatchSampler reshuffling a sample onto another DP rank.
    """
    import torch.distributed as dist

    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=30))
    try:
        if shared:
            cache = SharedConditionStore(1 << 20, root)
            cache.bind(dist.group.WORLD)
        else:
            cache = ConditionFeatureCache(1 << 20)
        decisions, encodes = [], 0
        for epoch in range(2):
            key = f"sample-{(rank + epoch) % 2}"
            hit = all_ranks_hit(cache.contains(key), dist.group.WORLD)
            decisions.append(hit)
            if hit:
                assert torch.equal(cache.get(key)["kv"], torch.full((2,), float((rank + epoch) % 2)))
            else:
                feature = torch.full((2,), float((rank + epoch) % 2))
                dist.all_reduce(torch.zeros(1), group=dist.group.WORLD)  # stands in for FSDP all-gathers
                cache.put(key, {"kv": feature})
                encodes += 1
                if shared:
                    assert cache.flush()
            dist.barrier(group=dist.group.WORLD)
        torch.save({"decisions": decisions, "encodes": encodes}, f"{output}-{rank}.pt")
        cache.close()
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not torch.distributed.is_gloo_available() or os.name == "nt", reason="CPU Gloo multi-process is unavailable"
)
@pytest.mark.parametrize("shared", [True, False])
def test_samples_moving_across_ranks_hit_only_with_node_shared_store(tmp_path, shared):
    import torch.multiprocessing as mp

    rendezvous = (tmp_path / "rendezvous").as_uri()
    root = tmp_path / "shm"
    mp.spawn(_epoch_swap_worker, args=(rendezvous, str(root), shared, str(tmp_path / "out")), nprocs=2, join=True)
    for rank in range(2):
        result = torch.load(tmp_path / f"out-{rank}.pt", weights_only=True)
        expected = [False, True] if shared else [False, False]
        assert result == {"decisions": expected, "encodes": 2 - sum(expected)}
    # The stable lifecycle-lock inode remains; no run data/owners remain.
    assert not root.exists() or all(path.name == cc._ROOT_LOCK and path.is_file() for path in root.iterdir())
