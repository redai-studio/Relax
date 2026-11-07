"""Quota-backed control credits survive restarts, competing writers and
faults."""

import multiprocessing
import os
import threading
import time
from pathlib import Path

import pytest

from relax.distributed.weight_sync import store as module
from relax.distributed.weight_sync.deadline import time_budget
from relax.distributed.weight_sync.store import VersionStore, stored_bytes, writer_lock


def reopen(root, budget=1 << 26):
    return VersionStore(root, disk_budget=budget, max_weight_bytes=1 << 20)


@pytest.fixture
def source(tmp_path):
    value = tmp_path / "weight"
    value.write_bytes(bytes(64))
    store = reopen(tmp_path / "store")
    target = store.publish("s", 0, {"w": ("U8", [64], value)})
    return store, target


def test_restarted_controls_consume_counted_credit_without_object_scans(source, monkeypatch):
    store, target = source
    store.register("reader", "s")
    before = stored_bytes(store.root)
    original = module.stored_bytes
    scans = []

    def scan(path):
        scans.append(path)
        return original(path)

    monkeypatch.setattr(module, "stored_bytes", scan)
    for n in range(20):
        reopen(store.root).pin("s", 0, f"owner-{n}")
    reopen(store.root).acknowledge(
        "reader", {"phase": "ACTIVE", "target": {k: target[k] for k in ("stream", "version", "root")}}
    )
    assert not scans
    assert original(store.root) <= before
    assert reopen(store.root).consumer_status("reader")["cursor"] == 0


def _pin_many(root, prefix):
    for n in range(8):
        reopen(root).pin("s", 0, f"{prefix}-{n}")


def test_competing_processes_never_double_spend(source):
    store, _ = source
    store.register("reader", "s")
    before = stored_bytes(store.root)
    context = multiprocessing.get_context("spawn")
    workers = [context.Process(target=_pin_many, args=(store.root, str(n))) for n in range(3)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(15)
        assert not worker.is_alive()
        assert worker.exitcode == 0
    assert len(list((store.root / "pins").glob("*.json"))) == 24
    assert stored_bytes(store.root) <= before <= store.disk_budget


def test_crash_after_durable_debit_cannot_reuse_credit(source, monkeypatch):
    store, _ = source
    store.register("reader", "s")
    credit = store.root / ".control-reservation-v1"
    before = credit.stat().st_size
    original = module.atomic_write

    def fail(path, data):
        if Path(path).parent.name == "pins":
            raise OSError("injected metadata failure")
        return original(path, data)

    monkeypatch.setattr(module, "atomic_write", fail)
    with pytest.raises(OSError, match="injected metadata"):
        store.pin("s", 0, "broken")
    after = credit.stat().st_size
    assert after < before and not (store.root / "pins/broken.json").exists()
    monkeypatch.setattr(module, "atomic_write", original)
    reopen(store.root).pin("s", 0, "recovered")
    assert credit.stat().st_size < after
    assert stored_bytes(store.root) <= store.disk_budget


@pytest.mark.parametrize("damage", ["missing", "partial", "exhausted"])
def test_missing_partial_or_exhausted_credit_refills_only_after_exact_scan(source, monkeypatch, damage):
    store, _ = source
    store.register("reader", "s")
    path = store.root / ".control-reservation-v1"
    if damage == "missing":
        path.unlink()
    elif damage == "partial":
        path.write_bytes(b"partial")
    else:
        header = path.read_bytes().split(b"\n", 1)[0] + b"\n"
        path.write_bytes(header)
    original, scans = module.stored_bytes, []

    def scan(root):
        scans.append(root)
        return original(root)

    monkeypatch.setattr(module, "stored_bytes", scan)
    reopen(store.root).pin("s", 0, "recovered")
    assert scans == [store.root]
    assert stored_bytes(store.root) <= store.disk_budget


def test_smaller_quota_does_not_reuse_old_credit(source):
    store, _ = source
    store.register("reader", "s")
    smaller = reopen(store.root, stored_bytes(store.root))
    with pytest.raises(OSError, match="budget exhausted"):
        smaller.pin("s", 0, "cannot-fit")
    assert not (store.root / "pins/cannot-fit.json").exists()


def test_bulk_admission_counts_prepaid_metadata(source):
    store, _ = source
    store.register("reader", "s")
    remaining = store.disk_budget - stored_bytes(store.root)
    other = reopen(store.root)
    with writer_lock(store.root / ".writer"):
        with pytest.raises(OSError, match="budget exhausted"):
            other._reserve(remaining + 1)


def test_tight_quota_uses_normal_admission_without_reservation(source):
    store, _ = source
    before = stored_bytes(store.root)
    exact = reopen(store.root, before + 128)
    exact.pin("s", 0, "small")
    assert not (store.root / ".control-reservation-v1").exists()
    assert stored_bytes(store.root) <= exact.disk_budget


@pytest.mark.parametrize("alias", ["hardlink", "symlink"])
def test_reservation_cannot_truncate_an_aliased_file(source, tmp_path, alias):
    store, _ = source
    store.register("reader", "s")
    path = store.root / ".control-reservation-v1"
    external = tmp_path / "external"
    if alias == "hardlink":
        os.link(path, external)
    else:
        external.write_bytes(path.read_bytes())
        path.unlink()
        path.symlink_to(external)
    before = external.read_bytes()
    with pytest.raises(OSError):
        reopen(store.root).pin("s", 0, "refuse-alias")
    assert external.read_bytes() == before


def test_outside_growth_does_not_grant_extra_control_capacity(source):
    store, _ = source
    store.register("reader", "s")
    outside = store.root / "unmanaged"
    outside.write_bytes(bytes(4096))
    before = stored_bytes(store.root)
    reopen(store.root).pin("s", 0, "already-admitted")
    assert stored_bytes(store.root) == before


def test_slow_credit_fsync_cannot_acknowledge_past_the_operation_deadline(source, monkeypatch):
    store, _ = source
    store.register("reader", "s")
    original = module.os.fsync

    def slow(fd):
        time.sleep(0.03)
        return original(fd)

    monkeypatch.setattr(module.os, "fsync", slow)
    with time_budget(0.01), pytest.raises(TimeoutError, match="control metadata reservation"):
        store.pin("s", 0, "timed-out")
    assert not (store.root / "pins/timed-out.json").exists()


def test_first_refill_waits_outside_metadata_lock_for_bulk_admission(source):
    store, _ = source
    entered, finished = threading.Event(), threading.Event()
    failures = []

    def control():
        entered.set()
        try:
            reopen(store.root).pin("s", 0, "concurrent")
        except BaseException as error:
            failures.append(error)
        finally:
            finished.set()

    with writer_lock(store.root / ".content-writer"):
        worker = threading.Thread(target=control)
        worker.start()
        assert entered.wait(1)
        time.sleep(0.05)
        assert not finished.is_set()
        # Publication can finish its manifest transaction without lock inversion.
        with writer_lock(store.root / ".writer", timeout=1):
            assert not (store.root / ".control-reservation-v1").exists()
    worker.join(2)
    assert not failures and finished.is_set()


def test_content_owner_can_refill_while_pinning_and_context_is_released(source):
    store, _ = source
    with writer_lock(store.root / ".content-writer"), time_budget(1):
        store.pin("s", 0, "owned-fetch")
    assert not module._held_content_locks.get()
    assert (store.root / "pins/owned-fetch.json").exists()


def test_bulk_writer_can_progress_while_controls_spend_existing_capacity(source):
    store, _ = source
    store.register("reader", "s")
    before = stored_bytes(store.root)
    failures = []

    def control():
        try:
            reopen(store.root).pin("s", 0, "fast-concurrent")
        except BaseException as error:
            failures.append(error)

    with writer_lock(store.root / ".content-writer"):
        worker = threading.Thread(target=control)
        worker.start()
        worker.join(1)
        assert not worker.is_alive() and not failures
    assert stored_bytes(store.root) == before


@pytest.mark.parametrize("control", ["register", "acknowledge", "unpin"])
def test_same_instance_controls_preserve_inflight_bulk_capacity(source, control):
    store, target = source
    store.register("reader", "s")
    store.pin("s", 0, "fetch-shared-instance")
    receipt = {"phase": "ACTIVE", "target": {key: target[key] for key in ("stream", "version", "root")}}
    actions = {
        "register": lambda: store.register("second-reader", "s"),
        "acknowledge": lambda: store.acknowledge("reader", receipt),
        "unpin": lambda: store.unpin("fetch-shared-instance"),
    }
    failures = []

    def apply_control():
        try:
            actions[control]()
        except BaseException as error:
            failures.append(error)

    with writer_lock(store.root / ".content-writer"):
        store.refresh_budget()
        # Bulk workers have admitted these bytes but have not yet written them.
        store._reserve(store.disk_budget - stored_bytes(store.root))
        worker = threading.Thread(target=apply_control)
        worker.start()
        worker.join(2)
        assert not worker.is_alive() and not failures
        # Spending prepaid credit must not forget the pending bulk allocation.
        with pytest.raises(OSError, match="budget exhausted"):
            store._reserve(1)
