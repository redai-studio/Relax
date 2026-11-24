"""Borrowed snapshot hash workers retain byte-view lifetime on every exit."""

import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import torch

from relax.distributed.weight_sync import checkpoint


def test_external_executor_reused_and_not_closed(tmp_path):
    value = torch.arange(128, dtype=torch.int32)
    with ThreadPoolExecutor(max_workers=1) as pool:
        worker = pool.submit(threading.get_ident).result()
        for index in range(3):
            hashes = {}
            table = checkpoint.snapshot_tensors(
                [("w", value)], tmp_path / str(index), hashes=hashes, hash_executor=pool
            )
            assert hashes == {"w": hashlib.sha256(table["w"][2].read_bytes()).hexdigest()}
            assert table["w"][:2] == ("I32", [128])
            assert pool.submit(threading.get_ident).result() == worker


def test_no_hash_request_does_not_submit_to_external_executor(tmp_path):
    class Unused:
        def submit(self, *args):
            raise AssertionError("hashes were not requested")

    checkpoint.snapshot_tensors([("w", torch.ones(2))], tmp_path, hash_executor=Unused())


@pytest.mark.parametrize("hash_fails", [False, True])
def test_write_failure_drains_external_work_and_preserves_primary_error(tmp_path, monkeypatch, hash_fails):
    hashing = threading.Event()
    release = threading.Event()
    failed_write = threading.Event()
    digest = checkpoint.digest
    opened = Path.open

    def slow(value):
        if isinstance(value, memoryview):
            hashing.set()
            assert release.wait(5)
            if hash_fails:
                raise ValueError("injected hash failure")
        return digest(value)

    def fail_open(path, *args, **kwargs):
        if args == ("wb",):
            assert hashing.wait(5)
            failed_write.set()
            raise OSError("injected snapshot write failure")
        return opened(path, *args, **kwargs)

    monkeypatch.setattr(checkpoint, "digest", slow)
    monkeypatch.setattr(Path, "open", fail_open)
    hashes = {"old": "untouched"}
    with ThreadPoolExecutor(max_workers=1) as pool, ThreadPoolExecutor(max_workers=1) as caller:
        pending = caller.submit(
            checkpoint.snapshot_tensors, [("w", torch.ones(2))], tmp_path, hashes=hashes, hash_executor=pool
        )
        try:
            assert failed_write.wait(5)
            assert not pending.done(), "snapshot returned while its CPU view was still in use"
        finally:
            release.set()
        with pytest.raises(OSError, match="snapshot write failure"):
            pending.result(5)
        assert pool.submit(lambda: 17).result() == 17
    assert hashes == {"old": "untouched"}


def test_hash_failure_keeps_hash_mapping_transactional_and_executor_usable(tmp_path, monkeypatch):
    real = checkpoint.digest

    def failed(value):
        if isinstance(value, memoryview):
            raise ValueError("injected hash failure")
        return real(value)

    monkeypatch.setattr(checkpoint, "digest", failed)
    hashes = {"old": "untouched"}
    with ThreadPoolExecutor(max_workers=1) as pool:
        with pytest.raises(ValueError, match="hash failure"):
            checkpoint.snapshot_tensors([("w", torch.ones(2))], tmp_path, hashes=hashes, hash_executor=pool)
        assert pool.submit(lambda: 17).result() == 17
    assert hashes == {"old": "untouched"}
