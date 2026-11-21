"""Optional snapshot hashes bind written CPU bytes without a file readback."""

import hashlib
from pathlib import Path

import pytest
import torch

from relax.distributed.weight_sync import checkpoint


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32, torch.int64, torch.bool])
def test_hashes_match_written_bytes_without_changing_table(tmp_path, monkeypatch, dtype):
    original = torch.arange(48).reshape(6, 8).to(dtype).T
    hashes = {"retained": "unrelated"}
    read_bytes = Path.read_bytes

    def forbid_readback(self):
        raise AssertionError("snapshot must not reread its output to produce the optional hash")

    seen = []
    digest = checkpoint.digest

    def record_digest(value):
        if isinstance(value, memoryview):
            assert value.readonly
            seen.append(value.nbytes)
        return digest(value)

    monkeypatch.setattr(Path, "read_bytes", forbid_readback)
    monkeypatch.setattr(checkpoint, "digest", record_digest)
    table = checkpoint.snapshot_tensors([("w", original)], tmp_path / "hashed", hashes=hashes)
    plain = checkpoint.snapshot_tensors([("w", original)], tmp_path / "plain")
    data = read_bytes(table["w"][2])
    assert table["w"][:2] == plain["w"][:2]
    assert table["w"][1] == [8, 6]
    assert data == read_bytes(plain["w"][2])
    assert hashes == {"retained": "unrelated", "w": hashlib.sha256(data).hexdigest()}
    assert seen == [original.numel() * original.element_size()]


def test_empty_and_scalar_tensors_keep_exact_byte_hashes(tmp_path):
    values = [("empty", torch.empty(0)), ("scalar", torch.tensor(-0.0))]
    hashes = {}
    table = checkpoint.snapshot_tensors(values, tmp_path, hashes=hashes)
    assert table["empty"][1] == [0]
    assert table["scalar"][1] == []
    assert hashes == {name: hashlib.sha256(entry[2].read_bytes()).hexdigest() for name, entry in table.items()}


def test_failed_snapshot_does_not_publish_partial_hash_mapping(tmp_path):
    hashes = {"before": "preserved"}
    with pytest.raises(ValueError, match="Duplicate tensor"):
        checkpoint.snapshot_tensors([("w", torch.ones(16)), ("w", torch.zeros(16))], tmp_path, hashes=hashes)
    assert hashes == {"before": "preserved"}


def test_short_write_rejects_hash_and_keeps_original_mapping(tmp_path, monkeypatch):
    original_open = Path.open
    hashes = {}

    class ShortWriter:
        def __init__(self, path):
            self.file = original_open(path, "wb")

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.file.close()

        def write(self, value):
            return self.file.write(value[:-1])

    def short_open(self, *args, **kwargs):
        return ShortWriter(self) if args == ("wb",) else original_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", short_open)
    with pytest.raises(OSError, match="Incomplete tensor snapshot write"):
        checkpoint.snapshot_tensors([("w", torch.ones(16))], tmp_path, hashes=hashes)
    assert not hashes


def test_hash_worker_finishes_before_failed_snapshot_releases_inputs(tmp_path, monkeypatch):
    import threading
    import time

    digest, opened = checkpoint.digest, Path.open
    started, finished = threading.Event(), threading.Event()

    def slow_digest(value):
        if isinstance(value, memoryview):
            started.set()
            time.sleep(0.04)
            try:
                return digest(value)
            finally:
                finished.set()
        return digest(value)

    def fail_open(self, *args, **kwargs):
        if args == ("wb",):
            assert started.wait(1)
            raise OSError("injected snapshot write failure")
        return opened(self, *args, **kwargs)

    monkeypatch.setattr(checkpoint, "digest", slow_digest)
    monkeypatch.setattr(Path, "open", fail_open)
    hashes = {}
    with pytest.raises(OSError, match="injected snapshot write failure"):
        checkpoint.snapshot_tensors([("w", torch.ones(16))], tmp_path, hashes=hashes)
    assert finished.is_set()
    assert not hashes
