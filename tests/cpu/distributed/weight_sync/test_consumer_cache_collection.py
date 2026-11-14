"""Long consumers reclaim old cache entries while independent readers retain
pins."""

import random
import threading
from contextlib import contextmanager

import pytest

from relax.distributed.weight_sync.checkpoint import validate_checkpoint
from relax.distributed.weight_sync.consumer import VersionConsumer
from relax.distributed.weight_sync.store import VersionStore, stored_bytes
from relax.distributed.weight_sync.transport import ObjectServer, SharedFSTransport, TCPTransport


@contextmanager
def delivery(source, mode):
    if mode == "sharedfs":
        yield SharedFSTransport(source)
        return
    with ObjectServer(("127.0.0.1", 0), source) as server:
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        try:
            yield TCPTransport(server.server_address)
        finally:
            server.shutdown()
            thread.join()


@pytest.mark.parametrize("mode", ["sharedfs", "tcp"])
@pytest.mark.parametrize("blocks", [False, True])
def test_long_sequence_reclaims_cache_and_preserves_slow_reader(tmp_path, mode, blocks):
    source = VersionStore(tmp_path / "source", disk_budget=64 * 1024**2, max_weight_bytes=1024**2)
    cache = VersionStore(
        tmp_path / "cache", disk_budget=8 * 1024**2, max_weight_bytes=1024**2, canonical_blocks=blocks
    )
    tensor = tmp_path / "tensor"
    rng = random.Random(0)
    current = held = None
    with delivery(source, mode) as transport:
        fast = VersionConsumer(transport, cache, tmp_path / "fast", checkpoint_budget=32 * 1024**2)
        slow = VersionConsumer(transport, cache, tmp_path / "slow", checkpoint_budget=32 * 1024**2)
        for version in range(40):
            content = rng.randbytes(256 * 1024)
            tensor.write_bytes(content)
            source.publish(
                "s",
                version,
                {"w": ("U8", [len(content)], tensor)},
                anchor_interval=2,
                compression=False,
                model_config={},
            )
            if held is None:
                held = slow.prepare("s", 0, "slow")
            prepared = fast.prepare("s", version, f"fast-{version}")
            validate_checkpoint(prepared.checkpoint, prepared.manifest)
            raw = (prepared.checkpoint / "model.safetensors").read_bytes()
            assert raw[8 + int.from_bytes(raw[:8], "little") :] == content
            if current is not None:
                fast.release(current, remove_checkpoint=True)
            current = prepared
            validate_checkpoint(held.checkpoint, held.manifest)
            assert cache.manifest("s", 0)["root"] == held.manifest["root"]
            assert stored_bytes(cache.root) < cache.disk_budget
        assert len(cache.versions("s")) <= 6
        assert len(source.versions("s")) == 40
        slow.release(held, remove_checkpoint=True)
        cache.collect()
        assert not cache.manifest_path("s", 0).exists()


def test_collection_error_does_not_block_valid_preparation(tmp_path, monkeypatch):
    source = VersionStore(tmp_path / "source", disk_budget=32 * 1024**2, max_weight_bytes=1024**2)
    tensor = tmp_path / "tensor"
    tensor.write_bytes(b"abcd")
    manifest = source.publish("s", 0, {"w": ("U8", [4], tensor)}, model_config={})
    cache = VersionStore(tmp_path / "cache", disk_budget=32 * 1024**2, max_weight_bytes=1024**2)
    consumer = VersionConsumer(
        SharedFSTransport(source), cache, tmp_path / "checkpoints", checkpoint_budget=32 * 1024**2
    )

    def unavailable(**kwargs):
        raise OSError("injected maintenance I/O failure")

    monkeypatch.setattr(cache, "collect", unavailable)
    prepared = consumer.prepare("s", 0, "reader")
    assert prepared.manifest == manifest
    validate_checkpoint(prepared.checkpoint, manifest)
    consumer.release(prepared)
    retried = consumer.prepare("s", 0, "next-reader")
    validate_checkpoint(retried.checkpoint, manifest)


def test_preparation_checks_new_and_reused_bytes_once_and_rejects_corruption(tmp_path, monkeypatch):
    from relax.distributed.weight_sync import checkpoint as checkpoint_module
    from relax.distributed.weight_sync import consumer as consumer_module

    source = VersionStore(tmp_path / "source", disk_budget=32 << 20, max_weight_bytes=1 << 20)
    tensor = tmp_path / "tensor"
    tensor.write_bytes(b"abcd")
    manifest = source.publish("s", 0, {"w": ("U8", [4], tensor)}, model_config={})
    consumer = VersionConsumer(
        SharedFSTransport(source),
        VersionStore(tmp_path / "cache", disk_budget=32 << 20, max_weight_bytes=1 << 20),
        tmp_path / "checkpoints",
        checkpoint_budget=32 << 20,
    )
    checked = []

    def validate(path, manifest):
        checked.append(path)
        return validate_checkpoint(path, manifest)

    monkeypatch.setattr(checkpoint_module, "validate_checkpoint", validate)
    monkeypatch.setattr(consumer_module, "validate_checkpoint", validate)
    first = consumer.prepare("s", 0, "first")
    assert len(checked) == 1 and checked[0].name.startswith(".checkpoint-")
    checked.clear()
    second = consumer.prepare("s", 0, "second", full=True)
    assert second.checkpoint == first.checkpoint
    assert checked == [first.checkpoint]
    assert "checkpoint_reuse_validation" in second.metrics["seconds"]
    path = first.checkpoint / "model.safetensors"
    content = path.read_bytes()
    path.write_bytes(content[:-1] + bytes([content[-1] ^ 1]))
    with pytest.raises(ValueError, match="content differs"):
        consumer.prepare("s", 0, "corrupt")
    with pytest.raises(ValueError, match="content differs"):
        validate_checkpoint(first.checkpoint, manifest)
