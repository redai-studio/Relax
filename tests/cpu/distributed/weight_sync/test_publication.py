# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import os
import subprocess
import sys
import threading
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

from relax.distributed.weight_sync.consumer import Consumer, verify_checkpoint
from relax.distributed.weight_sync.format import WeightSyncError, canonical_json, digest
from relax.distributed.weight_sync.publisher import Publisher
from relax.distributed.weight_sync.store import CapacityError, SharedFSStore
from relax.distributed.weight_sync.tcp import ObjectServer, TCPTransport


def publisher(store, **kwargs):
    return Publisher(store, "stream", {"model_config": {"model_type": "qwen3"}}, chunk_bytes=256, **kwargs)


def publish(pub, version, tensors):
    return pub.finalize_pending(pub.submit(str(version), tensors.items()))


def load_checkpoint(prepared):
    return {
        name: tensor for file in Path(prepared.path).glob("*.safetensors") for name, tensor in load_file(file).items()
    }


def test_publication_writer_lock_timeout_also_bounds_close(tmp_path):
    import fcntl
    from time import monotonic

    store = SharedFSStore(tmp_path / "store")
    pub = publisher(store, sync_timeout_s=0.1)
    with (store.root / ".writer.lock").open("ab") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        started = monotonic()
        handle = pub.submit("blocked", [("x", torch.ones(8))])
        with pytest.raises(TimeoutError, match="Store transaction deadline"):
            pub.finalize_pending(handle, timeout=2)
        pub.close()
        assert monotonic() - started < 2
    assert store.head("stream") is None


@pytest.mark.parametrize("transport_name", ["shared_fs", "tcp"])
def test_twenty_versions_anchor_delta_restart_and_bitwise_dtype_roundtrip(tmp_path, transport_name):
    store = SharedFSStore(tmp_path / "store")
    pub = publisher(store, anchor_interval=4)
    tensors = {
        "float": torch.arange(512, dtype=torch.float32),
        "bf16": torch.tensor([0x7FC1, 0x7FC2, -32768, 0], dtype=torch.int16).view(torch.bfloat16),
        "fp16": torch.tensor([0x7E01, 0x7E02, -32768, 0], dtype=torch.int16).view(torch.float16),
        "integer": torch.tensor([1, 2, -3], dtype=torch.int64),
        "empty": torch.empty((0, 4), dtype=torch.float16),
        "scalar": torch.tensor(1.5),
    }
    store.register_consumer("all", "stream")
    server = ObjectServer(("127.0.0.1", 0), store, request_timeout_s=2)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    transport = store if transport_name == "shared_fs" else TCPTransport(*server.server_address, timeout_s=5)
    try:
        for version in range(20):
            tensors["float"][version] = -version
            published = publish(pub, version, tensors)
            assert store.resolve("stream", version)["kind"] == ("FULL" if version % 4 == 0 else "DELTA")
            # A new client every time exercises persisted inventory, not a live Python base.
            consumer = Consumer(
                transport, SharedFSStore(tmp_path / "cache"), tmp_path / "checkpoints", shard_bytes=1024
            )
            prepared = consumer.prepare("stream", version)
            actual = load_checkpoint(prepared)
            assert actual.keys() == tensors.keys()
            for name, expected in tensors.items():
                assert actual[name].shape == expected.shape
                assert actual[name].dtype == expected.dtype
                assert torch.equal(actual[name].reshape(-1).view(torch.uint8), expected.reshape(-1).view(torch.uint8))
            store.acknowledge("all", version, published.root, str(version))
            transport.release(prepared.reference)
            store.collect()
        assert store.inventory()["consumers"]["all"]["last_active_version"] == 19
    finally:
        pub.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def test_capture_is_immutable_queue_is_bounded_and_commit_retry_is_idempotent(tmp_path):
    store = SharedFSStore(tmp_path)
    pub = publisher(store)
    tensor = torch.arange(512, dtype=torch.bfloat16)
    expected = tensor.clone()
    handle = pub.submit("update-0", [("x", tensor)])
    tensor.fill_(7)
    with pytest.raises(CapacityError, match="QUEUE"):
        pub.submit("other", [("x", tensor)])
    published = pub.finalize_pending(handle)
    consumer = Consumer(store, SharedFSStore(tmp_path / "cache"), tmp_path / "checkpoints")
    actual = load_checkpoint(consumer.prepare("stream", 0))["x"]
    assert torch.equal(actual, expected)
    pub.close()
    retry = publisher(store)
    assert publish(retry, "update-0", {"x": expected}) == published
    assert len(store.inventory()["streams"]["stream"]) == 1
    retry.close()


def test_failed_commit_does_not_advance_base_or_make_version_visible(tmp_path, monkeypatch):
    store = SharedFSStore(tmp_path)
    pub = publisher(store)
    original_save = store._save
    monkeypatch.setattr(store, "_save", lambda _: (_ for _ in ()).throw(OSError("injected index fsync failure")))
    with pytest.raises(OSError):
        publish(pub, 0, {"x": torch.arange(512)})
    assert store.head("stream") is None and pub.confirmed is None
    monkeypatch.setattr(store, "_save", original_save)
    assert publish(pub, 0, {"x": torch.arange(512)}).version == 0
    pub.close()


def test_same_target_full_fallback_seal_and_offline_process(tmp_path):
    store = SharedFSStore(tmp_path / "store")
    pub = publisher(store, anchor_interval=20)
    tensors = {"x": torch.arange(512, dtype=torch.bfloat16)}
    publish(pub, 0, tensors)
    tensors["x"][8] = 7
    target = publish(pub, 1, tensors)
    full = store.ensure_full("stream", 1)
    assert full != target.artifact_id
    assert store.manifest(full)["root"] == target.root
    assert store.record("stream", 1)["artifact_id"] == target.artifact_id
    store.seal("offline", "stream", [1])
    pub.close()
    script = """
import sys
from relax.distributed.weight_sync.store import SharedFSStore
from relax.distributed.weight_sync.consumer import Consumer
root=sys.argv[1]
c=Consumer(SharedFSStore(root+'/store'),SharedFSStore(root+'/fresh-cache'),root+'/offline')
p=c.prepare('stream',1,full=True)
assert p.root==sys.argv[2]
"""
    subprocess.run(
        [sys.executable, "-c", script, str(tmp_path), target.root],
        check=True,
        timeout=20,
        env=os.environ | {"CUDA_VISIBLE_DEVICES": ""},
    )


def test_all_versions_consumer_and_pins_protect_old_targets_from_gc(tmp_path):
    store = SharedFSStore(tmp_path)
    pub = publisher(store, anchor_interval=2)
    store.register_consumer("slow", "stream", 0)
    tensors = {"x": torch.arange(64)}
    for version in range(10):
        tensors["x"].fill_(version)
        publish(pub, version, tensors)
    store.collect()
    assert len(store.inventory()["streams"]["stream"]) == 10
    store.pin("old", "stream", 0)
    for version in range(10):
        root = store.record("stream", version)["root"]
        first = store.acknowledge("slow", version, root, str(version))
        assert store.acknowledge("slow", version, root, str(version)) == first
    store.collect()
    assert store.record("stream", 0)
    store.release("old")
    store.collect()
    with pytest.raises(WeightSyncError, match="UNAVAILABLE"):
        store.record("stream", 0)
    assert store.record("stream", 9)
    pub.close()


def test_manifest_and_local_checkpoint_corruption_are_rejected(tmp_path):
    store = SharedFSStore(tmp_path / "store")
    pub = publisher(store)
    result = publish(pub, 0, {"x": torch.arange(64)})
    consumer = Consumer(store, SharedFSStore(tmp_path / "cache"), tmp_path / "ckpt")
    prepared = consumer.prepare("stream", 0)
    manifest = store.resolve("stream", 0)
    shard = next(Path(prepared.path).glob("*.safetensors"))
    with shard.open("r+b") as output:
        output.seek(-1, 2)
        output.write(b"x")
    with pytest.raises(WeightSyncError, match="bytes differ"):
        verify_checkpoint(prepared.path, manifest["catalog"])
    altered = manifest | {"base_root": digest(b"wrong-base")}
    (store.root / "artifacts" / result.artifact_id).write_bytes(canonical_json(altered))
    with pytest.raises(WeightSyncError):
        store.manifest(result.artifact_id)
    pub.close()


def test_host_and_disk_quota_fail_before_visibility(tmp_path):
    store = SharedFSStore(tmp_path, disk_budget_bytes=512)
    pub = publisher(store, compression=False, host_budget_bytes=8192)
    with pytest.raises(CapacityError):
        pub.submit("large", [("x", torch.empty(10000))])
    with pytest.raises(CapacityError):
        publish(pub, 0, {"x": torch.arange(512)})
    assert store.head("stream") is None
    pub.close()
