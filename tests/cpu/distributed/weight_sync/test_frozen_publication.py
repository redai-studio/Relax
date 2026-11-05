# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import importlib.util
import json
import threading
from dataclasses import asdict, replace
from pathlib import Path

import torch
from safetensors.torch import save_file

from relax.distributed.weight_sync.config import WeightSyncConfig
from relax.distributed.weight_sync.consumer import Consumer
from relax.distributed.weight_sync.publisher import Publisher
from relax.distributed.weight_sync.store import SharedFSStore
from relax.distributed.weight_sync.tcp import ObjectServer, TCPTransport
from relax.distributed.weight_sync.tensor_diff import compare


def test_frozen_publication_reaches_tcp_consumer_for_full_and_delta(tmp_path):
    """Exercise actual frozen-HF publication and TCP reconstruction, without
    Ray or GPU startup."""
    script = Path(__file__).resolve().parents[4] / "task_delta_weight_sync/gpu_replay.py"
    spec = importlib.util.spec_from_file_location("frozen_publication_smoke", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original = WeightSyncConfig(
        mode="delta",
        stream="smoke",
        store_root=str(tmp_path / "original/store"),
        canonical_root=str(tmp_path / "original/canonical"),
        chunk_bytes=32,
        host_budget_bytes=65536,
        disk_budget_bytes=1048576,
        sync_timeout_s=30,
    )
    config_path = tmp_path / "sync.json"
    config_path.write_text(json.dumps(asdict(original)))
    source = SharedFSStore(original.store_root, canonical_root=original.canonical_root)
    expected = []
    for version in range(2):
        bits = torch.full((32,), 0x3F800000, dtype=torch.int32)
        bits[3] += version
        tensors = {"weight": bits.view(torch.float32)}
        export = tmp_path / "exports" / f"update-{version:06d}"
        export.mkdir(parents=True)
        save_file(tensors, export / "model.safetensors")
        publisher = Publisher(source, original.stream, {"fixture": "frozen"}, **original.publisher_kwargs())
        try:
            expected.append(publisher.finalize_pending(publisher.submit(str(version), tensors.items()), 30))
        finally:
            publisher.close()
    for full_only in [True, False]:
        arm = "full" if full_only else "delta"
        config = replace(
            original,
            store_root=str(tmp_path / arm / "store"),
            canonical_root=str(tmp_path / arm / "canonical"),
            full_only=full_only,
        )
        store = SharedFSStore(config.store_root, config.disk_budget_bytes, canonical_root=config.canonical_root)
        server = ObjectServer(("127.0.0.1", 0), store)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        transport = TCPTransport(*server.server_address, timeout_s=30)
        consumer = Consumer(transport, SharedFSStore(tmp_path / arm / "cache"), tmp_path / arm / "checkpoints")
        try:
            for version in range(2):
                published = module.publish_frozen_target(store, config, str(config_path), version)
                prepared = consumer.prepare(config.stream, version)
                assert published["root"] == prepared.root == expected[version].root
                assert published["kind"] == ("FULL" if full_only or version == 0 else "DELTA")
                assert compare(tmp_path / "exports" / f"update-{version:06d}", Path(prepared.path))["equal"]
                assert not consumer.last_metrics.get("fallback_reason")
                consumer.release(prepared)
            assert transport.metrics["payload_bytes"] > 0
        finally:
            server.shutdown()
            server.server_close()
            thread.join(5)
