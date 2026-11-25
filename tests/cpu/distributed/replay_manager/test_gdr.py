# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import pytest
import torch
from omegaconf import OmegaConf

from relax.distributed.replay_manager._gdr import reader_connection, require_gdr_client


def test_gdr_reader_does_not_mutate_shared_cpu_connection(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    connection = OmegaConf.create(
        {
            "backend": {
                "storage_backend": "MooncakeStore",
                "MooncakeStore": {
                    "protocol": "rdma",
                    "use_gdr": False,
                    "gdr_staging_buffer_mb": 1024,
                },
            }
        }
    )
    selected, enabled = reader_connection(connection)
    assert enabled and selected.backend.MooncakeStore.use_gdr
    assert selected.backend.MooncakeStore.gdr_staging_buffer_mb == 8
    assert not connection.backend.MooncakeStore.use_gdr
    assert connection.backend.MooncakeStore.gdr_staging_buffer_mb == 1024


def test_gdr_reader_rejects_uninitialized_gpu_and_tcp(monkeypatch):
    connection = OmegaConf.create(
        {
            "backend": {
                "storage_backend": "MooncakeStore",
                "MooncakeStore": {
                    "protocol": "rdma",
                    "use_gdr": False,
                },
            }
        }
    )
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    with pytest.raises(RuntimeError, match="Initialize"):
        reader_connection(connection)
    connection.backend.MooncakeStore.protocol = "tcp"
    with pytest.raises(ValueError, match="RDMA"):
        reader_connection(connection)


def test_cpu_backend_remains_usable_without_gpu_initialization():
    from types import SimpleNamespace

    connection = OmegaConf.create({"backend": {"storage_backend": "SimpleStorage"}})
    selected, enabled = reader_connection(connection)
    assert selected is connection and not enabled
    client = SimpleNamespace(storage_manager=SimpleNamespace(storage_client=SimpleNamespace(use_gdr=False)))
    with pytest.raises(RuntimeError, match="did not enable"):
        require_gdr_client(client)


def test_gdr_reader_rejects_cpu_payload_instead_of_implicit_h2d():
    from types import SimpleNamespace

    from relax.distributed.replay_manager._gdr import GdrBodyReader

    meta = SimpleNamespace(size=1, field_names=["replay_body"], global_indexes=[7])
    meta.select_fields = lambda fields: meta
    storage = SimpleNamespace(
        use_gdr=True, _gdr_staging=object(), get=lambda **kwargs: [torch.zeros(2, 65, dtype=torch.int32)]
    )
    manager = SimpleNamespace(
        storage_client=storage,
        _generate_keys=lambda fields, indexes: ["test-key"],
        _get_shape_type_custom_backend_meta_list=lambda metadata: ([(2, 65)], [torch.int32], [None]),
    )
    client = SimpleNamespace(storage_manager=manager, kv_retrieve_meta=lambda **kwargs: meta, close=lambda: None)
    reader = GdrBodyReader(lambda: client, partition="test")
    # Isolate the hardware boundary from this CPU-only fail-closed contract.
    reader._receive = lambda *args: [torch.zeros(2, 65, dtype=torch.int32)]
    with pytest.raises(RuntimeError, match="GPU bodies"):
        reader.read(["test-key"])
    assert reader.read_calls == reader.read_bytes == 0
    reader.close()


def test_gdr_reader_cleans_failed_client_initialization():
    from types import SimpleNamespace

    from relax.distributed.replay_manager._gdr import GdrBodyReader

    closed = []
    client = SimpleNamespace(
        storage_manager=SimpleNamespace(storage_client=SimpleNamespace(use_gdr=False)),
        close=lambda: closed.append(True),
    )
    reader = GdrBodyReader(lambda: client, partition="test")
    with pytest.raises(RuntimeError, match="did not enable"):
        reader.read(["test-key"])
    assert closed == [True] and reader._client is None
    reader.close()
    assert closed == [True]


@pytest.mark.parametrize("failed", [("gdr",), ("tq",), ("gdr", "tq")])
def test_body_backend_close_attempts_all_clients_and_retries_only_failures(failed):
    from relax.distributed.replay_manager.storage import TQBodyBackend

    class Client:
        def __init__(self, name):
            self.name = name
            self.calls = 0

        def close(self):
            self.calls += 1
            if self.name in failed and self.calls == 1:
                raise RuntimeError(f"{self.name} close failed")

    backend = TQBodyBackend("abcd", lambda: pytest.fail("Cleanup must not create a client"))
    gdr, tq = Client("gdr"), Client("tq")
    backend._gdr_reader, backend._client = gdr, tq
    with pytest.raises(RuntimeError, match=f"{failed[0]} close failed"):
        backend.close()
    assert gdr.calls == tq.calls == 1
    backend.close()
    assert gdr.calls == 1 + ("gdr" in failed)
    assert tq.calls == 1 + ("tq" in failed)
    assert backend._gdr_reader is backend._client is None
    backend.close()
    assert gdr.calls == 1 + ("gdr" in failed)
    assert tq.calls == 1 + ("tq" in failed)
