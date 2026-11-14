from types import SimpleNamespace

import numpy as np
import pytest

from relax.distributed.weight_sync.source_map import TensorMapping


@pytest.mark.parametrize(
    "shape,dimension,kind",
    [((6, 4), 0, "gather_gate_up"), ((3, 4), 1, "identity"), ((3, 4), 0, "identity"), ((3,), None, "identity")],
)
def test_gather_coordinates_match_full_concatenation(shape, dimension, kind):
    shards = [np.arange(np.prod(shape), dtype=np.uint16).reshape(shape) + 40000 * rank for rank in range(2)]
    if kind == "gather_gate_up":
        parts = [np.split(value, 2, axis=0) for value in shards]
        complete = np.concatenate([part[0] for part in parts] + [part[1] for part in parts], axis=0)
    elif dimension is not None:
        complete = np.concatenate(shards, axis=dimension)
    else:
        complete = shards[0]
    mapping = TensorMapping(shape, 2, dimension, (("source", complete.shape),), kind=kind)
    reconstructed = np.zeros_like(complete).reshape(-1)
    for rank in range(2 if dimension is not None else 1):
        positions, targets = mapping.destinations(rank, np.arange(np.prod(shape)))["source"]
        reconstructed[targets] = shards[rank].reshape(-1)[positions]
    assert reconstructed.tobytes() == complete.tobytes()


def test_bridge_reuse_validates_logical_cache_and_reconverts(tmp_path):
    torch = pytest.importorskip("torch")
    from relax.backends.megatron.weight_update.bridge_source import BridgeSourceAdapter
    from relax.distributed.weight_sync.checkpoint import snapshot_tensors
    from relax.distributed.weight_sync.source_patch import file_digest

    calls = []

    class Converter:
        def convert(self, name, tensor):
            calls.append(name)
            return [("logical", tensor.reshape(2, 4).T.contiguous())]

    adapter = object.__new__(BridgeSourceAdapter)
    adapter.iterator = SimpleNamespace(_bridge_converter=Converter())
    info = SimpleNamespace(name="source")
    original = torch.arange(8, dtype=torch.bfloat16)
    table = snapshot_tensors([("source", original)], tmp_path / "raw")
    first = tmp_path / "first"
    first.mkdir()
    result = adapter.finish_group(info, table, first, "FULL", None)
    previous = {
        "logical": result,
        "logical_groups": {"source": ["logical"]},
        "logical_hashes": {"logical": file_digest(result["logical"][2])},
    }
    second = tmp_path / "second"
    second.mkdir()
    adapter.finish_group(info, table, second, "REUSE", previous)
    assert len(calls) == 1
    result["logical"][2].write_bytes(b"\0" * 16)
    third = tmp_path / "third"
    third.mkdir()
    repaired = adapter.finish_group(info, table, third, "REUSE", previous)
    assert len(calls) == 2
    assert (
        repaired["logical"][2].read_bytes()
        == original.reshape(2, 4).T.contiguous().view(torch.uint8).numpy().tobytes()
    )
