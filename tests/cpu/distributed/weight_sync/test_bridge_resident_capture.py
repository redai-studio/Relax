from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from relax.backends.megatron.weight_update.bridge_source import BridgeSourceAdapter
from relax.distributed.weight_sync.checkpoint import snapshot_tensors
from relax.distributed.weight_sync.source_patch import file_digest


def adapter(convert):
    result = object.__new__(BridgeSourceAdapter)
    result.iterator = SimpleNamespace(_bridge_converter=SimpleNamespace(convert=convert))
    return result


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32, torch.int64])
def test_resident_conversion_matches_file_path_and_isolates_converter_mutation(tmp_path, dtype):
    info = SimpleNamespace(name="source")
    original = torch.arange(24, dtype=dtype).reshape(4, 6).T.contiguous()
    frozen = original.clone()

    def convert(name, value):
        value.add_(1)
        return [("logical", value.T.contiguous())]

    obj = adapter(convert)
    table = snapshot_tensors([("source", original)], tmp_path / "canonical")
    before_hash = file_digest(table["source"][2])
    for p in ["disk", "resident"]:
        (tmp_path / p).mkdir()
    disk = obj.finish_group(info, table, tmp_path / "disk", "FULL", None)
    # There is deliberately no file-based fallback in the resident route.
    with patch.object(torch, "from_file", side_effect=AssertionError("redundant file read")):
        prepared = obj.prepare_resident_group(info, [("source", original)], tmp_path / "resident")
        hashes = {}
        with ThreadPoolExecutor(2) as pool:
            resident = obj.write_group(prepared, obj.convert_group(info, prepared), hashes=hashes, hash_executor=pool)
            assert pool.submit(lambda: 3).result() == 3
    assert resident["logical"][2].read_bytes() == disk["logical"][2].read_bytes()
    assert hashes["logical"] == file_digest(resident["logical"][2])
    assert torch.equal(original, frozen)
    assert file_digest(table["source"][2]) == before_hash


def test_reuse_hashes_survive_and_corrupt_logical_reconverts(tmp_path):
    info = SimpleNamespace(name="source")
    calls = []

    def convert(name, tensor):
        calls.append(name)
        return [("logical", tensor.reshape(2, 4).T.contiguous())]

    obj = adapter(convert)
    tensor = torch.arange(8, dtype=torch.bfloat16)
    table = snapshot_tensors([("source", tensor)], tmp_path / "canonical")
    for p in ["first", "reuse", "repair"]:
        (tmp_path / p).mkdir()
    hashes = {}
    with ThreadPoolExecutor(2) as pool:
        prepared = obj.prepare_resident_group(info, [("source", tensor)], tmp_path / "first")
        first = obj.write_group(prepared, obj.convert_group(info, prepared), hashes=hashes, hash_executor=pool)
        previous = {"logical": first, "logical_groups": {"source": ["logical"]}, "logical_hashes": hashes}
        next_hashes = {}
        prepared = obj.prepare_group(info, table, tmp_path / "reuse", "REUSE", previous)
        reused = obj.write_group(prepared, obj.convert_group(info, prepared), hashes=next_hashes, hash_executor=pool)
        assert reused["logical"][2].read_bytes() == first["logical"][2].read_bytes()
        assert next_hashes == hashes and len(calls) == 1
        first["logical"][2].write_bytes(b"\0" * 16)
        prepared = obj.prepare_group(info, table, tmp_path / "repair", "REUSE", previous)
        repaired = obj.write_group(prepared, obj.convert_group(info, prepared), hashes={}, hash_executor=pool)
    assert len(calls) == 2
    assert file_digest(repaired["logical"][2]) == hashes["logical"]


def test_resident_wrong_group_rejected(tmp_path):
    obj = adapter(None)
    with pytest.raises(ValueError, match="differs"):
        obj.prepare_resident_group(SimpleNamespace(name="source"), [("other", torch.ones(1))], tmp_path)
