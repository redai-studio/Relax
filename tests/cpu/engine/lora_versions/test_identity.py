# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import hashlib
import json

import pytest
import torch

from relax.engine.lora_versions.identity import snapshot_adapter, tensor_fingerprint
from relax.engine.lora_versions.manager import LoRAVersionManager, PublicationError, Replica, SessionKey


def test_snapshot_owns_storage_and_metadata():
    source = torch.arange(12, dtype=torch.float32, requires_grad=True).reshape(3, 4).t()
    config = {"r": 8, "target_modules": ["q_proj", "v_proj"]}
    snapshot = snapshot_adapter(7, {"a": source}, config, base_fingerprint="base-sha256")
    expected = source.detach().clone()
    with torch.no_grad():
        source.fill_(99)
    config["target_modules"].clear()
    exported = snapshot.copy_tensors()
    assert exported["a"].device.type == "cpu"
    assert not exported["a"].requires_grad
    assert exported["a"].is_contiguous()
    assert torch.equal(exported["a"], expected)
    exported["a"].zero_()
    snapshot.config["target_modules"].clear()
    snapshot.fingerprints.clear()
    assert torch.equal(snapshot.copy_tensors()["a"], expected)
    assert snapshot.config["target_modules"] == ["q_proj", "v_proj"]
    assert snapshot.fingerprints
    assert snapshot.identity.digest == hashlib.sha256(snapshot.manifest_json.encode()).hexdigest()
    assert snapshot.identity.lora_name == f"relax_lora@7.{snapshot.identity.digest[:12]}"


def test_digest_is_canonical_and_independent_of_version_and_tensor_layout():
    a = torch.arange(12, dtype=torch.bfloat16).reshape(3, 4).t()
    first = snapshot_adapter(1, {"z": a, "a": a}, {"r": 8, "alpha": 16}, base_fingerprint="base")
    second = snapshot_adapter(2, {"a": a.contiguous(), "z": a.clone()}, {"alpha": 16, "r": 8}, base_fingerprint="base")
    assert first.identity.digest == second.identity.digest
    assert first.identity.lora_name != second.identity.lora_name
    assert first.fingerprints == second.fingerprints
    manifest = json.loads(first.manifest_json)
    assert [t["name"] for t in manifest["tensors"]] == ["a", "z"]
    assert manifest["tensors"][0]["dtype"] == "torch.bfloat16"


@pytest.mark.parametrize("change", ["base", "config", "name", "dtype", "shape", "bytes"])
def test_digest_covers_all_content_identity_fields(change):
    tensor = torch.arange(4, dtype=torch.float32)
    original = snapshot_adapter(0, {"a": tensor}, {"r": 8}, base_fingerprint="base")
    changed = snapshot_adapter(
        0,
        {
            "b" if change == "name" else "a": {
                "dtype": tensor.view(torch.int32),
                "shape": tensor.reshape(2, 2),
                "bytes": tensor.flip(0),
            }.get(change, tensor)
        },
        {"r": 16 if change == "config" else 8},
        base_fingerprint="other-base" if change == "base" else "base",
    )
    assert original.identity.digest != changed.identity.digest


def test_transport_fingerprint_detects_permutation_and_zero_truncation():
    data = torch.tensor([1, 2, 3, 0], dtype=torch.uint8)
    original = tensor_fingerprint(data)
    assert original != tensor_fingerprint(data.flip(0))
    assert original != tensor_fingerprint(data[:-1])
    assert original != tensor_fingerprint(data[1:])


def test_transport_fingerprint_matches_integer_reference_across_chunks():
    data = torch.zeros((1 << 20) + 7, dtype=torch.uint8)
    nonzero = {0: 255, (1 << 20) - 1: 17, (1 << 20) + 6: 89}
    mask = (1 << 64) - 1
    expected = data.numel()
    for position, value in nonzero.items():
        data[position] = value
        weight = (position + 0x9E3779B97F4A7C15) & mask
        weight = ((weight ^ (weight >> 30)) * 0xBF58476D1CE4E5B9) & mask
        weight = ((weight ^ (weight >> 27)) * 0x94D049BB133111EB) & mask
        weight ^= weight >> 31
        expected = (expected + value * weight) & mask
    if expected >= (1 << 63):
        expected -= 1 << 64
    assert int(tensor_fingerprint(data)) == expected


def test_scalar_and_empty_tensors_have_stable_manifests():
    snapshot = snapshot_adapter(
        0, {"scalar": torch.tensor(1.0, dtype=torch.bfloat16), "empty": torch.empty(0)}, {}, base_fingerprint="base"
    )
    manifest = json.loads(snapshot.manifest_json)
    assert [t["shape"] for t in manifest["tensors"]] == [[0], []]
    assert dict(snapshot.fingerprints)["empty"] == 0


def test_snapshot_rejects_missing_identity_inputs_and_non_json_config():
    with pytest.raises(ValueError, match="requires"):
        snapshot_adapter(0, {}, {}, base_fingerprint="base")
    with pytest.raises(ValueError, match="requires"):
        snapshot_adapter(0, {"a": torch.ones(1)}, {}, base_fingerprint="")
    with pytest.raises(ValueError, match="JSON"):
        snapshot_adapter(0, {"a": torch.ones(1)}, {"alpha": float("nan")}, base_fingerprint="base")


def test_snapshot_publication_uses_received_tensor_fingerprints():
    snapshot = snapshot_adapter(0, {"a": torch.arange(8, dtype=torch.bfloat16)}, {"r": 8}, base_fingerprint="base")
    manager = LoRAVersionManager({"e0": "boot0", "e1": "boot1"})
    reservation = manager.reserve(snapshot.identity, snapshot.fingerprints)
    attempt = reservation.publication.attempt_id
    for engine in ("e0", "e1"):
        received = snapshot.copy_tensors()
        actual = {name: int(tensor_fingerprint(tensor)) for name, tensor in received.items()}
        assert manager.record_verified(0, attempt, Replica(engine, f"boot{engine[-1]}", engine), actual, pinned=True)
    manager.commit(0, attempt, None)
    manager.register_shard("s", "s0")
    assert manager.bind(SessionKey("s", "s0", 0)).lora_name == snapshot.identity.lora_name

    next_snapshot = snapshot_adapter(1, snapshot.copy_tensors(), snapshot.config, base_fingerprint="base")
    attempt = manager.reserve(next_snapshot.identity, next_snapshot.fingerprints).publication.attempt_id
    corrupted = next_snapshot.copy_tensors()
    corrupted["a"][0] += 1
    with pytest.raises(PublicationError, match="REPLICA_INVALID"):
        manager.record_verified(
            1,
            attempt,
            Replica("e0", "boot0", "corrupt"),
            {name: int(tensor_fingerprint(tensor)) for name, tensor in corrupted.items()},
            pinned=True,
        )
    assert manager.default_version == 0
