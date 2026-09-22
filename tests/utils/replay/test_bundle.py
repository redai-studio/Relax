# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Bundle writer/reader integrity tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from relax.utils.replay.bundle import (
    BundleReader,
    IncompleteBundleError,
    sha256_file,
    try_finalize_cohort,
    write_cohort_shard,
)
from relax.utils.replay.schema import ActorStepId, Identity
from relax.utils.replay.validate import validate_bundle
from tests.utils.replay.helpers import build_grpo_bundle, resign_metadata_checksums


def _payload_path(bundle: Path, name: str) -> Path:
    return bundle / "payloads" / f"{name}.pt"


def _rewrite_manifest(bundle: Path, mutate) -> None:
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mutate(manifest)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    resign_metadata_checksums(bundle)


def _replace_payload(bundle: Path, name: str, obj) -> None:
    """Re-sign a payload (manifest + rank shard) so checksum checks pass and
    the reader reaches the type/numerics validation."""
    path = _payload_path(bundle, name)
    torch.save(obj, path)
    new_sha256 = sha256_file(path)
    new_dtype = str(obj.dtype) if isinstance(obj, torch.Tensor) else "object"
    new_shape = list(obj.shape) if isinstance(obj, torch.Tensor) else []

    def mutate(manifest):
        spec = manifest["payloads"][name]
        spec["bytes"] = path.stat().st_size
        spec["sha256"] = new_sha256
        spec["dtype"] = new_dtype
        spec["shape"] = new_shape

    _rewrite_manifest(bundle, mutate)

    complete_path = bundle / "COMPLETE.0"
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    complete["payloads"][name] = new_sha256
    complete_path.write_text(json.dumps(complete), encoding="utf-8")
    resign_metadata_checksums(bundle)


def test_bundle_roundtrip(tmp_path):
    bundle, index, expected = build_grpo_bundle(tmp_path / "bundle")
    loaded = BundleReader(bundle).load()

    assert loaded.manifest.bundle_id == "b-00001"
    assert loaded.index.identity.actor_step_id.rollout_id == 120
    assert loaded.index.identity.actor_step_id.step_id == 0
    assert len(loaded.index.samples) == 4
    assert set(loaded.tensors) == {"old_log_probs", "log_probs", "ref_log_probs", "entropy", "kl", "advantages"}
    assert loaded.tensors["advantages"].tolist() == [-1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0]
    assert loaded.expected["loss.policy"]["loss"] == pytest.approx(expected["loss.policy"]["loss"])


def test_bundle_unsupported_version(tmp_path):
    bundle, _, _ = build_grpo_bundle(tmp_path / "bundle")

    def mutate(manifest):
        manifest["format_version"] = "2.0.0"

    _rewrite_manifest(bundle, mutate)
    with pytest.raises(ValueError):
        BundleReader(bundle).load()


def test_bundle_unknown_field_warns(tmp_path):
    bundle, _, _ = build_grpo_bundle(tmp_path / "bundle")

    def mutate(manifest):
        manifest["future_field"] = {"x": 1}

    _rewrite_manifest(bundle, mutate)
    result = validate_bundle(bundle)
    assert result.valid
    assert any("unknown field" in warning for warning in result.warnings)


def test_bundle_missing_commit_warns(tmp_path):
    bundle, _, _ = build_grpo_bundle(tmp_path / "bundle")

    def mutate(manifest):
        manifest["producer"]["commit"] = ""

    _rewrite_manifest(bundle, mutate)
    result = validate_bundle(bundle)
    assert result.valid
    assert any("producer commit" in warning for warning in result.warnings)


def test_bundle_missing_payload(tmp_path):
    bundle, _, _ = build_grpo_bundle(tmp_path / "bundle")
    _payload_path(bundle, "advantages").unlink()
    with pytest.raises(IncompleteBundleError):
        BundleReader(bundle).load()


def test_bundle_checksum_mismatch(tmp_path):
    bundle, _, _ = build_grpo_bundle(tmp_path / "bundle")
    path = _payload_path(bundle, "advantages")
    path.write_bytes(path.read_bytes() + b"\x00")
    with pytest.raises(IncompleteBundleError):
        BundleReader(bundle).load()


def test_bundle_missing_complete(tmp_path):
    bundle, _, _ = build_grpo_bundle(tmp_path / "bundle")
    (bundle / "COMPLETE").unlink()
    with pytest.raises(IncompleteBundleError):
        BundleReader(bundle).load()


def test_bundle_unsafe_object(tmp_path):
    bundle, _, _ = build_grpo_bundle(tmp_path / "bundle")
    _replace_payload(bundle, "advantages", {"not": "a tensor"})
    with pytest.raises(IncompleteBundleError):
        BundleReader(bundle).load()


def test_bundle_nan_inf_rejected(tmp_path):
    bundle, _, _ = build_grpo_bundle(tmp_path / "bundle", corrupt="nan")
    result = validate_bundle(bundle)
    assert not result.valid
    assert any("NaN" in error for error in result.errors)


def test_bundle_truncation(tmp_path):
    bundle, _, _ = build_grpo_bundle(tmp_path / "bundle")
    path = _payload_path(bundle, "advantages")
    path.write_bytes(path.read_bytes()[:-2])
    with pytest.raises(IncompleteBundleError):
        BundleReader(bundle).load()


def test_bundle_rollout_identity_valid(tmp_path):
    from relax.utils.replay.capture import build_bundle_from_record
    from tests.utils.replay.helpers import make_rollout_capture_record

    bundle_path = build_bundle_from_record(make_rollout_capture_record("b-rid"), tmp_path)
    assert validate_bundle(bundle_path).valid


def _rewrite_index(bundle: Path, mutate) -> None:
    index_path = bundle / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    mutate(index)
    index_path.write_text(json.dumps(index), encoding="utf-8")
    resign_metadata_checksums(bundle)


def test_bundle_identity_anchor_rejected(tmp_path):
    bundle, _, _ = build_grpo_bundle(tmp_path / "bundle")

    def both_anchors(index):
        index["identity"]["rollout_id"] = 120

    _rewrite_index(bundle, both_anchors)
    _sync_shard_identity(bundle)
    result = validate_bundle(bundle)
    assert not result.valid
    assert any("exactly one" in error for error in result.errors)

    def neither_anchor(index):
        index["identity"]["actor_step_id"] = None
        index["identity"]["rollout_id"] = None

    _rewrite_index(bundle, neither_anchor)
    _sync_shard_identity(bundle)
    result = validate_bundle(bundle)
    assert not result.valid
    assert any("exactly one" in error for error in result.errors)


def _sync_shard_identity(bundle: Path) -> None:
    """Copy index.identity onto COMPLETE.0 so reader identity checks pass."""
    identity = json.loads((bundle / "index.json").read_text(encoding="utf-8"))["identity"]
    shard_path = bundle / "COMPLETE.0"
    shard = json.loads(shard_path.read_text(encoding="utf-8"))
    shard["actor_step_id"] = identity["actor_step_id"]
    shard["rollout_id"] = identity.get("rollout_id")
    shard_path.write_text(json.dumps(shard), encoding="utf-8")


def test_bundle_metadata_checksum_mismatch(tmp_path):
    bundle, _, _ = build_grpo_bundle(tmp_path / "bundle")
    expected_path = bundle / "expected.json"
    expected = json.loads(expected_path.read_text(encoding="utf-8"))
    expected["loss.policy"]["loss"] = 0.0
    expected_path.write_text(json.dumps(expected), encoding="utf-8")
    with pytest.raises(IncompleteBundleError, match="metadata checksum"):
        BundleReader(bundle).load()


def test_bundle_shard_actor_step_mismatch(tmp_path):
    bundle, _, _ = build_grpo_bundle(tmp_path / "bundle")
    shard_path = bundle / "COMPLETE.0"
    shard = json.loads(shard_path.read_text(encoding="utf-8"))
    shard["actor_step_id"] = {"rollout_id": 999, "step_id": 0}
    shard_path.write_text(json.dumps(shard), encoding="utf-8")
    with pytest.raises(IncompleteBundleError, match="actor_step_id"):
        BundleReader(bundle).load()


def test_try_finalize_cohort_waits_then_completes(tmp_path):
    cohort = tmp_path / "cohort"
    identity = Identity(actor_step_id=ActorStepId(rollout_id=120, step_id=0))
    write_cohort_shard(cohort, rank=0, identity=identity, expected_ranks=[0, 1], payloads={"x": "abc"})
    assert try_finalize_cohort(cohort, [0, 1]) is False
    write_cohort_shard(cohort, rank=1, identity=identity, expected_ranks=[0, 1], payloads={"y": "def"})
    assert try_finalize_cohort(cohort, [0, 1]) is True
    complete = json.loads((cohort / "COMPLETE").read_text(encoding="utf-8"))
    assert complete["ranks"] == [0, 1]
    assert complete["shards"]["1"]["payloads"] == {"y": "def"}
    assert try_finalize_cohort(cohort, [0, 1]) is True


def test_bundle_reader_requires_parent_cohort_complete(tmp_path):
    cohort = tmp_path / "cohort"
    bundle, _, _ = build_grpo_bundle(cohort / "rank-0")
    complete_path = bundle / "COMPLETE"
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    complete["expected_ranks"] = [0, 1]
    complete_path.write_text(json.dumps(complete), encoding="utf-8")
    with pytest.raises(IncompleteBundleError, match="incomplete"):
        BundleReader(bundle).load()

    (cohort / "COMPLETE").write_text(json.dumps({"ranks": [0, 1], "shards": {}}), encoding="utf-8")
    loaded = BundleReader(bundle).load()
    assert loaded.manifest.bundle_id == "b-00001"


def test_bundle_reader_infers_expected_ranks_from_parent_shard(tmp_path):
    cohort = tmp_path / "cohort"
    bundle, _, _ = build_grpo_bundle(cohort / "rank-0")
    complete_path = bundle / "COMPLETE"
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    complete.pop("expected_ranks", None)
    complete_path.write_text(json.dumps(complete), encoding="utf-8")
    identity = Identity(actor_step_id=ActorStepId(rollout_id=120, step_id=0))
    write_cohort_shard(cohort, rank=0, identity=identity, expected_ranks=[0, 1], payloads={"x": "abc"})
    with pytest.raises(IncompleteBundleError, match="incomplete"):
        BundleReader(bundle).load()
