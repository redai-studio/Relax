# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Replay bundle writer and reader.

The writer is crash-safe: payloads and metadata land in a sibling temp
directory, then the directory is atomically renamed into place only after the
completion sentinel is written. The reader refuses to open a bundle that is
missing COMPLETE, a payload, a rank shard, or whose checksums do not match the
manifest — and it loads tensors with weights_only=True only.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from relax.utils.replay.schema import (
    BundleIndex,
    Identity,
    Manifest,
    PayloadSpec,
    index_from_dict,
    index_to_dict,
    manifest_from_dict,
    manifest_to_dict,
)


_COMPLETE = "COMPLETE"
_PAYLOAD_DIR = "payloads"
_MANIFEST = "manifest.json"
_INDEX = "index.json"
_EXPECTED = "expected.json"
_METADATA_FILES = (_MANIFEST, _INDEX, _EXPECTED)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")


def _write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    """Write JSON via a sibling temp file, then os.replace (POSIX-atomic)."""
    path = Path(path)
    tmp_path = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    _write_json(tmp_path, data)
    os.replace(tmp_path, path)


def _read_json(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def metadata_checksums(directory: str | Path) -> dict[str, str]:
    """SHA-256 of manifest.json, index.json and expected.json under
    directory."""
    directory = Path(directory)
    checksums: dict[str, str] = {}
    for name in _METADATA_FILES:
        path = directory / name
        if not path.is_file():
            raise IncompleteBundleError(f"missing {name} in {directory}")
        checksums[name] = sha256_file(path)
    return checksums


def _identity_anchor(identity: Identity) -> dict[str, Any]:
    """COMPLETE.<rank> identity fields that must match index.identity."""
    step = identity.actor_step_id
    return {
        "actor_step_id": ({"rollout_id": step.rollout_id, "step_id": step.step_id} if step is not None else None),
        "rollout_id": identity.rollout_id,
    }


def _validate_shard_anchor(shard: dict[str, Any], identity: Identity, *, rank: int) -> None:
    """Require actor_step_id + owned payloads, and match index.identity."""
    if "actor_step_id" not in shard:
        raise IncompleteBundleError(f"COMPLETE.{rank} missing required field 'actor_step_id'")
    if not isinstance(shard.get("payloads"), dict):
        raise IncompleteBundleError(f"COMPLETE.{rank} missing owned-record 'payloads' map")

    expected = _identity_anchor(identity)
    if shard["actor_step_id"] != expected["actor_step_id"]:
        raise IncompleteBundleError(
            f"COMPLETE.{rank} actor_step_id {shard['actor_step_id']!r} != index {expected['actor_step_id']!r}"
        )
    if expected["actor_step_id"] is None:
        if "rollout_id" not in shard:
            raise IncompleteBundleError(f"COMPLETE.{rank} missing rollout_id for a rollout-level bundle")
        if shard["rollout_id"] != expected["rollout_id"]:
            raise IncompleteBundleError(
                f"COMPLETE.{rank} rollout_id {shard['rollout_id']!r} != index {expected['rollout_id']!r}"
            )
    elif "rollout_id" in shard and shard["rollout_id"] != expected["rollout_id"]:
        raise IncompleteBundleError(
            f"COMPLETE.{rank} rollout_id {shard['rollout_id']!r} != index {expected['rollout_id']!r}"
        )


@dataclass
class LoadedBundle:
    """A fully validated bundle ready for replay."""

    manifest: Manifest
    index: BundleIndex
    expected: dict[str, Any]
    tensors: dict[str, torch.Tensor]

    @property
    def sample_ids(self) -> list[str]:
        return [record.sample_id for record in self.index.samples]

    @property
    def response_lengths(self) -> list[int]:
        return [record.response_length for record in self.index.samples]


class BundleWriter:
    """Writes one rank-local replay bundle atomically."""

    def __init__(
        self,
        path: str | Path,
        manifest: Manifest,
        index: BundleIndex,
        expected: dict[str, Any],
        *,
        rank: int = 0,
    ) -> None:
        self._final_path = Path(path)
        self._tmp_path = self._final_path.with_name(f".{self._final_path.name}.tmp-{os.getpid()}")
        self._manifest = manifest
        self._index = index
        self._expected = expected
        self._rank = rank
        self._payloads: dict[str, PayloadSpec] = {}

    def _payload_dir(self) -> Path:
        return self._tmp_path / _PAYLOAD_DIR

    def write_payload(self, name: str, tensor: torch.Tensor) -> None:
        """Store one detached CPU tensor payload under name."""
        tensor = tensor.detach().cpu().contiguous()
        path = self._payload_dir() / f"{name}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(tensor, path)
        self._payloads[name] = PayloadSpec(
            name=name,
            dtype=str(tensor.dtype),
            shape=list(tensor.shape),
            bytes=path.stat().st_size,
            sha256=sha256_file(path),
        )

    def finalize(self, ranks: list[int], *, expected_ranks: list[int] | None = None) -> dict[str, Any]:
        """Flush metadata, write COMPLETE sentinels and atomically publish.

        ranks is the shard set stored in this rank-local bundle.
        expected_ranks is the full producer set and defaults to ranks. A
        multi-rank capture persists expected_ranks so the reader can require
        the parent cohort COMPLETE.

        Returns the rank-local COMPLETE.<rank> payload so a multi-rank
        caller can publish the cohort shard without reading the bundle
        back.
        """
        expected = list(expected_ranks) if expected_ranks is not None else list(ranks)
        self._manifest.payloads = dict(self._payloads)
        _write_json(self._tmp_path / _MANIFEST, manifest_to_dict(self._manifest))
        _write_json(self._tmp_path / _INDEX, index_to_dict(self._index))
        _write_json(self._tmp_path / _EXPECTED, self._expected)

        metadata = metadata_checksums(self._tmp_path)
        rank_complete = {
            **_identity_anchor(self._index.identity),
            "payloads": {name: spec.sha256 for name, spec in self._payloads.items()},
            "metadata": metadata,
            "expected_ranks": expected,
        }
        _write_json(self._tmp_path / f"{_COMPLETE}.{self._rank}", rank_complete)
        _write_json(
            self._tmp_path / _COMPLETE,
            {"ranks": list(ranks), "metadata": metadata, "expected_ranks": expected},
        )

        if self._final_path.exists():
            raise FileExistsError(f"bundle already exists at {self._final_path}")
        os.replace(self._tmp_path, self._final_path)
        return rank_complete


def write_cohort_shard(
    path: str | Path,
    *,
    rank: int,
    identity: Identity,
    expected_ranks: list[int],
    payloads: dict[str, str] | None = None,
    metadata: dict[str, str] | None = None,
    relpath: str | None = None,
) -> None:
    """Publish one rank's COMPLETE.<rank> into a shared cohort directory.

    Does not block; other ranks may still be missing. try_finalize_cohort
    writes the final COMPLETE once every expected rank has published.
    """
    cohort_dir = Path(path)
    cohort_dir.mkdir(parents=True, exist_ok=True)
    shard = {
        **_identity_anchor(identity),
        "payloads": dict(payloads) if payloads is not None else {},
        "expected_ranks": list(expected_ranks),
    }
    if metadata is not None:
        shard["metadata"] = dict(metadata)
    if relpath is not None:
        shard["relpath"] = relpath
    _write_json_atomic(cohort_dir / f"{_COMPLETE}.{rank}", shard)


def try_finalize_cohort(path: str | Path, ranks: list[int]) -> bool:
    """If every expected COMPLETE.<rank> is present and consistent, write
    COMPLETE.

    Returns True when the cohort is finalized (including already-complete).
    Missing ranks return False without waiting. Safe to call from each rank's
    writer thread; concurrent finalizers atomically replace COMPLETE.

    The cohort COMPLETE is a coordinator sentinel (ranks + per-rank shards),
    not a BundleReader target. Replay still loads rank-<n>/; that rank-local
    COMPLETE records expected_ranks so the reader can require this parent
    sentinel before treating a multi-rank capture as complete.
    """
    cohort_dir = Path(path)
    complete_path = cohort_dir / _COMPLETE
    if complete_path.is_file():
        return True

    shards: dict[str, dict[str, Any]] = {}
    for rank in ranks:
        shard_path = cohort_dir / f"{_COMPLETE}.{rank}"
        if not shard_path.is_file():
            return False
        shard = _read_json(shard_path)
        if "actor_step_id" not in shard or not isinstance(shard.get("payloads"), dict):
            raise IncompleteBundleError(f"COMPLETE.{rank} missing actor_step_id or owned-record payloads")
        recorded_expected = shard.get("expected_ranks")
        if recorded_expected is not None and list(recorded_expected) != list(ranks):
            raise IncompleteBundleError(
                f"COMPLETE.{rank} expected_ranks {recorded_expected!r} != coordinator {list(ranks)!r}"
            )
        shards[str(rank)] = shard

    anchors = {
        (json.dumps(shard.get("actor_step_id"), sort_keys=True), shard.get("rollout_id")) for shard in shards.values()
    }
    if len(anchors) != 1:
        raise IncompleteBundleError("COMPLETE.<rank> identity anchors do not match across ranks")

    _write_json_atomic(complete_path, {"ranks": list(ranks), "shards": shards})
    return True


def parse_expected_ranks(complete_raw: dict[str, Any]) -> list[int] | None:
    """Return expected_ranks from a COMPLETE payload, or None if unset."""
    recorded = complete_raw.get("expected_ranks")
    if recorded is None:
        return None
    if not isinstance(recorded, list) or not recorded or not all(isinstance(rank, int) for rank in recorded):
        raise IncompleteBundleError(f"malformed {_COMPLETE}: 'expected_ranks' must be a non-empty integer list")
    return [int(rank) for rank in recorded]


def cohort_expected_ranks(cohort_dir: str | Path) -> list[int] | None:
    """Read a multi-rank expected_ranks list from any COMPLETE.<rank> shard.

    Returns None when the directory has no shard, or every shard is single-
    rank. Used by the CLI selector and as a fallback for legacy rank-local
    bundles that did not persist expected_ranks on COMPLETE.
    """
    cohort_dir = Path(cohort_dir)
    if not cohort_dir.is_dir():
        return None
    for path in sorted(cohort_dir.iterdir()):
        if not path.name.startswith(f"{_COMPLETE}.") or not path.is_file():
            continue
        try:
            shard = _read_json(path)
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            continue
        recorded = shard.get("expected_ranks")
        if isinstance(recorded, list) and len(recorded) > 1 and all(isinstance(rank, int) for rank in recorded):
            return [int(rank) for rank in recorded]
    return None


class IncompleteBundleError(ValueError):
    """Raised when a bundle is missing a sentinel, shard or payload."""


class BundleReader:
    """Reads and validates a replay bundle."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    def load(self, *, allow_partial_read: bool = False) -> LoadedBundle:
        """Load manifest, index, expected outputs and tensor payloads.

        allow_partial_read is for inspection of incomplete bundles; replay and
        validate always require a complete bundle, including the parent cohort
        COMPLETE when expected_ranks has more than one producer.
        """
        if not self._path.is_dir():
            raise IncompleteBundleError(f"bundle directory not found: {self._path}")

        complete = self._path / _COMPLETE
        if not complete.is_file():
            raise IncompleteBundleError(f"missing {_COMPLETE} sentinel in {self._path}")

        complete_raw = _read_json(complete)
        ranks = self._parse_ranks(complete_raw)
        if not allow_partial_read:
            self._require_parent_cohort_complete(complete_raw)
            self._validate_metadata_checksums(complete_raw.get("metadata"), required=True)

        manifest = manifest_from_dict(_read_json(self._path / _MANIFEST))
        index = index_from_dict(_read_json(self._path / _INDEX))
        expected = _read_json(self._path / _EXPECTED)

        if manifest.bundle_id != index.bundle_id:
            raise IncompleteBundleError(
                f"manifest bundle_id {manifest.bundle_id!r} != index bundle_id {index.bundle_id!r}"
            )

        self._validate_rank_shards(manifest, index.identity, ranks)
        self._validate_payloads(manifest)

        if not allow_partial_read:
            tensors = {name: self._read_payload(spec) for name, spec in manifest.payloads.items()}
        else:
            tensors = {}
        return LoadedBundle(manifest=manifest, index=index, expected=expected, tensors=tensors)

    def _parse_ranks(self, complete_raw: dict[str, Any]) -> list[int]:
        ranks = complete_raw.get("ranks")
        if not isinstance(ranks, list) or not all(isinstance(rank, int) for rank in ranks):
            raise IncompleteBundleError(f"malformed {_COMPLETE}: missing integer 'ranks' list")
        return ranks

    def _require_parent_cohort_complete(self, complete_raw: dict[str, Any]) -> None:
        """Fail closed when a multi-rank rank-local bundle's cohort is open."""
        expected = parse_expected_ranks(complete_raw)
        if expected is None and self._path.name.startswith("rank-"):
            expected = cohort_expected_ranks(self._path.parent)
        if expected is None or len(expected) <= 1:
            return
        parent = self._path.parent
        complete_path = parent / _COMPLETE
        if not complete_path.is_file():
            raise IncompleteBundleError(
                f"cohort {parent} is incomplete (missing {_COMPLETE}) for multi-rank bundle {self._path}"
            )
        parent_ranks = _read_json(complete_path).get("ranks")
        if not isinstance(parent_ranks, list) or list(parent_ranks) != list(expected):
            raise IncompleteBundleError(f"cohort {parent} ranks {parent_ranks!r} != expected_ranks {list(expected)!r}")

    def _validate_metadata_checksums(self, recorded: object, *, required: bool) -> None:
        if not isinstance(recorded, dict):
            if required:
                raise IncompleteBundleError(f"{_COMPLETE} missing metadata checksums")
            return
        actual = metadata_checksums(self._path)
        if recorded != actual:
            raise IncompleteBundleError("metadata checksum mismatch between COMPLETE and metadata files")

    def _validate_rank_shards(self, manifest: Manifest, identity: Identity, ranks: list[int]) -> None:
        accounted: dict[str, str] = {}
        for rank in ranks:
            shard_path = self._path / f"{_COMPLETE}.{rank}"
            if not shard_path.is_file():
                raise IncompleteBundleError(f"missing COMPLETE.{rank} in {self._path}")
            shard = _read_json(shard_path)
            _validate_shard_anchor(shard, identity, rank=rank)
            recorded_metadata = shard.get("metadata")
            if recorded_metadata is not None:
                self._validate_metadata_checksums(recorded_metadata, required=True)
            for name, checksum in shard["payloads"].items():
                if name in accounted:
                    raise IncompleteBundleError(f"payload {name!r} written by more than one rank")
                accounted[name] = checksum

        manifest_names = set(manifest.payloads)
        if set(accounted) != manifest_names:
            missing = sorted(manifest_names - set(accounted))
            extra = sorted(set(accounted) - manifest_names)
            raise IncompleteBundleError(f"rank shards do not match manifest (missing={missing}, extra={extra})")

        for name, checksum in accounted.items():
            if checksum != manifest.payloads[name].sha256:
                raise IncompleteBundleError(f"payload {name!r} checksum mismatch between shard and manifest")

    def _validate_payloads(self, manifest: Manifest) -> None:
        for name, spec in manifest.payloads.items():
            path = self._path / _PAYLOAD_DIR / f"{name}.pt"
            if not path.is_file():
                raise IncompleteBundleError(f"missing payload file {path}")
            if path.stat().st_size != spec.bytes:
                raise IncompleteBundleError(f"payload {name!r} byte size mismatch")
            if sha256_file(path) != spec.sha256:
                raise IncompleteBundleError(f"payload {name!r} checksum mismatch")

    def _read_payload(self, spec: PayloadSpec) -> torch.Tensor:
        path = self._path / _PAYLOAD_DIR / f"{spec.name}.pt"
        try:
            tensor = torch.load(path, weights_only=True)
        except Exception as exc:  # noqa: BLE001 — reject any unsafe/unpicklable payload as incomplete
            raise IncompleteBundleError(f"payload {spec.name!r} could not be loaded safely: {exc}") from exc
        if not isinstance(tensor, torch.Tensor):
            raise IncompleteBundleError(f"payload {spec.name!r} is not a tensor (weights_only loader rejected it)")
        if str(tensor.dtype) != spec.dtype:
            raise IncompleteBundleError(f"payload {spec.name!r} dtype {tensor.dtype} != manifest {spec.dtype}")
        if list(tensor.shape) != spec.shape:
            raise IncompleteBundleError(f"payload {spec.name!r} shape {tuple(tensor.shape)} != manifest {spec.shape}")
        return tensor
