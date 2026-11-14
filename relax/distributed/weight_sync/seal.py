# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Verified, retained version sets for replay after the producer exits."""

import threading

from .codec import decode
from .deadline import checkpoint
from .parallel_io import ordered_map
from .store import (
    DTYPES,
    VersionStore,
    atomic_write,
    canonical,
    digest,
    load_json,
    sync_directory,
    valid_id,
    writer_lock,
)


def read_seal(store: VersionStore, name: str) -> dict:
    """Read the immutable inventory and check its publication identities."""
    record = load_json((store.root / "seals" / (valid_id(name) + ".json")).read_bytes())
    body = {key: value for key, value in record.items() if key != "inventory_id"}
    if (
        record.get("format") != 1
        or record.get("name") != name
        or record.get("inventory_id") != digest(canonical(body))
    ):
        raise ValueError("Invalid sealed inventory; restore its original file before collecting versions")
    for entry in record["closure"]:
        manifest = store.manifest(record["stream"], entry["version"])
        if any(manifest[field] != entry[field] for field in ("root", "artifact_id")):
            raise ValueError("Sealed publication identity changed; restore its recorded package")
    return record


def seal_versions(store: VersionStore, stream: str, versions: list[int], name: str) -> dict:
    """Verify encoded dependencies and retain an explicit set atomically.

    The inventory contains no source process or external canonical paths.
    Copying the store's manifests, objects and seals is sufficient for replay.
    """
    valid_id(name)
    valid_id(stream)
    if not versions or len(versions) > 4096 or any(type(v) is not int or v < 0 for v in versions):
        raise ValueError("Seal requires 1 to 4096 nonnegative version numbers")
    versions = sorted(set(versions))
    path = store.root / "seals" / (name + ".json")
    with writer_lock(store.root / ".content-writer", timeout=120), writer_lock(store.root / ".writer", timeout=30):
        store.refresh_budget()
        if path.exists():
            prior = read_seal(store, name)
            if prior["stream"] != stream or prior["versions"] != versions:
                raise ValueError("Seal name already identifies another set; choose a new name")
        manifests = {}
        for version in versions:
            current = store.manifest(stream, version)
            while current["version"] not in manifests:
                if len(manifests) >= 4096:
                    raise ValueError("Sealed dependency closure exceeds 4096 versions")
                manifests[current["version"]] = current
                if current["kind"] == "full":
                    break
                base = store.manifest(stream, current["base_version"])
                if base["root"] != current["base_root"]:
                    raise ValueError("Sealed dependency root mismatch")
                current = base
        objects = {}
        for version, manifest in sorted(manifests.items()):
            if manifest["schema_context"]["model_config"] is None:
                raise ValueError("Sealing requires the published model configuration")
            previous = (
                {tensor["name"]: tensor for tensor in manifests[manifest["base_version"]]["tensors"]}
                if manifest["kind"] == "delta"
                else {}
            )
            repair_lock = threading.Lock()

            def read_base(tensor, index):
                chunk = previous[tensor["name"]]["chunks"][index]
                try:
                    return store.read_chunk(stream, manifest["base_version"], tensor["name"], chunk)
                except (FileNotFoundError, ValueError):
                    with repair_lock:
                        try:
                            return store.read_chunk(stream, manifest["base_version"], tensor["name"], chunk)
                        except (FileNotFoundError, ValueError):
                            store._materialize(stream, manifest["base_version"])
                            return store.read_chunk(stream, manifest["base_version"], tensor["name"], chunk)

            def verify(job):
                tensor, index, chunk = job
                checkpoint("offline seal verification")
                object_path = store.object_path(chunk["object"])
                if object_path.stat().st_size != chunk["encoded_size"]:
                    raise ValueError("Sealed object size differs from its manifest")
                payload = object_path.read_bytes()
                if digest(payload) != chunk["object"]:
                    raise ValueError("Sealed object content differs from its manifest")
                base = None
                if chunk["kind"] not in ("raw", "raw_zlib"):
                    base = read_base(tensor, index)
                value = decode(chunk["kind"], payload, base, DTYPES[tensor["dtype"]], chunk["size"])
                # Raw decode returns the exact immutable payload just hashed.
                # Other codecs must hash their reconstructed output separately.
                output = chunk["object"] if chunk["kind"] == "raw" else digest(value)
                if output != chunk["output"]:
                    raise ValueError("Sealed object does not reconstruct its recorded weight")
                return chunk["object"], chunk["encoded_size"]

            jobs = (
                (tensor, index, chunk)
                for tensor in manifest["tensors"]
                for index, chunk in enumerate(tensor["chunks"])
            )
            # Verify the delivery objects directly, without first reading the
            # entire target cache. Every delta checks its base bytes as consumed
            # and repairs missing/corrupt base caches before decoding. Drain one
            # version before advancing to its dependents.
            for key, size in ordered_map(verify, jobs, 8):
                objects[key] = size
        record = {
            "format": 1,
            "name": name,
            "stream": stream,
            "versions": versions,
            "closure": [
                {field: m[field] for field in ("version", "root", "artifact_id")} for _, m in sorted(manifests.items())
            ],
            "object_count": len(objects),
            "encoded_bytes": sum(objects.values()),
            "objects_id": digest(canonical(objects)),
        }
        record["inventory_id"] = digest(canonical(record))
        encoded = canonical(record)
        if path.exists():
            if path.read_bytes() != encoded:
                raise ValueError("Sealed inventory changed; restore its recorded content")
        else:
            store._reserve(len(encoded))
            atomic_write(path, encoded)
        return record


def release_seal(store: VersionStore, name: str) -> None:
    """Release this set's retention; collection remains an explicit action."""
    path = store.root / "seals" / (valid_id(name) + ".json")
    with writer_lock(store.root / ".writer", timeout=30):
        path.unlink(missing_ok=True)
        if path.parent.exists():
            sync_directory(path.parent)
