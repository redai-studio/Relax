# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Immutable engine-format snapshots on a shared POSIX filesystem.

Use a separate root per training run. Writers publish complete directories by
rename within that root; readers must see the same files at the absolute path.
These tensors already use the engine's layout, including repacked GDN weights,
and must not pass through a generic HF/PEFT conversion again.

Final artifacts are retained for retries and engine recovery. Removing a GPU
replica does not authorize deleting its shared artifact.
"""

import errno
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .identity import AdapterSnapshot, VersionIdentity, snapshot_adapter


@dataclass(frozen=True)
class AdapterArtifact:
    identity: VersionIdentity
    manifest_json: str
    _fingerprints: tuple[tuple[str, int], ...]
    path: Path

    @property
    def config(self) -> dict[str, Any]:
        return json.loads(self.manifest_json)["adapter_config"]

    @property
    def fingerprints(self) -> dict[str, int]:
        return dict(self._fingerprints)


def read_adapter_artifact(path: str | Path) -> AdapterArtifact:
    """Validate the stored bytes and return metadata without retaining
    tensors."""
    from safetensors.torch import load_file

    path = Path(path).resolve()
    metadata = json.loads((path / "manifest.json").read_text())
    identity = VersionIdentity(**metadata["identity"])
    config = json.loads((path / "adapter_config.json").read_text())
    tensors = load_file(str(path / "adapter_model.safetensors"), device="cpu")
    snapshot = snapshot_adapter(
        identity.version_id, tensors, config, base_fingerprint=metadata["snapshot"]["base_fingerprint"]
    )
    if snapshot.identity != identity or json.loads(snapshot.manifest_json) != metadata["snapshot"]:
        raise ValueError("artifact content does not match its manifest")
    return AdapterArtifact(identity, snapshot.manifest_json, tuple(snapshot.fingerprints.items()), path)


def write_adapter_artifact(snapshot: AdapterSnapshot, root: str | Path) -> AdapterArtifact:
    """Publish once, or validate and reuse an identical completed version.

    The filesystem must support atomic same-filesystem directory rename and
    fsync. Concurrent writers can only lose to an already complete directory. A
    failed write removes only its own staging directory, never a final path.
    """
    from safetensors.torch import save_file

    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    destination = root / f"version-{snapshot.identity.version_id}"
    if not destination.exists():
        staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=root))
        try:
            save_file(snapshot.copy_tensors(), str(staging / "adapter_model.safetensors"))
            (staging / "adapter_config.json").write_text(json.dumps(snapshot.config, sort_keys=True))
            metadata = {
                "identity": {"version_id": snapshot.identity.version_id, "digest": snapshot.identity.digest},
                "snapshot": json.loads(snapshot.manifest_json),
            }
            (staging / "manifest.json").write_text(json.dumps(metadata, sort_keys=True))
            for name in ("adapter_model.safetensors", "adapter_config.json", "manifest.json"):
                with (staging / name).open("rb") as stream:
                    os.fsync(stream.fileno())
            directory = os.open(staging, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
            try:
                os.rename(staging, destination)
            except OSError as exc:
                if exc.errno not in (errno.EEXIST, errno.ENOTEMPTY):
                    raise
                # A concurrent writer won; verify its complete content below.
            directory = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
    artifact = read_adapter_artifact(destination)
    if artifact.identity != snapshot.identity:
        raise ValueError("VERSION_CONFLICT: artifact version already contains different content")
    return artifact
