# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Version identity for full adapters exported by ``LoraAdapterSync``."""

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    import torch


@dataclass(frozen=True)
class VersionIdentity:
    version_id: int
    digest: str

    def __post_init__(self) -> None:
        if self.version_id < 0:
            raise ValueError("version_id must be nonnegative")
        if len(self.digest) != 64 or any(c not in "0123456789abcdef" for c in self.digest):
            raise ValueError("digest must be a lowercase SHA-256 hex digest")

    @property
    def lora_name(self) -> str:
        return f"relax_lora@{self.version_id}.{self.digest[:12]}"


def tensor_fingerprint(tensor: "torch.Tensor") -> "torch.Tensor":
    """Position-weighted byte checksum, returned as a device int64 scalar.

    SplitMix64 weights and the sum wrap modulo 2**64; byte length seeds the sum
    so dropping zero bytes is detected too. Chunking bounds scratch memory and
    uses absolute byte positions, independent of chunk boundaries. This is a
    transport check, not a substitute for the snapshot's SHA-256.
    """
    import torch

    data = tensor.detach().contiguous().reshape(-1).view(torch.uint8)
    checksum = torch.full((), data.numel(), dtype=torch.int64, device=data.device)
    for start in range(0, data.numel(), 1 << 20):
        end = min(start + (1 << 20), data.numel())
        positions = torch.arange(start, end, dtype=torch.int64, device=data.device)
        weights = positions + (-7046029254386353131)
        weights = (weights ^ ((weights >> 30) & ((1 << 34) - 1))) * (-4658895280553007687)
        weights = (weights ^ ((weights >> 27) & ((1 << 37) - 1))) * (-7723592293110705685)
        weights = weights ^ ((weights >> 31) & ((1 << 33) - 1))
        checksum += (data[start:end].to(torch.int64) * weights).sum()
    return checksum


@dataclass(frozen=True)
class AdapterSnapshot:
    """Own detached CPU storage; consumers receive copies, never live params.

    Build only with ``snapshot_adapter`` after the existing export/gather has
    produced a full adapter. No collective or training-state mutation occurs
    here. The manager needs only ``identity`` and ``fingerprints``.
    """

    identity: VersionIdentity
    manifest_json: str
    _fingerprints: tuple[tuple[str, int], ...]
    _tensors: dict[str, "torch.Tensor"] = field(repr=False, compare=False)

    def copy_tensors(self) -> dict[str, "torch.Tensor"]:
        """Return independently owned tensors for transport or persistence."""
        return {name: tensor.clone() for name, tensor in self._tensors.items()}

    @property
    def config(self) -> dict[str, Any]:
        return json.loads(self.manifest_json)["adapter_config"]

    @property
    def fingerprints(self) -> dict[str, int]:
        return dict(self._fingerprints)


def snapshot_adapter(
    version_id: int,
    tensors: Mapping[str, "torch.Tensor"],
    config: Mapping[str, Any],
    *,
    base_fingerprint: str,
) -> AdapterSnapshot:
    """Freeze a full export at a training sync point before hashing/sending it.

    ``base_fingerprint`` identifies base contents, not their local pathname.
    Tensor bytes are hashed in logical contiguous order; dtype and shape are
    included in the manifest. Version IDs deliberately do not affect digest.
    """
    import torch

    if not tensors or not base_fingerprint:
        raise ValueError("a snapshot requires adapter tensors and a base fingerprint")
    owned = {}
    manifest_tensors = []
    fingerprints = []
    for name in sorted(tensors):
        tensor = tensors[name]
        if not name or tensor.layout != torch.strided or tensor.is_quantized:
            raise ValueError("adapter tensors must be named, dense, non-quantized tensors")
        copied = tensor.detach().to(device="cpu", copy=True).contiguous()
        owned[name] = copied
        data = copied.reshape(-1).view(torch.uint8).numpy()
        manifest_tensors.append(
            {
                "name": name,
                "dtype": str(copied.dtype),
                "shape": list(copied.shape),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
        fingerprints.append((name, int(tensor_fingerprint(copied))))
    manifest = {
        "format_version": 1,
        "base_fingerprint": base_fingerprint,
        "adapter_config": dict(config),
        "tensors": manifest_tensors,
    }
    manifest_json = json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    identity = VersionIdentity(version_id, hashlib.sha256(manifest_json.encode("utf-8")).hexdigest())
    return AdapterSnapshot(identity, manifest_json, tuple(fingerprints), owned)
