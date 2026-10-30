# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Preserve installed training extensions for reuse with the identical Torch
ABI."""

import hashlib
import importlib.metadata as md
import json
import platform
import sys
from pathlib import Path

import torch
from wheel.wheelfile import WheelFile


PACKAGES = (
    "flash-attn",
    "flash-attn-3",
    "fast-hadamard-transform",
    "flash-mla",
    "transformer-engine",
    "transformer-engine-torch",
    "transformer-engine-cu12",
    "torch-memory-saver",
    "apex",
    "fake-int4-quant-cuda",
    "nvidia-modelopt",
)


def fingerprint() -> dict[str, object]:
    return {
        "torch": md.version("torch"),
        "cuda": torch.version.cuda,
        "cxx11_abi": torch.compiled_with_cxx11_abi(),
        "python": list(sys.version_info[:2]),
        "machine": platform.machine(),
        "nccl": md.version("nvidia-nccl-cu12"),
    }


def export(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"abi": fingerprint(), "packages": {}, "wheels": {}}
    for name in PACKAGES:
        dist = md.distribution(name)
        root = Path(dist.locate_file(""))
        stem = f"{dist.metadata['Name'].replace('-', '_')}-{dist.version}"
        metadata_dir = f"{stem}.dist-info"
        files = [Path(p) for p in dist.files or []]
        if name == "flash-attn-3":
            files = [p.relative_to(root) for p in (root / "flash_attn_3").rglob("*") if p.is_file()]
            files.append(Path("flash_attn_config.py"))
            wheel_metadata = "Wheel-Version: 1.0\nRoot-Is-Purelib: false\nTag: cp312-cp312-linux_x86_64\n"
        else:
            wheel_metadata = dist.read_text("WHEEL")
        if name == "transformer-engine-cu12":
            # The published library wheel has the same incorrect cp310 metadata.
            # Its CPython extension is shipped separately in transformer-engine-torch.
            assert dist.version == "2.14.1"
            assert not any(".cpython-" in p.name and p.suffix == ".so" for p in files)
            wheel_metadata = wheel_metadata.replace(
                "cp310-cp310-manylinux_2_28_x86_64", "py3-none-manylinux_2_28_x86_64"
            )
        tag = next(line.removeprefix("Tag: ") for line in wheel_metadata.splitlines() if line.startswith("Tag: "))
        destination = output / f"{stem}-{tag}.whl"
        with WheelFile(destination, "w") as wheel:
            for relative in files:
                if ".." in relative.parts or "__pycache__" in relative.parts or relative.suffix == ".pyc":
                    continue
                if relative.name in {"RECORD", "INSTALLER", "REQUESTED", "direct_url.json"}:
                    continue
                if name == "transformer-engine-cu12" and relative.name == "WHEEL":
                    continue
                path = root / relative
                if path.is_file():
                    wheel.write(path, str(relative))
            if name == "flash-attn-3":
                wheel.writestr(f"{metadata_dir}/METADATA", dist.read_text("METADATA") or dist.read_text("PKG-INFO"))
                wheel.writestr(f"{metadata_dir}/WHEEL", wheel_metadata)
            elif name == "transformer-engine-cu12":
                wheel.writestr(f"{metadata_dir}/WHEEL", wheel_metadata)
        with destination.open("rb") as stream:
            manifest["wheels"][destination.name] = hashlib.file_digest(stream, "sha256").hexdigest()
        manifest["packages"][name] = dist.version
        print(destination.name, flush=True)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def check(output: Path) -> None:
    manifest = json.loads((output / "manifest.json").read_text())
    if manifest["abi"] != fingerprint():
        raise RuntimeError(f"Training wheel ABI mismatch: {manifest['abi']} != {fingerprint()}")
    for name, expected in manifest["wheels"].items():
        with (output / name).open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
                raise RuntimeError(f"Training wheel checksum mismatch: {name}")


if __name__ == "__main__":
    {"export": export, "check": check}[sys.argv[1]](Path(sys.argv[2]))
