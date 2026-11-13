# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Build the recipe-local FLA 0.4.2 source closure without changing global
FLA."""

import argparse
import hashlib
import json
import subprocess
import tempfile
import urllib.request
import zipfile
from pathlib import Path


WHEELS = (
    (
        "https://files.pythonhosted.org/packages/ee/36/3c303f92bafea7c3f97d68bbb83d18cc42e30cd0bfb1b7cfe589360f11d6/fla_core-0.4.2-py3-none-any.whl",
        "cba3db29380002da3cbfc0db94d6efac19aaf528900d19c05c2765e8f3cc485b",
    ),
    (
        "https://files.pythonhosted.org/packages/60/ee/a3cba17965482b35c4990af90bad108e82c32edcb59911c37f318b5f4198/flash_linear_attention-0.4.2-py3-none-any.whl",
        "c08be006ce4dbe1be81f54938ee8e6fc7968cfba397c8d06c7669e97b8c44c0d",
    ),
)
SOURCE_SHA256 = "08d6957c0fa1008d8a02cd9e824d18bb6f71cabcca07f503fdf0bfeafdd15a19"


def source_hash(package: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(package.rglob("*.py")):
        digest.update(path.relative_to(package).as_posix().encode() + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()


def unpack(wheel: Path, expected: str, target: Path) -> None:
    if hashlib.sha256(wheel.read_bytes()).hexdigest() != expected:
        raise ValueError(f"Wheel checksum mismatch: {wheel.name}")
    with zipfile.ZipFile(wheel) as archive:
        for name in archive.namelist():
            if Path(name).is_absolute() or ".." in Path(name).parts:
                raise ValueError(f"Unsafe wheel member: {name}")
        archive.extractall(target)


def install(target: Path, patch: Path, wheels: list[Path]) -> dict:
    if target.exists():
        raise ValueError(f"Refusing to overwrite existing compatibility runtime: {target}")
    target.mkdir(parents=True)
    for wheel, (_, expected) in zip(wheels, WHEELS, strict=True):
        unpack(wheel, expected, target)
    subprocess.run(
        ["patch", "--batch", "--forward", "--fuzz=0", "-p1", "-i", str(patch.resolve())], cwd=target, check=True
    )
    actual = source_hash(target / "fla")
    if actual != SOURCE_SHA256:
        raise ValueError(f"Legacy FLA source closure differs from the validated runtime: {actual}")
    return {"version": "0.4.2", "root": str(target), "source_sha256": actual, "wheels": WHEELS, "gpu_validated": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--patch", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as directory:
        wheels = []
        for url, _ in WHEELS:
            wheel = Path(directory) / url.rsplit("/", 1)[1]
            with urllib.request.urlopen(url, timeout=120) as response:
                wheel.write_bytes(response.read())
            wheels.append(wheel)
        report = install(args.target, args.patch, wheels)
    args.manifest.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
