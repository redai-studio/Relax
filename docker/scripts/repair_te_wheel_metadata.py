# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Correct TE 2.14.1's cu12 library wheel tag to match its published filename.

The upstream py3-none-manylinux_2_28_x86_64 wheel contains an erroneous
cp310-cp310 tag. This distribution supplies shared libraries; the separate
transformer-engine-torch distribution supplies the Python extension.
"""

import base64
import csv
import hashlib
import importlib.metadata as md
import subprocess
import sys
from pathlib import Path

from packaging.tags import parse_tag, sys_tags


EXPECTED_TAG = "py3-none-manylinux_2_28_x86_64"


def repair(distribution: md.Distribution, patch: Path) -> None:
    if distribution.version != "2.14.1":
        raise ValueError(f"Unexpected TE library version: {distribution.version}")
    if not parse_tag(EXPECTED_TAG).intersection(sys_tags()):
        raise ValueError("TE wheel repair only supports manylinux_2_28 x86_64")
    files = list(distribution.files or ())
    wheel_entry = next(path for path in files if str(path).endswith(".dist-info/WHEEL"))
    wheel = Path(distribution.locate_file(wheel_entry))
    metadata = wheel.read_text()
    tags = [line for line in metadata.splitlines() if line.startswith("Tag:")]
    if tags == [f"Tag: {EXPECTED_TAG}"]:
        return
    if tags != ["Tag: cp310-cp310-manylinux_2_28_x86_64"]:
        raise ValueError(f"Unexpected TE wheel tags: {tags}")
    # pip records compiled __pycache__/*.cpython-312.pyc files as well.
    # Only native extensions carry a CPython binary ABI; bytecode is harmless.
    if any(".cpython-" in path.name and path.suffix in {".so", ".pyd"} for path in files):
        raise ValueError("Refusing to retag a CPython extension")
    subprocess.run(
        ["patch", "--batch", "--forward", "--fuzz=0", "-p1", "-i", str(patch.resolve())],
        cwd=wheel.parent,
        check=True,
    )
    record = wheel.parent / "RECORD"
    with record.open(newline="") as stream:
        rows = list(csv.reader(stream))
    content = wheel.read_bytes()
    digest = base64.urlsafe_b64encode(hashlib.sha256(content).digest()).rstrip(b"=").decode()
    for row in rows:
        if row[0] == str(wheel_entry):
            row[1:] = [f"sha256={digest}", str(len(content))]
            break
    else:
        raise ValueError("TE RECORD does not contain WHEEL")
    with record.open("w", newline="") as stream:
        csv.writer(stream).writerows(rows)


if __name__ == "__main__":
    repair(md.distribution("transformer-engine-cu12"), Path(sys.argv[1]))
