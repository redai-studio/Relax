#!/usr/bin/env bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Build only. This script never installs a package or changes the source checkout.
set -euo pipefail

if [[ $# -lt 1 || $# -gt 3 ]]; then
    echo "Usage: TQ_BUILD_PYTHON=<isolated-python> $0 OUTPUT_DIR [PATCH_FILE] [SOURCE_ARCHIVE]" >&2
    exit 2
fi
tq_patch_sha=baabe0f9a84fee210bcc4da51a844da81e21b4ac555c8da38c1f4646f9662b6d
if [[ "${TQ_PATCH_SHA256:-${tq_patch_sha}}" != "${tq_patch_sha}" ]]; then
    echo "TQ_PATCH_SHA256 does not match the pinned TransferQueue patch" >&2
    exit 2
fi
tq_python="${TQ_BUILD_PYTHON:-python3}"
tq_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
tq_patch="${2:-${tq_script_dir}/../patch/transferqueue/9784ad0-assignment-transform.patch}"
tq_out="$("${tq_python}" -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve())' "$1")"
tq_patch="$("${tq_python}" -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve())' "${tq_patch}")"
tq_commit=9784ad0d8a5f0db93ff8326379ab2a6d06e640a0
tq_archive_sha=5da896714784e4f3c03f54ef6454cc9893f6e143d447cf520e0c3bafd17aab31
tq_version=0.1.10.dev0+relax.9784ad0.lifecycle.ack

"${tq_python}" -c 'import build, setuptools, wheel' || {
    echo "Missing build tooling. Use an isolated build venv; do not install into the training environment." >&2
    exit 2
}
tq_work="$(mktemp -d "${TMPDIR:-/tmp}/relax-tq-build.XXXXXXXX")"
trap 'rm -rf -- "${tq_work}"' EXIT
tq_archive="${3:-${tq_work}/source.tar.gz}"
if [[ $# -lt 3 ]]; then
    curl --fail --location --silent --show-error \
        "https://codeload.github.com/redai-studio/TransferQueue/tar.gz/${tq_commit}" -o "${tq_archive}"
fi

# Verify the exact source and patch before extracting or applying anything.
"${tq_python}" - "${tq_archive}" "${tq_archive_sha}" "${tq_patch}" "${tq_patch_sha}" \
    "${tq_work}" "${tq_commit}" <<'PY'
import hashlib
import pathlib
import shutil
import sys
import tarfile

archive, archive_sha, patch, patch_sha, work, commit = sys.argv[1:]
for name, expected in ((archive, archive_sha), (patch, patch_sha)):
    if hashlib.sha256(pathlib.Path(name).read_bytes()).hexdigest() != expected:
        raise SystemExit(f"SHA256 mismatch: {name}")
source = pathlib.Path(work) / "source"
source.mkdir()
prefix = f"TransferQueue-{commit}"
with tarfile.open(archive, "r:gz") as tar:
    epoch = tar.getmember(prefix).mtime
    for member in tar.getmembers():
        parts = pathlib.PurePosixPath(member.name).parts
        if not parts or parts[0] != prefix or ".." in parts:
            raise SystemExit(f"Unexpected archive path: {member.name}")
        target = source.joinpath(*parts[1:])
        if member.isdir():
            target.mkdir(parents=True, exist_ok=True)
        elif member.isfile():
            target.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(member) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            target.chmod(member.mode & 0o777)
        else:
            raise SystemExit(f"Unsupported archive member: {member.name}")
baseline = {
    "client.py": "9a52e3e293b557e7230743d1cd09b6707ce8ce42827e73c3ff7b32d886b2dc2a",
    "controller.py": "5ad6e2988516ea474fa3f80e409215ae9020681c70f04c6e550f90848790c64e",
}
for name, expected in baseline.items():
    if hashlib.sha256((source / "transfer_queue" / name).read_bytes()).hexdigest() != expected:
        raise SystemExit(f"Unexpected upstream baseline: {name}")
(pathlib.Path(work) / "source_epoch").write_text(str(epoch))
PY

patch --batch --forward --fuzz=0 -d "${tq_work}/source" -p1 < "${tq_patch}"
"${tq_python}" - "${tq_work}/source" "${tq_commit}" "${tq_archive_sha}" \
    "${tq_patch_sha}" "${tq_version}" <<'PY'
import hashlib
import json
import pathlib
import sys

source = pathlib.Path(sys.argv[1])
commit, archive_sha, patch_sha, version = sys.argv[2:]
version_file = source / "transfer_queue/version/version"
if version_file.read_text().strip() != "0.1.10.dev0":
    raise SystemExit("Unexpected upstream version")
version_file.write_text(version + "\n")
metadata = {
    "version": version,
    "source_commit": commit,
    "source_archive_sha256": archive_sha,
    "patch_sha256": patch_sha,
    "patches": [
        {"name": "9784ad0-assignment-transform.patch", "sha256": patch_sha},
    ],
    "source_files_sha256": {
        name: hashlib.sha256((source / "transfer_queue" / name).read_bytes()).hexdigest()
        for name in ("client.py", "controller.py", "sampler/seqlen_balanced_sampler.py")
    },
}
(source / "transfer_queue/_relax_build.json").write_text(json.dumps(metadata, sort_keys=True, indent=2) + "\n")
# The upstream package-data list does not include JSON files.
project = source / "pyproject.toml"
text = project.read_text()
needle = '    "*.yaml"\n'
if text.count(needle) != 1:
    raise SystemExit("Unexpected upstream package-data configuration")
project.write_text(text.replace(needle, '    "*.yaml",\n    "_relax_build.json"\n'))
PY

export SOURCE_DATE_EPOCH="$(cat "${tq_work}/source_epoch")"
"${tq_python}" -m build --wheel --no-isolation --outdir "${tq_work}/dist" "${tq_work}/source"

# Verify packaged sources and RECORD before publishing the wheel and receipt.
"${tq_python}" - "${tq_work}" "${tq_out}" "${tq_version}" <<'PY'
import base64
import csv
import hashlib
import importlib.metadata
import io
import json
import pathlib
import shutil
import sys
import zipfile
from email.parser import BytesParser

work, out, version = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), sys.argv[3]
wheels = list((work / "dist").glob("*.whl"))
if len(wheels) != 1:
    raise SystemExit("Expected exactly one wheel")
wheel = wheels[0]
with zipfile.ZipFile(wheel) as z:
    metadata = json.loads(z.read("transfer_queue/_relax_build.json"))
    for name, expected in metadata["source_files_sha256"].items():
        if hashlib.sha256(z.read(f"transfer_queue/{name}")).hexdigest() != expected:
            raise SystemExit(f"Packaged source mismatch: {name}")
    entries = z.namelist()
    dist_meta = [x for x in entries if x.endswith(".dist-info/METADATA")]
    records = [x for x in entries if x.endswith(".dist-info/RECORD")]
    if len(dist_meta) != 1 or len(records) != 1:
        raise SystemExit("Unexpected wheel metadata layout")
    if BytesParser().parsebytes(z.read(dist_meta[0]))["Version"] != version:
        raise SystemExit("Unexpected wheel version")
    recorded = set()
    for name, digest, size in csv.reader(io.StringIO(z.read(records[0]).decode())):
        recorded.add(name)
        data = z.read(name)
        if name == records[0] and not digest and not size:
            continue
        expected = "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
        if digest != expected or size != str(len(data)):
            raise SystemExit(f"Invalid wheel RECORD: {name}")
    if set(entries) != recorded:
        raise SystemExit("Wheel entries are missing from RECORD")
receipt = {
    "wheel": wheel.name,
    "wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
    "wheel_bytes": wheel.stat().st_size,
    "build": metadata,
    "python": sys.version.split()[0],
    "build_tools": {name: importlib.metadata.version(name) for name in ("build", "setuptools", "wheel")},
}
out.mkdir(parents=True, exist_ok=True)
receipt_path = out / (wheel.stem + ".build.json")
if (out / wheel.name).exists() or receipt_path.exists():
    raise SystemExit("Refusing to overwrite an existing build")
with (out / wheel.name).open("xb") as dst, wheel.open("rb") as src:
    shutil.copyfileobj(src, dst)
with receipt_path.open("x") as dst:
    json.dump(receipt, dst, sort_keys=True, indent=2)
    dst.write("\n")
print(json.dumps({"wheel": str(out / wheel.name), "sha256": receipt["wheel_sha256"], "receipt": str(receipt_path)}))
PY
