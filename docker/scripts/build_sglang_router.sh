#!/usr/bin/env bash
set -euo pipefail

PATCH_DIR=$(realpath "${1:?usage: build_sglang_router.sh PATCH_DIR WHEEL_DIR}")
WHEEL_DIR=$(realpath -m "${2:?usage: build_sglang_router.sh PATCH_DIR WHEEL_DIR}")
ROUTER_COMMIT=9daabcd801c861088d93b8ce8a2a831490ce4bee
ARCHIVE_SHA256=27c0ddc073814b9fd7447e4a481e580fd225c662fd2bfdfe5ef5502510da23df
BUILD_DIR=$(mktemp -d)
trap 'rm -rf "$BUILD_DIR"' EXIT

# A supplied archive is still checked against the pinned upstream source hash.
ARCHIVE=${ROUTER_SOURCE_ARCHIVE:-$BUILD_DIR/source.tar.gz}
if [ -z "${ROUTER_SOURCE_ARCHIVE:-}" ]; then
    curl --fail --location --retry 3 \
        "https://codeload.github.com/zhuzilin/sgl-router/tar.gz/$ROUTER_COMMIT" -o "$ARCHIVE"
fi
printf '%s  %s\n' "$ARCHIVE_SHA256" "$ARCHIVE" | sha256sum --check
mkdir -p "$BUILD_DIR/source" "$WHEEL_DIR"
tar -xzf "$ARCHIVE" -C "$BUILD_DIR/source" --strip-components=1
cd "$BUILD_DIR/source"
git apply --check "$PATCH_DIR/v0.3.2-9daabcd-pd-dp-aware.patch"
git apply "$PATCH_DIR/v0.3.2-9daabcd-pd-dp-aware.patch"
cp "$PATCH_DIR/v0.3.2-9daabcd-Cargo.lock" bindings/python/Cargo.lock

# Identify the patched wheel independently of an inherited stock/slime wheel.
python - "$PATCH_DIR" "$ROUTER_COMMIT" "$ARCHIVE_SHA256" <<'PY'
import hashlib
import json
from pathlib import Path
import subprocess
import sys

patch_dir = Path(sys.argv[1])
package = Path("bindings/python/src/sglang_router")
version = "0.3.2+slime.dpaware"
project = Path("bindings/python/pyproject.toml")
text = project.read_text()
assert text.count('version = "0.3.2"') == 1
project.write_text(text.replace('version = "0.3.2"', f'version = "{version}"'))
(package / "version.py").write_text(f'__version__ = "{version}"\n')
record = {
    "version": version,
    "source_commit": sys.argv[2],
    "source_archive_sha256": sys.argv[3],
    "patch_sha256": hashlib.sha256((patch_dir / "v0.3.2-9daabcd-pd-dp-aware.patch").read_bytes()).hexdigest(),
    "cargo_lock_sha256": hashlib.sha256(Path("bindings/python/Cargo.lock").read_bytes()).hexdigest(),
    "rustc": subprocess.check_output(["rustc", "--version"], text=True).strip(),
}
(package / "_relax_build.json").write_text(json.dumps(record, indent=2) + "\n")
PY

# The builder and final image share TRAIN_IMAGE, including libc and OpenSSL.
python -m maturin build --release --locked --compatibility linux \
    --manifest-path bindings/python/Cargo.toml --out "$WHEEL_DIR"
