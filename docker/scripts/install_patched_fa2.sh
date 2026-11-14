#!/bin/sh
set -eu

patch_file=$1
build_dir=$(mktemp -d)
trap 'rm -rf -- "$build_dir"' EXIT
cd "$build_dir"

python -m pip download --no-binary flash-attn --no-deps --no-build-isolation --no-cache-dir flash-attn==2.7.4.post1
tar -xzf flash_attn-2.7.4.post1.tar.gz
cd flash_attn-2.7.4.post1
patch --batch --forward --fuzz=0 -p1 < "$patch_file"
FLASH_ATTENTION_FORCE_BUILD=TRUE python -m pip -v install --force-reinstall --no-deps --no-build-isolation --no-cache-dir .
