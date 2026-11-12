#!/usr/bin/env bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
set -euo pipefail

source_dir="${1:?Provide the Git checkout}"
source_sha="${2:?Provide the exact Relax commit}"
install_dir="${3:-/opt}"
bare_repo="$install_dir/relax-source.git"
checkout="$install_dir/relax"

# The build context is read-only and can have a different owner inside Docker.
test "$(git -c safe.directory="$source_dir" -C "$source_dir" rev-parse HEAD)" = "$source_sha"
test ! -e "$bare_repo"
test ! -e "$checkout/.git"

staging="$(mktemp -d)"
trap 'rm -rf "$staging"' EXIT
mkdir -p "$checkout"
git -c safe.directory="$source_dir" clone --bare --no-local --depth 1 --single-branch -- \
    "$source_dir" "$bare_repo"
git -C "$bare_repo" update-ref refs/heads/relax-runtime "$source_sha"
git -C "$bare_repo" symbolic-ref HEAD refs/heads/relax-runtime
git clone --no-local --depth 1 --single-branch --branch relax-runtime -- "$bare_repo" "$staging/checkout"
cp -a "$staging/checkout/." "$checkout/"

# QS fetches before it checks out the requested commit.
git -C "$checkout" fetch
git -C "$checkout" checkout --detach "$source_sha"
test "$(git -C "$checkout" rev-parse HEAD)" = "$source_sha"
git -C "$checkout" diff --exit-code
printf '[relax-source] checked out %s\n' "$source_sha"
