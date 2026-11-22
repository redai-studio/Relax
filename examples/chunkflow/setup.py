# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Apply or verify the example integration against its pinned Relax source."""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path


TARGETS = (
    "relax/utils/arguments.py",
    "relax/backends/megatron/model_provider.py",
    "relax/backends/megatron/model.py",
)


def _git(root: Path, *args: str) -> bytes:
    result = subprocess.run(["git", "-C", str(root), *args], capture_output=True)
    if result.returncode:
        raise RuntimeError(result.stderr.decode().strip() or f"git {' '.join(args)} failed")
    return result.stdout


def prepare(root: Path, *, check: bool = False, reverse: bool = False) -> None:
    """Accept only pinned core sources or the exact already-applied patch.

    Descendant checkouts are allowed when their Relax sources still match the
    pin. This lets the example live in the same repository without requiring
    its introduction commit to be its own dependency.
    """
    root = root.resolve()
    example = root / "examples/chunkflow"
    revision = (example / "relax.version").read_text().strip()
    patch = example / "relax.patch"
    changed = _git(
        root,
        "diff",
        "--name-only",
        revision,
        "--",
        "relax",
        *(f":(exclude){path}" for path in TARGETS),
    )
    if changed:
        raise RuntimeError(f"Relax sources differ from the supported commit {revision}:\n{changed.decode().strip()}")
    originals = {path: _git(root, "show", f"{revision}:{path}") for path in TARGETS}
    # Build the expected patched sources outside the user's checkout before
    # deciding whether it is safe to touch any files.
    with tempfile.TemporaryDirectory(prefix="relax-example-") as temporary:
        staging = Path(temporary)
        for path, contents in originals.items():
            target = staging / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(contents)
        _git(staging, "apply", str(patch))
        patched = {path: (staging / path).read_bytes() for path in TARGETS}

    current = {path: (root / path).read_bytes() for path in TARGETS}
    if current != originals and current != patched:
        raise RuntimeError("Relax integration files contain incompatible or partial changes; no files were modified")
    if check:
        if current != patched:
            raise RuntimeError("Apply the example first: python examples/chunkflow/setup.py")
        return
    desired = originals if reverse else patched
    if current == desired:
        return
    options = ("--reverse",) if reverse else ()
    _git(root, "apply", "--check", *options, str(patch))
    _git(root, "apply", *options, str(patch))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--relax-root", type=Path, default=Path(__file__).resolve().parents[2])
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--check", action="store_true", help="Verify that the exact integration patch is applied")
    action.add_argument("--reverse", action="store_true", help="Remove the integration patch")
    args = parser.parse_args()
    try:
        prepare(args.relax_root, check=args.check, reverse=args.reverse)
    except (OSError, RuntimeError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
