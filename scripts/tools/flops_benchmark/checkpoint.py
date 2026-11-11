# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Create a native checkpoint view with a stable fingerprint for paired
runs."""

import argparse
import json
from pathlib import Path

from scripts.tools.flops_benchmark.common import digest, file_digest, write_json


def prepare_view(source: Path, destination: Path) -> str:
    source = source.resolve()
    tracker = source / "latest_checkpointed_iteration.txt"
    iteration = int(tracker.read_text().strip())
    native = source / f"iter_{iteration:07d}"
    if iteration != 19 or not native.is_dir():
        raise ValueError("A complete native warmup checkpoint is required")
    files = {str(path.relative_to(source)): file_digest(path) for path in sorted(native.rglob("*")) if path.is_file()}
    if not files:
        raise ValueError("Empty warmup checkpoint")
    destination.mkdir(parents=True, exist_ok=False)
    # Omitting the SFT cursor uses the production replay path for changed data/budgets.
    (destination / tracker.name).symlink_to(tracker)
    (destination / native.name).symlink_to(native, target_is_directory=True)
    identity = digest({"iteration": iteration, "files": files})
    write_json(
        destination / "fingerprint.json",
        {"initial_state_id": identity, "iteration": iteration, "source": str(source), "files": files},
    )
    return identity


def verify_view(destination: Path) -> str:
    manifest = json.loads((destination / "fingerprint.json").read_text())
    tracker = destination / "latest_checkpointed_iteration.txt"
    if int(tracker.read_text()) != manifest["iteration"] or manifest["iteration"] != 19:
        raise ValueError("Warmup checkpoint iteration changed")
    native = destination / f"iter_{manifest['iteration']:07d}"
    files = {
        str(path.relative_to(destination)): file_digest(path) for path in sorted(native.rglob("*")) if path.is_file()
    }
    identity = digest({"iteration": manifest["iteration"], "files": files})
    if files != manifest["files"] or identity != manifest["initial_state_id"]:
        raise ValueError("Warmup checkpoint contents changed")
    return identity


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    prepare_view(args.source, args.destination)


if __name__ == "__main__":
    main()
