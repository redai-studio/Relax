# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Download pinned evaluation splits and record their source revisions."""

import argparse
import json
from pathlib import Path


def download(output: Path, mathvista_revision: str, mmmu_revision: str) -> None:
    from huggingface_hub import HfApi, snapshot_download

    output.mkdir(parents=True, exist_ok=True)
    provenance = {}
    for name, repo, revision, patterns in (
        ("mathvista", "AI4Math/MathVista", mathvista_revision, ["data/testmini*.parquet"]),
        ("mmmu", "MMMU/MMMU", mmmu_revision, ["*/validation*.parquet"]),
    ):
        resolved = HfApi().dataset_info(repo, revision=revision).sha
        destination = output / "source" / name
        marker = destination / "source-revision.json"
        expected = {"repo": repo, "revision": resolved}
        if destination.exists() and any(destination.iterdir()):
            if not marker.exists() or json.loads(marker.read_text()) != expected:
                raise ValueError(
                    f"Use a fresh --output directory: {destination} has unknown or different revision data"
                )
        destination.mkdir(parents=True, exist_ok=True)
        # Mark before downloading so an interrupted transfer can resume safely.
        marker.write_text(json.dumps(expected, indent=2) + "\n")
        snapshot_download(
            repo,
            repo_type="dataset",
            revision=resolved,
            allow_patterns=patterns,
            local_dir=destination,
            max_workers=3,
        )
        provenance[name] = {"repo": repo, "revision": resolved, "patterns": patterns}
    (output / "sources.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mathvista-revision", default="2b6ad69445fbb5695c9b165475e8decdbeb97747")
    parser.add_argument("--mmmu-revision", default="876ce5cb130f7f7e290ce4d9984357737d4db5cf")
    args = parser.parse_args()
    download(args.output, args.mathvista_revision, args.mmmu_revision)
