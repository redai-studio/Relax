# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Prepare local GPU test artifacts; no downloads, server launch or
training."""

import argparse
import json
from pathlib import Path

from relax.engine.lora.artifact import PROVENANCE
from relax.engine.lora.snapshot import snapshot_adapter

from .fixtures import make_fixtures


def prepare(model: Path, output: Path) -> None:
    model = model.resolve(strict=True)
    output = output.resolve()
    # Never replace an experiment's fixtures, registered tolerances or results.
    output.mkdir(parents=True, exist_ok=False)
    fixtures = output / "fixtures"
    make_fixtures(model, fixtures, rank=8)
    base = json.loads((fixtures / "A" / PROVENANCE).read_text())["base_model_digest"]
    snapshots = {}
    for version, source in (("A", "A"), ("B", "B"), ("C", "B")):
        snapshot = snapshot_adapter(fixtures / source, output / "store", version_id=version, base_model_digest=base)
        snapshots[version] = {"path": str(snapshot.path), "digest": snapshot.digest}
    profile = json.loads(Path(__file__).with_name("profile.json").read_text())
    profile.update(
        model_path=str(model),
        barrier_directory=str(output / "results"),
        test_control_file=str(output / "results" / "control.json"),
    )
    (output / "results").mkdir()
    (output / "results" / "control.json").write_text("{}\n")
    (output / "verification.json").write_text(json.dumps(profile, indent=2) + "\n")
    # JSON is also valid YAML for PublicationConfig.read().
    publication = {
        "artifact_store": str(output / "store"),
        "capacity": 2,
        "engines_per_gpu": 1,
        "bootstrap_version_id": "A",
        "auto_publish": False,
    }
    (output / "publication.yaml").write_text(json.dumps(publication, indent=2) + "\n")
    (output / "artifacts.json").write_text(
        json.dumps(
            {"model_path": str(model), "base_model_digest": base, "producer": "relax.fixture", "versions": snapshots},
            indent=2,
        )
        + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, required=True, help="New directory; existing experiments are not overwritten"
    )
    args = parser.parse_args()
    prepare(args.model, args.output)


if __name__ == "__main__":
    main()
