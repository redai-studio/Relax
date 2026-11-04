import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
from runner import campaign


def test_public_cli_runs_from_example_directory_and_accepts_selected_devices(tmp_path):
    completed = subprocess.run(
        [sys.executable, "-m", "runner.campaign", "run", "--help"],
        cwd=campaign.KIT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "--prepared" in completed.stdout and "--devices" in completed.stdout
    args = campaign.parser().parse_args(
        ["run", "--prepared", "assets/prepared.json", "--work", str(tmp_path), "--devices", "5", "--world", "1"]
    )
    assert args.devices == "5" and args.world == 1


def test_prepared_validation_reads_actual_asset_bytes(tmp_path, monkeypatch):
    from assets.prepare import identity

    def file(path: Path, content: bytes) -> dict:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return {"path": path.name, "size": len(content), "sha256": hashlib.sha256(content).hexdigest()}

    kit = tmp_path / "example"
    weights = file(tmp_path / "model" / "weights.bin", b"pinned-model")
    image = file(tmp_path / "data" / "image.png", b"pinned-image")
    lock = {"models": {"toy": {"files": [weights]}}, "data": {"files": []}}
    lock_path = kit / "assets" / "manifest.json"
    lock_path.parent.mkdir(parents=True)
    lock_path.write_text(json.dumps(lock))
    metadata = {}
    hashes = {}
    for subset, count in (("debug8", 8), ("high128", 128), ("low1024", 1024)):
        path = tmp_path / "data" / f"{subset}.json"
        path.write_text(json.dumps([{"image": "image.png"}] * count))
        metadata[subset] = str(path)
        hashes[subset] = campaign.digest(path)
    ready = tmp_path / "data" / "data-ready.json"
    ready.write_text(json.dumps({"source_identity": identity([]), "metadata_sha256": hashes, "images": [image]}))
    prepared = {
        "state": "READY",
        "input_manifest_sha256": campaign.digest(lock_path),
        "models": {"toy": {"identity": identity([weights]), "path": str(tmp_path / "model")}},
        "data": {"root": str(tmp_path / "data"), "receipt": str(ready), **metadata},
    }
    path = tmp_path / "prepared.json"
    path.write_text(json.dumps(prepared))
    monkeypatch.setattr(campaign, "KIT", kit)
    assert campaign.validate_prepared(path) == prepared
    (tmp_path / "model" / "weights.bin").write_bytes(b"broken-model")
    with pytest.raises(ValueError, match="SHA-256"):
        campaign.validate_prepared(path)


def test_fixed_native_recipe_cannot_silently_ignore_config_edits():
    config = json.loads((campaign.KIT / "configs/campaign.json").read_text())
    campaign.validate_recipe(config)
    for change in ({"global_batch": 4}, {"learning_rate": 2e-4}):
        with pytest.raises(ValueError, match="fixed native benchmark"):
            campaign.validate_recipe(config | change)
