import json
import shutil
import subprocess
import sys

import pytest

from relax.distributed.weight_sync.seal import read_seal, release_seal, seal_versions
from relax.distributed.weight_sync.store import VersionStore


def make_store(path):
    return VersionStore(path, disk_budget=8 * 1024**2, max_weight_bytes=1024)


def publish_sequence(root, count=7):
    store = make_store(root / "source")
    expected = {}
    value = bytearray(range(128))
    for version in range(count):
        value[version] ^= 7
        expected[version] = bytes(value)
        source = root / "input.bin"
        source.write_bytes(value)
        store.publish(
            "run", version, {"w": ("BF16", [64], source)}, anchor_interval=2, chunk_bytes=64, model_config={}
        )
    return store, expected


def test_seal_retains_dependency_closure_until_explicit_release(tmp_path):
    store, _ = publish_sequence(tmp_path)
    record = seal_versions(store, "run", [3, 1], "evaluation")
    assert record["versions"] == [1, 3]
    assert [entry["version"] for entry in record["closure"]] == [0, 1, 2, 3]
    assert seal_versions(store, "run", [1, 3], "evaluation") == record
    store.collect()
    assert read_seal(store, "evaluation") == record
    release_seal(store, "evaluation")
    assert store.manifest("run", 1)["version"] == 1
    store.collect()
    with pytest.raises(FileNotFoundError):
        store.manifest("run", 1)


def test_sealed_files_replay_without_original_store_or_process(tmp_path):
    store, expected = publish_sequence(tmp_path)
    sealed = seal_versions(store, "run", [1, 3], "evaluation")
    detached = tmp_path / "detached"
    # Only delivery objects, exact manifests and the inventory move. Source
    # canonical weights, caches and input checkpoint are deliberately absent.
    for directory in ("objects", "seals"):
        shutil.copytree(store.root / directory, detached / directory)
    for entry in sealed["closure"]:
        original = store.manifest_path("run", entry["version"])
        target = detached / original.relative_to(store.root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, target)
    store.root.rename(tmp_path / "unavailable-source")
    (tmp_path / "input.bin").unlink()
    common = [
        sys.executable,
        "-m",
        "relax.distributed.weight_sync",
        "--disk-budget",
        "8388608",
        "--max-weight-bytes",
        "1024",
    ]
    for version in sealed["versions"]:
        output = tmp_path / f"checkpoint-{version}"
        result = subprocess.run(
            [
                *common,
                "--store",
                str(tmp_path / "fresh-reader"),
                "fetch",
                "--source-store",
                str(detached),
                "--stream",
                "run",
                "--version",
                str(version),
                "--checkpoint-output",
                str(output),
            ],
            capture_output=True,
            check=True,
        )
        assert json.loads(result.stdout)["root"] == next(
            e["root"] for e in sealed["closure"] if e["version"] == version
        )
        assert (output / "model.safetensors").read_bytes()[-128:] == expected[version]


@pytest.mark.parametrize("damage", ["object", "missing_base", "missing_config"])
def test_invalid_delivery_never_exposes_finished_seal(tmp_path, damage):
    store, _ = publish_sequence(tmp_path)
    if damage == "object":
        key = store.manifest("run", 1)["tensors"][0]["chunks"][0]["object"]
        store.object_path(key).write_bytes(b"corrupt")
    elif damage == "missing_base":
        store.manifest_path("run", 0).unlink()
    else:
        source = tmp_path / "input.bin"
        store.publish("opaque", 0, {"w": ("BF16", [64], source)})
    with pytest.raises((ValueError, FileNotFoundError)):
        seal_versions(
            store,
            "opaque" if damage == "missing_config" else "run",
            [0] if damage == "missing_config" else [1],
            "failed",
        )
    assert not (store.root / "seals/failed.json").exists()


def test_conflicting_name_preserves_original_seal(tmp_path):
    store, _ = publish_sequence(tmp_path)
    original = seal_versions(store, "run", [1], "review")
    with pytest.raises(ValueError, match="another set"):
        seal_versions(store, "run", [3], "review")
    assert read_seal(store, "review") == original


def test_interrupted_inventory_commit_is_retryable(tmp_path, monkeypatch):
    from relax.distributed.weight_sync import seal

    store, _ = publish_sequence(tmp_path)
    original = seal.atomic_write
    with monkeypatch.context() as patch:
        patch.setattr(seal, "atomic_write", lambda *args: (_ for _ in ()).throw(OSError("injected write failure")))
        with pytest.raises(OSError, match="injected"):
            seal_versions(store, "run", [1], "review")
    assert not (store.root / "seals/review.json").exists()
    assert seal.atomic_write == original
    assert seal_versions(store, "run", [1], "review")["versions"] == [1]


def test_seal_cli_and_corrupt_inventory_collection(tmp_path):
    store, _ = publish_sequence(tmp_path)
    command = [
        sys.executable,
        "-m",
        "relax.distributed.weight_sync",
        "--store",
        str(store.root),
        "--disk-budget",
        "8388608",
        "--max-weight-bytes",
        "1024",
    ]
    result = subprocess.run(
        [*command, "seal", "--name", "archive", "--stream", "run", "--versions", "1", "3"],
        capture_output=True,
        check=True,
    )
    assert json.loads(result.stdout)["versions"] == [1, 3]
    path = store.root / "seals/archive.json"
    original = path.read_bytes()
    path.write_bytes(original.replace(b'"versions":[1,3]', b'"versions":[3]'))
    with pytest.raises(ValueError, match="inventory"):
        store.collect()
    path.write_bytes(original)
    subprocess.run([*command, "release-seal", "--name", "archive"], check=True, capture_output=True)
    assert not path.exists()
