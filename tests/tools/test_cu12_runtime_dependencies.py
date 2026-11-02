# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Check the TE metadata repair without changing the host installation."""

import base64
import csv
import hashlib
import importlib.metadata as md
import importlib.util
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest
from packaging.tags import parse_tag


ROOT = Path(__file__).resolve().parents[2]
PATCH = ROOT / "docker/patch/transformer_engine/2.14.1-cu12-wheel-tag.patch"


@pytest.fixture
def repair_module(monkeypatch):
    spec = importlib.util.spec_from_file_location("repair_te", ROOT / "docker/scripts/repair_te_wheel_metadata.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "sys_tags", lambda: iter(parse_tag(module.EXPECTED_TAG)))
    return module


def make_distribution(tmp_path, tag="cp310-cp310-manylinux_2_28_x86_64", version="2.14.1", extension=False):
    info = tmp_path / f"transformer_engine_cu12-{version}.dist-info"
    info.mkdir()
    (info / "METADATA").write_text(f"Metadata-Version: 2.1\nName: transformer-engine-cu12\nVersion: {version}\n")
    (info / "WHEEL").write_text(
        f"Wheel-Version: 1.0\nGenerator: setuptools (82.0.1)\nRoot-Is-Purelib: false\nTag: {tag}\n\n"
    )
    entries = [f"{info.name}/WHEEL,,", f"{info.name}/RECORD,,", "transformer_engine/libtransformer_engine.so,,"]
    if extension:
        entries.append("transformer_engine/extension.cpython-310-x86_64-linux-gnu.so,,")
        binary = tmp_path / "transformer_engine/extension.cpython-310-x86_64-linux-gnu.so"
        binary.parent.mkdir()
        binary.touch()
    (info / "RECORD").write_text("\n".join(entries) + "\n")
    return md.Distribution.at(info), info


def test_te_library_tag_repair_updates_record_and_is_idempotent(tmp_path, repair_module):
    distribution, info = make_distribution(tmp_path)
    repair_module.repair(distribution, PATCH)
    content = (info / "WHEEL").read_bytes()
    assert b"Tag: py3-none-manylinux_2_28_x86_64" in content
    with (info / "RECORD").open(newline="") as stream:
        row = next(row for row in csv.reader(stream) if row[0].endswith("/WHEEL"))
    digest = base64.urlsafe_b64encode(hashlib.sha256(content).digest()).rstrip(b"=").decode()
    assert row[1:] == ["sha256=" + digest, str(len(content))]
    record = (info / "RECORD").read_bytes()
    repair_module.repair(distribution, PATCH)
    assert (info / "WHEEL").read_bytes() == content
    assert (info / "RECORD").read_bytes() == record


def test_te_repair_accepts_pip_generated_cpython_bytecode(tmp_path, repair_module):
    _, info = make_distribution(tmp_path)
    wheel = tmp_path / "transformer_engine_cu12-2.14.1-py3-none-manylinux_2_28_x86_64.whl"
    # Reproduce the upstream filename/internal-tag mismatch through real pip.
    # This fixture exercises packaging only, not the TE binary or its GPU ABI.
    entries = {
        f"{info.name}/METADATA": (info / "METADATA").read_bytes(),
        f"{info.name}/WHEEL": (info / "WHEEL").read_bytes(),
        "transformer_engine/__init__.py": b"VERSION = '2.14.1'\n",
    }
    entries[f"{info.name}/RECORD"] = "".join(f"{name},,\n" for name in entries).encode()
    with zipfile.ZipFile(wheel, "w") as archive:
        for name, content in entries.items():
            archive.writestr(name, content)
    target = tmp_path / "installed"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-index",
            "--no-deps",
            "--compile",
            "--platform",
            "manylinux_2_28_x86_64",
            "--target",
            str(target),
            str(wheel),
        ],
        check=True,
        capture_output=True,
    )
    installed = md.Distribution.at(target / info.name)
    bytecode = [path for path in installed.files if ".cpython-" in path.name and path.suffix == ".pyc"]
    assert bytecode, "pip must generate the bytecode that triggered the build failure"
    repair_module.repair(installed, PATCH)
    assert f"Tag: {repair_module.EXPECTED_TAG}" in installed.read_text("WHEEL")
    assert all(Path(installed.locate_file(path)).is_file() for path in bytecode)


@pytest.mark.parametrize(
    "options,message",
    [
        ({"tag": "cp311-cp311-manylinux_2_28_x86_64"}, "Unexpected TE wheel tags"),
        ({"version": "2.15.0"}, "Unexpected TE library version"),
        ({"extension": True}, "CPython extension"),
    ],
)
def test_te_repair_rejects_unrecognized_binary_metadata(tmp_path, repair_module, options, message):
    distribution, info = make_distribution(tmp_path, **options)
    before = (info / "WHEEL").read_bytes()
    with pytest.raises(ValueError, match=message):
        repair_module.repair(distribution, PATCH)
    assert (info / "WHEEL").read_bytes() == before


def test_te_repair_does_not_hide_unsupported_host(tmp_path, repair_module, monkeypatch):
    distribution, _ = make_distribution(tmp_path)
    monkeypatch.setattr(repair_module, "sys_tags", lambda: iter(parse_tag("py3-none-manylinux_2_28_aarch64")))
    with pytest.raises(ValueError, match="only supports"):
        repair_module.repair(distribution, PATCH)
