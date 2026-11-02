# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Installation upgrades preserve original and intermediate runtime backups."""

import difflib
import importlib.util
import json
import sys
from pathlib import Path

import pytest


def test_upgrade_backups_and_unknown_source_are_transactional(tmp_path, monkeypatch):
    source = Path(__file__).resolve().parents[2] / "scripts/tools/apply-unified-image-compat.py"
    spec = importlib.util.spec_from_file_location("image_compat", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fake_script = tmp_path / "repo/scripts/tools/install.py"
    monkeypatch.setattr(module, "__file__", str(fake_script))
    catalog = tmp_path / "repo/docker/patch/image-compat/qs-20261021.json"
    catalog.parent.mkdir(parents=True)
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    first, second = runtime / "first.py", runtime / "second.py"
    first.write_text("VALUE = 1\n")
    second.write_text("VALUE = 1\n")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"megatron_sources": {}}))
    original_manifest = manifest.read_bytes()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(fake_script),
            "--megatron-root",
            str(runtime),
            "--sglang-root",
            str(runtime),
            "--transferqueue-root",
            str(runtime),
            "--manifest",
            str(manifest),
        ],
    )

    def record(name, before, after):
        return dict(
            component="megatron",
            path=name,
            before=module.digest(before.encode()),
            after=module.digest(after.encode()),
            patch="".join(
                difflib.unified_diff(
                    before.splitlines(True), after.splitlines(True), fromfile="a/" + name, tofile="b/" + name
                )
            ),
        )

    catalog.write_text(json.dumps({"files": [record("first.py", "VALUE = 1\n", "VALUE = 2\n")]}))
    module.main()
    intermediate = manifest.read_bytes()
    module.main()
    assert manifest.read_bytes() == intermediate
    catalog.write_text(
        json.dumps(
            {
                "files": [
                    record("first.py", "VALUE = 1\n", "VALUE = 2\n"),
                    record("second.py", "VALUE = 1\n", "VALUE = 2\n"),
                ]
            }
        )
    )
    second.write_text("UNRECOGNIZED = True\n")
    with pytest.raises(ValueError, match="Unknown source version"):
        module.main()
    assert manifest.read_bytes() == intermediate
    assert first.read_text() == "VALUE = 2\n"
    second.write_text("VALUE = 1\n")
    module.main()
    assert second.read_text() == "VALUE = 2\n"
    base_backup = manifest.with_name(manifest.name + ".pre-unified-image-compat")
    assert base_backup.read_bytes() == original_manifest
    versioned = base_backup.with_name(base_backup.name + "." + module.digest(intermediate))
    assert versioned.read_bytes() == intermediate
    assert first.with_name("first.py.pre-unified-image-compat").read_text() == "VALUE = 1\n"
    module.main()

    upgraded = record("first.py", "VALUE = 1\n", "VALUE = 3\n")
    delta = record("first.py", "VALUE = 2\n", "VALUE = 3\n")
    upgraded["upgrades"] = [{"before": delta["before"], "patch": delta["patch"]}]
    catalog.write_text(json.dumps({"files": [upgraded, record("second.py", "VALUE = 1\n", "VALUE = 2\n")]}))
    module.main()
    assert first.read_text() == "VALUE = 3\n"
    intermediate_backup = first.with_name("first.py.pre-unified-image-compat." + delta["before"])
    assert intermediate_backup.read_text() == "VALUE = 2\n"
    latest_manifest = manifest.read_bytes()
    module.main()
    assert manifest.read_bytes() == latest_manifest
    # The same catalog also installs directly from the original image source.
    first.write_text("VALUE = 1\n")
    manifest.write_bytes(original_manifest)
    module.main()
    assert first.read_text() == "VALUE = 3\n"
