# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import json
import os
import stat

import pytest

from relax.entrypoints.repetition import diagnose_dumps


pytestmark = pytest.mark.skipif(os.name != "posix", reason="POSIX permission bits require a POSIX filesystem")


@pytest.mark.parametrize("mode", [0o600, 0o640, 0o644, 0o664])
def test_repetition_report_preserves_existing_permissions(tmp_path, mode):
    source = tmp_path / "samples.jsonl"
    source.write_text(json.dumps({"response": "hello"}) + "\n", encoding="utf-8")
    output = tmp_path / "report.json"
    output.write_text("previous report", encoding="utf-8")
    output.chmod(mode)

    diagnose_dumps([source], output)

    assert stat.S_IMODE(output.stat().st_mode) == mode
    assert json.loads(output.read_text(encoding="utf-8"))["summary"]["samples"] == 1


def test_repetition_new_report_is_private(tmp_path):
    source = tmp_path / "samples.jsonl"
    source.write_text(json.dumps({"response": "hello"}) + "\n", encoding="utf-8")
    output = tmp_path / "report.json"

    diagnose_dumps([source], output)

    assert stat.S_IMODE(output.stat().st_mode) == 0o600


def test_repetition_failed_report_preserves_permissions_and_contents(tmp_path):
    source = tmp_path / "samples.jsonl"
    source.write_text('{"response": null}\n', encoding="utf-8")
    output = tmp_path / "report.json"
    output.write_text("previous report", encoding="utf-8")
    output.chmod(0o640)

    with pytest.raises(ValueError):
        diagnose_dumps([source], output)

    assert stat.S_IMODE(output.stat().st_mode) == 0o640
    assert output.read_text(encoding="utf-8") == "previous report"
    assert not list(tmp_path.glob(".report.json.*.tmp"))
