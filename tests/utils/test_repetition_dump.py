# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import json
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from relax.entrypoints.repetition import diagnose_dumps, discover_dumps
from relax.utils.repetition import analyze_repetition
from relax.utils.rollout_dump import iter_rollout_records
from relax.utils.training.train_dump_utils import (
    save_debug_rollout_data,
    save_eval_summary_jsonl,
    save_rollout_result_jsonl,
)
from relax.utils.types import Sample


def _samples():
    return [
        Sample(index=17, group_index=2, response="工具观察：重复内容。" * 2000, response_length=3),
        Sample(index=18, group_index=2, response="正常回复", response_length=2),
        Sample(index=19, group_index=3, response="", response_length=0),
    ]


def test_repetition_real_writers_reader_and_cli_round_trip(tmp_path):
    samples = _samples()
    args = SimpleNamespace(
        rollout_result_dir=str(tmp_path / "dumps"), save_debug_rollout_data=str(tmp_path / "{rollout_id}.pt")
    )
    save_rollout_result_jsonl(args, 7, samples)
    save_eval_summary_jsonl(args, 8, {"中文任务": {"samples": samples}})
    save_debug_rollout_data(args, samples, 9, evaluation=False)
    report_path = tmp_path / "report.json"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "relax.entrypoints.repetition",
            str(tmp_path / "dumps"),
            str(tmp_path / "9.pt"),
            "--output",
            str(report_path),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["summary"]["samples"] == 9
    assert report["summary"]["repetition_frac"] == pytest.approx(1 / 3)
    assert report["offsets"] == {"unit": "unicode_code_points", "start": "inclusive", "end": "exclusive", "base": 0}
    assert report["detection"]["compression_level"] == 9
    assert report["detection"]["comparison"] == ">"
    for entry in report["samples"]:
        original = samples[entry["record_index"]]
        assert entry["has_repetition"] == analyze_repetition(original.response).has_repetition
        assert entry["response_chars"] == len(original.response)
        assert entry["covered_chars"] <= entry["response_chars"]
        assert "response" not in entry
        if entry["source"].endswith("9.pt"):
            assert entry["index"] == original.index
            assert entry["sample_index"] is None
            assert entry["rollout_id"] == 9
            assert entry["line_number"] is None
        else:
            assert entry["sample_index"] == entry["record_index"]
            assert entry["line_number"] == entry["record_index"] + 1
            assert entry["group_index"] == original.group_index
            if entry["rollout_id"] == 8:
                assert entry["dataset"] == "中文任务"
    assert "hits=" in result.stderr or "hits=" in result.stdout


def test_repetition_jsonl_preserves_whitespace_unicode_and_physical_line(tmp_path):
    path = tmp_path / "samples.jsonl"
    response = "\n  中😀e\u0301\r\n工具观察：\t "
    path.write_text(
        "\n" + json.dumps({"response": response, "sample_index": "external-id"}) + "\n\n", encoding="utf-8"
    )
    (record,) = iter_rollout_records(path)
    assert record.response == response
    assert record.line_number == 2
    assert record.record_index == 0
    assert record.sample_index == "external-id"


@pytest.mark.parametrize(
    "line, message",
    [
        ('{"response":', "invalid JSON"),
        ("[]", "expected a sample object"),
        ("{}", "response must be present"),
        ('{"response": null}', "response must be present"),
        ('{"response": 123}', "response must be present"),
    ],
)
def test_repetition_bad_record_never_publishes_partial_report(tmp_path, line, message):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"response": "fine"}\n' + line + "\n", encoding="utf-8")
    output = tmp_path / "report.json"
    output.write_text("previous report", encoding="utf-8")
    with pytest.raises(ValueError, match=f"line 2.*{message}"):
        diagnose_dumps([path], output)
    assert output.read_text(encoding="utf-8") == "previous report"
    assert not list(tmp_path.glob("*.tmp"))


def test_repetition_empty_dump_has_valid_zero_summary(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_text("\n", encoding="utf-8")
    output = tmp_path / "report.json"
    summary = diagnose_dumps([path], output)
    assert summary["samples"] == 0
    assert summary["repetition_frac"] == 0.0
    assert summary["max_compression_ratio"] is None
    assert json.loads(output.read_text(encoding="utf-8"))["samples"] == []


def test_repetition_discovery_deduplicates_and_protects_inputs(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text('{"response":"x"}\n', encoding="utf-8")
    output = tmp_path / "report.json"
    assert discover_dumps([path, tmp_path, path], output) == [path.resolve()]
    with pytest.raises(ValueError, match="overwrite"):
        diagnose_dumps([path], path)
    with pytest.raises(FileNotFoundError):
        diagnose_dumps([tmp_path / "absent"], output)
    with pytest.raises(ValueError, match="No rollout dumps"):
        discover_dumps([], output)


def test_repetition_torch_explicit_format_and_legacy_opt_in(tmp_path):
    import numpy as np

    path = tmp_path / "debug.dump"
    samples = _samples()
    samples[0].rollout_routed_experts = np.array([1, 2])
    args = SimpleNamespace(save_debug_rollout_data=str(path))
    save_debug_rollout_data(args, samples, 5, evaluation=False)
    with pytest.raises(ValueError, match="unknown dump extension"):
        list(iter_rollout_records(path))
    with pytest.raises(ValueError, match="trusted-torch"):
        list(iter_rollout_records(path, input_format="torch"))
    records = list(iter_rollout_records(path, input_format="torch", trusted_torch=True))
    assert [record.response for record in records] == [sample.response for sample in samples]
    assert all(record.rollout_id == 5 for record in records)


@pytest.mark.parametrize("dump", [{}, {"samples": {}}, [1, 2], {"samples": [{"response": None}]}])
def test_repetition_torch_rejects_wrong_schema(tmp_path, dump):
    path = tmp_path / "bad.pt"
    torch.save(dump, path)
    with pytest.raises(ValueError):
        list(iter_rollout_records(path))


def test_repetition_report_rejects_non_json_identifiers_with_location(tmp_path):
    path = tmp_path / "bad-id.pt"
    torch.save({"samples": [{"response": "hello", "index": torch.tensor(1)}]}, path)
    output = tmp_path / "report.json"
    with pytest.raises(ValueError, match="record 0: sample identifiers must be JSON-compatible"):
        diagnose_dumps([path], output)
    assert not output.exists()


def test_repetition_cli_rejects_invalid_configuration(tmp_path):
    output = tmp_path / "report.json"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "relax.entrypoints.repetition",
            str(tmp_path),
            "--output",
            str(output),
            "--stride",
            "10001",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "stride must not exceed" in result.stderr
    assert not output.exists()


def test_repetition_cli_jsonl_does_not_import_training_dependencies(tmp_path):
    """Import and run the production CLI while rejecting heavy imports
    outright."""
    path = tmp_path / "sample.jsonl"
    path.write_text('{"response":"hello"}\n', encoding="utf-8")
    script = """
import sys
class RejectHeavyImports:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'numpy', 'ray', 'sglang', 'transformers'}:
            raise AssertionError('unexpected heavy import: ' + fullname)
sys.meta_path.insert(0, RejectHeavyImports())
from relax.entrypoints.repetition import main
main(sys.argv[1:])
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(path), "--output", str(tmp_path / "report.json")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
