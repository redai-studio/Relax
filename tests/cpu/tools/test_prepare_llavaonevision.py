# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import base64
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts.tools import prepare_llavaonevision as prepare


def _source(path):
    good = {
        "id": "good",
        "conversations": [{"from": "human", "value": "<image>look"}, {"from": "gpt", "value": "yes"}],
        "image": [{"bytes": b"test-image", "path": None}],
    }
    bad = {**good, "id": "bad", "conversations": [{"from": "human", "value": "<image>look"}]}
    pq.write_table(pa.Table.from_pylist([good, bad, {**good, "id": "second"}]), path, row_group_size=1)


def test_onevision_shard_extracts_images_rejects_bad_rows_and_resumes(tmp_path):
    source = tmp_path / "source.parquet"
    _source(source)
    relative = Path("subset/source.parquet")
    stats = prepare.convert_shard(source, relative, tmp_path)
    assert (stats["rows"], stats["accepted"], stats["rejected"], stats["images"]) == (3, 2, 1, 2)
    output = tmp_path / "sft/train/subset/source.jsonl"
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert [row["metadata"]["id"] for row in rows] == ["good", "second"]
    assert all(
        Path(row["images"][0]).is_absolute() and Path(row["images"][0]).read_bytes() == b"test-image" for row in rows
    )
    before = output.stat().st_mtime_ns
    assert prepare.convert_shard(source, relative, tmp_path) == stats
    assert output.stat().st_mtime_ns == before
    assert not output.with_suffix(".jsonl.tmp").exists()


def test_onevision_sample_counts_only_valid_rows_and_reuses_plan(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    source = raw / "source.parquet"
    _source(source)
    prepare.write_json(
        tmp_path / "source_manifest.json",
        {"revision": "fixture", "files": [{"path": source.name, "size": source.stat().st_size}]},
    )
    output = prepare.sample(tmp_path, count=2, seed=7, workers=1)
    ready = json.loads((output / "READY.json").read_text())
    assert ready["rows"] == 2 and ready["population_rejected"] == 1
    rows = [json.loads(line) for line in (output / "train/source.jsonl").read_text().splitlines()]
    assert all(base64.b64decode(row["images"][0].split(",", 1)[1]) == b"test-image" for row in rows)
    assert prepare.sample(tmp_path, count=2, seed=7, workers=1) == output
    with pytest.raises(ValueError, match="different parameters"):
        prepare.sample(tmp_path, count=2, seed=8, workers=1)
    with pytest.raises(ValueError, match="only 2 valid rows"):
        prepare.sample(tmp_path, count=3, seed=7, workers=1)
