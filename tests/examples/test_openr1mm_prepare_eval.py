# Copyright (c) 2026 Relax Authors. All Rights Reserved.
import io
import json

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from examples.openr1mm.prepare_eval import prepare


def test_prepare_eval_overlap_and_multimodal_reference_order(tmp_path):
    def image(color):
        buf = io.BytesIO()
        Image.new("RGB", (2, 2), color).save(buf, format="PNG")
        return buf.getvalue()

    red, blue, green = image("red"), image("blue"), image("green")
    train = tmp_path / "train.parquet"
    pq.write_table(pa.Table.from_pylist([{"image": [red]}]), train)
    source = tmp_path / "source"
    mv = source / "mathvista/data"
    mm = source / "mmmu/Art"
    mv.mkdir(parents=True)
    mm.mkdir(parents=True)
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "pid": str(i),
                    "decoded_image": {"bytes": value},
                    "query": "Count?",
                    "choices": [],
                    "answer": "2",
                    "answer_type": "integer",
                    "precision": None,
                }
                for i, value in enumerate([red, blue])
            ]
        ),
        mv / "testmini-0.parquet",
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "id": "multi",
                    "question": "Compare <image 2> and <image 1>.",
                    "options": "['<image 2>', 'Other']",
                    "question_type": "multiple-choice",
                    "answer": "A",
                    "image_1": {"bytes": blue},
                    "image_2": {"bytes": green},
                }
            ]
        ),
        mm / "validation-0.parquet",
    )
    output = tmp_path / "out"
    report = prepare(source, train, output)
    assert report["mathvista_testmini"] == {"total": 2, "disjoint": 1, "train_image_overlap_ids": ["0"]}
    assert pq.read_table(output / "mathvista_testmini_disjoint.parquet")["id"].to_pylist() == ["1"]
    row = pq.read_table(output / "mmmu_validation_disjoint.parquet").to_pylist()[0]
    assert row["image"] == [green, blue, green]
    assert row["prompt"][0]["content"].count("<image>") == 3
    assert "<image 2>" not in row["prompt"][0]["content"]
    assert json.loads(row["label"])["correct_index"] == 0
    assert json.loads((output / "overlap-report.json").read_text()) == report


def test_download_eval_rejects_mixed_revisions(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import huggingface_hub
    import pytest

    from examples.openr1mm.download_eval import download

    calls = []
    monkeypatch.setattr(
        huggingface_hub,
        "HfApi",
        lambda: SimpleNamespace(dataset_info=lambda repo, revision: SimpleNamespace(sha=revision)),
    )
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda *args, **kwargs: calls.append(kwargs))
    download(tmp_path, "mv-v1", "mm-v1")
    download(tmp_path, "mv-v1", "mm-v1")
    assert len(calls) == 4
    with pytest.raises(ValueError, match="fresh --output"):
        download(tmp_path, "mv-v2", "mm-v1")
    assert len(calls) == 4
    assert json.loads((tmp_path / "sources.json").read_text())["mathvista"]["revision"] == "mv-v1"
