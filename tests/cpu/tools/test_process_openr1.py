# Copyright (c) 2026 Relax Authors. All Rights Reserved.
import io
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image

from scripts.tools.process_openr1 import prepare_eval as prepare
from scripts.tools.process_openr1 import prepare_sft


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
    config = json.loads((output / "eval.yaml").read_text())
    assert len(config["eval"]["datasets"]) == 2
    assert config["eval"]["datasets"][0]["path"] == str((output / "mathvista_testmini_disjoint.parquet").resolve())


def test_download_eval_rejects_mixed_revisions(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import huggingface_hub
    import pytest

    from scripts.tools.process_openr1 import download_eval as download

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
    with pytest.raises(ValueError, match="fresh --output-dir"):
        download(tmp_path, "mv-v2", "mm-v1")
    assert len(calls) == 4
    assert json.loads((tmp_path / "sources.json").read_text())["mathvista"]["revision"] == "mv-v1"


def test_process_openr1_default_train_mode_is_compatible(tmp_path):
    from scripts.tools.process_openr1 import main

    source = tmp_path / "raw.parquet"
    output = tmp_path / "train.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [{"problem": " Question? ", "image": {"bytes": b"original-image"}, "solution": "answer"}]
        ),
        source,
    )
    main(["--input-dir", str(source), "--output-dir", str(output)])
    assert pq.read_table(output).to_pylist() == [
        {"prompt": [{"role": "user", "content": "<image>Question?"}], "image": [b"original-image"], "label": "answer"}
    ]


def _sft_source(tmp_path, *, bad_second_row=False):
    data = io.BytesIO()
    Image.new("RGB", (2, 2), "blue").save(data, format="PNG")
    source = tmp_path / "sft.parquet"
    rows = [
        {"problem": " Question? ", "solution": "<think>reason</think>\nanswer", "image": {"bytes": data.getvalue()}}
    ]
    if bad_second_row:
        rows.append({**rows[0], "solution": "  "})
    pq.write_table(pa.Table.from_pylist(rows), source)
    return source, data.getvalue()


def test_prepare_sft_keeps_solution_and_extracts_real_image(tmp_path):
    source, image = _sft_source(tmp_path)
    output = tmp_path / "out"
    prepare_sft(source, output)
    row = json.loads((output / "sft.jsonl").read_text())
    assert row["id"] == "openr1mm-0"
    assert row["messages"] == [
        {"role": "user", "content": "<image>\nQuestion?"},
        {"role": "assistant", "content": "<think>reason</think>\nanswer"},
    ]
    from pathlib import Path

    assert Path(row["images"][0]).read_bytes() == image
    assert json.loads((output / "conversion.json").read_text())["rows"] == 1


@pytest.mark.parametrize("existing", [False, True])
def test_prepare_sft_bad_row_never_publishes_or_overwrites_dataset(tmp_path, existing):
    source, _ = _sft_source(tmp_path, bad_second_row=True)
    output = tmp_path / "out"
    output.mkdir()
    dataset = output / "sft.jsonl"
    if existing:
        dataset.write_text("previous dataset\n")
    with pytest.raises(ValueError, match="Empty problem or solution at row 1"):
        prepare_sft(source, output)
    if existing:
        assert dataset.read_text() == "previous dataset\n"
    else:
        assert not dataset.exists()
    assert not list(output.glob(".sft-*"))


def test_prepare_sft_empty_source_is_rejected_without_publishing(tmp_path):
    source = tmp_path / "empty.parquet"
    pq.write_table(pa.Table.from_pylist([]), source)
    output = tmp_path / "out"
    with pytest.raises(ValueError, match="No samples"):
        prepare_sft(source, output)
    assert not (output / "sft.jsonl").exists()
