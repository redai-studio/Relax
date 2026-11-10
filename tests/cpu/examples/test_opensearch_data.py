# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""OpenSearch data retains long trajectories and isolates related evaluation
rows."""

import importlib.util
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest
from PIL import Image


TOOLS = Path(__file__).resolve().parents[3] / "examples" / "opensearch_vl" / "tools"


@pytest.fixture
def prepare(monkeypatch):
    monkeypatch.syspath_prepend(str(TOOLS))
    spec = importlib.util.spec_from_file_location("opensearch_prepare", TOOLS / "prepare_data.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_split_keeps_all_rows_and_shared_image_question_groups(prepare):
    rows = [
        {
            "id": str(index),
            "messages": [{}, {"content": f"question {index}"}],
            "metadata": {"image_hashes": [f"image {index}"], "expanded_tokens": 100000},
        }
        for index in range(30)
    ]
    # Two image links and a question link form one transitive component.
    rows[1]["metadata"]["image_hashes"] = ["image 0", "image 2"]
    rows[3]["messages"][1]["content"] = "question 2"
    splits = prepare.select_splits(rows, [0, 8, 8], 42)
    assert [len(splits[name]) for name in ("train", "val", "test")] == [14, 8, 8]
    assert sorted(row["id"] for split in splits.values() for row in split) == sorted(row["id"] for row in rows)
    for field in ("images", "questions"):
        keys = []
        for split in splits.values():
            keys.append(
                {
                    key
                    for row in split
                    for key in (
                        row["metadata"]["image_hashes"] if field == "images" else [row["messages"][1]["content"]]
                    )
                }
            )
        assert not keys[0] & keys[1] and not keys[0] & keys[2] and not keys[1] & keys[2]
    assert splits == prepare.select_splits(rows, [0, 8, 8], 42)


@pytest.mark.parametrize("archived", [False, True])
def test_prepare_cli_retains_long_rows_without_a_model(tmp_path, prepare, archived):
    source = tmp_path / "source"
    domain = source / "fvqa"
    (domain / "images").mkdir(parents=True)
    rows = []
    for index in range(24):
        Image.new("RGB", (64, 32), (index, 0, 0)).save(domain / "images" / f"{index}.png")
        rows.append(
            {
                "system": "Use tools.",
                "conversations": [
                    {"from": "human", "value": f"<image>question {index}"},
                    {
                        "from": "gpt",
                        "value": '<think>reason</think><tool_call>{"name":"search","arguments":{}}</tool_call>',
                    },
                    {"from": "observation", "value": "long observation " * 40000 if index == 0 else "result"},
                    {"from": "gpt", "value": "<response>answer</response>"},
                ],
                "images": [f"images/{index}.png"],
                "tools": json.dumps(json.dumps([{"type": "function", "function": {"name": "search"}}])),
            }
        )
    (domain / prepare.SOURCE_FILES["fvqa"]).write_text(json.dumps(rows))
    if archived:
        with zipfile.ZipFile(domain / "images.zip", "w") as bundle:
            for path in (domain / "images").iterdir():
                bundle.write(path, path.relative_to(domain))
                path.unlink()
        (domain / "images").rmdir()
    (source / "source-revision.json").write_text(json.dumps({"revision": prepare.DATA_REVISION}))
    output = tmp_path / "prepared"
    subprocess.run(
        [
            sys.executable,
            str(TOOLS / "prepare_data.py"),
            "--source",
            str(source),
            "--output",
            str(output),
            "--domains",
            "fvqa",
            "--val-size",
            "8",
            "--test-size",
            "8",
            "--workers",
            "2",
        ],
        check=True,
        timeout=60,
    )
    prepared = [
        json.loads(line)
        for name in ("train", "val", "test")
        for line in (output / f"{name}.jsonl").read_text().splitlines()
    ]
    assert len(prepared) == 24
    long = next(row for row in prepared if row["messages"][1]["content"] == "<image>question 0")
    assert long["messages"][3]["content"] == "long observation " * 40000
    assert all(Path(path).is_file() for row in prepared for path in row["images"])
    assert [m["role"] for m in long["messages"]] == ["system", "user", "assistant", "tool", "assistant"]
    assert isinstance(long["tools"], list)
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["max_length"] is None and manifest["rejected_count"] == 0
    assert len(list((output / "images").glob("*.png"))) == 24
    assert (domain / "images").exists() is not archived


def test_zip_and_file_images_have_identical_normalized_pixels(tmp_path, prepare):
    domain = tmp_path / "fvqa"
    (domain / "images").mkdir(parents=True)
    path = domain / "images" / "sample.png"
    Image.new("RGB", (1024, 768), (17, 33, 71)).save(path)
    extracted, extracted_hash = prepare.prepare_image(path, tmp_path / "extracted", domain)
    with zipfile.ZipFile(domain / "images.zip", "w") as bundle:
        bundle.write(path, "images/sample.png")
        bundle.writestr("../must-not-extract", b"unused")
    path.unlink()
    path.parent.rmdir()
    prepare.image_bundle.cache_clear()
    archived, archived_hash = prepare.prepare_image(path, tmp_path / "archived", domain)
    assert archived_hash == extracted_hash
    with Image.open(extracted) as expected, Image.open(archived) as actual:
        assert actual.size == expected.size and actual.tobytes() == expected.tobytes()
    assert not (tmp_path / "must-not-extract").exists()
