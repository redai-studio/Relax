# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Diffusion data-prep scripts: record normalization, deterministic split,
gate."""

from __future__ import annotations

import importlib.util
import pathlib

import pytest


_ROOT = pathlib.Path(__file__).resolve().parents[2] / "examples" / "diffusion"


def _load(name):
    spec = importlib.util.spec_from_file_location(f"_diff_{name}", _ROOT / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


prepare = _load("prepare_data")
curate = _load("curate_data")
inspect = _load("inspect_data")


def test_normalize_record_rejects_retired_tasks():
    rec = prepare.normalize_record("a cat", "t2i", "s0")
    assert rec["prompt"] == "a cat" and rec["metadata"]["task"] == "t2i"
    assert prepare.TASKS == ("t2i",)
    # i2i / t2v / i2v / v2v / t2av are not supported on this branch.
    for retired in ("i2i", "t2v", "i2v", "v2v", "t2av"):
        with pytest.raises(ValueError):
            prepare.normalize_record("edit", retired, "s1", images=["/a.png"])


def test_curate_split_is_deterministic():
    recs = [{"prompt": f"p{i}", "metadata": {"task": "t2i", "sample_id": f"s{i}"}} for i in range(200)]
    train1, eval1 = curate.curate(recs, eval_fraction=0.1, check_media=False)
    train2, eval2 = curate.curate(recs, eval_fraction=0.1, check_media=False)
    assert [r["metadata"]["sample_id"] for r in eval1] == [r["metadata"]["sample_id"] for r in eval2]
    # no overlap between splits
    tr_ids = {r["metadata"]["sample_id"] for r in train1}
    ev_ids = {r["metadata"]["sample_id"] for r in eval1}
    assert not (tr_ids & ev_ids)
    assert len(eval1) > 0  # ~10% landed in eval


def test_curate_dedup():
    recs = [{"prompt": "same", "metadata": {"task": "t2i", "sample_id": f"s{i}"}} for i in range(5)]
    train, evalset = curate.curate(recs, eval_fraction=0.0, check_media=False)
    assert len(train) + len(evalset) == 1  # all identical prompts deduped to one


def test_curate_min_prompt_words_and_eval_size():
    """The alignment preset: >=6-word prompts, a fixed-size held-out split."""
    recs = [
        {"prompt": " ".join(f"w{i}_{j}" for j in range(i % 10)), "metadata": {"task": "t2i", "sample_id": f"s{i}"}}
        for i in range(300)
    ]
    train, evalset = curate.curate(recs, 0.0, check_media=False, min_prompt_words=6, eval_size=20)
    assert len(evalset) == 20
    assert all(curate.prompt_word_count(r) >= 6 for r in train + evalset)
    # Deterministic and disjoint.
    train2, eval2 = curate.curate(recs, 0.0, check_media=False, min_prompt_words=6, eval_size=20)
    assert [r["metadata"]["sample_id"] for r in eval2] == [r["metadata"]["sample_id"] for r in evalset]
    assert not ({r["metadata"]["sample_id"] for r in train} & {r["metadata"]["sample_id"] for r in evalset})


def test_inspect_check_record():
    good = {"prompt": "a snowy street at dusk", "metadata": {"task": "t2i", "sample_id": "s"}}
    assert inspect.check_record(good) == []
    # A retired task is rejected by the gate.
    retired = {"prompt": "add snow", "metadata": {"task": "i2i", "sample_id": "s"}}
    assert any("metadata.task" in e for e in inspect.check_record(retired))
    # Placeholder without the matching media is still an error.
    dangling = {"prompt": "<image> add snow", "metadata": {"task": "t2i", "sample_id": "s"}}
    assert any("images" in e for e in inspect.check_record(dangling))


def test_inspect_split_leakage():
    assert inspect.find_split_leakage(["a", "b"], ["c"]) == set()
    assert inspect.find_split_leakage(["a", "b"], ["b", "c"]) == {"b"}


def test_inspect_split_leakage_ignores_missing_sample_ids(tmp_path):
    train = tmp_path / "train.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    train.write_text('{"prompt":"p","metadata":{"task":"t2i"}}\n', encoding="utf-8")
    eval_path.write_text('{"prompt":"q","metadata":{"task":"t2i","sample_id":"e0"}}\n', encoding="utf-8")

    assert inspect._inspect_file(str(train), decode_media=False) == (1, 0)
    assert inspect._inspect_file(str(eval_path), decode_media=False) == (1, 0)
    assert inspect._sample_ids(str(train)) == []
    assert inspect._sample_ids(str(eval_path)) == ["e0"]
    assert inspect.find_split_leakage(inspect._sample_ids(str(train)), inspect._sample_ids(str(eval_path))) == set()
