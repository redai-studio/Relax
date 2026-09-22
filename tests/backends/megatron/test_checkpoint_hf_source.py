# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from types import SimpleNamespace

import pytest


pytest.importorskip("megatron.training.checkpointing", exc_type=ImportError)

from relax.backends.megatron import checkpoint
from relax.utils.model_source import ModelSource


def test_select_hf_load_source_remaps_replaced_hf_path():
    args = SimpleNamespace(
        model_source=ModelSource("s3://bucket/model/"),
        _model_source_original_hf_checkpoint="/models/original",
        hf_checkpoint="/dev/shm/model",
        ref_load="/dev/shm/model",
    )

    assert checkpoint._select_hf_load_source(args, "/models/original/") == "/dev/shm/model"


def test_select_hf_load_source_preserves_distinct_int4_reference():
    args = SimpleNamespace(
        model_source=ModelSource("s3://bucket/packed-model/"),
        _model_source_original_hf_checkpoint="/models/packed-int4",
        hf_checkpoint="/dev/shm/packed-model",
        ref_load="/models/bf16-reference",
    )

    assert checkpoint._select_hf_load_source(args, args.ref_load) == args.ref_load


def test_load_checkpoint_preserves_valid_megatron_resume(monkeypatch, tmp_path):
    resume_path = tmp_path / "model"
    resume_path.mkdir()
    (resume_path / "latest_checkpointed_iteration.txt").write_text("1")
    args = SimpleNamespace(load=str(resume_path))
    expected = (1, 2)

    monkeypatch.setattr(checkpoint, "get_args", lambda: args)
    monkeypatch.setattr(checkpoint, "_alias_renamed_transfer_queue_enum", lambda: None)
    monkeypatch.setattr(checkpoint, "_load_checkpoint_megatron", lambda **_kwargs: expected)

    def fail_hf_load(**_kwargs):
        raise AssertionError("valid Megatron resume must not reach the HF loader")

    monkeypatch.setattr(checkpoint, "_load_checkpoint_hf", fail_hf_load)

    assert checkpoint.load_checkpoint(None, None, None, {}, False) == expected
