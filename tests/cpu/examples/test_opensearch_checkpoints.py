# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from argparse import Namespace

import pytest

from examples.opensearch_vl.tools.checkpoint_retention import keep_latest_hf


def test_completed_hf_export_prunes_older_exports_only(tmp_path):
    old = tmp_path / "final-hf-199"
    current = tmp_path / "final-hf-399"
    other = tmp_path / "model-cache"
    for directory in (old, current, other):
        directory.mkdir()
        (directory / "config.json").write_text("{}")
    native = tmp_path / "relax" / "iter_0000399"
    native.mkdir(parents=True)
    keep_latest_hf(Namespace(), str(current), 399)
    assert not old.exists()
    assert current.is_dir() and other.is_dir() and native.is_dir()


def test_incomplete_hf_export_preserves_previous_recovery_point(tmp_path):
    old = tmp_path / "final-hf-199"
    old.mkdir()
    incomplete = tmp_path / "final-hf-399"
    incomplete.mkdir()
    with pytest.raises(ValueError, match="Invalid completed HF export"):
        keep_latest_hf(Namespace(), str(incomplete), 399)
    assert old.is_dir()
