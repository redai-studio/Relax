# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from types import SimpleNamespace

from relax.utils.rotate_ckpt import rotate_ckpt


def test_rotate_ckpt_committed_only_ignores_failed_newer_directory(tmp_path):
    committed = tmp_path / "iter_0000001"
    committed.mkdir()
    (committed / "COMMITTED").write_text("1", encoding="utf-8")
    failed = tmp_path / "iter_0000002"
    failed.mkdir()

    args = SimpleNamespace(max_actor_ckpt_to_keep=1, rotate_ckpt=False, save=str(tmp_path))
    rotate_ckpt(args, global_step=2, save_dir=str(tmp_path), committed_only=True)

    assert committed.is_dir()
    assert failed.is_dir()


def test_rotate_ckpt_committed_only_never_deletes_latest_valid_checkpoint(tmp_path):
    committed = tmp_path / "iter_0000001"
    committed.mkdir()
    (committed / "COMMITTED").write_text("1", encoding="utf-8")

    args = SimpleNamespace(max_actor_ckpt_to_keep=0, rotate_ckpt=False, save=str(tmp_path))
    rotate_ckpt(args, global_step=1, save_dir=str(tmp_path), committed_only=True)

    assert committed.is_dir()
