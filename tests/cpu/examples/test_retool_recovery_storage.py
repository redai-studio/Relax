# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Fresh synthetic checkpoint metadata tests; never execute historical
trajectories."""

import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from examples.retool import checkpoint, recovery


class TestBoundedCheckpointSelection(unittest.TestCase):
    def test_claim_registry_uses_explicit_local_runtime_not_experiment_root(self):
        with tempfile.TemporaryDirectory() as folder:
            runtime = Path(folder) / "local-runtime"
            runtime.mkdir()
            shared = Path(folder) / "not-created-shared-root"
            spec = importlib.util.spec_from_file_location(
                "fresh_resource_fixture",
                recovery.__file__.replace(
                    "recovery.py",
                    "resources.py",
                ),
            )
            module = importlib.util.module_from_spec(spec)
            with mock.patch.dict(os.environ, {"ROOT": str(shared), "RT": str(runtime)}):
                spec.loader.exec_module(module)
            self.assertEqual(module.REG, runtime / "claims")
            self.assertTrue(module.REG.is_dir())
            self.assertFalse(shared.exists())

    def state(self, root, step):
        directory = root / "dataset"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"retool_{step}.json").write_text(
            json.dumps(
                dict(
                    cursor=step + 1,
                    batch=64,
                    K=16,
                    seed=20261031,
                    rng_state_version=1,
                    rng_state=[17],
                )
            )
        )

    def test_exact_debug_cp2_without_directory_enumeration(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "checkpoints/rl"
            self.state(root, 2)
            (root / "latest_checkpointed_iteration.txt").write_text("2")
            with mock.patch.object(Path, "iterdir", side_effect=AssertionError("directory enumeration")):
                with mock.patch.object(Path, "glob", side_effect=AssertionError("directory enumeration")):
                    selected = recovery.select_checkpoint(
                        [root],
                        "rl",
                        200,
                        Path(folder) / "views",
                        validator=lambda view, sft: 2,
                    )
            self.assertEqual((selected["step"], selected["completed"]), (2, 3))

    def test_invalid_latest_tries_only_fixed_formal_boundary(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "checkpoints/rl"
            self.state(root, 24)
            (root / "latest_checkpointed_iteration.txt").write_text("49")
            seen = []

            def validate(view, sft):
                step = int((view / "latest_checkpointed_iteration.txt").read_text())
                seen.append(step)
                if step == 49:
                    raise ValueError("fresh fixture: incomplete latest checkpoint")
                return step

            selected = recovery.select_checkpoint([root], "rl", 200, Path(folder) / "views", validate)
            self.assertEqual(seen, [49, 24])
            self.assertEqual(selected["step"], 24)
            self.assertEqual(len(selected["rejected"]), 1)

    def test_invalid_cp2_fails_closed_without_inventing_earlier_saves(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "checkpoints/rl"
            root.mkdir(parents=True)
            (root / "latest_checkpointed_iteration.txt").write_text("2")
            validate = mock.Mock(side_effect=ValueError("fresh fixture: missing metadata"))
            with self.assertRaisesRegex(AssertionError, "none accepted"):
                recovery.select_checkpoint([root], "rl", 200, Path(folder) / "views", validate)
            self.assertEqual(validate.call_count, 1)

    def test_existing_empty_or_initialization_root_requires_tracker(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with mock.patch.object(Path, "iterdir", side_effect=AssertionError("directory enumeration")):
                with self.assertRaisesRegex(ValueError, "committed"):
                    checkpoint.validate_checkpoint(root)
                (root / "transformer_config.json").write_text("{}")
                with self.assertRaisesRegex(ValueError, "committed"):
                    checkpoint.validate_checkpoint(root)
                self.assertIsNone(checkpoint.validate_checkpoint(root / "not-started"))

    def test_missing_tracker_needs_failed_wait_and_nonempty_zero_update_log(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "checkpoints/rl"
            root.mkdir(parents=True)
            run = Path(folder) / "gpu_rl"
            run.mkdir()
            with self.assertRaises(FileNotFoundError):
                recovery.require_zero_update_failure(root)
            wait = run / "training_actual_wait.json"
            log = run / "train.log"
            wait.write_text('{"returncode":1,"actual_parent_wait":true}')
            with self.assertRaises(FileNotFoundError):
                recovery.require_zero_update_failure(root)
            log.write_text("")
            with self.assertRaisesRegex(AssertionError, "Empty native log"):
                recovery.require_zero_update_failure(root)
            log.write_text("fresh initialization fixture\nSTEP_END step=1\n")
            with self.assertRaisesRegex(AssertionError, "Uncheckpointed"):
                recovery.require_zero_update_failure(root)
            # Removing the checkpoint directory must not hide actual updates.
            root.rmdir()
            with self.assertRaisesRegex(AssertionError, "Uncheckpointed"):
                recovery.require_zero_update_failure(root)
            root.mkdir()
            log.write_text("fresh initialization failed fixture\n")
            wait.write_text('{"returncode":0,"actual_parent_wait":true}')
            with self.assertRaisesRegex(AssertionError, "No failed native exit"):
                recovery.require_zero_update_failure(root)
            for record in ('{"returncode":1}', '{"returncode":1,"actual_parent_wait":false}'):
                wait.write_text(record)
                with self.assertRaisesRegex(AssertionError, "No failed native exit"):
                    recovery.require_zero_update_failure(root)
            wait.write_text('{"returncode":1,"actual_parent_wait":true}')
            recovery.require_zero_update_failure(root)

    def test_dcp_extents_stat_each_unique_storage_file_once(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            directory = root / "iter_0000002"
            directory.mkdir()
            storage = directory / "new.distcp"
            storage.write_bytes(b"x" * 32)
            metadata = SimpleNamespace(
                state_dict_metadata={
                    name: None
                    for name in (
                        "embedding.word_embeddings.weight",
                        "decoder.layers.0",
                        "output_layer.weight",
                        "optimizer",
                        "rng_state",
                    )
                },
                storage_data={
                    index: SimpleNamespace(relative_path="new.distcp", offset=offset, length=8)
                    for index, offset in enumerate((0, 8, 24))
                },
            )
            reader = SimpleNamespace(read_metadata=lambda: metadata)
            module = SimpleNamespace(FileSystemReader=lambda path: reader)
            original_stat = Path.stat
            calls = []

            def stat(path, *args, **kwargs):
                if path == storage:
                    calls.append(path)
                return original_stat(path, *args, **kwargs)

            with mock.patch.dict(sys.modules, {"torch.distributed.checkpoint": module}):
                with mock.patch.object(checkpoint, "validate_checkpoint", return_value=2):
                    with mock.patch.object(Path, "stat", stat):
                        self.assertEqual(recovery.validate_native(root, False), 2)
            self.assertEqual(calls, [storage])
            storage.write_bytes(b"x" * 31)
            with mock.patch.dict(sys.modules, {"torch.distributed.checkpoint": module}):
                with mock.patch.object(checkpoint, "validate_checkpoint", return_value=2):
                    with self.assertRaises(AssertionError):
                        recovery.validate_native(root, False)
            storage.unlink()
            with mock.patch.dict(sys.modules, {"torch.distributed.checkpoint": module}):
                with mock.patch.object(checkpoint, "validate_checkpoint", return_value=2):
                    with self.assertRaises(FileNotFoundError):
                        recovery.validate_native(root, False)
            outside = root / "external.distcp"
            outside.write_bytes(b"x" * 32)
            storage.symlink_to(outside)
            with mock.patch.dict(sys.modules, {"torch.distributed.checkpoint": module}):
                with mock.patch.object(checkpoint, "validate_checkpoint", return_value=2):
                    with self.assertRaises(AssertionError):
                        recovery.validate_native(root, False)


if __name__ == "__main__":
    unittest.main()
