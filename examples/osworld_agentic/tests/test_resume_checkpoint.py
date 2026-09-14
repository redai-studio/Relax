# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""CPU regression for the shared runner's checkpoint selection."""

from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path


_MODEL_NAME = "Qwen3-VL-8B-osworld-outcome"
_TRACKER = "latest_checkpointed_iteration.txt"
_ARGV_MARKER = b"\n__OSWORLD_TEST_CKPT_ARGS__\0"


class TestResumeCheckpoint(unittest.TestCase):
    def setUp(self) -> None:
        configured = os.environ.get("OSWORLD_TEST_EXAMPLE_DIR")
        example_dir = Path(configured) if configured else Path(__file__).resolve().parents[1]
        launcher = example_dir / "run_qwen3vl_8B_osworld_outcome.sh"
        missing = [str(path) for path in (launcher,) if not path.is_file()]
        if missing:
            if configured:
                self.fail(f"OSWORLD_TEST_EXAMPLE_DIR is missing real launcher sources: {missing}")
            self.skipTest("Real launchers are absent; set OSWORLD_TEST_EXAMPLE_DIR to run this regression")
        self.checkpoint_source = self._section(
            launcher,
            "# Model and resume checkpoint arguments.",
            "# NCCL collective timeout.",
        )
        temporary = tempfile.TemporaryDirectory(prefix="osworld-resume-test-")
        self.addCleanup(temporary.cleanup)
        self.folder = Path(temporary.name)
        self.model_dir = self.folder / "hf-model"
        self.hf_checkpoint = self.model_dir / "Qwen3-VL-8B-Thinking"
        self.save_dir = self.folder / "new-run"
        self.target = self.save_dir / _MODEL_NAME
        self.resume_dir = self.folder / "source checkpoint with spaces" / _MODEL_NAME
        self.hf_checkpoint.mkdir(parents=True)
        self.save_dir.mkdir()
        self.env = dict(
            os.environ,
            MODEL_DIR=str(self.model_dir),
            SAVE_DIR=str(self.save_dir),
            OSWORLD_RESUME_DIR="",
            RESUME_FROM_SAVE="0",
        )

    def _section(self, path: Path, start: str, end: str) -> str:
        source = path.read_text()
        self.assertEqual(source.count(start), 1, f"Expected one section start in {path}")
        beginning = source.index(start)
        ending = source.index(end, beginning)
        return source[beginning:ending]

    @staticmethod
    def _checkpoint(root: Path) -> None:
        root.mkdir(parents=True)
        (root / _TRACKER).write_text("0\n")

    def _run(self, source: str, **overrides: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["bash", "-eu", "-c", source],
            env=dict(self.env, **overrides),
            cwd=self.folder,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=15,
            check=False,
        )

    def _argv(self, **overrides: str) -> list[str]:
        output = self._run(
            self.checkpoint_source
            + r"""
printf '\n__OSWORLD_TEST_CKPT_ARGS__\0'
printf '%s\0' "${CKPT_ARGS[@]}"
""",
            **overrides,
        )
        self.assertEqual(output.returncode, 0, output.stderr.decode(errors="replace"))
        self.assertEqual(output.stdout.count(_ARGV_MARKER), 1, output.stdout)
        encoded = output.stdout.split(_ARGV_MARKER, 1)[1]
        self.assertTrue(encoded.endswith(b"\0"), "Expected NUL-delimited complete argv")
        return [value.decode() for value in encoded[:-1].split(b"\0")]

    def _assert_option(self, argv: list[str], name: str, value: Path) -> None:
        self.assertEqual(argv.count(name), 1, f"Duplicate or missing {name}: {argv}")
        self.assertEqual(argv[argv.index(name) + 1], str(value))

    def test_explicit_and_legacy_resume_cannot_be_combined(self) -> None:
        self._checkpoint(self.resume_dir)

        output = self._run(self.checkpoint_source, OSWORLD_RESUME_DIR=str(self.resume_dir), RESUME_FROM_SAVE="1")
        self.assertNotEqual(output.returncode, 0, "Conflicting resume sources unexpectedly passed")
        self.assertTrue(output.stdout or output.stderr, "Rejected configuration should explain its failure")

    def test_legacy_launcher_resume_keeps_its_existing_argument_contract(self) -> None:
        self._checkpoint(self.target)

        # The legacy lower-level launcher still supports same-directory resume.
        argv = self._argv(RESUME_FROM_SAVE="1")

        self._assert_option(argv, "--load", self.target)
        self._assert_option(argv, "--ref-load", self.hf_checkpoint)
        self._assert_option(argv, "--save", self.target)


if __name__ == "__main__":
    unittest.main()
