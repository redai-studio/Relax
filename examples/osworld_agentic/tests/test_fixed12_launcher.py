# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Exercise the real fixed12 shell chain without starting Ray or a GPU job."""

import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


_EXAMPLE = Path(__file__).resolve().parents[1]
_ROOT = _EXAMPLE.parents[1]


class TestFixed12Launcher(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory(prefix="osworld-recipe-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.example = self.root / "examples/osworld_agentic"
        self.example.mkdir(parents=True)
        for name in ("launch_fixed12_minibatch3.sh", "run_qwen3vl_8B_osworld_outcome.sh"):
            shutil.copyfile(_EXAMPLE / name, self.example / name)
        model_config = self.root / "scripts/models"
        model_config.mkdir(parents=True)
        shutil.copyfile(_ROOT / "scripts/models/qwen3-vl-8B.sh", model_config / "qwen3-vl-8B.sh")
        self.data = self.root / "fixed12.parquet"
        self.data.write_bytes(b"Existing fixture; the shell does not parse task data.")
        self.capture = self.root / "ray-argv.json"
        command_dir = self.root / "bin"
        command_dir.mkdir()
        ray = command_dir / "ray"
        ray.write_text(
            "#!/usr/bin/env python3\nimport json, os, sys\n"
            "with open(os.environ['OSWORLD_TEST_ARGV'], 'w') as target:\n"
            "    json.dump(sys.argv[1:], target)\n"
        )
        ray.chmod(0o755)
        self.env = dict(
            os.environ,
            PATH=f"{command_dir}:{os.environ['PATH']}",
            MODEL_DIR=str(self.root / "model with spaces"),
            DATA_DIR=str(self.root / "data"),
            SAVE_DIR=str(self.root / "new run"),
            OSWORLD_REPO=str(self.root / "OSWorld"),
            MEGATRON=str(self.root / "Megatron-LM"),
            OSWORLD_TRAIN_DATA=str(self.data),
            OSWORLD_TEST_ARGV=str(self.capture),
            OSWORLD_RESUME_DIR="",
            RESUME_FROM_SAVE="0",
            RAY_JOB_ADDRESS="http://test-ray:8265",
        )

    def run_launcher(self, **overrides: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["bash", str(self.example / "launch_fixed12_minibatch3.sh")],
            env=dict(self.env, **overrides),
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )

    def test_real_argv_preserves_native_batches_and_forwards_worker_environment(self) -> None:
        # A machine env file is loaded once, before the recipe fixes its batch plan.
        (self.example / "env.sh").write_text("export ROLLOUT_BATCH_SIZE=74\nexport GLOBAL_BATCH_SIZE=592\n")
        result = self.run_launcher()
        self.assertEqual(result.returncode, 0, result.stderr)
        argv = json.loads(self.capture.read_text())
        for option, value in (
            ("--rollout-batch-size", "12"),
            ("--n-samples-per-prompt", "8"),
            ("--global-batch-size", "32"),
            ("--num-steps-per-rollout", "3"),
            ("--lr", "1e-6"),
            ("--lr-decay-iters", "135"),
            ("--hf-checkpoint", f"{self.env['MODEL_DIR']}/Qwen3-VL-8B-Thinking"),
            ("--save", f"{self.env['SAVE_DIR']}/Qwen3-VL-8B-osworld-outcome"),
        ):
            with self.subTest(option=option):
                self.assertEqual(argv[argv.index(option) + 1], value)
        self.assertIn("examples.osworld_agentic.train_fixed12_minibatch3", argv)
        self.assertIn("--skip-eval-before-train", argv)
        self.assertIn("--address=http://test-ray:8265", argv)
        for absent in ("--eval-interval", "--eval-prompt-data", "--load", "--true-on-policy-mode"):
            self.assertNotIn(absent, argv)
        runtime = json.loads(argv[argv.index("--runtime-env-json") + 1])["env_vars"]
        for name, value in (
            ("OSWORLD_PER_TRAJ_NORM", "0"),
            ("RELAX_MM_PIXEL_DTYPE", "bf16"),
            ("RELAX_MM_IMAGE_POOL", "1"),
            ("RELAX_MM_RELEASE_AFTER_TRANSFER", "1"),
            ("RELAX_TRANSFER_BATCH_GROUPS", "9"),
            ("RELAX_EVAL_ON_EPOCH_BOUNDARY", "0"),
        ):
            with self.subTest(variable=name):
                self.assertEqual(runtime[name], value)
                self.assertIn(name, runtime["RELAX_PROPAGATE_ENV_VARS"].split(","))

    def test_base_recipe_rejects_resume_and_existing_destination_before_ray(self) -> None:
        for overrides in ({"OSWORLD_RESUME_DIR": "/checkpoint"}, {"RESUME_FROM_SAVE": "1"}):
            with self.subTest(overrides=overrides):
                self.assertNotEqual(self.run_launcher(**overrides).returncode, 0)
        target = Path(self.env["SAVE_DIR"]) / "Qwen3-VL-8B-osworld-outcome"
        target.mkdir(parents=True)
        (target / "latest_checkpointed_iteration.txt").write_text("9\n")
        self.assertNotEqual(self.run_launcher().returncode, 0)
        self.assertFalse(self.capture.exists())

    def test_missing_explicit_task_data_fails_before_ray(self) -> None:
        result = self.run_launcher(OSWORLD_TRAIN_DATA=str(self.root / "missing.parquet"))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("OSWORLD_TRAIN_DATA", result.stderr)
        self.assertFalse(self.capture.exists())

    def test_fixed12_manifest_has_twelve_unique_tasks(self) -> None:
        ids = (_EXAMPLE / "scripts/manifests/train_fixed12.txt").read_text().splitlines()
        self.assertEqual(len(ids), 12)
        self.assertEqual(len(set(ids)), 12)


if __name__ == "__main__":
    unittest.main()
