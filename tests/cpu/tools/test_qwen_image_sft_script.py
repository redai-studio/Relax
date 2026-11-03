# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[3] / "scripts/training/diffusion/run-qwen-image-sft.sh"


class TestQwenImageSFTScript(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def preview(self, extra=(), **overrides):
        env = {
            **os.environ,
            "DRY_RUN": "1",
            "MODEL_PATH": "/models/with spaces/Qwen-Image",
            "DATA_ROOT": "/data/coco",
            "METADATA_PATH": "/data/caption subset.json",
            "DATASET_SIZE": "24",
            "SAVE_DIR": str(self.root / "save"),
            "RUN_DIR": str(self.root / "logs"),
            "RAY_DASHBOARD_ADDRESS": "http://ray-head:8265",
            "RUNTIME_ENV_JSON": '{"env_vars":{"EXISTING":"yes","PATH":"/worker/bin"}}',
            "DRIVER_PYTHONPATH": "/imports/with spaces",
            "USE_CLEARML": "0",
            "SAVE_CHECKPOINTS": "0",
            "IMAGE_PREFETCH_DEPTH": "0",
            "GPUS": "8",
            "GPUS_PER_NODE": "8",
            "GLOBAL_BATCH_SIZE": "8",
            "MICRO_BATCH_SIZE": "1",
            "HEIGHT": "512",
            "WIDTH": "512",
            "MAX_SEQUENCE_LENGTH": "512",
            "TEXT_ENCODER_PLACEMENT": "sharded",
            "CAPTION_DROPOUT": "0",
            "RANDOM_CROP": "0",
            "DETERMINISTIC_ATTENTION": "0",
            "GRADIENT_CHECKPOINTING": "1",
        }
        env.update(overrides)
        return subprocess.run(["bash", str(SCRIPT), *extra], env=env, capture_output=True, text=True, timeout=10)

    def test_qwen_image_preview_preserves_paths_runtime_and_has_no_side_effects(self):
        result = self.preview()
        self.assertEqual(result.returncode, 0, result.stderr)
        args = json.loads(result.stdout)
        self.assertEqual(args[args.index("--generation-task") + 1], "t2i")
        self.assertEqual(
            args[args.index("--model-adapter-path") + 1], "relax.models.qwen_image.sft_adapter.QwenImageSFTAdapter"
        )
        self.assertEqual(args[args.index("--model-path") + 1], "/models/with spaces/Qwen-Image")
        self.assertEqual(args[args.index("--prompt-data") + 1], "/data/caption subset.json")
        runtime = json.loads(args[args.index("--runtime-env-json") + 1])["env_vars"]
        self.assertEqual(runtime, {"EXISTING": "yes", "PATH": "/worker/bin"})
        self.assertIn("--no-wait", args)
        self.assertNotIn("--save", args)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_qwen_image_preview_preserves_effective_batch_and_resume(self):
        checkpoint = str(self.root / "checkpoint with spaces")
        result = self.preview(GLOBAL_BATCH_SIZE="32", LOAD_DIR=checkpoint, SAVE_CHECKPOINTS="1", LR_WARMUP_ITERS="0")
        self.assertEqual(result.returncode, 0, result.stderr)
        args = json.loads(result.stdout)
        self.assertEqual(args[args.index("--global-batch-size") + 1], "32")
        self.assertEqual(args[args.index("--rollout-batch-size") + 1], "32")
        self.assertEqual(args[args.index("--micro-batch-size") + 1], "1")
        self.assertEqual(args[args.index("--load") + 1], checkpoint)
        self.assertEqual(args[args.index("--lr-warmup-iters") + 1], "0")
        self.assertIn("--save", args)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_qwen_image_preview_passes_placement_and_random_preprocessing(self):
        result = self.preview(TEXT_ENCODER_PLACEMENT="replicated", RANDOM_CROP="1", CAPTION_DROPOUT="0.2")
        self.assertEqual(result.returncode, 0, result.stderr)
        args = json.loads(result.stdout)
        config = json.loads(args[args.index("--t2i-sft-config") + 1])
        self.assertEqual(config["text_encoder_placement"], "replicated")
        self.assertTrue(config["random_crop"])
        self.assertEqual(config["caption_dropout"], 0.2)

    def test_qwen_image_preview_rejects_invalid_configuration_before_submission(self):
        for overrides in (
            {"GLOBAL_BATCH_SIZE": "12"},
            {"GPUS_PER_NODE": "3"},
            {"MICRO_BATCH_SIZE": "2", "GLOBAL_BATCH_SIZE": "32"},
            {"CAPTION_DROPOUT": "nan"},
            {"CAPTION_DROPOUT": "1.1"},
            {"HEIGHT": "511"},
            {"MAX_SEQUENCE_LENGTH": "513"},
            {"TEXT_ENCODER_PLACEMENT": "offload"},
            {"IMAGE_PREFETCH_DEPTH": "1"},
            {"RANDOM_CROP": "true"},
        ):
            with self.subTest(overrides=overrides):
                result = self.preview(**overrides)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(list(self.root.iterdir()), [])

    def test_qwen_image_preview_rejects_hidden_batch_override(self):
        result = self.preview(extra=("--global-batch-size=4",))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("GLOBAL_BATCH_SIZE/MICRO_BATCH_SIZE", result.stderr)


if __name__ == "__main__":
    unittest.main()
