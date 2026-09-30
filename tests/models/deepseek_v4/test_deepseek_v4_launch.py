# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Check SFT/RL QAT launch wiring with fake commands, without Ray/ML imports or
jobs."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
PREFIX = "RELAX_DSV4_FP4_"
PROVIDER = "relax.models.deepseek_v4.provider.model_provider"
SCRIPT = ROOT / "scripts/training/sft/run-deepseek-v4-flash-sft-0731-128xgpu.sh"
RL_SCRIPT = ROOT / "scripts/training/text/run-deepseek-v4-flash-0731-128xgpu.sh"


class DeepSeekV4LaunchTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory(prefix="relax-dsv4-launch-")
        self.addCleanup(temporary.cleanup)
        self.folder = Path(temporary.name)
        bin_dir = self.folder / "bin"
        bin_dir.mkdir()
        programs = {
            "bash": "#!/bin/bash\nexit 0\n",  # Stub ClearML; launch() uses /bin/bash directly.
            "mkdir": "#!/bin/bash\nexit 0\n",
            "tee": "#!/bin/bash\ncat\n",
            "stdbuf": '#!/bin/bash\nshift 2\nexec "$@"\n',
            "ray": f'#!{sys.executable}\nimport json,os,sys\nopen(os.environ["ARGV_CAPTURE"],"w").write(json.dumps(sys.argv[1:]))\n',
        }
        for name, source in programs.items():
            path = bin_dir / name
            path.write_text(source)
            path.chmod(0o755)
        self.capture = self.folder / "argv.json"
        excluded = {
            "RELAX_EXTRA_ENV_ALLOWLIST",
            "RELAX_PROPAGATE_ENV_VARS",
            "WORKING_DIR",
            "HF_CKPT",
            "PROMPT_DATA",
        }
        self.environment = {
            k: v for k, v in os.environ.items() if not k.startswith("RELAX_DSV4_") and k not in excluded
        }
        self.environment.update(
            PATH=str(bin_dir) + os.pathsep + os.environ["PATH"],
            RELAX_ENTRYPOINT_MODE="argv-check",
            MODEL_CONFIG_DIR=str(ROOT / "scripts/models"),
            RAY_ADDRESS="http://argv.invalid",
            ARGV_CAPTURE=str(self.capture),
            MODEL_DIR=str(self.folder / "models"),
            DATA_DIR=str(self.folder / "data"),
            LOG_DIR=str(self.folder / "logs"),
            SAVE_DIR=str(self.folder / "save"),
            PYTHONDONTWRITEBYTECODE="1",
            NO_VCS_VERSION="1",
        )

    def launch(
        self,
        fp4: bool,
        flags: dict[str, str] | None = None,
        runtime: dict[str, Any] | None = None,
        *,
        success: bool = True,
        script: Path = SCRIPT,
    ) -> tuple[list[str], dict[str, str]]:
        if fp4:
            script = script.with_name(script.stem + "-fp4.sh")
        self.capture.unlink(missing_ok=True)
        environment = self.environment | {"RUNTIME_ENV_JSON": json.dumps(runtime or {"env_vars": {}})} | (flags or {})
        result = subprocess.run(
            ["/bin/bash", str(script)], env=environment, capture_output=True, text=True, timeout=20
        )
        if not success:
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(self.capture.exists())
            return [], {}
        self.assertEqual(result.returncode, 0, result.stderr[-2000:])
        argv = json.loads(self.capture.read_text())
        runtime = json.loads(next(v.split("=", 1)[1] for v in argv if v.startswith("--runtime-env-json=")))
        return argv, runtime["env_vars"]

    def test_sft_fp4_is_opt_in(self) -> None:
        argv, env = self.launch(False)
        for flag in ("--custom-model-provider-path", "--fp8-format", "--fp8-recipe"):
            self.assertNotIn(flag, argv)
        self.assertFalse(any(key.startswith(PREFIX) for key in env))

        argv, env = self.launch(True)
        for flag, value in {
            "--transformer-impl": "transformer_engine",
            "--fp8-format": "e4m3",
            "--fp8-recipe": "blockwise",
            "--custom-model-provider-path": PROVIDER,
        }.items():
            self.assertEqual(argv.count(flag), 1)
            self.assertEqual(argv[argv.index(flag) + 1], value)
        for flag in ("--bf16", "--optimizer-cpu-offload", "--use-precision-aware-optimizer"):
            self.assertIn(flag, argv)
        self.assertNotIn("--no-save-optim", argv)
        for key, value in {"MODE": "native", "EXPERT_COMPUTE": "bf16", "INDEXER": "0", "BF16_SCORES": "1"}.items():
            self.assertEqual(env[PREFIX + key], value)
        self.assertEqual(env["NVTE_FP8_BLOCK_SCALING_FP32_SCALES"], "1")

    def test_fp4_settings_override_stale_runtime_and_clean_lists(self) -> None:
        old = "RELAX_DSV4_FP4_V3_STATS_DIR"
        listed = ",".join((old, "UNRELATED_ENV"))
        incoming = {
            "env_vars": {
                old: "/old",
                PREFIX + "MODE": "stock_fp8",
                PREFIX + "STATS_DIR": "/stale",
                "UNRELATED_ENV": "keep",
                "RELAX_EXTRA_ENV_ALLOWLIST": listed,
                "RELAX_PROPAGATE_ENV_VARS": listed,
            }
        }
        _, defaults = self.launch(True, runtime=incoming)
        self.assertNotIn(PREFIX + "STATS_DIR", defaults)
        self.assertEqual(defaults[PREFIX + "MODE"], "native")
        flags = {
            PREFIX + "MODE": "stock_fp8",
            PREFIX + "EXPERT_COMPUTE": "fp8",
            PREFIX + "STATS_DIR": str(self.folder / "stats"),
        }
        _, env = self.launch(True, flags, incoming)
        self.assertEqual({key: env[key] for key in flags}, flags)
        self.assertEqual(env["UNRELATED_ENV"], "keep")
        self.assertNotIn(old, env)
        for name in ("RELAX_EXTRA_ENV_ALLOWLIST", "RELAX_PROPAGATE_ENV_VARS"):
            self.assertNotIn(old, env.get(name, "").split(","))
            self.assertIn("UNRELATED_ENV", env[name].split(","))
        self.assertTrue(set(flags) <= set(env["RELAX_PROPAGATE_ENV_VARS"].split(",")))

    def test_fp4_rejects_obsolete_or_invalid_shell_settings(self) -> None:
        for key, value in (
            ("RELAX_DSV4_QAT_VERSION", "v3"),
            (PREFIX + "STATS_INTERVAL", "0"),
            (PREFIX + "MODE", "invalid"),
        ):
            with self.subTest(key=key):
                self.launch(True, {key: value}, success=False)

    def test_rl_selects_stock_fp8_in_training_actor_environment(self) -> None:
        stale = {"env_vars": {PREFIX + "MODE": "native", "UNRELATED_ENV": "keep"}}
        argv, env = self.launch(True, runtime=stale, script=RL_SCRIPT)
        train_env = json.loads(argv[argv.index("--train-env-vars") + 1])
        self.assertEqual((env | train_env)[PREFIX + "MODE"], "stock_fp8")
        self.assertEqual(env["UNRELATED_ENV"], "keep")
        self.assertEqual(argv[argv.index("--custom-model-provider-path") + 1], PROVIDER)
        self.assertEqual(argv[argv.index("--fp8-recipe") + 1], "blockwise")
        self.assertEqual(env["NVTE_FP8_BLOCK_SCALING_FP32_SCALES"], "1")


if __name__ == "__main__":
    unittest.main()
