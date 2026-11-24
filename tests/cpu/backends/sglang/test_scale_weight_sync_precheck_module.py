# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Tests for the installable NCCL precheck module."""

import subprocess
import sys


def test_precheck_module_exposes_cli_help():
    result = subprocess.run(
        [sys.executable, "-m", "relax.backends.sglang._scale_weight_sync_precheck", "--help"],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0
    assert "--master-address" in result.stdout
