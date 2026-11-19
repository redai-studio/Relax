# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Tests for the installable NCCL precheck module."""

import json
import os
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


def test_probe_verdict_starts_on_its_own_line(monkeypatch, capfd):
    import torch

    from relax.backends.sglang import _scale_weight_sync_precheck as probe

    def fail(_device):
        os.write(sys.stdout.fileno(), b"NCCL log without a trailing newline")
        raise RuntimeError("CUDA unavailable")

    monkeypatch.setattr(torch.cuda, "set_device", fail)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "probe",
            "--master-address=127.0.0.1",
            "--master-port=6000",
            "--rank=0",
            "--device-id=0",
            "--timeout-secs=1",
            "--run-token=framing",
        ],
    )
    assert probe.main() == 1
    verdict = json.loads(capfd.readouterr().out.splitlines()[-1])
    assert not verdict["success"]
    assert verdict["run_token"] == "framing"
