# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""CPU utilities must remain importable without the optional training
backend."""

import subprocess
import sys
from pathlib import Path


def test_backend_compat_import_without_megatron():
    script = """
import importlib.abc
import sys

class NoMegatron(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "megatron" or fullname.startswith("megatron."):
            raise ModuleNotFoundError("No module named 'megatron'", name="megatron")

sys.meta_path.insert(0, NoMegatron())
from relax.backends.megatron import compat
assert compat.__name__ == "relax.backends.megatron.compat"
assert "relax.models.common.chunked_mtp_loss" not in sys.modules
assert "relax.models.common.chunked_grad_coalesce" not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[3],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
