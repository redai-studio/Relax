# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""The image's unconditional GDN helper import must be available in CR2."""

import subprocess
import sys
from pathlib import Path


def test_gdn_static_helper_imports_without_optional_kernel_dependencies():
    script = """
import importlib.abc
import os
import sys

class NoOptionalBackend(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "megatron" or fullname.startswith("megatron."):
            raise ModuleNotFoundError("No module named 'megatron'", name="megatron")
        if fullname == "fla" or fullname.startswith("fla."):
            raise AssertionError("Default GDN preparation must not import FLA kernels")

sys.meta_path.insert(0, NoOptionalBackend())
os.environ.pop("RELAX_QWEN38_GDN_OUTPUT_SAFE_CONFIG", None)
from relax.models.qwen3_8.legacy_precision import prep_compile
assert prep_compile() is False
assert "fla.ops.common.chunk_o" not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[3],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
