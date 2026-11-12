# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Checkpoint-service imports must not require the optional Megatron stack."""

from __future__ import annotations

import subprocess
import sys
import textwrap


_BLOCK_MEGATRON = """
import importlib.abc
import sys

class BlockMegatron(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "megatron" or fullname.startswith("megatron."):
            raise ModuleNotFoundError("blocked for test", name=fullname)
        return None

sys.meta_path.insert(0, BlockMegatron())
"""


def _run(code: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(_BLOCK_MEGATRON + code)],
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )


def test_checkpoint_service_imports_without_megatron():
    result = _run("\nimport relax.distributed.checkpoint_service\n")
    assert result.returncode == 0, result.stderr


def test_device_direct_use_reports_clear_missing_megatron_dependency():
    result = _run(
        """
from relax.distributed.checkpoint_service.backends.device_direct import _load_megatron_dependencies

try:
    _load_megatron_dependencies()
except ModuleNotFoundError as exc:
    assert "requires the optional Megatron dependencies" in str(exc)
else:
    raise AssertionError("DeviceDirect dependency load unexpectedly succeeded")
"""
    )
    assert result.returncode == 0, result.stderr
