# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Locate installed SGLang source without importing its GPU kernels."""

import importlib.util
import os
from pathlib import Path

import pytest


def sglang_source_path(relative: str) -> Path:
    root = os.environ.get("SGLANG_TEST_SOURCE_ROOT")
    source_override = root is not None
    if root is None:
        spec = importlib.util.find_spec("sglang")
        if spec is None:
            pytest.skip("Requires SGLang source or SGLANG_TEST_SOURCE_ROOT")
        root = str(Path(spec.origin).parent)
    path = Path(root) / relative
    if not path.exists():
        if not source_override:
            pytest.skip(f"Requires a SGLang version with {relative}")
        pytest.fail(f"Missing SGLang source: {path}")
    return path
