# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


EXAMPLE_DIR = Path(__file__).resolve().parents[3] / "examples" / "nemo_gym_agentic"
PACKAGE_NAME = "relax_nemo_gym_example"
PACKAGE_SPEC = importlib.util.spec_from_file_location(
    PACKAGE_NAME,
    EXAMPLE_DIR / "__init__.py",
    submodule_search_locations=[str(EXAMPLE_DIR)],
)
if PACKAGE_SPEC is None or PACKAGE_SPEC.loader is None:
    raise RuntimeError(f"Unable to load NeMo Gym example package from {EXAMPLE_DIR}")
PACKAGE_MODULE = importlib.util.module_from_spec(PACKAGE_SPEC)
sys.modules[PACKAGE_NAME] = PACKAGE_MODULE
PACKAGE_SPEC.loader.exec_module(PACKAGE_MODULE)
