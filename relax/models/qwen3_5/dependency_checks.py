# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Verify the Docker-built runtime without replacing Python import paths."""

import hashlib
import json
from pathlib import Path


def runtime_manifest() -> dict:
    return json.loads(Path("/etc/relax-qwen35-runtime.json").read_text())


def verify_megatron_sources() -> None:
    import inspect

    from megatron.core.transformer.moe import experts

    root = Path(inspect.getfile(experts)).resolve().parents[4]
    for name, expected in runtime_manifest()["megatron_sources"].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, name
