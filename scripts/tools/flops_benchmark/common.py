# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import ast
import hashlib
import json
import re
from pathlib import Path
from typing import Any


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def read_metrics(path: Path, kind: str) -> dict[int, dict[str, Any]]:
    """Read the existing full-precision training dictionaries, including Ray
    prefixes."""
    prefix = "perf" if kind == "perf" else "step"
    pattern = re.compile(rf"\b{prefix} (\d+): (\{{.*\}})")
    results: dict[int, dict[str, Any]] = {}
    for line in path.read_text(errors="replace").splitlines():
        line = re.sub(r"\x1b\[[0-9;]*m", "", line)
        match = pattern.search(line)
        if not match:
            continue
        step = int(match[1])
        value = ast.literal_eval(match[2])
        if step in results and results[step] != value:
            raise ValueError(f"Conflicting {kind} records for step {step} in {path}")
        results[step] = value
    if not results:
        raise ValueError(f"No {kind} metrics in {path}")
    return results
