# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Verify public evidence checksums, arithmetic and optional source-copy
invariants."""

from __future__ import annotations

import argparse
import hashlib
import math
import re
import sys
from pathlib import Path
from typing import Any

from recompute import read_json, recompute


PRIVATE_PATTERNS = (
    re.compile(
        r"(?<![A-Za-z0-9:])/(?:gpfs|home|root|Users|tmp|opt|usr|var|data|scratch|mnt|run)(?:/[^\s'\"<>;,)\]}]*)?"
    ),
    re.compile(r"\bgpu-\d+\b", re.IGNORECASE),
    re.compile(r"GPU-[0-9a-fA-F-]{8,}"),
)
PLACEHOLDER = re.compile(r"<(?:PATH|USER|HOST|GPU_UUID)_\d{4}>")


def original_string(value: str) -> str:
    """Replace only the documented private-string categories for comparison."""
    for pattern in PRIVATE_PATTERNS:
        value = pattern.sub("<REDACTED>", value)
    return value


def compare_copy(original: Any, public: Any) -> None:
    """Check every numeric value, parameter key and nonprivate string
    recursively."""
    if isinstance(original, dict):
        if original.keys() != public.keys():
            raise ValueError("Complete-copy JSON keys changed")
        for key in original:
            compare_copy(original[key], public[key])
    elif isinstance(original, list):
        if len(original) != len(public):
            raise ValueError("Complete-copy JSON list length changed")
        for a, b in zip(original, public):
            compare_copy(a, b)
    elif isinstance(original, str):
        if original_string(original) != PLACEHOLDER.sub("<REDACTED>", public):
            raise ValueError("A nonprivate source string changed")
    elif original != public or isinstance(original, bool) != isinstance(public, bool):
        raise ValueError("A numeric/boolean/null source value changed")


def verify(root: Path, source: Path | None = None) -> int:
    """Check the manifest and all published performance aggregates without a
    GPU."""
    manifest = read_json(root / "MANIFEST.json")
    for row in manifest["files"]:
        data = (root / row["path"]).read_bytes()
        if len(data) != row["bytes"] or hashlib.sha256(data).hexdigest() != row["sha256"]:
            raise ValueError(f"Public bytes/SHA mismatch: {row['path']}")
        if source is not None:
            for original in row["sources"]:
                data = (source / original["path"]).read_bytes()
                if len(data) != original["bytes"] or hashlib.sha256(data).hexdigest() != original["sha256"]:
                    raise ValueError(f"Original source bytes/SHA mismatch: {original['path']}")
            if row["content"] == "complete original JSON with private strings replaced":
                original = source / row["sources"][0]["path"]
                compare_copy(read_json(original), read_json(root / row["path"]))
    actual, summary = recompute(root), read_json(root / "summary.json")
    for name, row in actual["performance"].items():
        for key in ("mean_gain", "median_gain", "min_gain", "max_gain", "sample_stddev_gain"):
            expected = summary["performance"][name][key]
            # Statistics implementations can round differently across Python versions.
            tolerance = 4 * max(math.ulp(row[key]), math.ulp(expected))
            if not math.isclose(row[key], expected, rel_tol=0, abs_tol=tolerance):
                raise ValueError(f"Performance summary differs: {name}/{key}")
    return len(manifest["files"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, nargs="?", default=Path(__file__).resolve().parent)
    parser.add_argument("--source", type=Path, help="Optional unredacted received-packet directory")
    args = parser.parse_args()
    count = verify(args.directory, args.source)
    sys.stdout.write(f"Verified {count} public members and all 18 performance segments.\n")
