# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Read exact SGLang patch sources without importing SGLang.

Set SGLANG_SOURCE_DIR to the target checkout, its python directory, or its
sglang package directory. Otherwise use installed source when available. Both
pristine and already patched files are accepted after checking the original Git
blob hash; no files are changed and no sources are downloaded.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import os
import re
import unittest
from functools import lru_cache
from pathlib import Path


PATCH_PATH = Path(__file__).resolve().parents[2] / "docker/patch/sglang/v0.5.21.patch"
HUNK_PATTERN = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def _section(patch: str, target: str) -> str:
    prefix = f"a/{target} b/{target}\n"
    sections = [section for section in patch.split("diff --git ") if section.startswith(prefix)]
    if len(sections) != 1:
        raise AssertionError(f"Expected one patch section for {target}, found {len(sections)}")
    return sections[0]


def apply_file_patch(source: str, patch: str, target: str, *, reverse: bool = False) -> str:
    """Apply exact hunk positions and contents in memory, without fuzzy
    matching."""
    lines = source.splitlines(keepends=True)
    diff = _section(patch, target).splitlines(keepends=True)
    output: list[str] = []
    cursor = 0
    index = next((i for i, line in enumerate(diff) if HUNK_PATTERN.match(line)), len(diff))
    if index == len(diff):
        raise AssertionError(f"No patch hunks for {target}")
    while index < len(diff):
        match = HUNK_PATTERN.match(diff[index])
        if match is None:
            raise AssertionError(f"Malformed diff hunk for {target}: {diff[index]!r}")
        old_count, new_count = int(match[2] or "1"), int(match[4] or "1")
        old_start, new_start = int(match[1]) - bool(old_count), int(match[3]) - bool(new_count)
        old: list[str] = []
        new: list[str] = []
        index += 1
        while index < len(diff) and not HUNK_PATTERN.match(diff[index]):
            line = diff[index]
            if line.startswith("\\ No newline at end of file"):
                previous = diff[index - 1][0]
                if previous in " -":
                    old[-1] = old[-1].removesuffix("\n")
                if previous in " +":
                    new[-1] = new[-1].removesuffix("\n")
            elif line.startswith((" ", "-", "+")):
                if line[0] in " -":
                    old.append(line[1:])
                if line[0] in " +":
                    new.append(line[1:])
            else:
                raise AssertionError(f"Malformed diff line for {target}: {line!r}")
            index += 1
        if len(old) != old_count or len(new) != new_count:
            raise AssertionError(f"Inconsistent diff counts for {target}")
        if reverse:
            old, new = new, old
            old_start, new_start = new_start, old_start
        if old_start < cursor or old_start > len(lines) or lines[old_start : old_start + len(old)] != old:
            raise AssertionError(f"Source does not match patch hunk for {target}:{old_start + 1}")
        output.extend(lines[cursor:old_start])
        if len(output) != new_start:
            raise AssertionError(f"Inconsistent diff positions for {target}")
        output.extend(new)
        cursor = old_start + len(old)
    output.extend(lines[cursor:])
    return "".join(output)


def _blob_hash(source: str) -> str:
    data = source.encode("utf-8")
    return hashlib.sha1(b"blob " + str(len(data)).encode("ascii") + b"\0" + data).hexdigest()


def _validated_source(source: str, patch: str, target: str, extra_patches: tuple[str, ...] = ()) -> str:
    patches = (patch, *extra_patches)
    match = re.search(r"^index ([0-9a-f]+)\.\.", _section(patch, target), re.MULTILINE)
    if match is None:
        raise AssertionError(f"Missing original blob hash for {target}")
    original_hash = match[1]
    if _blob_hash(source).startswith(original_hash):
        for current in patches:
            source = apply_file_patch(source, current, target)
        return source
    # Verify already patched files against the current hunks, even if someone
    # edited the patch without refreshing its new-side index hash.
    original = source
    for current in reversed(patches):
        original = apply_file_patch(original, current, target, reverse=True)
    if not _blob_hash(original).startswith(original_hash):
        raise AssertionError(f"Source does not match the target SGLang blob for {target}")
    result = original
    for current in patches:
        result = apply_file_patch(result, current, target)
    if result != source:
        raise AssertionError(f"Patched source does not round-trip for {target}")
    return result


@lru_cache(maxsize=None)
def patched_source(target: str, *, include_per_position: bool = False) -> str:
    explicit = os.environ.get("SGLANG_SOURCE_DIR")
    if explicit:
        root = Path(explicit).expanduser()
        candidates = (root / "python/sglang", root / "sglang", root)
    else:
        spec = importlib.util.find_spec("sglang")
        candidates = tuple(Path(path) for path in spec.submodule_search_locations or ()) if spec else ()
    package = next((path for path in candidates if (path / "srt").is_dir()), None)
    if package is None:
        message = "Target SGLang source is unavailable; set SGLANG_SOURCE_DIR to a clean v0.5.21 checkout"
        if explicit:
            raise AssertionError(f"{message}: {explicit}")
        raise unittest.SkipTest(message)
    source_path = package / target.removeprefix("python/sglang/")
    try:
        source = source_path.read_bytes().decode("utf-8")
        patches = [PATCH_PATH.read_text()]
        if include_per_position:
            patches.append((PATCH_PATH.parent.parent / "latest/sglang_per_pos_topk.patch").read_text())
            header = f"diff --git a/{target} b/{target}\n"
            patches = [patch for patch in patches if header in patch]
        return _validated_source(source, patches[0], target, tuple(patches[1:]))
    except (OSError, AssertionError) as error:
        message = f"Target SGLang source validation failed at {source_path}: {error}"
        if explicit:
            raise AssertionError(message) from error
        raise unittest.SkipTest(f"{message}; set SGLANG_SOURCE_DIR to a clean v0.5.21 checkout") from error


def extract_functions(source: str, target: str, names: set[str]) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
    """Extract complete functions from a complete, parsed source file."""
    methods: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {}
    for node in ast.walk(ast.parse(source, filename=target)):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names:
            if node.name in methods:
                raise AssertionError(f"Ambiguous function in {target}: {node.name}")
            methods[node.name] = node
    missing = names - methods.keys()
    if missing:
        raise AssertionError(f"Missing functions in {target}: {sorted(missing)}")
    return list(methods.values())


def patched_functions(target: str, names: set[str]) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
    return extract_functions(patched_source(target), target, names)
