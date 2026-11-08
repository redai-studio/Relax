# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Load exact source bodies without importing Ray, Megatron or GPU packages."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import sys
import types
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
SLIME_SHA = "71c2679b1a87dc06c591a446b09561c425f74f38"
TQ_SHA = "9784ad0d8a5f0db93ff8326379ab2a6d06e640a0"
RELAX_SHA = "3673d7eb566f2fd05dbf0e450ae4746a0d7758f8"
PATCH = ROOT / "docker/patch/transferqueue/9784ad0-assignment-transform.patch"
REFERENCE_MANIFEST_SHA256 = "7ca415e44f0a0606530b289448188e6bf180c3cca849e39619587588bdc681c2"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reference_manifest() -> dict:
    """Read the immutable P1 provenance anchor before trusting its hashes."""
    data = (HERE / "reference/versions.json").read_bytes()
    if hashlib.sha256(data).hexdigest() != REFERENCE_MANIFEST_SHA256:
        raise ValueError("Fixed reference manifest integrity mismatch")
    return json.loads(data)


def verified_source(path: Path, text: str | None = None) -> str:
    """Check fixed reference bytes before AST parsing or execution."""
    data = path.read_bytes()
    relative = None
    reference = HERE / "reference"
    # A symlink in a reference slot must not turn it into an unchecked source.
    for candidate, base in (
        (Path(os.path.abspath(path)), Path(os.path.abspath(reference))),
        (path.resolve(), reference.resolve()),
    ):
        try:
            relative = candidate.relative_to(base).as_posix()
            break
        except ValueError:
            continue
    if relative is None:
        return data.decode("utf-8") if text is None else text
    manifest = reference_manifest()
    expected = manifest["sha256"].get("tests/no16/reference/" + relative)
    if expected is None or hashlib.sha256(data).hexdigest() != expected:
        raise ValueError(f"Fixed reference source integrity mismatch: {relative}")
    if text is not None and hashlib.sha256(text.encode("utf-8")).hexdigest() != expected:
        patched = manifest["transfer_queue"]["patched_sha256"]
        if (
            relative != "transfer_queue/seqlen_balanced_sampler.py.txt"
            or hashlib.sha256(text.encode("utf-8")).hexdigest() != patched
        ):
            raise ValueError(f"Fixed reference text override integrity mismatch: {relative}")
    return data.decode("utf-8") if text is None else text


def patched_seqlen_source() -> str:
    """Apply the repository's pinned single-file patch, requiring exact
    context."""
    manifest = reference_manifest()
    original = verified_source(HERE / "reference/transfer_queue/seqlen_balanced_sampler.py.txt")
    lines = original.splitlines(keepends=True)
    patch_bytes = PATCH.read_bytes()
    if hashlib.sha256(patch_bytes).hexdigest() != manifest["sha256"][manifest["transfer_queue"]["patch"]]:
        raise ValueError("Fixed TransferQueue patch integrity mismatch")
    patch = patch_bytes.decode("utf-8").splitlines(keepends=True)
    result: list[str] = []
    position = 0
    in_hunk = False
    for line in patch:
        match = re.match(r"@@ -(\d+)(?:,\d+)? \+\d+(?:,\d+)? @@", line)
        if match:
            start = int(match.group(1)) - 1
            result.extend(lines[position:start])
            position = start
            in_hunk = True
        elif in_hunk and line[0] in " +-":
            if line[0] != "+":
                if position >= len(lines) or lines[position] != line[1:]:
                    raise ValueError("Pinned TransferQueue patch context mismatch")
                position += 1
            if line[0] != "-":
                result.append(line[1:])
    if not in_hunk:
        raise ValueError("Pinned patch has no hunks")
    result.extend(lines[position:])
    patched = "".join(result)
    if hashlib.sha256(patched.encode("utf-8")).hexdigest() != manifest["transfer_queue"]["patched_sha256"]:
        raise ValueError("Patched TransferQueue source integrity mismatch")
    return patched


def load(
    path: Path,
    *,
    injected: dict[str, Any] | None = None,
    omitted_imports: tuple[str, ...] = (),
    names: set[str] | None = None,
    text: str | None = None,
    class_name: str | None = None,
) -> types.ModuleType:
    """Preserve selected bodies; only dependency imports/annotations are
    isolated."""
    tree = ast.parse(verified_source(path, text), filename=str(path))

    class DependencyImports(ast.NodeTransformer):
        def visit_ImportFrom(self, node: ast.ImportFrom) -> ast.AST | None:
            if node.module and node.module.startswith(omitted_imports):
                return None
            return node

        def visit_Try(self, node: ast.Try) -> ast.AST:
            self.generic_visit(node)
            if not node.body:
                node.body = [ast.copy_location(ast.Pass(), node)]
            return node

    tree = DependencyImports().visit(tree)
    body = (
        tree.body
        if class_name is None
        else next(node.body for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    )
    nodes = []
    for node in body:
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith(omitted_imports):
            continue
        if names is not None and getattr(node, "name", None) not in names:
            continue
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            continue
        nodes.append(node)
    module = types.ModuleType("_no16_source_" + str(len([name for name in sys.modules if name.startswith("_no16_")])))
    module.__dict__.update(injected or {})
    # Dataclasses and deepcopy need a registered module, even for isolated source.
    sys.modules[module.__name__] = module
    future = ast.parse("from __future__ import annotations").body
    code = ast.fix_missing_locations(ast.Module(body=future + nodes, type_ignores=[]))
    exec(compile(code, str(path), "exec"), module.__dict__)
    return module


def slime() -> types.ModuleType:
    reference = HERE / "reference/slime_2017"
    balancing = load(reference / "seqlen_balancing.py.txt")
    flops = load(reference / "flops_utils.py.txt")
    return load(
        reference / "dp_schedule.py.txt",
        omitted_imports=("slime.",),
        injected={
            "calculate_fwd_flops": flops.calculate_fwd_flops,
            "get_seqlen_balanced_partitions": balancing.get_seqlen_balanced_partitions,
            "first_fit_pack": balancing.first_fit_pack,
            "expand_bins_by_splitting": balancing.expand_bins_by_splitting,
        },
    )


def samplers() -> dict[str, Any]:
    reference = HERE / "reference/transfer_queue"
    base = load(reference / "base.py.txt")
    group = load(
        reference / "grpo_group_n_sampler.py.txt",
        omitted_imports=("transfer_queue",),
        injected={"BaseSampler": base.BaseSampler},
    )
    seqlen = load(
        reference / "seqlen_balanced_sampler.py.txt",
        omitted_imports=("transfer_queue",),
        injected={"GRPOGroupNSampler": group.GRPOGroupNSampler},
        text=patched_seqlen_source(),
    )
    streaming = load(
        reference / "streaming_token_budget_sampler.py.txt",
        omitted_imports=("transfer_queue",),
        injected={
            "GRPOGroupNSampler": group.GRPOGroupNSampler,
            "get_seqlen_balanced_partitions": seqlen.get_seqlen_balanced_partitions,
        },
    )
    identity = load(
        ROOT / "relax/utils/data/identity_window_sampler.py",
        omitted_imports=("transfer_queue",),
        injected={"StreamingTokenBudgetSampler": streaming.StreamingTokenBudgetSampler},
    )
    wholepack = load(
        ROOT / "relax/utils/data/wholepack.py",
        omitted_imports=("transfer_queue", "relax.utils.logging_utils"),
        injected={
            "BaseSampler": base.BaseSampler,
            "SeqlenBalancedSampler": seqlen.SeqlenBalancedSampler,
            "get_logger": lambda name: types.SimpleNamespace(info=lambda *a, **kw: None),
        },
    )
    return {
        "sft": seqlen.SeqlenBalancedSampler,
        "rl": identity.IdentityWindowSampler,
        "wholepack": wholepack,
        "base": base.BaseSampler,
    }
