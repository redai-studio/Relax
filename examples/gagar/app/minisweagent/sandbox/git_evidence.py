# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Freeze the SWE actor's changes independently of dirty task images."""

import re
import shlex
from typing import Any


_BASELINE_COMMAND = r"""set -e
cd "$(git rev-parse --show-toplevel)"
index=$(mktemp)
trap 'rm -f "$index"' EXIT
# Git uses the index mtime to detect racy-clean entries.
cp -p "$(git rev-parse --git-path index)" "$index"
GIT_INDEX_FILE="$index" git add -u -- .
head=$(git rev-parse HEAD)
tree=$(GIT_INDEX_FILE="$index" git write-tree)
printf '\n%s\n%s\n' "$head" "$tree"
"""


def _execute(env: Any, command: str) -> dict:
    result = env.execute({"command": command}, cwd="/testbed", check_finished=False)
    if result.get("returncode") != 0 or result.get("exception_info"):
        raise RuntimeError(f"git evidence failed: {str(result)[:1000]}")
    return result


def capture_baseline(env: Any, evidence: dict) -> None:
    """Snapshot post-setup tracked state without changing HEAD or the real
    index."""
    try:
        result = _execute(env, _BASELINE_COMMAND)
        head, tree = result["output"].strip().splitlines()[-2:]
        if not all(re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", value) for value in (head, tree)):
            raise ValueError("git baseline did not return two object IDs")
        evidence["baseline"] = {
            "head": head,
            "tree": tree,
            "patch": _execute(env, f"git diff --binary {shlex.quote(head)} {shlex.quote(tree)}"),
        }
    except Exception as exc:
        # Evidence is optional for the original binary judge, but mandatory for grading.
        evidence["capture_error"] = f"baseline: {type(exc).__name__}: {exc}"


def capture_patch(env: Any, evidence: dict) -> None:
    """Keep the original HEAD diff for audit and expose only actor edits as
    patch."""
    try:
        evidence["head_patch"] = _execute(env, "git diff --binary HEAD")
        if "capture_error" not in evidence:
            tree = evidence["baseline"]["tree"]
            evidence["patch"] = _execute(env, f"git diff --binary {shlex.quote(tree)}")
    except Exception as exc:
        evidence["capture_error"] = f"patch: {type(exc).__name__}: {exc}"
