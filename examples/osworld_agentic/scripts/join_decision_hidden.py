# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Join process-sharded decision vectors with OSWorld terminal outcomes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterator


def records(root: Path, kind: str) -> Iterator[dict[str, Any]]:
    for path in sorted(root.glob(f"{kind}-*.jsonl")):
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                yield json.loads(line)


def joined_records(root: Path) -> Iterator[dict[str, Any]]:
    outcomes = {(r["task_id"], r["session_id"]): r for r in records(root, "outcomes")}
    for record in records(root, "requests"):
        outcome = outcomes.get((record["task_id"], record["session_id"]))
        yield {
            **record,
            "reward": outcome["reward"] if outcome else None,
            "success": outcome["success"] if outcome else None,
            "outcome_available": outcome is not None,
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    for row in joined_records(args.directory):
        print(json.dumps(row, ensure_ascii=False, allow_nan=False))
