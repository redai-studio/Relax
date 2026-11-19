# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Append-only, process-owned research sidecars; independent of archive
quotas."""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any


_SHARD = uuid.uuid4().hex


def enabled() -> bool:
    return bool(os.environ.get("RELAX_DECISION_HIDDEN_DIR"))


def write_record(kind: str, record: dict[str, Any]) -> None:
    root = os.environ.get("RELAX_DECISION_HIDDEN_DIR")
    if not root:
        return
    path = Path(root)
    path.mkdir(parents=True, exist_ok=True)
    # PID also isolates forked workers; UUID isolates hosts and restarts.
    with (path / f"{kind}-{_SHARD}-{os.getpid()}.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
