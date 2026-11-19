# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Source-row identity, captured before templating, filtering or shuffling."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any


class SourceRow(dict):
    """Keep provenance outside user columns and custom prompt callbacks."""

    def __init__(self, data: dict[str, Any], path: str, row_id: int) -> None:
        super().__init__(data)
        self.source_path = str(Path(path).resolve())
        self.source_row_id = str(row_id)


def snapshot_source(data: SourceRow, *, metadata_key: str, split: str) -> dict[str, Any]:
    metadata = data.get(metadata_key) or {}
    metadata = metadata if isinstance(metadata, dict) else {}
    source = {
        "dataset_id": metadata.get("dataset_id", data.source_path),
        "dataset_version": metadata.get("dataset_version"),
        "split": metadata.get("split", split),
        "source_row_id": data.source_row_id,
        "source_path": data.source_path,
        "input_payload": copy.deepcopy(dict(data)),
        "input_payload_ref": None,
        "identity_kind": "source_row",
    }
    source["prompt_id"] = prompt_identity(source)
    return source


def prompt_identity(source: dict[str, Any]) -> str:
    # Include content so an unversioned file changed in place cannot alias an
    # earlier question. Equal content at different source rows stays distinct.
    identity = json.dumps(source, sort_keys=True, ensure_ascii=False, default=str, separators=(",", ":"))
    return "prompt_" + hashlib.sha256(identity.encode("utf-8")).hexdigest()
