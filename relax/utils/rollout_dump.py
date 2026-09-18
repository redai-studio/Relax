# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Read unmodified response records from existing rollout dump formats."""

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal


DumpFormat = Literal["auto", "jsonl", "torch"]


@dataclass(frozen=True)
class RolloutRecord:
    source: str
    record_index: int
    line_number: int | None
    rollout_id: Any
    sample_index: Any
    index: Any
    group_index: Any
    dataset: Any
    response: str

    def identity(self) -> dict[str, Any]:
        """Keep physical location as well as original, possibly absent IDs."""
        return {name: value for name, value in self.__dict__.items() if name != "response"}


def _record(
    raw: Any, path: Path, position: int, line_number: int | None = None, rollout_id: Any = None
) -> RolloutRecord:
    location = f"{path}:line {line_number}" if line_number is not None else f"{path}:record {position}"
    if not isinstance(raw, dict):
        raise ValueError(f"{location}: expected a sample object")
    if not isinstance(raw.get("response"), str):
        raise ValueError(f"{location}: response must be present and a string")
    # Do not infer sample_index from index: these fields have different meanings
    # in lightweight JSONL summaries and complete debug dumps.
    return RolloutRecord(
        str(path),
        position,
        line_number,
        raw.get("rollout_id", rollout_id),
        raw.get("sample_index"),
        raw.get("index"),
        raw.get("group_index"),
        raw.get("dataset"),
        raw["response"],
    )


def iter_rollout_records(
    path: str | Path, *, input_format: DumpFormat = "auto", trusted_torch: bool = False
) -> Iterator[RolloutRecord]:
    """Read JSONL incrementally or a complete Torch debug dump on CPU.

    Invalid records fail with their source location, never silently disappear.
    Torch's restricted loader is the default. Legacy dumps containing pickled
    objects require explicit trusted_torch=True, which must only be used for
    trusted files. JSONL requires no torch or training-stack dependencies.
    """
    path = Path(path).resolve()
    if input_format not in ("auto", "jsonl", "torch"):
        raise ValueError(f"Unsupported input format: {input_format}")
    if input_format == "auto":
        if path.suffix.lower() == ".jsonl":
            input_format = "jsonl"
        elif path.suffix.lower() in (".pt", ".pth"):
            input_format = "torch"
        else:
            raise ValueError(f"{path}: unknown dump extension; specify --input-format jsonl or torch")
    if input_format == "jsonl":
        position = 0
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:line {line_number}: invalid JSON: {exc.msg}") from exc
                yield _record(raw, path, position, line_number)
                position += 1
        return

    import torch

    try:
        dump = torch.load(path, map_location="cpu", weights_only=not trusted_torch)
    except Exception as exc:
        hint = " For trusted legacy pickle dumps only, use --trusted-torch." if not trusted_torch else ""
        raise ValueError(f"{path}: cannot load Torch dump ({type(exc).__name__}).{hint}") from exc
    if not isinstance(dump, dict) or not isinstance(dump.get("samples"), list):
        raise ValueError(f"{path}: expected a Torch dump with a samples list")
    for position, raw in enumerate(dump["samples"]):
        yield _record(raw, path, position, rollout_id=dump.get("rollout_id"))
