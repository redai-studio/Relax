# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Diagnose full responses in existing rollout .pt dumps or JSONL results."""

import argparse
import json
import sys
from collections.abc import Iterator, Sequence
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Any

from relax.utils.repetition import detect_repetition


def iter_dump_samples(path: Path) -> Iterator[tuple[int, dict[str, Any], Any]]:
    """Yield position, sample and rollout ID without loading JSONL in full.

    PyTorch dumps use the existing pickle format and must come from a trusted
    source. Tensors are loaded onto the CPU; Ray and model services are unused.
    """
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    sample = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
                yield line_number - 1, sample, sample.get("rollout_id") if isinstance(sample, dict) else None
    elif path.suffix == ".pt":
        import torch

        dump = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(dump, dict) or not isinstance(dump.get("samples"), list):
            raise ValueError(f"{path}: expected a rollout dump with a 'samples' list")
        for position, sample in enumerate(dump["samples"]):
            yield position, sample, dump.get("rollout_id")
    else:
        raise ValueError(f"{path}: expected a .pt or .jsonl rollout dump")


def diagnose_dump(
    path: Path,
    *,
    window_size: int = 10_000,
    stride: int = 5_000,
    threshold: float = 10.0,
) -> dict[str, Any]:
    """Build a JSON-compatible report, keeping sample IDs and no response
    text."""
    # Validate options even for an empty dump.
    detect_repetition("", window_size=window_size, stride=stride, threshold=threshold)
    reports = []
    for position, sample, rollout_id in iter_dump_samples(path):
        if not isinstance(sample, dict) or not isinstance(sample.get("response"), str):
            raise ValueError(f"{path}: sample at position {position} must have a string 'response'")
        result = detect_repetition(sample["response"], window_size=window_size, stride=stride, threshold=threshold)
        reports.append(
            {
                "sample_position": position,
                "rollout_id": rollout_id,
                "index": sample.get("index"),
                "sample_index": sample.get("sample_index"),
                "group_index": sample.get("group_index"),
                "dataset": sample.get("dataset"),
                "response_characters": len(sample["response"]),
                **asdict(result),
            }
        )
    hits = sum(report["has_repetition"] for report in reports)
    return {
        "source": str(path),
        "detector": {
            "window_size": window_size,
            "stride": stride,
            "threshold": threshold,
            "algorithm": "zlib",
            "level": 9,
            "offset_unit": "unicode_code_point",
            "interval": "[start, end)",
        },
        "summary": {
            "sample_count": len(reports),
            "repetitive_sample_count": hits,
            "repetition_frac": hits / len(reports) if reports else 0.0,
        },
        "samples": reports,
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dump", type=Path, help="Existing .jsonl results or a trusted .pt rollout dump")
    parser.add_argument("--output", type=Path, help="JSON report path; defaults to stdout")
    parser.add_argument("--window-size", type=int, default=10_000, help="Window size in Unicode code points")
    parser.add_argument("--stride", type=int, default=5_000, help="Stride in Unicode code points, at most window size")
    parser.add_argument(
        "--threshold", type=float, default=10.0, help="Hit when the compression ratio is strictly greater"
    )
    args = parser.parse_args(argv)
    if args.output is not None and args.output.resolve() == args.dump.resolve():
        parser.error("--output must not overwrite an input dump")
    try:
        report = diagnose_dump(args.dump, window_size=args.window_size, stride=args.stride, threshold=args.threshold)
        with args.output.open("w", encoding="utf-8") if args.output else nullcontext(sys.stdout) as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
    except (OSError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
