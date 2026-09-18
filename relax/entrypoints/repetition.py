# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Diagnose repetition: python -m relax.entrypoints.repetition INPUT --output
REPORT.json."""

import argparse
import json
import os
import tempfile
import time
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from relax.utils.logging_utils import get_logger
from relax.utils.repetition import RepetitionConfig, analyze_repetition
from relax.utils.rollout_dump import DumpFormat, iter_rollout_records


logger = get_logger(__name__)


def discover_dumps(inputs: Sequence[str | Path], output: Path, input_format: DumpFormat = "auto") -> list[Path]:
    """Expand directories recursively, sort and deduplicate physical paths."""
    paths: set[Path] = set()
    for item in inputs:
        path = Path(item).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Input does not exist: {path}")
        if path.is_file():
            if path == output.resolve():
                raise ValueError("Report output must not overwrite an input dump")
            paths.add(path)
        elif path.is_dir():
            suffixes = {".jsonl"} if input_format == "jsonl" else {".pt", ".pth"}
            if input_format == "auto":
                suffixes.add(".jsonl")
            paths.update(
                candidate.resolve()
                for candidate in path.rglob("*")
                if candidate.is_file()
                and candidate.suffix.lower() in suffixes
                and candidate.resolve() != output.resolve()
            )
    if not paths:
        raise ValueError("No rollout dumps found (directory discovery accepts .jsonl, .pt and .pth)")
    return sorted(paths)


def diagnose_dumps(
    inputs: Sequence[str | Path],
    output: str | Path,
    *,
    config: RepetitionConfig = RepetitionConfig(),
    input_format: DumpFormat = "auto",
    trusted_torch: bool = False,
) -> dict[str, Any]:
    """Stream a complete JSON report and atomically publish it on success.

    Memory is bounded by one JSONL record and its hits, not the entire dump or
    report. Torch serialization itself requires loading one complete dump. A
    failed run leaves any existing report untouched.
    """
    output = Path(output).resolve()
    sources = discover_dumps(inputs, output, input_format)
    output.parent.mkdir(parents=True, exist_ok=True)
    header = {
        "schema_version": 1,
        "offsets": {"unit": "unicode_code_points", "start": "inclusive", "end": "exclusive", "base": 0},
        "detection": {
            **asdict(config),
            "algorithm": "zlib",
            "encoding": "utf-8",
            "compression_level": 9,
            "comparison": ">",
            "short_response_policy": "one_window",
        },
        "sources": [str(path) for path in sources],
    }
    summary: dict[str, Any] = {
        "samples": 0,
        "repetitive_samples": 0,
        "response_chars": 0,
        "covered_chars": 0,
        "scanned_windows": 0,
        "max_compression_ratio": None,
    }
    started = time.perf_counter()
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=output.parent, prefix=f".{output.name}.", suffix=".tmp", delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(json.dumps(header, ensure_ascii=False, allow_nan=False)[:-1] + ', "samples": [\n')
            first = True
            for source in sources:
                for record in iter_rollout_records(source, input_format=input_format, trusted_torch=trusted_torch):
                    try:
                        result = analyze_repetition(record.response, config)
                    except UnicodeError as exc:
                        raise ValueError(
                            f"{source}:record {record.record_index}: response is not valid UTF-8"
                        ) from exc
                    if not first:
                        stream.write(",\n")
                    try:
                        json.dump(
                            {**record.identity(), **result.to_dict()}, stream, ensure_ascii=False, allow_nan=False
                        )
                    except (TypeError, ValueError) as exc:
                        raise ValueError(
                            f"{source}:record {record.record_index}: sample identifiers must be JSON-compatible"
                        ) from exc
                    first = False
                    summary["samples"] += 1
                    summary["repetitive_samples"] += int(result.has_repetition)
                    for key in ("response_chars", "covered_chars", "scanned_windows"):
                        summary[key] += getattr(result, key)
                    maximum = result.max_compression_ratio
                    if maximum is not None:
                        previous = summary["max_compression_ratio"]
                        summary["max_compression_ratio"] = maximum if previous is None else max(previous, maximum)
                    if result.has_repetition:
                        logger.info(
                            "%s record=%s rollout_id=%s sample_index=%s index=%s hits=%s",
                            source,
                            record.record_index,
                            record.rollout_id,
                            record.sample_index,
                            record.index,
                            [(hit.start, hit.end, round(hit.compression_ratio, 4)) for hit in result.hits],
                        )
            count = summary["samples"]
            summary["repetition_frac"] = summary["repetitive_samples"] / count if count else 0.0
            summary["elapsed_seconds"] = time.perf_counter() - started
            stream.write('\n], "summary": ')
            json.dump(summary, stream, ensure_ascii=False, allow_nan=False)
            stream.write("}\n")
            stream.flush()
            os.fsync(stream.fileno())
        if os.name == "posix" and output.exists():
            temporary.chmod(output.stat().st_mode & 0o777)
        os.replace(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    logger.info("Report saved to %s: %s", output, summary)
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="Dump files or directories (recursively discovered)")
    parser.add_argument("--output", required=True, help="Destination JSON report")
    parser.add_argument("--window-size", type=int, default=10_000, help="Window size in Unicode code points")
    parser.add_argument("--stride", type=int, default=5_000, help="Stride in Unicode code points")
    parser.add_argument(
        "--threshold", type=float, default=10.0, help="Strict original/compressed byte ratio threshold"
    )
    parser.add_argument("--input-format", choices=("auto", "jsonl", "torch"), default="auto")
    parser.add_argument(
        "--trusted-torch", action="store_true", help="Allow pickle objects in trusted legacy Torch dumps"
    )
    args = parser.parse_args(argv)
    try:
        config = RepetitionConfig(args.window_size, args.stride, args.threshold)
        diagnose_dumps(
            args.inputs, args.output, config=config, input_format=args.input_format, trusted_torch=args.trusted_torch
        )
    except (OSError, ValueError, ImportError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main()
