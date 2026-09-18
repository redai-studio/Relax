# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Offline repetition diagnosis over dumped rollout data.

Reads the two dump formats Relax writes and reports, per sample, whether the
response contains repetition anywhere in the full text, together with the
suspected character intervals:

``.jsonl``
    Written on every rollout step by
    :func:`relax.utils.training.train_dump_utils.save_rollout_result_jsonl`
    and ``save_eval_summary_jsonl``. Always on whenever
    ``--rollout-result-dir`` is set, so this is usually the file at hand.
    Read incrementally and without ``torch``.

``.pt``
    Written by ``--save-debug-rollout-data``. Holds pickled Python objects,
    so it is only read from a trusted source; ``torch`` is imported lazily,
    on this path alone.

Usage::

    python -m relax.entrypoints.diagnose_repetition <dump.jsonl|dump.pt> [...] -o report.json

Detection reuses :func:`relax.utils.repetition.scan_repetition`, so the
offline verdict matches the online ``rollout/repetition_frac`` metric for the
same text and thresholds.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Iterator

from relax.utils.logging_utils import get_logger
from relax.utils.repetition import (
    REPETITION_COMPRESSION_RATIO_THRESHOLD,
    REPETITION_WINDOW_SIZE_CHARS,
    REPETITION_WINDOW_STRIDE_CHARS,
    scan_repetition,
)


logger = get_logger(__name__)


def _iter_jsonl_samples(path: Path) -> Iterator[dict]:
    """Yield one record per non-blank line, without loading the whole file."""
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {e.msg}") from e
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object, got {type(record).__name__}")
            _validate_sample(record, f"{path}:{line_number}")
            yield record


def _load_pt_samples(path: Path) -> tuple[Any, list[dict]]:
    """Load a ``.pt`` rollout dump, returning ``(rollout_id, samples)``."""
    import torch

    with open(path, "rb") as f:
        # weights_only=False is required: a dump holds plain Python objects
        # (sample dicts, PIL images, numpy arrays), not just tensors, and
        # weights_only=True raises UnpicklingError on any multimodal dump.
        # The input is the operator's own training output, matching how
        # relax.utils.utils.get_debug_data loads the same files.
        payload = torch.load(f, map_location="cpu", weights_only=False)

    if not isinstance(payload, dict) or "samples" not in payload:
        raise ValueError(f"{path} is not a rollout dump: expected a dict with a 'samples' key")

    samples = payload["samples"]
    if not isinstance(samples, list):
        raise ValueError(f"{path}: 'samples' must be a list, got {type(samples).__name__}")

    for position, sample in enumerate(samples):
        _validate_sample(sample, f"{path}: sample at position {position}")
    return payload.get("rollout_id"), samples


def load_rollout_dump(path: str | Path) -> tuple[Any, Iterable[dict]]:
    """Load one dumped rollout file, returning ``(rollout_id, samples)``.

    Dispatches on the suffix: ``.jsonl`` is streamed as plain JSON, ``.pt`` is
    unpickled via ``torch``. For a ``.jsonl`` file the rollout id lives on each
    record rather than on the file, so it is reported per sample instead.
    """
    path = Path(path)
    if path.suffix == ".jsonl":
        return None, _iter_jsonl_samples(path)
    if path.suffix == ".pt":
        return _load_pt_samples(path)
    raise ValueError(f"{path}: expected a .jsonl or .pt rollout dump, got '{path.suffix or 'no suffix'}'")


def _validate_sample(sample: Any, location: str) -> None:
    if not isinstance(sample, dict) or not isinstance(sample.get("response"), str):
        raise ValueError(f"{location}: expected a sample with a string 'response'")


def _sample_identity(sample: dict, position: int) -> dict[str, Any]:
    """Identify a sample well enough to find it again in the dump."""
    return {
        "position": position,
        "dataset": sample.get("dataset"),
        "index": sample.get("index"),
        "sample_index": sample.get("sample_index"),
        "group_index": sample.get("group_index"),
        "rollout_id": sample.get("rollout_id"),
    }


def diagnose_samples(
    samples: Iterable[dict],
    *,
    window_size: int = REPETITION_WINDOW_SIZE_CHARS,
    stride: int = REPETITION_WINDOW_STRIDE_CHARS,
    threshold: float = REPETITION_COMPRESSION_RATIO_THRESHOLD,
    only_hits: bool = False,
) -> dict[str, Any]:
    """Scan every sample's ``response`` and summarize the hits."""
    scan_repetition("", window_size=window_size, stride=stride, threshold=threshold)
    results: list[dict[str, Any]] = []
    num_samples = 0
    num_hits = 0

    for position, sample in enumerate(samples):
        num_samples += 1
        _validate_sample(sample, f"sample at position {position}")
        response = sample["response"]
        report = scan_repetition(response, window_size=window_size, stride=stride, threshold=threshold)
        if report.has_repetition:
            num_hits += 1
        elif only_hits:
            continue
        results.append(_sample_identity(sample, position) | report.to_dict())

    return {
        "num_samples": num_samples,
        "num_repetitive_samples": num_hits,
        # Same definition as the online rollout/repetition_frac metric.
        "repetition_frac": (num_hits / num_samples) if num_samples else 0.0,
        "samples": results,
    }


def diagnose_dumps(
    paths: Iterable[str | Path],
    *,
    window_size: int = REPETITION_WINDOW_SIZE_CHARS,
    stride: int = REPETITION_WINDOW_STRIDE_CHARS,
    threshold: float = REPETITION_COMPRESSION_RATIO_THRESHOLD,
    only_hits: bool = False,
) -> dict[str, Any]:
    """Diagnose one or more rollout dumps and build a JSON-ready report."""
    scan_repetition("", window_size=window_size, stride=stride, threshold=threshold)
    files: list[dict[str, Any]] = []
    total_samples = 0
    total_hits = 0

    for path in paths:
        rollout_id, samples = load_rollout_dump(path)
        summary = diagnose_samples(
            samples,
            window_size=window_size,
            stride=stride,
            threshold=threshold,
            only_hits=only_hits,
        )
        total_samples += summary["num_samples"]
        total_hits += summary["num_repetitive_samples"]
        logger.info(
            "%s: %d/%d samples contain repetition",
            path,
            summary["num_repetitive_samples"],
            summary["num_samples"],
        )
        files.append({"path": str(path), "rollout_id": rollout_id} | summary)

    return {
        "config": {
            "window_size": window_size,
            "stride": stride,
            "threshold": threshold,
            # Offsets are character positions into the response, half-open
            # [start, end) slices; an interval marks a suspected repetitive
            # window, not an exact repetition boundary.
            "offset_unit": "characters",
            "interval_convention": "half-open [start, end)",
        },
        "num_samples": total_samples,
        "num_repetitive_samples": total_hits,
        "repetition_frac": (total_hits / total_samples) if total_samples else 0.0,
        "files": files,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Diagnose full-text repetition in dumped rollout responses.",
    )
    parser.add_argument(
        "dumps",
        nargs="+",
        help="Rollout dumps: .jsonl results (always written with --rollout-result-dir) "
        "or trusted .pt files (from --save-debug-rollout-data).",
    )
    parser.add_argument("-o", "--output", default=None, help="Write the JSON report here instead of stdout.")
    parser.add_argument(
        "--window-size",
        type=int,
        default=REPETITION_WINDOW_SIZE_CHARS,
        help="Scan window size in characters.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=REPETITION_WINDOW_STRIDE_CHARS,
        help="Distance between window starts, in characters; may not exceed --window-size.",
    )
    parser.add_argument("--threshold", type=float, default=REPETITION_COMPRESSION_RATIO_THRESHOLD)
    parser.add_argument(
        "--only-hits",
        action="store_true",
        help="Report only samples that hit, instead of every sample.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    # Report bad window parameters as a CLI usage error (clean message, exit
    # code 2) rather than letting the scan raise a traceback with exit code 0.
    try:
        scan_repetition("", window_size=args.window_size, stride=args.stride, threshold=args.threshold)
    except ValueError as e:
        parser.error(str(e))

    if args.output is not None:
        output = Path(args.output)
        # Writing the report over an input would destroy the data being
        # diagnosed, and the dumps are training output that cannot be re-created.
        for dump in args.dumps:
            dump_path = Path(dump)
            if dump_path.exists() and output.resolve() == dump_path.resolve():
                parser.error(f"--output must not overwrite an input dump: {dump}")

    try:
        report = diagnose_dumps(
            args.dumps,
            window_size=args.window_size,
            stride=args.stride,
            threshold=args.threshold,
            only_hits=args.only_hits,
        )
        payload = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)
        if args.output:
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(payload, encoding="utf-8")
            logger.info("Wrote repetition report to %s", output)
        else:
            print(payload)  # noqa: T201 - CLI report goes to stdout by design
    except (OSError, ValueError) as e:
        parser.error(str(e))


if __name__ == "__main__":
    main()
