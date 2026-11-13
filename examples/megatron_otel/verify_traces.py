#!/usr/bin/env python3

# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Check native Megatron spans and a successful policy update in console logs.

Accept Lens's single-line console JSON, including Ray prefixes and ANSI colors.
The completed spans must share a run and an ancestor ``relax.train.step`` span.
This checks exported execution evidence, not numerical accuracy or RL quality.
"""

import argparse
import json
import re
import sys
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any


_ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_NAMES = (
    "relax.train.step",
    "relax.optimizer.step",
    "megatron.microbatch.forward",
    "megatron.microbatch.backward",
    "megatron.grad_sync.start",
    "megatron.grad_sync.finish",
)


def console_spans(line: str) -> Iterator[dict[str, Any]]:
    """Decode span objects without assuming a particular Ray log prefix."""
    line = _ANSI.sub("", line)
    decoder = json.JSONDecoder()
    offset = 0
    while (start := line.find("{", offset)) >= 0:
        try:
            value, offset = decoder.raw_decode(line, start)
        except json.JSONDecodeError:
            offset = start + 1
            continue
        if (
            isinstance(value, dict)
            and isinstance(value.get("name"), str)
            and isinstance(value.get("context"), dict)
            and isinstance(value.get("resource"), dict)
        ):
            yield value


def span_id(value: Any, width: int) -> str | None:
    """Normalize the hexadecimal IDs produced by the OTel console exporter."""
    if not isinstance(value, str):
        return None
    normalized = value.lower().removeprefix("0x")
    if len(normalized) != width or not re.fullmatch(r"[0-9a-f]+", normalized) or int(normalized, 16) == 0:
        return None
    return normalized


def step_ancestor(
    key: tuple[str, str], spans: dict[tuple[str, str], dict[str, Any]], steps: set[tuple[str, str]]
) -> tuple[str, str] | None:
    """Follow parents within the same trace, tolerating intermediate spans."""
    seen = set()
    while key in spans and key not in seen:
        if key in steps:
            return key
        seen.add(key)
        parent = span_id(spans[key].get("parent_id"), 16)
        if parent is None:
            break
        key = key[0], parent
    return None


def verify(logs: list[str], run_id: str) -> dict[str, Any]:
    spans: dict[tuple[str, str], dict[str, Any]] = {}
    decoded = duplicates = 0
    errors: list[str] = []
    for filename in logs:
        try:
            stream = sys.stdin if filename == "-" else Path(filename).open(encoding="utf-8", errors="replace")
            try:
                for line in stream:
                    for record in console_spans(line):
                        decoded += 1
                        attributes = record["resource"].get("attributes", {})
                        if not isinstance(attributes, dict):
                            continue
                        record_run = attributes.get("relax.run.id", attributes.get("nemo.run.id"))
                        if record_run != run_id:
                            continue
                        trace = span_id(record["context"].get("trace_id"), 32)
                        current = span_id(record["context"].get("span_id"), 16)
                        if trace is None or current is None:
                            continue
                        if not isinstance(record.get("attributes"), dict):
                            record["attributes"] = {}
                        key = trace, current
                        if key in spans:
                            duplicates += 1
                        else:
                            spans[key] = record
            finally:
                if stream is not sys.stdin:
                    stream.close()
        except OSError as exc:
            errors.append(f"Cannot read {filename}: {exc}")

    counts = Counter(record["name"] for record in spans.values())
    steps = {
        key
        for key, record in spans.items()
        if record["name"] == "relax.train.step" and record.get("attributes", {}).get("relax.role") == "actor"
    }
    grouped: dict[tuple[str, str], Counter[str]] = {key: Counter() for key in steps}
    for key, record in spans.items():
        name = record["name"]
        if name not in _NAMES or name == "relax.train.step":
            continue
        ancestor = step_ancestor(key, spans, steps)
        if ancestor is None:
            continue
        if name == "relax.optimizer.step":
            if record.get("attributes", {}).get("relax.optimizer.update_successful") is True:
                grouped[ancestor]["successful_optimizer_updates"] += 1
        else:
            grouped[ancestor][name] += 1

    accepted = []
    for key, found in grouped.items():
        if not (
            found["megatron.microbatch.forward"]
            and found["megatron.microbatch.backward"]
            and found["successful_optimizer_updates"]
        ):
            continue
        attributes = spans[key].get("attributes", {})
        accepted.append(
            {
                "trace_id": key[0],
                "step_span_id": key[1],
                "rollout_id": attributes.get("relax.rollout_id"),
                "step_id": attributes.get("relax.step_id"),
                "span_counts": dict(found),
            }
        )

    if not spans:
        errors.append("No console spans with valid IDs and the requested run ID were found.")
    if not steps:
        errors.append("No exported actor relax.train.step span was found for this run.")
    if not counts["megatron.microbatch.forward"]:
        errors.append("No native megatron.microbatch.forward span was found for this run.")
    if not counts["megatron.microbatch.backward"]:
        errors.append("No native megatron.microbatch.backward span was found for this run.")
    if not accepted:
        errors.append(
            "No actor step contains native forward/backward spans and a successful relax.optimizer.step "
            "connected through trace_id/parent_id ancestry."
        )
    return {
        "passed": bool(accepted) and not errors,
        "run_id": run_id,
        "decoded_span_records": decoded,
        "matching_unique_spans": len(spans),
        "duplicate_records": duplicates,
        "span_counts": {name: counts[name] for name in _NAMES},
        "verified_actor_steps": len(accepted),
        "examples": accepted[:5],
        "error_count": len(errors),
        "errors": errors[:10],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", help="Console log files; use - for stdin")
    parser.add_argument("--run-id", required=True, help="The recipe's NEMO_LENS_RUN_ID")
    args = parser.parse_args()
    result = verify(args.logs, args.run_id)
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
