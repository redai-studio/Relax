# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Offline, exclusive-snapshot export and local SFT path conversion."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from relax.utils.logging_utils import get_logger

from .schema import ASSET_PREFIX, SCHEMA_VERSION, asset_ids, encode, path_fingerprint, validate
from .store import ArchiveStore


logger = get_logger(__name__)


def relative_assets(value: Any) -> Any:
    if isinstance(value, str) and value.startswith(ASSET_PREFIX):
        return "assets/" + value[len(ASSET_PREFIX) :]
    if isinstance(value, dict):
        return {k: relative_assets(v) for k, v in value.items()}
    if isinstance(value, list):
        return [relative_assets(v) for v in value]
    return value


def sft_samples(trajectory: dict, skipped: Counter) -> list[dict]:
    calls, actions = trajectory["model_calls"], trajectory["actions"]
    # Conservative correction policy: no targets from attempts with a known
    # execution error or parse/transport retry. Raw history is never rewritten.
    if any(a["status"] == "failure" or a.get("error") for a in actions):
        skipped["execution_failure"] += len(calls)
        return []
    if any(c.get("parse_status") != "parsed" or not c.get("selected") for c in calls):
        skipped["retry_or_correction"] += len(calls)
        return []
    samples = []
    for call in calls:
        executed = [a for a in actions if a["call_id"] == call["call_id"]]
        if not executed or any(
            a.get("origin") == "harness"
            or (a["status"] == "unknown" and a.get("execution_evidence") != "rpc_completed")
            for a in executed
        ):
            skipped["unconfirmed_execution"] += 1
            continue
        request = call["request"]
        reply = call.get("assistant_message")
        if not reply or reply.get("role") != "assistant":
            skipped["missing_assistant"] += 1
            continue
        messages = [*request["messages"], reply]
        allowed = {"role", "content", "tool_calls", "reasoning_content", "learn"}
        if any(
            m.get("role") not in {"system", "user", "assistant", "tool", "function_call"}
            or "content" not in m
            or set(m) - allowed
            for m in messages
        ):
            skipped["unsupported_loader_message_fields"] += 1
            continue
        samples.append(
            {
                "messages": [{**m, "learn": i == len(messages) - 1} for i, m in enumerate(messages)],
                "tools": request.get("tools") or [],
                "metadata": {
                    "trajectory_id": trajectory["trajectory_id"],
                    "call_id": call["call_id"],
                    "task_source": trajectory["task_source"],
                    "task_id": trajectory["task_id"],
                    "apply_chat_template_kwargs": request.get("chat_template_kwargs") or {},
                    "supervision": "last_assistant_only",
                },
            }
        )
    return samples


def export_archive(store: ArchiveStore, destination: str | Path) -> dict:
    destination = Path(destination).resolve()
    if destination == store.root or store.root in destination.parents:
        raise ValueError("export must be outside the archive budget directory")
    # mkdir(exist_ok=False) is the no-overwrite guard. Partial exports have no
    # manifest and are deliberately not reusable; choose a new destination.
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "assets").mkdir()
    reasons, sft_skipped = Counter(), Counter()
    selected = samples_count = total = 0
    seen: set[str] = set()
    sources: set[str] = set()
    with (
        store.mutex,
        (destination / "trajectories.jsonl").open("wb") as full,
        (destination / "samples.jsonl").open("wb") as samples,
    ):
        for path in sorted((store.root / "trajectories").glob("*.jsonl")):
            total += 1
            try:
                from .schema import materialize

                trajectory = materialize(store._events(path))
                if trajectory.get("integrity") != "complete":
                    raise ValueError("missing complete commit record")
                validate(trajectory)
                store.verify_assets(trajectory)
                fingerprint = path_fingerprint(trajectory)
                if fingerprint in seen:
                    reasons["duplicate_path"] += 1
                    continue
            except (ValueError, KeyError, TypeError, OSError) as exc:
                reasons[str(exc)] += 1
                continue
            seen.add(fingerprint)
            trajectory["integrity"] = "complete"
            for key in asset_ids(trajectory):
                target = destination / "assets" / key
                if not target.exists():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(store.asset_path(key), target)
            full.write(encode(relative_assets(trajectory)) + b"\n")
            for sample in sft_samples(trajectory, sft_skipped):
                samples.write(encode(relative_assets(sample)) + b"\n")
                samples_count += 1
            selected += 1
            sources.add(trajectory["task_source"])
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "export_version": 1,
        "sources": sorted(sources),
        "trajectories_considered": total,
        "trajectories_selected": selected,
        "trajectories_skipped": total - selected,
        "skip_reasons": dict(reasons),
        "duplicates": reasons["duplicate_path"],
        "missing": sum(n for reason, n in reasons.items() if "missing" in reason or "corrupted" in reason),
        "sft_samples": samples_count,
        "sft_skipped": dict(sft_skipped),
        "selection": "confirmed success, complete associations and verified assets; exact path deduplication",
        "sft_policy": "selected parsed calls only; no retry/correction trajectories or known execution failures; "
        "RPC completion plus successful evaluator may qualify, without claiming semantic action success",
        "supervision": "messages[*].learn=false except the final assistant reply; tools and template kwargs preserved",
        "snapshot": "exclusive archive owner lock held for the entire export; all Agents and writers must be stopped",
        "incomplete_staging_excluded": len(list((store.root / "staging").glob("*.jsonl"))),
    }
    (destination / "manifest.json").write_bytes(encode(manifest))
    return manifest


def convert_sft(source: str | Path, output: str | Path) -> None:
    """Create a deployment-local JSONL: only image URLs become absolute
    paths."""
    source = Path(source).resolve()

    def convert(value: Any) -> Any:
        if isinstance(value, dict):
            result = {k: convert(v) for k, v in value.items()}
            if result.get("type") in ("image", "image_url"):
                key = "image_url" if "image_url" in result else "image"
                source_value = result[key]
                relative = source_value.get("url") if isinstance(source_value, dict) else source_value
                path = (source.parent / relative).resolve()
                if source.parent not in path.parents or not path.is_file():
                    raise ValueError("missing or escaping exported image")
                result[key] = {**source_value, "url": str(path)} if isinstance(source_value, dict) else str(path)
            return result
        if isinstance(value, list):
            return [convert(v) for v in value]
        return value

    with Path(output).open("xb") as handle:
        for line in source.read_text().splitlines():
            handle.write(encode(convert(json.loads(line))) + b"\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    export = sub.add_parser("export")
    export.add_argument("--archive-dir", required=True)
    export.add_argument("--output-dir", required=True)
    export.add_argument("--shared-filesystem", action="store_true", help="Use the direct-write shared filesystem lock")
    recover = sub.add_parser("recover")
    recover.add_argument("--archive-dir", required=True)
    recover.add_argument("--confirm-no-active-sessions", action="store_true", required=True)
    recover.add_argument(
        "--shared-filesystem", action="store_true", help="Use the direct-write shared filesystem lock"
    )
    convert = sub.add_parser("convert-sft")
    convert.add_argument("--samples", required=True)
    convert.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "convert-sft":
        convert_sft(args.samples, args.output)
        return
    store = ArchiveStore(args.archive_dir, allow_shared_filesystem=args.shared_filesystem)
    try:
        if args.command == "export":
            logger.info("Archive export: %s", export_archive(store, args.output_dir))
        else:
            logger.info("Removed %s abandoned candidates", store.recover())
    finally:
        store.close()


if __name__ == "__main__":
    main()
