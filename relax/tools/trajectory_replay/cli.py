# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Command-line entry point for offline trajectory replay.

Usage:

    python -m relax.tools.trajectory_replay inspect  <bundle>
    python -m relax.tools.trajectory_replay validate <bundle>
    python -m relax.tools.trajectory_replay replay   <bundle> [--stage all]
        [--sample s-0 --group g-1 --batch mb-0007 --step 120:0 --rollout 120]

inspect works on incomplete bundles (no COMPLETE required); validate
and replay require a complete, checksum-validated bundle. replay
supports selecting a single sample / group / micro-batch (each expands to its
semantic-group closure); cohort-level stages (loss) are skipped for a partial
selection. --step ROLLOUT_ID:STEP_ID selects that actor step by exact
actor_step_id match. --rollout ROLLOUT_ID selects a rollout-level bundle.
On a capture directory, the matching bundle is picked; the two flags are
mutually exclusive.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from relax.utils.logging_utils import get_logger
from relax.utils.replay.bundle import cohort_expected_ranks
from relax.utils.replay.report import ReplayReport
from relax.utils.replay.runner import replay as run_replay
from relax.utils.replay.schema import index_from_dict, manifest_from_dict
from relax.utils.replay.validate import validate_bundle


logger = get_logger(__name__)


def _inspect(bundle_path: Path) -> int:
    try:
        manifest = manifest_from_dict(json.loads((bundle_path / "manifest.json").read_text(encoding="utf-8")))
        index = index_from_dict(json.loads((bundle_path / "index.json").read_text(encoding="utf-8")))
    except FileNotFoundError as exc:
        logger.error("incomplete bundle: %s", exc)
        return 1
    except (ValueError, json.JSONDecodeError) as exc:
        logger.error("invalid bundle metadata: %s", exc)
        return 1

    identity = index.identity
    logger.info("bundle %s (format %s)", manifest.bundle_id, manifest.format_version)
    if identity.actor_step_id is not None:
        logger.info(
            "actor step: rollout_id=%s step_id=%s", identity.actor_step_id.rollout_id, identity.actor_step_id.step_id
        )
    else:
        logger.info("rollout: rollout_id=%s", identity.rollout_id)
    logger.info("samples: %d", len(index.samples))
    logger.info("producer commit: %s", manifest.producer.commit or "<unset>")
    for stage, contract in sorted(manifest.stage_contracts.items()):
        logger.info("stage %-20s version=%-4s capability=%s", stage.value, contract.version, contract.capability.value)
    for name, spec in sorted(manifest.payloads.items()):
        logger.info("payload %-24s %s shape=%s bytes=%d", name, spec.dtype, spec.shape, spec.bytes)
    return 0


def _validate(bundle_path: Path) -> int:
    result = validate_bundle(bundle_path)
    for message in result.warnings:
        logger.warning("%s", message)
    for message in result.errors:
        logger.error("%s", message)
    if result.valid:
        logger.info("bundle valid: %s", bundle_path)
        return 0
    logger.error("bundle invalid: %s", bundle_path)
    return 1


def _format_report(report: ReplayReport) -> str:
    lines = [f"bundle {report.bundle_id} — first divergent stage: {report.first_divergent_stage or '<none>'}", ""]
    for result in report.stages:
        line = f"[{result.status.value:>7}] {result.stage}"
        if result.message:
            line += f" — {result.message}"
        if result.max_abs_error is not None:
            line += f" (max_abs_error={result.max_abs_error:.3e}, mismatches={result.mismatch_count})"
        lines.append(line)
        for divergence in result.divergences:
            where = divergence.sample_id or "-"
            offset = f"@{divergence.token_offset}" if divergence.token_offset is not None else ""
            lines.append(
                f"          {divergence.field} {where}{offset}: expected={divergence.expected} "
                f"actual={divergence.actual} abs_err={divergence.abs_error}"
            )
    return "\n".join(lines)


def _parse_step(step: str) -> tuple[int, int] | None:
    try:
        rollout_id, step_id = (int(part) for part in step.split(":"))
    except ValueError:
        return None
    return rollout_id, step_id


def _is_bundle(path: Path) -> bool:
    return (path / "index.json").is_file() and (path / "manifest.json").is_file()


def _iter_bundles(root: Path) -> list[Path]:
    """Return bundle directories under root.

    A capture directory has complete bundles as children, or cohort directories
    whose rank-* children are complete bundles.
    """
    if _is_bundle(root):
        return [root]
    if not root.is_dir():
        return []
    found: list[Path] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        if _is_bundle(child):
            found.append(child)
            continue
        found.extend(
            grandchild
            for grandchild in sorted(child.iterdir())
            if grandchild.is_dir() and grandchild.name.startswith("rank-") and _is_bundle(grandchild)
        )
    return found


def _cohort_complete(cohort_dir: Path) -> bool:
    return (cohort_dir / "COMPLETE").is_file()


def _select_matching_bundle(hits: list[Path], selector: str) -> Path:
    """Pick one hit; multi-rank rank-* siblings require the cohort COMPLETE."""
    parents = {path.parent for path in hits}
    rank_local = all(path.name.startswith("rank-") for path in hits)
    if rank_local and len(parents) == 1:
        parent = next(iter(parents))
        needs_complete = len(hits) > 1 or cohort_expected_ranks(parent) is not None
        if needs_complete and not _cohort_complete(parent):
            raise ValueError(f"cohort {parent} is incomplete (missing COMPLETE) for {selector}")
        return sorted(hits, key=lambda path: path.name)[0]
    if len(hits) == 1:
        return hits[0]
    names = ", ".join(path.name for path in hits)
    raise ValueError(f"multiple bundles match {selector}: {names}")


def _load_index(bundle_path: Path):
    return index_from_dict(json.loads((bundle_path / "index.json").read_text(encoding="utf-8")))


def _identity_matches_step(index, rollout_id: int, step_id: int) -> bool:
    """True when index is the requested actor step (exact actor_step_id)."""
    actual = index.identity.actor_step_id
    return actual is not None and (actual.rollout_id, actual.step_id) == (rollout_id, step_id)


def _identity_matches_rollout(index, rollout_id: int) -> bool:
    """True when index is a rollout-level bundle for rollout_id."""
    return index.identity.actor_step_id is None and index.identity.rollout_id == rollout_id


def _resolve_bundle(bundle_path: Path, step: str | None, rollout: int | None) -> Path:
    """Resolve a bundle path or a capture directory to one bundle."""
    if _is_bundle(bundle_path):
        return bundle_path
    bundles = _iter_bundles(bundle_path)
    if not bundles:
        raise FileNotFoundError(f"{bundle_path} is not a replay bundle (missing index.json / manifest.json)")
    if step is None and rollout is None:
        if len(bundles) == 1:
            return bundles[0]
        raise ValueError(
            f"{bundle_path} contains {len(bundles)} bundles; pass a specific bundle, "
            "--step ROLLOUT_ID:STEP_ID, or --rollout ROLLOUT_ID"
        )
    if step is not None and rollout is not None:
        raise ValueError("--step and --rollout are mutually exclusive")
    hits: list[Path] = []
    if step is not None:
        parsed = _parse_step(step)
        if parsed is None:
            raise ValueError(f"invalid --step {step!r} (expected ROLLOUT_ID:STEP_ID)")
        rollout_id, step_id = parsed
        hits = [path for path in bundles if _identity_matches_step(_load_index(path), rollout_id, step_id)]
        selector = f"actor step {rollout_id}:{step_id}"
    else:
        assert rollout is not None
        hits = [path for path in bundles if _identity_matches_rollout(_load_index(path), rollout)]
        selector = f"rollout {rollout}"
    if not hits:
        raise ValueError(f"no bundle in {bundle_path} matches {selector}")
    return _select_matching_bundle(hits, selector)


def _replay(
    bundle_path: Path,
    stages: str,
    sample: list[str],
    group: list[str],
    batch: list[str],
    step: str | None,
    rollout: int | None,
) -> int:
    if step is not None and rollout is not None:
        logger.error("--step and --rollout are mutually exclusive")
        return 1
    if step is not None and _parse_step(step) is None:
        logger.error("invalid --step %r (expected ROLLOUT_ID:STEP_ID)", step)
        return 1
    try:
        bundle_path = _resolve_bundle(bundle_path, step, rollout)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        logger.error("%s", exc)
        return 1
    if step is not None:
        parsed = _parse_step(step)
        assert parsed is not None
        rollout_id, step_id = parsed
        if not _identity_matches_step(_load_index(bundle_path), rollout_id, step_id):
            logger.error("bundle is not the requested actor step %d:%d", rollout_id, step_id)
            return 1
    if rollout is not None and not _identity_matches_rollout(_load_index(bundle_path), rollout):
        logger.error("bundle is not the requested rollout %d", rollout)
        return 1

    requested_stages: frozenset[str] | None = None
    if stages != "all":
        requested_stages = frozenset(part.strip() for part in stages.split(",") if part.strip())
        if not requested_stages:
            logger.error("empty --stage filter")
            return 1

    try:
        report = run_replay(
            bundle_path,
            sample_ids=sample or None,
            group_ids=group or None,
            batch_ids=batch or None,
            requested_stages=requested_stages,
        )
    except Exception as exc:  # noqa: BLE001 — surface any replay failure to the user
        logger.error("replay failed: %s", exc)
        return 1

    if requested_stages is not None:
        report.stages = [stage for stage in report.stages if stage.stage in requested_stages]

    for line in _format_report(report).splitlines():
        logger.info("%s", line)
    return 0 if report.passed else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="relax.tools.trajectory_replay", description="Offline trajectory replay")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="show identity, capability and payload summary")
    inspect_parser.add_argument("bundle", type=Path)

    validate_parser = subparsers.add_parser("validate", help="validate format, integrity and closure")
    validate_parser.add_argument("bundle", type=Path)

    replay_parser = subparsers.add_parser("replay", help="recompute stages and compare against expected outputs")
    replay_parser.add_argument("bundle", type=Path)
    replay_parser.add_argument("--stage", default="all", help="comma-separated stage filter (default: all)")
    replay_parser.add_argument("--sample", action="append", default=[], help="sample_id to replay (repeatable)")
    replay_parser.add_argument("--group", action="append", default=[], help="semantic group id to replay (repeatable)")
    replay_parser.add_argument("--batch", action="append", default=[], help="micro-batch id to replay (repeatable)")
    replay_parser.add_argument(
        "--step",
        help="select actor step ROLLOUT_ID:STEP_ID by exact actor_step_id",
    )
    replay_parser.add_argument(
        "--rollout",
        type=int,
        default=None,
        help="select a rollout-level bundle by rollout_id",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "inspect":
        return _inspect(args.bundle)
    if args.command == "validate":
        return _validate(args.bundle)
    if args.command == "replay":
        return _replay(args.bundle, args.stage, args.sample, args.group, args.batch, args.step, args.rollout)
    return 1


if __name__ == "__main__":
    sys.exit(main())
