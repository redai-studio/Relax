# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Read-only provenance and published-version accounting for actor loss tokens.

This module deliberately has no Torch, Ray, or SGLang dependency. Token offsets
are absolute in Sample.tokens; prepared rows contain counts from the CPU mask.
"""

from __future__ import annotations

import json
import os
import tempfile
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

from relax.utils.env import Envs


AUDIT_KEY = "weight_version_audit"
AUDIT_FIELD = "weight_audit_payload"
PREPARED_FIELD = "weight_audit_rows"
_VERSION_PREFIX = "relax-audit-v1:"
_MAX_REPORT_BYTES = 8 * 1024 * 1024
_pending_summaries: dict[Path, dict[str, Any]] = {}
_unread_summaries: set[Path] = set()


def enabled() -> bool:
    """Enable collection and transport together through one registered knob."""
    return bool(Envs.RELAX_WEIGHT_AUDIT_DIR)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_frontier() -> dict[str, Any] | None:
    """Read the shared publication snapshot, or explicitly leave it unknown."""
    directory = Envs.RELAX_WEIGHT_AUDIT_DIR
    if not directory:
        return None
    try:
        value = json.loads((Path(directory) / "frontier.json").read_text())
        if (
            not isinstance(value, dict)
            or value.get("schema_version") != 1
            or not isinstance(value.get("policy_id"), str)
            or not isinstance(value.get("lineage_id"), str)
            or type(value.get("published_seq")) is not int
            or value["published_seq"] < 0
            or value.get("evidence_valid") is False
        ):
            return None
        return value
    except (OSError, ValueError, TypeError):
        return None


class PublicationLedger:
    """One publisher's successful updates; raw updater attempts are not ages.

    The caller confirms all policy loads before calling publish(), then applies
    its returned tag before resuming generation. A fresh publisher has a fresh
    lineage, including after a restart. No initial model is inferred as seq=0.
    """

    def __init__(self, directory: str, policy_id: str):
        self.directory = Path(directory)
        self.policy_id = policy_id
        self.lineage_id = uuid.uuid4().hex
        self.published_seq = 0

    def publish(self) -> str:
        """Confirm one complete policy publication, never an actor_fwd-only
        call."""
        self.published_seq += 1
        next_seq = self.published_seq
        _atomic_json(
            self.directory / "frontier.json",
            {
                "schema_version": 1,
                "policy_id": self.policy_id,
                "lineage_id": self.lineage_id,
                "published_seq": next_seq,
            },
        )
        return f"{_VERSION_PREFIX}{self.lineage_id}:{next_seq}"

    def invalidate(self) -> None:
        """Do not trust retained adapter labels after an unconfirmed update."""
        # Retained tags must stay invalid even if their responses arrive after recovery.
        self.lineage_id = uuid.uuid4().hex
        _atomic_json(self.directory / "frontier.json", {"schema_version": 1, "evidence_valid": False})


def _version(raw: Any, frontier: dict[str, Any] | None) -> dict[str, Any]:
    value = {"raw_version": raw, "published_seq": None, "lineage_id": None}
    if raw is None:
        return {**value, "unknown_reason": "missing_backend_version"}
    if not isinstance(raw, str) or not raw.startswith(_VERSION_PREFIX):
        return {**value, "unknown_reason": "unconfirmed_publication"}
    lineage, separator, seq = raw[len(_VERSION_PREFIX) :].rpartition(":")
    if not separator or not seq.isdecimal() or not lineage or int(seq) <= 0:
        return {**value, "unknown_reason": "invalid_publication_tag"}
    if frontier is None:
        return {**value, "unknown_reason": "publication_frontier_unavailable"}
    if lineage != frontier["lineage_id"] or int(seq) > frontier["published_seq"]:
        return {**value, "unknown_reason": "publication_scope_mismatch"}
    return {
        **value,
        "published_seq": int(seq),
        "policy_id": frontier["policy_id"],
        "lineage_id": lineage,
        "unknown_reason": None,
    }


def generation_spans(
    meta: dict[str, Any], count: int, *, offset: int, request_id: str, attempt_id: str, piece_index: int
) -> list[dict[str, Any]]:
    """Map backend output intervals to accepted new tokens, preserving holes.

    Prefer explicit backend intervals. A scalar is accepted only with the
    controlled DCS abort-before-update contract and a confirmed publication
    tag.
    """
    if not count:
        return []
    frontier = read_frontier()
    mapping_valid = meta.get("weight_audit_token_mapping_valid", True)
    pieces = meta.get("weight_versions") if mapping_valid else None
    if not isinstance(pieces, list):
        pieces = [
            {
                "start": 0,
                "end": count,
                "version": meta.get("weight_version"),
                "unproven": not meta.get("_weight_audit_scalar_bound", False) or not mapping_valid,
            }
        ]
    result = []
    previous = 0
    for number, piece in enumerate(pieces):
        start, end = piece.get("start"), piece.get("end")
        if type(start) is not int or type(end) is not int or not previous <= start < end <= count:
            return generation_spans(
                {}, count, offset=offset, request_id=request_id, attempt_id=attempt_id, piece_index=piece_index
            )
        version = _version(piece.get("version"), frontier)
        if piece.get("unproven"):
            version.update(
                published_seq=None,
                unknown_reason=("missing_backend_version_intervals" if mapping_valid else "unmapped_token_range"),
            )
        result.append(
            {
                **version,
                "token_start": offset + start,
                "token_end": offset + end,
                "request_id": request_id,
                "attempt_id": attempt_id,
                "segment_id": f"{attempt_id}/{piece_index}/{number}",
            }
        )
        previous = end
    return result


def encode_payload(provenance: dict[str, Any]) -> list[int]:
    """Use the existing numeric TensorDict/TransferQueue row transport."""
    return list(json.dumps(provenance, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8"))


def prepare_row(provenance: dict[str, Any], total: int, response: int, mask: list[int]) -> dict[str, Any]:
    """Count each source interval against the actual final CPU loss mask."""
    if len(mask) != response or not 0 <= response <= total or any(value not in (0, 1) for value in mask):
        raise ValueError("Audit requires an exact binary response-relative loss mask")
    prefix = [0]
    for value in mask:
        prefix.append(prefix[-1] + int(value))
    prompt = total - response
    previous = prompt
    spans = []
    covered = 0
    for source in sorted(provenance.get("spans", []), key=lambda item: item["token_start"]):
        start, end = source["token_start"], source["token_end"]
        if type(start) is not int or type(end) is not int or not previous <= start < end <= total:
            raise ValueError("Audit intervals must be disjoint and inside the response suffix")
        count = prefix[end - prompt] - prefix[start - prompt]
        spans.append({**source, "eligible_tokens": count})
        covered += count
        previous = end
    return {**provenance, "eligible_tokens": prefix[-1], "uncovered_tokens": prefix[-1] - covered, "spans": spans}


def prepare_batch(data: dict[str, Any]) -> None:
    """Decode audit-only fields before masks leave CPU; never change training
    fields."""
    payloads = data.pop(AUDIT_FIELD)
    prepared = []
    for payload, total, response, mask in zip(
        payloads, data["total_lengths"], data["response_lengths"], data["loss_masks"], strict=True
    ):
        if getattr(getattr(mask, "device", None), "type", "cpu") != "cpu":
            raise ValueError("Audit preparation must happen before GPU ingestion")
        values = mask.tolist() if hasattr(mask, "tolist") else list(mask)
        encoded = payload.tolist() if hasattr(payload, "tolist") else payload
        try:
            provenance = json.loads(bytes(encoded).decode("utf-8"))
            if not isinstance(provenance, dict) or provenance.get("schema_version", 1) != 1:
                raise ValueError("Unsupported audit schema")
            row = prepare_row(provenance, int(total), int(response), values)
        except (ValueError, TypeError, KeyError, UnicodeError):
            row = prepare_row({}, int(total), int(response), values)
            row["audit_error"] = "invalid_provenance_payload"
        prepared.append(row)
    data[PREPARED_FIELD] = prepared


def unavailable_rows(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Keep valid original mask counts when only the audit transport is
    corrupt."""
    rows = []
    for index, total in enumerate(data["total_lengths"]):
        try:
            mask = data["loss_masks"][index]
            if getattr(getattr(mask, "device", None), "type", "cpu") != "cpu":
                raise ValueError("CPU mask unavailable")
            values = mask.tolist() if hasattr(mask, "tolist") else list(mask)
            row = prepare_row({}, int(total), int(data["response_lengths"][index]), values)
        except (ValueError, TypeError, KeyError, IndexError):
            row = {"eligible_tokens": 0, "uncovered_tokens": 0, "spans": []}
        rows.append({**row, "audit_error": "preparation_failed"})
    return rows


def _coverage_metrics(total: int, known: int, age_count: int, age_sum: int) -> dict[str, float | None]:
    return {
        "missing_token_rate": (total - known) / total if total else None,
        "version_coverage": known / total if total else None,
        "age_coverage": age_count / total if total else None,
        "token_weighted_age": age_sum / age_count if age_count else None,
    }


def _finish_progress(
    finish: dict[str, Any] | None, frontier: dict[str, Any] | None, newest_seq: int | None
) -> tuple[int | None, int | None]:
    """Compare completion and consumption only within one ordered publication
    lineage."""
    if not (
        frontier
        and finish
        and (finish.get("lineage_id"), finish.get("policy_id")) == (frontier["lineage_id"], frontier["policy_id"])
        and (newest_seq is None or newest_seq <= finish["published_seq"])
        and finish["published_seq"] <= frontier["published_seq"]
    ):
        return None, None
    advance = frontier["published_seq"] - finish["published_seq"]
    finish_gap = finish["published_seq"] - newest_seq if newest_seq is not None else None
    return advance, finish_gap


def audit_row(row: dict[str, Any], frontier: dict[str, Any] | None) -> dict[str, Any]:
    """Audit one actual actor-loss exposure, never a generation or a
    prefetch."""
    total = row["eligible_tokens"]
    known = age_count = age_sum = negative = 0
    versions, ages, details = [], [], []
    source_scopes = {
        (span.get("policy_id"), span.get("lineage_id"))
        for span in row["spans"]
        if span.get("published_seq") is not None and span["eligible_tokens"]
    }
    missing = Counter({"uncovered_token_range": row["uncovered_tokens"]})
    for span in row["spans"]:
        count = span["eligible_tokens"]
        seq = span.get("published_seq")
        reason = span.get("unknown_reason")
        if frontier is None and len(source_scopes) > 1:
            reason = "mixed_publication_lineages"
            seq = None
        if (
            seq is not None
            and frontier is not None
            and (span.get("policy_id"), span.get("lineage_id")) != (frontier["policy_id"], frontier["lineage_id"])
        ):
            reason = "publication_scope_mismatch"
            seq = None
        age = None
        if seq is not None and count:
            known += count
            versions.append(seq)
            if frontier is not None:
                delta = frontier["published_seq"] - seq
                if delta >= 0:
                    age = delta
                    ages.append(age)
                    age_count += count
                    age_sum += count * age
                else:
                    negative += count
        elif count:
            missing[reason or "unknown_version"] += count
        details.append({**span, "age": age, "unknown_reason": reason})
    lo, hi = (min(versions), max(versions)) if versions else (None, None)
    finish = row.get("finish_snapshot")
    advance, finish_gap = _finish_progress(finish, frontier, hi) if total else (None, None)
    return {
        "sample_uid": row.get("sample_uid"),
        "physical_row_index": row.get("physical_row_index"),
        "session_id": row.get("session_id"),
        "leaf_state_hash": row.get("leaf_state_hash"),
        "finish_boundary": "session_response_commit",
        "finish_snapshot": finish,
        "consumption_snapshot": frontier,
        "eligible_tokens": total,
        "known_version_tokens": known,
        "age_valid_tokens": age_count,
        "age_token_sum": age_sum,
        "negative_age_tokens": negative,
        **_coverage_metrics(total, known, age_count, age_sum),
        "oldest_token_age": max(ages) if ages else None,
        "newest_token_age": min(ages) if ages else None,
        "generation_min_seq": lo,
        "generation_max_seq": hi,
        "observed_generation_span": hi - lo if versions else None,
        "newest_source_gap": (
            frontier["published_seq"] - hi if frontier and versions and frontier["published_seq"] >= hi else None
        ),
        "generation_span_complete": bool(total and known == total),
        "post_generation_published_advance": advance,
        "finish_frontier_gap": finish_gap,
        "unknown_reasons": {key: value for key, value in missing.items() if value},
        "audit_error": row.get("audit_error"),
        "audit_errors": int(bool(row.get("audit_error"))),
        "spans": details,
    }


def aggregate(reports: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge counts, not per-row or per-rank averages."""
    keys = (
        "eligible_tokens",
        "known_version_tokens",
        "age_valid_tokens",
        "age_token_sum",
        "negative_age_tokens",
        "audit_errors",
    )
    result = {key: sum(report[key] for report in reports) for key in keys}
    result.update(
        _coverage_metrics(
            result["eligible_tokens"],
            result["known_version_tokens"],
            result["age_valid_tokens"],
            result["age_token_sum"],
        ),
        aggregate_complete=result["audit_errors"] == 0,
    )
    result["consumed_rows"] = sum(
        report.get("consumed_rows", int(report["eligible_tokens"] > 0)) for report in reports
    )
    result["complete_generation_rows"] = sum(
        report.get("complete_generation_rows", int(bool(report.get("generation_span_complete")))) for report in reports
    )
    for name, source in (
        ("generation_span", "observed_generation_span"),
        ("post_generation_advance", "post_generation_published_advance"),
    ):
        count = sum(report.get(f"{name}_count", int(report.get(source) is not None)) for report in reports)
        total_value = sum(report.get(f"{name}_sum", report.get(source) or 0) for report in reports)
        values = [report.get(f"{name}_max", report.get(source)) for report in reports]
        result.update(
            {
                f"{name}_count": count,
                f"{name}_sum": total_value,
                f"{name}_mean": total_value / count if count else None,
                f"{name}_max": max((value for value in values if value is not None), default=None),
            }
        )
    for name, operation in (("oldest_token_age", max), ("newest_token_age", min)):
        result[name] = operation((report[name] for report in reports if report.get(name) is not None), default=None)
    reasons = Counter()
    for report in reports:
        reasons.update(report.get("unknown_reasons", {}))
    result["unknown_reasons"] = dict(reasons)
    return result


class AuditStep:
    """Stage loss-callback observations, committing only after the schedule
    succeeds."""

    def __init__(self, consumption_id: str):
        self.consumption_id = consumption_id
        self.frontier = read_frontier()
        self.observed: dict[int, list[dict[str, Any]]] = {}
        self.summary: dict[str, Any] | None = None
        self.masked_rows: dict[int, tuple[list[dict[str, Any]], Any]] = {}

    def observe(
        self, ordinal: int, rows: list[dict[str, Any]] | None, *, dummy: bool = False, counts: Any = None
    ) -> None:
        if not dummy and rows is not None and ordinal not in self.observed:
            if counts is None:
                self.observed[ordinal] = [
                    {**audit_row(row, self.frontier), "microbatch_index": ordinal} for row in rows
                ]
            else:
                # Device counts are read only after the forward/backward schedule succeeds.
                self.masked_rows[ordinal] = (rows, counts)
                self.observed[ordinal] = []

    def commit(self, rank: int) -> dict[str, Any]:
        """Write bounded details; retain failed cumulative writes for the next
        step."""
        error = None
        if self.summary is None:
            for ordinal, (rows, counts) in self.masked_rows.items():
                values = iter(counts.tolist())
                for row in rows:
                    total = next(values)
                    spans = [{**span, "eligible_tokens": next(values)} for span in row["spans"]]
                    adjusted = {
                        **row,
                        "eligible_tokens": total,
                        "spans": spans,
                        "uncovered_tokens": total - sum(span["eligible_tokens"] for span in spans),
                    }
                    self.observed[ordinal].append({**audit_row(adjusted, self.frontier), "microbatch_index": ordinal})
            reports = [report for batch in self.observed.values() for report in batch]
            summary = {"consumption_id": self.consumption_id, "rank": rank, **aggregate(reports)}
            directory = Path(Envs.RELAX_WEIGHT_AUDIT_DIR)
            lineage = self.frontier["lineage_id"] if self.frontier else "unknown"
            path = directory / lineage / f"consumed-rank-{rank}.jsonl"
            summary_path = path.with_suffix(".summary.json")
            previous = _pending_summaries.get(summary_path)
            if previous is None:
                _unread_summaries.add(summary_path)
            cumulative = aggregate([previous, summary]) if previous else aggregate([summary])
            cumulative.update(
                rank=rank,
                last_consumption_id=self.consumption_id,
                consumptions=(previous["consumptions"] if previous else 0) + 1,
                detail_records_dropped=previous["detail_records_dropped"] if previous else 0,
            )
            # Stage once before I/O so retries and later steps cannot lose or duplicate counts.
            _pending_summaries[summary_path] = cumulative
            self.summary = summary
            self.observed.clear()
            self.masked_rows.clear()
            try:
                dropped = self._append_details(path, reports, rank)
            except OSError as exc:
                cumulative["audit_errors"] += 1
                cumulative["aggregate_complete"] = False
                dropped = len(reports)
                error = exc
            summary["detail_records_dropped"] = dropped
            cumulative["detail_records_dropped"] += dropped
        # Also retry old lineages, which may never receive another consumption.
        for summary_path, cumulative in list(_pending_summaries.items()):
            try:
                if summary_path in _unread_summaries:
                    try:
                        previous = json.loads(summary_path.read_text())
                    except FileNotFoundError:
                        previous = None
                    if previous is not None:
                        merged = aggregate([previous, cumulative])
                        merged.update(
                            rank=cumulative["rank"],
                            last_consumption_id=cumulative["last_consumption_id"],
                            consumptions=previous["consumptions"] + cumulative["consumptions"],
                            detail_records_dropped=previous["detail_records_dropped"]
                            + cumulative["detail_records_dropped"],
                        )
                        cumulative.clear()
                        cumulative.update(merged)
                    _unread_summaries.remove(summary_path)
                _atomic_json(summary_path, cumulative)
            except (OSError, ValueError) as exc:
                cumulative["audit_errors"] += 1
                cumulative["aggregate_complete"] = False
                error = exc
            else:
                del _pending_summaries[summary_path]
        if error is not None:
            self.summary["audit_errors"] += 1
            self.summary["aggregate_complete"] = False
            raise error
        return self.summary

    def _append_details(self, path: Path, reports: list[dict[str, Any]], rank: int) -> int:
        """Append details within the byte limit and return the number
        omitted."""
        path.parent.mkdir(parents=True, exist_ok=True)
        size = path.stat().st_size if path.exists() else 0
        dropped = 0
        with path.open("a", encoding="utf-8") as handle:
            for report in reports:
                record = {**report, "consumption_id": self.consumption_id, "rank": rank}
                line = json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n"
                encoded_size = len(line.encode("utf-8"))
                if size + encoded_size > _MAX_REPORT_BYTES:
                    dropped += 1
                    continue
                handle.write(line)
                size += encoded_size
        return dropped
