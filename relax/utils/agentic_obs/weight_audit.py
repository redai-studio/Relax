# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Pure publication state and token-source accounting for Agentic audits."""

from __future__ import annotations

import copy
from bisect import bisect_right
from typing import Any


class PublicationLedger:
    """Single-owner, idempotent successful-publication counter.

    Beginning an update invalidates the frontier until that exact update is
    committed. Publisher restarts create a new comparison domain. Raw backend
    labels are opaque strings, never distances between published policies.
    """

    def __init__(self) -> None:
        self._epoch: str | None = None
        self._policy: str | None = None
        self._ordinal = 0
        self._pending: tuple[str, str, str] | None = None
        self._current: dict[str, Any] | None = None
        self._last: tuple[str, str, str] | None = None
        self._retired: set[str] = set()

    def begin(self, policy: str, epoch: str, raw_version: str) -> bool:
        attempt = (policy, epoch, raw_version)
        if epoch in self._retired:
            return False
        if self._pending == attempt or (self._last == attempt and self._current is not None):
            return True
        if self._epoch != epoch or self._policy != policy:
            if self._epoch is not None:
                self._retired.add(self._epoch)
            self._policy, self._epoch, self._ordinal = policy, epoch, 0
        self._pending = attempt
        self._current = None
        return True

    def commit(self, policy: str, epoch: str, raw_version: str) -> dict[str, Any] | None:
        attempt = (policy, epoch, raw_version)
        if attempt == self._last and self._current is not None:
            return self.snapshot()
        if self._pending != attempt:
            return None
        self._ordinal += 1
        self._current = {"policy": policy, "epoch": epoch, "ordinal": self._ordinal, "raw_version": raw_version}
        self._last, self._pending = attempt, None
        return self.snapshot()

    def snapshot(self) -> dict[str, Any] | None:
        if self._current is not None:
            return copy.deepcopy(self._current)
        if self._epoch is not None:
            return {"policy": self._policy, "epoch": self._epoch, "ordinal": None, "status": "updating"}
        return None


def version_age(newer: Any, older: Any) -> int | None:
    """Compare certified ordinals only within the same policy lifetime."""
    if not isinstance(newer, dict) or not isinstance(older, dict):
        return None
    if not newer.get("policy") or not newer.get("epoch"):
        return None
    if (newer.get("policy"), newer.get("epoch")) != (older.get("policy"), older.get("epoch")):
        return None
    left, right = newer.get("ordinal"), older.get("ordinal")
    if type(left) is not int or type(right) is not int or right < 1 or left < right:
        return None
    return left - right


def attempt_source(raw_version: Any, before: Any, after: Any) -> dict[str, Any] | None:
    """Resolve a scalar label across one publisher lifetime.

    Partial abort may overlap publication: the retained prefix still belongs to
    the confirmed pre-update version. A publisher restart, missing snapshot or
    mixed backend label cannot establish this relationship.
    """
    if not isinstance(raw_version, (str, int)) or isinstance(raw_version, bool):
        return None
    if not isinstance(before, dict) or not isinstance(after, dict):
        return None
    if (before.get("policy"), before.get("epoch")) != (after.get("policy"), after.get("epoch")):
        return None
    for version in (before, after):
        if version_age(version, version) == 0 and str(raw_version) == version.get("raw_version"):
            return copy.deepcopy(version)
    return None


def audit_sample(
    trajectory: dict[str, Any] | None,
    consumed: dict[str, Any],
    consumption_version: dict[str, Any] | None,
) -> dict[str, Any]:
    """Intersect committed source intervals with a final response-space mask.

    Missing/invalid/overlapping intervals remain unknown; none of these cases
    is repaired by assigning the newest version. No token arrays are rebuilt.
    """
    mask = consumed.get("effective_loss_mask")
    report: dict[str, Any] = {
        "sample_export_id": consumed.get("sample_export_id"),
        "session_id": trajectory.get("session_id") if trajectory else None,
        "consumption_version": consumption_version,
        "mask_status": consumed.get("effective_loss_mask_status", "missing"),
        "effective_tokens": None,
        "known_version_tokens": 0,
        "known_age_tokens": 0,
        "weighted_age_sum": 0,
        "token_weighted_age": None,
        "version_missing_rate": None,
        "age_missing_rate": None,
        "generation_span": None,
        "known_generation_span": None,
        "post_generation_lag": None,
        "segments": [],
    }
    length = consumed.get("response_length")
    if type(length) is not int or length < 0 or not isinstance(mask, dict):
        return report
    ranges = mask.get("ranges")
    if not isinstance(ranges, list):
        return report
    starts, ends, prefix = [], [], [0]
    for interval in ranges:
        if not isinstance(interval, (list, tuple)) or len(interval) != 3:
            return report
        start, end, value = interval
        if (
            type(start) is not int
            or type(end) is not int
            or not 0 <= start < end <= length
            or value != 1
            or (ends and start < ends[-1])
        ):
            return report
        starts.append(start)
        ends.append(end)
        prefix.append(prefix[-1] + end - start)

    def covered_before(position: int) -> int:
        index = bisect_right(starts, position) - 1
        return 0 if index < 0 else prefix[index] + min(position, ends[index]) - starts[index]

    total = prefix[-1]
    report["effective_tokens"] = total
    nodes = trajectory.get("lineage", []) if trajectory else []
    response_nodes = [node for node in nodes if node.get("kind") == "resp"]
    end_version = response_nodes[-1].get("generation_end_version") if response_nodes else None
    report["generation_end_version"] = end_version
    report["post_generation_lag"] = version_age(consumption_version, end_version)
    cursor, started, turn = 0, False, 0
    versions = []
    first_response = next((index for index, node in enumerate(nodes) if node.get("kind") == "resp"), len(nodes))
    coordinate_length = sum(int(node.get("token_count", 0)) for node in nodes[first_response:])
    coordinates_valid = coordinate_length == length
    for node in nodes:
        if node.get("kind") == "resp":
            started = True
            turn += 1
        if not started:
            continue
        length = int(node.get("token_count", 0))
        segments = node.get("generation_segments", []) if node.get("kind") == "resp" else []
        valid = []
        for segment in segments:
            start, end = segment.get("start"), segment.get("end")
            if type(start) is int and type(end) is int and 0 <= start < end <= length:
                valid.append(segment)
        for segment in valid:
            start, end = segment["start"], segment["end"]
            overlaps = sum(other["start"] < end and start < other["end"] for other in valid) != 1
            count = covered_before(cursor + end) - covered_before(cursor + start) if coordinates_valid else 0
            source = segment.get("version") if not overlaps else None
            known = version_age(source, source) == 0
            age = version_age(consumption_version, source)
            report["known_version_tokens"] += count if known else 0
            report["known_age_tokens"] += count if age is not None else 0
            report["weighted_age_sum"] += count * age if age is not None else 0
            if known and count:
                versions.append(source)
            report["segments"].append(
                {
                    **segment,
                    "turn": turn,
                    "state_hash": node.get("state_hash"),
                    "response_start": cursor + start,
                    "response_end": cursor + end,
                    "effective_tokens": count,
                    "consumption_age": age,
                    "generation_age": version_age(end_version, source),
                    "overlap": overlaps,
                }
            )
        cursor += length
    # Overlapping spans must not double-count unknown token coverage either.
    report["known_version_tokens"] = min(total, report["known_version_tokens"])
    report["known_age_tokens"] = min(total, report["known_age_tokens"])
    if total:
        report["version_missing_rate"] = 1 - report["known_version_tokens"] / total
        report["age_missing_rate"] = 1 - report["known_age_tokens"] / total
    if report["known_age_tokens"]:
        report["token_weighted_age"] = report["weighted_age_sum"] / report["known_age_tokens"]
    if versions and len({(v["policy"], v["epoch"]) for v in versions}) == 1:
        report["known_generation_span"] = max(v["ordinal"] for v in versions) - min(v["ordinal"] for v in versions)
        if report["known_version_tokens"] == total:
            report["generation_span"] = report["known_generation_span"]
    report["coordinates_valid"] = coordinates_valid
    return report
