# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Pure reader-side merging of local optimizer facts; no training
collectives."""

from __future__ import annotations

import copy
from collections import defaultdict
from typing import Any, Iterable


def merge_train_updates(stages: Iterable[dict[str, Any]], inputs: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge full-v3 DP boundaries, preserving full-v2 global events unchanged.

    Missing/conflicting owners mean unknown. Input completeness additionally
    requires the persisted train fragments. Optimizer counts are relative to
    actor-group initialization and remain unknown after any gap or conflicting
    sequence.
    """
    grouped = defaultdict(list)
    fragments = defaultdict(list)
    result = []
    for record in stages:
        if record.get("scope") == "update":
            result.append(copy.deepcopy(record))
        elif record.get("scope") == "update_fragment":
            grouped[(record.get("run_id"), record.get("train_generation_id"), record.get("update_id"))].append(record)
    for record in inputs:
        fragments[(record.get("run_id"), record.get("train_generation_id"), record.get("update_id"))].append(record)
    summaries = defaultdict(list)
    conflicting_sequences = defaultdict(set)
    for key, records in grouped.items():
        # Re-reading an identical record is idempotent. A changed record or a
        # second attempt under the same update identity is a conflict.
        unique = []
        for record in records:
            if record not in unique:
                unique.append(record)
        first = unique[0]
        sizes = {r.get("dp_world_size") for r in unique}
        sequences = {r.get("update_sequence") for r in unique}
        size = first.get("dp_world_size")
        valid = len(sizes) == len(sequences) == 1 and type(size) is int and size > 0
        valid = valid and isinstance(key[1], str) and bool(key[1])
        identities = {(r.get("rollout_id"), r.get("step_id")) for r in unique}
        valid = valid and len(identities) == 1
        sequence = first.get("update_sequence")
        valid = valid and type(sequence) is int and sequence >= 0
        owners = defaultdict(list)
        for record in unique:
            owners[record.get("dp_rank")].append(record)
        valid = valid and set(owners) == set(range(size or 0))
        ends = []
        for pair in owners.values():
            starts = [r for r in pair if r.get("status") == "start"]
            outcomes = [r for r in pair if r.get("status") != "start"]
            if len(starts) != 1 or len(outcomes) != 1:
                valid = False
                continue
            start, end = starts[0], outcomes[0]
            valid = valid and start.get("stage_id") == end.get("stage_id")
            valid = valid and start.get("event_seq") == 1 and end.get("event_seq") == 2
            valid = valid and start.get("step_id") == end.get("step_id")
            ends.append(end)
        outcomes_known = valid and all(
            type(r.get("optimizer_successful")) is bool
            and r.get("status") == ("completed" if r["optimizer_successful"] else "failed")
            for r in ends
        )
        status = "unknown"
        if outcomes_known:
            status = "committed" if all(r["optimizer_successful"] for r in ends) else "failed"
        complete = outcomes_known and all(r.get("input_complete") is True for r in ends)
        input_owners = defaultdict(list)
        attempts = {end.get("dp_rank"): end.get("stage_id") for end in ends}
        for record in fragments[key]:
            if not record.get("stage_id") or record["stage_id"] != attempts.get(record.get("dp_rank")):
                continue
            if record not in input_owners[record.get("dp_rank")]:
                input_owners[record.get("dp_rank")].append(record)
        complete = complete and set(input_owners) == set(range(size or 0))
        complete = complete and all(
            len(rows) == 1 and rows[0].get("input_complete") is True and rows[0].get("dp_world_size") == size
            for rows in input_owners.values()
        )
        summary = {
            "run_id": key[0],
            "train_generation_id": key[1],
            "update_id": key[2],
            "scope": "update",
            "stage": "train",
            "status": status,
            "rollout_id": first.get("rollout_id"),
            "step_id": first.get("step_id"),
            "started_at": min(
                (r["occurred_at"] for r in unique if r.get("status") == "start" and r.get("occurred_at")), default=None
            ),
            "occurred_at": max((r["occurred_at"] for r in ends if r.get("occurred_at")), default=None),
            "update_sequence": sequence if len(sequences) == 1 else None,
            "optimizer_step": None,
            "group_ids": sorted({g for r in ends for g in r.get("group_ids", [])}),
            "groups_finalized": bool(valid and all(r.get("groups_finalized") is True for r in ends)),
            "input_complete": bool(complete),
        }
        if not valid:
            conflicting_sequences[key[:2]].update(sequences)
        summaries[key[:2]].append(summary)
    for generation, updates in summaries.items():
        by_sequence = defaultdict(list)
        for update in updates:
            by_sequence[update["update_sequence"]].append(update)
        count = 0
        sequence = 0
        while sequence not in conflicting_sequences[generation] and len(by_sequence.get(sequence, [])) == 1:
            update = by_sequence[sequence][0]
            if update["status"] == "unknown":
                break
            if update["status"] == "committed":
                count += 1
                update["optimizer_step"] = count
            sequence += 1
        result.extend(updates)
    return result
