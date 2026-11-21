# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Directional socket accounting and conservative communication comparisons."""

import threading
import time
import uuid


def aggregate_events(reports: list[dict]) -> dict:
    """Merge rank/attempt reports without combining incompatible byte views.

    The same receipt may be collected twice. Event identifiers make repeated
    collection idempotent, while retries with new event identifiers remain.
    Missing stages are reported, never replaced with unobserved zeroes.
    """
    events, updates, gaps = {}, set(), []
    participants, stages_by_update, event_updates = {}, {}, {}
    for report in reports:
        updates.add(report["update_id"])
        gaps.extend(report.get("unmeasured", []))
        participant = report.get("source_participant")
        if participant:
            world = participant["world_size"]
            rank = participant["rank"]
            if type(world) is not int or type(rank) is not int or not 0 <= rank < world:
                raise ValueError("Invalid source participant")
            group = participants.setdefault(report["update_id"], {"world_size": world, "ranks": set()})
            if group["world_size"] != world:
                raise ValueError("Source participant set changed within an update")
            group["ranks"].add(rank)
        for event in report["links"]:
            key = event["event_id"]
            if key in events and (events[key] != event or event_updates[key] != report["update_id"]):
                raise ValueError("Conflicting communication event identifier")
            if (
                event["stage"] not in ("G", "W", "X", "T", "C")
                or type(event["bytes"]) is not int
                or event["bytes"] < 0
            ):
                raise ValueError("Invalid communication event")
            events[key] = event
            event_updates[key] = report["update_id"]
            stages_by_update.setdefault(report["update_id"], set()).add(event["stage"])
    views = {}
    for event in events.values():
        scope = views.setdefault(event["scope"], {})
        scope[event["stage"]] = scope.get(event["stage"], 0) + event["bytes"]
    measured = {event["stage"] for event in events.values()}
    return {
        "updates": sorted(updates),
        "bytes_by_scope": views,
        "events": len(events),
        "missing_stages": sorted(set("GWXTC") - measured),
        "missing_stages_by_update": {
            key: sorted(set("GWXTC") - stages_by_update.get(key, set())) for key in sorted(updates)
        },
        "unmeasured": gaps,
        "missing_source_ranks": {
            key: sorted(set(range(value["world_size"])) - value["ranks"])
            for key, value in participants.items()
            if len(value["ranks"]) != value["world_size"]
        },
        "physical_link_coverage_complete": False,
    }


class SocketLedger:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._bytes = {"sent": {"X": 0, "C": 0}, "received": {"X": 0, "C": 0}}
        self._events = 0
        self._instance_id = uuid.uuid4().hex

    def add(self, direction: str, stage: str, count: int) -> None:
        if direction not in self._bytes or stage not in ("X", "C") or type(count) is not int or count < 0:
            raise ValueError("Invalid socket counter")
        with self._lock:
            self._bytes[direction][stage] += count
            self._events += 1

    def report(self) -> dict:
        with self._lock:
            return {
                "measurement_level": "socket_bytes",
                "instance_id": self._instance_id,
                "physical_link_coverage_complete": False,
                "monotonic_ns": time.monotonic_ns(),
                "events": self._events,
                "bytes": {direction: dict(values) for direction, values in self._bytes.items()},
            }


def compare_communication(baseline: dict, delta: dict) -> dict:
    """Compare a frozen sequence using one accounting view for both paths.

    Each stage supplies lower/upper byte bounds and evidence. Unknown stages
    keep the result pending; application and wire views cannot be mixed.
    """
    if baseline["scope"] != delta["scope"] or baseline["sequence"] != delta["sequence"]:
        raise ValueError("Compare the same frozen sequence and accounting scope")
    totals = []
    missing = []
    for name, report in (("baseline", baseline), ("delta", delta)):
        lower, upper = 0, 0
        for stage in ("G", "W", "X", "T", "C"):
            item = report["stages"].get(stage)
            if not item or item.get("lower") is None or item.get("upper") is None or not item.get("evidence"):
                missing.append(f"{name}.{stage}")
                continue
            if (
                any(type(item[key]) is not int for key in ("lower", "upper"))
                or not 0 <= item["lower"] <= item["upper"]
            ):
                raise ValueError("Invalid communication bounds")
            lower += item["lower"]
            upper += item["upper"]
        totals.append((lower, upper))
    if missing or totals[0][0] == 0:
        return {"status": "incomplete", "missing": missing, "saving_lower_bound": None}
    saving = 1 - totals[1][1] / totals[0][0]
    return {
        "status": "measured",
        "saving_lower_bound": saving,
        "meets_50_percent": saving >= 0.5,
        "baseline": {"lower": totals[0][0], "upper": totals[0][1]},
        "delta": {"lower": totals[1][0], "upper": totals[1][1]},
    }
